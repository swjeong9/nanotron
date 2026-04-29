"""Single-run benchmark 결과 → 시계열 + 통계 + 이론치 비교.

생성물 (모두 ``examples/heterogeneous/figures/single_run/`` 아래, gitignore):
- ``timeseries.png``  — 양 노드의 power / temperature / SMACT / TENSO /
   DRAMA / DCGM PCIe / NIC TX·RX (실측) 시계열
- ``stats.md``        — 한 step 평균 wall-clock + DCGM 평균/최대 + 이론치
   대비 측정값
- ``stats.json``      — 위 정보 raw

분석 핵심: ``benchmark_single.sh`` 가 만든 ``meta.json`` 의 mbs/ga/seq
값으로 이론치 (compute FLOPs, comm bytes) 를 계산하고, 측정 step time /
NIC bandwidth 와 비교한다.

Usage:
    uv run --no-project --with matplotlib python \\
        examples/heterogeneous/plot_single.py \\
        --run-dir /opt/dlami/nvme/single_run
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.font_manager as fm

_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_NOTO)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# Llama 3.2 1B architecture (model_config 와 일관).
LLAMA32_1B_HIDDEN = 2048
LLAMA32_1B_VOCAB = 128256
LLAMA32_1B_NUM_LAYERS = 16
LLAMA32_1B_PARAMS = 1.236e9      # ~1.24 B (embedding + decoder; lm_head 는 tied)
# Per-component params (Llama 3.2 1B):
# - embedding/lm_head (tied): vocab × hidden = 128256 × 2048 = 263M
# - 16 decoder layers: (1.236e9 - 263e6) ÷ 16 ≈ 60.7M / layer
# - final_norm: hidden = 2048 (negligible)
LLAMA32_1B_EMBED_PARAMS = LLAMA32_1B_VOCAB * LLAMA32_1B_HIDDEN
LLAMA32_1B_DECODER_PARAMS_PER_LAYER = (LLAMA32_1B_PARAMS - LLAMA32_1B_EMBED_PARAMS) / LLAMA32_1B_NUM_LAYERS

# 학습 step time 의 "전형적" 추정 (실측 기준 — project_background.md §6.1.1).
# 일반 dense matmul 에서 sustainable 한 값 — peak 의 25-50% 정도.
L4_BF16_TFLOPS = 30
A10G_BF16_TFLOPS = 70  # NVIDIA spec (FP16); A10G 는 BF16 도 동일.
# MFU 계산용 peak BF16 dense TC — NVIDIA datasheet 의 sparse 미적용 수치.
# - L4 (AD104, Ada): BF16 TC dense = 121 TFLOPs
# - A10G (GA102, Ampere): BF16 TC dense = 125 TFLOPs (FP16 와 동일)
L4_BF16_PEAK_TFLOPS = 121
A10G_BF16_PEAK_TFLOPS = 125
# AWS g5.xlarge / g6.xlarge 의 ENA 네트워크 대역폭.
# - "Up to 10 Gbps" 가 datasheet spec 이지만 이는 **burst credit** (24h 당 ~30분).
# - **sustained baseline 은 1.25 Gbps = 156 MB/s** (small instance 공통). 5-10분
#   짧은 benchmark 에서는 burst 로 1250 MB/s peak 가능, 시간 단위 학습에서는
#   baseline 156 MB/s 에 묶임. 측정 평균 100-110 MB/s 가 baseline 한계 근처라는
#   사실과 부합.
# - 본 인스턴스는 EFA 미지원 (RDMA 안 됨) → NCCL Socket plugin (TCP) fallback.
ENA_BURST_GBPS = 10
ENA_BASELINE_GBPS = 1.25

# steady-state 평균 계산 시 사용할 step index — step 0 (warmup, NCCL/cuBLAS
# 초기화 + 데이터로더 cold-start 포함) 만 제외하고 step 1..N-1 (= 1-based
# step 2..N) 사용.
STEADY_SLICE = slice(1, None)


# =============================================================================
# 로그 / DCGM / NIC 파싱
# =============================================================================
_TS_RE = re.compile(r"^\[?\d?[0-9;]*m?(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def parse_log_ts(line: str, year: int) -> float | None:
    m = _TS_RE.match(line.lstrip("\x1b").lstrip())
    if not m:
        return None
    mon, day, h, mn, s = map(int, m.groups())
    return datetime(year, mon, day, h, mn, s, tzinfo=timezone.utc).timestamp()


def extract_step_timings(log_path: Path) -> List[float]:
    """forward+backward 만의 wall-clock (Before → After train_batch_iter)."""
    return [a - b for b, a, _ in extract_step_boundaries(log_path)]


_MODEL_FLOPS_RE = re.compile(r"\[ModelFLOPs\]\s+(\w+)=(\d+)")
_STAGE_FLOPS_RE = re.compile(r"\[StageFLOPs\]\s+stage\s+(\d+):\s+cost=(\d+)\s+\(([\d.]+)%\)\s+modules=\[([^\]]*)\]")


def extract_flops_log(log_path: Path) -> dict:
    """학습 시작 시 ``[ModelFLOPs]`` / ``[StageFLOPs]`` 줄 파싱.

    nanotron 의 ``get_block_compute_costs()`` 를 통해 module 별/stage 별
    상대 FLOPs (per-token forward 의 attention 무시 추정치) 를 출력한 것을
    그대로 읽어 들임.

    return shape: ``{"per_module": {ClassName: cost, ...},
                     "per_stage": [{"rank": 0, "cost": X, "pct": ..., "modules": "LlamaDecoderLayer×8"}, ...]}``
    """
    out = {"per_module": {}, "per_stage": []}
    if not log_path.exists():
        return out
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _MODEL_FLOPS_RE.search(line)
        if m:
            out["per_module"][m.group(1)] = int(m.group(2))
            continue
        m = _STAGE_FLOPS_RE.search(line)
        if m:
            out["per_stage"].append({
                "rank": int(m.group(1)),
                "cost": int(m.group(2)),
                "pct": float(m.group(3)),
                "modules": m.group(4),
            })
    out["per_stage"].sort(key=lambda x: x["rank"])
    return out


def extract_step_boundaries(log_path: Path) -> List[tuple]:
    """``[(before_ts, fwdbwd_end_ts, step_end_ts), ...]`` 의 unix wallclock triple.

    각 시각의 의미:
    - ``before_ts``: ``Before train_batch_iter`` — step 시작
    - ``fwdbwd_end_ts``: ``After train_batch_iter`` — forward+backward 끝
      (sync_tied_weights / clip / optimizer.step 직전)
    - ``step_end_ts``: ``After training_step`` — optimizer/lr_scheduler 까지 끝
      (실질적 step 종료. 다음 step 의 ``Before`` 직전.)

    ``After training_step`` 이 없으면 (예: trainer.py 미패치) ``step_end_ts =
    fwdbwd_end_ts`` 로 fallback.
    """
    if not log_path.exists():
        return []
    triples = []
    before = None
    fwdbwd_end = None
    year = datetime.now(timezone.utc).year
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Before train_batch_iter" in line:
            # 새 step 시작. 이전 step 이 ``After training_step`` 없이 끝났으면
            # fwdbwd_end 로 fallback (구식 trainer.py 호환).
            if before is not None and fwdbwd_end is not None:
                triples.append((before, fwdbwd_end, fwdbwd_end))
            before = parse_log_ts(line, year)
            fwdbwd_end = None
        elif "After train_batch_iter" in line and before is not None:
            fwdbwd_end = parse_log_ts(line, year)
        elif "After training_step" in line and before is not None and fwdbwd_end is not None:
            step_end = parse_log_ts(line, year)
            if step_end is not None:
                triples.append((before, fwdbwd_end, step_end))
            before = None
            fwdbwd_end = None
    if before is not None and fwdbwd_end is not None:
        triples.append((before, fwdbwd_end, fwdbwd_end))
    return triples


def dcgm_jsonl(txt: Path, awk: Path) -> List[dict]:
    if not txt.exists():
        return []
    out = subprocess.run(
        ["awk", "-f", str(awk), str(txt)],
        capture_output=True, text=True, check=True,
    ).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def parse_nic(path: Path) -> List[dict]:
    """``ts RX_bytes RX_packets TX_bytes TX_packets`` 한 줄짜리 sample 들.

    cumulative bytes 라 인접 두 sample 의 차이 / dt 로 MB/s 계산."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        toks = line.split()
        if len(toks) < 5:
            continue
        try:
            rows.append({
                "ts": float(toks[0]),
                "rx_bytes": int(toks[1]),
                "rx_pkts": int(toks[2]),
                "tx_bytes": int(toks[3]),
                "tx_pkts": int(toks[4]),
            })
        except ValueError:
            continue
    return rows


def nic_to_rate(rows: List[dict]) -> List[dict]:
    """누적 bytes → bytes/sec (rate) 로 변환.

    sanity filter: dt 가 50 ms 미만이면 → bash sleep 의 jitter 또는
    counter wraparound 의심 (잘못 들어오면 분모 작아져 rate 가 비현실적).
    dt > 5 sec 면 sampler 가 일시 정지된 구간 → 의미 없는 평균이라 skip.
    또한 byte delta 가 음수 (counter rollover) 면 skip.
    """
    out = []
    for prev, cur in zip(rows[:-1], rows[1:]):
        dt = cur["ts"] - prev["ts"]
        if dt < 0.05 or dt > 5.0:
            continue
        rx_delta = cur["rx_bytes"] - prev["rx_bytes"]
        tx_delta = cur["tx_bytes"] - prev["tx_bytes"]
        if rx_delta < 0 or tx_delta < 0:
            continue
        out.append({
            "ts": cur["ts"],
            "rx_MBps": rx_delta / dt / 1e6,
            "tx_MBps": tx_delta / dt / 1e6,
        })
    return out


def relative_time(rows, key="ts") -> List[float]:
    if not rows:
        return []
    t0 = rows[0][key]
    return [r[key] - t0 for r in rows]


def active_window(rows: List[dict]) -> List[dict]:
    return [r for r in rows if r.get("SMCLK", 0) > 1500]


def mean_safe(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


# =============================================================================
# Plot
# =============================================================================
def plot_timeseries(meta: dict, dcgm0, dcgm1, nic0, nic1, step_boundaries, out_path: Path):
    """모든 subplot 의 x=0 을 ``첫 학습 step 의 Before train_batch_iter`` 시점
    으로 정렬한다.

    Alignment 방식:
    - NIC: ``/proc/net/dev`` sample 의 unix ts 를 그대로 사용 → x = ts -
      train_start_unix. 정확.
    - DCGM: row_index → 절대시간 변환에 ``meta['dcgm_start_ts_node{0,1}']``
      (benchmark_single.sh 가 dcgmi dmon 띄우기 직전에 기록한 unix) 를 사용.
      DCGM 은 1Hz polling 이므로 row_idx 가 곧 dcgmi 시작 후 elapsed sec.
      ``dcgm_start_ts`` 가 meta 에 없는 경우엔 ``첫 SMCLK boost row``
      휴리스틱으로 fallback (구식 run 결과 호환용).
    """
    fig, axes = plt.subplots(7, 1, figsize=(13, 16), sharex=True)
    fig.suptitle(f"Single run | mbs={meta['mbs']} ga={meta['ga']} seq={meta['seq_len']} "
                 f"(GBS={meta['gbs_seqs']} sequences = {meta['gbs_tokens_per_step']} tokens / step) | "
                 f"NODE 0 = L4 / NODE 1 = A10G")

    # 학습 시작 unix (없으면 NIC 첫 active sample fallback)
    if step_boundaries:
        train_start_unix = step_boundaries[0][0]
    elif nic0:
        train_start_unix = nic0[0]["ts"]
    else:
        train_start_unix = 0

    def _dcgm_align(rows, dcgm_start_unix):
        """DCGM row_index → t (학습 시작 = 0)."""
        if not rows:
            return []
        if dcgm_start_unix and dcgm_start_unix > 0 and train_start_unix > 0:
            # row_idx 가 dcgmi 시작 후 elapsed sec (1 Hz polling).
            # x = (dcgmi_start_unix + row_idx) - train_start_unix
            offset_sec = train_start_unix - dcgm_start_unix
            return [r["ts"] - offset_sec for r in rows]
        # Fallback: first-SMCLK-boost row → t=0
        boost = [i for i, r in enumerate(rows) if r.get("SMCLK", 0) > 1500]
        offset = boost[0] if boost else 0
        return [r["ts"] - offset for r in rows]

    def _nic_align(rows):
        return [r["ts"] - train_start_unix for r in rows]

    dcgm_start_n0 = float(meta.get("dcgm_start_ts_node0") or 0)
    dcgm_start_n1 = float(meta.get("dcgm_start_ts_node1") or 0)
    t_d0 = _dcgm_align(dcgm0, dcgm_start_n0)
    t_d1 = _dcgm_align(dcgm1, dcgm_start_n1)
    nic0_rate = nic_to_rate(nic0)
    nic1_rate = nic_to_rate(nic1)
    t_n0 = _nic_align(nic0_rate)
    t_n1 = _nic_align(nic1_rate)

    # (1) Power
    if dcgm0:
        axes[0].plot(t_d0, [r["POWER"] for r in dcgm0], label="NODE 0 (L4)", color="tab:blue")
    if dcgm1:
        axes[0].plot(t_d1, [r["POWER"] for r in dcgm1], label="NODE 1 (A10G)", color="tab:orange")
    axes[0].set_ylabel("GPU power [W]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # (2) Temperature
    if dcgm0:
        axes[1].plot(t_d0, [r["TMPTR"] for r in dcgm0], color="tab:blue")
    if dcgm1:
        axes[1].plot(t_d1, [r["TMPTR"] for r in dcgm1], color="tab:orange")
    axes[1].set_ylabel("GPU temp [°C]")
    axes[1].grid(True, alpha=0.3)

    # (3) SMACT
    if dcgm0:
        axes[2].plot(t_d0, [r["SMACT"] for r in dcgm0], color="tab:blue")
    if dcgm1:
        axes[2].plot(t_d1, [r["SMACT"] for r in dcgm1], color="tab:orange")
    axes[2].set_ylabel("SM active fraction\n(DCGM SMACT, 0..1)")
    axes[2].grid(True, alpha=0.3)

    # (4) Tensor core
    if dcgm0:
        axes[3].plot(t_d0, [r["TENSO"] for r in dcgm0], color="tab:blue")
    if dcgm1:
        axes[3].plot(t_d1, [r["TENSO"] for r in dcgm1], color="tab:orange")
    axes[3].set_ylabel("Tensor-core util\n(DCGM TENSO, 0..1)")
    axes[3].grid(True, alpha=0.3)

    # (5) DRAM bandwidth
    if dcgm0:
        axes[4].plot(t_d0, [r["DRAMA"] for r in dcgm0], color="tab:blue")
    if dcgm1:
        axes[4].plot(t_d1, [r["DRAMA"] for r in dcgm1], color="tab:orange")
    axes[4].set_ylabel("DRAM bandwidth\n(DCGM DRAMA, 0..1)")
    axes[4].grid(True, alpha=0.3)

    # (6) DCGM PCIe (proxy)
    if dcgm0:
        axes[5].plot(t_d0, [r["PCITX"] / 1e6 for r in dcgm0], label="NODE 0 PCITX", color="tab:blue", linestyle="--")
        axes[5].plot(t_d0, [r["PCIRX"] / 1e6 for r in dcgm0], label="NODE 0 PCIRX", color="tab:blue", linestyle=":")
    if dcgm1:
        axes[5].plot(t_d1, [r["PCITX"] / 1e6 for r in dcgm1], label="NODE 1 PCITX", color="tab:orange", linestyle="--")
        axes[5].plot(t_d1, [r["PCIRX"] / 1e6 for r in dcgm1], label="NODE 1 PCIRX", color="tab:orange", linestyle=":")
    axes[5].set_ylabel("PCIe (GPU↔CPU)\n[MB/s]")
    axes[5].legend(loc="upper right", fontsize=8)
    axes[5].grid(True, alpha=0.3)

    # (7) NIC actual bandwidth (from /proc/net/dev)
    if nic0_rate:
        axes[6].plot(t_n0, [r["tx_MBps"] for r in nic0_rate], label="NODE 0 TX (NIC)", color="tab:blue", linestyle="--")
        axes[6].plot(t_n0, [r["rx_MBps"] for r in nic0_rate], label="NODE 0 RX (NIC)", color="tab:blue", linestyle=":")
    if nic1_rate:
        axes[6].plot(t_n1, [r["tx_MBps"] for r in nic1_rate], label="NODE 1 TX (NIC)", color="tab:orange", linestyle="--")
        axes[6].plot(t_n1, [r["rx_MBps"] for r in nic1_rate], label="NODE 1 RX (NIC)", color="tab:orange", linestyle=":")
    # ENA — burst (10 Gbps) 와 baseline (1.25 Gbps) 두 개 선
    axes[6].axhline(ENA_BURST_GBPS * 125, color="red", linestyle=":", alpha=0.5,
                    label=f"ENA burst ({ENA_BURST_GBPS} Gbps, 24h 당 ~30분)")
    axes[6].axhline(ENA_BASELINE_GBPS * 125, color="purple", linestyle=":", alpha=0.5,
                    label=f"ENA baseline ({ENA_BASELINE_GBPS} Gbps, sustained)")
    axes[6].set_ylabel("NIC bandwidth\n(/proc/net/dev) [MB/s]")
    axes[6].set_xlabel("elapsed [s] from training start (first 'Before train_batch_iter')")
    axes[6].legend(loc="upper right", fontsize=8)
    axes[6].grid(True, alpha=0.3)

    # 학습 step boundary 를 모든 axis 에 vertical line 으로.
    # 3 종류 line:
    # - iter 시작 (Before train_batch_iter)        → green dashed, alpha 0.5
    # - fwd/bwd 끝 (After train_batch_iter)        → orange dotted, alpha 0.4
    # - step 끝 (After training_step, optimizer 완료) → grey dashed, alpha 0.5
    # 마지막 step end 는 학습 전체 종료라 red 강조.
    if step_boundaries:
        t0 = step_boundaries[0][0]
        for ax in axes:
            for i, (b, fb_end, step_end) in enumerate(step_boundaries):
                ax.axvline(b - t0, color="green", linestyle="--", alpha=0.5, linewidth=0.7)
                ax.axvline(fb_end - t0, color="tab:orange", linestyle=":", alpha=0.4, linewidth=0.6)
                if i == len(step_boundaries) - 1:
                    ax.axvline(step_end - t0, color="red", linestyle="--", alpha=0.7, linewidth=1.0)
                else:
                    ax.axvline(step_end - t0, color="grey", linestyle="--", alpha=0.5, linewidth=0.7)

        axes[0].annotate("iter 1 start\n(Before train_batch_iter)",
                         xy=(0, axes[0].get_ylim()[1]),
                         xytext=(2, axes[0].get_ylim()[1] * 0.95),
                         fontsize=7, color="green")
        last_step_end = step_boundaries[-1][2] - t0
        axes[0].annotate(f"training end\n(After training_step #{len(step_boundaries)})",
                         xy=(last_step_end, axes[0].get_ylim()[1]),
                         xytext=(last_step_end + 2, axes[0].get_ylim()[1] * 0.85),
                         fontsize=7, color="red")
        # 첫 두 step 만 라벨 (legend 역할)
        for i in range(min(2, len(step_boundaries))):
            b, fb, se = step_boundaries[i]
            axes[2].annotate(f"step {i+1}\nfwd/bwd end",
                             xy=(fb - t0, 0.4),
                             xytext=(fb - t0 + 0.5, 0.4 - 0.1 * i),
                             fontsize=6, color="tab:orange", alpha=0.8)
            if i > 0:
                axes[2].annotate(f"step {i}\nstep end",
                                 xy=(se - t0, 0.7),
                                 xytext=(se - t0 + 0.5, 0.7 - 0.05 * i),
                                 fontsize=6, color="grey", alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =============================================================================
# Summary bar chart
# =============================================================================
def plot_bars(meta: dict, step_boundaries, dcgm0, dcgm1, flops_log: dict, out_path: Path):
    """Steady-state 1 step 의 cluster summary 를 bar chart 로.

    3 panel:
    1. Throughput (cluster tokens/sec)
    2. MFU (cluster, L4, A10G) — 6 · cost · tokens / peak_BF16_TC_dense
       *cost* 는 nanotron 의 ``get_block_compute_costs()`` 가 출력한 module 별
       params (= per-token forward 의 matmul-only 개수). FLOPs 환산은 6× 곱.
       Stage 분배는 ``[StageFLOPs]`` log 라인으로 파싱 — manual partition /
       cost-based default 모두 동일하게 동작.
       Log 가 없으면 (구식 nanotron) hardcoded Llama 3.2 1B + [8,8] fallback.
    3. Avg power per node (W) — DCGM POWER active 구간 평균
    """
    if not step_boundaries:
        return

    step_total_times = [se - b for b, _, se in step_boundaries]
    fwdbwd_times = [fb - b for b, fb, _ in step_boundaries]
    steady_total = mean_safe(step_total_times[STEADY_SLICE])
    steady_fwdbwd = mean_safe(fwdbwd_times[STEADY_SLICE])
    if steady_total is None:
        return

    tokens_per_step = meta["gbs_tokens_per_step"]
    throughput_tps = tokens_per_step / steady_total

    # Per-stage FLOPs/step 계산.
    # nanotron 의 ``[StageFLOPs]`` log 가 있으면 그 값 사용 (manual partition 도
    # 자동 반영). 없으면 hardcoded Llama 3.2 1B + [8,8] fallback.
    if flops_log and flops_log.get("per_stage"):
        # cost = per-token forward matmul params count → 6× 가 per-token fwd+bwd FLOPs
        flops_per_stage = [s["cost"] * 6 * tokens_per_step for s in flops_log["per_stage"]]
        pp_partition = meta.get("pp_layer_partition") or [
            sum(int(part.split("×")[1]) for part in s["modules"].split(", ")
                if "DecoderLayer" in part)
            for s in flops_log["per_stage"]
        ]
        # 짧은 라벨: dec layer 수 + 나머지 module (lm_head/loss/embed 등) 만.
        modules_per_stage = []
        for stage_idx, s in enumerate(flops_log["per_stage"]):
            parts = [p.strip() for p in s["modules"].split(",")]
            n_dec = sum(int(p.split("×")[1]) for p in parts if "DecoderLayer" in p)
            extras = [p for p in parts if "DecoderLayer" not in p]
            label = f"{n_dec} dec layers"
            if extras:
                # "Embedding×1" → "embed", "TensorParallelColumnLinear×1" → "lm_head" 등
                aliases = {"Embedding": "embed", "TensorParallelColumnLinear": "lm_head",
                           "TritonRMSNorm": "norm", "Loss": "loss"}
                short = []
                for e in extras:
                    name = e.split("×")[0]
                    short.append(aliases.get(name, name))
                label += "\n+ " + " + ".join(short)
            modules_per_stage.append(label)
    else:
        pp_partition = meta.get("pp_layer_partition") or [LLAMA32_1B_NUM_LAYERS // 2,
                                                           LLAMA32_1B_NUM_LAYERS // 2]
        flops_decoder_per_layer = 6 * LLAMA32_1B_DECODER_PARAMS_PER_LAYER * tokens_per_step
        flops_lm_head = 6 * LLAMA32_1B_EMBED_PARAMS * tokens_per_step
        flops_per_stage = [layers * flops_decoder_per_layer for layers in pp_partition]
        flops_per_stage[-1] += flops_lm_head
        modules_per_stage = [f"{n} dec layers" for n in pp_partition]
        modules_per_stage[-1] += "\n+ lm_head"

    flops_total = sum(flops_per_stage)
    cluster_achieved_tflops = flops_total / steady_total / 1e12
    cluster_peak_tflops = L4_BF16_PEAK_TFLOPS + A10G_BF16_PEAK_TFLOPS
    mfu_cluster = cluster_achieved_tflops / cluster_peak_tflops

    # Per-stage MFU = (per-stage achieved TFLOPs/s) / per-GPU peak.
    # Steady-state 1F1B 에서 양 stage 가 같은 wall-clock 으로 진행하므로
    # per-stage 시간 = step time. 따라서 per-stage achieved = stage_FLOPs / step_time.
    achieved_l4_tflops = flops_per_stage[0] / steady_total / 1e12
    achieved_a10g_tflops = flops_per_stage[1] / steady_total / 1e12
    mfu_l4 = achieved_l4_tflops / L4_BF16_PEAK_TFLOPS
    mfu_a10g = achieved_a10g_tflops / A10G_BF16_PEAK_TFLOPS

    # Power: 다른 metric 과 동일하게 steady-state (step 2..N) wallclock 윈도우만 평균.
    # SMCLK boost heuristic 은 warmup + cleanup 포함이라 더 넓음 → 일관성 유지 위해 동일 윈도우 사용.
    steady_starts = [b for b, _, _ in step_boundaries[STEADY_SLICE]]
    steady_ends = [se for _, _, se in step_boundaries[STEADY_SLICE]]
    if steady_starts and steady_ends:
        steady_t0_unix = steady_starts[0]
        steady_t1_unix = steady_ends[-1]
    else:
        steady_t0_unix = step_boundaries[0][0]
        steady_t1_unix = step_boundaries[-1][2]

    def _dcgm_in_steady(rows, dcgm_start_unix):
        if not rows or not dcgm_start_unix:
            return rows
        return [r for r in rows
                if steady_t0_unix <= dcgm_start_unix + r["ts"] <= steady_t1_unix]

    dcgm_start_n0 = float(meta.get("dcgm_start_ts_node0") or 0)
    dcgm_start_n1 = float(meta.get("dcgm_start_ts_node1") or 0)
    n0_steady = _dcgm_in_steady(dcgm0, dcgm_start_n0)
    n1_steady = _dcgm_in_steady(dcgm1, dcgm_start_n1)
    avg_power_l4 = mean_safe(r["POWER"] for r in n0_steady) if n0_steady else 0
    avg_power_a10g = mean_safe(r["POWER"] for r in n1_steady) if n1_steady else 0

    partition_str = "-".join(str(n) for n in pp_partition)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Single run summary (step 2..{len(step_boundaries)} steady-state) | "
                 f"mbs={meta['mbs']} ga={meta['ga']} GBS={meta['gbs_seqs']} "
                 f"split={partition_str} | NODE 0 = L4 / NODE 1 = A10G")

    # (1) Throughput
    bars1 = axes[0].bar([f"split {partition_str}"], [throughput_tps],
                        color="tab:gray", width=0.5)
    axes[0].set_ylabel("throughput [tokens/sec]")
    axes[0].set_title("Throughput")
    for b, v in zip(bars1, [throughput_tps]):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\ntokens/s",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, throughput_tps * 1.25)

    # (2) MFU. xticks = "cluster" / "L4:0" / "A10G:0" (NUM 은 같은 노드 내 GPU index;
    # PP=2 single GPU/node 환경에서는 모두 0 — single-node multi-GPU 실험으로
    # 확장될 때 의미 있어짐).
    mfu_vals = [mfu_cluster * 100, mfu_l4 * 100, mfu_a10g * 100]
    mfu_xticks = ["cluster", "L4:0", "A10G:0"]
    bars2 = axes[1].bar(mfu_xticks, mfu_vals,
                        color=["tab:gray", "tab:blue", "tab:orange"], width=0.6)
    axes[1].set_ylabel("MFU [%]")
    axes[1].set_title("Model FLOPs Utilization (MFU)")
    for b, v in zip(bars2, mfu_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set_ylim(0, max(50, max(mfu_vals) * 1.2))
    legend_handles = [
        plt.Line2D([0], [0], color="tab:gray", lw=8,
                   label=f"cluster ({cluster_achieved_tflops:.1f} / {cluster_peak_tflops} TFLOPS)"),
        plt.Line2D([0], [0], color="tab:blue", lw=8,
                   label=f"L4 ({achieved_l4_tflops:.1f} / {L4_BF16_PEAK_TFLOPS} TFLOPS)"),
        plt.Line2D([0], [0], color="tab:orange", lw=8,
                   label=f"A10G ({achieved_a10g_tflops:.1f} / {A10G_BF16_PEAK_TFLOPS} TFLOPS)"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=8)

    # (3) Average power per node — same xticks scheme.
    power_vals = [avg_power_l4, avg_power_a10g]
    power_xticks = ["L4:0", "A10G:0"]
    bars3 = axes[2].bar(power_xticks, power_vals,
                        color=["tab:blue", "tab:orange"], width=0.6)
    axes[2].set_ylabel("avg power [W]")
    axes[2].set_title("Average GPU power")
    for b, v in zip(bars3, power_vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} W",
                     ha="center", va="bottom", fontsize=9)
    # TDP — L4 datacenter spec 72W, A10G (AWS 변종, 일반 A10 의 300W 버전) 300W.
    L4_TDP_W = 72
    A10G_TDP_W = 300
    axes[2].axhline(L4_TDP_W, ls=":", color="tab:blue", alpha=0.4, linewidth=0.8)
    axes[2].axhline(A10G_TDP_W, ls=":", color="tab:orange", alpha=0.4, linewidth=0.8)
    axes[2].set_ylim(0, max(A10G_TDP_W, max(power_vals)) * 1.15)
    power_legend_handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=8, label=f"L4 (TDP {L4_TDP_W}W)"),
        plt.Line2D([0], [0], color="tab:orange", lw=8, label=f"A10G (TDP {A10G_TDP_W}W)"),
    ]
    axes[2].legend(handles=power_legend_handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "throughput_tokens_per_sec": throughput_tps,
        "pp_partition": pp_partition,
        "flops_per_stage_TF_per_step": [f / 1e12 for f in flops_per_stage],
        "achieved_tflops_cluster": cluster_achieved_tflops,
        "achieved_tflops_l4": achieved_l4_tflops,
        "achieved_tflops_a10g": achieved_a10g_tflops,
        "mfu_cluster_pct": mfu_cluster * 100,
        "mfu_l4_pct": mfu_l4 * 100,
        "mfu_a10g_pct": mfu_a10g * 100,
        "avg_power_l4_w": avg_power_l4,
        "avg_power_a10g_w": avg_power_a10g,
        "steady_step_sec": steady_total,
        "steady_fwdbwd_sec": steady_fwdbwd,
    }


# =============================================================================
# 이론치 vs 측정
# =============================================================================
def first_principles(meta: dict, steady_step_sec: float) -> dict:
    """이론 compute / comm 과 실측 wall-clock 을 비교."""
    mbs = meta["mbs"]; ga = meta["ga"]; seq = meta["seq_len"]
    hidden = LLAMA32_1B_HIDDEN
    bf16 = 2  # bytes

    tokens_per_step = mbs * seq * ga
    # 한 microbatch 의 활성화 cross-stage 전송 (forward 1 회) bytes:
    activation_per_mb = mbs * seq * hidden * bf16          # 8.39 MB at our config
    # 한 microbatch 의 grad cross-stage 전송 (backward 1 회) bytes:
    grad_per_mb = activation_per_mb
    # per iteration 양 방향 합:
    bytes_per_step = ga * (activation_per_mb + grad_per_mb)

    # 측정 throughput
    measured_tps = tokens_per_step / steady_step_sec if steady_step_sec else 0
    measured_avg_nic_MBps = bytes_per_step / 1e6 / steady_step_sec if steady_step_sec else 0

    # 이론 compute time per stage (1B 모델, 절반씩 PP=2 분할, BF16):
    # 전체 학습 FLOPs ≈ 6 N × token (forward+backward 합산, attn 무시).
    flops_total = 6 * LLAMA32_1B_PARAMS * tokens_per_step
    flops_per_stage = flops_total / 2
    compute_l4 = flops_per_stage / (L4_BF16_TFLOPS * 1e12)
    compute_a10g = flops_per_stage / (A10G_BF16_TFLOPS * 1e12)

    # 이론 comm time — burst 와 baseline 두 가지 모두 (5분 짧은 benchmark 에서는
    # burst 가 가능하지만, 시간 단위 학습에서는 baseline 156 MB/s 에 묶임).
    comm_at_burst = bytes_per_step / (ENA_BURST_GBPS * 1e9 / 8)
    comm_at_baseline = bytes_per_step / (ENA_BASELINE_GBPS * 1e9 / 8)

    return {
        "tokens_per_step": tokens_per_step,
        "bytes_per_step_GB": round(bytes_per_step / 1e9, 3),
        "measured_step_sec": round(steady_step_sec, 3) if steady_step_sec else None,
        "measured_throughput_tokens_per_sec": round(measured_tps, 1) if measured_tps else None,
        "measured_avg_nic_bytes_per_step_MBps": round(measured_avg_nic_MBps, 1) if measured_avg_nic_MBps else None,
        "theoretical_compute_per_stage_sec_l4": round(compute_l4, 2),
        "theoretical_compute_per_stage_sec_a10g": round(compute_a10g, 2),
        "theoretical_comm_at_ena_burst_sec": round(comm_at_burst, 2),
        "theoretical_comm_at_ena_baseline_sec": round(comm_at_baseline, 2),
        "implied_idle_sec": round(steady_step_sec - max(compute_l4, compute_a10g) - comm_at_burst, 2)
            if steady_step_sec else None,
    }


def write_stats(meta: dict, dcgm0, dcgm1, nic0_rate, nic1_rate, step_boundaries, fp: dict, path: Path):
    n0_active = active_window(dcgm0)
    n1_active = active_window(dcgm1)

    # NIC peak / mean (학습 active 구간만 — 임의 휴리스틱: NIC tx 가 1MB/s 이상)
    nic0_active = [r for r in nic0_rate if r["tx_MBps"] > 1.0 or r["rx_MBps"] > 1.0]
    nic1_active = [r for r in nic1_rate if r["tx_MBps"] > 1.0 or r["rx_MBps"] > 1.0]

    # 두 종류의 step time:
    # - fwdbwd_times = After train_batch_iter - Before train_batch_iter (forward+backward 만)
    # - step_total_times = After training_step - Before train_batch_iter (optimizer 포함)
    fwdbwd_times = [fb - b for b, fb, _ in step_boundaries]
    step_total_times = [se - b for b, _, se in step_boundaries]
    optimizer_times = [se - fb for _, fb, se in step_boundaries]

    lines = ["# Single-run benchmark 결과\n"]
    lines.append(f"**Config**: mbs={meta['mbs']}, ga={meta['ga']}, seq={meta['seq_len']}, "
                 f"GBS={meta['gbs_seqs']} sequences = {meta['gbs_tokens_per_step']} tokens / step\n")
    lines.append(f"**Wall clock**: 총 {meta['elapsed_sec']:.1f} s, train_steps={meta['train_steps']}\n")
    lines.append(f"**fwd/bwd time** (Before → After train_batch_iter): "
                 f"{[round(t, 2) for t in fwdbwd_times]}")
    lines.append(f"**step total** (Before → After training_step, optimizer 포함): "
                 f"{[round(t, 2) for t in step_total_times]}")
    lines.append(f"**optimizer + tied sync** (After train_batch_iter → After training_step): "
                 f"{[round(t, 2) for t in optimizer_times]}\n")

    if step_total_times:
        warmup_total = step_total_times[0]
        steady_total = mean_safe(step_total_times[STEADY_SLICE]) if len(step_total_times) > 1 else None
        steady_fwdbwd = mean_safe(fwdbwd_times[STEADY_SLICE]) if len(fwdbwd_times) > 1 else None
        steady_optim = mean_safe(optimizer_times[STEADY_SLICE]) if len(optimizer_times) > 1 else None
        lines.append(f"- step 1 (warmup) total: {warmup_total:.2f} s")
        if steady_total is not None:
            lines.append(f"- steady-state (step 2..{len(step_total_times)}) **total** 평균: "
                         f"**{steady_total:.2f} s** (fwd/bwd {steady_fwdbwd:.2f}s + "
                         f"optimizer/tied {steady_optim:.2f}s)\n")

    lines.append("\n## DCGM 평균 (학습 active 구간만)\n")
    lines.append("| 지표 | NODE 0 (L4) | NODE 1 (A10G) |")
    lines.append("|---|---:|---:|")
    if n0_active and n1_active:
        rows = [
            ("avg power [W]", "POWER", "{:.1f}"),
            ("max power [W]", "POWER", "{:.1f}", max),
            ("avg temp [°C]", "TMPTR", "{:.1f}"),
            ("max temp [°C]", "TMPTR", "{:.1f}", max),
            ("avg SMACT", "SMACT", "{:.3f}"),
            ("avg TENSO (BF16/FP16 matmul)", "TENSO", "{:.3f}"),
            ("avg DRAMA (DRAM BW use)", "DRAMA", "{:.3f}"),
        ]
        for row in rows:
            label, key, fmt = row[0], row[1], row[2]
            agg = row[3] if len(row) > 3 else mean_safe
            v0 = agg(r[key] for r in n0_active)
            v1 = agg(r[key] for r in n1_active)
            lines.append(f"| {label} | {fmt.format(v0)} | {fmt.format(v1)} |")

    lines.append("\n## NIC 실측 (`/proc/net/dev` 차분)\n")
    lines.append("| 지표 | NODE 0 (enp39s0) | NODE 1 (ens5) |")
    lines.append("|---|---:|---:|")
    if nic0_active and nic1_active:
        rows = [
            ("active samples count", None),
            ("avg TX [MB/s]", "tx_MBps", mean_safe),
            ("max TX [MB/s]", "tx_MBps", max),
            ("avg RX [MB/s]", "rx_MBps", mean_safe),
            ("max RX [MB/s]", "rx_MBps", max),
        ]
        lines[-1]  # avoid lint
        lines.append(f"| samples (≥1MB/s) | {len(nic0_active)} | {len(nic1_active)} |")
        for label, key, agg in rows[1:]:
            v0 = agg(r[key] for r in nic0_active)
            v1 = agg(r[key] for r in nic1_active)
            lines.append(f"| {label} | {v0:.1f} | {v1:.1f} |")

    lines.append("\n## 이론치 vs 실측\n")
    lines.append(f"- 한 step 의 이론 cross-stage 전송 (forward + backward) 합: "
                 f"`2 × ga × mbs × seq × hidden × 2B = 2 × {meta['ga']} × {meta['mbs']} × "
                 f"{meta['seq_len']} × {LLAMA32_1B_HIDDEN} × 2 = "
                 f"**{fp['bytes_per_step_GB']} GB / step**`")
    lines.append(f"- ENA bandwidth — burst {ENA_BURST_GBPS} Gbps 면 한 step 통신 "
                 f"`{fp['theoretical_comm_at_ena_burst_sec']} s` (24h 당 ~30분 한정), "
                 f"baseline {ENA_BASELINE_GBPS} Gbps (sustained) 면 "
                 f"`{fp['theoretical_comm_at_ena_baseline_sec']} s`. EFA 미지원 인스턴스라 "
                 f"NCCL Socket plugin (TCP) fallback.")
    lines.append(f"- 이론 compute (per stage, 6N × tokens / sustained TFLOPs 추정): "
                 f"L4 측 `{fp['theoretical_compute_per_stage_sec_l4']} s` "
                 f"({L4_BF16_TFLOPS} TFLOPs 기준), A10G 측 "
                 f"`{fp['theoretical_compute_per_stage_sec_a10g']} s` "
                 f"({A10G_BF16_TFLOPS} TFLOPs 기준).")
    lines.append(f"- 측정 steady step: `{fp['measured_step_sec']} s` → throughput "
                 f"`{fp['measured_throughput_tokens_per_sec']} tokens/s`.")
    if fp['measured_avg_nic_bytes_per_step_MBps']:
        lines.append(f"- 측정 평균 NIC bytes/step: "
                     f"`{fp['measured_avg_nic_bytes_per_step_MBps']} MB/s` "
                     f"(burst cap {ENA_BURST_GBPS*125:.0f} MB/s 대비 "
                     f"{fp['measured_avg_nic_bytes_per_step_MBps'] / (ENA_BURST_GBPS*125) * 100:.1f}%, "
                     f"baseline {ENA_BASELINE_GBPS*125:.0f} MB/s 대비 "
                     f"{fp['measured_avg_nic_bytes_per_step_MBps'] / (ENA_BASELINE_GBPS*125) * 100:.1f}%).")
    lines.append(f"- 측정과의 차이 (= NCCL P2P latency / pipeline bubble / kernel launch 등): "
                 f"`{fp['implied_idle_sec']} s` (burst comm 가정)")

    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        help="Benchmark output dir (defaults to last single_run dir under "
             "/opt/dlami/nvme/runs/<model>/<descriptor>/).",
    )
    ap.add_argument("--awk", default="/home/ubuntu/nanotron/examples/heterogeneous/dcgm_text_to_jsonl.awk")
    ap.add_argument(
        "--figures-dir",
        help="Figures output dir. Auto-derived from meta if omitted.",
    )
    args = ap.parse_args()

    # run-dir resolution: explicit arg > /opt/dlami/nvme/runs/*/* (most recent)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        candidates = sorted(Path("/opt/dlami/nvme/runs").glob("*/*"),
                            key=lambda p: p.stat().st_mtime, reverse=True) \
                     if Path("/opt/dlami/nvme/runs").exists() else []
        if not candidates:
            raise SystemExit("No --run-dir given and no /opt/dlami/nvme/runs/*/* found")
        run_dir = candidates[0]
        print(f"[plot] auto-detected run_dir = {run_dir}")

    awk = Path(args.awk)
    meta = json.loads((run_dir / "meta.json").read_text())

    # PP partition 정보 — meta 에 이미 있으면 그대로, 없으면 config 에서 읽어 합침.
    if "pp_layer_partition" not in meta and "config_path" in meta:
        cfg = Path(f"/home/ubuntu/nanotron/{meta['config_path']}")
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("pp_layer_partition:"):
                    rhs = stripped.split(":", 1)[1].strip().lstrip("[").rstrip("]")
                    meta["pp_layer_partition"] = [int(x.strip()) for x in rhs.split(",") if x.strip()]
                    break

    # figures-dir resolution: explicit arg > derived from meta (model/descriptor)
    if args.figures_dir:
        fig_dir = Path(args.figures_dir)
    else:
        model = meta.get("model", "unknown_model")
        descriptor = meta.get("descriptor", run_dir.name)
        fig_dir = Path("/home/ubuntu/nanotron/examples/heterogeneous/figures") / model / descriptor
        print(f"[plot] auto-derived figures_dir = {fig_dir}")
    fig_dir.mkdir(parents=True, exist_ok=True)

    dcgm0 = dcgm_jsonl(run_dir / "dcgm_node0.txt", awk)
    dcgm1 = dcgm_jsonl(run_dir / "dcgm_node1.txt", awk)
    nic0 = parse_nic(run_dir / "nic_node0.txt")
    nic1 = parse_nic(run_dir / "nic_node1.txt")
    step_boundaries = extract_step_boundaries(run_dir / "train_node0.log")

    plot_timeseries(meta, dcgm0, dcgm1, nic0, nic1, step_boundaries, fig_dir / "timeseries.png")
    print(f"saved {fig_dir / 'timeseries.png'}")

    flops_log = extract_flops_log(run_dir / "train_node0.log")
    if flops_log["per_module"]:
        print(f"loaded module FLOPs: {flops_log['per_module']}")
    bar_summary = plot_bars(meta, step_boundaries, dcgm0, dcgm1, flops_log, fig_dir / "bars.png")
    print(f"saved {fig_dir / 'bars.png'}")

    # 이론치 비교는 "true step total time" (optimizer 포함) 으로.
    step_total_times = [se - b for b, _, se in step_boundaries]
    steady_total = mean_safe(step_total_times[STEADY_SLICE]) if len(step_total_times) > 1 else None
    fp = first_principles(meta, steady_total or 0.0)

    nic0_rate = nic_to_rate(nic0)
    nic1_rate = nic_to_rate(nic1)
    write_stats(meta, dcgm0, dcgm1, nic0_rate, nic1_rate, step_boundaries, fp, fig_dir / "stats.md")
    (fig_dir / "stats.json").write_text(json.dumps({
        "meta": meta,
        "step_boundaries_unix": step_boundaries,
        "fwdbwd_times": [fb - b for b, fb, _ in step_boundaries],
        "step_total_times": step_total_times,
        "first_principles": fp,
        "bar_summary": bar_summary,
        "flops_log": flops_log,
    }, indent=2))
    print(f"saved {fig_dir / 'stats.md'} and stats.json")


if __name__ == "__main__":
    main()
