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

# GPU spec 테이블 — meta.json 의 ``gpu_node{0,1}`` 문자열에서 GPU type 을 파싱해 lookup.
# - peak_bf16_tflops: NVIDIA datasheet dense BF16/FP16 TC (sparsity 미적용).
# - sustained_bf16_tflops: 일반 dense matmul 에서 실측 가능한 sustained 값. peak 의 25-50%.
# - tdp_w: NVIDIA datasheet TDP (datacenter spec).
GPU_SPECS = {
    "L4":   {"peak_bf16_tflops": 121,   "sustained_bf16_tflops": 30,  "tdp_w": 72},
    "L40S": {"peak_bf16_tflops": 362,   "sustained_bf16_tflops": 100, "tdp_w": 350},
    "A10G": {"peak_bf16_tflops": 125,   "sustained_bf16_tflops": 70,  "tdp_w": 150},
    "A100": {"peak_bf16_tflops": 312,   "sustained_bf16_tflops": 200, "tdp_w": 400},
    "H100": {"peak_bf16_tflops": 989,   "sustained_bf16_tflops": 600, "tdp_w": 700},
    # V100 SXM2 32GB (Volta) — BF16 미지원이지만 FP16 Tensor Core 가 동일 throughput.
    # peak field 는 cluster 가 사용하는 dtype 의 peak TC TFLOPS 의미 (V100 cluster 는 FP16).
    "V100": {"peak_bf16_tflops": 125,   "sustained_bf16_tflops": 70,  "tdp_w": 300},
}
GPU_TYPE_PATTERN = re.compile(
    r"\b(H100|A100|A10G|L40S|L4|V100)\b"   # longest-first 조심
)


def parse_gpu_type(meta: dict, node_idx: int) -> str:
    """meta['gpu_node{idx}'] 문자열에서 GPU type 추출. 매칭 실패 시 'GPU' fallback."""
    s = meta.get(f"gpu_node{node_idx}", "")
    m = GPU_TYPE_PATTERN.search(s)
    return m.group(1) if m else "GPU"


def gpu_spec(gpu_type: str, key: str, default=0):
    return GPU_SPECS.get(gpu_type, {}).get(key, default)
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
_MEM_RE = re.compile(
    r"Memory usage:\s*([\d.]+)MiB\.\s*Peak allocated:?\s*([\d.]+)MiB\.?\s*Peak reserved:\s*([\d.]+)MiB"
)
# nanotron 의 INFO 라인은 ``[INFO|PP=N|TP=M]`` 으로 rank 정보 인코딩.
# log_memory (logging/base.py) 가 ``[rank N]`` 명시 접두사를 붙이는 경우도 있음.
_PPTP_RE = re.compile(r"\[INFO\|PP=(\d+)\|TP=(\d+)\]")
_RANK_PREFIX_RE = re.compile(r"\[rank (\d+)\]")


def extract_memory_peaks(log_path: Path) -> dict:
    """``Memory usage / Peak allocated / Peak reserved`` 줄을 (PP, TP) 별 파싱.

    rank 라벨 우선순위:
      1) ``[rank N]`` (logging/base.py 의 log_memory 명시 접두사 — single-GPU 호환)
      2) ``[INFO|PP=N|TP=M]`` (nanotron 의 표준 INFO 라인 형식)

    multi-GPU 노드 (TP=4 등) 에선 4 rank 가 각자 log → per-rank dict 에 4 entry.

    return:
      ``{"per_rank": {"pp0_tp0": {max_live_MiB, max_alloc_MiB, max_reserved_MiB}, ...},
         "max_live_MiB": <max across ranks>, ...}``
    """
    if not log_path.exists():
        return {}
    per_rank = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _MEM_RE.search(line)
        if not m:
            continue
        rank_match = _RANK_PREFIX_RE.search(line)
        if rank_match:
            rank = f"rank{rank_match.group(1)}"
        else:
            pptp_match = _PPTP_RE.search(line)
            if pptp_match:
                rank = f"pp{pptp_match.group(1)}_tp{pptp_match.group(2)}"
            else:
                rank = "unknown"
        live, alloc, reserved = float(m.group(1)), float(m.group(2)), float(m.group(3))
        cur = per_rank.setdefault(rank, {"max_live": 0.0, "max_alloc": 0.0, "max_reserved": 0.0})
        cur["max_live"] = max(cur["max_live"], live)
        cur["max_alloc"] = max(cur["max_alloc"], alloc)
        cur["max_reserved"] = max(cur["max_reserved"], reserved)
    if not per_rank:
        return {}
    out = {
        "per_rank": {
            rank: {
                "max_live_MiB": round(v["max_live"], 1),
                "max_alloc_MiB": round(v["max_alloc"], 1),
                "max_reserved_MiB": round(v["max_reserved"], 1),
            }
            for rank, v in sorted(per_rank.items())
        },
        "max_live_MiB": round(max(v["max_live"] for v in per_rank.values()), 1),
        "max_alloc_MiB": round(max(v["max_alloc"] for v in per_rank.values()), 1),
        "max_reserved_MiB": round(max(v["max_reserved"] for v in per_rank.values()), 1),
    }
    return out


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


def aggregate_dcgm_per_gpu(rows: List[dict]) -> dict:
    """Group DCGM samples by 'entity' (GPU 0..N-1), compute avg/max for key metrics.

    Caller responsible for time/active filtering before passing rows in.
    Returns {entity_str: {avg_power_W, max_power_W, avg_temp_C, max_temp_C,
                          avg_SMACT, avg_TENSO, avg_DRAMA, n_samples}}.
    """
    from collections import defaultdict
    by_gpu = defaultdict(list)
    for r in rows:
        ent = r.get("entity")
        if ent:
            by_gpu[ent].append(r)
    out = {}
    for gpu in sorted(by_gpu):
        xs = by_gpu[gpu]
        out[gpu] = {
            "avg_power_W": round(mean_safe(r.get("POWER") for r in xs) or 0, 2),
            "max_power_W": round(max((r.get("POWER", 0) for r in xs), default=0), 2),
            "avg_temp_C": round(mean_safe(r.get("TMPTR") for r in xs) or 0, 1),
            "max_temp_C": round(max((r.get("TMPTR", 0) for r in xs), default=0), 1),
            "avg_SMACT": round(mean_safe(r.get("SMACT") for r in xs) or 0, 4),
            "avg_TENSO": round(mean_safe(r.get("TENSO") for r in xs) or 0, 4),
            "avg_DRAMA": round(mean_safe(r.get("DRAMA") for r in xs) or 0, 4),
            "n_samples": len(xs),
        }
    return out


def parse_nvidia_smi_per_gpu(path: Path) -> List[List[float]]:
    """``ts gpu0,gpu1,gpu2,gpu3`` (4-GPU TP=4 노드의 comma-separated mem.used MiB).

    Returns: list of [gpu0, gpu1, gpu2, gpu3] per sample row.
    Single-GPU 노드면 [val] 만 들어 있어 호환됨.
    """
    if not path.exists():
        return []
    samples = []
    for line in path.read_text(errors="ignore").splitlines():
        toks = line.split()
        if len(toks) < 2:
            continue
        try:
            vals = [float(v) for v in toks[1].split(",")]
            samples.append(vals)
        except (ValueError, IndexError):
            continue
    return samples


def per_gpu_max_nvsmi(samples: List[List[float]]) -> List[float]:
    """각 GPU index 별로 모든 sample 의 max.

    sanity bound: nvidia-smi sampler 의 race condition 으로 newline 손실 시 mem 값이
    다음 라인 timestamp 와 concat 되어 ~10^15 같은 garbage 발생. GPU 메모리는
    물리적으로 100 GB 이하라 100,000 MiB 초과 sample 은 corrupt 로 간주하고 제외.
    """
    if not samples:
        return []
    n_gpus = max(len(s) for s in samples)
    SANITY_MAX_MIB = 100_000  # 100 GB / GPU 이상이면 corrupt
    return [
        round(max((s[i] for s in samples if i < len(s) and s[i] < SANITY_MAX_MIB), default=0), 1)
        for i in range(n_gpus)
    ]


# =============================================================================
# Plot helpers
# =============================================================================
def _group_dcgm_by_gpu(rows: List[dict]) -> dict:
    """{'GPU 0': [row, row, ...], 'GPU 1': [...], ...}"""
    from collections import defaultdict
    out = defaultdict(list)
    for r in rows:
        ent = r.get("entity")
        if ent:
            out[ent].append(r)
    return dict(sorted(out.items()))


def _add_step_boundary_lines(ax, step_boundaries, t0):
    """모든 axis 에 step boundary vertical line 추가."""
    if not step_boundaries:
        return
    for i, (b, fb_end, step_end) in enumerate(step_boundaries):
        ax.axvline(b - t0, color="green", linestyle="--", alpha=0.5, linewidth=0.7)
        ax.axvline(fb_end - t0, color="tab:orange", linestyle=":", alpha=0.4, linewidth=0.6)
        if i == len(step_boundaries) - 1:
            ax.axvline(step_end - t0, color="red", linestyle="--", alpha=0.7, linewidth=1.0)
        else:
            ax.axvline(step_end - t0, color="grey", linestyle="--", alpha=0.5, linewidth=0.7)


def _dcgm_align_t(rows, dcgm_start_unix, train_start_unix):
    """DCGM row → x (학습 시작 = 0).

    DCGM ``ts`` 두 가지 포맷 지원:
    - **wallclock** (값이 unix epoch, > 1e9): dcgm_dmon_wrap.sh prefix 모드 → 직접 빼면 됨
    - **sample_idx** (값이 작은 정수, < 1e9): 옛 awk 모드, dcgm_start 기준 상대 초로 가정
    """
    if not rows:
        return []
    is_wallclock = rows[0].get("ts", 0) > 1e9
    if is_wallclock:
        return [r["ts"] - train_start_unix for r in rows]
    if dcgm_start_unix and dcgm_start_unix > 0 and train_start_unix > 0:
        offset_sec = train_start_unix - dcgm_start_unix
        return [r["ts"] - offset_sec for r in rows]
    boost = [i for i, r in enumerate(rows) if r.get("SMCLK", 0) > 1500]
    offset = boost[0] if boost else 0
    return [r["ts"] - offset for r in rows]


def _dcgm_ts_to_unix(r, dcgm_start_unix):
    """DCGM row 의 ``ts`` field 를 unix epoch 로 변환 (포맷 자동 감지)."""
    ts = r.get("ts", 0)
    return ts if ts > 1e9 else dcgm_start_unix + ts


def plot_per_gpu_metric(
    meta: dict,
    dcgm0: List[dict], dcgm1: List[dict],
    step_boundaries,
    metric_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
):
    """N (NODE 0 GPU) + N (NODE 1 GPU) 의 metric 시계열을 2N subplot 으로.

    layout: 2 row × N col.  row 0 = NODE 0, row 1 = NODE 1.
    GPU type 라벨은 ``meta['gpu_node{0,1}']`` 에서 자동 추출.

    metric_key: DCGM column name (e.g. "POWER", "TMPTR", "SMACT", "TENSO", "DRAMA").
    """
    gpu_type_n0 = parse_gpu_type(meta, 0)
    gpu_type_n1 = parse_gpu_type(meta, 1)
    by_gpu0 = _group_dcgm_by_gpu(dcgm0)
    by_gpu1 = _group_dcgm_by_gpu(dcgm1)
    n_gpus = max(len(by_gpu0), len(by_gpu1), 1)

    fig, axes = plt.subplots(2, n_gpus, figsize=(4 * n_gpus, 5.5),
                              sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f"{title} | {meta.get('descriptor', '?')} | "
                 f"NODE 0 = {gpu_type_n0}, NODE 1 = {gpu_type_n1}", fontsize=11)

    train_start = step_boundaries[0][0] if step_boundaries else 0
    dcgm_start_n0 = float(meta.get("dcgm_start_ts_node0") or 0)
    dcgm_start_n1 = float(meta.get("dcgm_start_ts_node1") or 0)

    def _plot_one(ax, gpu_rows, dcgm_start, color, label):
        if not gpu_rows:
            ax.set_visible(False)
            return
        t = _dcgm_align_t(gpu_rows, dcgm_start, train_start)
        ax.plot(t, [r.get(metric_key, 0) for r in gpu_rows], color=color, linewidth=0.8)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        _add_step_boundary_lines(ax, step_boundaries, train_start)

    for col, (gpu, rows) in enumerate(by_gpu0.items()):
        _plot_one(axes[0][col], rows, dcgm_start_n0, "tab:blue", f"{gpu_type_n0} / {gpu}")
    for col, (gpu, rows) in enumerate(by_gpu1.items()):
        _plot_one(axes[1][col], rows, dcgm_start_n1, "tab:orange", f"{gpu_type_n1} / {gpu}")

    for ax in axes[0]:
        ax.set_ylabel(ylabel) if ax == axes[0][0] else None
    for ax in axes[1]:
        ax.set_xlabel("elapsed [s] from training start")
        ax.set_ylabel(ylabel) if ax == axes[1][0] else None

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_node_bandwidth(meta, dcgm0, dcgm1, nic0, nic1, step_boundaries, out_path: Path):
    """Node-level 시계열 — NIC (TX/RX), PCIe (TX/RX, GPU↔CPU mean across 4 GPU),
    DRAM bandwidth (mean across 4 GPU).

    PCIe / DRAM 은 GPU 별이지만 node 합/평균 으로 간략화. node 단위가 핵심.
    """
    gpu_type_n0 = parse_gpu_type(meta, 0)
    gpu_type_n1 = parse_gpu_type(meta, 1)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"Node-level bandwidth | mbs={meta['mbs']} ga={meta['ga']} "
                 f"seq={meta['seq_len']} | NODE 0 = {gpu_type_n0} / NODE 1 = {gpu_type_n1}", fontsize=11)

    train_start = step_boundaries[0][0] if step_boundaries else 0
    dcgm_start_n0 = float(meta.get("dcgm_start_ts_node0") or 0)
    dcgm_start_n1 = float(meta.get("dcgm_start_ts_node1") or 0)

    # Per-GPU rows → group by ts, then avg/sum.
    def _by_ts(rows):
        from collections import defaultdict
        out = defaultdict(list)
        for r in rows:
            out[r["ts"]].append(r)
        return out

    def _node_mean_at_each_ts(rows, key):
        """Returns (ts_list, mean_list) — at each ts, mean across all GPUs of that node."""
        tsd = _by_ts(rows)
        ts_sorted = sorted(tsd.keys())
        return ts_sorted, [mean_safe(r.get(key, 0) for r in tsd[t]) or 0 for t in ts_sorted]

    def _node_sum_at_each_ts(rows, key):
        """Sum across GPUs (PCIe/DRAM은 합으로 보는 게 직관적)."""
        tsd = _by_ts(rows)
        ts_sorted = sorted(tsd.keys())
        return ts_sorted, [sum(r.get(key, 0) for r in tsd[t]) for t in ts_sorted]

    # (1) DRAM bandwidth (mean across 4 GPU per node)
    if dcgm0:
        ts0, vals = _node_mean_at_each_ts(dcgm0, "DRAMA")
        x0 = _dcgm_align_t([{"ts": t} for t in ts0], dcgm_start_n0, train_start)
        axes[0].plot(x0, vals, label=f"NODE 0 ({gpu_type_n0}) mean", color="tab:blue")
    if dcgm1:
        ts1, vals = _node_mean_at_each_ts(dcgm1, "DRAMA")
        x1 = _dcgm_align_t([{"ts": t} for t in ts1], dcgm_start_n1, train_start)
        axes[0].plot(x1, vals, label=f"NODE 1 ({gpu_type_n1}) mean", color="tab:orange")
    axes[0].set_ylabel("DRAM bandwidth\n(DCGM DRAMA, 0..1; mean across GPUs)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # (2) PCIe TX+RX (sum across 4 GPU per node)
    if dcgm0:
        ts0, tx_sum = _node_sum_at_each_ts(dcgm0, "PCITX")
        _, rx_sum = _node_sum_at_each_ts(dcgm0, "PCIRX")
        x0 = _dcgm_align_t([{"ts": t} for t in ts0], dcgm_start_n0, train_start)
        axes[1].plot(x0, [v / 1e6 for v in tx_sum], label="NODE 0 PCITX (sum)",
                     color="tab:blue", linestyle="--")
        axes[1].plot(x0, [v / 1e6 for v in rx_sum], label="NODE 0 PCIRX (sum)",
                     color="tab:blue", linestyle=":")
    if dcgm1:
        ts1, tx_sum = _node_sum_at_each_ts(dcgm1, "PCITX")
        _, rx_sum = _node_sum_at_each_ts(dcgm1, "PCIRX")
        x1 = _dcgm_align_t([{"ts": t} for t in ts1], dcgm_start_n1, train_start)
        axes[1].plot(x1, [v / 1e6 for v in tx_sum], label="NODE 1 PCITX (sum)",
                     color="tab:orange", linestyle="--")
        axes[1].plot(x1, [v / 1e6 for v in rx_sum], label="NODE 1 PCIRX (sum)",
                     color="tab:orange", linestyle=":")
    axes[1].set_ylabel("PCIe (GPU↔CPU)\n[MB/s; sum across GPUs]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # (3) NIC actual bandwidth (per-node, /proc/net/dev, no per-GPU)
    nic0_rate = nic_to_rate(nic0)
    nic1_rate = nic_to_rate(nic1)
    if nic0_rate:
        x = [r["ts"] - train_start for r in nic0_rate]
        axes[2].plot(x, [r["tx_MBps"] for r in nic0_rate], label="NODE 0 TX",
                     color="tab:blue", linestyle="--")
        axes[2].plot(x, [r["rx_MBps"] for r in nic0_rate], label="NODE 0 RX",
                     color="tab:blue", linestyle=":")
    if nic1_rate:
        x = [r["ts"] - train_start for r in nic1_rate]
        axes[2].plot(x, [r["tx_MBps"] for r in nic1_rate], label="NODE 1 TX",
                     color="tab:orange", linestyle="--")
        axes[2].plot(x, [r["rx_MBps"] for r in nic1_rate], label="NODE 1 RX",
                     color="tab:orange", linestyle=":")
    axes[2].axhline(ENA_BURST_GBPS * 125, color="red", linestyle=":", alpha=0.5,
                    label=f"ENA burst ({ENA_BURST_GBPS} Gbps)")
    axes[2].axhline(ENA_BASELINE_GBPS * 125, color="purple", linestyle=":", alpha=0.5,
                    label=f"ENA baseline ({ENA_BASELINE_GBPS} Gbps)")
    axes[2].set_ylabel("NIC bandwidth\n(/proc/net/dev) [MB/s]")
    axes[2].set_xlabel("elapsed [s] from training start")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        _add_step_boundary_lines(ax, step_boundaries, train_start)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =============================================================================
# Plot (legacy combined timeseries — kept for backward compat, not used in main)
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
    gpu_type_n0 = parse_gpu_type(meta, 0)
    gpu_type_n1 = parse_gpu_type(meta, 1)
    fig, axes = plt.subplots(7, 1, figsize=(13, 16), sharex=True)
    fig.suptitle(f"Single run | mbs={meta['mbs']} ga={meta['ga']} seq={meta['seq_len']} "
                 f"(GBS={meta['gbs_seqs']} sequences = {meta['gbs_tokens_per_step']} tokens / step) | "
                 f"NODE 0 = {gpu_type_n0} / NODE 1 = {gpu_type_n1}")

    # 학습 시작 unix (없으면 NIC 첫 active sample fallback)
    if step_boundaries:
        train_start_unix = step_boundaries[0][0]
    elif nic0:
        train_start_unix = nic0[0]["ts"]
    else:
        train_start_unix = 0

    def _dcgm_align(rows, dcgm_start_unix):
        return _dcgm_align_t(rows, dcgm_start_unix, train_start_unix)

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
        axes[0].plot(t_d0, [r["POWER"] for r in dcgm0], label=f"NODE 0 ({gpu_type_n0})", color="tab:blue")
    if dcgm1:
        axes[0].plot(t_d1, [r["POWER"] for r in dcgm1], label=f"NODE 1 ({gpu_type_n1})", color="tab:orange")
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
    # TP-aware + node-aware peak: 각 노드의 GPU 수 = (pp/nodes) × tp = stages_per_node × tp.
    # 본 cluster 는 2 노드 fixed → 각 노드 GPU = pp × tp / 2.
    # 이전에는 stage 당 GPU (= tp) 만 곱해 PP>2 인 경우 cluster peak 가 실제의 절반/¼/⅛ 로 underestimate.
    tp = int(meta.get("tp", 1) or 1)
    pp = int(meta.get("pp", len(flops_per_stage)) or 1)
    gpu_type_n0 = parse_gpu_type(meta, 0)
    gpu_type_n1 = parse_gpu_type(meta, 1)
    peak_n0 = gpu_spec(gpu_type_n0, "peak_bf16_tflops", 1)
    peak_n1 = gpu_spec(gpu_type_n1, "peak_bf16_tflops", 1)
    n_n0_gpus = (pp * tp) // 2                  # NODE 0 의 총 GPU (stages_per_node × tp)
    n_n1_gpus = (pp * tp) // 2                  # NODE 1 의 총 GPU
    cluster_peak_tflops = n_n0_gpus * peak_n0 + n_n1_gpus * peak_n1
    mfu_cluster = cluster_achieved_tflops / cluster_peak_tflops

    # Per-node achieved TFLOPS = 노드 측 stage 들의 FLOPs 합 / step_time.
    # 1F1B steady-state 에서 모든 stage 의 wall-clock 동일하므로 step_time 으로 나눔.
    half = pp // 2
    flops_n0_stage = sum(flops_per_stage[:half]) if half else flops_per_stage[0]
    flops_n1_stage = sum(flops_per_stage[half:]) if half else flops_per_stage[-1]
    achieved_n0_stage_tflops = flops_n0_stage / steady_total / 1e12
    achieved_n1_stage_tflops = flops_n1_stage / steady_total / 1e12
    achieved_n0_per_gpu = achieved_n0_stage_tflops / n_n0_gpus
    achieved_n1_per_gpu = achieved_n1_stage_tflops / n_n1_gpus
    mfu_n0 = achieved_n0_per_gpu / peak_n0
    mfu_n1 = achieved_n1_per_gpu / peak_n1

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
                if steady_t0_unix <= _dcgm_ts_to_unix(r, dcgm_start_unix) <= steady_t1_unix]

    dcgm_start_n0 = float(meta.get("dcgm_start_ts_node0") or 0)
    dcgm_start_n1 = float(meta.get("dcgm_start_ts_node1") or 0)
    n0_steady = _dcgm_in_steady(dcgm0, dcgm_start_n0)
    n1_steady = _dcgm_in_steady(dcgm1, dcgm_start_n1)
    avg_power_n0 = mean_safe(r["POWER"] for r in n0_steady) if n0_steady else 0
    avg_power_n1 = mean_safe(r["POWER"] for r in n1_steady) if n1_steady else 0

    partition_str = "-".join(str(n) for n in pp_partition)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(f"Single run summary (step 2..{len(step_boundaries)} steady-state) | "
                 f"mbs={meta['mbs']} ga={meta['ga']} GBS={meta['gbs_seqs']} "
                 f"split={partition_str} | NODE 0 = {gpu_type_n0} / NODE 1 = {gpu_type_n1}")

    # (1) Throughput
    bars1 = axes[0].bar([f"split {partition_str}"], [throughput_tps],
                        color="tab:gray", width=0.5)
    axes[0].set_ylabel("throughput [tokens/sec]")
    axes[0].set_title("Throughput")
    for b, v in zip(bars1, [throughput_tps]):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}\ntokens/s",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, throughput_tps * 1.25)

    # (2) MFU per-GPU. 한 stage 의 TP GPU 가 균등 sharding → 모두 동일한 per-GPU MFU.
    # cluster MFU + (n_n0 + n_n1) GPU MFU = (1 + 2*tp) bars.
    mfu_vals = [mfu_cluster * 100]
    mfu_xticks = ["cluster"]
    mfu_colors = ["tab:gray"]
    for i in range(n_n0_gpus):
        mfu_vals.append(mfu_n0 * 100)
        mfu_xticks.append(f"{gpu_type_n0}:{i}")
        mfu_colors.append("tab:blue")
    for i in range(n_n1_gpus):
        mfu_vals.append(mfu_n1 * 100)
        mfu_xticks.append(f"{gpu_type_n1}:{i}")
        mfu_colors.append("tab:orange")
    bars2 = axes[1].bar(mfu_xticks, mfu_vals, color=mfu_colors, width=0.7)
    axes[1].set_ylabel("MFU [%]")
    axes[1].set_title("Model FLOPs Utilization (per-GPU)")
    for b, v in zip(bars2, mfu_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_ylim(0, max(50, max(mfu_vals) * 1.25))
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)
    legend_handles = [
        plt.Line2D([0], [0], color="tab:gray", lw=8,
                   label=f"cluster ({cluster_achieved_tflops:.1f}/{cluster_peak_tflops:.0f} TF/s)"),
        plt.Line2D([0], [0], color="tab:blue", lw=8,
                   label=f"{gpu_type_n0} each ({achieved_n0_per_gpu:.1f}/{peak_n0} TF/s)"),
        plt.Line2D([0], [0], color="tab:orange", lw=8,
                   label=f"{gpu_type_n1} each ({achieved_n1_per_gpu:.1f}/{peak_n1} TF/s)"),
    ]
    axes[1].legend(handles=legend_handles, loc="upper right", fontsize=7)

    # (3) Average power per GPU — TP=N 이라 NODE 0 GPU 0..N-1 + NODE 1 GPU 0..N-1 합 2N bars.
    by_gpu0 = _group_dcgm_by_gpu(n0_steady)
    by_gpu1 = _group_dcgm_by_gpu(n1_steady)

    power_xticks, power_vals, bar_colors = [], [], []
    for gpu, rows in by_gpu0.items():
        idx = gpu.split()[-1]                        # "GPU 0" → "0"
        power_xticks.append(f"{gpu_type_n0}:{idx}")
        power_vals.append(mean_safe(r["POWER"] for r in rows) or 0)
        bar_colors.append("tab:blue")
    for gpu, rows in by_gpu1.items():
        idx = gpu.split()[-1]
        power_xticks.append(f"{gpu_type_n1}:{idx}")
        power_vals.append(mean_safe(r["POWER"] for r in rows) or 0)
        bar_colors.append("tab:orange")
    if not power_vals:
        # fallback (single-GPU 환경 호환)
        power_xticks = [f"{gpu_type_n0}:0", f"{gpu_type_n1}:0"]
        power_vals = [avg_power_n0 or 0, avg_power_n1 or 0]
        bar_colors = ["tab:blue", "tab:orange"]

    bars3 = axes[2].bar(power_xticks, power_vals, color=bar_colors, width=0.7)
    axes[2].set_ylabel("avg power [W]")
    axes[2].set_title("Average GPU power (per-GPU)")
    for b, v in zip(bars3, power_vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=8)
    # TDP — meta 의 GPU type 별 NVIDIA datasheet TDP. axhline 으로 시각적 reference.
    tdp_n0 = gpu_spec(gpu_type_n0, "tdp_w", 0)
    tdp_n1 = gpu_spec(gpu_type_n1, "tdp_w", 0)
    if tdp_n0:
        axes[2].axhline(tdp_n0, ls=":", color="tab:blue", alpha=0.4, linewidth=0.8)
    if tdp_n1:
        axes[2].axhline(tdp_n1, ls=":", color="tab:orange", alpha=0.4, linewidth=0.8)
    axes[2].set_ylim(0, max(max(tdp_n0, tdp_n1, 1), max(power_vals)) * 1.15)
    axes[2].tick_params(axis="x", rotation=30, labelsize=8)
    power_legend_handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=8, label=f"{gpu_type_n0} (TDP {tdp_n0}W)"),
        plt.Line2D([0], [0], color="tab:orange", lw=8, label=f"{gpu_type_n1} (TDP {tdp_n1}W)"),
    ]
    axes[2].legend(handles=power_legend_handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return {
        "throughput_tokens_per_sec": throughput_tps,
        "pp_partition": pp_partition,
        "tp": tp,
        "gpu_type_node0": gpu_type_n0,
        "gpu_type_node1": gpu_type_n1,
        "flops_per_stage_TF_per_step": [f / 1e12 for f in flops_per_stage],
        "cluster_peak_tflops": cluster_peak_tflops,
        "achieved_tflops_cluster": cluster_achieved_tflops,
        "achieved_tflops_node0_stage": achieved_n0_stage_tflops,     # sum across TP GPUs
        "achieved_tflops_node1_stage": achieved_n1_stage_tflops,
        "achieved_tflops_node0_per_gpu": achieved_n0_per_gpu,
        "achieved_tflops_node1_per_gpu": achieved_n1_per_gpu,
        "mfu_cluster_pct": mfu_cluster * 100,
        "mfu_node0_pct": mfu_n0 * 100,                               # per-GPU
        "mfu_node1_pct": mfu_n1 * 100,                               # per-GPU
        "avg_power_node0_w": avg_power_n0,
        "avg_power_node1_w": avg_power_n1,
        "steady_step_sec": steady_total,
        "avg_latency_per_step_sec": steady_total,
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
    gpu_type_n0 = parse_gpu_type(meta, 0)
    gpu_type_n1 = parse_gpu_type(meta, 1)
    sustained_n0 = gpu_spec(gpu_type_n0, "sustained_bf16_tflops", 1)
    sustained_n1 = gpu_spec(gpu_type_n1, "sustained_bf16_tflops", 1)
    compute_n0 = flops_per_stage / (sustained_n0 * 1e12)
    compute_n1 = flops_per_stage / (sustained_n1 * 1e12)

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
        "gpu_type_node0": gpu_type_n0,
        "gpu_type_node1": gpu_type_n1,
        "theoretical_compute_per_stage_sec_node0": round(compute_n0, 2),
        "theoretical_compute_per_stage_sec_node1": round(compute_n1, 2),
        "theoretical_comm_at_ena_burst_sec": round(comm_at_burst, 2),
        "theoretical_comm_at_ena_baseline_sec": round(comm_at_baseline, 2),
        "implied_idle_sec": round(steady_step_sec - max(compute_n0, compute_n1) - comm_at_burst, 2)
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
            lines.append(f"- steady-state (step 2..{len(step_total_times)}) **avg latency per step**: "
                         f"**{steady_total:.2f} s** (fwd/bwd {steady_fwdbwd:.2f}s + "
                         f"optimizer/tied {steady_optim:.2f}s)\n")

    gpu_type_n0 = parse_gpu_type(meta, 0)
    gpu_type_n1 = parse_gpu_type(meta, 1)
    lines.append("\n## DCGM 평균 (학습 active 구간만)\n")
    lines.append(f"| 지표 | NODE 0 ({gpu_type_n0}) | NODE 1 ({gpu_type_n1}) |")
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
    sustained_n0 = gpu_spec(gpu_type_n0, "sustained_bf16_tflops", 0)
    sustained_n1 = gpu_spec(gpu_type_n1, "sustained_bf16_tflops", 0)
    lines.append(f"- 이론 compute (per stage, 6N × tokens / sustained TFLOPs 추정): "
                 f"{gpu_type_n0} 측 `{fp['theoretical_compute_per_stage_sec_node0']} s` "
                 f"({sustained_n0} TFLOPs 기준), {gpu_type_n1} 측 "
                 f"`{fp['theoretical_compute_per_stage_sec_node1']} s` "
                 f"({sustained_n1} TFLOPs 기준).")
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
        help="Raw benchmark output dir (defaults to most recent under "
             "/opt/dlami/nvme/runs/<cluster>/<model>/<descriptor>/).",
    )
    ap.add_argument("--awk", default="/home/ubuntu/nanotron/examples/heterogeneous/dcgm_text_to_jsonl.awk")
    ap.add_argument("--figures-dir", help="PNG out (auto if omitted).")
    ap.add_argument("--data-dir", help="stats/json out (auto if omitted).")
    args = ap.parse_args()

    # run-dir resolution: explicit arg > /opt/dlami/nvme/runs/*/*/* (most recent)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        candidates = []
        if Path("/opt/dlami/nvme/runs").exists():
            for p in Path("/opt/dlami/nvme/runs").rglob("meta.json"):
                candidates.append(p.parent)
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit("No --run-dir given and no /opt/dlami/nvme/runs/**/meta.json found")
        run_dir = candidates[0]
        print(f"[plot] auto-detected run_dir = {run_dir}")

    awk = Path(args.awk)
    meta = json.loads((run_dir / "meta.json").read_text())

    # PP partition 정보 — 우선순위: (1) meta 의 pp_layer_partition_str (benchmark
    # override 값, 항상 정확), (2) meta 의 pp_layer_partition list, (3) config 파일.
    # config 파일은 sweep 종료 후 baseline 으로 restore 되어 있어 잘못된 값을 줄 수
    # 있으니 마지막 수단으로만.
    if "pp_layer_partition_str" in meta:
        meta["pp_layer_partition"] = [int(x) for x in meta["pp_layer_partition_str"].split("-") if x]
    elif "pp_layer_partition" not in meta and "config_path" in meta:
        cfg = Path(f"/home/ubuntu/nanotron/{meta['config_path']}")
        if cfg.exists():
            for line in cfg.read_text().splitlines():
                stripped = line.strip()
                if stripped.startswith("pp_layer_partition:"):
                    rhs = stripped.split(":", 1)[1].strip().lstrip("[").rstrip("]")
                    meta["pp_layer_partition"] = [int(x.strip()) for x in rhs.split(",") if x.strip()]
                    break

    cluster = meta.get("cluster", "unknown_cluster")
    model = meta.get("model", "unknown_model")
    descriptor = meta.get("descriptor", run_dir.name)

    EX_HET = Path("/home/ubuntu/nanotron/examples/heterogeneous")
    fig_dir = Path(args.figures_dir) if args.figures_dir \
        else EX_HET / "figures" / cluster / model / descriptor
    data_dir = Path(args.data_dir) if args.data_dir \
        else EX_HET / "data" / cluster / model / descriptor
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot] figures → {fig_dir}")
    print(f"[plot] data    → {data_dir}")

    # OOM / failed run: skip plot 생성, OOM 표기만 stats.md 에 남기고 종료.
    # raw 는 디버깅 위해 항상 아카이브 (정상 run 과 동일 경로). archival 은 main 끝에서 일괄.
    oom = bool(meta.get("oom", False))
    completed_steps = int(meta.get("completed_steps", 0))
    if oom or completed_steps < 2:
        reason = "OOM" if oom else f"insufficient steps completed ({completed_steps})"
        (data_dir / "stats.md").write_text(
            f"# {descriptor} — FAILED ({reason})\n\n"
            f"- cluster: {cluster}\n- model: {model}\n- descriptor: {descriptor}\n"
            f"- oom: {oom}\n- completed_steps: {completed_steps}\n"
        )
        (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (data_dir / "stats.json").write_text(json.dumps({
            "meta": meta,
            "oom": oom,
            "completed_steps": completed_steps,
            "failed": True,
        }, indent=2))
        print(f"[plot] {descriptor}: skipped plotting ({reason})")
        # Archive raw 디버깅 위해 항상 (정상 run 과 동일).
        raw_root = Path("/home/ubuntu/nanotron/examples/heterogeneous/data/raw")
        raw_dst = raw_root / cluster / model / descriptor
        raw_dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in run_dir.iterdir():
            if f.is_file():
                dst = raw_dst / f.name
                dst.write_bytes(f.read_bytes())
                copied.append(f.name)
        print(f"[plot] archived raw → {raw_dst} ({len(copied)} files, failed run)")
        return

    dcgm0 = dcgm_jsonl(run_dir / "dcgm_node0.txt", awk)
    dcgm1 = dcgm_jsonl(run_dir / "dcgm_node1.txt", awk)
    nic0 = parse_nic(run_dir / "nic_node0.txt")
    nic1 = parse_nic(run_dir / "nic_node1.txt")
    step_boundaries = extract_step_boundaries(run_dir / "train_node0.log")

    # Per-GPU timeseries (4 L4 + 4 A10G subplots) — power, temp, SMACT, TENSO 분리.
    plot_per_gpu_metric(meta, dcgm0, dcgm1, step_boundaries,
                        "POWER", "GPU power [W]", "GPU power per GPU",
                        fig_dir / "power_per_gpu.png")
    plot_per_gpu_metric(meta, dcgm0, dcgm1, step_boundaries,
                        "TMPTR", "GPU temp [°C]", "GPU temperature per GPU",
                        fig_dir / "temp_per_gpu.png")
    plot_per_gpu_metric(meta, dcgm0, dcgm1, step_boundaries,
                        "SMACT", "SMACT (0..1)", "SM active fraction per GPU",
                        fig_dir / "smact_per_gpu.png")
    plot_per_gpu_metric(meta, dcgm0, dcgm1, step_boundaries,
                        "TENSO", "TENSO (0..1)", "Tensor-core utilization per GPU",
                        fig_dir / "tenso_per_gpu.png")
    # Node-level bandwidth (NIC, PCIe sum, DRAM mean) — per-GPU 대신 node 합/평균.
    plot_node_bandwidth(meta, dcgm0, dcgm1, nic0, nic1, step_boundaries,
                        fig_dir / "node_bandwidth.png")
    print(f"saved per-GPU metrics + node_bandwidth → {fig_dir}")

    flops_log = extract_flops_log(run_dir / "train_node0.log")
    if flops_log["per_module"]:
        print(f"loaded module FLOPs: {flops_log['per_module']}")

    # Per-rank memory peaks (nanotron의 log_memory + nvidia-smi 둘 다).
    gpu_type_n0 = parse_gpu_type(meta, 0)
    gpu_type_n1 = parse_gpu_type(meta, 1)
    memory_peaks = {
        "node0": extract_memory_peaks(run_dir / "train_node0.log"),
        "node1": extract_memory_peaks(run_dir / "train_node1.log"),
    }
    # Per-GPU nvidia-smi (comma-separated samples per node, TP=N 환경에서 N GPU).
    nvsmi_n0 = parse_nvidia_smi_per_gpu(run_dir / "nvidia_smi_node0.txt")
    nvsmi_n1 = parse_nvidia_smi_per_gpu(run_dir / "nvidia_smi_node1.txt")
    memory_peaks["nvidia_smi_per_gpu_node0"] = per_gpu_max_nvsmi(nvsmi_n0)
    memory_peaks["nvidia_smi_per_gpu_node1"] = per_gpu_max_nvsmi(nvsmi_n1)
    # node 전체의 max (TP=N GPU 중 어느 하나라도 가장 높았던 값) — OOM threshold 용도.
    memory_peaks["nvidia_smi_max_MiB_node0"] = (
        max(memory_peaks["nvidia_smi_per_gpu_node0"], default=0)
        or meta.get("nvidia_smi_max_used_MiB_node0", 0))
    memory_peaks["nvidia_smi_max_MiB_node1"] = (
        max(memory_peaks["nvidia_smi_per_gpu_node1"], default=0)
        or meta.get("nvidia_smi_max_used_MiB_node1", 0))
    if memory_peaks["node0"]:
        print(f"{gpu_type_n0:5s} peak reserved (PyTorch, per-rank max): {memory_peaks['node0'].get('max_reserved_MiB', 0):.0f} MiB | "
              f"nvidia-smi per-GPU max: {memory_peaks['nvidia_smi_per_gpu_node0']}")
    if memory_peaks["node1"]:
        print(f"{gpu_type_n1:5s} peak reserved (PyTorch, per-rank max): {memory_peaks['node1'].get('max_reserved_MiB', 0):.0f} MiB | "
              f"nvidia-smi per-GPU max: {memory_peaks['nvidia_smi_per_gpu_node1']}")

    # Per-GPU DCGM 통계 — bar_summary 의 node-level 평균과 별도로 GPU 별 분포
    # (avg/max power, temp, SMACT/TENSO/DRAMA) 보존. steady-state 동일 window 사용.
    n0_active = active_window(dcgm0)
    n1_active = active_window(dcgm1)
    dcgm_per_gpu = {
        "node0_active": aggregate_dcgm_per_gpu(n0_active),
        "node1_active": aggregate_dcgm_per_gpu(n1_active),
    }

    bar_summary = plot_bars(meta, step_boundaries, dcgm0, dcgm1, flops_log, fig_dir / "bars.png")
    print(f"saved {fig_dir / 'bars.png'}")

    # 이론치 비교는 "true step total time" (optimizer 포함) 으로.
    step_total_times = [se - b for b, _, se in step_boundaries]
    steady_total = mean_safe(step_total_times[STEADY_SLICE]) if len(step_total_times) > 1 else None
    fp = first_principles(meta, steady_total or 0.0)

    nic0_rate = nic_to_rate(nic0)
    nic1_rate = nic_to_rate(nic1)
    write_stats(meta, dcgm0, dcgm1, nic0_rate, nic1_rate, step_boundaries, fp, data_dir / "stats.md")
    (data_dir / "stats.json").write_text(json.dumps({
        "meta": meta,
        "step_boundaries_unix": step_boundaries,
        "fwdbwd_times": [fb - b for b, fb, _ in step_boundaries],
        "step_total_times": step_total_times,
        "first_principles": fp,
        "bar_summary": bar_summary,
        "flops_log": flops_log,
        "memory_peaks": memory_peaks,
        "dcgm_per_gpu": dcgm_per_gpu,
    }, indent=2))
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {data_dir / 'stats.md'} and stats.json + meta.json")

    # Archive raw: data/raw/<cluster>/<model>/<descriptor>/ (dev EBS 영속).
    # /opt/dlami/nvme/runs/ 는 instance ephemeral 이라 stop 시 삭제 — 영속 archive 필요.
    cluster = meta.get("cluster", "unknown_cluster")
    model = meta.get("model", "unknown_model")
    descriptor = meta.get("descriptor", run_dir.name)
    raw_root = Path("/home/ubuntu/nanotron/examples/heterogeneous/data/raw")
    raw_dst = raw_root / cluster / model / descriptor
    raw_dst.mkdir(parents=True, exist_ok=True)
    import shutil
    copied = []
    for f in run_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, raw_dst / f.name)
            copied.append(f.name)
    print(f"archived raw → {raw_dst} ({len(copied)} files)")


if __name__ == "__main__":
    main()
