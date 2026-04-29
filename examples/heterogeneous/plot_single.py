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
LLAMA32_1B_PARAMS = 1.236e9      # ~1.24 B
# L4 / A10G 의 BF16 dense throughput (실측 기준 — project_background.md §6.1.1).
L4_BF16_TFLOPS = 30
A10G_BF16_TFLOPS = 70  # NVIDIA spec (FP16); A10G 는 BF16 도 동일.
ENA_BANDWIDTH_GBPS = 10  # g5/g6.xlarge baseline


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
    return [a - b for b, a in extract_step_boundaries(log_path)]


def extract_step_boundaries(log_path: Path) -> List[tuple]:
    """``[(before_ts, after_ts), ...]`` 의 unix wallclock 쌍."""
    if not log_path.exists():
        return []
    pairs = []
    cur = None
    year = datetime.now(timezone.utc).year
    for line in log_path.read_text(errors="ignore").splitlines():
        if "Before train_batch_iter" in line:
            cur = parse_log_ts(line, year)
        elif "After train_batch_iter" in line and cur is not None:
            after = parse_log_ts(line, year)
            if after is not None:
                pairs.append((cur, after))
            cur = None
    return pairs


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
    """모든 subplot 의 x=0 을 ``첫 학습 step 의 Before train_batch_iter``
    시점으로 정렬한다. 이렇게 하면 학습 시작 / 각 iter 종료 / 학습 종료를
    같은 x 좌표로 표시 가능.

    DCGM 은 절대시간이 없는 row_index 만 가지므로 SMCLK 가 처음으로 boost
    clock (>1500 MHz) 에 도달한 row 를 학습 시작 시점으로 간주해 align.
    NIC 은 unix wallclock 이 있으므로 직접 ``training_start_unix`` 와 빼면 됨.
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

    def _dcgm_align(rows):
        """DCGM row_index → t (학습 시작 = 0). 첫 SMCLK boost row 를 학습 시작으로."""
        if not rows:
            return []
        boost = [i for i, r in enumerate(rows) if r.get("SMCLK", 0) > 1500]
        offset = boost[0] if boost else 0
        return [r["ts"] - offset for r in rows]

    def _nic_align(rows):
        return [r["ts"] - train_start_unix for r in rows]

    t_d0 = _dcgm_align(dcgm0)
    t_d1 = _dcgm_align(dcgm1)
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
    # ENA 1.25 GB/s = 1250 MB/s 한계 표시
    axes[6].axhline(1250, color="red", linestyle=":", alpha=0.5,
                    label=f"ENA cap (~{ENA_BANDWIDTH_GBPS} Gbps)")
    axes[6].set_ylabel("NIC bandwidth\n(/proc/net/dev) [MB/s]")
    axes[6].set_xlabel("elapsed [s] from training start (first 'Before train_batch_iter')")
    axes[6].legend(loc="upper right", fontsize=8)
    axes[6].grid(True, alpha=0.3)

    # 학습 step boundary 를 모든 axis 에 vertical dashed line 으로.
    # 색은 시작/종료 구분: training start = green, iter end = grey, training end = red.
    if step_boundaries:
        for ax in axes:
            # iter 1 시작 = t=0 (이미 정렬됨)
            ax.axvline(0, color="green", linestyle="--", alpha=0.6, linewidth=0.8)
            # 각 iter 의 종료
            for i, (b, a) in enumerate(step_boundaries):
                rel_end = a - step_boundaries[0][0]
                ax.axvline(rel_end, color="grey", linestyle="--", alpha=0.4, linewidth=0.6)
            # training 전체 종료 = 마지막 iter end
            last_end = step_boundaries[-1][1] - step_boundaries[0][0]
            ax.axvline(last_end, color="red", linestyle="--", alpha=0.6, linewidth=0.8)

        # 첫 axis 에만 라벨로 표시
        axes[0].annotate("training start", xy=(0, axes[0].get_ylim()[1]),
                         xytext=(2, axes[0].get_ylim()[1] * 0.95),
                         fontsize=8, color="green")
        axes[0].annotate(f"training end\n(after step {len(step_boundaries)})",
                         xy=(last_end, axes[0].get_ylim()[1]),
                         xytext=(last_end + 2, axes[0].get_ylim()[1] * 0.85),
                         fontsize=8, color="red")
        # iter 종료는 너무 빽빽할 수 있어 첫 두 개만 라벨
        for i in range(min(2, len(step_boundaries))):
            rel_end = step_boundaries[i][1] - step_boundaries[0][0]
            axes[2].annotate(f"iter {i+1} end", xy=(rel_end, 0.5),
                             xytext=(rel_end + 1, 0.4 - 0.1 * i),
                             fontsize=7, color="grey", alpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


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

    # 이론 comm time at full ENA bandwidth (1.25 GB/s):
    comm_at_ena = bytes_per_step / (ENA_BANDWIDTH_GBPS * 1e9 / 8)

    return {
        "tokens_per_step": tokens_per_step,
        "bytes_per_step_GB": round(bytes_per_step / 1e9, 3),
        "measured_step_sec": round(steady_step_sec, 3) if steady_step_sec else None,
        "measured_throughput_tokens_per_sec": round(measured_tps, 1) if measured_tps else None,
        "measured_avg_nic_bytes_per_step_MBps": round(measured_avg_nic_MBps, 1) if measured_avg_nic_MBps else None,
        "theoretical_compute_per_stage_sec_l4": round(compute_l4, 2),
        "theoretical_compute_per_stage_sec_a10g": round(compute_a10g, 2),
        "theoretical_comm_at_ena_full_sec": round(comm_at_ena, 2),
        "implied_idle_sec": round(steady_step_sec - max(compute_l4, compute_a10g) - comm_at_ena, 2)
            if steady_step_sec else None,
    }


def write_stats(meta: dict, dcgm0, dcgm1, nic0_rate, nic1_rate, step_times, fp: dict, path: Path):
    n0_active = active_window(dcgm0)
    n1_active = active_window(dcgm1)

    # NIC peak / mean (학습 active 구간만 — 임의 휴리스틱: NIC tx 가 1MB/s 이상)
    nic0_active = [r for r in nic0_rate if r["tx_MBps"] > 1.0 or r["rx_MBps"] > 1.0]
    nic1_active = [r for r in nic1_rate if r["tx_MBps"] > 1.0 or r["rx_MBps"] > 1.0]

    lines = ["# Single-run benchmark 결과\n"]
    lines.append(f"**Config**: mbs={meta['mbs']}, ga={meta['ga']}, seq={meta['seq_len']}, "
                 f"GBS={meta['gbs_seqs']} sequences = {meta['gbs_tokens_per_step']} tokens / step\n")
    lines.append(f"**Wall clock**: 총 {meta['elapsed_sec']:.1f} s, train_steps={meta['train_steps']}\n")
    lines.append(f"**Step times** (Before/After train_batch_iter 차): {[round(t, 2) for t in step_times]}\n")

    if step_times:
        warmup = step_times[0]
        steady = mean_safe(step_times[1:]) if len(step_times) > 1 else None
        lines.append(f"- step 1 (warmup): {warmup:.2f} s")
        if steady is not None:
            lines.append(f"- steady-state (step 2..N) 평균: **{steady:.2f} s**\n")

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
    lines.append(f"- ENA {ENA_BANDWIDTH_GBPS} Gbps 풀로 다 쓰면 한 step 통신만 "
                 f"`{fp['theoretical_comm_at_ena_full_sec']} s` 가 한계.")
    lines.append(f"- 이론 compute (per stage, 6N × tokens / TFLOPs): L4 측 "
                 f"`{fp['theoretical_compute_per_stage_sec_l4']} s`, A10G 측 "
                 f"`{fp['theoretical_compute_per_stage_sec_a10g']} s`.")
    lines.append(f"- 측정 steady step: `{fp['measured_step_sec']} s` → throughput "
                 f"`{fp['measured_throughput_tokens_per_sec']} tokens/s`.")
    lines.append(f"- 측정 평균 NIC bytes/step: `{fp['measured_avg_nic_bytes_per_step_MBps']} MB/s` "
                 f"(이론 ENA cap 대비 {fp['measured_avg_nic_bytes_per_step_MBps'] / 1250 * 100:.1f}%)."
                 if fp['measured_avg_nic_bytes_per_step_MBps'] else "")
    lines.append(f"- 이론적으로 compute + comm 만 쓰면 step 시간 ≥ "
                 f"`max(compute_l4, compute_a10g) + comm = "
                 f"max({fp['theoretical_compute_per_stage_sec_l4']}, "
                 f"{fp['theoretical_compute_per_stage_sec_a10g']}) + "
                 f"{fp['theoretical_comm_at_ena_full_sec']} = "
                 f"{max(fp['theoretical_compute_per_stage_sec_l4'], fp['theoretical_compute_per_stage_sec_a10g']) + fp['theoretical_comm_at_ena_full_sec']:.2f} s`.")
    lines.append(f"- 측정과의 차이 (= 통신 latency / NCCL 오버헤드 / kernel launch 등으로 추정): "
                 f"`{fp['implied_idle_sec']} s`")

    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/opt/dlami/nvme/single_run")
    ap.add_argument("--awk", default="/home/ubuntu/nanotron/examples/heterogeneous/dcgm_text_to_jsonl.awk")
    ap.add_argument(
        "--figures-dir",
        default="/home/ubuntu/nanotron/examples/heterogeneous/figures/single_run",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    awk = Path(args.awk)

    meta = json.loads((run_dir / "meta.json").read_text())
    dcgm0 = dcgm_jsonl(run_dir / "dcgm_node0.txt", awk)
    dcgm1 = dcgm_jsonl(run_dir / "dcgm_node1.txt", awk)
    nic0 = parse_nic(run_dir / "nic_node0.txt")
    nic1 = parse_nic(run_dir / "nic_node1.txt")
    step_times = extract_step_timings(run_dir / "train_node0.log")

    step_boundaries = extract_step_boundaries(run_dir / "train_node0.log")
    plot_timeseries(meta, dcgm0, dcgm1, nic0, nic1, step_boundaries, fig_dir / "timeseries.png")
    print(f"saved {fig_dir / 'timeseries.png'}")

    steady = mean_safe(step_times[1:]) if len(step_times) > 1 else None
    fp = first_principles(meta, steady or 0.0)

    nic0_rate = nic_to_rate(nic0)
    nic1_rate = nic_to_rate(nic1)
    write_stats(meta, dcgm0, dcgm1, nic0_rate, nic1_rate, step_times, fp, fig_dir / "stats.md")
    (fig_dir / "stats.json").write_text(json.dumps({
        "meta": meta,
        "step_times": step_times,
        "first_principles": fp,
    }, indent=2))
    print(f"saved {fig_dir / 'stats.md'} and stats.json")


if __name__ == "__main__":
    main()
