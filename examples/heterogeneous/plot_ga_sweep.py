"""ga sweep 결과 시각화 + 통계.

[디렉토리 구조]
- 입력 raw (DCGM text, train log, meta.json) : ``--sweep-dir`` (보통
  ``/opt/dlami/nvme/ga_sweep/``) 의 flat layout (``dcgm_node{0,1}_ga<N>.txt``
  등) 또는 per-ga 서브디렉토리. 두 layout 모두 지원.
- 출력 figure : ``--figures-dir`` (default
  ``examples/heterogeneous/figures/``) 아래에:
    figures/ga_sweep/ga<N>/timeseries.png      ← 각 config 별 시계열
    figures/ga_sweep/summary/step_time.png      ← ga 축 비교
    figures/ga_sweep/summary/throughput.png
    figures/ga_sweep/summary/power.png
    figures/ga_sweep/summary/activity.png
    figures/ga_sweep/summary/stats.json
    figures/ga_sweep/summary/stats.md

  ``examples/heterogeneous/figures/`` 는 ``.gitignore`` 에 등록되어 git
  추적되지 않음.

Usage:
    uv run --no-project --with matplotlib python \\
        examples/heterogeneous/plot_ga_sweep.py

학습 hyper-parameter 는 config 와 합치되 — micro_batch_size=2, seq_len=1024 가
고정인 가정. ga 만 변경.
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

# 한국어 라벨이 ``DejaVu Sans`` (matplotlib default) 에 없는 glyph 라 ▭ 로
# 깨지므로 Noto CJK 를 명시 등록.
_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_NOTO)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass  # font 가 없는 환경에서는 그냥 영어 fallback


# 학습 config 와 일치해야 함 (config_llama32_1b_alpaca_pp2.yaml).
MICRO_BATCH_SIZE = 2
SEQ_LEN = 1024


# =============================================================================
# Log / DCGM 파싱
# =============================================================================
_TS_RE = re.compile(r"^\[?\d?[0-9;]*m?(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})")


def parse_log_ts(line: str, year: int) -> float | None:
    m = _TS_RE.match(line.lstrip("\x1b").lstrip())
    if not m:
        return None
    mon, day, h, mn, s = map(int, m.groups())
    return datetime(year, mon, day, h, mn, s, tzinfo=timezone.utc).timestamp()


def extract_step_timings(log_path: Path) -> List[float]:
    """``Before train_batch_iter`` ↔ ``After train_batch_iter`` 시간 차."""
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
    return [a - b for b, a in pairs]


def dcgm_jsonl(txt: Path, awk: Path) -> List[dict]:
    if not txt.exists():
        return []
    out = subprocess.run(
        ["awk", "-f", str(awk), str(txt)],
        capture_output=True, text=True, check=True,
    ).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def active_window(rows: List[dict]) -> List[dict]:
    """idle 구간 제외 — SMCLK > 1500MHz 일 때만 학습 active 로 간주."""
    return [r for r in rows if r.get("SMCLK", 0) > 1500]


def relative_time(rows: List[dict]) -> List[float]:
    """첫 sample 의 ts 를 0 으로 둔 상대 초."""
    if not rows:
        return []
    t0 = rows[0]["ts"]
    return [r["ts"] - t0 for r in rows]


# =============================================================================
# Plot 1: 시계열 (한 ga)
# =============================================================================
def plot_timeseries(ga: int, rows0: List[dict], rows1: List[dict], out_path: Path):
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    fig.suptitle(f"ga={ga} | DCGM time-series (NODE 0 = L4, NODE 1 = A10G)")

    if rows0:
        t0 = relative_time(rows0)
        axes[0].plot(t0, [r["POWER"] for r in rows0], label="NODE 0 (L4)", color="tab:blue")
    if rows1:
        t1 = relative_time(rows1)
        axes[0].plot(t1, [r["POWER"] for r in rows1], label="NODE 1 (A10G)", color="tab:orange")
    axes[0].set_ylabel("POWER [W]")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    if rows0:
        axes[1].plot(t0, [r["SMACT"] for r in rows0], color="tab:blue")
    if rows1:
        axes[1].plot(t1, [r["SMACT"] for r in rows1], color="tab:orange")
    axes[1].set_ylabel("SMACT [0..1]")
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

    if rows0:
        axes[2].plot(t0, [r["TENSO"] for r in rows0], color="tab:blue")
    if rows1:
        axes[2].plot(t1, [r["TENSO"] for r in rows1], color="tab:orange")
    axes[2].set_ylabel("TENSO [0..1]")
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))

    if rows0:
        axes[3].plot(t0, [r["DRAMA"] for r in rows0], color="tab:blue")
    if rows1:
        axes[3].plot(t1, [r["DRAMA"] for r in rows1], color="tab:orange")
    axes[3].set_ylabel("DRAMA [0..1]")
    axes[3].grid(True, alpha=0.3)

    # PCIe 는 byte counter (per-sample delta) — MB/s 로 환산해서 시계열.
    # delta sample 간격 = 1 sec (DCGM dmon -d 1000) → bytes/s == bytes per sample.
    if rows0:
        axes[4].plot(t0, [r["PCITX"] / 1e6 for r in rows0], label="NODE 0 PCITX", color="tab:blue", linestyle="--")
        axes[4].plot(t0, [r["PCIRX"] / 1e6 for r in rows0], label="NODE 0 PCIRX", color="tab:blue", linestyle=":")
    if rows1:
        axes[4].plot(t1, [r["PCITX"] / 1e6 for r in rows1], label="NODE 1 PCITX", color="tab:orange", linestyle="--")
        axes[4].plot(t1, [r["PCIRX"] / 1e6 for r in rows1], label="NODE 1 PCIRX", color="tab:orange", linestyle=":")
    axes[4].set_ylabel("PCIe [MB/s]")
    axes[4].set_xlabel("relative time [s] (한 노드 기준)")
    axes[4].legend(loc="upper right", fontsize=8)
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# =============================================================================
# Plot 2: ga 별 summary
# =============================================================================
def plot_summary(stats: List[dict], out_dir: Path):
    gas = [s["ga"] for s in stats]

    # (a) steady step time + ideal linear
    fig, ax = plt.subplots(figsize=(8, 5))
    steady = [s["steady_state_sec"] for s in stats]
    ax.plot(gas, steady, "o-", label="실측 steady-state step")
    if steady[0]:
        ideal = [steady[0] * g / gas[0] for g in gas]
        ax.plot(gas, ideal, "k--", alpha=0.5, label=f"ga 에 정확히 비례 (ga={gas[0]} 기준)")
    ax.set_xlabel("ga (batch_accumulation_per_replica)")
    ax.set_ylabel("steady-state step time [s]")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Step time vs ga — 선형이면 통신/compute 가 ga 에 비례")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_step_time.png", dpi=120)
    plt.close(fig)

    # (b) throughput
    fig, ax = plt.subplots(figsize=(8, 5))
    tps = [(MICRO_BATCH_SIZE * SEQ_LEN * s["ga"]) / s["steady_state_sec"] if s["steady_state_sec"] else 0 for s in stats]
    ax.plot(gas, tps, "o-")
    ax.set_xlabel("ga")
    ax.set_ylabel("throughput [tokens/s]")
    ax.set_xscale("log", base=2)
    ax.set_title(f"Throughput (mbs={MICRO_BATCH_SIZE}, seq={SEQ_LEN}, GBS=mbs·ga)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "summary_throughput.png", dpi=120)
    plt.close(fig)

    # (c) avg power per node
    fig, ax = plt.subplots(figsize=(8, 5))
    pwr0 = [s["node0"]["active_avg_POWER_W"] for s in stats]
    pwr1 = [s["node1"]["active_avg_POWER_W"] for s in stats]
    ax.plot(gas, pwr0, "o-", color="tab:blue", label="NODE 0 (L4) — TDP 72W")
    ax.plot(gas, pwr1, "o-", color="tab:orange", label="NODE 1 (A10G) — TDP 150W")
    ax.set_xlabel("ga")
    ax.set_ylabel("avg POWER [W] (학습 중 active 구간만)")
    ax.set_xscale("log", base=2)
    ax.set_title("Power draw vs ga")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "summary_power.png", dpi=120)
    plt.close(fig)

    # (d) SMACT / TENSO per node
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in [(axes[0], "active_avg_SMACT", "SMACT (SM 활성도)"), (axes[1], "active_avg_TENSO", "TENSO (Tensor core 활성도)")]:
        v0 = [s["node0"][key] for s in stats]
        v1 = [s["node1"][key] for s in stats]
        ax.plot(gas, v0, "o-", color="tab:blue", label="NODE 0 (L4)")
        ax.plot(gas, v1, "o-", color="tab:orange", label="NODE 1 (A10G)")
        ax.set_xlabel("ga")
        ax.set_ylabel(f"avg {key.split('_')[-1]} [0..1]")
        ax.set_xscale("log", base=2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("GPU 활성도 vs ga — 1 에 가까울수록 GPU 가 idle 시간 없이 활용됨")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_activity.png", dpi=120)
    plt.close(fig)


# =============================================================================
# 통계 dict 생성 (analyze_ga_sweep 와 비슷, 더 풍부)
# =============================================================================
def mean_safe(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def summarize_node(rows: List[dict]) -> dict:
    active = active_window(rows)
    if not active:
        return {k: None for k in ["active_avg_POWER_W", "active_avg_SMACT", "active_avg_SMOCC",
                                   "active_avg_TENSO", "active_avg_DRAMA", "active_avg_GRACT",
                                   "active_max_POWER_W", "active_n_samples",
                                   "active_total_PCITX_GB", "active_total_PCIRX_GB"]}
    return {
        "active_n_samples": len(active),
        "active_avg_POWER_W": round(mean_safe(r["POWER"] for r in active), 2),
        "active_max_POWER_W": round(max(r["POWER"] for r in active), 2),
        "active_avg_SMACT": round(mean_safe(r["SMACT"] for r in active), 4),
        "active_avg_SMOCC": round(mean_safe(r["SMOCC"] for r in active), 4),
        "active_avg_TENSO": round(mean_safe(r["TENSO"] for r in active), 4),
        "active_avg_DRAMA": round(mean_safe(r["DRAMA"] for r in active), 4),
        "active_avg_GRACT": round(mean_safe(r["GRACT"] for r in active), 4),
        "active_total_PCITX_GB": round(sum(r["PCITX"] for r in active) / 1e9, 3),
        "active_total_PCIRX_GB": round(sum(r["PCIRX"] for r in active) / 1e9, 3),
    }


def write_markdown(stats: List[dict], path: Path):
    lines = ["# ga sweep 결과 요약\n"]
    lines.append("학습 hyper-parameter: micro_batch_size=2, sequence_length=1024, train_steps=5, "
                 "PP=2 (NODE 0 = L4 / NODE 1 = A10G), checkpoint save 비활성.\n")
    lines.append("학습 step 1 은 NCCL warm-up + cuBLAS context 초기화로 "
                 "steady-state 보다 ~30s 더 걸림. ``steady_state_sec`` 은 step 2..N 평균.\n")
    lines.append("\n## 표\n")
    lines.append("| ga | elapsed[s] | warmup[s] | steady[s] | throughput[tokens/s] | "
                 "L4 power[W] | A10G power[W] | L4 SMACT | A10G SMACT | L4 TENSO | A10G TENSO | "
                 "L4 PCITX[GB] | A10G PCITX[GB] |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        tps = (MICRO_BATCH_SIZE * SEQ_LEN * s["ga"]) / s["steady_state_sec"] if s["steady_state_sec"] else 0
        n0 = s["node0"]; n1 = s["node1"]
        lines.append(f"| {s['ga']} | {s['elapsed_sec']:.1f} | {s['warmup_sec']:.1f} | "
                     f"{s['steady_state_sec']:.2f} | {tps:.0f} | "
                     f"{n0.get('active_avg_POWER_W') or 0:.1f} | {n1.get('active_avg_POWER_W') or 0:.1f} | "
                     f"{n0.get('active_avg_SMACT') or 0:.3f} | {n1.get('active_avg_SMACT') or 0:.3f} | "
                     f"{n0.get('active_avg_TENSO') or 0:.3f} | {n1.get('active_avg_TENSO') or 0:.3f} | "
                     f"{n0.get('active_total_PCITX_GB') or 0:.2f} | "
                     f"{n1.get('active_total_PCITX_GB') or 0:.2f} |")
    path.write_text("\n".join(lines))


def _resolve_raw_paths(sweep: Path, ga: int) -> dict:
    """flat layout (``dcgm_node0_ga4.txt``) 또는 per-ga subdir layout
    (``ga4/dcgm_node0.txt``) 둘 다 지원."""
    sub = sweep / f"ga{ga}"
    if sub.exists():
        return {
            "dcgm0": sub / "dcgm_node0.txt",
            "dcgm1": sub / "dcgm_node1.txt",
            "log0": sub / "train_node0.log",
            "log1": sub / "train_node1.log",
            "meta": sub / "meta.json",
        }
    return {
        "dcgm0": sweep / f"dcgm_node0_ga{ga}.txt",
        "dcgm1": sweep / f"dcgm_node1_ga{ga}.txt",
        "log0": sweep / f"train_node0_ga{ga}.log",
        "log1": sweep / f"train_node1_ga{ga}.log",
        "meta": sweep / f"meta_ga{ga}.json",
    }


def _list_meta_files(sweep: Path) -> List[Path]:
    """flat 또는 per-ga subdir 두 layout 의 meta.json 리스트."""
    flat = sorted(sweep.glob("meta_ga*.json"), key=lambda p: int(p.stem.split("ga")[1]))
    if flat:
        return flat
    sub = sorted(sweep.glob("ga*/meta.json"), key=lambda p: int(p.parent.name[2:]))
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="/opt/dlami/nvme/ga_sweep")
    ap.add_argument("--awk", default="/home/ubuntu/nanotron/examples/heterogeneous/dcgm_text_to_jsonl.awk")
    ap.add_argument(
        "--figures-dir",
        default="/home/ubuntu/nanotron/examples/heterogeneous/figures/ga_sweep",
        help="그림 저장 root. 각 ga 는 ga<N>/ 서브디렉토리, summary 는 summary/ 에.",
    )
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    awk = Path(args.awk)
    fig_root = Path(args.figures_dir)
    fig_root.mkdir(parents=True, exist_ok=True)
    summary_dir = fig_root / "summary"
    summary_dir.mkdir(exist_ok=True)

    metas = _list_meta_files(sweep)
    if not metas:
        print(f"no meta_ga*.json (flat) or ga*/meta.json (subdir) in {sweep}", file=sys.stderr)
        sys.exit(1)

    stats = []
    for m in metas:
        meta = json.loads(m.read_text())
        ga = meta["ga"]
        paths = _resolve_raw_paths(sweep, ga)
        rows0 = dcgm_jsonl(paths["dcgm0"], awk)
        rows1 = dcgm_jsonl(paths["dcgm1"], awk)

        timings = extract_step_timings(paths["log0"])
        warmup = timings[0] if timings else 0
        steady = mean_safe(timings[1:]) if len(timings) > 1 else None

        stats.append({
            "ga": ga,
            "elapsed_sec": meta["elapsed_sec"],
            "n_steps_logged": len(timings),
            "warmup_sec": warmup,
            "steady_state_sec": steady,
            "step_times": [round(t, 3) for t in timings],
            "node0": summarize_node(rows0),
            "node1": summarize_node(rows1),
        })

        # 각 ga 마다 자기 폴더 안에 시계열 plot
        ga_fig_dir = fig_root / f"ga{ga}"
        ga_fig_dir.mkdir(exist_ok=True)
        plot_timeseries(ga, rows0, rows1, ga_fig_dir / "timeseries.png")
        print(f"saved {ga_fig_dir / 'timeseries.png'}")

    # ga 비교 summary plots + 통계 표 → summary/
    plot_summary(stats, summary_dir)
    print(f"saved summary plots in {summary_dir}")

    (summary_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    write_markdown(stats, summary_dir / "stats.md")
    print(f"saved {summary_dir / 'stats.json'} and {summary_dir / 'stats.md'}")


if __name__ == "__main__":
    main()
