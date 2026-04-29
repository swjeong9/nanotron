"""Cross-partition 비교 그래프.

``examples/heterogeneous/data/<cluster>/<model>/*/stats.json`` 을 모두 읽어
partition 별 throughput / MFU / power / memory 를 한 figure 에 비교.

OOM iteration 도 빈 자리 + "OOM" 라벨로 표시 (어떤 partition 이 fit 가능한지를
직접 보임).

Usage:
    uv run --no-project --with matplotlib python \\
        examples/heterogeneous/plot_compare.py \\
        --cluster l4__a10g_pp2 --model llama32_1b
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_NOTO)
    matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def load_partition_data(cluster: str, model: str, data_root: Path,
                        seq_filter: Optional[int] = None,
                        recompute_filter: Optional[bool] = None) -> List[Dict]:
    """Returns sorted list of {partition_str, partition_tuple, ...metrics}.

    seq_filter / recompute_filter: 일치 안 하면 skip. None 이면 전부.
    """
    base = data_root / cluster / model
    if not base.exists():
        return []
    rows = []
    for stat_path in base.glob("*/stats.json"):
        with open(stat_path) as f:
            s = json.load(f)
        meta = s.get("meta", {})
        ps = meta.get("pp_layer_partition_str") or ""
        if not ps:
            continue
        if seq_filter is not None and meta.get("seq_len") != seq_filter:
            continue
        if recompute_filter is not None and bool(meta.get("recompute_layer", False)) != recompute_filter:
            continue
        # ``11-5`` → (11, 5)
        partition_tuple = tuple(int(x) for x in ps.split("-"))
        bs = s.get("bar_summary") or {}
        mp = s.get("memory_peaks") or {}
        n0 = mp.get("node0_l4") if isinstance(mp.get("node0_l4"), dict) else {}
        n1 = mp.get("node1_a10g") if isinstance(mp.get("node1_a10g"), dict) else {}
        rows.append({
            "partition_str": ps,
            "partition_tuple": partition_tuple,
            "seq_len": meta.get("seq_len"),
            "recompute_layer": bool(meta.get("recompute_layer", False)),
            "oom": bool(meta.get("oom", False)) or s.get("failed", False),
            "completed_steps": int(meta.get("completed_steps", 0)),
            "throughput": bs.get("throughput_tokens_per_sec", 0),
            "step_sec": bs.get("steady_step_sec", 0),
            "mfu_cluster": bs.get("mfu_cluster_pct", 0),
            "mfu_l4": bs.get("mfu_l4_pct", 0),
            "mfu_a10g": bs.get("mfu_a10g_pct", 0),
            "power_l4": bs.get("avg_power_l4_w", 0),
            "power_a10g": bs.get("avg_power_a10g_w", 0),
            "mem_l4_reserved_MiB": (n0 or {}).get("max_reserved_MiB", 0) or 0,
            "mem_a10g_reserved_MiB": (n1 or {}).get("max_reserved_MiB", 0) or 0,
            "mem_l4_nvsmi_MiB": mp.get("nvidia_smi_max_MiB_node0", 0) or 0,
            "mem_a10g_nvsmi_MiB": mp.get("nvidia_smi_max_MiB_node1", 0) or 0,
        })
    # Sort by partition tuple (1-15 first, 15-1 last)
    rows.sort(key=lambda r: r["partition_tuple"])
    return rows


def annotate_oom(ax, x: int, y_max: float):
    """OOM partition 자리에 회색 박스 + 'OOM' 라벨."""
    ax.axvspan(x - 0.4, x + 0.4, alpha=0.1, color="red")
    ax.text(x, y_max * 0.5, "OOM", ha="center", va="center",
            color="red", fontsize=9, fontweight="bold", rotation=90)


def plot_compare(rows: List[Dict], cluster: str, model: str,
                 out_path: Path, subtitle: str = ""):
    if not rows:
        print("[plot_compare] no data")
        return

    n = len(rows)
    xs = np.arange(n)
    labels = [r["partition_str"] for r in rows]
    fit_mask = np.array([not r["oom"] and r["completed_steps"] >= 2 for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    title = f"Partition sweep: {cluster} / {model}"
    if subtitle:
        title += f" | {subtitle}"
    title += f" | each x-tick = pp_layer_partition (sum={sum(rows[0]['partition_tuple'])})"
    fig.suptitle(title, fontsize=11)

    # === (1) Throughput + step time ===
    ax = axes[0, 0]
    tputs = np.array([r["throughput"] if r["throughput"] else np.nan for r in rows])
    bar_colors = ["tab:gray" if fit_mask[i] else "lightgray" for i in range(n)]
    bars = ax.bar(xs, np.where(fit_mask, tputs, 0), color=bar_colors, width=0.7)
    y_max = float(np.nanmax(tputs)) if np.any(~np.isnan(tputs)) else 1
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
        else:
            ax.text(i, tputs[i], f"{int(tputs[i])}", ha="center", va="bottom", fontsize=7)
    # Highlight max
    if np.any(fit_mask):
        best_idx = int(np.nanargmax(np.where(fit_mask, tputs, np.nan)))
        bars[best_idx].set_color("tab:green")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("throughput [tokens/sec]")
    ax.set_title("Cluster throughput (best = green)")

    # === (2) MFU (grouped: cluster, L4, A10G) ===
    ax = axes[0, 1]
    width = 0.27
    mfu_c = np.array([r["mfu_cluster"] for r in rows])
    mfu_l = np.array([r["mfu_l4"] for r in rows])
    mfu_a = np.array([r["mfu_a10g"] for r in rows])
    ax.bar(xs - width, np.where(fit_mask, mfu_c, 0), width=width,
           color="tab:gray", label="cluster")
    ax.bar(xs, np.where(fit_mask, mfu_l, 0), width=width,
           color="tab:blue", label="L4 (stage 0)")
    ax.bar(xs + width, np.where(fit_mask, mfu_a, 0), width=width,
           color="tab:orange", label="A10G (stage 1)")
    y_max = max(50, float(np.nanmax(mfu_c)) * 1.15 if np.any(fit_mask) else 50)
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MFU [%]")
    ax.set_title("Model FLOPs Utilization")
    ax.legend(fontsize=8, loc="upper left")

    # === (3) Power (stacked: L4 + A10G + total marker) ===
    ax = axes[1, 0]
    p_l = np.array([r["power_l4"] for r in rows])
    p_a = np.array([r["power_a10g"] for r in rows])
    p_total = p_l + p_a
    # Stacked bar: L4 아래 + A10G 위
    ax.bar(xs, np.where(fit_mask, p_l, 0), width=0.7,
           color="tab:blue", label="L4 (TDP 72W)")
    ax.bar(xs, np.where(fit_mask, p_a, 0), bottom=np.where(fit_mask, p_l, 0),
           width=0.7, color="tab:orange", label="A10G (TDP 300W)")
    # Total 값 위에 라벨
    for i, r in enumerate(rows):
        if fit_mask[i]:
            ax.text(i, p_total[i], f"{int(p_total[i])}W",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    y_max = max(400, float(np.max(p_total[fit_mask])) * 1.15) if np.any(fit_mask) else 400
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("avg power [W]")
    ax.set_title("GPU power per partition (stacked = cluster total)")
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=8, loc="upper right")

    # === (4) Memory (stacked: L4 + A10G + cluster total label) ===
    ax = axes[1, 1]
    m_l = np.array([r["mem_l4_nvsmi_MiB"] for r in rows], dtype=float)
    m_a = np.array([r["mem_a10g_nvsmi_MiB"] for r in rows], dtype=float)
    m_total = m_l + m_a
    ax.bar(xs, np.where(fit_mask, m_l, 0), width=0.7,
           color="tab:blue", label="L4 (nvidia-smi max)")
    ax.bar(xs, np.where(fit_mask, m_a, 0), bottom=np.where(fit_mask, m_l, 0),
           width=0.7, color="tab:orange", label="A10G (nvidia-smi max)")
    for i, r in enumerate(rows):
        if fit_mask[i]:
            ax.text(i, m_total[i], f"{int(m_total[i] / 1024)} GiB",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")
    y_max = max(48 * 1024, float(np.max(m_total[fit_mask])) * 1.10) if np.any(fit_mask) else 48 * 1024
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("memory used [MiB]")
    ax.set_title("Peak GPU memory per partition (stacked = cluster total)")
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="l4__a10g_pp2")
    ap.add_argument("--model", default="llama32_1b")
    ap.add_argument("--seq", type=int, default=None,
                    help="filter by sequence_length (e.g. 1024). None = all configs.")
    ap.add_argument("--recompute", choices=["true", "false", "any"], default="any")
    ap.add_argument("--data-root",
                    default="/home/ubuntu/nanotron/examples/heterogeneous/data")
    ap.add_argument("--out",
                    default="/home/ubuntu/nanotron/examples/heterogeneous/figures")
    args = ap.parse_args()

    recompute_filter = None if args.recompute == "any" else (args.recompute == "true")
    rows = load_partition_data(args.cluster, args.model, Path(args.data_root),
                               seq_filter=args.seq, recompute_filter=recompute_filter)
    if not rows:
        raise SystemExit(f"No data after filter (seq={args.seq}, recompute={args.recompute})")
    print(f"loaded {len(rows)} partitions: " + ", ".join(r["partition_str"] for r in rows))

    # 출력 dir 에 filter 정보 포함
    sub_parts = []
    if args.seq is not None:
        sub_parts.append(f"seq{args.seq}")
    if args.recompute != "any":
        sub_parts.append(f"recomp_{args.recompute}")
    sub = "_".join(sub_parts) if sub_parts else "all"
    out_dir = Path(args.out) / args.cluster / args.model / f"comparison_{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    subtitle_parts = []
    if args.seq is not None:
        subtitle_parts.append(f"seq={args.seq}")
    if args.recompute != "any":
        subtitle_parts.append(f"recompute={args.recompute}")
    subtitle = ", ".join(subtitle_parts)

    plot_compare(rows, args.cluster, args.model,
                 out_dir / "partition_compare.png", subtitle=subtitle)


if __name__ == "__main__":
    main()
