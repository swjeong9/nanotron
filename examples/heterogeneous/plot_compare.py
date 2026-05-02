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

# Element 별 font size — readability 우선. 추후 figure 크기 따라 일괄 scale 하고 싶으면
# 한 곳에서 조절. xtick / title / subtitle / legend 추가 확대; ytick / bar value 는 유지
# (해당 위치들은 이미 충분).
FS_TICK_X = 20     # x-tick label (partition string)
FS_TICK_Y = 13     # y-tick label (그대로)
FS_AXIS = 17       # axis label, panel title
FS_BAR_VALUE = 10  # bar 위 숫자 라벨 (그대로)
FS_LEGEND = 15     # legend
FS_OOM = 12        # OOM 주석
FS_SUPTITLE = 26   # overall figure title (suptitle = main + sub line)

matplotlib.rcParams.update({
    "font.size": FS_TICK_Y,
    "axes.labelsize": FS_AXIS,
    "axes.titlesize": FS_AXIS,
    "xtick.labelsize": FS_TICK_X,
    "ytick.labelsize": FS_TICK_Y,
})


def load_partition_data(cluster: str, model: str, data_root: Path,
                        seq_filter: Optional[int] = None,
                        recompute_filter: Optional[bool] = None,
                        pp_filter: Optional[int] = None,
                        tp_filter: Optional[int] = None) -> List[Dict]:
    """Returns sorted list of {partition_str, partition_tuple, ...metrics}.

    Filters: seq / recompute / pp / tp — 일치 안 하면 skip. None 이면 전부.

    Field 이름은 backwards-compat — node0/node1 (신) 우선, 없으면 l4/a10g (구) fallback.
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
        if pp_filter is not None and int(meta.get("pp", 0)) != pp_filter:
            continue
        if tp_filter is not None and int(meta.get("tp", 0)) != tp_filter:
            continue
        # ``11-5`` → (11, 5)
        partition_tuple = tuple(int(x) for x in ps.split("-"))
        bs = s.get("bar_summary") or {}
        mp = s.get("memory_peaks") or {}
        # node0/node1 (신) 우선 — 없으면 node0_l4/node1_a10g (구) fallback
        n0 = mp.get("node0") or mp.get("node0_l4") or {}
        n1 = mp.get("node1") or mp.get("node1_a10g") or {}
        if not isinstance(n0, dict): n0 = {}
        if not isinstance(n1, dict): n1 = {}
        tp = int(meta.get("tp", 1)) or 1
        # Stage 당 power (per-GPU avg × num_gpus_in_stage = tp)
        avg_p_n0 = bs.get("avg_power_node0_w", bs.get("avg_power_l4_w", 0)) or 0
        avg_p_n1 = bs.get("avg_power_node1_w", bs.get("avg_power_a10g_w", 0)) or 0
        rows.append({
            "partition_str": ps,
            "partition_tuple": partition_tuple,
            "pp": int(meta.get("pp", 0)),
            "tp": tp,
            "seq_len": meta.get("seq_len"),
            "recompute_layer": bool(meta.get("recompute_layer", False)),
            "oom": bool(meta.get("oom", False)) or s.get("failed", False),
            "completed_steps": int(meta.get("completed_steps", 0)),
            "throughput": bs.get("throughput_tokens_per_sec", 0),
            "step_sec": bs.get("steady_step_sec", 0),
            "avg_latency_per_step_sec": bs.get("avg_latency_per_step_sec",
                                               bs.get("steady_step_sec", 0)),
            "mfu_cluster": bs.get("mfu_cluster_pct", 0),
            "mfu_node0": bs.get("mfu_node0_pct", bs.get("mfu_l4_pct", 0)),
            "mfu_node1": bs.get("mfu_node1_pct", bs.get("mfu_a10g_pct", 0)),
            # Per-GPU avg + stage total
            "power_node0_avg_per_gpu": avg_p_n0,
            "power_node1_avg_per_gpu": avg_p_n1,
            "power_node0_sum": avg_p_n0 * tp,
            "power_node1_sum": avg_p_n1 * tp,
            "mem_node0_reserved_MiB": n0.get("max_reserved_MiB", 0) or 0,
            "mem_node1_reserved_MiB": n1.get("max_reserved_MiB", 0) or 0,
            "mem_node0_nvsmi_MiB": mp.get("nvidia_smi_max_MiB_node0", 0) or 0,
            "mem_node1_nvsmi_MiB": mp.get("nvidia_smi_max_MiB_node1", 0) or 0,
        })
    rows.sort(key=lambda r: r["partition_tuple"])
    return rows


_GPU_TDP = {"L4": 72, "L40S": 350, "A10G": 150, "A100": 400, "H100": 700, "V100": 300}


def _infer_gpu_types(rows: List[Dict], cluster: str):
    """Returns (gpu_type_n0, gpu_type_n1, tdp_n0, tdp_n1)."""
    import re
    # cluster 이름에서 추출 시도 (e.g. g6e_48xl__p4d_24xl → L40S, A100; p3dn_24xl__p4dn → V100, A100)
    name = cluster.lower()
    if "p3dn" in name or "p3_" in name:
        n0 = "V100"
    elif "g6e" in name:
        n0 = "L40S"
    elif "g6_" in name or "g6.12xl" in name:
        n0 = "L4"
    elif "l4" in name:
        n0 = "L4"
    else:
        n0 = "GPU"
    if "p4d" in name or "a100" in name:
        n1 = "A100"
    elif "g5" in name or "a10g" in name:
        n1 = "A10G"
    elif "h100" in name:
        n1 = "H100"
    else:
        n1 = "GPU"
    return n0, n1, _GPU_TDP.get(n0, 0), _GPU_TDP.get(n1, 0)


def annotate_oom(ax, x: int, y_max: float):
    """OOM partition: x-axis 바로 위에 'OOM' 가로 라벨만. 박스/배경 없음."""
    trans = ax.get_xaxis_transform()
    ax.text(x, 0.02, "OOM", transform=trans, ha="center", va="bottom",
            color="red", fontsize=FS_OOM, fontweight="bold")


def plot_compare(rows: List[Dict], cluster: str, model: str,
                 out_path: Path, subtitle: str = ""):
    if not rows:
        print("[plot_compare] no data")
        return

    # GPU type / TDP — meta 의 gpu_type_node{0,1} (있으면) 사용, 없으면 cluster 이름에서 휴리스틱.
    gpu_type_n0, gpu_type_n1, tdp_n0, tdp_n1 = _infer_gpu_types(rows, cluster)

    n = len(rows)
    xs = np.arange(n)
    labels = [r["partition_str"] for r in rows]
    fit_mask = np.array([not r["oom"] and r["completed_steps"] >= 2 for r in rows])

    # 5 panels in 1 column — full width per panel for readable x-tick labels with 27 partitions.
    fig, axes = plt.subplots(5, 1, figsize=(16, 22), sharex=True)
    main_title = f"Partition sweep — {cluster} / {model}"
    sub_parts = []
    if subtitle:
        sub_parts.append(subtitle)
    sub_parts.append(f"x = pp_layer_partition (sum={sum(rows[0]['partition_tuple'])})")
    sub_line = " | ".join(sub_parts)
    fig.suptitle(f"{main_title}\n{sub_line}", fontsize=FS_SUPTITLE, y=0.995)

    # === (1) Throughput + step time ===
    ax = axes[0]
    tputs = np.array([r["throughput"] if r["throughput"] else np.nan for r in rows])
    bar_colors = ["tab:gray" if fit_mask[i] else "lightgray" for i in range(n)]
    bars = ax.bar(xs, np.where(fit_mask, tputs, 0), color=bar_colors, width=0.7)
    y_max = float(np.nanmax(tputs)) if np.any(~np.isnan(tputs)) else 1
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
        else:
            ax.text(i, tputs[i], f"{int(tputs[i])}", ha="center", va="bottom", fontsize=FS_BAR_VALUE)
    # Highlight max
    if np.any(fit_mask):
        best_idx = int(np.nanargmax(np.where(fit_mask, tputs, np.nan)))
        bars[best_idx].set_color("tab:green")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=FS_TICK_X)
    ax.set_ylabel("throughput [tokens/sec]")
    ax.set_title("Cluster throughput (best = green)")

    # === (2) Avg latency per step (cost-relevant) ===
    ax = axes[1]
    lats = np.array([r["avg_latency_per_step_sec"] if r["avg_latency_per_step_sec"] else np.nan
                     for r in rows])
    bar_colors = ["tab:gray" if fit_mask[i] else "lightgray" for i in range(n)]
    bars = ax.bar(xs, np.where(fit_mask, lats, 0), color=bar_colors, width=0.7)
    y_max = float(np.nanmax(lats)) if np.any(~np.isnan(lats)) else 1
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
        else:
            ax.text(i, lats[i], f"{lats[i]:.2f}s", ha="center", va="bottom", fontsize=FS_BAR_VALUE)
    # Lowest latency = best (green)
    if np.any(fit_mask):
        best_idx = int(np.nanargmin(np.where(fit_mask, lats, np.nan)))
        bars[best_idx].set_color("tab:green")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=FS_TICK_X)
    ax.set_ylabel("avg latency per step [s]")
    ax.set_title("Avg latency per step (lowest = green) — for cost calc")

    # === (3) MFU (grouped: cluster, NODE 0, NODE 1) ===
    ax = axes[2]
    width = 0.27
    mfu_c = np.array([r["mfu_cluster"] for r in rows])
    mfu_l = np.array([r["mfu_node0"] for r in rows])
    mfu_a = np.array([r["mfu_node1"] for r in rows])
    ax.bar(xs - width, np.where(fit_mask, mfu_c, 0), width=width,
           color="tab:gray", label="cluster")
    ax.bar(xs, np.where(fit_mask, mfu_l, 0), width=width,
           color="tab:blue", label=f"{gpu_type_n0} (NODE 0)")
    ax.bar(xs + width, np.where(fit_mask, mfu_a, 0), width=width,
           color="tab:orange", label=f"{gpu_type_n1} (NODE 1)")
    y_max = max(50, float(np.nanmax(mfu_c)) * 1.15 if np.any(fit_mask) else 50)
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=FS_TICK_X)
    ax.set_ylabel("MFU [%]")
    ax.set_title("Model FLOPs Utilization")
    ax.legend(fontsize=FS_LEGEND, loc="upper left", ncol=3)

    # === (4) Power (stacked node sum: NODE 0 + NODE 1 = cluster total) ===
    # 각 node 의 8 GPU 모두 power × 8 (cluster 의 stage 가 N=PP/2 개 노드 별로 나뉘는데,
    # 현재 cluster 셋업에선 한 노드 = 8 GPU 고정 — power_node*_sum 자체가 stage 합산이라
    # 노드별 GPU 개수 (8) 만큼 곱해야 cluster total).
    ax = axes[3]
    # power_node*_sum = avg per-GPU × tp (stage 1 개의 GPU 개수). 한 노드의 stage 수 =
    # 8 / tp 라서 노드 총 power = power_node*_sum × (8 / tp). 등가로 avg × 8.
    n_gpus_per_node = 8
    p_l = np.array([r["power_node0_avg_per_gpu"] * n_gpus_per_node for r in rows])
    p_a = np.array([r["power_node1_avg_per_gpu"] * n_gpus_per_node for r in rows])
    p_total = p_l + p_a
    label_l = f"{n_gpus_per_node}×{gpu_type_n0} (NODE 0, TDP {tdp_n0}W/GPU)"
    label_a = f"{n_gpus_per_node}×{gpu_type_n1} (NODE 1, TDP {tdp_n1}W/GPU)"
    ax.bar(xs, np.where(fit_mask, p_l, 0), width=0.7,
           color="tab:blue", label=label_l)
    ax.bar(xs, np.where(fit_mask, p_a, 0), bottom=np.where(fit_mask, p_l, 0),
           width=0.7, color="tab:orange", label=label_a)
    # Total 값 위에 라벨
    for i, r in enumerate(rows):
        if fit_mask[i]:
            ax.text(i, p_total[i], f"{int(p_total[i])}W",
                    ha="center", va="bottom", fontsize=FS_BAR_VALUE, fontweight="bold")
    # y range = max × 1.3 (legend 가 bar 가리지 않도록 헤드룸 확보)
    y_max = float(np.max(p_total[fit_mask])) * 1.3 if np.any(fit_mask) else 800
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=FS_TICK_X)
    ax.set_ylabel("stage total power [W]")
    ax.set_title("GPU power per partition (stage sum, stacked = cluster total)")
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=FS_LEGEND, loc="upper right", ncol=2)

    # === (5) Memory (stacked: NODE 0 + NODE 1 + cluster total label) ===
    ax = axes[4]
    m_l = np.array([r["mem_node0_nvsmi_MiB"] for r in rows], dtype=float)
    m_a = np.array([r["mem_node1_nvsmi_MiB"] for r in rows], dtype=float)
    m_total = m_l + m_a
    ax.bar(xs, np.where(fit_mask, m_l, 0), width=0.7,
           color="tab:blue", label=f"{gpu_type_n0} (nvidia-smi max)")
    ax.bar(xs, np.where(fit_mask, m_a, 0), bottom=np.where(fit_mask, m_l, 0),
           width=0.7, color="tab:orange", label=f"{gpu_type_n1} (nvidia-smi max)")
    for i, r in enumerate(rows):
        if fit_mask[i]:
            ax.text(i, m_total[i], f"{int(m_total[i] / 1024)} GiB",
                    ha="center", va="bottom", fontsize=FS_BAR_VALUE, fontweight="bold")
    # y range = max × 1.3 (legend 가 bar 가리지 않도록 헤드룸 확보)
    y_max = float(np.max(m_total[fit_mask])) * 1.3 if np.any(fit_mask) else 48 * 1024
    for i, r in enumerate(rows):
        if r["oom"] or r["completed_steps"] < 2:
            annotate_oom(ax, i, y_max)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="center", fontsize=FS_TICK_X)
    ax.set_ylabel("memory used [MiB]")
    # nvidia_smi_max_MiB_node{0,1} = max sample over all GPUs in node × time.
    # In multi-GPU stage (e.g. TP=4) this is "per-GPU peak in stage", not stage sum.
    ax.set_title("Peak GPU memory (per-GPU max in stage; OOM threshold)")
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=FS_LEGEND, loc="upper left", ncol=2)


    # Suptitle 과 axes 사이 간격 축소: rect[3] 0.97 → 0.985 (axes 가 위로 더 차지),
    # suptitle y=0.995 그대로 → 두 사이 gap 이 figure 의 1% 로 축소.
    fig.tight_layout(rect=[0, 0, 1, 0.985])
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
    ap.add_argument("--pp", type=int, default=None, help="filter by pp (e.g. 4)")
    ap.add_argument("--tp", type=int, default=None, help="filter by tp (e.g. 4)")
    ap.add_argument("--trim-oom", action="store_true",
                    help="앞/뒤 contiguous OOM 영역 중 fit 영역 직전/직후 1점만 남기고 나머지 OOM 점 숨김.")
    ap.add_argument("--data-root",
                    default="/home/ubuntu/nanotron/examples/heterogeneous/data")
    ap.add_argument("--out",
                    default="/home/ubuntu/nanotron/examples/heterogeneous/figures")
    args = ap.parse_args()

    recompute_filter = None if args.recompute == "any" else (args.recompute == "true")
    rows = load_partition_data(args.cluster, args.model, Path(args.data_root),
                               seq_filter=args.seq, recompute_filter=recompute_filter,
                               pp_filter=args.pp, tp_filter=args.tp)
    if not rows:
        raise SystemExit(f"No data after filter (seq={args.seq}, recompute={args.recompute}, pp={args.pp}, tp={args.tp})")

    if args.trim_oom and rows:
        # Front: contiguous OOM 의 마지막 1점만 유지
        front_keep = 0
        while front_keep < len(rows) and rows[front_keep]["oom"]:
            front_keep += 1
        # rows[0..front_keep-1] 가 OOM. rows[front_keep-1] 만 유지 (front_keep > 0 이면).
        # rows[front_keep..] 가 fit 시작.
        if front_keep > 0:
            rows = rows[front_keep - 1:]
        # Back: contiguous OOM 의 첫 1점만 유지
        n = len(rows)
        back_start = n
        while back_start > 0 and rows[back_start - 1]["oom"]:
            back_start -= 1
        # rows[back_start..n-1] 가 OOM. rows[back_start] 만 유지.
        if back_start < n:
            rows = rows[:back_start + 1]
    print(f"loaded {len(rows)} partitions: " + ", ".join(r["partition_str"] for r in rows))

    # 출력 dir 에 filter 정보 포함
    sub_parts = []
    if args.pp is not None:
        sub_parts.append(f"pp{args.pp}")
    if args.tp is not None:
        sub_parts.append(f"tp{args.tp}")
    if args.seq is not None:
        sub_parts.append(f"seq{args.seq}")
    if args.recompute != "any":
        sub_parts.append(f"recomp_{args.recompute}")
    sub = "_".join(sub_parts) if sub_parts else "all"
    out_dir = Path(args.out) / args.cluster / args.model / f"comparison_{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)

    subtitle_parts = []
    if args.pp is not None:
        subtitle_parts.append(f"PP={args.pp}")
    if args.tp is not None:
        subtitle_parts.append(f"TP={args.tp}")
    if args.seq is not None:
        subtitle_parts.append(f"seq={args.seq}")
    if args.recompute != "any":
        subtitle_parts.append(f"recompute={args.recompute}")
    subtitle = ", ".join(subtitle_parts)

    plot_compare(rows, args.cluster, args.model,
                 out_dir / "partition_compare.png", subtitle=subtitle)


if __name__ == "__main__":
    main()
