#!/usr/bin/env bash
# Qwen 3 14B (40 layers) 의 PP=4 TP=4 partition sweep.
#
# 자유도 = "g6e 에 몇 layer / p4d 에 몇 layer" (intra-node 는 같은 하드웨어라 균등).
#
#   stage 0 = stage 1 = N         (g6e.48xl 의 두 stage, L40S)
#   stage 2 = stage 3 = 20 - N    (p4d 의 두 stage, A100)
#   N ∈ [1, 19]  →  19 점
#
# Usage:
#   bash sweep_pp4_tp4.sh                  # default: N = 1..19
#   START_N=4 END_N=16 bash sweep_pp4_tp4.sh

set -uo pipefail
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOTRON=/home/ubuntu/nanotron
cd "$NANOTRON"

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp4_tp4.yaml}"
NUM_LAYERS=${NUM_LAYERS:-40}
START_N="${START_N:-1}"
END_N="${END_N:-19}"

echo "=== PP=4 TP=4 sweep — cluster=g6e_48xl__p4d_24xl model=qwen3_14b ==="
echo "=== rule: stage0=stage1=N, stage2=stage3=$((NUM_LAYERS / 2))-N (g6e:p4d 분배) ==="
echo "=== range: N = $START_N .. $END_N (총 $((END_N - START_N + 1)) 점) ==="

bash "$CLUSTER_DIR/sync.sh" || exit 2

START_TS=$(date +%s)
HALF=$((NUM_LAYERS / 2))   # 20 layer 가 한 노드에 분배되는 baseline
for N in $(seq "$START_N" "$END_N"); do
    OTHER=$((HALF - N))
    PARTITION="${N}-${N}-${OTHER}-${OTHER}"
    echo
    echo "==============================================="
    echo "  partition $PARTITION  (g6e 에 $((N*2)) layer, p4d 에 $((OTHER*2)) layer)"
    echo "  $(date '+%H:%M:%S')"
    echo "==============================================="
    bash "$CLUSTER_DIR/benchmark_single.sh" "$CONFIG" "$PARTITION" 2>&1 | tail -30
    uv run --no-project --with matplotlib python \
        "$NANOTRON/examples/heterogeneous/plot_single.py" 2>&1 | tail -10
done
END_TS=$(date +%s)
echo
echo "=== Sweep complete in $((END_TS - START_TS)) sec ==="
