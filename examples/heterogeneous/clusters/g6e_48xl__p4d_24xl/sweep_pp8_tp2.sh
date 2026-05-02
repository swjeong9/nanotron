#!/usr/bin/env bash
# Qwen 3 14B (40 layers) 의 PP=8 TP=2 partition sweep.
#
# PP=8 = 8 stage. 양 노드 균등 (8 GPU 씩) 이라 각 노드에 4 stage 씩 (TP=2 라 stage 당 2 GPU):
#   stage 0,1,2,3 = N         (g6e.48xl 의 4 stage, L40S 각각)
#   stage 4,5,6,7 = 10 - N    (p4d 의 4 stage, A100 각각)
#   sum = 4N + 4(10-N) = 40 ✓
#   N ∈ [1, 9]  →  9 점
#
# Usage:
#   bash sweep_pp8_tp2.sh                  # default: N = 1..9
#   START_N=2 END_N=8 bash sweep_pp8_tp2.sh

set -uo pipefail
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOTRON=/home/ubuntu/nanotron
cd "$NANOTRON"

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp8_tp2.yaml}"
NUM_LAYERS=${NUM_LAYERS:-40}
START_N="${START_N:-1}"
END_N="${END_N:-9}"

echo "=== PP=8 TP=2 sweep — cluster=g6e_48xl__p4d_24xl model=qwen3_14b ==="
echo "=== rule: stage 0..3 = N, stage 4..7 = $((NUM_LAYERS / 8))-N (g6e:p4d 분배) ==="
echo "=== range: N = $START_N .. $END_N (총 $((END_N - START_N + 1)) 점) ==="

bash "$CLUSTER_DIR/sync.sh" || exit 2

START_TS=$(date +%s)
QUARTER=$((NUM_LAYERS / 8))   # 5 layer per stage at balanced
for N in $(seq "$START_N" "$END_N"); do
    OTHER=$((QUARTER * 2 - N))   # NUM_LAYERS / 4 - N = 10 - N
    PARTITION="${N}-${N}-${N}-${N}-${OTHER}-${OTHER}-${OTHER}-${OTHER}"
    echo
    echo "==============================================="
    echo "  partition $PARTITION  (g6e 에 $((N*4)) layer, p4d 에 $((OTHER*4)) layer)"
    echo "  $(date '+%H:%M:%S')"
    echo "==============================================="
    bash "$CLUSTER_DIR/benchmark_single.sh" "$CONFIG" "$PARTITION" 2>&1 | tail -30
    uv run --no-project --with matplotlib python \
        "$NANOTRON/examples/heterogeneous/plot_single.py" 2>&1 | tail -10
done
END_TS=$(date +%s)
echo
echo "=== Sweep complete in $((END_TS - START_TS)) sec ==="
