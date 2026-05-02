#!/usr/bin/env bash
# Qwen 3 14B (40 layers) 의 PP=2 TP=8 partition sweep.
#
# 2-way partition. 자유도 1 차원: stage 0 layer 수 N → stage 1 = (40 - N).
# Default range: stage0 ∈ {2, 4, ..., 38} stride 2 (19 점).
#
# 본 셋업의 stage 위치:
#   - stage 0: NODE 0 = g6e.48xlarge (8× L40S 48GB) — TP=8 intra-node
#   - stage 1: NODE 1 = p4d.24xlarge (8× A100 40GB) — TP=8 intra-node + lm_head
#
# Usage:
#   bash sweep_pp2_tp8.sh                                       # default: stage0 2,4,...,38 (19 점)
#   START_STAGE0=12 END_STAGE0=28 STEP=2 bash sweep_pp2_tp8.sh
#   START_STAGE0=1 END_STAGE0=39 STEP=1 bash sweep_pp2_tp8.sh   # full sweep (39 점) — 시간 김

set -uo pipefail
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOTRON=/home/ubuntu/nanotron
cd "$NANOTRON"

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp2_tp8.yaml}"
NUM_LAYERS=${NUM_LAYERS:-40}
START_STAGE0="${START_STAGE0:-2}"
END_STAGE0="${END_STAGE0:-38}"
STEP="${STEP:-2}"

NUM_POINTS=$(( (END_STAGE0 - START_STAGE0) / STEP + 1 ))
echo "=== PP=2 TP=8 sweep — cluster=g6e_48xl__p4d_24xl model=qwen3_14b ==="
echo "=== range: stage0 = $START_STAGE0 .. $END_STAGE0 step $STEP (sum = $NUM_LAYERS) ==="
echo "=== rule: stage 1 = (sum - stage0); 총 $NUM_POINTS 점 ==="

bash "$CLUSTER_DIR/sync.sh" || exit 2

START_TS=$(date +%s)
for STAGE0 in $(seq "$START_STAGE0" "$STEP" "$END_STAGE0"); do
    STAGE1=$((NUM_LAYERS - STAGE0))
    PARTITION="${STAGE0}-${STAGE1}"
    echo
    echo "==============================================="
    echo "  partition $PARTITION  ($(date '+%H:%M:%S'))"
    echo "==============================================="
    bash "$CLUSTER_DIR/benchmark_single.sh" "$CONFIG" "$PARTITION" 2>&1 | tail -30
    uv run --no-project --with matplotlib python \
        "$NANOTRON/examples/heterogeneous/plot_single.py" 2>&1 | tail -10
done
END_TS=$(date +%s)
echo
echo "=== Sweep complete in $((END_TS - START_TS)) sec ==="
