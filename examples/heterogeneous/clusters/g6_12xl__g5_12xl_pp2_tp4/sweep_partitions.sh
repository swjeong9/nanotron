#!/usr/bin/env bash
# Llama 3.2 3B (28 layers) 의 PP=2 partition sweep — TP=4 + PP=2.
# 28 layer 의 균형 = [14, 14]. 균형 ± N 으로 sweep.
#
# Usage:
#   bash sweep_partitions.sh                            # default: stage0 1..27
#   START_STAGE0=10 END_STAGE0=18 bash sweep_partitions.sh   # 균형 ± 4 만

set -uo pipefail
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOTRON=/home/ubuntu/nanotron
cd "$NANOTRON"

CONFIG="${1:-examples/heterogeneous/configs/llama32_3b/alpaca_pp2_tp4.yaml}"
NUM_LAYERS=${NUM_LAYERS:-28}
START_STAGE0="${START_STAGE0:-1}"
END_STAGE0="${END_STAGE0:-$((NUM_LAYERS - 1))}"

echo "=== Sweep cluster=g6_12xl__g5_12xl_pp2_tp4 model=llama32_3b ==="
echo "=== range: stage0 = $START_STAGE0 .. $END_STAGE0 (sum = $NUM_LAYERS) ==="

# 한 번 sync 미리
bash "$CLUSTER_DIR/sync.sh" || exit 2

START_TS=$(date +%s)
for STAGE0 in $(seq "$START_STAGE0" "$END_STAGE0"); do
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
