#!/usr/bin/env bash
# Qwen 3 14B (40 layers) 의 PP=4 partition sweep — TP=4 + PP=4.
#
# 4-way partition 은 자유도 3 차원이라 full grid 가 매우 큼. 본 sweep 은
# **stage 0 (slow L40S) 의 layer 수만 변화** 시키고 stage 1/2/3 은 균등 분배.
# stage 0 가 cluster throughput bottleneck 이므로 가장 큰 변동 자유도.
#
# 더 정교한 sweep (stage 0 + stage 1 동시 변화) 이 필요하면 별도 script.
#
# Usage:
#   bash sweep_partitions.sh                            # default: stage0 4..16
#   START_STAGE0=6 END_STAGE0=14 bash sweep_partitions.sh

set -uo pipefail
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOTRON=/home/ubuntu/nanotron
cd "$NANOTRON"

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp4_tp4.yaml}"
NUM_LAYERS=${NUM_LAYERS:-40}
START_STAGE0="${START_STAGE0:-4}"
END_STAGE0="${END_STAGE0:-16}"

echo "=== Sweep cluster=g6e_12xl_x2__p4d_24xl_pp4_tp4 model=qwen3_14b ==="
echo "=== range: stage0 = $START_STAGE0 .. $END_STAGE0 (sum = $NUM_LAYERS, 4-way split) ==="
echo "=== rule: stage 1/2/3 = (sum - stage0) 을 3 분할 (균등, 마지막에 잔여 떨굼) ==="

# 한 번 sync 미리
bash "$CLUSTER_DIR/sync.sh" || exit 2

START_TS=$(date +%s)
for STAGE0 in $(seq "$START_STAGE0" "$END_STAGE0"); do
    REMAINING=$((NUM_LAYERS - STAGE0))
    BASE=$((REMAINING / 3))
    REMAINDER=$((REMAINING - BASE * 3))
    STAGE1=$BASE
    STAGE2=$BASE
    # 잔여 layer 는 stage 3 (lm_head 와 같은 노드, A100 NVLink, 빠름) 에 떨굼.
    STAGE3=$((BASE + REMAINDER))
    PARTITION="${STAGE0}-${STAGE1}-${STAGE2}-${STAGE3}"
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
