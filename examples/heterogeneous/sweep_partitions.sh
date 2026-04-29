#!/usr/bin/env bash
# 모든 가능한 PP=2 layer partition 을 [1, 15] 부터 [15, 1] 까지 (16 layers
# 총합) 자동 sweep. 각 iteration:
#   1. ``benchmark_single.sh`` 호출 (partition override + raw 데이터 기록)
#   2. ``plot_single.py`` 호출 (figures + data 저장)
#   3. OOM / 학습 실패면 plot 은 skip 되고 data dir 에 OOM 표기만
#
# Total estimate: 정상 step 22s × 10 step + warmup ~30s + cleanup ~10s ≈ 5분 / iter
# × 15 partitions ≈ 75 분. OOM iteration 은 더 빨리 끝남 (≤ 1분).
#
# Usage:
#   bash sweep_partitions.sh
#   bash sweep_partitions.sh configs/llama32_1b/alpaca_pp2_split8-8.yaml

set -uo pipefail   # ``-e`` 일부러 빼서 한 partition OOM 이라도 이어서 돌게.

cd "$(dirname "$0")/../.."

CONFIG="${1:-examples/heterogeneous/configs/llama32_1b/alpaca_pp2.yaml}"
START_STAGE0="${START_STAGE0:-1}"   # env var, default 1 (전체 sweep)
END_STAGE0="${END_STAGE0:-}"         # env var, default = NUM_LAYERS - 1

# 강제 sync — sweep 시작 전 한 번. (각 iter 의 benchmark_single 도 자체 sync 하지만
# 첫 iter 시작 직후 mismatch 가 발견되면 sweep 통째로 무의미해지니 미리 검증)
bash "$(dirname "$0")/sync_to_node1.sh" || exit 2
NUM_LAYERS=${NUM_LAYERS:-16}

echo "=== Sweep partition: 1..$((NUM_LAYERS - 1)) for $CONFIG ==="

END_STAGE0=${END_STAGE0:-$((NUM_LAYERS - 1))}
START_TS=$(date +%s)
echo "=== Sweep range: stage0 = $START_STAGE0 .. $END_STAGE0 ==="
for STAGE0 in $(seq "$START_STAGE0" "$END_STAGE0"); do
    STAGE1=$((NUM_LAYERS - STAGE0))
    PARTITION="${STAGE0}-${STAGE1}"
    echo
    echo "==============================================="
    echo "  partition $PARTITION  ($(date '+%H:%M:%S'))"
    echo "==============================================="
    bash examples/heterogeneous/benchmark_single.sh "$CONFIG" "$PARTITION" \
        2>&1 | tail -30
    uv run --no-project --with matplotlib python \
        examples/heterogeneous/plot_single.py 2>&1 | tail -10
done
END_TS=$(date +%s)
echo
echo "=== Sweep complete in $((END_TS - START_TS)) sec ==="
echo "data:    examples/heterogeneous/data/"
echo "figures: examples/heterogeneous/figures/"
