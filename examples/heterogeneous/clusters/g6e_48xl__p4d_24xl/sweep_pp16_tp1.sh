#!/usr/bin/env bash
# Qwen 3 14B (40 layers) 의 PP=16 TP=1 partition sweep.
#
# 16 stage = 8 stage / node × 2 nodes. TP=1 → cluster GPU = pp = 16.
# Stage 당 layer 수 N (g6e 측) ∈ {1, 2, 3, 4}, p4d 측 = (40 - 8N) / 8 = 5 - N.
# 균등 (g6e=p4d=20) 은 8N=20 안 떨어져 별도: balanced [3×4 + 2×4 / 3×4 + 2×4] — verify 에서 이미 측정.
#
# Sweep partition pattern (각 노드 안 stage 들 동일 layer):
#   N=1: g6e [1,1,1,1,1,1,1,1] / p4d [4,4,4,4,4,4,4,4]   sum=8+32=40
#   N=2: g6e [2,2,2,2,2,2,2,2] / p4d [3,3,3,3,3,3,3,3]   sum=16+24=40
#   N=3: g6e [3,3,3,3,3,3,3,3] / p4d [2,2,2,2,2,2,2,2]   sum=24+16=40
#   N=4: g6e [4,4,4,4,4,4,4,4] / p4d [1,1,1,1,1,1,1,1]   sum=32+8=40
#
# Usage:
#   bash sweep_pp16_tp1.sh                  # default: N = 1..4
#   START_N=2 END_N=3 bash sweep_pp16_tp1.sh

set -uo pipefail
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NANOTRON=/home/ubuntu/nanotron
cd "$NANOTRON"

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp16_tp1.yaml}"
START_N="${START_N:-1}"
END_N="${END_N:-4}"

echo "=== PP=16 TP=1 sweep — cluster=g6e_48xl__p4d_24xl model=qwen3_14b ==="
echo "=== rule: g6e 8 stage 모두 N layer, p4d 8 stage 모두 (5-N) layer (sum=40) ==="
echo "=== range: N = $START_N .. $END_N (총 $((END_N - START_N + 1)) 점) ==="

bash "$CLUSTER_DIR/sync.sh" || exit 2

START_TS=$(date +%s)
for N in $(seq "$START_N" "$END_N"); do
    OTHER=$((5 - N))
    # 16 entries: 8x N (g6e) + 8x OTHER (p4d)
    G6E_PART=$(yes "$N" | head -8 | tr '\n' '-' | sed 's/-$//')
    P4D_PART=$(yes "$OTHER" | head -8 | tr '\n' '-' | sed 's/-$//')
    PARTITION="${G6E_PART}-${P4D_PART}"
    echo
    echo "==============================================="
    echo "  partition $PARTITION  (g6e 에 $((N*8)) layer, p4d 에 $((OTHER*8)) layer)"
    echo "  $(date '+%H:%M:%S')"
    echo "==============================================="
    bash "$CLUSTER_DIR/benchmark_single.sh" "$CONFIG" "$PARTITION" 2>&1 | tail -30
    uv run --no-project --with matplotlib python \
        "$NANOTRON/examples/heterogeneous/plot_single.py" 2>&1 | tail -10
done
END_TS=$(date +%s)
echo
echo "=== Sweep complete in $((END_TS - START_TS)) sec ==="
