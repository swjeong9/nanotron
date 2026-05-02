#!/usr/bin/env bash
# A100×8 (p4d.24xlarge 단일 노드) 의 OOM demo launcher.
#
# 동일한 hyperparameter (mbs=4, ga=64, s=8192, full recomp) 로
# 4 가지 PP × TP = 8 조합을 각각 실행해 nvtop 으로 OOM 발생 여부 시연.
#
# **이 스크립트는 p4d 노드 위에서 직접 실행** (single-node, no rdzv).
# nvtop 은 별도 ssh terminal 에서 띄워 watch.
#
# Usage (PP TP 둘 다 명시, PP × TP = 8):
#   bash launch.sh 1 8     # PP=1 TP=8 — no PP, all TP. partition [40].
#   bash launch.sh 2 4     # PP=2 TP=4. partition [20, 20].
#   bash launch.sh 4 2     # PP=4 TP=2. partition [10, 10, 10, 10].
#   bash launch.sh 8 1     # PP=8 TP=1 — no TP, all PP. partition [5]×8.

set -euo pipefail
cd "$(dirname "$0")/../../../.."

PP="${1:?Usage: $0 <PP> <TP>  (PP*TP must = 8)}"
TP="${2:?Usage: $0 <PP> <TP>  (PP*TP must = 8)}"

if [ "$((PP * TP))" -ne 8 ]; then
    echo "[launch] ✗ PP*TP must = 8 (got PP=$PP TP=$TP, product=$((PP * TP)))" >&2
    exit 2
fi

CONFIG_REL="examples/heterogeneous/configs/qwen3_14b/alpaca_a100x8_oom_demo.yaml"
CONFIG="/home/ubuntu/nanotron/$CONFIG_REL"
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 2; }

NUM_LAYERS=40
if [ "$((NUM_LAYERS % PP))" -ne 0 ]; then
    echo "[launch] ✗ NUM_LAYERS=$NUM_LAYERS not divisible by PP=$PP — use PP ∈ {1,2,4,5,8}" >&2
    exit 2
fi
LAYER_PER_STAGE=$((NUM_LAYERS / PP))
PARTITION_VALUES=()
for ((i=0; i<PP; i++)); do PARTITION_VALUES+=("$LAYER_PER_STAGE"); done
PARTITION_BRACKET="[$(IFS=,; echo "${PARTITION_VALUES[*]}")]"
PARTITION_BRACKET="${PARTITION_BRACKET//,/, }"

# config sed override
sed -i "s|^  pp: .*|  pp: $PP|" "$CONFIG"
sed -i "s|^  tp: .*|  tp: $TP|" "$CONFIG"
sed -i "s|^  pp_layer_partition: .*|  pp_layer_partition: $PARTITION_BRACKET|" "$CONFIG"

# yaml 에서 mbs/ga/seq 읽어 descriptor 생성 (yaml 변경 시 자동 반영).
MBS=$(awk '/^  micro_batch_size:/ { print $2 }' "$CONFIG")
GA=$(awk '/^  batch_accumulation_per_replica:/ { print $2 }' "$CONFIG")
SEQ=$(awk '/^  sequence_length:/ { print $2 }' "$CONFIG")
DESCRIPTOR="pp${PP}_tp${TP}_mbs${MBS}_ga${GA}_seq${SEQ}"
echo "[launch] PP=$PP TP=$TP partition=$PARTITION_BRACKET descriptor=$DESCRIPTOR"
echo "[launch] config diff:"
grep -E '^  (pp|tp|pp_layer_partition|micro_batch_size|sequence_length|batch_accumulation_per_replica|recompute_layer):' "$CONFIG"

# Cleanup any stale processes
pkill -9 -f run_train 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
sleep 1

mkdir -p /opt/dlami/nvme/oom_demo_logs
LOG_FILE="/opt/dlami/nvme/oom_demo_logs/${DESCRIPTOR}.log"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN     # OOM 시 NCCL noise 줄이기
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

echo "[launch] starting torchrun --nproc_per_node=8 (single node, world_size=8)"
echo "[launch] log → $LOG_FILE"
echo "[launch] (별도 terminal 에서: nvtop  ← 8 GPU memory + util 실시간)"
echo

# Single-node launch (no master_addr/rdzv needed beyond localhost).
uv run torchrun \
  --standalone \
  --nproc_per_node=8 \
  run_train.py \
  --config-file "$CONFIG_REL" \
  2>&1 | tee "$LOG_FILE"

# OOM detection in log — Triton kernel 의 "out of memory" 도 함께 잡음 (PyTorch 의
# 표준 ``CUDA out of memory`` 외 Triton FA / RMSNorm 등에서 다른 메시지 사용).
if grep -q -i -E "CUDA out of memory|OutOfMemoryError|Triton Error \[CUDA\]: out of memory" "$LOG_FILE"; then
    echo
    echo "[launch] ✗ OOM detected — PP=$PP TP=$TP A100×8 으로는 14B mbs=$MBS s=$SEQ 이 안 들어감."
    exit 1
elif grep -q "iteration: 5 / 5" "$LOG_FILE"; then
    echo
    echo "[launch] ✓ 5 iterations 완주 — PP=$PP TP=$TP 는 fit 함."
fi
