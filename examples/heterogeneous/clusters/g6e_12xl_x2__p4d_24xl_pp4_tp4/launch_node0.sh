#!/usr/bin/env bash
# NODE 0 (g6e.12xlarge, 4× L40S) — Stage 0 of PP=4 with intra-node TP=4.
# Master rendezvous endpoint (master_addr = NODE 0 자기 자신).
#
# Usage:
#   RDZV_HOST=<NODE0_IP> bash launch_node0.sh [config_path]

set -euo pipefail
cd "$(dirname "$0")/../../../.."

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp4_tp4.yaml}"
RDZV_HOST="${RDZV_HOST:?RDZV_HOST must be set to NODE 0 IP (this node)}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                              # EFA 재시도 (local_pg fix 적용 후).
                                                      # EFA 미부착 환경에선 1 로 override 후 ENA TCP fallback.
# inter-node TCP socket interface (NCCL fallback / NCCL_IB_DISABLE=1 시) — ``ip -br link`` 로 확인 후 변경.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp39s0}"
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

uv run torchrun \
  --nproc_per_node=4 \
  --nnodes=3 \
  --node_rank=0 \
  --rdzv_backend=static \
  --master_addr=${RDZV_HOST} \
  --master_port=29500 \
  --max_restarts=0 \
  run_train.py \
  --config-file "$CONFIG" \
  2>&1 | tee /opt/dlami/nvme/pp4_node0.log
