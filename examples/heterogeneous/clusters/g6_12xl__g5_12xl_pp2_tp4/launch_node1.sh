#!/usr/bin/env bash
# NODE 1 (g5.12xlarge, 4× A10G) — stage 1 of PP=2 with intra-node TP=4.
# rdzv_endpoint points at NODE 0. Run AFTER NODE 0 starts (within ~30s).
#
# Usage:
#   bash launch_node1.sh [config_path]

set -euo pipefail
cd "$(dirname "$0")/../../../.."

CONFIG="${1:-examples/heterogeneous/configs/llama32_3b/alpaca_pp2_tp4.yaml}"
RDZV_HOST="${RDZV_HOST:?RDZV_HOST must be set to NODE 0 IP}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens5}"
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

uv run torchrun \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=1 \
  --rdzv_backend=static \
  --master_addr=${RDZV_HOST} \
  --master_port=29500 \
  --max_restarts=0 \
  run_train.py \
  --config-file "$CONFIG" \
  2>&1 | tee /opt/dlami/nvme/pp2_node1.log
