#!/usr/bin/env bash
# Multi-node PP=2 launch — NODE 0 (master / 172.31.31.40 / g6.xlarge / L4).
# Pair with launch_pp2_node1.sh on the other node. Start node1 within ~30s.
#
# Usage:
#   bash launch_pp2_node0.sh [config_path]
# Default config_path:
#   examples/heterogeneous/configs/llama32_1b/alpaca_pp2_split8-8.yaml

set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG="${1:-examples/heterogeneous/configs/llama32_1b/alpaca_pp2_split8-8.yaml}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp39s0}"
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

uv run torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=0 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=172.31.31.40:29500 \
  --max_restarts=0 \
  run_train.py \
  --config-file "$CONFIG" \
  2>&1 | tee /opt/dlami/nvme/pp2_node0.log
