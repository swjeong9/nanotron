#!/usr/bin/env bash
# Multi-node PP=2 launch — NODE 1 (172.31.40.226 / g5.xlarge / A10G).
# rdzv_endpoint points at NODE 0. Run AFTER NODE 0 starts (within ~30s).

set -euo pipefail
cd "$(dirname "$0")/../.."

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens5}"
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

uv run torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=172.31.31.40:29500 \
  --max_restarts=0 \
  run_train.py \
  --config-file examples/heterogeneous/config_llama32_1b_alpaca_pp2.yaml \
  2>&1 | tee /opt/dlami/nvme/pp2_node1.log
