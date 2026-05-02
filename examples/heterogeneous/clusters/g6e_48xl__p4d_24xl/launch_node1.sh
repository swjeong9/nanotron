#!/usr/bin/env bash
# NODE 1 (p4d.24xlarge, 8× A100) — Stages 2 + 3 of PP=4. nproc_per_node=8.
# rank 8–11 = Stage 2 (TP=4 intra-p4d, NVSwitch), rank 12–15 = Stage 3.
# rdzv_endpoint = NODE 0. NODE 0 :29500 가 열린 뒤 launch.
#
# Usage:
#   RDZV_HOST=<NODE0_IP> bash launch_node1.sh [config_path]

set -euo pipefail
cd "$(dirname "$0")/../../../.."

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp4_tp4.yaml}"
RDZV_HOST="${RDZV_HOST:?RDZV_HOST must be set to NODE 0 IP}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
# EFA 비활성화 — 위 launch_node0.sh comment 참조 (cross-instance OFI RC: 265 hang 회피).
export NCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=none
export FI_PROVIDER='^efa'
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens32}"   # p4d standard
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

uv run torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --rdzv_backend=static \
  --master_addr=${RDZV_HOST} \
  --master_port=29500 \
  --max_restarts=0 \
  run_train.py \
  --config-file "$CONFIG" \
  2>&1 | tee /opt/dlami/nvme/torchrun_node1.log
