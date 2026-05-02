#!/usr/bin/env bash
# NODE 0 (g6e.48xlarge, 8× L40S) — Stages 0 + 1 of PP=4. nproc_per_node=8.
# rank 0–3 = Stage 0 (TP=4 intra-node, PCIe), rank 4–7 = Stage 1.
# Master rendezvous endpoint.
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
# EFA 강제 비활성화 — heterogeneous (g6e × p4d) cross-instance EFA 가
# AWS-OFI-NCCL plugin 에서 RC: 265 / "Unknown error" 발생시키며 hang.
# GDRDMA-asymmetric (p4d=GDR / g6e=Shared) 가능성 의심. TCP fallback 으로 안정성 확보.
export NCCL_IB_DISABLE=1                              # IB/RDMA 비활성
export NCCL_NET_PLUGIN=none                           # AWS-OFI-NCCL plugin auto-load 차단 (libfabric 우회)
export FI_PROVIDER='^efa'                             # libfabric 의 efa provider exclude (belt-and-suspenders)
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp135s0}"  # g6e.48xl default route NIC (this instance — verify with `ip route get 1`)
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

uv run torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --rdzv_backend=static \
  --master_addr=${RDZV_HOST} \
  --master_port=29500 \
  --max_restarts=0 \
  run_train.py \
  --config-file "$CONFIG" \
  2>&1 | tee /opt/dlami/nvme/torchrun_node0.log
