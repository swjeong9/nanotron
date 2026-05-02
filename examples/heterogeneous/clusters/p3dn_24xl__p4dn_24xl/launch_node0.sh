#!/usr/bin/env bash
# NODE 0 (p3dn.24xlarge, 8× V100 32GB SXM2 — NVLink intra-node).
# 기본 PP=2 TP=8 의 stage 0 (rank 0-7 모두 TP=8). 다른 setup 일 시 stage 분배 변경.
# Master rendezvous endpoint.
#
# Usage:
#   RDZV_HOST=<NODE0_IP> bash launch_node0.sh [config_path]

set -euo pipefail
cd "$(dirname "$0")/../../../.."

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp2_tp8_fp16.yaml}"
RDZV_HOST="${RDZV_HOST:?RDZV_HOST must be set to NODE 0 IP (this node)}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
# EFA 강제 비활성화 — cross-instance (p3dn × p4dn) EFA 가 AWS-OFI-NCCL plugin 에서
# AWS 의 known limitation 으로 hang/error. TCP fallback 으로 안정성 확보.
# (자세히: ../g6e_48xl__p4d_24xl/ 의 디버깅 기록 + feedback_efa_cross_instance 메모리 참조)
export NCCL_IB_DISABLE=1                              # IB/RDMA 비활성
export NCCL_NET_PLUGIN=none                           # AWS-OFI-NCCL plugin auto-load 차단 (libfabric 우회)
export FI_PROVIDER='^efa'                             # libfabric 의 efa provider exclude (belt-and-suspenders)
# p3dn default route NIC: 인스턴스 launch 후 `ip route get 1 | awk '{print $5}'` 로 확인 후 override.
# 일반적으로 p3dn.24xlarge 는 ``ens5`` 또는 ``enp135s0`` 등.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens5}"
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
