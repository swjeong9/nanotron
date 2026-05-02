#!/usr/bin/env bash
# NODE 2 (p4d.24xlarge, 8× A100 40GB) — Stage 2 + Stage 3 of PP=4.
# 8 GPU 가 두 개의 TP=4 그룹 (intra-node) 으로 갈라져 stage 2 (rank 8–11) 와
# stage 3 (rank 12–15) 를 동시에 처리. NVLink/NVSwitch 로 양 stage 모두 빠른 TP comm.
# rdzv_endpoint = NODE 0. NODE 0 :29500 가 열린 뒤 launch.
#
# Usage:
#   RDZV_HOST=<NODE0_IP> bash launch_node2.sh [config_path]

set -euo pipefail
cd "$(dirname "$0")/../../../.."

CONFIG="${1:-examples/heterogeneous/configs/qwen3_14b/alpaca_pp4_tp4.yaml}"
RDZV_HOST="${RDZV_HOST:?RDZV_HOST must be set to NODE 0 IP}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=INFO
# p4d.24xlarge 는 EFA 지원 → IB 미사용 강제 해제 검토 (성능). 안전 default 는 disable.
export NCCL_IB_DISABLE=0                              # EFA 재시도 (local_pg fix 적용 후).
# p4d 의 primary NIC interface (NCCL_IB_DISABLE=1 fallback 용). ``ip -br link`` 로 확인.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens32}"
export OMP_NUM_THREADS=4
export PATH="$HOME/.local/bin:$PATH"

uv run torchrun \
  --nproc_per_node=8 \
  --nnodes=3 \
  --node_rank=2 \
  --rdzv_backend=static \
  --master_addr=${RDZV_HOST} \
  --master_port=29500 \
  --max_restarts=0 \
  run_train.py \
  --config-file "$CONFIG" \
  2>&1 | tee /opt/dlami/nvme/pp4_node2.log
