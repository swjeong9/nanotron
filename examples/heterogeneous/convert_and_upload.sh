#!/usr/bin/env bash
# HF Llama → nanotron 변환 + S3 업로드 한 방 스크립트.
#
# GPU 가 있는 노드 (예: g6.12xlarge) 에서 실행. 3B / 8B 등 모델 이름만 바꿔서 재사용.
#
# Usage (해당 노드의 nanotron 디렉토리에서):
#   bash examples/heterogeneous/convert_and_upload.sh meta-llama/Llama-3.2-3B llama-3.2-3b
#   bash examples/heterogeneous/convert_and_upload.sh meta-llama/Llama-3.1-8B llama-3.1-8b
#
# 처리:
#   1. HF model 다운로드 → /opt/dlami/nvme/<S3_NAME>_hf/
#   2. nanotron 변환 → /opt/dlami/nvme/<S3_NAME>_nanotron/
#   3. S3 업로드 → s3://swj-nanotron-model/<S3_NAME>/nanotron/
#   4. (선택) HF 임시 디렉토리 정리
#
# 양 worker 노드 (학습 노드) 에서는 다음과 같이 가져옴:
#   aws s3 sync s3://swj-nanotron-model/<S3_NAME>/nanotron/ /opt/dlami/nvme/<S3_NAME_LOCAL>_nanotron/

set -euo pipefail

HF_REPO="${1:?Usage: $0 <hf_repo> <s3_name>  e.g. meta-llama/Llama-3.2-3B llama-3.2-3b}"
S3_NAME="${2:?Usage: $0 <hf_repo> <s3_name>}"
S3_BUCKET="${S3_BUCKET:-s3://swj-nanotron-model}"

NANOTRON=/home/ubuntu/nanotron
HF_DIR="/opt/dlami/nvme/${S3_NAME}_hf"
NT_DIR="/opt/dlami/nvme/${S3_NAME}_nanotron"

echo "[convert] hf_repo=$HF_REPO  s3=$S3_BUCKET/$S3_NAME/nanotron/"
echo "[convert] hf_dir=$HF_DIR  nt_dir=$NT_DIR"

mkdir -p "$HF_DIR" "$NT_DIR"

# 1) HF download
echo "[convert] step 1: download HF weights"
cd "$NANOTRON"
uv run --no-project python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_REPO', local_dir='$HF_DIR')
print('HF download done')
"

# 2) Convert (GPU). Original convert script 은 ``examples/llama/convert_hf_to_nanotron.py``.
# Relative import 회피 위해 ``-m`` 플래그 사용.
echo "[convert] step 2: HF → nanotron conversion"
uv run torchrun --nproc_per_node=1 -m examples.llama.convert_hf_to_nanotron \
    --checkpoint_path="$HF_DIR" \
    --save_path="$NT_DIR"

# 3) S3 upload
echo "[convert] step 3: upload to S3"
aws s3 sync "$NT_DIR/" "$S3_BUCKET/$S3_NAME/nanotron/"

echo
echo "[convert] ✓ done"
echo "  s3://$S3_BUCKET/$S3_NAME/nanotron/"
echo
echo "Cleanup (선택):"
echo "  rm -rf $HF_DIR  # HF 원본 (디스크 회수)"
echo "  rm -rf $NT_DIR  # 변환 결과 (S3 에 이미 올라감)"
