#!/usr/bin/env bash
# HF → nanotron 변환 + S3 업로드 한 방 스크립트.
#
# GPU 가 있는 노드 (예: g6.12xlarge) 에서 실행. Llama / Qwen 3 모두 지원.
# Architecture 는 HF config.json 의 ``architectures`` field 에서 자동 감지 —
# 사용자가 ARCH 잘못 박아 silent miscompile 되는 사고 방지.
#
# Usage (해당 노드의 nanotron 디렉토리에서):
#   bash examples/heterogeneous/convert_and_upload.sh meta-llama/Llama-3.2-3B llama-3.2-3b
#   bash examples/heterogeneous/convert_and_upload.sh meta-llama/Llama-3.1-8B llama-3.1-8b
#   bash examples/heterogeneous/convert_and_upload.sh Qwen/Qwen3-0.6B          qwen-3-0.6b
#   bash examples/heterogeneous/convert_and_upload.sh Qwen/Qwen3-14B           qwen-3-14b
#
# 처리:
#   1. HF model 다운로드 → /opt/dlami/nvme/<S3_NAME>_hf/
#   2. config.json 의 architectures 로 arch 자동 감지 (override: ARCH=llama|qwen3)
#   3. nanotron 변환 → /opt/dlami/nvme/<S3_NAME>_nanotron/
#   4. S3 업로드 → s3://swj-nanotron-model/<S3_NAME>/nanotron/
#
# 양 worker 노드 (학습 노드) 에서는 다음과 같이 가져옴:
#   aws s3 sync s3://swj-nanotron-model/<S3_NAME>/nanotron/ /opt/dlami/nvme/<S3_NAME_LOCAL>_nanotron/

set -euo pipefail

HF_REPO="${1:?Usage: $0 <hf_repo> <s3_name>  e.g. Qwen/Qwen3-14B qwen-3-14b}"
S3_NAME="${2:?Usage: $0 <hf_repo> <s3_name>}"
S3_BUCKET="${S3_BUCKET:-s3://swj-nanotron-model}"
ARCH_OVERRIDE="${ARCH:-}"     # 빈 값이면 auto-detect, 박으면 강제 사용

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

# 2) Architecture 자동 감지 (또는 override 사용)
HF_CONFIG="$HF_DIR/config.json"
[ -f "$HF_CONFIG" ] || { echo "[convert] ✗ $HF_CONFIG not found after HF download" >&2; exit 2; }

# config.json 의 architectures[0] 로 결정. 모르는 arch 는 즉시 abort.
DETECTED_ARCH=$(uv run --no-project python -c "
import json, sys
arch_list = json.load(open('$HF_CONFIG')).get('architectures') or []
if not arch_list:
    print('UNKNOWN', file=sys.stderr); sys.exit(2)
hf_arch = arch_list[0]
mapping = {
    'LlamaForCausalLM': 'llama',
    'Qwen3ForCausalLM': 'qwen3',
}
nt_arch = mapping.get(hf_arch)
if nt_arch is None:
    sys.stderr.write(f'unsupported HF architecture: {hf_arch}\n')
    sys.exit(2)
print(nt_arch)
") || { echo "[convert] ✗ arch detection failed" >&2; exit 2; }

if [ -n "$ARCH_OVERRIDE" ]; then
    ARCH="$ARCH_OVERRIDE"
    if [ "$ARCH" != "$DETECTED_ARCH" ]; then
        echo "[convert] ⚠ ARCH override = '$ARCH' but detected '$DETECTED_ARCH' from $HF_CONFIG"
        echo "[convert]   override 가 의도된 게 아니면 ARCH 환경변수를 비우고 재실행 권장."
    fi
else
    ARCH="$DETECTED_ARCH"
    echo "[convert] step 2a: detected arch = $ARCH (from config.json architectures)"
fi

# 3) Convert (GPU).
echo "[convert] step 2b: HF → nanotron conversion ($ARCH)"
case "$ARCH" in
    llama)
        uv run torchrun --nproc_per_node=1 -m examples.llama.convert_hf_to_nanotron \
            --checkpoint_path="$HF_DIR" \
            --save_path="$NT_DIR"
        ;;
    qwen3)
        uv run torchrun --nproc_per_node=1 -m examples.qwen.convert_hf_qwen3_to_nanotron \
            --hf-model-path="$HF_DIR" \
            --save-path="$NT_DIR"
        ;;
    *)
        echo "[convert] ✗ unknown ARCH=$ARCH (expected: llama | qwen3)" >&2
        exit 1
        ;;
esac

# 4) Post-conversion sanity check (Qwen3 면 q_norm 디렉토리 + _use_qk_norm=true 확인)
if [ "$ARCH" = "qwen3" ]; then
    NT_CONFIG="$NT_DIR/model_config.json"
    Q_NORM_DIR="$NT_DIR/model/model/decoder/0/pp_block/attn/q_norm"
    if [ ! -d "$Q_NORM_DIR" ]; then
        echo "[convert] ✗ Qwen3 변환 결과에 q_norm 디렉토리 없음: $Q_NORM_DIR" >&2
        echo "[convert]   examples/qwen/convert_hf_qwen3_to_nanotron.py 가 실제 실행됐는지 확인" >&2
        exit 3
    fi
    if ! grep -q '"_use_qk_norm": true' "$NT_CONFIG"; then
        echo "[convert] ✗ $NT_CONFIG 에 _use_qk_norm: true 누락 — 변환 경로 의심" >&2
        exit 3
    fi
    echo "[convert] ✓ Qwen3 sanity check (q_norm dir + _use_qk_norm=true) passed"
fi

# 5) S3 upload
echo "[convert] step 3: upload to S3"
aws s3 sync "$NT_DIR/" "$S3_BUCKET/$S3_NAME/nanotron/"

echo
echo "[convert] ✓ done (arch=$ARCH)"
echo "  s3://$S3_BUCKET/$S3_NAME/nanotron/"
echo
echo "Cleanup (선택):"
echo "  rm -rf $HF_DIR  # HF 원본 (디스크 회수)"
echo "  rm -rf $NT_DIR  # 변환 결과 (S3 에 이미 올라감)"
