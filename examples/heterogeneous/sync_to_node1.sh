#!/usr/bin/env bash
# 모든 multi-node 실험은 NODE 1 도 NODE 0 과 byte-identical 한 nanotron 트리를
# 가져야 한다. 그렇지 않으면 partition override / config / launch script 가
# silent 하게 desync 되어 partition mismatch (e.g. NODE 0 = [1, 15], NODE 1 =
# 옛날 baseline [8, 8]) 가 발생, 결과가 무의미해진다.
#
# 본 스크립트는:
#   1. examples/heterogeneous/ 와 src/nanotron/ 을 NODE 1 으로 ``--delete`` rsync
#      (NODE 1 의 stale 파일은 모두 제거)
#   2. 핵심 파일 md5sum 비교로 byte-identical 검증 — mismatch 시 exit 2
#
# Usage:
#   bash sync_to_node1.sh                 # NODE 0 에서 호출, NODE 1 으로 push
#
# Auto-called by ``benchmark_single.sh``, ``probe_memory.sh``, ``sweep_partitions.sh``.
# 직접 호출도 OK.

set -euo pipefail

NANOTRON=/home/ubuntu/nanotron
NODE1_IP=${NODE1_IP:-172.31.40.226}

cd "$NANOTRON"

echo "[sync] rsync examples/heterogeneous/ + src/nanotron/ → NODE1 (delete-after)"
rsync -aq --delete examples/heterogeneous/ "ubuntu@$NODE1_IP:$NANOTRON/examples/heterogeneous/"
rsync -aq --delete src/nanotron/ "ubuntu@$NODE1_IP:$NANOTRON/src/nanotron/"

# Sanity check: 핵심 파일 md5 이 일치하는지 확인. 일치 안 하면 진행 불가.
CRITICAL_FILES=(
    "examples/heterogeneous/launch_pp2_node1.sh"
    "examples/heterogeneous/launch_pp2_node0.sh"
    "src/nanotron/logging/base.py"
    "src/nanotron/models/base.py"
    "src/nanotron/trainer.py"
)

LOCAL_MD5=$(md5sum "${CRITICAL_FILES[@]}")
REMOTE_MD5=$(ssh -o BatchMode=yes "ubuntu@$NODE1_IP" \
    "cd $NANOTRON && md5sum ${CRITICAL_FILES[*]}" 2>/dev/null || echo "FAILED")

if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then
    echo "[sync] ✗ md5 mismatch — aborting"
    diff <(echo "$LOCAL_MD5") <(echo "$REMOTE_MD5")
    exit 2
fi

# config 도 검증 — partition override 직전이면 일치하지만 다른 process 의
# 잔여 변경이 있으면 mismatch 가 뜸.
CONFIG_LOCAL=$(md5sum examples/heterogeneous/configs/llama32_1b/alpaca_pp2.yaml 2>/dev/null || echo "missing")
CONFIG_REMOTE=$(ssh -o BatchMode=yes "ubuntu@$NODE1_IP" \
    "cd $NANOTRON && md5sum examples/heterogeneous/configs/llama32_1b/alpaca_pp2.yaml" 2>/dev/null || echo "missing")
if [ "$CONFIG_LOCAL" != "$CONFIG_REMOTE" ]; then
    echo "[sync] ✗ alpaca_pp2.yaml md5 mismatch — aborting"
    echo "  local:  $CONFIG_LOCAL"
    echo "  remote: $CONFIG_REMOTE"
    exit 2
fi

echo "[sync] ✓ NODE 1 in sync (${#CRITICAL_FILES[@]} core files + alpaca_pp2.yaml)"
