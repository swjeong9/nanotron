#!/usr/bin/env bash
# Multi-node sync 강제 — examples/heterogeneous/ + src/nanotron/ 트리를 양 worker
# 노드 (g6_12xl_node, g5_12xl_node) 에 ``rsync --delete`` push 후 md5 검증.
#
# 핵심 파일 mismatch 시 즉시 abort. nodes.json 에서 worker IP 읽음.

set -euo pipefail

NANOTRON=/home/ubuntu/nanotron
NODES_JSON="$NANOTRON/examples/heterogeneous/nodes.json"

if [ ! -f "$NODES_JSON" ]; then
    echo "[sync] ✗ $NODES_JSON not found" >&2
    exit 2
fi

NODE0_IP=$(uv run --no-project python -c "
import json
nodes = json.load(open('$NODES_JSON'))['nodes']
print(next(n['private_ip'] for n in nodes if n['instance_type'].startswith('g6.12x')))
")
NODE1_IP=$(uv run --no-project python -c "
import json
nodes = json.load(open('$NODES_JSON'))['nodes']
print(next(n['private_ip'] for n in nodes if n['instance_type'].startswith('g5.12x')))
")

echo "[sync] target NODE 0 (g6.12xlarge): $NODE0_IP"
echo "[sync] target NODE 1 (g5.12xlarge): $NODE1_IP"

cd "$NANOTRON"
for ip in "$NODE0_IP" "$NODE1_IP"; do
    echo "[sync] rsync → $ip"
    rsync -aq --delete examples/heterogeneous/ "ubuntu@$ip:$NANOTRON/examples/heterogeneous/"
    rsync -aq --delete src/nanotron/ "ubuntu@$ip:$NANOTRON/src/nanotron/"
    # run_train.py 도 root-level 파일이라 별도 push 필요. sft path 변경 등이 여기에.
    rsync -aq run_train.py "ubuntu@$ip:$NANOTRON/run_train.py"
done

# 핵심 파일 md5 양 노드 비교.
CRITICAL_FILES=(
    "run_train.py"
    "examples/heterogeneous/clusters/g6_12xl__g5_12xl_pp2_tp4/launch_node0.sh"
    "examples/heterogeneous/clusters/g6_12xl__g5_12xl_pp2_tp4/launch_node1.sh"
    "examples/heterogeneous/configs/llama32_3b/alpaca_pp2_tp4.yaml"
    "src/nanotron/logging/base.py"
    "src/nanotron/models/base.py"
    "src/nanotron/trainer.py"
)
LOCAL_MD5=$(md5sum "${CRITICAL_FILES[@]}")
for ip in "$NODE0_IP" "$NODE1_IP"; do
    REMOTE_MD5=$(ssh -o BatchMode=yes "ubuntu@$ip" \
        "cd $NANOTRON && md5sum ${CRITICAL_FILES[*]}" 2>/dev/null || echo "FAILED")
    if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then
        echo "[sync] ✗ md5 mismatch on $ip — aborting" >&2
        diff <(echo "$LOCAL_MD5") <(echo "$REMOTE_MD5") >&2
        exit 2
    fi
done

echo "[sync] ✓ both nodes in sync (${#CRITICAL_FILES[@]} core files)"
