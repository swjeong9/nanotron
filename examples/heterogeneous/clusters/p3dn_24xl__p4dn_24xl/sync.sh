#!/usr/bin/env bash
# 2-node sync (p3dn.24xlarge + p4dn.24xlarge) — examples/heterogeneous + src/nanotron + run_train.py
# 를 양 worker 에 ``rsync --delete`` push 후 md5 검증. mismatch 즉시 abort.
#
# nodes.json 에서 ``node_rank`` field 로 노드 식별.

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
print(next(n['private_ip'] for n in nodes if n.get('node_rank') == 0))
")
NODE1_IP=$(uv run --no-project python -c "
import json
nodes = json.load(open('$NODES_JSON'))['nodes']
print(next(n['private_ip'] for n in nodes if n.get('node_rank') == 1))
")

echo "[sync] target NODE 0 (p3dn.24xlarge, stages 0+1): $NODE0_IP"
echo "[sync] target NODE 1 (p4dn.24xlarge,  stages 2+3): $NODE1_IP"

cd "$NANOTRON"
ALL_IPS=("$NODE0_IP" "$NODE1_IP")
for ip in "${ALL_IPS[@]}"; do
    echo "[sync] rsync → $ip"
    rsync -aq --delete examples/heterogeneous/ "ubuntu@$ip:$NANOTRON/examples/heterogeneous/"
    rsync -aq --delete src/nanotron/ "ubuntu@$ip:$NANOTRON/src/nanotron/"
    rsync -aq run_train.py "ubuntu@$ip:$NANOTRON/run_train.py"
    rsync -aq --delete examples/qwen/ "ubuntu@$ip:$NANOTRON/examples/qwen/"
done

# 핵심 파일 md5 양 노드 비교
CRITICAL_FILES=(
    "run_train.py"
    "examples/heterogeneous/clusters/p3dn_24xl__p4dn_24xl/launch_node0.sh"
    "examples/heterogeneous/clusters/p3dn_24xl__p4dn_24xl/launch_node1.sh"
    "examples/heterogeneous/configs/qwen3_14b/alpaca_pp2_tp8_fp16.yaml"
    "examples/heterogeneous/dcgm_dmon_wrap.sh"
    "src/nanotron/parallel/context.py"
    "src/nanotron/config/models_config.py"
    "src/nanotron/models/qwen.py"
    "src/nanotron/trainer.py"
)
LOCAL_MD5=$(md5sum "${CRITICAL_FILES[@]}")
for ip in "${ALL_IPS[@]}"; do
    REMOTE_MD5=$(ssh -o BatchMode=yes "ubuntu@$ip" \
        "cd $NANOTRON && md5sum ${CRITICAL_FILES[*]}" 2>/dev/null || echo "FAILED")
    if [ "$LOCAL_MD5" != "$REMOTE_MD5" ]; then
        echo "[sync] ✗ md5 mismatch on $ip — aborting" >&2
        diff <(echo "$LOCAL_MD5") <(echo "$REMOTE_MD5") >&2
        exit 2
    fi
done

echo "[sync] ✓ both nodes in sync (${#CRITICAL_FILES[@]} core files)"

# Data / model presence check
DATA_CHECKS=(
    "/opt/dlami/nvme/alpaca_sft_local/train.parquet"
    "/opt/dlami/nvme/qwen-3-14b_nanotron/model_config.json"
    "/opt/dlami/nvme/qwen-3-14b_nanotron/model/model/decoder/0"
    "/opt/dlami/nvme/qwen-3-14b_nanotron/model/model/decoder/39"
    "/opt/dlami/nvme/qwen-3-14b_nanotron/tokenizer.json"
)
for ip in "${ALL_IPS[@]}"; do
    for path in "${DATA_CHECKS[@]}"; do
        if ! ssh -o BatchMode=yes "ubuntu@$ip" "test -e $path" 2>/dev/null; then
            echo "[sync] ✗ missing on $ip: $path" >&2
            echo "[sync]   re-run prepare_alpaca.py + 'aws s3 sync s3://swj-nanotron-model/qwen-3-14b/nanotron/ /opt/dlami/nvme/qwen-3-14b_nanotron/' on the worker" >&2
            exit 3
        fi
    done
done
echo "[sync] ✓ data + checkpoint + tokenizer present on both nodes"
