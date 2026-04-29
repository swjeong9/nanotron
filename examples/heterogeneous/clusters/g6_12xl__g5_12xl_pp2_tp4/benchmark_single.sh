#!/usr/bin/env bash
# Production-like 학습 한 번 measurement (g6.12xl + g5.12xl, PP=2 TP=4).
#
# 양 노드 (worker 0 = g6.12xl, worker 1 = g5.12xl) 에 ssh launch + dev 에서
# 결과 수집. dev 노드는 worker 가 아니므로 GPU 없어도 됨.
#
# Usage:
#   bash benchmark_single.sh                                 # default config + 균등 partition
#   bash benchmark_single.sh <config_path> <partition>       # e.g. ../configs/llama32_3b/alpaca_pp2_tp4.yaml 12-16

set -uo pipefail   # -e 빼고 (한 partition fail 해도 sweep 이어가도록)

NANOTRON=/home/ubuntu/nanotron
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NODES_JSON="$NANOTRON/examples/heterogeneous/nodes.json"
DEFAULT_CONFIG="examples/heterogeneous/configs/llama32_3b/alpaca_pp2_tp4.yaml"

CONFIG_REL="${1:-$DEFAULT_CONFIG}"
PARTITION_OVERRIDE="${2:-}"
CONFIG="$NANOTRON/$CONFIG_REL"
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 2; }

# Worker IPs from nodes.json
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

# Sync 강제 — partition override 이전에 한 번
bash "$CLUSTER_DIR/sync.sh"

# config sed 강제 — mbs/ga/train_steps + partition override
ORIGINAL_PARTITION_LINE=$(grep '^  pp_layer_partition:' "$CONFIG" || echo "")
sed -i 's/^  micro_batch_size: .*/  micro_batch_size: 2/' "$CONFIG"
sed -i 's/^  batch_accumulation_per_replica: .*/  batch_accumulation_per_replica: 64/' "$CONFIG"
sed -i 's/^  train_steps: .*/  train_steps: 10/' "$CONFIG"
sed -i 's/^  sequence_length: .*/  sequence_length: 1024/' "$CONFIG"
sed -i 's/^      dataset_overwrite_cache: .*/      dataset_overwrite_cache: false/' "$CONFIG"

if [ -n "$PARTITION_OVERRIDE" ]; then
    NEW_PARTITION_BRACKET="[${PARTITION_OVERRIDE//-/, }]"
    sed -i "s|^  pp_layer_partition: .*|  pp_layer_partition: $NEW_PARTITION_BRACKET|" "$CONFIG"
fi

# 변경된 config 양 노드 재 sync + verify
for ip in "$NODE0_IP" "$NODE1_IP"; do
    rsync -aq "$CONFIG" "ubuntu@$ip:$CONFIG"
done
LOCAL_CFG_MD5=$(md5sum "$CONFIG" | awk '{print $1}')
for ip in "$NODE0_IP" "$NODE1_IP"; do
    REMOTE_CFG_MD5=$(ssh -o BatchMode=yes "ubuntu@$ip" "md5sum $CONFIG" 2>/dev/null | awk '{print $1}')
    if [ "$LOCAL_CFG_MD5" != "$REMOTE_CFG_MD5" ]; then
        echo "[benchmark] ✗ config md5 mismatch on $ip; aborting" >&2
        exit 2
    fi
done

# Auto-derive descriptor
MBS=$(awk '/^  micro_batch_size:/ { print $2 }' "$CONFIG")
GA=$(awk '/^  batch_accumulation_per_replica:/ { print $2 }' "$CONFIG")
SEQ=$(awk '/^  sequence_length:/ { print $2 }' "$CONFIG")
TS=$(awk '/^  train_steps:/ { print $2 }' "$CONFIG")
PARTITION=$(grep '^  pp_layer_partition:' "$CONFIG" \
            | sed -E 's/^.*: *\[//;s/\] *$//;s/ //g;s/,/-/g')
[ -z "$PARTITION" ] && PARTITION="auto"
RECOMPUTE_LAYER=$(grep '^  recompute_layer:' "$CONFIG" | awk '{ print $2 }')
RECOMPUTE_LAYER=${RECOMPUTE_LAYER:-false}
RECOMPUTE_TAG=""
[ "$RECOMPUTE_LAYER" = "true" ] && RECOMPUTE_TAG="_recomp"
DESCRIPTOR="mbs${MBS}_ga${GA}_seq${SEQ}_tp4${RECOMPUTE_TAG}_split${PARTITION}"

MODEL=$(basename "$(dirname "$CONFIG")")
CLUSTER="g6_12xl__g5_12xl_pp2_tp4"

OUT_DIR="/opt/dlami/nvme/runs/$CLUSTER/$MODEL/$DESCRIPTOR"
echo "[benchmark] cluster=$CLUSTER model=$MODEL descriptor=$DESCRIPTOR"
echo "[benchmark] OUT_DIR=$OUT_DIR"

FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010"

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR"/*

# 1) cleanup 양 노드
for ip in "$NODE0_IP" "$NODE1_IP"; do
    ssh -o BatchMode=yes "ubuntu@$ip" \
        "pkill -9 -f run_train; pkill -9 -f torchrun; pkill -f 'dcgmi dmon'; pkill -f 'while :'" 2>/dev/null || true
done
sleep 2

# 2) DCGM dmon 양 노드 background. 4 GPU 라 모든 GPU 데이터 capture.
DCGM_START_TS_NODE0=$(ssh -o BatchMode=yes "ubuntu@$NODE0_IP" \
    "ts=\$(date +%s.%N); nohup dcgmi dmon -e \"$FIELDS\" -d 1000 \
     > /opt/dlami/nvme/dcgm_node0.txt 2>&1 & echo \$ts" 2>/dev/null || echo "0")
DCGM_START_TS_NODE1=$(ssh -o BatchMode=yes "ubuntu@$NODE1_IP" \
    "ts=\$(date +%s.%N); nohup dcgmi dmon -e \"$FIELDS\" -d 1000 \
     > /opt/dlami/nvme/dcgm_node1.txt 2>&1 & echo \$ts" 2>/dev/null || echo "0")

# 3) NIC + nvidia-smi sampler 양 노드 background.
# NIC interface 자동 감지 — primary route 의 dev.
for ip in "$NODE0_IP" "$NODE1_IP"; do
    ssh -o BatchMode=yes "ubuntu@$ip" "
        IFACE=\$(ip route get 1 | awk '{print \$5; exit}')
        nohup bash -c \"
        while :; do
            ts=\\\$(date +%s.%N)
            line=\\\$(awk -v iface=\\\"\$IFACE\\\" '\\\$1 ~ iface\\\":\\\" { print \\\$2, \\\$3, \\\$10, \\\$11 }' /proc/net/dev)
            echo \\\"\\\$ts \\\$line\\\"
            sleep 0.1
        done
        \" > /opt/dlami/nvme/nic_${ip##*.}.txt 2>&1 &

        nohup bash -c '
        while :; do
            ts=\$(date +%s.%N)
            used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd,)
            echo \"\$ts \$used\"
            sleep 1
        done
        ' > /opt/dlami/nvme/nvidia_smi_${ip##*.}.txt 2>&1 &
    " || true
done

sleep 1

# 4) 학습 시작 — NODE 0 master, NODE 1 worker
START_TS=$(date -u +%s.%N)

ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "
    cd $NANOTRON
    RDZV_HOST=$NODE0_IP nohup bash $CLUSTER_DIR/launch_node0.sh \"$CONFIG_REL\" \
        > /opt/dlami/nvme/train_node0.log 2>&1 &
" || true

# NODE 0 의 :29500 가 열릴 때까지 대기 (NODE 1 launch 전)
NODE0_READY_DEADLINE=$(($(date +%s) + 60))
while ! ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "ss -tnlp 2>/dev/null | grep -q ':29500 '" 2>/dev/null; do
    if [ $(date +%s) -ge $NODE0_READY_DEADLINE ]; then
        echo "[benchmark] NODE 0 :29500 never opened; aborting"
        break
    fi
    sleep 1
done

# NODE 1 launch (foreground for monitoring its exit)
timeout 1800 ssh -o BatchMode=yes "ubuntu@$NODE1_IP" "
    cd $NANOTRON
    RDZV_HOST=$NODE0_IP bash $CLUSTER_DIR/launch_node1.sh \"$CONFIG_REL\" \
        > /opt/dlami/nvme/train_node1.log 2>&1
" || NODE1_RC=$?

# NODE 1 끝났으면 NODE 0 도 곧 끝나야 함. 60s 대기 후 강제 kill.
NODE0_DEADLINE=$(($(date +%s) + 60))
while ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "pgrep -f run_train > /dev/null" 2>/dev/null; do
    if [ $(date +%s) -ge $NODE0_DEADLINE ]; then
        ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "pkill -9 -f run_train; pkill -9 -f torchrun" 2>/dev/null
        NODE0_RC=124
        break
    fi
    sleep 2
done

if [ "${NODE1_RC:-0}" -ne 0 ]; then
    echo "[benchmark] ✗ NODE 1 exit rc=$NODE1_RC — killing NODE 0 immediately"
    ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "pkill -9 -f run_train; pkill -9 -f torchrun" 2>/dev/null
    NODE0_RC=125
fi

END_TS=$(date -u +%s.%N)

# 5) 정리 — sampler 들 stop
for ip in "$NODE0_IP" "$NODE1_IP"; do
    ssh -o BatchMode=yes "ubuntu@$ip" \
        "pkill -f 'dcgmi dmon'; pkill -f 'while :'" 2>/dev/null || true
done
sleep 1

# 6) 결과 회수 — 양 노드 모두 dev 로 scp
scp -q "ubuntu@$NODE0_IP:/opt/dlami/nvme/dcgm_node0.txt" "$OUT_DIR/dcgm_node0.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/dcgm_node1.txt" "$OUT_DIR/dcgm_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE0_IP:/opt/dlami/nvme/nic_${NODE0_IP##*.}.txt" "$OUT_DIR/nic_node0.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/nic_${NODE1_IP##*.}.txt" "$OUT_DIR/nic_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE0_IP:/opt/dlami/nvme/nvidia_smi_${NODE0_IP##*.}.txt" "$OUT_DIR/nvidia_smi_node0.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/nvidia_smi_${NODE1_IP##*.}.txt" "$OUT_DIR/nvidia_smi_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE0_IP:/opt/dlami/nvme/train_node0.log" "$OUT_DIR/train_node0.log" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/train_node1.log" "$OUT_DIR/train_node1.log" 2>/dev/null || true

# 7) OOM detection
OOM=false
if grep -q -E "CUDA out of memory|OutOfMemoryError" \
    "$OUT_DIR/train_node0.log" "$OUT_DIR/train_node1.log" 2>/dev/null; then
    OOM=true
fi
# NODE 0 의 4 ranks (TP=4) 모두 같은 step 마다 "After training_step" 출력 →
# raw count 가 nproc_per_node 배 부풀려짐. 4 로 나눠 실제 step 수 산출.
COMPLETED_STEPS_RAW=$(grep -c "After training_step" "$OUT_DIR/train_node0.log" 2>/dev/null || true)
COMPLETED_STEPS_RAW=${COMPLETED_STEPS_RAW:-0}
NPROC_PER_NODE=4
COMPLETED_STEPS=$((COMPLETED_STEPS_RAW / NPROC_PER_NODE))

# 8) max nvidia-smi memory (양 노드 4 GPU 의 max)
NVSMI_MAX_NODE0=$(awk '{for(i=2;i<=NF;i++) if($i+0>m) m=$i+0} END{print m+0}' \
    "$OUT_DIR/nvidia_smi_node0.txt" 2>/dev/null || echo 0)
NVSMI_MAX_NODE1=$(awk '{for(i=2;i<=NF;i++) if($i+0>m) m=$i+0} END{print m+0}' \
    "$OUT_DIR/nvidia_smi_node1.txt" 2>/dev/null || echo 0)
NVSMI_MAX_NODE0=${NVSMI_MAX_NODE0:-0}
NVSMI_MAX_NODE1=${NVSMI_MAX_NODE1:-0}

cat > "$OUT_DIR/meta.json" <<EOF
{
  "config_path": "$CONFIG_REL",
  "model": "$MODEL",
  "cluster": "$CLUSTER",
  "gpu_node0": "L4 ×4 (g6.12xlarge)",
  "gpu_node1": "A10G ×4 (g5.12xlarge)",
  "descriptor": "$DESCRIPTOR",
  "config": "production-like (mbs=$MBS ga=$GA seq=$SEQ tp=4 train_steps=$TS partition=$PARTITION)",
  "mbs": $MBS,
  "ga": $GA,
  "seq_len": $SEQ,
  "train_steps": $TS,
  "tp": 4,
  "pp": 2,
  "gbs_seqs": $((MBS * GA)),
  "gbs_tokens_per_step": $((MBS * GA * SEQ)),
  "pp_layer_partition_str": "$PARTITION",
  "recompute_layer": $RECOMPUTE_LAYER,
  "start_ts_utc": "$START_TS",
  "end_ts_utc": "$END_TS",
  "elapsed_sec": $(awk "BEGIN { print $END_TS - $START_TS }"),
  "dcgm_start_ts_node0": "$DCGM_START_TS_NODE0",
  "dcgm_start_ts_node1": "$DCGM_START_TS_NODE1",
  "oom": $OOM,
  "completed_steps": $COMPLETED_STEPS,
  "node0_rc": ${NODE0_RC:-0},
  "node1_rc": ${NODE1_RC:-0},
  "nvidia_smi_max_used_MiB_node0": $NVSMI_MAX_NODE0,
  "nvidia_smi_max_used_MiB_node1": $NVSMI_MAX_NODE1
}
EOF

# 9) Restore baseline partition (sweep 친화)
if [ -n "$PARTITION_OVERRIDE" ] && [ -n "$ORIGINAL_PARTITION_LINE" ]; then
    sed -i "s|^  pp_layer_partition: .*|$ORIGINAL_PARTITION_LINE|" "$CONFIG"
    for ip in "$NODE0_IP" "$NODE1_IP"; do
        rsync -aq "$CONFIG" "ubuntu@$ip:$CONFIG" 2>/dev/null || true
    done
fi

echo
echo "Single-run benchmark complete (oom=$OOM, completed_steps=$COMPLETED_STEPS)"
echo "Results in $OUT_DIR"
echo "elapsed: $(awk "BEGIN { print $END_TS - $START_TS }") sec"
