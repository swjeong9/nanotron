#!/usr/bin/env bash
# 3-node benchmark single run (2× g6e.12xl + 1× p4d.24xl, PP=4 TP=4).
#
# Usage:
#   bash benchmark_single.sh                                 # default config + 균등 partition
#   bash benchmark_single.sh <config_path> <partition>       # e.g. ../configs/qwen3_14b/alpaca_pp4_tp4.yaml 8-11-11-10

set -uo pipefail   # -e 빼고 (한 partition fail 해도 sweep 이어가도록)

NANOTRON=/home/ubuntu/nanotron
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NODES_JSON="$NANOTRON/examples/heterogeneous/nodes.json"
DEFAULT_CONFIG="examples/heterogeneous/configs/qwen3_14b/alpaca_pp4_tp4.yaml"

CONFIG_REL="${1:-$DEFAULT_CONFIG}"
PARTITION_OVERRIDE="${2:-}"
CONFIG="$NANOTRON/$CONFIG_REL"
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 2; }

# Worker IPs from nodes.json (by node_rank).
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
NODE2_IP=$(uv run --no-project python -c "
import json
nodes = json.load(open('$NODES_JSON'))['nodes']
print(next(n['private_ip'] for n in nodes if n.get('node_rank') == 2))
")
ALL_IPS=("$NODE0_IP" "$NODE1_IP" "$NODE2_IP")
NPROC_PER_NODE=(4 4 8)

# Sync 강제 — partition override 이전에 한 번
bash "$CLUSTER_DIR/sync.sh"

# config sed 강제 — production-like benchmark 의 invariants + partition override.
# ga=16 : alpaca 662 packed bins (s=8192) ÷ (mbs=4 × ga=16) = 10.3 step → 5 step 안전 fit.
# ga=64 면 256 bins/step × 5 = 1280 필요해 데이터 부족 (assertion fail).
ORIGINAL_PARTITION_LINE=$(grep '^  pp_layer_partition:' "$CONFIG" || echo "")
sed -i 's/^  micro_batch_size: .*/  micro_batch_size: 4/' "$CONFIG"
sed -i 's/^  batch_accumulation_per_replica: .*/  batch_accumulation_per_replica: 16/' "$CONFIG"
sed -i 's/^  train_steps: .*/  train_steps: 5/' "$CONFIG"
sed -i 's/^  sequence_length: .*/  sequence_length: 8192/' "$CONFIG"
sed -i 's/^      dataset_overwrite_cache: .*/      dataset_overwrite_cache: false/' "$CONFIG"

if [ -n "$PARTITION_OVERRIDE" ]; then
    NEW_PARTITION_BRACKET="[${PARTITION_OVERRIDE//-/, }]"
    sed -i "s|^  pp_layer_partition: .*|  pp_layer_partition: $NEW_PARTITION_BRACKET|" "$CONFIG"
fi

# 변경된 config 양 노드 재 sync + verify
for ip in "${ALL_IPS[@]}"; do
    rsync -aq "$CONFIG" "ubuntu@$ip:$CONFIG"
done
LOCAL_CFG_MD5=$(md5sum "$CONFIG" | awk '{print $1}')
for ip in "${ALL_IPS[@]}"; do
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
CLUSTER="g6e_12xl_x2__p4d_24xl_pp4_tp4"

OUT_DIR="/opt/dlami/nvme/runs/$CLUSTER/$MODEL/$DESCRIPTOR"
echo "[benchmark] cluster=$CLUSTER model=$MODEL descriptor=$DESCRIPTOR"
echo "[benchmark] OUT_DIR=$OUT_DIR"

FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010"

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR"/*

# 1) cleanup 모든 노드
for ip in "${ALL_IPS[@]}"; do
    ssh -o BatchMode=yes "ubuntu@$ip" \
        "pkill -9 -f run_train; pkill -9 -f torchrun; pkill -f 'dcgmi dmon'; pkill -f 'while :'" 2>/dev/null || true
done
sleep 2

# 2) DCGM dmon 모든 노드 background.
declare -a DCGM_START_TS
for i in 0 1 2; do
    ip="${ALL_IPS[$i]}"
    DCGM_START_TS[$i]=$(ssh -o BatchMode=yes "ubuntu@$ip" \
        "ts=\$(date +%s.%N); nohup dcgmi dmon -e \"$FIELDS\" -d 1000 \
         > /opt/dlami/nvme/dcgm_node${i}.txt 2>&1 & echo \$ts" 2>/dev/null || echo "0")
done

# 3) NIC + nvidia-smi sampler 모든 노드 background.
for i in 0 1 2; do
    ip="${ALL_IPS[$i]}"
    ssh -o BatchMode=yes "ubuntu@$ip" "
        IFACE=\$(ip route get 1 | awk '{print \$5; exit}')
        nohup bash -c \"
        while :; do
            ts=\\\$(date +%s.%N)
            line=\\\$(awk -v iface=\\\"\$IFACE\\\" '\\\$1 ~ iface\\\":\\\" { print \\\$2, \\\$3, \\\$10, \\\$11 }' /proc/net/dev)
            echo \\\"\\\$ts \\\$line\\\"
            sleep 0.1
        done
        \" > /opt/dlami/nvme/nic_node${i}.txt 2>&1 &

        nohup bash -c '
        while :; do
            ts=\$(date +%s.%N)
            used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd,)
            echo \"\$ts \$used\"
            sleep 1
        done
        ' > /opt/dlami/nvme/nvidia_smi_node${i}.txt 2>&1 &
    " || true
done

sleep 1

# 4) 학습 시작 — NODE 0 master, NODE 1 + NODE 2 worker
START_TS=$(date -u +%s.%N)

ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "
    cd $NANOTRON
    RDZV_HOST=$NODE0_IP nohup bash $CLUSTER_DIR/launch_node0.sh \"$CONFIG_REL\" \
        > /opt/dlami/nvme/train_node0.log 2>&1 &
" || true

# NODE 0 의 :29500 가 열릴 때까지 대기 (다른 노드 launch 전)
NODE0_READY_DEADLINE=$(($(date +%s) + 60))
while ! ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "ss -tnlp 2>/dev/null | grep -q ':29500 '" 2>/dev/null; do
    if [ $(date +%s) -ge $NODE0_READY_DEADLINE ]; then
        # uv run 이 첫 실행 시 패키지 resync + torch import 로 60s 초과할 수 있음.
        # rendezvous 는 worker launch 후에도 따라잡으므로 break 만 — script 는 abort X.
        echo "[benchmark] :29500 wait timed out (60s) — continuing anyway, workers will block on rendezvous if master not up yet"
        break
    fi
    sleep 1
done

# NODE 1 launch (background, 그 다음 NODE 2 도 background)
ssh -o BatchMode=yes "ubuntu@$NODE1_IP" "
    cd $NANOTRON
    RDZV_HOST=$NODE0_IP nohup bash $CLUSTER_DIR/launch_node1.sh \"$CONFIG_REL\" \
        > /opt/dlami/nvme/train_node1.log 2>&1 &
" || true

# NODE 2 launch (foreground 으로 monitoring — 마지막 stage 라 죽으면 전체 abort)
timeout 1800 ssh -o BatchMode=yes "ubuntu@$NODE2_IP" "
    cd $NANOTRON
    RDZV_HOST=$NODE0_IP bash $CLUSTER_DIR/launch_node2.sh \"$CONFIG_REL\" \
        > /opt/dlami/nvme/train_node2.log 2>&1
" || NODE2_RC=$?

# NODE 2 끝났으면 NODE 0/1 도 곧 끝나야 함. 60s 대기 후 강제 kill.
for ip in "$NODE0_IP" "$NODE1_IP"; do
    DEADLINE=$(($(date +%s) + 60))
    while ssh -o BatchMode=yes "ubuntu@$ip" "pgrep -f run_train > /dev/null" 2>/dev/null; do
        if [ $(date +%s) -ge $DEADLINE ]; then
            ssh -o BatchMode=yes "ubuntu@$ip" "pkill -9 -f run_train; pkill -9 -f torchrun" 2>/dev/null
            break
        fi
        sleep 2
    done
done

if [ "${NODE2_RC:-0}" -ne 0 ]; then
    echo "[benchmark] ✗ NODE 2 exit rc=$NODE2_RC — killing NODE 0/1 immediately"
    for ip in "$NODE0_IP" "$NODE1_IP"; do
        ssh -o BatchMode=yes "ubuntu@$ip" "pkill -9 -f run_train; pkill -9 -f torchrun" 2>/dev/null
    done
fi

END_TS=$(date -u +%s.%N)

# 5) 정리 — sampler 들 stop
for ip in "${ALL_IPS[@]}"; do
    ssh -o BatchMode=yes "ubuntu@$ip" \
        "pkill -f 'dcgmi dmon'; pkill -f 'while :'" 2>/dev/null || true
done
sleep 1

# 6) 결과 회수 — 모든 노드 dev 로 scp
for i in 0 1 2; do
    ip="${ALL_IPS[$i]}"
    scp -q "ubuntu@$ip:/opt/dlami/nvme/dcgm_node${i}.txt"        "$OUT_DIR/dcgm_node${i}.txt" 2>/dev/null || true
    scp -q "ubuntu@$ip:/opt/dlami/nvme/nic_node${i}.txt"         "$OUT_DIR/nic_node${i}.txt" 2>/dev/null || true
    scp -q "ubuntu@$ip:/opt/dlami/nvme/nvidia_smi_node${i}.txt"  "$OUT_DIR/nvidia_smi_node${i}.txt" 2>/dev/null || true
    scp -q "ubuntu@$ip:/opt/dlami/nvme/train_node${i}.log"       "$OUT_DIR/train_node${i}.log" 2>/dev/null || true
done

# 7) OOM detection (모든 노드 log 검사)
OOM=false
if grep -q -E "CUDA out of memory|OutOfMemoryError" \
    "$OUT_DIR/train_node0.log" "$OUT_DIR/train_node1.log" "$OUT_DIR/train_node2.log" 2>/dev/null; then
    OOM=true
fi

# NODE 0 의 4 ranks (TP=4) 모두 같은 step 마다 "After training_step" 출력 →
# raw count 가 nproc_per_node 배 부풀려짐. 4 로 나눠 실제 step 수 산출.
COMPLETED_STEPS_RAW=$(grep -c "After training_step" "$OUT_DIR/train_node0.log" 2>/dev/null || true)
COMPLETED_STEPS_RAW=${COMPLETED_STEPS_RAW:-0}
COMPLETED_STEPS=$((COMPLETED_STEPS_RAW / 4))

# 8) max nvidia-smi memory (각 노드의 GPU max)
declare -a NVSMI_MAX
for i in 0 1 2; do
    NVSMI_MAX[$i]=$(awk '{for(i=2;i<=NF;i++) if($i+0>m) m=$i+0} END{print m+0}' \
        "$OUT_DIR/nvidia_smi_node${i}.txt" 2>/dev/null || echo 0)
    NVSMI_MAX[$i]=${NVSMI_MAX[$i]:-0}
done

cat > "$OUT_DIR/meta.json" <<EOF
{
  "config_path": "$CONFIG_REL",
  "model": "$MODEL",
  "cluster": "$CLUSTER",
  "gpu_node0": "L40S ×4 (g6e.12xlarge, stage 0)",
  "gpu_node1": "L40S ×4 (g6e.12xlarge, stage 1)",
  "gpu_node2": "A100 ×8 (p4d.24xlarge, stages 2 + 3)",
  "descriptor": "$DESCRIPTOR",
  "config": "production-like (mbs=$MBS ga=$GA seq=$SEQ tp=4 pp=4 train_steps=$TS partition=$PARTITION)",
  "mbs": $MBS,
  "ga": $GA,
  "seq_len": $SEQ,
  "train_steps": $TS,
  "tp": 4,
  "pp": 4,
  "gbs_seqs": $((MBS * GA)),
  "gbs_tokens_per_step": $((MBS * GA * SEQ)),
  "pp_layer_partition_str": "$PARTITION",
  "recompute_layer": $RECOMPUTE_LAYER,
  "start_ts_utc": "$START_TS",
  "end_ts_utc": "$END_TS",
  "elapsed_sec": $(awk "BEGIN { print $END_TS - $START_TS }"),
  "dcgm_start_ts_node0": "${DCGM_START_TS[0]}",
  "dcgm_start_ts_node1": "${DCGM_START_TS[1]}",
  "dcgm_start_ts_node2": "${DCGM_START_TS[2]}",
  "oom": $OOM,
  "completed_steps": $COMPLETED_STEPS,
  "node0_rc": ${NODE0_RC:-0},
  "node1_rc": ${NODE1_RC:-0},
  "node2_rc": ${NODE2_RC:-0},
  "nvidia_smi_max_used_MiB_node0": ${NVSMI_MAX[0]},
  "nvidia_smi_max_used_MiB_node1": ${NVSMI_MAX[1]},
  "nvidia_smi_max_used_MiB_node2": ${NVSMI_MAX[2]}
}
EOF

# 9) Restore baseline partition (sweep 친화)
if [ -n "$PARTITION_OVERRIDE" ] && [ -n "$ORIGINAL_PARTITION_LINE" ]; then
    sed -i "s|^  pp_layer_partition: .*|$ORIGINAL_PARTITION_LINE|" "$CONFIG"
    for ip in "${ALL_IPS[@]}"; do
        rsync -aq "$CONFIG" "ubuntu@$ip:$CONFIG" 2>/dev/null || true
    done
fi

echo
echo "Single-run benchmark complete (oom=$OOM, completed_steps=$COMPLETED_STEPS)"
echo "Results in $OUT_DIR"
echo "elapsed: $(awk "BEGIN { print $END_TS - $START_TS }") sec"
