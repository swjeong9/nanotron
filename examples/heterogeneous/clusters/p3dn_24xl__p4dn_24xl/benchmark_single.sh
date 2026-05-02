#!/usr/bin/env bash
# 2-node benchmark single run (p3dn.24xl V100 + p4dn.24xl A100, FP16).
#
# Usage:
#   bash benchmark_single.sh                                 # default config + 균등 partition
#   bash benchmark_single.sh <config_path> <partition>       # e.g. ../configs/qwen3_14b/alpaca_pp2_tp8_fp16.yaml 20-20

set -uo pipefail   # -e 빼고 (한 partition fail 해도 sweep 이어가도록)

NANOTRON=/home/ubuntu/nanotron
CLUSTER_DIR="$(cd "$(dirname "$0")" && pwd)"
NODES_JSON="$NANOTRON/examples/heterogeneous/nodes.json"
DEFAULT_CONFIG="examples/heterogeneous/configs/qwen3_14b/alpaca_pp2_tp8_fp16.yaml"

CONFIG_REL="${1:-$DEFAULT_CONFIG}"
PARTITION_OVERRIDE="${2:-}"
CONFIG="$NANOTRON/$CONFIG_REL"
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 2; }

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
ALL_IPS=("$NODE0_IP" "$NODE1_IP")

# Sync 강제
bash "$CLUSTER_DIR/sync.sh"

# config invariants + partition override.
# mbs / ga 는 yaml 자체에서 지정한 값 그대로 사용 (PP=8 은 mbs=1 ga=32, PP=2/4 는 mbs=2 ga=16).
# train_steps / sequence_length 만 sweep-wide 하게 강제.
ORIGINAL_PARTITION_LINE=$(grep '^  pp_layer_partition:' "$CONFIG" || echo "")
sed -i 's/^  train_steps: .*/  train_steps: 3/' "$CONFIG"
sed -i 's/^  sequence_length: .*/  sequence_length: 8192/' "$CONFIG"
sed -i 's/^      dataset_overwrite_cache: .*/      dataset_overwrite_cache: false/' "$CONFIG"

if [ -n "$PARTITION_OVERRIDE" ]; then
    NEW_PARTITION_BRACKET="[${PARTITION_OVERRIDE//-/, }]"
    sed -i "s|^  pp_layer_partition: .*|  pp_layer_partition: $NEW_PARTITION_BRACKET|" "$CONFIG"
fi

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

# Auto-derive descriptor — TP / PP 도 yaml 에서 읽어 PP=4 TP=4 / PP=2 TP=8 같은 다른 셋업 모두 지원.
MBS=$(awk '/^  micro_batch_size:/ { print $2 }' "$CONFIG")
GA=$(awk '/^  batch_accumulation_per_replica:/ { print $2 }' "$CONFIG")
SEQ=$(awk '/^  sequence_length:/ { print $2 }' "$CONFIG")
TS=$(awk '/^  train_steps:/ { print $2 }' "$CONFIG")
TP=$(awk '/^  tp:/ { print $2 }' "$CONFIG")
PP=$(awk '/^  pp:/ { print $2 }' "$CONFIG")
PARTITION=$(grep '^  pp_layer_partition:' "$CONFIG" \
            | sed -E 's/^.*: *\[//;s/\] *$//;s/ //g;s/,/-/g')
[ -z "$PARTITION" ] && PARTITION="auto"
RECOMPUTE_LAYER=$(grep '^  recompute_layer:' "$CONFIG" | awk '{ print $2 }')
RECOMPUTE_LAYER=${RECOMPUTE_LAYER:-false}
RECOMPUTE_TAG=""
[ "$RECOMPUTE_LAYER" = "true" ] && RECOMPUTE_TAG="_recomp"
DESCRIPTOR="mbs${MBS}_ga${GA}_seq${SEQ}_pp${PP}_tp${TP}${RECOMPUTE_TAG}_split${PARTITION}"

MODEL=$(basename "$(dirname "$CONFIG")")
CLUSTER="p3dn_24xl__p4dn_24xl"

OUT_DIR="/opt/dlami/nvme/runs/$CLUSTER/$MODEL/$DESCRIPTOR"
echo "[benchmark] cluster=$CLUSTER model=$MODEL descriptor=$DESCRIPTOR"
echo "[benchmark] OUT_DIR=$OUT_DIR"

FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010"

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR"/*

# 1) cleanup 양 노드
for ip in "${ALL_IPS[@]}"; do
    ssh -o BatchMode=yes "ubuntu@$ip" \
        "pkill -9 -f run_train; pkill -9 -f torchrun; pkill -f 'dcgmi dmon'; pkill -f 'while :'" 2>/dev/null || true
done
sleep 2

# 2) DCGM dmon background — wallclock-prefix wrapper (line 별 epoch.ms 부착).
#    dcgmi 가 -d 1000 이여도 실제 ~3 Hz 로 emit 하는 이슈 우회.
declare -a DCGM_START_TS
for i in 0 1; do
    ip="${ALL_IPS[$i]}"
    DCGM_START_TS[$i]=$(ssh -o BatchMode=yes "ubuntu@$ip" \
        "ts=\$(date +%s.%N); nohup bash $NANOTRON/examples/heterogeneous/dcgm_dmon_wrap.sh \"$FIELDS\" 1000 \
         > /opt/dlami/nvme/dcgm_node${i}.txt 2>&1 & echo \$ts" 2>/dev/null || echo "0")
done

# 3) NIC + nvidia-smi sampler background.
for i in 0 1; do
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

# 4) 학습 시작 — NODE 0 master, NODE 1 worker
START_TS=$(date -u +%s.%N)

ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "
    cd $NANOTRON
    RDZV_HOST=$NODE0_IP nohup bash $CLUSTER_DIR/launch_node0.sh \"$CONFIG_REL\" \
        > /opt/dlami/nvme/train_node0.log 2>&1 &
" || true

# NODE 0 :29500 대기 (uv resync 등으로 60s 초과해도 break 만, abort 안 함)
NODE0_READY_DEADLINE=$(($(date +%s) + 60))
while ! ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "ss -tnlp 2>/dev/null | grep -q ':29500 '" 2>/dev/null; do
    if [ $(date +%s) -ge $NODE0_READY_DEADLINE ]; then
        echo "[benchmark] :29500 wait timed out (60s) — continuing anyway, workers will block on rendezvous if master not up yet"
        break
    fi
    sleep 1
done

# NODE 1 launch (foreground monitoring — output stage 라 죽으면 전체 abort)
timeout 1800 ssh -o BatchMode=yes "ubuntu@$NODE1_IP" "
    cd $NANOTRON
    RDZV_HOST=$NODE0_IP bash $CLUSTER_DIR/launch_node1.sh \"$CONFIG_REL\" \
        > /opt/dlami/nvme/train_node1.log 2>&1
" || NODE1_RC=$?

# NODE 1 끝났으면 NODE 0 도 곧. 60s 대기 후 강제 kill
DEADLINE=$(($(date +%s) + 60))
while ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "pgrep -f run_train > /dev/null" 2>/dev/null; do
    if [ $(date +%s) -ge $DEADLINE ]; then
        ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "pkill -9 -f run_train; pkill -9 -f torchrun" 2>/dev/null
        break
    fi
    sleep 2
done

if [ "${NODE1_RC:-0}" -ne 0 ]; then
    echo "[benchmark] ✗ NODE 1 exit rc=$NODE1_RC — killing NODE 0 immediately"
    ssh -o BatchMode=yes "ubuntu@$NODE0_IP" "pkill -9 -f run_train; pkill -9 -f torchrun" 2>/dev/null
fi

END_TS=$(date -u +%s.%N)

# 5) 정리
for ip in "${ALL_IPS[@]}"; do
    ssh -o BatchMode=yes "ubuntu@$ip" \
        "pkill -f 'dcgmi dmon'; pkill -f 'while :'" 2>/dev/null || true
done
sleep 1

# 6) 결과 회수
for i in 0 1; do
    ip="${ALL_IPS[$i]}"
    scp -q "ubuntu@$ip:/opt/dlami/nvme/dcgm_node${i}.txt"        "$OUT_DIR/dcgm_node${i}.txt" 2>/dev/null || true
    scp -q "ubuntu@$ip:/opt/dlami/nvme/nic_node${i}.txt"         "$OUT_DIR/nic_node${i}.txt" 2>/dev/null || true
    scp -q "ubuntu@$ip:/opt/dlami/nvme/nvidia_smi_node${i}.txt"  "$OUT_DIR/nvidia_smi_node${i}.txt" 2>/dev/null || true
    scp -q "ubuntu@$ip:/opt/dlami/nvme/train_node${i}.log"       "$OUT_DIR/train_node${i}.log" 2>/dev/null || true
    scp -q "ubuntu@$ip:/opt/dlami/nvme/torchrun_node${i}.log"         "$OUT_DIR/torchrun_node${i}.log" 2>/dev/null || true
done

# 7) OOM detection (Triton 도 포함)
OOM=false
if grep -q -i -E "CUDA out of memory|OutOfMemoryError|Triton Error \[CUDA\]: out of memory" \
    "$OUT_DIR/train_node0.log" "$OUT_DIR/train_node1.log" "$OUT_DIR/torchrun_node0.log" "$OUT_DIR/torchrun_node1.log" 2>/dev/null; then
    OOM=true
fi

COMPLETED_STEPS_RAW=$(grep -c "After training_step" "$OUT_DIR/torchrun_node0.log" 2>/dev/null || true)
COMPLETED_STEPS_RAW=${COMPLETED_STEPS_RAW:-0}
NPROC=8
COMPLETED_STEPS=$((COMPLETED_STEPS_RAW / NPROC))

# 8) max nvidia-smi memory (각 노드의 GPU max)
declare -a NVSMI_MAX
for i in 0 1; do
    NVSMI_MAX[$i]=$(awk '{for(i=2;i<=NF;i++) if($i+0>m) m=$i+0} END{print m+0}' \
        "$OUT_DIR/nvidia_smi_node${i}.txt" 2>/dev/null || echo 0)
    NVSMI_MAX[$i]=${NVSMI_MAX[$i]:-0}
done

cat > "$OUT_DIR/meta.json" <<EOF
{
  "config_path": "$CONFIG_REL",
  "model": "$MODEL",
  "cluster": "$CLUSTER",
  "gpu_node0": "V100 ×8 (p3dn.24xlarge)",
  "gpu_node1": "A100 ×8 (p4dn.24xlarge)",
  "descriptor": "$DESCRIPTOR",
  "config": "production-like (mbs=$MBS ga=$GA seq=$SEQ tp=$TP pp=$PP train_steps=$TS partition=$PARTITION)",
  "mbs": $MBS,
  "ga": $GA,
  "seq_len": $SEQ,
  "train_steps": $TS,
  "tp": $TP,
  "pp": $PP,
  "gbs_seqs": $((MBS * GA)),
  "gbs_tokens_per_step": $((MBS * GA * SEQ)),
  "pp_layer_partition_str": "$PARTITION",
  "recompute_layer": $RECOMPUTE_LAYER,
  "start_ts_utc": "$START_TS",
  "end_ts_utc": "$END_TS",
  "elapsed_sec": $(awk "BEGIN { print $END_TS - $START_TS }"),
  "dcgm_start_ts_node0": "${DCGM_START_TS[0]}",
  "dcgm_start_ts_node1": "${DCGM_START_TS[1]}",
  "oom": $OOM,
  "completed_steps": $COMPLETED_STEPS,
  "node0_rc": ${NODE0_RC:-0},
  "node1_rc": ${NODE1_RC:-0},
  "nvidia_smi_max_used_MiB_node0": ${NVSMI_MAX[0]},
  "nvidia_smi_max_used_MiB_node1": ${NVSMI_MAX[1]}
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
