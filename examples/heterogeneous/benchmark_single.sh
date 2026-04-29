#!/usr/bin/env bash
# Production-like 한 config 한 번 깊이 측정 — 병목 진단용.
#
# 동시에 수집:
#   1. DCGM 16 field 시계열 (양 노드)              → power/util/temp/PCIe
#   2. /proc/net/dev 의 NIC RX/TX bytes (양 노드)  → 실제 inter-node bandwidth
#   3. 학습 stdout (Before/After train_batch_iter / training_step timestamp)
#
# 결과 위치 (cluster + model + descriptor 자동 도출):
#   /opt/dlami/nvme/runs/<cluster>/<model>/<descriptor>/
#   ├── meta.json
#   ├── dcgm_node{0,1}.txt
#   ├── nic_node{0,1}.txt
#   └── train_node{0,1}.log
#
# Usage:
#   bash benchmark_single.sh                                                  # default config
#   bash benchmark_single.sh configs/llama32_1b/alpaca_pp2_split8-8.yaml      # 명시 config
#   bash benchmark_single.sh configs/llama32_1b/alpaca_pp2_split8-8.yaml 11-5 # partition override

set -euo pipefail

NANOTRON=/home/ubuntu/nanotron
NODE1_IP=172.31.40.226
DEFAULT_CONFIG="examples/heterogeneous/configs/llama32_1b/alpaca_pp2.yaml"

CONFIG_REL="${1:-$DEFAULT_CONFIG}"
PARTITION_OVERRIDE="${2:-}"
CONFIG="$NANOTRON/$CONFIG_REL"
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 2; }

# 모든 multi-node 실행 직전 NODE 1 와 sync 강제 — md5 verify 까지. 실패 시 abort.
bash "$NANOTRON/examples/heterogeneous/sync_to_node1.sh"

# Save baseline state for restore at end (sweep usability — 하나의 yaml 여러
# partition 으로 sweep 시 매 iteration 마다 baseline 으로 돌려놓는다).
ORIGINAL_PARTITION_LINE=$(grep '^  pp_layer_partition:' "$CONFIG" || echo "")

# config 의 mbs/ga/train_steps 강제 + dataset 캐시 사용.
sed -i 's/^  micro_batch_size: .*/  micro_batch_size: 2/' "$CONFIG"
sed -i 's/^  batch_accumulation_per_replica: .*/  batch_accumulation_per_replica: 64/' "$CONFIG"
sed -i 's/^  train_steps: .*/  train_steps: 10/' "$CONFIG"
sed -i 's/^      dataset_overwrite_cache: .*/      dataset_overwrite_cache: false/' "$CONFIG"

# Partition override: ``--partition 11-5`` → ``pp_layer_partition: [11, 5]``
if [ -n "$PARTITION_OVERRIDE" ]; then
    NEW_PARTITION_BRACKET="[${PARTITION_OVERRIDE//-/, }]"
    sed -i "s|^  pp_layer_partition: .*|  pp_layer_partition: $NEW_PARTITION_BRACKET|" "$CONFIG"
fi

# Partition override 직후 config 만 다시 push (sync_to_node1 이 이미 전체 트리를
# 동기화했지만, 그 뒤 sed 으로 partition override 했으므로 한 번 더 NODE 1 에 push).
ssh -o BatchMode=yes ubuntu@$NODE1_IP "mkdir -p $(dirname "$CONFIG")" 2>/dev/null || true
rsync -aq "$CONFIG" "ubuntu@$NODE1_IP:$CONFIG"
# 양 노드의 config md5 일치 검증.
LOCAL_CFG_MD5=$(md5sum "$CONFIG" | awk '{print $1}')
REMOTE_CFG_MD5=$(ssh -o BatchMode=yes ubuntu@$NODE1_IP "md5sum $CONFIG" 2>/dev/null | awk '{print $1}')
if [ "$LOCAL_CFG_MD5" != "$REMOTE_CFG_MD5" ]; then
    echo "[benchmark] ✗ config md5 mismatch after partition override; aborting" >&2
    exit 2
fi

# Auto-derive descriptor from config (model = parent dir, partition + mbs + ga).
MODEL=$(basename "$(dirname "$CONFIG")")    # e.g. "llama32_1b"
MBS=$(awk '/^  micro_batch_size:/ { print $2 }' "$CONFIG")
GA=$(awk '/^  batch_accumulation_per_replica:/ { print $2 }' "$CONFIG")
SEQ=$(awk '/^  sequence_length:/ { print $2 }' "$CONFIG")
TS=$(awk '/^  train_steps:/ { print $2 }' "$CONFIG")
PARTITION=$(grep '^  pp_layer_partition:' "$CONFIG" \
            | sed -E 's/^.*: *\[//;s/\] *$//;s/ //g;s/,/-/g')
[ -z "$PARTITION" ] && PARTITION="auto"

# recompute_layer 켜져 있으면 descriptor 에 ``recomp`` 태그 추가 — no-recompute
# 결과와 디렉토리 충돌 방지.
RECOMPUTE_LAYER=$(grep '^  recompute_layer:' "$CONFIG" | awk '{ print $2 }')
RECOMPUTE_LAYER=${RECOMPUTE_LAYER:-false}
RECOMPUTE_TAG=""
if [ "$RECOMPUTE_LAYER" = "true" ]; then
    RECOMPUTE_TAG="_recomp"
fi
DESCRIPTOR="mbs${MBS}_ga${GA}_seq${SEQ}${RECOMPUTE_TAG}_split${PARTITION}"

# Cluster 식별 — nvidia-smi 에서 GPU 종류 받아 ``l4__a10g_pp2`` 식으로.
gpu_short() {
    # "NVIDIA L4" → "l4", "NVIDIA A10G" → "a10g"
    local raw="$1"
    raw=${raw#NVIDIA }
    raw=${raw#GeForce }
    raw=$(echo "$raw" | tr 'A-Z' 'a-z' | tr ' /' '__' | sed 's/__*/_/g;s/^_//;s/_$//')
    echo "$raw"
}
GPU0_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU1_NAME=$(ssh -o BatchMode=yes ubuntu@$NODE1_IP "nvidia-smi --query-gpu=name --format=csv,noheader,nounits" 2>/dev/null | head -1)
GPU0_SHORT=$(gpu_short "${GPU0_NAME:-unknown}")
GPU1_SHORT=$(gpu_short "${GPU1_NAME:-unknown}")
CLUSTER="${GPU0_SHORT}__${GPU1_SHORT}_pp2"

OUT_DIR="/opt/dlami/nvme/runs/$CLUSTER/$MODEL/$DESCRIPTOR"
echo "[benchmark] cluster=$CLUSTER model=$MODEL descriptor=$DESCRIPTOR"
echo "[benchmark] OUT_DIR=$OUT_DIR"

FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010"

mkdir -p "$OUT_DIR"
rm -rf "$OUT_DIR"/*

# 1) cleanup
pkill -9 -f run_train 2>/dev/null || true
pkill -9 -f torchrun 2>/dev/null || true
pkill -f "dcgmi dmon" 2>/dev/null || true
ssh -o BatchMode=yes ubuntu@$NODE1_IP "pkill -9 -f run_train; pkill -9 -f torchrun; pkill -f 'dcgmi dmon'" 2>/dev/null || true
sleep 1
rm -f /opt/dlami/nvme/pp2_node0.log
ssh -o BatchMode=yes ubuntu@$NODE1_IP "rm -f /opt/dlami/nvme/pp2_node1.log" 2>/dev/null || true

# 2) DCGM dmon 양 노드 백그라운드. row_index → 절대시각 복원을 위해
# DCGM_START_TS_NODE{0,1} 직접 기록.
DCGM_START_TS_NODE0=$(date +%s.%N)
nohup dcgmi dmon -e "$FIELDS" -d 1000 \
    > "$OUT_DIR/dcgm_node0.txt" 2>&1 &
DCGM_START_TS_NODE1=$(ssh -o BatchMode=yes ubuntu@$NODE1_IP \
    "ts=\$(date +%s.%N); nohup dcgmi dmon -e \"$FIELDS\" -d 1000 \
     > /opt/dlami/nvme/dcgm_node1.txt 2>&1 & echo \$ts" 2>/dev/null || echo "0")

# 3) NIC sampler 양 노드 백그라운드 — /proc/net/dev 10Hz (sleep 0.1).
NIC_NODE0=enp39s0
NIC_NODE1=ens5

nohup bash -c "
while :; do
  ts=\$(date +%s.%N)
  line=\$(awk -v iface=\"$NIC_NODE0\" '\$1 ~ iface\":\" { print \$2, \$3, \$10, \$11 }' /proc/net/dev)
  echo \"\$ts \$line\"
  sleep 0.1
done
" > "$OUT_DIR/nic_node0.txt" 2>&1 &
NIC_PID_NODE0=$!

# nvidia-smi memory.used 1Hz polling 양 노드 — DCGM 의 FBUSD 와 별개로
# 직접 측정 (sustained memory pressure 시계열). schema: ``ts memory_used MiB``.
nohup bash -c "
while :; do
  ts=\$(date +%s.%N)
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo \"\$ts \$used\"
  sleep 1
done
" > "$OUT_DIR/nvidia_smi_node0.txt" 2>&1 &
SMI_PID_NODE0=$!

ssh -o BatchMode=yes ubuntu@$NODE1_IP "
nohup bash -c '
while :; do
  ts=\$(date +%s.%N)
  used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo \"\$ts \$used\"
  sleep 1
done
' > /opt/dlami/nvme/nvidia_smi_node1.txt 2>&1 &
" || true

ssh -o BatchMode=yes ubuntu@$NODE1_IP "
nohup bash -c '
while :; do
  ts=\$(date +%s.%N)
  line=\$(awk -v iface=\"$NIC_NODE1\" \"\\\$1 ~ iface\\\":\\\" { print \\\$2, \\\$3, \\\$10, \\\$11 }\" /proc/net/dev)
  echo \"\$ts \$line\"
  sleep 0.1
done
' > /opt/dlami/nvme/nic_node1.txt 2>&1 &
" || true

# 3.5) iperf3 baseline ceiling.
ssh -o BatchMode=yes ubuntu@$NODE1_IP "pkill -f 'iperf3 -s'; nohup iperf3 -s -1 > /tmp/iperf3s.log 2>&1 &" || true
sleep 1
IPERF_OUT=$(iperf3 -c $NODE1_IP -t 5 -i 1 -J 2>/dev/null || echo "{}")
IPERF_BPS=$(echo "$IPERF_OUT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('end',{}).get('sum_sent',{}).get('bits_per_second', 0))" 2>/dev/null || echo 0)
IPERF_GBPS=$(awk "BEGIN { printf \"%.2f\", $IPERF_BPS / 1e9 }")
echo "iperf3 baseline NODE0 → NODE1: $IPERF_GBPS Gbps"

sleep 1

# 4) 학습 시작
START_TS=$(date -u +%s.%N)

bash "$NANOTRON/examples/heterogeneous/launch_pp2_node0.sh" "$CONFIG_REL" \
    > "$OUT_DIR/train_node0.log" 2>&1 &
NODE0_PID=$!

# Rendezvous 가 절대 안 되면 영원히 멈추니 60s timeout.
PORT_DEADLINE=$(($(date +%s) + 60))
while ! ss -tnlp 2>/dev/null | grep -q ":29500 "; do
    if [ $(date +%s) -ge $PORT_DEADLINE ]; then
        echo "[benchmark] :29500 never opened; aborting"
        break
    fi
    sleep 1
done

timeout 1200 ssh -o BatchMode=yes ubuntu@$NODE1_IP \
    "bash $NANOTRON/examples/heterogeneous/launch_pp2_node1.sh \"$CONFIG_REL\" \
     > /opt/dlami/nvme/train_node1.log 2>&1" || NODE1_RC=$?

# NODE 1 fail 즉시 NODE 0 abort — NCCL block 으로 NODE 0 가 영영 안 끝나므로.
if [ "${NODE1_RC:-0}" -ne 0 ]; then
    echo "[benchmark] ✗ NODE 1 exit rc=$NODE1_RC — killing NODE 0 immediately"
    pkill -9 -f run_train 2>/dev/null || true
    pkill -9 -f torchrun 2>/dev/null || true
    kill -9 "$NODE0_PID" 2>/dev/null || true
    NODE0_RC=125
else
    SECONDS=0
    while kill -0 "$NODE0_PID" 2>/dev/null && [ $SECONDS -lt 60 ]; do sleep 1; done
    if kill -0 "$NODE0_PID" 2>/dev/null; then
        kill -9 "$NODE0_PID" 2>/dev/null || true
        NODE0_RC=124
    else
        wait "$NODE0_PID" 2>/dev/null || NODE0_RC=$?
    fi
fi
END_TS=$(date -u +%s.%N)

# 5) 정리
pkill -f "dcgmi dmon" 2>/dev/null || true
ssh -o BatchMode=yes ubuntu@$NODE1_IP "pkill -f 'dcgmi dmon'; pkill -f 'while :' " 2>/dev/null || true
kill "$NIC_PID_NODE0" 2>/dev/null || true
kill "$SMI_PID_NODE0" 2>/dev/null || true
sleep 1

# 6) 결과 회수
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/dcgm_node1.txt" "$OUT_DIR/dcgm_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/nic_node1.txt" "$OUT_DIR/nic_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/nvidia_smi_node1.txt" "$OUT_DIR/nvidia_smi_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/train_node1.log" "$OUT_DIR/train_node1.log" 2>/dev/null || true

# 6.5) nvidia-smi memory.used 의 max 양 노드에서 추출 (MiB).
NVSMI_MAX_NODE0=$(awk '{print $2}' "$OUT_DIR/nvidia_smi_node0.txt" 2>/dev/null \
    | grep -E "^[0-9]+$" | sort -n | tail -1)
NVSMI_MAX_NODE1=$(awk '{print $2}' "$OUT_DIR/nvidia_smi_node1.txt" 2>/dev/null \
    | grep -E "^[0-9]+$" | sort -n | tail -1)
NVSMI_MAX_NODE0=${NVSMI_MAX_NODE0:-0}
NVSMI_MAX_NODE1=${NVSMI_MAX_NODE1:-0}

# 7) OOM detection — torch / CUDA 의 OOM 시그니처를 grep.
OOM=false
if grep -q -E "CUDA out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|HIP out of memory" \
    "$OUT_DIR/train_node0.log" "$OUT_DIR/train_node1.log" 2>/dev/null; then
    OOM=true
    echo "[benchmark] OOM detected"
fi
# 추가 단서: NCCL 등으로 step 0 도 안 끝남 (After training_step 한 줄도 없음).
# grep -c 가 0 match 일 때 exit 1 + "0" 출력 → ``|| echo 0`` 추가로 "0\n0" 두 줄
# 만들어 meta.json JSON 파싱 깨뜨림. ``|| true`` 가 정답 (grep 가 이미 "0" 출력하니).
COMPLETED_STEPS=$(grep -c "After training_step" "$OUT_DIR/train_node0.log" 2>/dev/null || true)
COMPLETED_STEPS=${COMPLETED_STEPS:-0}

cat > "$OUT_DIR/meta.json" <<EOF
{
  "config_path": "$CONFIG_REL",
  "model": "$MODEL",
  "cluster": "$CLUSTER",
  "gpu_node0": "${GPU0_NAME:-unknown}",
  "gpu_node1": "${GPU1_NAME:-unknown}",
  "descriptor": "$DESCRIPTOR",
  "config": "production-like (mbs=$MBS ga=$GA seq=$SEQ train_steps=$TS partition=$PARTITION)",
  "mbs": $MBS,
  "ga": $GA,
  "seq_len": $SEQ,
  "train_steps": $TS,
  "gbs_seqs": $((MBS * GA)),
  "gbs_tokens_per_step": $((MBS * GA * SEQ)),
  "pp_layer_partition_str": "$PARTITION",
  "recompute_layer": $RECOMPUTE_LAYER,
  "start_ts_utc": "$START_TS",
  "end_ts_utc": "$END_TS",
  "elapsed_sec": $(awk "BEGIN { print $END_TS - $START_TS }"),
  "node0_iface": "$NIC_NODE0",
  "node1_iface": "$NIC_NODE1",
  "dcgm_start_ts_node0": "$DCGM_START_TS_NODE0",
  "dcgm_start_ts_node1": "$DCGM_START_TS_NODE1",
  "iperf_baseline_gbps": $IPERF_GBPS,
  "oom": $OOM,
  "completed_steps": $COMPLETED_STEPS,
  "node0_rc": ${NODE0_RC:-0},
  "node1_rc": ${NODE1_RC:-0},
  "nvidia_smi_max_used_MiB_node0": $NVSMI_MAX_NODE0,
  "nvidia_smi_max_used_MiB_node1": $NVSMI_MAX_NODE1
}
EOF

# 8) Restore baseline partition (sweep 친화적).
if [ -n "$PARTITION_OVERRIDE" ] && [ -n "$ORIGINAL_PARTITION_LINE" ]; then
    sed -i "s|^  pp_layer_partition: .*|$ORIGINAL_PARTITION_LINE|" "$CONFIG"
    rsync -aq "$CONFIG" "ubuntu@$NODE1_IP:$CONFIG" 2>/dev/null || true
fi

echo
echo "=========================================="
echo "Single-run benchmark complete (oom=$OOM, completed_steps=$COMPLETED_STEPS)"
echo "Results in $OUT_DIR"
ls -la "$OUT_DIR"
echo "elapsed: $(awk "BEGIN { print $END_TS - $START_TS }") sec"
echo
echo "Next: uv run --no-project --with matplotlib python \\"
echo "    examples/heterogeneous/plot_single.py --run-dir $OUT_DIR"
