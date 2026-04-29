#!/usr/bin/env bash
# Production-like 한 config 한 번 깊이 측정 — 병목 진단용.
#
# 동시에 수집:
#   1. DCGM 16 field 시계열 (양 노드)              → power/util/temp/PCIe
#   2. /proc/net/dev 의 NIC RX/TX bytes (양 노드)  → 실제 inter-node bandwidth
#   3. 학습 stdout (Before/After train_batch_iter / training_step timestamp)
#
# 결과 위치 (config 의 model + mbs/ga/partition 으로 자동 도출):
#   /opt/dlami/nvme/runs/<model>/<descriptor>/
#   ├── meta.json
#   ├── dcgm_node{0,1}.txt
#   ├── nic_node{0,1}.txt
#   └── train_node{0,1}.log
#
# Usage:
#   bash benchmark_single.sh                                          # default config
#   bash benchmark_single.sh configs/llama32_1b/alpaca_pp2_split8-8.yaml

set -euo pipefail

NANOTRON=/home/ubuntu/nanotron
NODE1_IP=172.31.40.226
DEFAULT_CONFIG="examples/heterogeneous/configs/llama32_1b/alpaca_pp2_split8-8.yaml"

CONFIG_REL="${1:-$DEFAULT_CONFIG}"
CONFIG="$NANOTRON/$CONFIG_REL"
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 2; }

# config 의 mbs/ga/train_steps 강제 + dataset 캐시 사용.
sed -i 's/^  micro_batch_size: .*/  micro_batch_size: 2/' "$CONFIG"
sed -i 's/^  batch_accumulation_per_replica: .*/  batch_accumulation_per_replica: 64/' "$CONFIG"
sed -i 's/^  train_steps: .*/  train_steps: 10/' "$CONFIG"
sed -i 's/^      dataset_overwrite_cache: .*/      dataset_overwrite_cache: false/' "$CONFIG"
ssh -o BatchMode=yes ubuntu@$NODE1_IP "mkdir -p $(dirname "$CONFIG")" 2>/dev/null || true
rsync -aq "$CONFIG" "ubuntu@$NODE1_IP:$CONFIG"

# Auto-derive descriptor from config (model = parent dir, partition + mbs + ga).
MODEL=$(basename "$(dirname "$CONFIG")")    # e.g. "llama32_1b"
MBS=$(awk '/^  micro_batch_size:/ { print $2 }' "$CONFIG")
GA=$(awk '/^  batch_accumulation_per_replica:/ { print $2 }' "$CONFIG")
SEQ=$(awk '/^  sequence_length:/ { print $2 }' "$CONFIG")
TS=$(awk '/^  train_steps:/ { print $2 }' "$CONFIG")
PARTITION=$(grep '^  pp_layer_partition:' "$CONFIG" \
            | sed -E 's/^.*: *\[//;s/\] *$//;s/ //g;s/,/-/g')
[ -z "$PARTITION" ] && PARTITION="auto"
DESCRIPTOR="mbs${MBS}_ga${GA}_split${PARTITION}"

OUT_DIR="/opt/dlami/nvme/runs/$MODEL/$DESCRIPTOR"
echo "[benchmark] model=$MODEL descriptor=$DESCRIPTOR"
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

# 2) DCGM dmon 양 노드 백그라운드. dcgmi dmon 출력에 timestamp 가 없으므로
# row_index → 절대시각 복원을 위해 ``DCGM_START_TS_NODE{0,1}`` 를 직접 기록.
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

# 3.5) iperf3 baseline ceiling — NODE 0 ↔ NODE 1 의 ENA 실측 cap.
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

while ! ss -tnlp 2>/dev/null | grep -q ":29500 "; do sleep 1; done

timeout 1200 ssh -o BatchMode=yes ubuntu@$NODE1_IP \
    "bash $NANOTRON/examples/heterogeneous/launch_pp2_node1.sh \"$CONFIG_REL\" \
     > /opt/dlami/nvme/train_node1.log 2>&1" || NODE1_RC=$?

SECONDS=0
while kill -0 "$NODE0_PID" 2>/dev/null && [ $SECONDS -lt 60 ]; do sleep 1; done
if kill -0 "$NODE0_PID" 2>/dev/null; then
    kill -9 "$NODE0_PID" 2>/dev/null || true
    NODE0_RC=124
else
    wait "$NODE0_PID" 2>/dev/null || NODE0_RC=$?
fi
END_TS=$(date -u +%s.%N)

# 5) 정리
pkill -f "dcgmi dmon" 2>/dev/null || true
ssh -o BatchMode=yes ubuntu@$NODE1_IP "pkill -f 'dcgmi dmon'; pkill -f 'while :' " 2>/dev/null || true
kill "$NIC_PID_NODE0" 2>/dev/null || true
sleep 1

# 6) 결과 회수
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/dcgm_node1.txt" "$OUT_DIR/dcgm_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/nic_node1.txt" "$OUT_DIR/nic_node1.txt" 2>/dev/null || true
scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/train_node1.log" "$OUT_DIR/train_node1.log" 2>/dev/null || true

cat > "$OUT_DIR/meta.json" <<EOF
{
  "config_path": "$CONFIG_REL",
  "model": "$MODEL",
  "descriptor": "$DESCRIPTOR",
  "config": "production-like (mbs=$MBS ga=$GA seq=$SEQ train_steps=$TS partition=$PARTITION)",
  "mbs": $MBS,
  "ga": $GA,
  "seq_len": $SEQ,
  "train_steps": $TS,
  "gbs_seqs": $((MBS * GA)),
  "gbs_tokens_per_step": $((MBS * GA * SEQ)),
  "pp_layer_partition_str": "$PARTITION",
  "start_ts_utc": "$START_TS",
  "end_ts_utc": "$END_TS",
  "elapsed_sec": $(awk "BEGIN { print $END_TS - $START_TS }"),
  "node0_iface": "$NIC_NODE0",
  "node1_iface": "$NIC_NODE1",
  "dcgm_start_ts_node0": "$DCGM_START_TS_NODE0",
  "dcgm_start_ts_node1": "$DCGM_START_TS_NODE1",
  "iperf_baseline_gbps": $IPERF_GBPS,
  "node0_rc": ${NODE0_RC:-0},
  "node1_rc": ${NODE1_RC:-0}
}
EOF

echo
echo "=========================================="
echo "Single-run benchmark complete."
echo "Results in $OUT_DIR"
ls -la "$OUT_DIR"
echo "elapsed: $(awk "BEGIN { print $END_TS - $START_TS }") sec"
echo
echo "Next: uv run --no-project --with matplotlib python \\"
echo "    examples/heterogeneous/plot_single.py --run-dir $OUT_DIR"
