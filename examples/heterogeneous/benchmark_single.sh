#!/usr/bin/env bash
# Production-like 한 config 한 번 깊이 측정 — 병목 진단용.
#
# 동시에 수집:
#   1. DCGM 16 field 시계열 (양 노드)         → power/util/temp/PCIe
#   2. /proc/net/dev 의 NIC RX/TX bytes (양 노드)  → 실제 inter-node bandwidth
#   3. 학습 stdout (Before/After train_batch_iter timestamp)
#
# config 는 ``config_llama32_1b_alpaca_pp2.yaml`` 그대로 사용. 학습 step 수만
# 더 길게 잡아 measurement noise 를 줄임 (default train_steps=5 → 10).
#
# 결과 위치: /opt/dlami/nvme/single_run/
#   meta.json, dcgm_node{0,1}.txt, nic_node{0,1}.txt, train_node{0,1}.log
#
# Usage:
#   bash benchmark_single.sh

set -euo pipefail

NANOTRON=/home/ubuntu/nanotron
NODE1_IP=172.31.40.226
OUT_DIR=/opt/dlami/nvme/single_run
CONFIG=$NANOTRON/examples/heterogeneous/config_llama32_1b_alpaca_pp2.yaml

FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010"

# config 의 mbs/ga/train_steps 강제 (단일 production-like 측정 한 번).
sed -i 's/^  micro_batch_size: .*/  micro_batch_size: 2/' "$CONFIG"
sed -i 's/^  batch_accumulation_per_replica: .*/  batch_accumulation_per_replica: 64/' "$CONFIG"
sed -i 's/^  train_steps: .*/  train_steps: 10/' "$CONFIG"
sed -i 's/^      dataset_overwrite_cache: .*/      dataset_overwrite_cache: false/' "$CONFIG"
rsync -aq "$CONFIG" "ubuntu@$NODE1_IP:$CONFIG"

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

# 2) DCGM dmon 양 노드 백그라운드
nohup dcgmi dmon -e "$FIELDS" -d 1000 \
    > "$OUT_DIR/dcgm_node0.txt" 2>&1 &
ssh -o BatchMode=yes ubuntu@$NODE1_IP \
    "nohup dcgmi dmon -e \"$FIELDS\" -d 1000 \
     > /opt/dlami/nvme/dcgm_node1.txt 2>&1 &" || true

# 3) NIC sampler 양 노드 백그라운드 — /proc/net/dev 10Hz (sleep 0.1).
# schema: ``ts_unix RX_bytes RX_packets TX_bytes TX_packets``.
# 0.5s 였으나 PP 의 microbatch 단위 burst (172ms 간격) 를 놓치는 aliasing 때문에
# 100ms 로 올림.
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
# 학습 시작 전에 짧게 (5 sec) 양방향 — 측정 결과를 meta 에 추가.
ssh -o BatchMode=yes ubuntu@$NODE1_IP "pkill -f 'iperf3 -s'; nohup iperf3 -s -1 > /tmp/iperf3s.log 2>&1 &" || true
sleep 1
IPERF_OUT=$(iperf3 -c $NODE1_IP -t 5 -i 1 -J 2>/dev/null || echo "{}")
IPERF_BPS=$(echo "$IPERF_OUT" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('end',{}).get('sum_sent',{}).get('bits_per_second', 0))" 2>/dev/null || echo 0)
IPERF_GBPS=$(awk "BEGIN { printf \"%.2f\", $IPERF_BPS / 1e9 }")
echo "iperf3 baseline NODE0 → NODE1: $IPERF_GBPS Gbps"

sleep 1

# 4) 학습 시작
START_TS=$(date -u +%s.%N)

bash "$NANOTRON/examples/heterogeneous/launch_pp2_node0.sh" \
    > "$OUT_DIR/train_node0.log" 2>&1 &
NODE0_PID=$!

while ! ss -tnlp 2>/dev/null | grep -q ":29500 "; do sleep 1; done

timeout 1200 ssh -o BatchMode=yes ubuntu@$NODE1_IP \
    "bash /home/ubuntu/nanotron/examples/heterogeneous/launch_pp2_node1.sh \
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

MBS=$(awk '/^  micro_batch_size:/ { print $2 }' "$CONFIG")
GA=$(awk '/^  batch_accumulation_per_replica:/ { print $2 }' "$CONFIG")
SEQ=$(awk '/^  sequence_length:/ { print $2 }' "$CONFIG")
TS=$(awk '/^  train_steps:/ { print $2 }' "$CONFIG")
cat > "$OUT_DIR/meta.json" <<EOF
{
  "config": "production-like (mbs=$MBS ga=$GA seq=$SEQ train_steps=$TS)",
  "mbs": $MBS,
  "ga": $GA,
  "seq_len": $SEQ,
  "train_steps": $TS,
  "gbs_seqs": $((MBS * GA)),
  "gbs_tokens_per_step": $((MBS * GA * SEQ)),
  "start_ts_utc": "$START_TS",
  "end_ts_utc": "$END_TS",
  "elapsed_sec": $(awk "BEGIN { print $END_TS - $START_TS }"),
  "node0_iface": "$NIC_NODE0",
  "node1_iface": "$NIC_NODE1",
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
