#!/usr/bin/env bash
# ga (batch_accumulation_per_replica) sweep with DCGM 시계열 수집.
# NODE 0 (this host) 에서 실행. NODE 1 은 ssh 로 trigger.
#
# 각 ga 값 마다:
#   1. config 의 ga 만 sed 로 in-place 변경 + rsync NODE 1
#   2. DCGM dmon 양 노드 background (text → JSONL via awk)
#   3. 학습 launch (5 step) → 종료까지 wait
#   4. DCGM 정리, 결과 파일 보존
#
# 결과: /opt/dlami/nvme/ga_sweep/ga<N>/ 아래 (per-config 서브디렉토리)
#   train_node{0,1}.log    : 학습 stdout
#   dcgm_node{0,1}.txt     : DCGM dmon raw text
#   meta.json              : 시작/종료 timestamp 등 메타
#
# Plot/figures 는 본 script 가 만들지 않음. sweep 후 ``plot_ga_sweep.py`` 로
# ``examples/heterogeneous/figures/ga_sweep/`` 에 생성 (gitignore).

set -euo pipefail

NANOTRON=/home/ubuntu/nanotron
NODE1_IP=172.31.40.226
OUT_DIR=/opt/dlami/nvme/ga_sweep
CONFIG=$NANOTRON/examples/heterogeneous/config_llama32_1b_alpaca_pp2.yaml

# NVLink 부재 (L4 / A10G 모두) 이므로 1011/1012 제외한 16 field.
# (project_background.md §6.4 + dcgm_test_report.md 검증 기준)
FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010"

GA_VALUES="${1:-4 8 16 32 64}"

mkdir -p "$OUT_DIR"

# Cache 재사용: ga 만 바뀌고 dataset/processing 은 동일 → cache hit 으로 매 run 마다
# packing 다시 안 함. 첫 run 만 packing, 이후 ~5초 안에 dataset load.
sed -i 's/^      dataset_overwrite_cache: .*/      dataset_overwrite_cache: false/' "$CONFIG"

cleanup_dcgm() {
    pkill -f "dcgmi dmon" 2>/dev/null || true
    ssh -o BatchMode=yes ubuntu@$NODE1_IP "pkill -f 'dcgmi dmon'" 2>/dev/null || true
}

cleanup_train() {
    pkill -9 -f run_train 2>/dev/null || true
    pkill -9 -f torchrun 2>/dev/null || true
    ssh -o BatchMode=yes ubuntu@$NODE1_IP "pkill -9 -f run_train; pkill -9 -f torchrun" 2>/dev/null || true
}

for ga in $GA_VALUES; do
    echo "=========================================="
    echo "=== ga=$ga ==="
    echo "=========================================="

    # 1) config 업데이트 (ga 만)
    sed -i "s/^  batch_accumulation_per_replica: .*/  batch_accumulation_per_replica: $ga/" "$CONFIG"
    rsync -aq "$CONFIG" "ubuntu@$NODE1_IP:$CONFIG"

    # 2) 이전 학습 / DCGM 잔여 정리
    cleanup_train
    cleanup_dcgm
    sleep 1
    rm -f /opt/dlami/nvme/pp2_node{0,1}.log
    ssh -o BatchMode=yes ubuntu@$NODE1_IP "rm -f /opt/dlami/nvme/pp2_node1.log" 2>/dev/null || true

    # 3) DCGM dmon 백그라운드 (text 출력, 1초 간격)
    nohup dcgmi dmon -e "$FIELDS" -d 1000 \
        > "$OUT_DIR/dcgm_node0_ga${ga}.txt" 2>&1 &
    DCGM_PID_NODE0=$!
    ssh -o BatchMode=yes ubuntu@$NODE1_IP \
        "nohup dcgmi dmon -e \"$FIELDS\" -d 1000 \
         > /opt/dlami/nvme/dcgm_node1_ga${ga}.txt 2>&1 &" || true
    sleep 2  # DCGM 안정화

    # 4) 학습 시작 시각 기록 (DCGM 와 join 용)
    START_TS=$(date -u +%s.%N)

    # 5) NODE 0 launch (background)
    bash "$NANOTRON/examples/heterogeneous/launch_pp2_node0.sh" \
        > "$OUT_DIR/train_node0_ga${ga}.log" 2>&1 &
    NODE0_PID=$!

    # NODE 0 listen 대기
    while ! ss -tnlp 2>/dev/null | grep -q ":29500 "; do sleep 1; done

    # 6) NODE 1 launch (foreground via ssh — block 까지). Timeout 안전장치:
    # ga=64 가 hang 가능성 있어 12분 timeout. 정상이면 6분 안에 끝.
    timeout 720 ssh -o BatchMode=yes ubuntu@$NODE1_IP \
        "bash /home/ubuntu/nanotron/examples/heterogeneous/launch_pp2_node1.sh \
         > /opt/dlami/nvme/train_node1_ga${ga}.log 2>&1" || NODE1_RC=$?

    # 7) NODE 0 종료 대기 (최대 30 sec — NODE 1 이 끝났으면 NODE 0 도 자동 종료)
    SECONDS=0
    while kill -0 "$NODE0_PID" 2>/dev/null && [ $SECONDS -lt 30 ]; do sleep 1; done
    if kill -0 "$NODE0_PID" 2>/dev/null; then
        kill -9 "$NODE0_PID" 2>/dev/null || true
        NODE0_RC=124
    else
        wait "$NODE0_PID" 2>/dev/null || NODE0_RC=$?
    fi

    END_TS=$(date -u +%s.%N)

    # 8) DCGM 정리
    cleanup_dcgm
    sleep 1

    # 9) 결과 파일 회수
    scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/dcgm_node1_ga${ga}.txt" \
        "$OUT_DIR/dcgm_node1_ga${ga}.txt" 2>/dev/null || true
    scp -q "ubuntu@$NODE1_IP:/opt/dlami/nvme/train_node1_ga${ga}.log" \
        "$OUT_DIR/train_node1_ga${ga}.log" 2>/dev/null || true

    # 10) 메타 저장
    cat > "$OUT_DIR/meta_ga${ga}.json" <<EOF
{
  "ga": $ga,
  "start_ts_utc": "$START_TS",
  "end_ts_utc": "$END_TS",
  "elapsed_sec": $(awk "BEGIN { print $END_TS - $START_TS }"),
  "node0_rc": ${NODE0_RC:-0},
  "node1_rc": ${NODE1_RC:-0}
}
EOF
    echo "ga=$ga done in $(awk "BEGIN { print $END_TS - $START_TS }") sec"
    NODE0_RC=0
    NODE1_RC=0
done

echo "=========================================="
echo "Sweep complete. Results in $OUT_DIR"
ls -la "$OUT_DIR"
