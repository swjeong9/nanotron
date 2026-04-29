# Heterogeneous PP=2 Benchmark 측정 방법

본 문서는 ``examples/heterogeneous/`` 의 학습 성능 측정 도구
(``benchmark_single.sh`` 등) 가 **무엇을 어떻게 측정하는지**, 그리고 그
측정이 **실제로 학습이 도는 동안 동작하는지** 확인할 수 있도록 정리한다.

## 0. ★ 다른 노드 sync 강제 정책 (필수)

**모든 multi-node 실행 entry-point 는 시작 직전에 NODE 1 와의 byte-identical
sync 를 강제한다**. 이전에 NODE 1 의 ``launch_pp2_node1.sh`` 가 옛날 버전이라
`config_path` arg 를 무시하고 OLD config 하드코딩 → NODE 0 만 partition
override 가 적용되어 NODE 1 은 항상 `[8, 8]` 로 학습 → partition mismatch
환경에서 sweep 결과가 모두 무의미해지는 사고가 있었다.

재발 방지를 위해:
1. ``benchmark_single.sh``, ``sweep_partitions.sh`` 가 시작 시 자동으로
   ``sync_to_node1.sh`` 를 호출.
2. ``sync_to_node1.sh`` 는 ``rsync -aq --delete`` 로 ``examples/heterogeneous/``
   와 ``src/nanotron/`` 트리를 NODE 1 에 push (NODE 1 의 stale 파일 모두 제거).
3. 핵심 파일 (``launch_pp2_node{0,1}.sh``, ``logging/base.py``, ``models/base.py``,
   ``trainer.py``, ``configs/llama32_1b/alpaca_pp2.yaml``) 의 md5sum 양 노드 비교,
   하나라도 mismatch 면 즉시 abort.
4. `partition` override 를 위해 sed 으로 config 수정한 후 NODE 1 에 다시 rsync 하고
   md5 verify 한 번 더.

수동 실험 시에도 multi-node 인 경우 반드시 ``sync_to_node1.sh`` 를 먼저 실행할 것.

## 1. 측정 대상

| 항목 | 출처 | 단위 / 의미 |
|---|---|---|
| GPU 전력 | DCGM field 155 (`POWER`) | W |
| GPU 온도 | DCGM field 150 (`TMPTR`) | °C |
| GPU SM 활성도 | DCGM field 1002 (`SMACT`) | 0..1 — 한 cycle 에 활성 warp 가 하나라도 있는 SM 의 비율 |
| Tensor core 활성도 | DCGM field 1004 (`TENSO`) | 0..1 — BF16/FP16 matmul 비율 |
| GPU DRAM 대역폭 사용률 | DCGM field 1005 (`DRAMA`) | 0..1 |
| PCIe TX/RX bytes (GPU↔CPU) | DCGM field 1009/1010 (`PCITX`/`PCIRX`) | bytes per 1Hz sample |
| NIC RX/TX bytes (노드 간) | `/proc/net/dev` 의 device 행 | cumulative bytes (delta 로 변환해서 사용) |
| 학습 step 시간 | nanotron stdout 의 `Before train_batch_iter` / `After train_batch_iter` 짝 | wallclock 초 |

DCGM 16 field 중 NVLink 관련 1011/1012 는 **L4 / A10G 모두 NVLink 부재** 라
제외 ([docs/dcgm_test_report.md](../../docs/dcgm_test_report.md) Stage 1 검증).

## 2. 측정 시간 윈도우 — sampler 가 학습을 cover 하는가

``benchmark_single.sh`` 의 흐름을 보면:

```
[t0]  pkill -9 → 이전 process 잔해 정리
[t1]  dcgmi dmon 양 노드 백그라운드 시작 (1 Hz)
[t2]  /proc/net/dev sampler 양 노드 백그라운드 시작 (2 Hz)
[t3]  sleep 2  → 백그라운드 process 가 안정화될 시간 확보
[t4]  ★ START_TS 기록 (학습 시작 wallclock)
[t5]  bash launch_pp2_node0.sh & (NODE 0 백그라운드 launch)
[t6]  ":29500" 가 listening 될 때까지 대기 (rendezvous)
[t7]  ssh node1 "bash launch_pp2_node1.sh"  (학습 종료까지 foreground)
[t8]  ★ END_TS 기록 (학습 종료 wallclock)
[t9]  pkill dcgmi dmon, NIC sampler
[t10] scp 으로 NODE 1 의 결과 파일 회수
```

즉:
- **DCGM/NIC 샘플러는 학습 시작 전 (t1, t2) 부터 학습 종료 후 (t9) 까지** 한
  번도 끊기지 않고 동작.
- 학습 코드 자체에는 instrumentation 추가 없음 → measurement 가 학습
  속도에 영향 없음.
- nanotron 이 매 step 마다 자동 출력하는 `Before/After train_batch_iter`
  로그 timestamp 만 사용 → **overhead 0**.

샘플 간격:
- DCGM: `dcgmi dmon -d 1000` → **1 sample/sec**
- NIC: bash `while :; ... sleep 0.5; done` → **2 sample/sec**
  (cumulative bytes 라 delta 로 변환)

학습이 30 sec 라면 DCGM 30 sample, NIC 60 sample 이 수집된다.

## 3. NIC sampling 의 정확한 명령

NIC sampler 는 `/proc/net/dev` 한 줄을 0.5 초마다 읽어 cumulative RX/TX
bytes 를 stdout 에 찍는 단순 bash 백그라운드:

```bash
# NODE 0 — 인터페이스 enp39s0
while :; do
    ts=$(date +%s.%N)
    line=$(awk -v iface="enp39s0" '$1 ~ iface":" { print $2, $3, $10, $11 }' /proc/net/dev)
    echo "$ts $line"   # ts RX_bytes RX_packets TX_bytes TX_packets
    sleep 0.5
done > nic_node0.txt
```

**왜 PCIe 외에 NIC 도 보는가**: DCGM `PCITX/PCIRX` 는 GPU↔CPU PCIe 트래픽
이라 NCCL TCP 의 NIC 트래픽의 _근사치_ 만 (PCIe 위에는 NCCL 외 weight load
같은 다른 traffic 도 섞임). 노드 간 정확한 bandwidth 는 NIC 직접 측정이
필요.

## 4. DCGM sampling 의 정확한 명령

```bash
dcgmi dmon -e 155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010 \
           -d 1000 \
           > dcgm_node0.txt
```

`-d 1000` = 1000 ms 간격 = **1 Hz**. text 출력은 [`dcgm_text_to_jsonl.awk`](dcgm_text_to_jsonl.awk)
로 JSONL 변환 후 plot 입력.

각 sample 의 timestamp 는 `ts = row_index` (정수 초) 로 부여한다 — DCGM dmon
text 자체에 timestamp 가 없고, awk 가 일괄 파싱이라 절대 시간 부여가
어렵기 때문. 1 Hz sampling 이라 row_index 가 곧 초 단위 상대시각.

## 5. 학습이 측정 윈도우 안에서 도는지 직접 확인하는 법

학습 stdout (예: `train_node0.log`) 에 step 별로 다음과 같이 출력:
```
04/29 05:08:53 [INFO|PP=0]: Before train_batch_iter
04/29 05:08:54 [INFO|PP=0]: After train_batch_iter
```
이 시각을 ``meta.json`` 의 `start_ts_utc` 와 비교하면 sampler 들이 학습
전체를 cover 했는지 확인 가능.

DCGM 시계열에서는 **활성 / 휴면 구간이 SMCLK 으로 잘 구분**된다:
- 학습 미시작 (idle): `SMCLK ≈ 210` (L4) / `300` (A10G)
- 학습 진행: `SMCLK ≈ 2040+` (boost clock)

분석 script 는 이 SMCLK 변화를 boundary 로 사용해 active/idle 구간을 분리한
뒤 평균 power/SMACT/TENSO 를 active 구간에서만 계산한다.

## 6. 측정 한계 / 주의사항

1. **DCGM `PCITX` 와 NIC TX 는 거의 같지만 동일하지 않음** — `PCITX` 에는
   NCCL 외 weight load (디스크→CPU→GPU), cuBLAS workspace 등이 섞임.
   노드 간 bandwidth 의 정확한 측정은 NIC `/proc/net/dev` 측을 우선.

2. **DCGM sampling rate 는 1 Hz 라 burst 트래픽을 놓침** — 100 ms 이내의
   peak bandwidth 는 보이지 않음. 평균치 위주 분석.

3. **활성 구간 boundary 가 SMCLK 휴리스틱** — A10G 의 idle clock 이 학습
   중 일시적으로 떨어지면 실수로 idle 로 분류될 가능성. 의심되면 raw
   data 로 검증.

4. **첫 iteration (step 1) 은 NCCL warm-up + cuBLAS context 초기화로
   30s 정도 더 걸림** — `steady_state_sec` 통계는 step 2..N 만 사용.

5. **NIC sampler 의 `sleep 0.5` 는 정확히 500 ms 가 아님** — bash + awk
   호출 overhead 로 실제로는 ~520–550 ms. delta 계산할 때 ts 차이로
   normalize 하면 영향 없음.

## 7. 산출물 schema (single-run benchmark 기준)

```
/opt/dlami/nvme/single_run/
├── meta.json         {ga, mbs, seq_len, gbs_seqs, start_ts_utc, end_ts_utc, elapsed_sec, ...}
├── dcgm_node0.txt    DCGM dmon raw text (NODE 0, L4)
├── dcgm_node1.txt    DCGM dmon raw text (NODE 1, A10G)
├── nic_node0.txt     /proc/net/dev sample (NODE 0, 한 줄당 "ts RX_bytes RX_packets TX_bytes TX_packets")
├── nic_node1.txt     동상 (NODE 1)
├── train_node0.log   nanotron stdout (NODE 0)
└── train_node1.log   nanotron stdout (NODE 1)
```

## 8. 1차 분석 sanity check 표

다음 표는 plot 결과를 볼 때 _맞게 측정됐는지_ 빠르게 가늠하는 기준이다.

| 검증 항목 | 기대값 (production-like config: mbs=2, ga=64, seq=1024) |
|---|---|
| 학습 wall-clock | step1 warmup ~30 s, steady step ~20-25 s |
| L4 평균 전력 | 30-60 W (idle 12W ~ TDP 72W) |
| A10G 평균 전력 | 150-200 W (idle 30W ~ TDP 300W; A10G 는 일반 A10 의 300W 변종) |
| L4 SMACT | < 0.2 (대부분 P2P 대기 — bottleneck 시) |
| A10G SMACT | < 0.4 (L4 보다 빠른 GPU 라 wait 비율 더 작음) |
| NIC TX peak (양 노드) | 100-1000 MB/s (ENA 1.25 GB/s cap 내) |
| 이론 대비 NIC 평균 | per iter total transfer ≈ 2 × ga × mbs × seq × hidden × 2B = 2 × 64 × 2 × 1024 × 2048 × 2 = 1.07 GB → step time 25s 면 평균 ~43 MB/s |
| GPU 온도 | thermal throttle 임계값 (~85°C) 미만 |

이 값과 크게 어긋나면 측정 / 환경 문제 의심.
