# Single-run benchmark 결과

**Config**: mbs=2, ga=64, seq=1024, GBS=128 sequences = 131072 tokens / step

**Wall clock**: 총 312.7 s, train_steps=10

**fwd/bwd time** (Before → After train_batch_iter): [54.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0]
**step total** (Before → After training_step, optimizer 포함): [55.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0]
**optimizer + tied sync** (After train_batch_iter → After training_step): [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

- step 1 (warmup) total: 55.00 s
- steady-state (step 2..10) **total** 평균: **22.00 s** (fwd/bwd 22.00s + optimizer/tied 0.00s)


## DCGM 평균 (학습 active 구간만)

| 지표 | NODE 0 (L4) | NODE 1 (A10G) |
|---|---:|---:|
| avg power [W] | 47.9 | 167.9 |
| max power [W] | 63.7 | 209.4 |
| avg temp [°C] | 62.2 | 53.9 |
| max temp [°C] | 69.0 | 60.0 |
| avg SMACT | 0.274 | 0.470 |
| avg TENSO (BF16/FP16 matmul) | 0.122 | 0.146 |
| avg DRAMA (DRAM BW use) | 0.192 | 0.241 |

## NIC 실측 (`/proc/net/dev` 차분)

| 지표 | NODE 0 (enp39s0) | NODE 1 (ens5) |
|---|---:|---:|
| samples (≥1MB/s) | 1449 | 291 |
| avg TX [MB/s] | 100.8 | 166.5 |
| max TX [MB/s] | 1078.5 | 1121.4 |
| avg RX [MB/s] | 99.7 | 158.6 |
| max RX [MB/s] | 1059.6 | 1139.1 |

## 이론치 vs 실측

- 한 step 의 이론 cross-stage 전송 (forward + backward) 합: `2 × ga × mbs × seq × hidden × 2B = 2 × 64 × 2 × 1024 × 2048 × 2 = **1.074 GB / step**`
- ENA bandwidth — burst 10 Gbps 면 한 step 통신 `0.86 s` (24h 당 ~30분 한정), baseline 1.25 Gbps (sustained) 면 `6.87 s`. EFA 미지원 인스턴스라 NCCL Socket plugin (TCP) fallback.
- 이론 compute (per stage, 6N × tokens / sustained TFLOPs 추정): L4 측 `16.2 s` (30 TFLOPs 기준), A10G 측 `6.94 s` (70 TFLOPs 기준).
- 측정 steady step: `22.0 s` → throughput `5957.8 tokens/s`.
- 측정 평균 NIC bytes/step: `48.8 MB/s` (burst cap 1250 MB/s 대비 3.9%, baseline 156 MB/s 대비 31.2%).
- 측정과의 차이 (= NCCL P2P latency / pipeline bubble / kernel launch 등): `4.94 s` (burst comm 가정)