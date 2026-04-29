# Single-run benchmark 결과

**Config**: mbs=2, ga=64, seq=1024, GBS=128 sequences = 131072 tokens / step

**Wall clock**: 총 313.3 s, train_steps=10

**fwd/bwd time** (Before → After train_batch_iter): [52.0, 22.0, 22.0, 22.0, 22.0, 22.0, 23.0, 22.0, 22.0, 22.0]
**step total** (Before → After training_step, optimizer 포함): [54.0, 22.0, 23.0, 22.0, 22.0, 22.0, 23.0, 22.0, 22.0, 22.0]
**optimizer + tied sync** (After train_batch_iter → After training_step): [2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

- step 1 (warmup) total: 54.00 s
- steady-state (step 2..10) **total** 평균: **22.22 s** (fwd/bwd 22.11s + optimizer/tied 0.11s)


## DCGM 평균 (학습 active 구간만)

| 지표 | NODE 0 (L4) | NODE 1 (A10G) |
|---|---:|---:|
| avg power [W] | 47.2 | 172.8 |
| max power [W] | 59.7 | 214.7 |
| avg temp [°C] | 60.6 | 53.4 |
| max temp [°C] | 68.0 | 60.0 |
| avg SMACT | 0.244 | 0.502 |
| avg TENSO (BF16/FP16 matmul) | 0.104 | 0.156 |
| avg DRAMA (DRAM BW use) | 0.181 | 0.258 |

## NIC 실측 (`/proc/net/dev` 차분)

| 지표 | NODE 0 (enp39s0) | NODE 1 (ens5) |
|---|---:|---:|
| samples (≥1MB/s) | 1443 | 300 |
| avg TX [MB/s] | 101.2 | 168.4 |
| max TX [MB/s] | 1097.7 | 1129.8 |
| avg RX [MB/s] | 100.2 | 162.0 |
| max RX [MB/s] | 1082.0 | 1107.4 |

## 이론치 vs 실측

- 한 step 의 이론 cross-stage 전송 (forward + backward) 합: `2 × ga × mbs × seq × hidden × 2B = 2 × 64 × 2 × 1024 × 2048 × 2 = **1.074 GB / step**`
- ENA bandwidth — burst 10 Gbps 면 한 step 통신 `0.86 s` (24h 당 ~30분 한정), baseline 1.25 Gbps (sustained) 면 `6.87 s`. EFA 미지원 인스턴스라 NCCL Socket plugin (TCP) fallback.
- 이론 compute (per stage, 6N × tokens / sustained TFLOPs 추정): L4 측 `16.2 s` (30 TFLOPs 기준), A10G 측 `6.94 s` (70 TFLOPs 기준).
- 측정 steady step: `22.222 s` → throughput `5898.2 tokens/s`.
- 측정 평균 NIC bytes/step: `48.3 MB/s` (burst cap 1250 MB/s 대비 3.9%, baseline 156 MB/s 대비 30.9%).
- 측정과의 차이 (= NCCL P2P latency / pipeline bubble / kernel launch 등): `5.16 s` (burst comm 가정)