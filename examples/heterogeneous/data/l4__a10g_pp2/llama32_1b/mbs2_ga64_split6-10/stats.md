# Single-run benchmark 결과

**Config**: mbs=2, ga=64, seq=1024, GBS=128 sequences = 131072 tokens / step

**Wall clock**: 총 315.7 s, train_steps=10

**fwd/bwd time** (Before → After train_batch_iter): [52.0, 23.0, 22.0, 23.0, 23.0, 22.0, 22.0, 22.0, 22.0, 23.0]
**step total** (Before → After training_step, optimizer 포함): [53.0, 23.0, 22.0, 23.0, 23.0, 22.0, 22.0, 22.0, 22.0, 23.0]
**optimizer + tied sync** (After train_batch_iter → After training_step): [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

- step 1 (warmup) total: 53.00 s
- steady-state (step 2..10) **total** 평균: **22.44 s** (fwd/bwd 22.44s + optimizer/tied 0.00s)


## DCGM 평균 (학습 active 구간만)

| 지표 | NODE 0 (L4) | NODE 1 (A10G) |
|---|---:|---:|
| avg power [W] | 45.7 | 179.1 |
| max power [W] | 57.8 | 222.2 |
| avg temp [°C] | 58.4 | 54.4 |
| max temp [°C] | 65.0 | 61.0 |
| avg SMACT | 0.211 | 0.534 |
| avg TENSO (BF16/FP16 matmul) | 0.087 | 0.166 |
| avg DRAMA (DRAM BW use) | 0.163 | 0.275 |

## NIC 실측 (`/proc/net/dev` 차분)

| 지표 | NODE 0 (enp39s0) | NODE 1 (ens5) |
|---|---:|---:|
| samples (≥1MB/s) | 1443 | 675 |
| avg TX [MB/s] | 100.9 | 148.1 |
| max TX [MB/s] | 1110.7 | 1174.8 |
| avg RX [MB/s] | 99.9 | 148.3 |
| max RX [MB/s] | 1110.6 | 1183.0 |

## 이론치 vs 실측

- 한 step 의 이론 cross-stage 전송 (forward + backward) 합: `2 × ga × mbs × seq × hidden × 2B = 2 × 64 × 2 × 1024 × 2048 × 2 = **1.074 GB / step**`
- ENA bandwidth — burst 10 Gbps 면 한 step 통신 `0.86 s` (24h 당 ~30분 한정), baseline 1.25 Gbps (sustained) 면 `6.87 s`. EFA 미지원 인스턴스라 NCCL Socket plugin (TCP) fallback.
- 이론 compute (per stage, 6N × tokens / sustained TFLOPs 추정): L4 측 `16.2 s` (30 TFLOPs 기준), A10G 측 `6.94 s` (70 TFLOPs 기준).
- 측정 steady step: `22.444 s` → throughput `5839.8 tokens/s`.
- 측정 평균 NIC bytes/step: `47.8 MB/s` (burst cap 1250 MB/s 대비 3.8%, baseline 156 MB/s 대비 30.6%).
- 측정과의 차이 (= NCCL P2P latency / pipeline bubble / kernel launch 등): `5.38 s` (burst comm 가정)