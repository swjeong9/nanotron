# Llama 3.2 3B Partition Sweep Plan (g6.12xl × g5.12xl)

## 목표

Cluster `g6_12xl__g5_12xl_pp2_tp4` (4× L4 + 4× A10G, PP=2 TP=4) 에서 Llama 3.2 3B
의 PP partition 비대칭 split 영향 측정. 1B sweep 의 결과 ([11, 5] 가 best,
monotonic toward L4-heavy) 가 더 큰 모델 + TP=4 환경에서도 동일한지 확인.

## 환경 (확정)

| 노드 | 인스턴스 | GPU | VRAM | NIC | 역할 (static backend) |
|---|---|---|---:|---|---|
| NODE 0 | g6.12xlarge (172.31.9.136) | 4× L4 | 24 GB | enp39s0 | stage 0 (rank 0..3) |
| NODE 1 | g5.12xlarge (172.31.13.113) | 4× A10G | 24 GB | ens5 | stage 1 (rank 4..7) |

- launch script: `--rdzv_backend=static` + `--master_addr/port` (deterministic mapping 확정)
- baseline 측정 (`[14, 14]`, mbs=2 seq=1024, ga=64): step ~40 sec, peak 11.4 GB / 12.0 GB
  (L4 / A10G), throughput ~ 6,400 tokens/s (cluster total) 추정

## Hyperparameter (sweep 시 고정)

| 항목 | 값 |
|---|---:|
| sequence length | 1024 |
| micro-batch size | 2 |
| gradient accumulation | 64 |
| GBS (sequences) | 128 |
| tokens per step | 131 K |
| recompute_layer | false |
| TP | 4 |
| PP | 2 |
| DP | 1 |

(ga ≥ pp+1 충족 — 1F1B bubble 1.5%)

## Partition 공간

`pp_layer_partition: [a, 28-a]` for `a ∈ {1, 2, ..., 27}` — 총 **27 점**.

메모리 추정 (per GPU TP=4, mbs=2 seq=1024):
- per layer per mb activation = `2048 × 2 × 3072 × 16 / TP4` ... (TP-shard)
- per layer state = ~218M / TP4 = ~55M params × 14 bytes = ~770 MB / GPU
- stage k 의 in-flight mbs (1F1B PP=2) = 2 (k=0) / 1 (k=1)
- 즉:
  - L4 (stage 0) total = `n × 0.77 GB (state) + 2 × n × ~100 MB (activation) + 3 GB (overhead)` ≈ `0.97n + 3 GB`
  - A10G (stage 1) total = `(28-n) × 0.77 + 1 × (28-n) × 0.1 + 1 GB (lm_head logits) + 3 GB` ≈ `0.87(28-n) + 4 GB`

| `a` (L4) | L4 추정 | A10G 추정 | OOM? |
|---:|---:|---:|---|
| 1 | 4 GB | 27.6 GB | A10G OOM ❌ |
| 5 | 7.9 | 24.0 | A10G OOM ❌ tight |
| 10 | 12.7 | 19.7 | ✓ |
| 14 | 16.6 | 16.2 | ✓ (실측 11.4 / 12.0) |
| 18 | 20.5 | 12.7 | ✓ |
| 22 | 24.3 | 9.2 | L4 OOM ❌ tight |
| 27 | 29.2 | 4.9 | L4 OOM ❌ |

**예상 fit range**: `a ∈ {7, ..., 21}` (15 점). 양 끝 OOM 명확하면 sweep 가속 가능.

(실측 baseline [14, 14] L4 11.4 GB 면 모델대로 12.6 + overhead 3.8 GB — 추정 16.6 GB 보다 5 GB 작음 → TP=4 의 activation 분산이 추정보다 더 효과적. 실제 OOM 한계는 더 양 끝까지 fit 가능성)

## Sweep 절차

```bash
# dev 에서, 양 노드 sync + sweep 실행
bash examples/heterogeneous/clusters/g6_12xl__g5_12xl_pp2_tp4/sync.sh

# 균형 ± 10 (예상 fit range)
START_STAGE0=4 END_STAGE0=24 \
  bash examples/heterogeneous/clusters/g6_12xl__g5_12xl_pp2_tp4/sweep_partitions.sh

# 또는 전체
bash examples/heterogeneous/clusters/g6_12xl__g5_12xl_pp2_tp4/sweep_partitions.sh
```

각 iteration 의 raw data → `/opt/dlami/nvme/runs/g6_12xl__g5_12xl_pp2_tp4/llama32_3b/<descriptor>/`,
plot 결과 → `examples/heterogeneous/data/...` + `figures/...`.

Per iteration 시간 ~13 분. fit 27 점 다 돌리면 ~6 시간. fit-only 15 점 ~3.5 시간.

## 분석 포인트 (1B sweep 와 비교)

1. **monotonic toward L4-heavy?** — 1B 에서 [11, 5] 가 best 였음. 3B 에서도 패턴 유지?
   - 가설 A: 유지 (lm_head 가 stage 1 (A10G) 에 있어 A10G 이미 부담 큼)
   - 가설 B: 변화 (3B 에서 lm_head 비중 ~2.4 layer-eq 로 1B 의 4.3 layer-eq 보다 작음 → 비대칭 약함)

2. **TP=4 가 stage time 에 미치는 영향** — TP=1 1B 와 비교해 communication overhead 부담:
   - intra-node TP all-reduce (PCIe Gen4) 가 추가 시간
   - inter-node PP P2P 는 TP=4 라 mb 당 4 GPU 모두 보내고 받음 (data 양은 동일하지만 NCCL 라운드)

3. **1F1B bubble 4.5%** (ga=64, pp=2) — 1B 의 ga=64 와 동일

4. **per-GPU MFU** — A100/L40S 비교 분석 시 baseline 으로 사용. L4 / A10G 의 raw spec 저하 정확히 매기기

## 다음 단계 (sweep 후)

1. plot_compare.py 의 `--cluster g6_12xl__g5_12xl_pp2_tp4 --model llama32_3b` 으로 비교 그래프
2. partition_compare.png 분석 → motivation 의 위치 (cost-throughput frontier 강조)
3. **8B target 환경** (p4d.24xlarge + g6e.12xlarge × 2) 의 setup — clusters/p4d__g6e_12xl_pp4_tp4/ 신설
   - PROJECT_BACKGROUND.md §5 의 canonical setup
   - L40S spec 정정 반영 (BF16 raw 더 빠름)
4. 8B Llama 3.1 의 nanotron checkpoint 변환 + S3 업로드 (이미 plan 있음)

## 미결정 / Open

- 8B target 환경의 EFA 활성화 여부 (ENA / EFA 선택)
- 3B sweep 시 power monitoring (DCGM) 의 stage 별 평균 vs 즉시 단일 step 동안의 dynamic
- monotonic 패턴 시 cost-throughput frontier 측정으로 motivation reframe — 인스턴스 가격 데이터 수집
