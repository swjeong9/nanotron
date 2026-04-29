# Heterogeneous Distributed Training Motivation — 8B target plan

본 문서는 PROJECT_BACKGROUND.md §5 의 이기종 클러스터 학습 motivation 검증을
위한 production-like 학습 setup 의 결정 사항을 정리한다. 현재 진행 중인
**1B sweep** ([1, 15] ~ [15, 1] partition 비교) 결과 분석이 먼저, 그 후 본 8B
setup 으로 이동.

---

## 1. 클러스터 구성

| 노드 | 타입 | GPU | 수 | VRAM | 비고 |
|---|---|---|---:|---:|---|
| stage 0/1 hosts | p4d.24xlarge × 1 | A100 | 8 | 40 GB | NVLink intra-node, p4 EFA 가능 |
| stage 2/3 hosts | g6e.12xlarge × 2 | L40S | 8 (4 × 2) | 48 GB | PCIe intra-node, EFA 미확정 |
| **합계** | | | **16** | | dev 노드 (orchestration) 별도 |

- 노드 간 inter-connect: TCP socket (NCCL Socket plugin) 또는 EFA (지원시).
  현재 사고 시점 기준 EFA 채택 여부 미정 — 일단 TCP 가정.

## 2. Parallelism 결정

| 차원 | 값 | 근거 |
|---|---:|---|
| PP | 4 | 32 layer 를 4 stage 에 분산. stage 당 8 layer balanced 시 [8, 8, 8, 8] |
| TP | 4 | A100 NVLink 와 L40S PCIe 모두 4-way intra-node 가능 |
| DP | 1 | 16 GPU = 4 × 4 × 1. 추가 DP 가 필요하면 클러스터 확장 필요 |
| **world** | **16** | |

PP partition 의 비대칭성 ([n0, n1, n2, n3] with sum=32) 이 본 연구의 주요
변수. 균등 [8, 8, 8, 8] 부터 시작해 A100 stage 더 많이 / L40S stage 더 많이
양 방향으로 sweep.

## 3. 학습 hyperparameter

| 항목 | 값 | 비고 |
|---|---:|---|
| sequence length | **2048** | 단축 — production Llama 3.1 8B 의 8192 와 다름. heterogeneous training motivation 검증 만 목표라 단축 OK |
| micro-batch size (mbs) | **4** | s × b = 8192 의 fit-able product. kernel utilization 적정 |
| gradient accumulation (ga) | **64** | 1F1B bubble 4.5% (= 3/67), pp_size=4 의 5배 충분 |
| global batch size (GBS) | **256** | mbs × ga × dp = 4 × 64 × 1 (sequences) |
| tokens per step | **525K** | GBS × seq = 256 × 2048 |
| recompute strategy | **selective via flash_attn** (built-in) | nanotron 의 `recompute_layer: true` 는 깨져 있음 (§7-3). flash_attn 의 attention 자동 recompute 로 selective 와 같은 메모리 |

**결정 근거**:
- Memory: per layer per mb = `s · b · h · 16` (selective + TP=4) = 2048 × 4 × 4096 × 16 = **0.54 GB / GPU**
  - s × b = 8192 가 critical product. 같은 메모리로 가능한 다른 조합:
    `(s, b) ∈ {(1024, 8), (2048, 4), (4096, 2), (8192, 1)}`
- 위 중 mbs=4 가 kernel utilization 과 PP bubble 균형. mbs=1 은 kernel 비효율, mbs=8 은 seq=1024 라 학습 의미 약함.
- seq=2048 은 production Llama 3.1 8B 의 8192 보다 짧음. context length 측면 손해 있지만 본 motivation 의 목적은 heterogeneous parallelism 검증이라 OK.
- **production-equivalent setup 으로 가려면**: nanotron 의 recompute path 디버깅 또는 PP/TP 토폴로지 확장 필요 (Open Question).

## 4. 메모리 모델 (per-GPU)

Selective recompute via flash_attn (no SP), seq=2048 mbs=4:
- per layer per microbatch activation = `s · b · h · (10 + 24/t)` = `s·b·h × 16` bytes per GPU
- 2048 × 4 × 4096 × 16 = **0.54 GB / GPU**
- stage 0 (8 layers × 4 in-flight mbs in 1F1B PP=4) 가 가장 빡빡

| 항 | 값 |
|---|---:|
| State (BF16 weight + FP32 grad acc + AdamW m/v, per stage 8 dec + embed, ÷ TP=4) | ~8 GB |
| Activation (stage 0): 8 × 4 × 0.54 GB | ~17 GB |
| BF16 grad temp (during backward, 568M × 2) | ~1 GB |
| Logits intermediate (mbs × seq × vocab × 4 bytes fp32, stage 3 만): 4 × 2048 × 128256 × 4 | ~4 GB (stage 3 만) |
| Overhead (PyTorch cache + NCCL buffers + intermediate) | ~3 GB |
| **Per-GPU total** (stage 0, A100 binding) | **~29 GB / 40 GB** |
| **Per-GPU total** (stage 3 with lm_head, L40S 48GB) | ~30 GB / 48 GB |

정상 fit. stage 1, 2 는 in-flight mb 수 적어 (3, 2) 메모리 더 여유 (~25-27 GB).

**Margin**:
- A100 stage 0: 40 - 29 = ~11 GB headroom (NCCL/cache 변동 흡수)
- L40S stage 3: 48 - 30 = ~18 GB headroom

## 5. 의도적으로 안 쓰는 것

- **Sequence Parallel (SP)**: 본 motivation 검증의 추가 변수 줄이기 위해 미사용.
  켜면 selective-equivalent 메모리 절감 가능하지만 본 실험의 주제 외.
- **Selective recompute**: nanotron 미지원. 어차피 full 로 충분히 fit.
- **DP > 1**: 클러스터 확장 시 추가 검토. 현재는 16 GPU 환경.

## 6. 측정 지표

기존 1B sweep 의 측정 도구를 그대로 사용:
- 양 노드 (전체 ≥ 2 노드) `dcgmi dmon` + `/proc/net/dev` + `nvidia-smi memory.used` 1Hz polling
- nanotron stdout 의 `Before/After train_batch_iter` + `After training_step` timestamp
- `[ModelFLOPs]` / `[StageFLOPs]` log → per-stage compute cost 자동 추출
- per-rank `log_memory` (`Memory usage / Peak allocated / Peak reserved`)

집계 metric:
- step time (steady-state 평균, step 1 warmup 제외)
- throughput (tokens/sec, cluster total)
- per-stage MFU (achieved TFLOPS / peak BF16 TC)
- per-GPU 평균 power
- per-GPU peak memory (PyTorch reserved + nvidia-smi max)
- NIC bandwidth (avg / peak), PCIe traffic (DCGM PCITX/PCIRX)

## 7. 진행 순서

1. **(현재 진행 중)** 1B sweep [1, 15] ~ [15, 1] 완료 — partition 별 throughput / MFU /
   memory 패턴 분석. 이게 "어떤 partition 이 어떤 효과를 내는지" 의 baseline.
2. 1B sweep 결과 정리:
   - 파일: `examples/heterogeneous/data/l4__a10g_pp2/llama32_1b/<descriptor>/stats.json`
   - cross-partition 비교 plot (figures/<cluster>/<model>/comparison/) — 향후 추가
   - 결론: PP=2 cross-VPC TCP 환경에서의 partition 의 효과
3. **★ Recompute 동작 검증** — 본 plan 의 메모리 추정 (~27 GB / GPU on stage 0) 은
   `parallelism.recompute_layer: true` 가 nanotron 에서 정상 작동한다는 가정에 의존.

   **★ 2026-04-29 검증 결과: nanotron 의 `recompute_layer: true` 가 우리
   setup 에서 작동 안 함**. [8, 8] no-recompute (L4 17.4 GB / A10G 18.1 GB nvsmi)
   vs recompute=true (L4 22.5 GB / A10G 21.9 GB nvsmi) — **메모리 더 사용**, OOM.
   forward 의 sharded_cross_entropy 에서 OOM (1F1B 의 mb 들 사이 활성화 누적 + lm_head
   logits intermediate). 가능 원인: CheckpointFunction 이 1F1B engine 의 mb 큐와
   상호작용 / `_use_doc_masking: true` (variable seq) interaction / TensorPointer
   분기 처리.

   → **8B target 의 메모리 모델 신뢰 불가**. 다음 중 하나 필요:

   a. nanotron 의 `Qwen2DecoderLayer.forward` 의 recompute path 디버깅 (메모리
      profiler 로 어디서 누적되는지 확인). HF upstream PR 가능성.
   b. nanotron PR — selective recompute 추가 (attention 만 recompute) — 메모리/
      compute 균형
   c. 본 motivation 검증을 위해서는 mbs/seq 줄이거나 더 큰 cluster 필요.
      예: seq=2048 + selective recompute 흉내 (or seq=4096 with mbs=4 — no-recompute
      에서 fit)
   d. (대안) Megatron-LM 등 다른 framework 로 8B 학습 — recompute 가 검증된 path

4. **clusters/ 디렉터리 reorg** — 위에서 합의된 구조 (clusters/{l4_a10g, a100_8__l40s_8, ...})
   로 이동. 1B sweep 결과는 이미 `data/l4__a10g_pp2/...` 에 있어 영향 없음.
5. **8B 클러스터 setup** — `clusters/a100_8__l40s_8/` 새로 만들고:
   - launch script (multi-node, multi-GPU per node, intra-node TP=4)
   - sync script (3 노드 모두)
   - benchmark script (DCGM/NIC/nvidia-smi sampling 양쪽 노드)
6. **8B baseline run** — partition [8, 8, 8, 8] 로 정상 학습 + 본 plan 의 메모리 model 검증
7. **8B sweep** — stage 별 layer 분배 [a, b, c, d] (sum=32, all ≥ 1) 의 부분 공간 sweep.
   완전 sweep 은 35,960 점이라 불가능 — 의미 있는 부분만 (e.g. balanced ± 2 layer 변형,
   A100 쪽 / L40S 쪽으로 무게 이동)

## 8. 미결정 / Open Questions

- **EFA 가능 여부**: g6e.12xlarge 가 EFA 지원하면 TCP 보다 NCCL 성능 ↑.
  L40S 가 PCIe 라 RDMA 가능한지 확인 필요. p4d 는 EFA 표준.
- **g6e.12xlarge 의 정확한 GPU spec**: L40S 는 TF32 91 TFLOPS / BF16 91 TFLOPS dense.
  실측 sustained 는 공식 datasheet 와 다를 수 있어 1차 run 에서 calibration 필요.
- **Cluster 노드 수**: 2 vs 3 vs 4. 위 표는 1 + 2 = 3 노드. 토폴로지에 따라 stage 배치
  유연성 변화.
- **dev 노드 분리 여부**: 현재 sweep 은 dev = stage 0 노드. 8B 환경에서는 별도 dev
  노드 (CPU instance) 권장 — orchestration 만 담당.
