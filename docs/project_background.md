# Heterogeneous GPU LLM Fine-tuning Project — Plan

## 1. 프로젝트 목표

이기종 GPU 클러스터(A100 + L40S)에서 LLM distributed training을 수행할 때, **asymmetric parallelism** (PP stage별 다른 TP 차수, 비대칭 layer split)을 통해 throughput / cost / power consumption을 최적화하는 것이 목표.

핵심 motivation:
- Homogeneous GPU 클러스터는 성능은 좋지만 **availability 부족** (A100 16장을 일관되게 확보하기 어려움)
- 가용한 모든 GPU를 포괄적으로 사용해야 하는 상황에서, 이기종 환경의 비대칭성을 어떻게 다룰지가 핵심 문제
- 잘못 partitioning된 모델은 GPU **underutilization** 이 발생하면서도 power는 거의 full로 소비됨 → 에너지 효율성 저하

---

## 2. 하드웨어 환경

| 항목 | 사양 |
|---|---|
| Stage 0 (compute-heavy) | NVIDIA A100 **40GB** × 8 (단일 노드, NVLink mesh) |
| Stage 1 (compute-aux) | NVIDIA L40S 48GB × 8 (PCIe Gen4 only, NVLink 없음) |
| 총 GPU | 16장 |
| 총 VRAM | 320 + 384 = 704 GB |
| 노드 간 통신 | (TBD: IB / RoCE / Ethernet 확인 필요) |

특기사항:
- **A100은 40GB 버전**. 80GB가 아님.
- L40S는 BF16 dense throughput에서 A100보다 16% 빠르나, memory bandwidth 0.43×, NVLink 부재로 inter-GPU 통신은 PCIe로 강제됨
- 두 GPU 모두 FlashAttention v2까지만 지원 (FA3는 Hopper 전용)

---

## 3. Software Stack

### 3.1 Framework

**Hugging Face nanotron fork** 를 베이스로 사용.

선정 이유:
- DeepSpeed/Megatron-LM/NeMo/ColossalAI 모두 PP stage별 다른 TP 차수를 native로 지원하지 않음
- Megatron-Core fork는 작업량이 너무 큼 (의존성, Transformer Engine, mcore 등)
- nanotron은 코드베이스가 작고 (`ParallelContext`, `PipelineBlock`, `ProcessGroup` 추상이 명시적), hackable함
- 1F1B, AFAB, ZeRO-1, FP32 grad accumulation 등 필요한 학습 인프라가 갖춰져 있음

**Fork 전략**: HF nanotron `main`을 그대로 사용 (fork: `swjeong9/nanotron@main`, baseline commit `21b355e`).
- v0.5(`4799d24`)는 Llama 2 시대 (~2024 중반) 기준이라 `LlamaConfig`에 Llama 3.1+의 `rope_scaling`, 큰 vocab(128k), GQA 변형이 미흡할 가능성. main은 이를 모두 지원함이 확인됨 (§3.2)
- main에는 v0.5 이후 `llama3_ring_attention.py`(long-context), Qwen, MoE, SmolLM3 (commit `7bc9923`) 등이 추가되어 본 연구 baseline에도 도움
- 변환 script(`examples/llama/convert_hf_to_nanotron.py`)는 example일 뿐이라 main / v0.5 모두 별도 sanity check가 필요한 건 동일
- 단점: HF main은 unfinished work가 섞일 수 있음 → 본 작업은 별도 branch에서 진행하고 main은 upstream 추적용으로만

### 3.2 Llama 3.x 지원 — Architecture는 OK, 변환 검증 + S3 저장 전략

**Architecture 지원 (확인됨)**: main의 `src/nanotron/config/models_config.py`의 `LlamaConfig` 가 `rope_scaling: Optional[dict]`, `rope_theta: float`, `num_key_value_heads`(GQA), `vocab_size`(default 32000이지만 override 가능)을 모두 보유. `src/nanotron/nn/llama3_ring_attention.py`도 추가됨. 즉 Llama 3.1 8B / 3.2 1B의 architecture 자체는 main에서 지원.

**변환 sanity check 필요** (Phase 1):
- `examples/llama/convert_hf_to_nanotron.py`는 example이라 직접 수정 가능. 표면 grep 기준 `rope_scaling` 키워드 매핑이 명시적이지 않아 직접 검증 필요.
- HF → nanotron 변환 무결성 (forward 출력 1e-3 tolerance 내 일치)
- RoPE scaling 파라미터 매핑 (Llama 3.1)
- vocab 128k tokenizer + tied embedding (Llama 3.2)

**변환된 checkpoint의 저장 전략**:
- 변환 비용(8B는 ~15 GB, 1B는 ~2.5 GB)을 매번 재실행하지 않도록 S3에 저장:
  - **Bucket: `s3://swj-nanotron-model/`**
  - 경로 규약 (예시): `s3://swj-nanotron-model/llama-3.1-8b/nanotron/`, `s3://swj-nanotron-model/llama-3.2-1b/nanotron/`
- nanotron의 `s3` extra (`pip install -e ".[s3]"` → `boto3`, `s3fs`, `s5cmd`) 또는 [examples/config_tiny_llama_with_s3_upload.yaml](../examples/config_tiny_llama_with_s3_upload.yaml) 패턴 활용
- Stage 1 검증에서 한 번만 변환 → S3 업로드. Stage 2 / 본 클러스터에서는 S3에서 직접 download (재변환 없음)

만약 변환이 끝까지 막히면 Llama 2 13B 또는 Qwen 2.5 14B로 변경 검토 (PROJECT_BACKGROUND fallback).

### 3.3 필요한 수정 (Asymmetric Parallelism 구현)

| 작업 | 예상 |
|---|---|
| `ParallelContext`에 stage별 TP 리스트 (`tensor_parallel_sizes_per_stage`) | 반나절 |
| `PipelineBlock` device 매핑 stage→GPU 일반화 | 1일 |
| Stage 경계 P2P shape 재계산 (TP=8 → TP=4 시 receiver 측 all-gather) | 1일 |
| 비대칭 layer split (PP=2, 19/13) | 반나절 |
| Sanity check: 균등 split vs 비대칭 split loss 정합성 | 1~2일 |

**예상 작업량: 약 3~5일** (DP=1 고정으로 layer-wise grad AllReduce 등 복잡한 부분 제외)

부수 작업 (Llama 3.x conversion 검증, FA2 통합, Liger Kernel patch 등)을 포함하면 1주 정도. 비대칭 PP 구현 자체와는 분리해서 산정.

### 3.4 보조 라이브러리

| 라이브러리 | 용도 |
|---|---|
| FlashAttention 2 | attention 가속. 공식 prebuilt wheel이 sm_80 (A100) + sm_89 (L40S/L4) 모두 포함, `pip install flash-attn --no-build-isolation`로 즉시 설치 |
| Liger Kernel | RMSNorm, RoPE, SwiGLU, fused linear+CE — 메모리 ~60% 절감, throughput +20% |
| nanotron `s3` extra (`boto3`, `s3fs`, `s5cmd`) | 변환된 checkpoint를 `s3://swj-nanotron-model/` 에 저장/load. `pip install -e ".[s3]"` 로 설치. nanotron 자체에 `examples/config_tiny_llama_with_s3_upload.yaml` 등 S3 통합 예시 존재 |
| DCGM | GPU power / utilization 모니터링 (Stage 1 검증 결과 [docs/dcgm_test_report.md](dcgm_test_report.md)) |
| IPMI / PDU | host 전력 측정 (figure에 포함 권장) |

---

## 4. 학습 시나리오

### 4.1 모델 — Llama 3.1 8B

| 항목 | 값 |
|---|---|
| Parameters | 8.03 B |
| Hidden size (h) | 4096 |
| Layers (L) | 32 |
| Q heads (a) | 32 |
| KV heads | 8 (GQA) |
| FFN intermediate | 14336 |
| Vocab | 128,256 |

**Capacity 메시지**: 이론적으로는 A100 40GB×8에 fit하지만 (peak ~36 GB / 40 GB), PyTorch fragmentation, NCCL/cuBLAS/FA workspace 등으로 실제로는 OOM borderline. 16 GPU가 필요한데 A100 16장 availability 부족 → 이기종 환경 사용. 시스템 논문의 "theoretical fit ≠ practical fit" 패턴.

### 4.2 데이터셋 — Alpaca 변종 (확정 필요)

데이터셋 분포 (실제 확인된 값):
- instruction: 9~489 토큰
- input: 0~2.47k 토큰
- output: 0~4.18k 토큰
- text(combined): 154~4.5k 토큰
- 평균: ~127 토큰 (Llama tokenizer, 원본 alpaca 기준)
- 총 토큰: ~6.6 M (원본 기준, cleaned는 더 많을 수 있음)

> TODO: 실험 시작 전 정확한 dataset 이름과 토큰 통계 확정

### 4.3 Hyperparameters (모든 configuration 공통)

| 항목 | 값 | 근거 |
|---|---|---|
| Global batch size | 128 | Alpaca/Vicuna/Tulu fine-tuning 표준 |
| Micro-batch (b) | 4 | 메모리 한계 내 throughput 최적화 |
| DP | 1 | 본 실험은 PP/TP 변동만 측정, DP는 고정 |
| Gradient accumulation | 32 | global / (b × DP) = 128/4 |
| Sequence length | 8192 | 데이터셋 분포의 ~100% 보존 |
| Sequence packing | ON, varlen FA2 + cu_seqlens | padding 최소화 |
| Precision | BF16 mixed, FP32 master + Adam m/v | 표준 mixed-precision |
| Activation recompute | selective (Korthikanti) | wall-clock penalty <3% |
| Sequence parallel (SP) | ON (TP>1 인 모든 config) | 활성을 1/t 로 |
| Optimizer | AdamW | LR 2e-5 (실제 학습 품질은 측정 대상 아님) |
| Epochs | 1 | wall-clock 충분 |
| PYTORCH_CUDA_ALLOC_CONF | expandable_segments:True | fragmentation 회피 |
| Warm-up steps (측정 제외) | 100 | CUDA 캐시 / NCCL handshake 안정화 |
| Measurement steps | 200 (warm-up 이후) | throughput/power 평균 |

### 4.4 메모리 회계 (참고)

per-layer activation = `34 × s × b × h / t` bytes (selective recompute, SP on, BF16)

Llama 3.1 8B, b=4, s=8192, TP=8 SP on:
- per-layer activation: 544 MB
- whole-model activation (PP-invariant): 17.0 GB
- weight+grad+optim per GPU (TP=8): 16.06 GB
- buffer (NCCL/cuBLAS/FA/CUDA context): ~3 GB
- 합계: 36.0 GB (40 GB의 90%, fragmentation 마진 포함 시 borderline)

---

## 4.5 개발 / 검증 환경 (Development Cluster)

본 motivation figure는 A100 ×8 + L40S ×8 (16 GPU) 클러스터에서 측정하나, **nanotron asymmetric parallelism fork의 코드 검증을 A100 8장 클러스터에서 iteration마다 돌리는 것은 비용/availability 모두 비효율**. 작은 GPU 인스턴스에서 작은 모델로 정합성 검증을 끝낸 후, 본 클러스터에서는 측정만 진행.

본 연구의 핵심 기여는 **stage별 다른 TP 차수**가 아니라 **PP에서의 비대칭 stage** (다른 GPU 종류, 다른 TP, 다른 layer 수)이므로, 개발 단계에서 TP 검증은 후순위. PP=2 TP=1로 stage별 GPU 종류만 다른 환경에서 cross-stage P2P, layer-wise grad AllReduce, 비대칭 layer split까지 모두 검증 가능.

### 개발 검증 단계

**Stage 1: 단일 인스턴스 단일 GPU**
- 인스턴스: `g6.xlarge` (NVIDIA L4, 24 GB)
- 비용: ~$0.8/h on-demand
- 목적: nanotron fork 빌드 검증, Llama 3.2 1B forward/backward 정상 동작 확인, PP=1 TP=1 baseline loss 추이 검증

**Stage 2: 다중 인스턴스 (이기종 inter-node)**
- 인스턴스: `g5.xlarge` (A10G, 24 GB) + `g6.xlarge` (L4, 24 GB), 각 1 GPU
- AWS placement group + EFA 또는 같은 AZ 일반 네트워크
- 목적: PP=2 TP=1 inter-node 정합성, 비대칭 layer split, NCCL 토폴로지, EFA P2P 동작 검증
- 본 실험과 동일한 multi-instance NCCL 셋업이라 본 클러스터로 넘어갈 때 surprise 없음

### 개발용 모델 — Llama 3.2 1B

| 항목 | 값 |
|---|---|
| Parameters | 1.24 B |
| Hidden size | 2048 |
| Layers | 16 |
| Q heads / KV heads | 32 / 8 (GQA) |
| Tied embeddings | Yes |
| 24GB GPU 메모리 (PP=1, b=2, s=512) | ~10 GB / GPU |
| 24GB GPU 메모리 (PP=2 TP=1, b=2, s=512, stage 0) | ~7 GB / GPU |

선정 이유: 본 실험의 Llama 3.1 8B와 동일한 architecture family (RoPE, GQA, RMSNorm, SwiGLU). 코드 path가 그대로 재사용되므로 8B scale-up 시 surprise 최소.

대안: TinyLlama 1.1B (Llama 2 family, GQA 없음 — architecture 차이 있어 검증 가치 약간 낮음).

### 개발 단계 hyperparameters

| 항목 | 값 (개발) | 본 실험과의 차이 |
|---|---|---|
| Model | Llama 3.2 1B | 8B → 1B |
| Sequence length | 512 또는 1024 | 8192 → 512 |
| Micro-batch | 1 또는 2 | 4 → 1~2 |
| Global batch | 8~16 | 128 → 8~16 |
| Steps | 50~200 | 1 epoch → 짧게 |
| Activation recompute | OFF | selective → off (메모리 여유 충분) |

목적은 wall-clock이나 throughput이 아니라 **코드 정합성 sanity check**.

### 개발 단계 검증 항목

**Stage 1 (g6.xlarge 단일 GPU):**
1. nanotron fork 빌드, FlashAttention 2 sm_89 빌드 정상
2. Llama 3.2 1B HF → nanotron 변환 무결성
3. PP=1 TP=1 baseline loss 정상 시작 (vocab 128k이므로 초기 loss ≈ ln(128256) ≈ 11.76) 및 감소 추이
4. Alpaca packing pipeline 동작

**Stage 2 (g5.xlarge + g6.xlarge multi-instance):**
1. NCCL inter-node 통신 (EFA / placement group 기반) 정상 동작
2. PP=2 TP=1 동질 layer split (8/8) loss가 PP=1 baseline과 일치 — gradient correctness
3. **PP=2 TP=1 비대칭 layer split (10/6 등)** loss가 균등 split과 일치 — 본 연구 핵심 정합성
4. Stage 0 (A10G) ↔ Stage 1 (L4) cross-stage P2P 정상 (이기종 NCCL)
5. 1F1B in-flight micro-batch 메모리 회계 정확 (`torch.cuda.max_memory_allocated()` 검증)

> Stage별 다른 TP 차수 검증은 본 클러스터에서 처음 시도. 개발 클러스터의 단일 GPU per node 셋업으로는 검증 불가하나, 이는 PP 정합성이 검증된 코드 위에서 TP 차원 추가만 하는 작업이라 위험도 낮음.

### 비용 비교 (us-east-1 on-demand 기준 참고)

| 클러스터 | 시간당 | 1주 fulltime 개발 |
|---|---:|---:|
| Stage 1: g6.xlarge | ~$0.8 | ~$135 |
| Stage 2: g5.xlarge + g6.xlarge | ~$1.8 | ~$300 |
| 본 실험: A100 ×8 클러스터 | ~$32.8+ | ~$5,500+ |

본 실험은 측정 단계에서만 잡고, 그 이전의 iteration은 모두 개발 클러스터에서 진행.

---

## 5. Configuration Sweep

motivation figure 및 ablation을 위한 6가지 설정. 모든 config는 동일한 global batch (128), seq_len (8192), packed dataset 사용.

| ID | 하드웨어 | TP / PP | layer split | 의도하는 메시지 |
|---|---|---|---|---|
| **A** | A100 ×8 only | TP=8, PP=1 | — | Homogeneous baseline (theoretical fit, fragmentation borderline) |
| **A'** | A100 ×16 (가설) | TP=8, PP=2 | 16/16 | A의 안전판 (실제로는 availability 없어 못 받음 — motivation에 사용) |
| **B** | A100 ×8 + L40S ×8 | PP=2, both TP=8 | 16/16 | **Naive symmetric**: L40S TP=8이 PCIe all-reduce에 갇혀 병목 → 더 많은 GPU/전력 쓰고도 성능 비슷 (anti-baseline) |
| **C** | A100 ×8 + L40S ×8 | PP=2, A100 TP=8, L40S TP=4 (TP 비대칭) | 16/16 | TP를 줄여서 L40S PCIe 부담 완화 |
| **D** | A100 ×8 + L40S ×8 | PP=2, A100 TP=8, L40S TP=4 | **19 / 13** (layer 비대칭) | **Asymmetric layer split**: stage 처리시간 균등화 → pipeline bubble 최소 (본 연구 기여) |
| **E** | A100 ×8 + L40S ×8 | PP=4 fine-grained | 11 / 9 / 7 / 5 | PP 깊이가 깊어질수록 bubble 증가 vs balance — D보다 나은지 ablation |

D가 본 연구의 핵심 기여, B는 "그냥 GPU 더하면 안 됨"의 anti-example.

---

## 6. 측정 항목

### 6.1 Primary Metrics

- **Throughput (tokens/sec)**: 100 step sliding window 단위로 기록 (시간에 따른 변화 추적)
- **End-to-end wall-clock**: 1 epoch 학습 전체 시간 (실제 운영 지표)
- **Power (Watts)**: GPU 합산 + host 포함 시스템 전력 (별도 로깅)
- **Energy efficiency (tokens/Joule)**: 1 epoch 누적 기준 핵심 metric

### 6.2 측정 분리 — 학습 코드 vs 외부 모니터링

**학습 코드 내부 (매 step, sync 없는 항목만)**:
- step_time (`time.perf_counter()`)
- tokens_processed
- `torch.cuda.max_memory_allocated()` — PyTorch 내부 카운터, GPU sync 없음
- loss는 매 100 step마다만 `.item()` (CPU↔GPU sync 비용 회피)
- W&B/TensorBoard log는 매 10 step에 한 번 (async batch)

**외부 모니터링 (학습 코드와 무관, 0 overhead)**:
- DCGM daemon (별도 프로세스)
- IPMI / PDU host 전력

학습 시작 전 background로 모니터링 daemon 띄우고, 학습 시작/종료 timestamp만 학습 코드에서 기록. 나중에 timestamp로 join.

### 6.3 Throughput 측정 방식

기록 단위:
- 매 step마다 step_time, tokens 로깅 (overhead 없는 것만)
- 100 step rolling average로 시각화 (smoothing)
- Warm-up 첫 100 step은 시각화 제외, raw log 보존
- 1 epoch 종료 시 end-to-end throughput (총 토큰 / 총 시간) 별도 산출

기록 이유:
- Step별 변동성 (fragmentation 누적 영향)
- Configuration 간 시간 추이 비교
- LR warmup 영향
- End-to-end는 200 step 평균과 다를 수 있음 (dataloader 재시작, checkpoint, NCCL 재연결 등 amortize 안 되는 비용)

### 6.4 DCGM 메트릭 — "관련 있는 것 전부" 정책

본 연구의 figure에 어떤 metric이 쓰일지 사전에 100% 알 수 없음. DCGM 수집은 학습 throughput에 0 overhead이므로 **관련 가능성 있는 모든 field를 수집**, 분석 단계에서 선택.

수집 대상 field (DCGM 4.5.2 기준 short name과 함께):

| Field ID | Long name (Short) | 본 연구 활용 |
|---:|---|---|
| 155 | power_usage (POWER) | 핵심 — 에너지 효율 |
| 156 | total_energy_consumption (TOTEC) | 누적 에너지 (시간 적분 불필요) |
| 150 | gpu_temp (TMPTR) | thermal throttling 확인 |
| 100 | sm_clock (SMCLK) | clock throttling |
| 101 | memory_clock (MMCLK) | 보조 |
| 203 | gpu_utilization (GPUTL) | coarse, SMACT가 더 정확 |
| 204 | mem_copy_utilization (MCUTL) | 보조 |
| 1001 | gr_engine_active (GRACT) | graphics engine 활성도 |
| 1002 | sm_active (SMACT) | 핵심 — SM 점유율 |
| 1003 | sm_occupancy (SMOCC) | 핵심 — SM 활용도 |
| 1004 | tensor_active (TENSO) | 핵심 — **tensor core 사용률 (BF16/FP16 포함)** |
| 1005 | dram_active (DRAMA) | 핵심 — memory bandwidth |
| 1007 | fp32_active (FP32A) | 보조 |
| 1008 | fp16_active (FP16A) | 보조 — **BF16은 잡히지 않음, BF16 활용도는 TENSO 사용** |
| 1009 | pcie_tx_bytes (PCITX) | 핵심 — L40S PCIe 통신량 |
| 1010 | pcie_rx_bytes (PCIRX) | 핵심 |
| 1011 | nvlink_tx_bytes (NVLTX) | 핵심 — **A100에만 적용 (L40S/L4는 NVLink 부재)** |
| 1012 | nvlink_rx_bytes (NVLRX) | A100에만 |

총 18개 (A100 stage). L40S stage는 1011/1012 제외한 16개 — 한 field라도 unsupported이면 dcgmi dmon이 watch 등록을 atomic하게 거부하므로 stage별로 분리해야 한다.

> 주의 1: profiling field (1001~1012)는 DCGM이 multiplexing으로 round-robin sampling. 1초 해상도라도 모든 field가 매 초 갱신되지는 않음. NVIDIA hardware counter 한계.
> 주의 2: 일부 GPU 모델에서 1002/1003 sampling 시 hang 사례 보고 (DCGM Issue #144). 첫 sanity run에서 확인 필요. (Stage 1 / L4에서는 미재현)
> 주의 3: `dcgmproftester` 류 stress 도구와 `dcgmi dmon`을 동시에 띄우면 profile field sampling이 0으로 보고됨 (multi-client watch 충돌). 학습 측정 중에는 dcgmi dmon 단독으로만.

### 6.5 시계열 수집 방법

본 연구는 단일 클러스터 단발성 측정이므로 Prometheus 같은 시계열 DB는 과함.

**`dcgmi dmon` text 출력 + awk로 JSONL 변환**

DCGM 4.5.2의 `dcgmi dmon`은 `-j` flag를 지원하지 않음 (`-j`는 `dcgmi stats` subcommand의 "job id" 인자). text 출력을 awk로 JSONL 변환한다 (helper: [scripts/dcgm_text_to_jsonl.awk](../scripts/dcgm_text_to_jsonl.awk) 작성 예정).

```bash
# A100 노드 (NVLink 있음, 18 field)
A100_FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010,1011,1012"
dcgmi dmon -i 0,1,2,3,4,5,6,7 -e $A100_FIELDS -d 1000 \
  | tee dcgm_a100.txt | awk -f scripts/dcgm_text_to_jsonl.awk > dcgm_a100.jsonl &
A100_PID=$!

# L40S 노드 (NVLink 부재, 16 field — 1011/1012 제외)
L40S_FIELDS="155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010"
dcgmi dmon -i 0,1,2,3,4,5,6,7 -e $L40S_FIELDS -d 1000 \
  | tee dcgm_l40s.txt | awk -f scripts/dcgm_text_to_jsonl.awk > dcgm_l40s.jsonl &
L40S_PID=$!

# 학습 실행
torchrun ... run_train.py

kill $A100_PID $L40S_PID
```

후처리:
```python
import pandas as pd
df_a100 = pd.read_json('dcgm_a100.jsonl', lines=True)  # ts, gpu_id, POWER, TOTEC, ... 18 col
df_l40s = pd.read_json('dcgm_l40s.jsonl', lines=True)
df_train = pd.read_csv('train_steps.csv')  # 학습 코드가 step별 timestamp 기록
# unix timestamp로 merge
```

shell script로 완결, 외부 인프라 불필요. 학습 코드는 step별로 (start_ts, end_ts, tokens)만 자기 log에 기록하면 됨 — DCGM과 awk가 모두 같은 unix time을 쓰므로 별도 동기화 불필요. 단, awk `systime()` 호출은 line-by-line이라 timestamp 정밀도는 ±1초 (학습 step 단위 join에는 충분).

> Prometheus + DCGM-Exporter는 multi-host 장기 모니터링용. 본 연구는 단일 클러스터에서 5번 정도 측정하는 단발성이라 셋업 오버헤드 대비 이득 적음. 연구실에 이미 Prometheus 인프라 있으면 그대로 사용 가능.

### 6.6 측정 정확성을 위한 통제

- 모든 config 동일한 random seed (deterministic input)
- `torch.backends.cudnn.benchmark=True` 동일 적용
- NCCL: `NCCL_DEBUG=INFO`로 첫 run에서 PCIe vs NVLink 경로 확인
- `NCCL_TOPO_FILE` 또는 `NCCL_P2P_DISABLE=0`, `NCCL_NET_GDR_LEVEL=2` 명시
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation 회피)
- 매 config 시작 전 GPU reset 또는 process 완전 재시작 (이전 state 영향 차단)
- DCGM/NVML 모니터링은 학습 시작 전부터 종료 후까지 계속 돌려서 baseline (idle) 도 캡쳐

---

## 7. Risk & Mitigation

| Risk | 영향 | Mitigation |
|---|---|---|
| nanotron fork 작업이 예상 초과 | 전체 일정 지연 | 가장 단순한 PP=2 case 먼저 구현, PP=4는 후순위 |
| Config A에서 OOM 안남 (이론 36GB라 들어감) | Capacity motivation 약화 | b=8 또는 s=16384로 늘려 OOM 명확화. 또는 메시지를 "throughput/power 효율"로 축소 |
| L40S sm_89 FA2 빌드 실패 | L40S 측 학습 불가 | xformers fallback, 또는 native attention + selective recompute |
| Inter-node bandwidth 부족 | PP의 P2P가 너무 느림 | NCCL_NET_GDR_LEVEL 조정, 또는 같은 host 내 mixed GPU로 변경 |
| nanotron의 DP=1 + PP 비대칭 정합성 문제 | 학습 진행 불가 | 단위 테스트에서 미리 확인, 단순 case부터 검증 |
| Llama 3.1 8B가 시나리오에 충분히 maxout 안 됨 | 메시지 약화 | Qwen 2.5 14B로 plan B 준비 (메모리 회계 이미 수립됨) |
