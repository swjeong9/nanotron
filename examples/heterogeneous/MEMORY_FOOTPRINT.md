# Heterogeneous PP 학습의 GPU 메모리 추정

본 문서는 nanotron 으로 LLM 을 PP (Pipeline Parallelism) + TP (Tensor Parallelism)
환경에서 학습할 때, 각 GPU 의 메모리 사용량을 분석적으로 계산하는 방법을 설명한다.
실측 데이터와 비교 검증을 포함.

대상 setup:
- **mixed precision**: BF16 weights + FP32 gradients/optimizer (`accumulate_grad_in_fp32: true`)
- **AdamW** optimizer, `zero_stage: 0` (state 복제 없음, DP-shard 안 함)
- **Flash Attention 2** (attention 활성화 backward 에서 recompute)
- **Selective activation recompute** 가정 (Megatron-LM 의 표준 접근)

---

## 1. 메모리 구성

GPU 메모리는 **4 가지 컴포넌트** 의 합:

| # | 컴포넌트 | scaling factor |
|---|---|---|
| 1 | **Model state** (weights + grad + optimizer) | params × bytes_per_param / TP |
| 2 | **Activations** (in-flight microbatches) | layers × in_flight × per_layer_per_mb |
| 3 | **LM head intermediates** (logits, loss) | b × s × vocab × bytes / TP |
| 4 | **Framework overhead** (CUDA, NCCL, PyTorch caching) | ~1-2 GB constant + caching slack |

---

## 2. Model state per GPU

AdamW + grad accumulation in FP32 의 경우:

```
bytes_per_param = BF16_weight (2) + FP32_grad_accum (4) + AdamW_m (4) + AdamW_v (4)
                = 14 bytes/param
```

`zero_stage: 0` 면 DP 간 sharding 없음. TP 가 layer 의 weight 를 4-way 분산:

```
state_per_GPU = (layer_count × params_per_layer + extras) × 14 / TP   bytes
```

**Llama 3.2 3B 기준** (28 layers, hidden 3072):
- per layer params ≈ 107M (attention QKV+O + FFN gate/up/down)
- embedding/lm_head ≈ 394M (vocab=128256, tied weight 라 한 번만 저장)

| Partition | Stage 0 layers (L4) | Stage 1 layers (A10G) | extras |
|---|:---:|:---:|---|
| [14, 14] | 14 | 14 | embed → stage 0, lm_head → stage 1 |
| [1, 27] | 1 | 27 | (동일) |

State per GPU 예 (TP=4):
- Stage 0 [14 layer]: `(14 × 107M + 394M) × 14 / 4 = 6.6 GB`
- Stage 1 [14 layer]: `(14 × 107M + 0) × 14 / 4 = 5.2 GB` (embed 가 stage 0 에 있으니까 stage 1 는 lm_head 만, 근데 tied 라 추가 안 됨)
- 단, **tied embedding 의 grad sync** 를 위해 두 stage 모두 embedding param 의 사본을 가져야 함 → 둘 다 ~5-6 GB

(실제 nanotron 에선 `tie_word_embeddings: true` 일 때 양 stage 의 rank 0 (TP=0) 만
embedding 보유. 자세한 건 `src/nanotron/parallel/tied_parameters.py` 참조.)

---

## 3. Activations (Megatron-LM 공식)

**핵심 공식** (selective recomputation + flash attention 가정, no sequence parallelism):

```
activation_per_layer_per_mb_per_GPU = s · b · h · (10 + 24/t)   bytes
```

### 출처

원 논문: **["Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)**
(Korthikanti et al., NVIDIA, 2022). Section 4.1 "Selective Activation Recomputation",
Table 2 + 본문 분석.

논문의 핵심 결과:
- **No optimization**: `s · b · h · (34 + 5·a·s/h)` bytes per transformer layer per microbatch
- **Selective activation recomputation** (attention 활성화는 backward 에서 recompute): drop `5·a·s/h`
  → `s · b · h · 34` bytes per layer per mb (no TP, no SP)
- **+ Tensor Parallelism (TP)**: 활성화의 일부 (attention output, FFN intermediates) 만 TP-sharded.
  TP-replicated 부분 (LN, dropout, residual) 은 그대로 → `s · b · h · (10 + 24/t)`

논문의 공식 `34` 은 우리의 `(10 + 24/t)` 의 `t=1` 케이스 (TP 없음). TP=t 면 24 부분만 sharded.

논문 Table 2 의 정확한 분해:
| 컴포넌트 | bytes | TP-shardable? | 우리 공식의 항 |
|---|---|:---:|---|
| Attention input | 2·sbh | shardable (Q, K, V projections) | 24/t 의 일부 |
| QKV matmul output | 6·sbh | ✓ | 24/t 의 일부 |
| Attention scratch | 5·a·s²·h | ✓ | flash attn 으로 0 |
| Attention output projection | 2·sbh | ✓ | 24/t 의 일부 |
| Dropout mask | 1·sbh | ✗ (replicated) | 10 의 일부 |
| LayerNorm | 2·sbh | ✗ | 10 의 일부 |
| MLP up projection input | 2·sbh | ✗ | 10 의 일부 |
| MLP intermediate | 8·sbh | ✓ | 24/t 의 일부 |
| MLP down projection input | 8·sbh | ✓ | 24/t 의 일부 |
| Residual streams + 기타 | 5·sbh | ✗ | 10 의 일부 |
| **합 (full, no opt)** | **34·sbh + 5·a·s²·h** | | |
| **selective + flash + TP=t** | **(10 + 24/t)·sbh** | | ✓ |

`(10 + 24/t)` 형태는 HuggingFace optimization docs, OPT/BLOOM training blog 등 secondary
source 에서도 자주 사용. 논문의 결과를 TP-share 가능성으로 분해한 표현.

### 분해 확인

- `s` = sequence length
- `b` = micro batch size
- `h` = hidden size
- `t` = TP size
- **10 · s · b · h**: LayerNorm input/output, dropout mask, residual stream — TP 와 무관 (각 GPU 가 동일 사본)
- **24/t · s · b · h**: attention QKV/output projection, GeLU/SwiGLU intermediate, FFN matmul outputs — TP-sharded

이 공식은 **flash attention 사용 시** 정확 (attention scratch `5·a·s²/h` 항이 0).
표준 attention 이면 그 항이 추가됨.

### Sequence Parallelism (SP) 사용 시

SP 가 적용되면 LN/dropout/residual 도 sequence dim 으로 t-way 분산 → `(10 + 24)/t = 34/t`.
우리 setup 은 **SP 미사용** 이라 `(10 + 24/t)` 가 정확.

**예 — 3B mbs=2 seq=1024 TP=4**:
```
1024 × 2 × 3072 × (10 + 24/4) = 1024 × 2 × 3072 × 16 bytes = 100.7 MB / GPU / layer / mb
```

**예 — 3B mbs=8 seq=4096 TP=8** (p4d.24xl 환경):
```
4096 × 8 × 3072 × (10 + 24/8) = 4096 × 8 × 3072 × 13 bytes = 1.31 GB / GPU / layer / mb
```

**활성화 16× 증가의 함정**:
mbs×seq 가 16× 증가하면 activation memory 도 16× — 그러나 **state 는 안 변함**.
cluster 전체 메모리는 16× 가 아니라 ~3-5× 만 증가 (state + framework 가 큰 부분).

---

## 4. In-flight microbatches in 1F1B PP

1F1B (one-forward-one-backward) pipeline schedule 에서 각 stage 의 활성화 저장
in-flight 개수:

```
in_flight_at_stage_k = pp_size - k   (warmup peak 기준)
```

**PP=2 의 경우**:
- Stage 0: **2 in-flight** (warmup 시 fwd 2개, 그 후 1F1B steady)
- Stage 1: **1 in-flight** (downstream 이 항상 막 받자마자 처리)

**PP=3 의 경우**:
- Stage 0: 3 in-flight
- Stage 1: 2 in-flight
- Stage 2: 1 in-flight

→ stage 0 는 항상 가장 많은 활성화 보관 → 메모리 압박 큼.

---

## 5. LM head intermediates

LM head (output projection) 가 stage `pp_size-1` (last stage) 에 위치:

```
lm_head_logits_FP32 = b × s × vocab × 4   bytes      (forward)
                    / TP                              (TP-shardable)
```

cross-entropy loss 는 logits 를 FP32 로 cast 한 후 softmax 라 큰 메모리.

**예 — 3B (vocab=128256) mbs=2 seq=1024 TP=4**:
```
2 × 1024 × 128256 × 4 / 4 = 263 MB / GPU
```

**예 — 3B mbs=8 seq=4096 TP=4**:
```
8 × 4096 × 128256 × 4 / 4 = 4.2 GB / GPU         ← 매우 큼!
```

대규모 vocab 모델 (Llama, Qwen) 에선 lm_head 가 stage 1 의 메모리 주범 중 하나.

---

## 6. Framework overhead

대략적 상수:

| 항목 | 추정 |
|---|---:|
| CUDA context, primary stream | ~600 MB |
| NCCL communication buffers (TP+PP) | ~500 MB |
| PyTorch caching allocator slack | ~1-3 GB (peak alloc 의 20-50%) |
| nanotron internal buffers | ~200 MB |
| **합** | **~2-4 GB / GPU** |

PyTorch 의 caching allocator 는 free 후 다시 OS 에 안 돌려주고 cache 해서, **nvidia-smi
의 max 가 PyTorch 의 max_reserved 보다 1-2 GB 더 높게 보고됨**.

---

## 7. Worked example: Llama 3.2 3B [14, 14] mbs=2 seq=1024 TP=4 PP=2

config:
- s=1024, b=2, h=3072, vocab=128256
- TP=4, PP=2, ga=64 (GBS=131,072 tokens/step)
- 28 layers (per_layer ≈ 107M params)

### Stage 0 (L4 × 4) per-GPU 추정

| 컴포넌트 | 계산 | 값 |
|---|---|---:|
| State | `(14 × 107M + 394M_embed) × 14 / 4` | 6.62 GB |
| Activation | `1024 × 2 × 3072 × 16 × 14_layers × 2_inflight = 100.7 MB × 14 × 2` | 2.82 GB |
| Framework | constant | ~1.5 GB |
| **합 (active alloc)** | | **~10.9 GB** |
| PyTorch caching slack | +20-30% | +2.2 GB |
| **nvidia-smi peak 추정** | | **~13 GB / GPU** |

### Stage 1 (A10G × 4) per-GPU 추정

| 컴포넌트 | 계산 | 값 |
|---|---|---:|
| State | `(14 × 107M) × 14 / 4` (embed stage 0 에 있음) | 5.24 GB |
| LM head logits (FP32) | `2 × 1024 × 128256 × 4 / 4` | 263 MB |
| Activation | `100.7 MB × 14 × 1` | 1.41 GB |
| Framework | constant | ~1.5 GB |
| **합 (active alloc)** | | **~8.4 GB** |
| PyTorch caching slack | +20-30% | +2 GB |
| **nvidia-smi peak 추정** | | **~10.4 GB / GPU** |

### 실측 (sweep raw archive 에서)

```
[14, 14] L4   (NODE 0): nvidia-smi max = 12,134 MiB / GPU  → 11.85 GB
                        PyTorch max_reserved = 11,288 MiB    → 11.02 GB
[14, 14] A10G (NODE 1): nvidia-smi max = 11,776 MiB / GPU  → 11.50 GB
                        PyTorch max_reserved = 10,828 MiB    → 10.57 GB
```

추정 vs 실측 격차:
- L4: 추정 ~13 → 실측 11.85 (12% over-estimate)
- A10G: 추정 ~10.4 → 실측 11.50 (10% under-estimate)

**±15% 이내 정확**. 큰 항목 (state, activation) 이 잘 맞고 framework overhead 가 noise.
A10G 가 약간 under 인 이유: 우리 모델이 `(10 + 24/4) = 16` 으로 가정한 활성화 부분이
실제로는 약간 더 클 수 있음 (e.g., `tp_linear_async_communication: false` 의 sync buffer).

---

## 8. 자료 측정 방법

### 8.1 nvidia-smi (sustained max)

benchmark_single.sh 에서 1 Hz polling:
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd,
```
4 GPU 의 comma-separated 값. plot_single.py 의 `parse_nvidia_smi_per_gpu` 가 파싱.

### 8.2 PyTorch reserved (caching allocator 기준)

`src/nanotron/logging/base.py` 의 `log_memory()` 가 학습 중 주요 시점에 출력:
```
[INFO|PP=0|TP=0] Memory usage: <live>MiB. Peak allocated: <alloc>MiB. Peak reserved: <reserved>MiB
```

3 가지 값:
- **memory.allocated**: PyTorch 가 즉시 사용 중인 (live tensor 의 합)
- **max_allocated**: PyTorch 가 동안 가장 높았던 allocated
- **max_reserved**: PyTorch caching allocator 가 OS 로부터 받은 총량 (실 사용 + cache slack)

`max_reserved` 가 가장 신뢰할 수 있는 PyTorch 측 max.

### 8.3 stats.json 의 per-rank breakdown

`examples/heterogeneous/data/<cluster>/<model>/<descriptor>/stats.json` 의
`memory_peaks` 필드:

```json
{
  "memory_peaks": {
    "node0_l4": {
      "per_rank": {
        "pp0_tp0": {"max_live_MiB": 7802, "max_alloc_MiB": 11127, "max_reserved_MiB": 11288},
        "pp0_tp1": {...},
        ...
      },
      "max_live_MiB": 7802,           // max across ranks
      "max_alloc_MiB": 11127,
      "max_reserved_MiB": 11288
    },
    "nvidia_smi_per_gpu_node0_l4": [12134, 12134, 12134, 12134],   // per-GPU max
    "nvidia_smi_max_MiB_node0": 12134
  }
}
```

→ 4 rank 모두 거의 동일 (TP-shard 가 균등 분산하므로) — sanity check.

---

## 9. 예측 vs 실측 비교 워크플로우

```python
def estimate_memory_per_gpu(
    layers, in_flight, per_layer_params, embed_params,
    s, b, h, vocab, t, has_lm_head, has_embed,
    bytes_per_param=14, bf16=2, fp32=4
):
    # State
    state = (layers * per_layer_params + (embed_params if has_embed else 0)) * bytes_per_param / t

    # Activations (Megatron formula, selective + flash)
    act_per_layer_mb = s * b * h * (10 + 24 / t)
    activation = layers * in_flight * act_per_layer_mb

    # LM head (last stage)
    lm_head = (b * s * vocab * fp32 / t) if has_lm_head else 0

    # Framework constant
    framework = 1.5 * 2**30   # 1.5 GB

    return {
        "state_GB": state / 2**30,
        "activation_GB": activation / 2**30,
        "lm_head_GB": lm_head / 2**30,
        "framework_GB": framework / 2**30,
        "total_active_GB": (state + activation + lm_head + framework) / 2**30,
        "nvsmi_with_caching_GB": (state + activation + lm_head + framework) / 2**30 * 1.25,
    }
```

각 sweep 결과의 stats.json 과 비교하면 ±15% 이내 일치 예상. 더 큰 격차 시:
- `flash_attention_2` 가 정상 동작 안 함 (attention activation 추가) → +`5·a·s²/h` per layer
- nanotron 의 `_use_doc_masking` 등 추가 buffer 가 큼
- TP all-reduce buffer 가 예상보다 큼

---

## 10. 흔한 함정

| 함정 | 설명 |
|---|---|
| **"mbs×seq 16× → 메모리 16×"** | activation 만 16×. state, lm_head 일부, framework 는 변함 없음. cluster 합은 ~3-5× |
| **"TP 늘리면 메모리 1/N"** | `(10 + 24/t)` 의 10 부분은 TP 와 무관 → `t=4` 에서 16×, `t=8` 에서 13× — TP=8 이 TP=4 의 절반이 아님 |
| **stage 0 가 stage 1 보다 큰 메모리** | PP=2 의 1F1B in-flight: stage 0 가 2 mb 보관 → activation 2× more than stage 1 |
| **lm_head 무시** | vocab 큰 모델에서 logits FP32 cast 가 GB 단위. b·s 클 때 특히 |
| **"nvidia-smi = PyTorch reserved"** | nvidia-smi 가 1-2 GB 더 큼 (caching allocator slack + CUDA context) |
| **TP shard 가 정확히 균등하지 않음** | hidden 이 TP 로 안 떨어지는 경우 (e.g., 3072/4=768 OK, 256/4=64 OK) — 보통 정수 나눗셈이라 균등 |
| **Sequence Parallelism 가정 금지** | 본 setup 은 SP 안 씀. SP 면 `(10 + 24/t)` 가 `34/t` 로 줄어 더 작아짐 — 별도 분석 필요 |

---

## 11. 빠른 견적 — 다양한 config

(28 layers, hidden 3072, vocab 128256, AdamW + FP32 grad, Flash Attn 2)

### Stage 0 (embed 포함, in-flight=2 if PP=2)

| Config | TP | layers | per-GPU 추정 |
|---|:---:|:---:|---:|
| mbs=2 seq=1024 PP=2 [14, 14] | 4 | 14 | ~11 GB |
| mbs=8 seq=4096 PP=2 [14, 14] | 8 | 14 | ~36 GB **(A100 40GB tight!)** |
| mbs=8 seq=2048 PP=2 [14, 14] | 8 | 14 | ~21 GB |
| mbs=4 seq=4096 PP=2 [14, 14] | 8 | 14 | ~21 GB |

### Stage 1 (lm_head 포함, in-flight=1 if PP=2)

| Config | TP | layers | per-GPU 추정 |
|---|:---:|:---:|---:|
| mbs=2 seq=1024 PP=2 [14, 14] | 4 | 14 | ~10 GB |
| mbs=8 seq=4096 PP=2 [14, 14] | 4 | 14 | ~32 GB |
| mbs=8 seq=4096 PP=2 [14, 14] | 8 | 14 | ~17 GB |

→ **TP=8 이 stage 1 메모리 절감에 유리** (lm_head 가 4× 더 sharded).

---

## 12. References

- **["Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)** — Korthikanti et al. (NVIDIA, 2022). 활성화 메모리 공식 `s·b·h·(34 + 5·a·s/h)` (no opt) → `(34/t)` (SP) → `(10 + 24/t)` (selective+TP, no SP) 의 원천. Section 4.1, Table 2.
- **["Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"](https://arxiv.org/abs/2104.04473)** — Narayanan et al. (NVIDIA, 2021). 1F1B PP schedule + bubble 분석.
- [Megatron-LM official repo](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py) — 활성화 메모리 추정 코드 (`get_train_iteration_metrics` 등)
- [HuggingFace transformers — performance docs](https://huggingface.co/docs/transformers/perf_train_gpu_many) — 분산학습 메모리 분석 secondary source
- [PROJECT_BACKGROUND.md](docs/project_background.md) — 본 cluster 의 hardware spec
- [PLAN_3B_SWEEP.md](examples/heterogeneous/PLAN_3B_SWEEP.md) — 3B sweep 의 설정
- [plot_single.py](examples/heterogeneous/plot_single.py) — 메모리 측정 파싱 코드
