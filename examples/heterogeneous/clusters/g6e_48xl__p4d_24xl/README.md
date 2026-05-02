# Cluster: `g6e_48xl__p4d_24xl`

## 구성

2-node heterogeneous cluster (g6e.48xlarge + p4d.24xlarge). 16 GPU, **PP=4 × TP=4 × DP=1**.
nproc_per_node 가 양 노드에서 동일 (8) 이라 이전 3-node (4+4+8) 의 비대칭 NCCL deadlock issue 회피.

| 노드 | 인스턴스 | GPU | VRAM | TP group | PP stage |
|---|---|---|---|---|---|
| NODE 0 | g6e.48xlarge | 8× L40S | 48 GB | TP-0 (rank 0–3), TP-1 (rank 4–7) | Stage 0 + Stage 1 |
| NODE 1 | p4d.24xlarge | 8× A100 | 40 GB | TP-2 (rank 8–11), TP-3 (rank 12–15) | Stage 2 + Stage 3 |

총 16 GPU. 각 TP group (4 rank) 이 정확히 한 노드 내에 묶여 intra-node 만 사용. PP comm 은
Stage 0→1 (intra-g6e), Stage 1→2 (cross-node), Stage 2→3 (intra-p4d / NVSwitch).

## 이전 cluster 와 비교

| | 3-node (g6e×2 + p4d) | **2-node (이 cluster)** |
|---|---|---|
| 인스턴스 | 4+4+8 GPU 비대칭 | **8+8 GPU 균등** |
| nproc_per_node | 4 / 4 / 8 | **8 / 8** |
| `local_world_size` 비대칭 NCCL deadlock | 발생 (fix 필요했음) | **발생 안 함** |
| Cross-node PP link | 2 개 (0↔1, 1↔2) | **1 개 (1↔2)** — comm 부담 감소 |
| Stage 별 GPU 종류 | L40S / L40S / A100 / A100 | **L40S / L40S / A100 / A100** (동일) |

→ 모델 partition sweep 의미 그대로. 셋업 단순화 + cross-node 1 link 만 → throughput 안정성 ↑.

## Rank → 물리 GPU 매핑

torchrun sequential rank 할당 + nanotron TP-major group convention 이 자연스럽게 정합:

```
world rank 0–3   = NODE 0 (g6e.48xl) GPU 0–3   = TP-0  = Stage 0
world rank 4–7   = NODE 0 (g6e.48xl) GPU 4–7   = TP-1  = Stage 1
world rank 8–11  = NODE 1 (p4d)      GPU 0–3   = TP-2  = Stage 2
world rank 12–15 = NODE 1 (p4d)      GPU 4–7   = TP-3  = Stage 3
```

## 통신 토폴로지

| 종류 | 매체 | 비고 |
|---|---|---|
| TP AllReduce stage 0, 1 | g6e.48xl 의 PCIe Gen4 (L40S 간) | NVLink 없음, ~32 GB/s/dir |
| TP AllReduce stage 2, 3 | p4d 의 NVSwitch (A100 간) | ~600 GB/s/dir |
| PP 0→1 | g6e intra-node | 빠름 |
| PP 1→2 | cross-node ENA (또는 EFA) | 본 cluster 의 유일 cross-node link |
| PP 2→3 | p4d NVSwitch | 빠름 |

EFA 활성화 시 PP 1↔2 가 ENA TCP 대비 빠름. 단 SG 의 inbound + **outbound** self-ref 모두 필요
(이전 디버깅 결과). EFA 안 쓰면 ENA TCP fallback (NCCL_IB_DISABLE=1 + NCCL_NET_PLUGIN=none).

## 메모리 분포

PP=4 partition `[10,10,10,10]`, mbs=4, ga=16, s=8192, full recomp 가정. Stage 별 in-flight
microbatch 수 = `PP - stage_idx` (1F1B):

| Stage | in-flight | per-rank 메모리 (bbox) | 위치 |
|---|---|---|---|
| 0 | 4 | state ~18 GB + activation ~3 GB + framework ~1 GB ≈ **22 GB** | g6e (48 GB → 26 GB 헤드룸) |
| 1 | 3 | state ~15 GB + act ~2.5 GB + 1 GB ≈ **18 GB** | g6e (48 GB) |
| 2 | 2 | state ~15 GB + act ~1.7 GB + 1 GB ≈ **17 GB** | p4d (40 GB) |
| 3 | 1 | state ~18 GB + act ~0.8 GB + logits FP32 ~5 GB + 1 GB ≈ **25 GB** | p4d (40 GB → 15 GB 헤드룸) |

p4d 의 stage 3 가 가장 빡빡 (40 GB cap 의 62%). EFA 안 쓰면 NCCL 의 추가 buffer 가 더 잡혀 1–2 GB 더 소비.

## 전제

- `nodes.json` 에 2 노드 등록:
  ```json
  {
    "nodes": [
      {"node_rank": 0, "private_ip": "<G6E_IP>", "instance_type": "g6e.48xlarge"},
      {"node_rank": 1, "private_ip": "<P4D_IP>", "instance_type": "p4d.24xlarge"}
    ]
  }
  ```
- 양 노드의 nanotron 트리는 dev 가 rsync push.
- 양 노드의 `/opt/dlami/nvme/` 에 `alpaca_sft_local/` + `qwen-3-14b_nanotron/` 있어야 함.
  ```bash
  uv run python examples/heterogeneous/prepare_alpaca.py
  aws s3 sync s3://swj-nanotron-model/qwen-3-14b/nanotron/ /opt/dlami/nvme/qwen-3-14b_nanotron/
  ```

## Hyperparameter — 두 가지 (PP, TP) 셋업 비교

같은 cluster 위에서 두 가지 parallelism 설정 sweep. **mbs=2, train_steps=2** (1 warmup + 1 data).

### PP=4 TP=4 ([../../configs/qwen3_14b/alpaca_pp4_tp4.yaml](../../configs/qwen3_14b/alpaca_pp4_tp4.yaml))
```
pp=4, tp=4, partition=[10,10,10,10]   # 4-way split, 한 stage 가 한 노드 절반
mbs=2, ga=16, seq=8192, full_recomp=true
```
- Stage 0/1 → g6e (4 GPU each), Stage 2/3 → p4d (4 GPU each)
- Cross-node PP link: stage 1↔2 (1 link)
- Bubble: 3/19 = **16%** (PP=4 ga=16)

### PP=2 TP=8 ([../../configs/qwen3_14b/alpaca_pp2_tp8.yaml](../../configs/qwen3_14b/alpaca_pp2_tp8.yaml))
```
pp=2, tp=8, partition=[20,20]         # 2-way split, 한 stage 가 한 노드 전체
mbs=2, ga=16, seq=8192, full_recomp=true
```
- Stage 0 → g6e 전체 (8 L40S), Stage 1 → p4d 전체 (8 A100)
- Cross-node PP link: stage 0↔1 (1 link)
- Bubble: 1/17 = **5.9%** (PP=2 ga=16) — PP=4 보다 작음
- TP=8 이라 per-rank weight shard 가 절반 → state memory ~50%; activation 도 ½
- 단 TP=8 AllReduce message 더 큼 → intra-node bandwidth 한계 더 빨리 부딪힘

## Cross-instance EFA 미지원

AWS 공식: [EFA traffic between P4d/P4de and other instance types is not supported](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html). p4d (Nitro v3, RDMA-read only) 와 g6e (Nitro v4, RDMA read+write) 의 transport protocol mismatch — `RC: 265` (FI_ETRUNC) 발생 후 hang. Workaround: `NCCL_IB_DISABLE=1 + NCCL_NET_PLUGIN=none + FI_PROVIDER='^efa'` (launch_node{0,1}.sh 적용 상태). TCP/ENA fallback 으로 안정 동작하지만 throughput 은 EFA 대비 1/5~1/10. throughput 절대값보다 partition 변화에 따른 *상대* trend 가 의미 있음.

## 사용

```bash
# 1) sync (양 yaml 모두 push 됨)
bash examples/heterogeneous/clusters/g6e_48xl__p4d_24xl/sync.sh

# 2) PP=4 TP=4 sweep — N = 1..19 (총 19 점)
#    stage 0 = stage 1 = N (g6e), stage 2 = stage 3 = 20-N (p4d)
bash examples/heterogeneous/clusters/g6e_48xl__p4d_24xl/sweep_pp4_tp4.sh

# 3) PP=2 TP=8 sweep — stage0 ∈ {2,4,...,38} (default, 19 점, stride 2)
#    stage 0 = N (g6e), stage 1 = 40-N (p4d)
#    full sweep 1..39 은 STEP=1 START_STAGE0=1 END_STAGE0=39 override
bash examples/heterogeneous/clusters/g6e_48xl__p4d_24xl/sweep_pp2_tp8.sh

# 4) 단일 benchmark (config 별)
bash examples/heterogeneous/clusters/g6e_48xl__p4d_24xl/benchmark_single.sh \
    examples/heterogeneous/configs/qwen3_14b/alpaca_pp4_tp4.yaml [partition]
bash examples/heterogeneous/clusters/g6e_48xl__p4d_24xl/benchmark_single.sh \
    examples/heterogeneous/configs/qwen3_14b/alpaca_pp2_tp8.yaml [partition]
```

`benchmark_single.sh` 가 yaml 의 `pp:` / `tp:` field 자동 읽음. descriptor 형식:
`mbs{N}_ga{N}_seq{N}_pp{N}_tp{N}_recomp_split{partition}`. 두 sweep 의 결과가 같은
`runs/g6e_48xl__p4d_24xl/qwen3_14b/` 디렉토리에 PP/TP 별 다른 descriptor 로 분리되어 저장됨.

## 주의

- Master rendezvous = NODE 0 (g6e.48xl). NODE 1 는 NODE 0 의 :29500 가 열린 뒤 launch.
- NIC interface default: g6e=`enp39s0`, p4d=`ens32` — 다르면 `NCCL_SOCKET_IFNAME` env 로 override.
- `local_pg` deadlock fix ([src/nanotron/parallel/context.py](../../../../src/nanotron/parallel/context.py)) 적용 상태 — 이번 cluster 는 균등이라 trigger 안 되지만 fix 유지.
