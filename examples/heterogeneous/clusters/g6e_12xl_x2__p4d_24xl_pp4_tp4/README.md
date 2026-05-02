# Cluster: `g6e_12xl_x2__p4d_24xl_pp4_tp4`

## 구성

3-node heterogeneous cluster, 16 GPU 합계, **PP=4 × TP=4 × DP=1**.

| 노드 | 인스턴스 | GPU | VRAM | TP group | PP stage | 노드 내 GPU 통신 |
|---|---|---:|---:|---|---|---|
| NODE 0 | g6e.12xlarge | 4× L40S | 48 GB | TP-0 (rank 0–3) | Stage 0 | PCIe Gen4 |
| NODE 1 | g6e.12xlarge | 4× L40S | 48 GB | TP-1 (rank 4–7) | Stage 1 | PCIe Gen4 |
| NODE 2 | p4d.24xlarge | 8× A100 40GB | 40 GB | TP-2 (rank 8–11), TP-3 (rank 12–15) | Stage 2 + Stage 3 | NVSwitch |

총 16 GPU. 각 TP group (4 rank) 이 정확히 하나의 노드 (또는 p4d 의 절반) 에 묶여
intra-node AllReduce 만 사용 — cross-node TP traffic 없음.

## Rank → 물리 GPU 매핑

torchrun sequential rank 할당 + nanotron TP-major group convention 정합:

```
world rank 0–3   = NODE 0 (g6e #1) GPU 0–3   = TP-0  = Stage 0
world rank 4–7   = NODE 1 (g6e #2) GPU 0–3   = TP-1  = Stage 1
world rank 8–11  = NODE 2 (p4d) GPU 0–3      = TP-2  = Stage 2
world rank 12–15 = NODE 2 (p4d) GPU 4–7      = TP-3  = Stage 3
```

## 통신 토폴로지

| 종류 | 매체 | 비고 |
|---|---|---|
| TP AllReduce stage 0, 1 | g6e PCIe Gen4 (~32 GB/s/dir) | NVLink 없음 — TP 가 throughput bottleneck 후보 |
| TP AllReduce stage 2, 3 | p4d NVSwitch (~600 GB/s/dir) | 빠름 |
| PP 0→1 | cross-node (g6e #1 ↔ g6e #2) | ENA |
| PP 1→2 | cross-instance (g6e #2 ↔ p4d) | ENA / EFA mixed |
| PP 2→3 | intra-p4d NVLink | 빠름 |

## 전제

- `nodes.json` 에 3 노드 등록 (이 cluster 에 한해 instance_type 으로 식별):
  ```json
  {
    "nodes": [
      {"private_ip": "<G6E_1>", "instance_type": "g6e.12xlarge", "node_rank": 0},
      {"private_ip": "<G6E_2>", "instance_type": "g6e.12xlarge", "node_rank": 1},
      {"private_ip": "<P4D>",   "instance_type": "p4d.24xlarge", "node_rank": 2}
    ]
  }
  ```
  (g6e 가 두 대라 instance_type 만으론 식별 못 함 → `node_rank` field 명시.)

- 모든 worker 노드의 nanotron 트리는 dev 가 rsync push (sync.sh 처리).
- 모든 노드에 `/opt/dlami/nvme/alpaca_sft_local/` (preprocess 된 dataset) 과
  `/opt/dlami/nvme/qwen-3-14b_nanotron/` (변환된 checkpoint + tokenizer) 있어야 함.
  ```bash
  # dev 또는 한 worker 에서:
  uv run python examples/heterogeneous/prepare_alpaca.py
  ARCH=qwen3 bash examples/heterogeneous/convert_and_upload.sh Qwen/Qwen3-14B qwen-3-14b
  # 그 다음 worker 들에서:
  aws s3 sync s3://swj-nanotron-model/qwen-3-14b/ /opt/dlami/nvme/qwen-3-14b_nanotron/
  ```

## Hyperparameter (초기 설정 — [alpaca_pp4_tp4.yaml](../../configs/qwen3_14b/alpaca_pp4_tp4.yaml))

```
mbs=4, ga=64, seq=8192, train_steps=5, partition=[10,10,10,10], full_recomp=true
```

→ in-flight stage 0 = PP × b × s = 4 × 4 × 8192 = 131k token-equivalent activation.
→ stage 3 logits FP32 buffer = s·b·V/TP·4 = 8192·4·151936/4·4 = 4.98 GB (rank).
→ 추정 cluster total ≈ 335 GB (cluster budget 352 GB, 17 GB 헤드룸).

## 사용

```bash
# 1) sync (3 node 모두)
bash examples/heterogeneous/clusters/g6e_12xl_x2__p4d_24xl_pp4_tp4/sync.sh

# 2) single benchmark (default partition [10,10,10,10])
bash examples/heterogeneous/clusters/g6e_12xl_x2__p4d_24xl_pp4_tp4/benchmark_single.sh

# 3) partition sweep
bash examples/heterogeneous/clusters/g6e_12xl_x2__p4d_24xl_pp4_tp4/sweep_partitions.sh
```

## 주의

- PP=4 의 stage 매핑은 **TP-major rank 할당** + **각 노드의 nproc_per_node 정확 일치** 에 의존.
  g6e 노드는 `--nproc_per_node=4`, p4d 는 `--nproc_per_node=8` 로 launch 해야 위
  rank → 물리 GPU 정합이 유지됨.
- Master rendezvous endpoint = NODE 0 (g6e #1). 다른 노드는 NODE 0 :29500 가 열린 뒤 launch.
- DCGM dmon, nvidia-smi 샘플러는 3 노드 모두에서 background 로 띄움.
