# Cluster: `g6_12xl__g5_12xl_pp2_tp4`

## 구성

| 노드 | 인스턴스 | GPU | VRAM | 역할 (PP=2 시) |
|---|---|---:|---:|---|
| NODE 0 | g6.12xlarge | 4× L4 | 24 GB | stage 0 (intra-node TP=4) |
| NODE 1 | g5.12xlarge | 4× A10G | 24 GB | stage 1 (intra-node TP=4) |

총 **8 GPU** = TP=4 × PP=2 × DP=1.

dev 노드는 별도 (이 cluster 의 worker 가 아니어도 됨, ssh 로 launch 만).

## 전제

- `nodes.json` 에 노드 등록:
  ```json
  {
    "g6_12xl_node": {"ip": "<NODE0_IP>", "gpu": "L4", "count": 4, "instance": "g6.12xlarge"},
    "g5_12xl_node": {"ip": "<NODE1_IP>", "gpu": "A10G", "count": 4, "instance": "g5.12xlarge"}
  }
  ```
- 양 노드의 nanotron 트리는 dev 가 rsync 로 push (sync.sh 가 처리).
- 양 노드에 `/opt/dlami/nvme/alpaca_sft_local` (preprocess 된 dataset) 과
  `/opt/dlami/nvme/llama32_3b_nanotron/` (변환된 checkpoint) 가 있어야 함.
  ```bash
  # NODE 0 (또는 dev → NODE) 에서 한 번 실행:
  uv run python examples/heterogeneous/prepare_alpaca.py
  aws s3 sync s3://swj-nanotron-model/llama-3.2-3b/nanotron/ /opt/dlami/nvme/llama32_3b_nanotron/
  ```

## NCCL / 네트워크 인터페이스

g6.12xlarge / g5.12xlarge 의 primary NIC 이름이 다를 수 있음:
- 보통 `ens6` 또는 `eth0`. 확인:
  ```bash
  ip -br link | grep -v lo
  ```
- launch_node{0,1}.sh 의 `NCCL_SOCKET_IFNAME` 을 확인된 이름으로 수정.
- EFA 가능하면 `NCCL_IB_DISABLE=0` + EFA 프로빙. 본 cluster 의 EFA 지원 여부는 인스턴스 셋업 시 확인.

## 실행

```bash
# dev 에서, sync 한 번
bash clusters/g6_12xl__g5_12xl_pp2_tp4/sync.sh

# benchmark 한 번
bash clusters/g6_12xl__g5_12xl_pp2_tp4/benchmark_single.sh

# partition sweep (예: 12-16 ~ 16-12 균형 ± 4)
START_STAGE0=10 END_STAGE0=18 \
  bash clusters/g6_12xl__g5_12xl_pp2_tp4/sweep_partitions.sh
```

## 메모리 모델 (TP=4, mbs=2, seq=1024)

per layer per mb activation = `s · b · h · (10 + 24/t)` = 1024 × 2 × 3072 × 16 = **101 MB / GPU**

| 항 | Stage 0 (L4 24GB) | Stage 1 (A10G 24GB) |
|---|---:|---:|
| State (450M × 14 bytes) | 6.3 GB | 6.3 GB |
| Activation (14 layers × in-flight × 101 MB) | 14 × 2 × 0.101 = **2.8 GB** | 14 × 1 × 0.101 = **1.4 GB** |
| BF16 grad temp | 1 GB | 1 GB |
| lm_head logits (2 × 1024 × 128256 × 4 fp32) | — | 1 GB |
| Overhead (PyTorch / NCCL) | 3 GB | 3 GB |
| **Total** | **~13 GB** | **~13 GB** |

24 GB 의 ~22 GB usable cap 대비 **9 GB headroom 양쪽 모두**. 비대칭 partition (예 [10, 18], [18, 10]) 도 충분히 fit 예상.
