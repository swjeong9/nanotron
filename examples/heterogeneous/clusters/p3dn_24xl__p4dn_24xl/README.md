# Cluster: `p3dn_24xl__p4dn_24xl`

## 구성

2-node heterogeneous cluster (p3dn.24xlarge + p4dn.24xlarge). 16 GPU.

| 노드 | 인스턴스 | GPU | VRAM | Tensor Core | 인터커넥트 |
|---|---|---|---|---|---|
| NODE 0 | p3dn.24xlarge | 8× **V100** SXM2 | 32 GB | FP16 only (BF16 미지원) | NVLink (intra-node) |
| NODE 1 | p4dn.24xlarge | 8× A100 SXM4 | 40 GB | FP16 + BF16 | NVSwitch (intra-node) |

총 16 GPU.

## g6e+p4d cluster 와의 차이

| | g6e_48xl + p4d_24xl | **p3dn + p4dn** |
|---|---|---|
| NODE 0 GPU | L40S (Ada, PCIe Gen4) | **V100 (Volta, NVLink)** |
| NODE 0 VRAM | 48 GB | **32 GB** (적음 — fit range 좁아짐) |
| NODE 0 intra-node interconnect | PCIe Gen4 (~32 GB/s) | **NVLink** (~300 GB/s, 10× 빠름) |
| BF16 support | 양쪽 OK | **NODE 0 (V100) 미지원 → FP16 cluster** |

핵심 expectation: NODE 0 (p3dn V100) 의 **NVLink 가 PCIe 대비 TP AllReduce 훨씬 빠를 것** → TP=8 의 intra-node bandwidth bottleneck 해소. 단 V100 32GB 라 메모리 빡빡 (recompute_layer + mbs 줄이기 필요).

## 통신 토폴로지

| 종류 | 매체 | 비고 |
|---|---|---|
| TP AllReduce stage 0 | p3dn 의 **NVLink** (V100 간) | ~300 GB/s/dir |
| TP AllReduce stage 1 | p4dn 의 **NVSwitch** (A100 간) | ~600 GB/s/dir |
| PP cross-node (stage 0→1) | TCP/ENA fallback | EFA cross-instance 미지원 (AWS 공식) |

**EFA 비활성화 강제** (g6e+p4d 의 디버깅 결과 동일): NCCL_IB_DISABLE=1 + NCCL_NET_PLUGIN=none + FI_PROVIDER='^efa'.

## Hyperparameter (PP=2 TP=8 출발)

[`../../configs/qwen3_14b/alpaca_pp2_tp8_fp16.yaml`](../../configs/qwen3_14b/alpaca_pp2_tp8_fp16.yaml):
```
pp=2, tp=8, partition=[20,20]    # 균등; sweep 시 sed override
mbs=2, ga=16, seq=8192, train_steps=2, recompute_layer=true
dtype=float16                     # V100 BF16 미지원
```
- gbs = 32 sequences = 262,144 tokens / step (g6e+p4d 의 PP=2 와 동일)
- Stage 0 = p3dn 전체 (8 V100), Stage 1 = p4dn 전체 (8 A100)
- Cross-node PP link: stage 0↔1 (1 link)

## 전제

- `nodes.json` 에 2 노드 등록:
  ```json
  {
    "nodes": [
      {"node_rank": 0, "private_ip": "<P3DN_IP>", "instance_type": "p3dn.24xlarge"},
      {"node_rank": 1, "private_ip": "<P4DN_IP>", "instance_type": "p4dn.24xlarge"}
    ]
  }
  ```
- 양 노드의 nanotron 트리는 dev 가 rsync push (sync.sh).
- 양 노드의 `/opt/dlami/nvme/` 에 `alpaca_sft_local/` + `qwen-3-14b_nanotron/` 있어야 함.
  ```bash
  uv run python examples/heterogeneous/prepare_alpaca.py
  aws s3 sync s3://swj-nanotron-model/qwen-3-14b/nanotron/ /opt/dlami/nvme/qwen-3-14b_nanotron/
  ```

## 사용

```bash
# 1) sync — 양 worker 에 코드 + config 배포
bash examples/heterogeneous/clusters/p3dn_24xl__p4dn_24xl/sync.sh

# 2) Single-point verify (균등 [20, 20])
bash examples/heterogeneous/clusters/p3dn_24xl__p4dn_24xl/benchmark_single.sh \
    examples/heterogeneous/configs/qwen3_14b/alpaca_pp2_tp8_fp16.yaml 20-20

# 3) PP=2 TP=8 sweep
bash examples/heterogeneous/clusters/p3dn_24xl__p4dn_24xl/sweep_pp2_tp8.sh
```

## 주의

- Master rendezvous = NODE 0 (p3dn). NODE 1 는 NODE 0 의 :29500 가 열린 뒤 launch.
- NIC interface default: p3dn=`ens5` (가정), p4dn=`ens32` — 인스턴스 launch 후 `ip route get 1` 으로 확인 후 `NCCL_SOCKET_IFNAME` env 로 override.
- V100 32GB 는 stage 0 의 메모리 빡빡 — partition heavy stage 0 (예: 30+ layer) OOM 가능성.
- FP16 NaN 위험 — clip_grad=1.0 + LR 작게 (2e-5) 으로 기본 셋업.
