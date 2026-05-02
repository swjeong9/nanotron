# Cluster: `p4d_24xl_only` — A100×8 OOM demo

## 목적

같은 hyperparameter (mbs=4, ga=64, s=8192, full recomp, Qwen3-14B) 를
**A100×8 단일 p4d.24xlarge 노드** 에서 다양한 PP × TP=8 조합으로 돌려,
**OOM 이 나는 partition 들을 nvtop 화면으로 직접 시연**.

`g6e_12xl_x2__p4d_24xl_pp4_tp4` (heterogeneous 16 GPU) 가 fit 하는 같은 셋업이
A100×8 만으론 안 들어간다는 정성적 증거 수집.

## 구성

| | 값 |
|---|---|
| 노드 | p4d.24xlarge × 1 |
| GPU | 8× A100 40GB (NVSwitch) |
| World size | 8 |
| Hyperparameters | mbs=4, ga=64, s=8192, train_steps=5, full recomp, Qwen3-14B |

PP × TP = 8 조합 (균등 partition):

| PP | TP | partition | 한 stage 의 layer 수 | 메모/예상 |
|---|---|---|---|---|
| 1 | 8 | `[40]` | 40 | 단일 stage, 전 layer 한 번에 forward — TP=8 sharding 으로 state 분산 |
| 2 | 4 | `[20, 20]` | 20 | stage 1 의 logits buffer (FP32) 가 OOM 후보 |
| 4 | 2 | `[10, 10, 10, 10]` | 10 | stage 0 의 emb + state per-rank 가 무거움 |
| 8 | 1 | `[5]×8` | 5 | TP=1 이라 layer activation 안 줄어듦 — 가장 OOM 가능성 큼 |

이전 정량 추정 (16-GPU 셋업 분석 시):
- PP=4 TP=2: stage 0 state 만 40+ GB → **state alone OOM**
- PP=2 TP=4: stage 1 total 40.9 GB → **logits 으로 ~0.9 GB 초과 OOM**
- PP=1 TP=8: 38.4 GB → tight 하지만 fit (1.6 GB headroom)
- PP=8 TP=1: stage 0 act 13.4 GB + state 33 GB → **OOM**

→ 시연 예상: 4 점 중 PP=1 TP=8 만 통과, 나머지 3 점은 OOM.

## 사전 준비 (p4d 노드 위에서)

```bash
# 1) nanotron 트리 동기화 (dev 에서 push 또는 git clone)
ssh ubuntu@<p4d_ip>
git clone <repo>; cd nanotron; uv sync

# 2) Qwen3-14B nanotron checkpoint + tokenizer
aws s3 sync s3://swj-nanotron-model/qwen-3-14b/nanotron/ /opt/dlami/nvme/qwen-3-14b_nanotron/

# 3) Alpaca SFT data
uv run python examples/heterogeneous/prepare_alpaca.py
```

## 실행 (p4d 노드 위에서, 한 번에 한 조합씩)

별도 ssh terminal 에서 `nvtop` 띄워두고:

```bash
# Terminal A — nvtop 8 GPU 실시간 모니터
nvtop

# Terminal B — 한 조합 launch
cd /home/ubuntu/nanotron

bash examples/heterogeneous/clusters/p4d_24xl_only/launch.sh 1 8   # PP=1 TP=8
bash examples/heterogeneous/clusters/p4d_24xl_only/launch.sh 2 4   # PP=2 TP=4
bash examples/heterogeneous/clusters/p4d_24xl_only/launch.sh 4 2   # PP=4 TP=2
bash examples/heterogeneous/clusters/p4d_24xl_only/launch.sh 8 1   # PP=8 TP=1
```

각 run:
- ~30 초 안에 model build → optimizer init → 1 step forward 까지 진입.
- OOM 나는 조합은 `CUDA out of memory` 로 즉시 abort. nvtop 에서 어느 GPU 가
  먼저 천장 친지 시각적으로 캡처 가능 (40960 MiB cap).
- Fit 하는 조합은 5 step 까지 완주.
- log: `/opt/dlami/nvme/oom_demo_logs/pp{PP}_tp{TP}_mbs4_ga64_seq8192.log`.
- launch.sh 가 끝에 OOM 여부를 grep 으로 한 줄 요약.

## 변수

`launch.sh` 가 yaml 의 `pp`, `tp`, `pp_layer_partition` 만 sed 로 override.
나머지 (mbs=4, ga=64, s=8192, recompute_layer=true, Qwen3 14B model_config) 는
[alpaca_a100x8_oom_demo.yaml](../../configs/qwen3_14b/alpaca_a100x8_oom_demo.yaml) 에
박혀 있고 모든 4 조합에서 동일.

이 yaml 은 heterogeneous 16-GPU 의 `alpaca_pp4_tp4.yaml` 와 hyperparameter 가
동일 (model_config + tokens 섹션 byte-identical) — A100×8 vs A100×8+L40S×8 의 차이만이
OOM 의 원인 임을 보장.

## 관전 포인트 (nvtop)

- **State 단계**: 학습 시작 직후 (model build + optim init) 모든 GPU 메모리가
  계단식으로 차오름. PP=4 TP=2 / PP=8 TP=1 는 이 단계에서 이미 cap.
- **Activation peak**: 첫 forward+backward 진입하면 추가 ~5–10 GB spike.
  PP=2 TP=4 가 stage 1 의 logits buffer 로 cap.
- **OOM 직전 GPU 인덱스**: nvtop 으로 캡처 → "어느 stage 의 어느 rank 가 먼저 터졌는지"
  시연 자료로 활용.

## 정리

이 cluster 디렉토리는 **시연 전용** — 자동 sweep / 결과 파싱 / plot 없음.
모든 후처리는 user 가 nvtop 캡처 + log 로 직접.
