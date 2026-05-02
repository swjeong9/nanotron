# Heterogeneous-PP Sweep Raw Data

각 sweep iteration 의 원본 측정 데이터 archive. plot_single.py 가 만든 stats.json
의 ground-truth 입력. `.gitignore` 로 git 에서 제외 — S3 의 `swj-nanotron-data`
에 sync.

## 디렉토리 구조

```
data/raw/<cluster>/<model>/<descriptor>/
  meta.json              — benchmark_single.sh 의 run metadata
  train_node0.log        — NODE 0 (master) 의 nanotron stdout/stderr (rank 0..3)
  train_node1.log        — NODE 1 (worker) 의 nanotron stdout/stderr (rank 4..7)
  dcgm_node0.txt         — NODE 0 의 dcgmi dmon (1Hz, 4 GPU 전체)
  dcgm_node1.txt         — NODE 1 의 dcgmi dmon (1Hz, 4 GPU 전체)
  nic_node0.txt          — NODE 0 의 /proc/net/dev sample (10Hz)
  nic_node1.txt          — NODE 1 의 /proc/net/dev sample (10Hz)
  nvidia_smi_node0.txt   — NODE 0 의 nvidia-smi memory.used (1Hz, 4 GPU comma-sep)
  nvidia_smi_node1.txt   — NODE 1 의 nvidia-smi memory.used (1Hz, 4 GPU comma-sep)
```

## Download (sweep 결과 재현용)

```bash
aws s3 sync s3://swj-nanotron-data/raw/ examples/heterogeneous/data/raw/
```

특정 cluster 만:
```bash
aws s3 sync s3://swj-nanotron-data/raw/g6_12xl__g5_12xl_pp2_tp4/ \
            examples/heterogeneous/data/raw/g6_12xl__g5_12xl_pp2_tp4/
```

## Upload (sweep 진행자 / 본인 만)

```bash
aws s3 sync examples/heterogeneous/data/raw/ s3://swj-nanotron-data/raw/
```

## Re-process (raw → stats.json + figures)

```bash
# 단일 iteration
uv run --no-project --with matplotlib python examples/heterogeneous/plot_single.py \
    --run-dir examples/heterogeneous/data/raw/<cluster>/<model>/<descriptor>/

# Sweep 전체 비교 그래프
uv run --no-project --with matplotlib python examples/heterogeneous/plot_compare.py \
    --cluster <cluster> --model <model>
```

## 크기 가이드

| 항목 | size |
|---|---:|
| descriptor 1 개 | ~3.6 MB |
| 한 cluster sweep (27 partitions) | ~95 MB |
| 한 model 의 모든 cluster | ~수백 MB |

## 주의

- `train_node*.log` 는 rank 별 PP/TP 좌표 (`[INFO|PP=0|TP=0]` 등) + `Memory usage`
  + step 경계 (`Before/After train_batch_iter`, `After training_step`) 를 모두 포함.
  plot_single.py 의 핵심 입력.
- DCGM `ts` 는 row index 가 아닌 GPU 0 tick 기반 (1Hz polling 의 sample_idx).
  `dcgm_text_to_jsonl.awk` 참고.
- raw 가 손상/누락되면 stats.json 의 어떤 값도 신뢰할 수 없음. 항상 raw 가
  ground truth.
