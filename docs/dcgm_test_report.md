# DCGM Test Report — Stage 1 (g6.xlarge / NVIDIA L4)

본 보고서는 [project_background.md](project_background.md) §6.4–6.5에 따라 DCGM의 16/18개 field 수집 가능 여부와 실제 거동을 Stage 1 개발 환경에서 검증한 결과다. 본 클러스터(A100×8 + L40S×8) 측정 단계 전에 **재실행 가능한 reference**로 활용한다.

## 1. 환경

| 항목 | 값 |
|---|---|
| 일자 | 2026-04-28 |
| 인스턴스 | AWS `g6.xlarge` |
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Kernel | Linux 6.17.0-1007-aws |
| GPU | NVIDIA L4 (sm_89, 23 GB) |
| Driver | 580.126.09 |
| Driver-exposed CUDA runtime | **13.0** (`nvidia-smi`의 "CUDA Version") |
| Active CUDA toolkit | **12.9** (`/usr/local/cuda → cuda-12.9/`, `nvcc 12.9.86`). 디스크에 12.6/12.8/12.9/13.0 공존, symlink만 12.9 |
| DCGM | 4.5.2-1 (이미 설치, `nvidia-dcgm.service` active, port 5555) |
| PyTorch (부하 테스트용) | 2.11.0+cu126 (uv venv) |

> **중요**: `nvidia-smi`가 보여주는 "CUDA Version: 13.0"은 driver의 **runtime expose 한도**이지 활성 toolkit이 아니다. dcgmproftester 같은 도구는 이 driver-exposed 값을 보므로 binary major version과 일치해야 한다 (§5.1 참조).

## 2. 단계별 결과 요약

| 단계 | 명령(요약) | 결과 |
|---|---|---|
| 1. service / GPU discovery | `systemctl is-active nvidia-dcgm`, `dcgmi discovery -l` | active, GPU 1개 (L4, BDF 0000:31:00.0) |
| 2. safe field smoke | `dcgmi dmon -e 155,156,150,100,101,203,204 -d 1000 -c 5` | exit 0, 5 sample, idle POWER 12.5W, TMPTR 28°C |
| 3. profile field hang check | 1001~1012 단독, `timeout 8s ... -d 1000 -c 3` | 1001~1010 OK, **1011/1012 exit 250** ("Feature not supported" — L4 NVLink 부재) |
| 4a. 18-field 통합 (1011/1012 포함) | `-e 155,...,1012` | **exit 250** — 1011/1012 unsupported 때문에 watch setup 전체 실패 |
| 4b. 16-field 통합 (NVLink 제외) | `-e 155,...,1010` | exit 0, 8 sample, dcgmi watch 활성 시 idle POWER 12.5W → 16.9W |
| 5. load-aware (BF16 matmul) | PyTorch BF16 8192² matmul + 16-field dmon | exit 0, **POWER 72W (TDP 100%), TENSO 0.94, SMACT 0.996, TMPTR 38→47°C** |
| 6. JSONL 변환 + pandas | `awk -f dcgm_text_to_jsonl.awk` + `pd.read_json(lines=True)` | 18 row × 18 col, dtype 자동 추론 OK |

## 3. Field-by-field 결과 ([project_background.md](project_background.md) §6.4의 18 field ID)

> Stage 5 PyTorch BF16 matmul sustained load 18 samples 평균 (`load_torch_v2.jsonl`, 정정된 ID set 사용).

| ID | Long name (Short) | 단독(Stage 3) | 통합(Stage 4) | 부하 mean | 비고 |
|---:|---|---|---|---:|---|
| 100 | sm_clock (SMCLK) | OK | OK | 1042 MHz | idle 210, load boost |
| 101 | memory_clock (MMCLK) | OK | OK | 6251 MHz | L4 default mem clock, BF16 matmul에선 변동 없음 |
| 150 | gpu_temp (TMPTR) | OK | OK | **42.7 °C** | idle 28°C → load steady ~47°C |
| 155 | power_usage (POWER) | OK | OK | **72.04 W** | L4 TDP 72W의 ~100% |
| 156 | total_energy_consumption (TOTEC) | OK | OK | 누적 mJ | idle Δ ≈ 12 J/s, load Δ ≈ 72 J/s |
| 203 | gpu_utilization (GPUTL) | OK | OK | 100 | load 100% |
| 204 | mem_copy_utilization (MCUTL) | OK | OK | 41 | |
| 1001 | gr_engine_active (GRACT) | OK | OK | 0.985 | (첫 sample 0.734 init 포함) |
| 1002 | sm_active (SMACT) | **OK (Issue #144 hang 미재현)** | OK | 0.996 | |
| 1003 | sm_occupancy (SMOCC) | OK | OK | 0.166 | single-stream matmul의 occupancy 한계, 실제 학습은 더 높을 것 |
| 1004 | tensor_active (TENSO) | OK | OK | **0.940** | Tensor Core 94% 활성 (BF16) |
| 1005 | dram_active (DRAMA) | OK | OK | 0.315 | matmul은 compute-bound이라 31% 합리적 |
| 1007 | fp32_active (FP32A) | OK | OK | 0.002 | BF16이라 FP32 거의 미사용 |
| 1008 | fp16_active (FP16A) | OK | OK | **0.000** | **BF16은 FP16A에 안 잡힘** — Tensor Core 활성은 TENSO로 봐야 함 |
| 1009 | pcie_tx_bytes (PCITX) | OK | OK | 753 KB/s | matmul은 PCIe 거의 안 씀 (첫 sample 9 MB는 init) |
| 1010 | pcie_rx_bytes (PCIRX) | OK | OK | 1.5 MB/s | |
| 1011 | nvlink_tx_bytes (NVLTX) | **exit 250** | (전체 거부) | — | L4에 NVLink 없음 |
| 1012 | nvlink_rx_bytes (NVLRX) | **exit 250** | (전체 거부) | — | L4에 NVLink 없음 |

## 4. 알려진 이슈

### 4.1 DCGM 4.5.2 `dcgmi dmon`은 `-j` (JSON) flag를 지원하지 않음 ⚠⚠

[project_background.md](project_background.md) §6.5의 초기 명령 가정 `dcgmi dmon ... -j > out.jsonl`은 4.5.2에서 **실패** (exit 1, "PARSE ERROR: Couldn't find match for argument -j"). `dcgmi dmon`은 text 출력만 지원한다 (`-j`는 `dcgmi stats` subcommand에서 "job id"를 의미하는 다른 인자). 이미 §6.5는 text+awk 변환으로 정정됨.

**대응**: text 출력을 awk로 JSONL 변환. helper script는 `/tmp/dcgm_test/dcgm_text_to_jsonl.awk` (본 보고서 §6.1에 전문 포함).

### 4.2 NVLink unsupported field가 있으면 전체 watch가 거부됨 ⚠⚠

`dcgmi dmon -e <list>` 는 list 중 **한 field라도 unsupported**면 atomic하게 watch setup을 거부 (exit 250, "Error setting watches. Result: -6: Feature not supported"). 일부 field만 partial하게 받지 않는다.

L4(NVLink 없음)에서 18-field set은 1011/1012 때문에 거부됨. **L40S 동일** — [project_background.md](project_background.md) §2가 L40S는 PCIe Gen4 only, NVLink 없음으로 명시. 따라서:

- **A100 stage**: 18-field OK (NVLink 있음)
- **L40S stage**: 1011/1012 제외한 **16-field만 가능**
- 본 클러스터 측정 시 **stage별 다른 명령**을 사용해야 함 (§6.2 참조)

### 4.3 Multi-client profile field watch 충돌

`dcgmproftester13` (또는 다른 DCGM client)와 `dcgmi dmon`이 **동시에 같은 profile field**를 watch하면 sampling이 실패해 dcgmi 측 값이 0으로 나온다 (Stage 5 첫 실험에서 재현).

| 시나리오 | POWER | TENSO | SMACT |
|---|---:|---:|---:|
| dcgmproftester13 + dcgmi dmon (충돌) | 33 W (실측) | **0.000** | **0.000** |
| PyTorch matmul + dcgmi dmon (단독) | 72 W | 0.940 | 0.996 |

**대응**: 본 실험은 PyTorch (학습) 단독 + dcgmi dmon 단독이라 충돌 없을 것. dcgmproftester는 별개 sanity 도구로만 사용하고, 학습 측정 중에는 절대 동시 실행 금지.

### 4.4 DCGM Issue #144 (sm_active/sm_occupancy hang) — L4에서 재현 안 됨 ✓

Stage 3에서 1002 (sm_active), 1003 (sm_occupancy)을 **단독으로** `timeout 8s -d 1000 -c 3`으로 실행 → 정상 exit 0, hang 없음. **L4에서는 Issue #144 없음**.

본 클러스터(A100/L40S)에서는 GPU 모델이 다르므로 동일 sanity check를 다시 수행해야 한다.

### 4.5 BF16 matmul은 FP16A로 안 잡힘

Stage 5 결과: TENSO 0.94 (Tensor Core 활성)인데도 FP16A는 0. **BF16 dtype 사용은 FP16A counter에 반영되지 않는다**. 본 연구의 BF16 학습 활용도 측정에는 **TENSO를 사용**해야 하며 FP16A는 보조 지표로도 의미 없음. (이미 [project_background.md](project_background.md) §6.4에 반영)

### 4.6 `dcgmi dmon` watch 활성 자체가 idle power +4W 추가

| 상태 | nvidia-smi idle | dcgmi dmon idle |
|---|---:|---:|
| Power | 12.5 W | 16.9 W |

DCGM이 profile field를 sampling하기 위해 GPU에서 hardware counter를 활성화하는 overhead. 본 측정은 모두 "DCGM 활성 상태" 기준이므로 일관성에는 영향 없으나, "idle baseline"을 정의할 때 이 값을 사용해야 한다 (nvidia-smi와 차이 인지).

### 4.7 Multiplexing — Stage 5 sustained load에서는 미관측

[project_background.md](project_background.md) §6.4 주의 1 ("1초 해상도라도 모든 field가 매 초 갱신되지는 않음")은 본 실험의 Stage 5 sustained load 8 sample에서는 **미관측**: 모든 16 field가 매 줄에 채워짐. 본 클러스터에서 GPU 16개 × 16 field = 256 watch 시 multiplexing이 발현될 수 있어 다시 검증 필요.

### 4.8 `dcgmproftester12` vs `dcgmproftester13`

`dcgmproftester12`는 driver-exposed CUDA runtime을 query해 자기 build version과 비교한다. 본 환경(driver 580.126.09 = CUDA 13.0 expose)에선 12 vs 13 mismatch로 **즉시 거부** (exit 244, "Wrong version of dcgmproftester is used. Expected Cuda version is 12. Installed Cuda version is 13."). **항상 `dcgmproftester13`을 사용**해야 한다. 이 동작은 `nvcc` toolkit이나 `/usr/local/cuda` symlink와 무관.

## 5. JSONL 변환 ([project_background.md](project_background.md) §6.5 후처리 호환)

### 5.1 변환 helper (awk)

`dcgmi dmon` text → JSONL 변환:

```awk
#!/usr/bin/awk -f
/^#Entity/ {
  ncols = 0
  for (i = 2; i <= NF; i++) { ncols++; hdr[ncols] = $i }
  next
}
/^ID/ { next }
/^GPU [0-9]+/ {
  ts = systime()
  printf "{\"ts\":%d,\"gpu_id\":%s", ts, $2
  for (i = 1; i <= ncols; i++) {
    v = $(i + 2)
    if (v ~ /^-?[0-9]+(\.[0-9]+)?$/) printf ",\"%s\":%s", hdr[i], v
    else printf ",\"%s\":\"%s\"", hdr[i], v
  }
  printf "}\n"
}
```

각 record는 `ts` (unix epoch, awk `systime()` — dmon이 자체 timestamp를 출력하지 않아 변환 시점 wall-clock 사용), `gpu_id`, 그리고 16개 short-name 키. `dmon -d 1000` (1초 간격) 호출에서 awk는 line-by-line 변환이므로 timestamp 정밀도는 ±1초. 학습 코드의 step별 timestamp와 join할 때 이 정도면 충분하지만, 더 정밀하게 필요하면 dmon stdout을 line buffered pipe로 받으며 receive 시점 timestamp를 사용해야 한다.

### 5.2 pandas 호환

```python
import pandas as pd
df = pd.read_json('dcgm.jsonl', lines=True)
# shape: (rows, 18), dtypes 자동 추론 (POWER float, TOTEC int 등)
```

**검증 결과**: 18 row × 18 col, dtype 자동 추론, `df.tail(18).describe()`로 sustained load 통계 즉시 산출 가능. **[project_background.md](project_background.md) §6.5의 후처리 코드는 변환 단계만 추가하면 그대로 사용 가능**.

## 6. 본 실험에서 사용할 명령

[project_background.md](project_background.md) §6.5에 stage별 정식 명령이 반영되어 있다. 본 절은 빠른 reference.

### 6.1 Stage 1 / Stage 2 / L40S 노드 — 16 field (NVLink 부재)

```bash
dcgmi dmon \
  -e 155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010 \
  -d 1000 \
  | tee dcgm.txt \
  | awk -f scripts/dcgm_text_to_jsonl.awk > dcgm.jsonl &
DCGM_PID=$!

# (학습 실행)
torchrun ... run_train.py

kill $DCGM_PID
```

### 6.2 A100 노드 — 18 field (NVLink 포함)

```bash
dcgmi dmon -i 0,1,2,3,4,5,6,7 \
  -e 155,156,150,100,101,203,204,1001,1002,1003,1004,1005,1007,1008,1009,1010,1011,1012 \
  -d 1000 | tee dcgm_a100.txt | awk -f scripts/dcgm_text_to_jsonl.awk > dcgm_a100.jsonl &
```

## 7. 본 클러스터에서의 차이 예상

| GPU | NVLink (1011/1012) | profile field sampling | TMPTR (150) |
|---|---|---|---|
| A100 (sm_80) | 작동 — 18-field OK | 정상 예상 (Issue #144 다시 sanity check) | 정상 |
| L40S (sm_89, NVLink 없음) | unsupported — 16-field 사용 | 정상 예상 (L4와 sm_89 동일 family) | 정상 (L4 검증 결과 기준) |

본 클러스터로 넘어갈 때 §3 표를 다시 채워서 보고서를 update해야 한다 (특히 1002/1003 hang 재현 여부, multi-GPU multiplexing 영향).

## 8. 재실행 체크리스트

본 클러스터에서 측정 시작 전 다음을 확인:

1. `systemctl is-active nvidia-dcgm` 모든 노드에서 active
2. `dcgmi discovery -l` 각 노드에서 GPU 8개 모두 가시
3. `timeout 8s dcgmi dmon -e 1002 -d 1000 -c 3` (Issue #144 재검증 — A100, L40S 둘 다)
4. 16-field 통합 (L40S) / 18-field 통합 (A100) 각각 short sample (`-c 5`)
5. 짧은 학습 워크로드(예: 100 step) 위에서 metric이 살아있는지 확인 — TENSO와 POWER가 idle 대비 명확히 상승해야 함
