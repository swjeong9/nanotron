#!/usr/bin/env bash
# dcgmi dmon wrapper — 각 출력 라인 앞에 wallclock epoch (sec.ms) prefix 를 붙임.
#
# Why: dcgmi 가 -d 1000 (1 Hz) 으로 켜도 실제로는 ~3 Hz 로 line emit (확인됨).
# downstream awk 가 sample_idx 로 ts 를 추정하면 시간축이 3× 압축됨. 이 wrapper 로
# 실제 wallclock 을 line 별 prepend → awk 가 정확한 ts 사용.
#
# Sample-grouping: 같은 dcgmi sample 안의 8 개 GPU line 들이 동일 ts 를 갖도록
# 한다 (``GPU 0`` 라인이 새 sample 시작 → 이때만 wallclock 갱신, 다음 ``GPU 0``
# 직전까지는 같은 ts 유지). plot 의 ``_by_ts`` grouping 이 정확히 동작.
#
# Usage: dcgm_dmon_wrap.sh <fields_csv> [<delay_ms>]
#   e.g. dcgm_dmon_wrap.sh 155,156,150 1000

set -euo pipefail
FIELDS="$1"
DELAY_MS="${2:-1000}"

exec python3 -u -c "
import subprocess, sys, time
proc = subprocess.Popen(
    ['dcgmi', 'dmon', '-e', '$FIELDS', '-d', '$DELAY_MS'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)
sample_ts = None
for line in iter(proc.stdout.readline, ''):
    line = line.rstrip()
    if line.startswith('GPU 0 '):
        # 새 dcgmi sample 의 시작 → wallclock 갱신
        sample_ts = time.time()
    elif sample_ts is None:
        # 파일 시작 시점의 header / unit / non-data 라인 — 임시 ts
        sample_ts = time.time()
    print(f'{sample_ts:.3f} {line}', flush=True)
"
