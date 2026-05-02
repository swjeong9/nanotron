# DCGM 4.5.2 의 ``dcgmi dmon`` text 출력을 JSONL 으로 변환.
#
# 입력 schema (dcgm_dmon_wrap.sh 가 wallclock prefix 를 line 별 부착):
#   첫 줄  : "<ts> #Entity   POWER   TOTEC   TMPTR  ...   PCIRX"   (wallclock + header)
#   두 번째: "<ts> ID         W      mJ      C    ..."             (단위 — skip)
#   이후    : "<ts> GPU 0     16.609  256...  42  ..."              (data row)
#   매 ~25 sample 마다 두 헤더 라인이 다시 나타남 (dcgmi 의 visual reset).
#
# Wallclock prefix:
#   dcgm_dmon_wrap.sh 가 모든 라인 앞에 epoch.ms (e.g. ``1777708107.135``) 을
#   prepend. 각 data row 의 ``ts`` 는 이 prefix 의 wallclock 그대로 사용.
#
# 핵심 주의: data 행은 entity ID 가 ``"GPU 0"`` 처럼 토큰 2 개라 NF 가 헤더보다
# 1 큼. wallclock prefix 추가로 모든 라인의 NF 가 +1.
#
# 출력: 각 data row 마다 1줄 JSON.
#
# 시간 처리: dcgm_dmon_wrap.sh 가 wallclock 을 line 별 부착해서 plot 측 alignment
# 가 trivial (그냥 unix epoch 그대로 쓰면 됨). dcgmi 자체 emit rate 가 -d 1000
# 이어도 실제 ~3 Hz 인 issue 회피.
#
# Backwards-compat: prefix 가 없는 (옛 데이터) 경우, ``$1`` 이 숫자가 아니면 prefix
# 없는 fallback path — sample_idx 기반 ts 사용 (1Hz 가정, 정확하지 않을 수 있음).
#
# Usage:
#   awk -f dcgm_text_to_jsonl.awk dcgm_node0.txt > dcgm_node0.jsonl

BEGIN {
    sample = 0
}

# Detect wallclock prefix: first line의 첫 토큰이 numeric 이면 prefix 모드.
# (ts 자체가 음수 / 매우 큰 정수 가능하니 정규식 매치)
NR == 1 {
    if ($1 ~ /^[0-9]+(\.[0-9]+)?$/) {
        has_prefix = 1
    } else {
        has_prefix = 0
    }
}

# Header: "#Entity" 또는 prefix 모드에선 "<ts> #Entity"
($0 ~ /(^|[ \t])#Entity/) {
    # field offset: prefix 모드면 +1 (첫 토큰은 ts).
    base = has_prefix ? 2 : 1
    n_fields = NF - base                    # "#Entity" 제외한 metric 개수
    for (i = 1; i <= n_fields; i++) {
        field[i] = $(i + base)              # field[1]=POWER, field[2]=TOTEC, ...
    }
    next
}

# 단위 행: "ID  W  mJ  ..." 또는 "<ts> ID  ..."
(($0 ~ /^ID/) || (has_prefix && $0 ~ /^[0-9.]+[ \t]+ID[ \t]/)) {
    next
}

# Data row: "GPU 0  ...  값들" 또는 "<ts> GPU 0  ...  값들"
(($0 ~ /^GPU [0-9]+/) || (has_prefix && $0 ~ /^[0-9.]+[ \t]+GPU [0-9]+/)) {
    if (has_prefix) {
        ts_str = $1                          # 원본 문자열 (정밀도 보존)
        gpu_idx = 3                          # $1=ts, $2="GPU", $3=<gpu_id>
        val_offset = 4                       # values start at $4
        entity = $2 " " $3                   # "GPU 0"
        gpu = $3 + 0
    } else {
        gpu_idx = 2
        val_offset = 3
        entity = $1 " " $2
        gpu = $2 + 0
        if (gpu == 0) sample++
        ts_str = (sample - 1)                # fallback: 1Hz 가정 (정수)
    }
    printf("{\"ts\":%s,\"entity\":\"%s\"", ts_str, entity)
    for (i = 1; i <= n_fields; i++) {
        v = $(i + val_offset - 1)
        if (v ~ /^-?[0-9]+(\.[0-9]+)?$/) {
            printf(",\"%s\":%s", field[i], v)
        } else {
            # dcgmi 가 "N/A" 등 non-numeric emit 하는 경우 (GPU 상태 따라 일부 metric).
            # downstream plot_single 의 statistics.mean 이 string 처리 못 하니 0 으로 치환.
            printf(",\"%s\":0", field[i])
        }
    }
    printf("}\n")
}
