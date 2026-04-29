# DCGM 4.5.2 의 ``dcgmi dmon`` text 출력을 JSONL 으로 변환.
#
# 입력 schema (dcgmi dmon -e <fields> -d 1000 의 stdout):
#   첫 줄  : "#Entity   POWER   TOTEC   TMPTR  ...   PCIRX"   (NF = 1 + n_fields)
#   두 번째: "ID         W      mJ      C    ..."             (단위 — skip)
#   이후    : "GPU 0     16.609  256...  42  ..."              (NF = 2 + n_fields)
#   매 ~25 sample 마다 두 헤더 라인이 다시 나타남 (dcgmi 의 visual reset).
#
# 핵심 주의: data 행은 entity ID 가 ``"GPU 0"`` 처럼 토큰 2 개라 NF 가 헤더보다
# 1 큼. 따라서 헤더의 i 번째 필드는 data 의 i+1 번째 값과 매칭.
#
# 출력: 각 샘플마다 1줄 JSON.
#
# 시간 처리: DCGM dmon text 자체에 timestamp 가 없고, awk 의 ``systime()`` 은
# 파일을 일괄 파싱하는 시점 (학습 종료 후) 이라 모든 row 가 거의 같은 ts 를
#받게 되어 시계열이 0 폭 vertical line 이 됨. 본 sweep 은 ``dcgmi dmon -d
# 1000`` (1Hz) 이므로 ``GPU 0`` 데이터 row 가 1 개 추가될 때마다 시간이 1 초
# 흘렀다고 간주: ``ts = row_index`` (정수 초). 같은 sweep 의 양 노드는 거의
# 같은 wallclock 에 dmon 시작 → 0 부터의 상대 시간으로 양 노드 비교 가능.
#
# Usage:
#   awk -f dcgm_text_to_jsonl.awk dcgm_node0_ga4.txt > dcgm_node0_ga4.jsonl

BEGIN {
    row = 0
}

/^#Entity/ {
    n_fields = NF - 1                       # "#Entity" 제외한 metric 개수
    for (i = 1; i <= n_fields; i++) {
        field[i] = $(i + 1)                 # field[1]=POWER, field[2]=TOTEC, ...
    }
    next
}

/^ID/ {
    next                                    # 단위 행은 무시
}

/^GPU [0-9]+/ {
    entity = $1 " " $2                      # "GPU 0"
    printf("{\"ts\":%d,\"entity\":\"%s\"", row, entity)
    for (i = 1; i <= n_fields; i++) {
        v = $(i + 2)                        # data 의 i+2 번째 = field[i] 의 값
        if (v ~ /^-?[0-9]+(\.[0-9]+)?$/) {
            printf(",\"%s\":%s", field[i], v)
        } else {
            printf(",\"%s\":\"%s\"", field[i], v)
        }
    }
    printf("}\n")
    row++
}
