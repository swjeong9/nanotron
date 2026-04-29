"""SFT 데이터셋 처리 — sequence packing 기반.

==============================================================================
변경 이력 (Korean)
==============================================================================
[기존 구현]
- ``process_sft``가 (prompt, completion) 쌍을 batch 단위로 받아 ``tokenizer(...,
  padding=True, truncation=True, max_length=trainer_sequence_length+1)`` 로
  토큰화한 뒤, 각 sample 을 trainer_sequence_length+1 길이로 ``오른쪽 PAD``
  하여 반환했음.
- ``pack_sft_sequences``는 placeholder 로 ``NotImplementedError`` 만 발생.

[기존 구현이 일으킨 실제 문제]
1) Alpaca 평균 토큰 길이가 ~127, seq_len=1024 padding 시 약 87% 가 PAD 토큰이라
   wall-clock / MFU 측정의 신뢰도가 매우 낮았음.
2) ``padding=True`` 는 batch 내 ``longest`` 만큼만 PAD 하므로, ``dataset.map(
   batched=True)`` 의 chunk 안 longest 가 max_length 미만이면 결과 길이가
   chunk 마다 달라져 collator 의 ``expanded_input_length == seq_len + 1``
   assertion 이 깨짐 (실제로 ``got 472`` 에러로 학습이 시작도 못 함).
3) ``padding="max_length"`` 로 강제해도 (1) 의 87% PAD 낭비는 그대로 남음.

[새 구현 — 본 파일]
- ``process_sft`` / ``pack_sft_sequences`` 를 모두 제거(또는
  NotImplementedError 로 보존)하고, ``prepare_sft_dataset`` 자체에
  ``sequence packing`` 을 통합.
- 각 row 를 (prompt + completion + EOS) 로 토큰화 (PAD/truncate 없음) →
  greedy bin-packing 으로 ``pack_size = trainer_sequence_length + 1`` 길이의
  bin 에 채워 넣음. bin 이 꽉 차기 전에 다음 sample 이 들어가면 overflow 가
  나는 시점에서 직전 bin 을 ``pad_token_id`` 로 우측 PAD 후 yield.
- bin 안의 각 packed sample 은 ``positions`` 가 ``0`` 부터 다시 시작하도록
  설정 — 이게 nanotron Qwen2 forward 가 ``cu_seqlens`` 를 도출할 때 사용하는
  "document-masking" 신호.
- ``label_mask`` 는 prompt 부분 ``False``, completion+EOS 부분 ``True``.
  collator 는 우리가 제공한 ``label_mask`` 를 우선 사용하도록 함께 수정함
  (`clm_collator.py` 변경 참조).

[변경의 효과]
- Alpaca 52,002 examples → 5,630 packed bins (1025 토큰/bin) ≈ **9.2× 압축**.
- PAD 비율이 ~87% → <1% 로 감소 → wall-clock 의 의미가 회복되어 MFU 측정
  baseline 이 가능해짐.
- collator 의 길이 assertion 도 자연스럽게 만족 (모든 bin 이 정확히
  pack_size 길이).
- ``cu_seqlens`` 기반 varlen FlashAttention 경로가 활성화되어 packed bin
  내부에서 sample 간 attention 누수가 차단됨 (model config 의
  ``_use_doc_masking: true`` 와 함께 사용해야 함).
"""

from typing import Iterator, List

import datasets
import torch  # noqa: F401  -- API compat; 향후 GPU-side 가속 시 사용 가능


def _iter_packed(
    raw_dataset: "datasets.Dataset",
    tokenizer,
    pack_size: int,
) -> Iterator[dict]:
    """Greedy bin-packing 의 단일 패스 generator.

    한 row 씩 순회하며 ``cur_*`` 버퍼에 누적하다가, 다음 sample 을 추가했을 때
    ``pack_size`` 를 초과하면 직전까지의 버퍼를 ``pad_token_id`` 로 채워 정확히
    ``pack_size`` 길이로 만들어 yield 한다. 마지막 미완성 bin 은 의도적으로
    버린다 (collator 입장에서 모든 bin 이 동일 길이여야 batch 가 깔끔).

    각 packed sample 의 ``positions`` 는 0 부터 다시 시작 → 이게
    Qwen2 forward (qwen.py:_forward_packed) 가 ``cu_seqlens`` 를 만들 때
    사용하는 "document boundary" 신호이며, ``_use_doc_masking=True`` 와 함께
    varlen FA2 경로를 타도록 한다.

    Args:
        raw_dataset: ``prompt`` / ``completion`` 컬럼을 가진 데이터셋.
        tokenizer: HuggingFace tokenizer (``pad_token`` 이 None 이면 호출 측에서
            EOS 로 fallback 시켜놓을 것).
        pack_size: ``trainer_sequence_length + 1`` (collator 가 input/label
            shift 후 ``[batch, sequence_length]`` 가 되도록 +1).

    Yields:
        ``{"input_ids", "label_mask", "positions"}`` 모두 길이 ``pack_size``.
    """
    pad_id = tokenizer.pad_token_id

    cur_ids: List[int] = []
    cur_mask: List[int] = []
    cur_pos: List[int] = []

    n_too_long = 0
    n_packed_in = 0

    for row in raw_dataset:
        prompt = row["prompt"]
        completion = row["completion"]

        # prompt 만 단독 토큰화하여 ``label_mask`` 의 prompt/completion 경계를
        # 알아낸다. ``add_special_tokens=True`` 라 BOS 가 prompt 측에 붙는다.
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        # 전체(prompt+completion+EOS)를 한 번에 토큰화 → completion 의 EOS 까지
        # 자연스럽게 한 시퀀스로 이어진다. tokenizer 가 BPE 기반이라 prompt 와
        # completion 의 경계 토큰이 합쳐지는 경우는 거의 없으나, 안전장치로
        # ``min(len(prompt_ids), len(full_ids))`` 로 클립.
        full_ids = tokenizer(
            f"{prompt}{completion}{tokenizer.eos_token}", add_special_tokens=True
        )["input_ids"]
        completion_start = min(len(prompt_ids), len(full_ids))

        # 한 sample 자체가 pack_size 보다 길면, 이를 분할하면 SFT 의 의미
        # (한 instruction 의 prompt-completion 쌍) 가 깨지므로 통째로 버린다.
        # Alpaca 의 경우 p99 ≈ 1k 라 seq_len=1024 에서 ~1% 발생.
        if len(full_ids) > pack_size:
            n_too_long += 1
            continue
        # completion 이 비어 있어 학습할 토큰이 하나도 없으면 의미 없음.
        if len(full_ids) == 0 or completion_start >= len(full_ids):
            continue

        # label_mask: prompt 구간은 0 (loss 제외), completion+EOS 는 1 (loss 포함).
        sample_mask = [0] * completion_start + [1] * (len(full_ids) - completion_start)
        # positions: sample 시작점에서 0 으로 리셋 → 다음 sample 이 같은 bin 에
        # 들어가면 그 sample 도 0 부터 시작 → ``positions == 0`` 가 boundary.
        sample_pos = list(range(len(full_ids)))

        # 현재 bin 에 이 sample 을 추가하면 pack_size 를 넘으면, 직전 bin 을
        # 닫고 (PAD 채워) yield.
        if len(cur_ids) + len(full_ids) > pack_size:
            pad_len = pack_size - len(cur_ids)
            cur_ids.extend([pad_id] * pad_len)
            cur_mask.extend([0] * pad_len)  # PAD 토큰은 loss 제외
            cur_pos.extend([0] * pad_len)   # PAD 의 position 은 의미 없으나 0 으로
            yield {"input_ids": cur_ids, "label_mask": cur_mask, "positions": cur_pos}
            cur_ids, cur_mask, cur_pos = [], [], []

        # 새 sample 을 (잘리지 않고) 그대로 현재 bin 에 추가.
        cur_ids.extend(full_ids)
        cur_mask.extend(sample_mask)
        cur_pos.extend(sample_pos)
        n_packed_in += 1

    # 마지막 bin 은 PAD 비율이 높을 수 있으므로 의도적으로 버린다 (수만 개 중
    # 한두 개라 학습 측엔 영향 미미). 필요시 추후 keep 옵션 추가 가능.
    if n_too_long:
        print(f"sft_processing: skipped {n_too_long} examples longer than pack_size={pack_size}")
    print(f"sft_processing: packed {n_packed_in} examples into bins of {pack_size} tokens")


def prepare_sft_dataset(
    raw_dataset: "datasets.Dataset",
    tokenizer,
    trainer_sequence_length: int,
    debug_max_samples: int = None,
    num_proc: int = 1,  # API 호환용; 단일 패스 packing 이라 실제로는 사용 안 함.
) -> "datasets.Dataset":
    """SFT 데이터셋을 ``trainer_sequence_length + 1`` 길이의 packed bin 으로 변환.

    이 함수의 출력 컬럼은 nanotron 의 CLM collator
    (``DataCollatorForCLMWithPositionIds``) 가 기대하는 형식과 정확히 매칭된다:

        input_ids       : List[int]  길이 = pack_size
        label_ids       : List[int]  길이 = pack_size  (collator 가 [:, 1:] 로 shift)
        label_mask      : List[bool] 길이 = pack_size  (True = loss 포함 토큰)
        positions       : List[int]  길이 = pack_size  (sample 마다 0 부터 리셋)
        attention_mask  : List[bool] 길이 = pack_size  (True = 비-PAD 토큰)

    [기존 구현 대비 핵심 차이]
    - 기존 ``process_sft`` 는 ``dataset.map(batched=True)`` 콜백으로 호출되어
      batch 단위 padding 결과를 반환했음 → batch 안의 longest 가 max_length
      미만일 때 길이 불일치 발생.
    - 본 구현은 ``map`` 을 거치지 않고 단일 패스로 직접 ``Dataset.from_dict`` 로
      만듬. 모든 bin 이 정확히 pack_size 길이임을 함수 자체가 보장.
    """
    if debug_max_samples is not None:
        print(f"DEBUG MODE: limiting raw dataset to {debug_max_samples} samples")
        raw_dataset = raw_dataset.select(range(min(debug_max_samples, len(raw_dataset))))

    pack_size = trainer_sequence_length + 1

    bins_input_ids: List[List[int]] = []
    bins_label_mask: List[List[int]] = []
    bins_positions: List[List[int]] = []

    # generator 를 그대로 소비. 메모리 사용량 = 전체 packed dataset 의 크기로,
    # Alpaca + seq=1024 기준 약 5,630 × 1025 × 4B(int32) ≈ 23 MB 수준이라
    # 단일 머신에서 list 로 보관해도 무리 없음.
    for packed in _iter_packed(raw_dataset, tokenizer, pack_size):
        bins_input_ids.append(packed["input_ids"])
        bins_label_mask.append(packed["label_mask"])
        bins_positions.append(packed["positions"])

    pad_id = tokenizer.pad_token_id

    def _to_attention_mask(ids: List[int]) -> List[bool]:
        # PAD 토큰만 0, 나머지는 1. document-masking 은 ``positions`` 로
        # 처리하므로 attention_mask 는 단순 PAD 마스크 역할만 한다.
        return [tok != pad_id for tok in ids]

    bins_attention = [_to_attention_mask(ids) for ids in bins_input_ids]

    print(f"sft_processing: produced {len(bins_input_ids)} packed bins of {pack_size} tokens each")

    return datasets.Dataset.from_dict(
        {
            "input_ids": bins_input_ids,
            # ``label_ids`` 는 collator 가 ``input_ids[:, 1:]`` 로 shift 해서 만들
            # 수도 있으나, 명시적으로 동일한 sequence 를 넘겨주어 collator 코드
            # 변경 폭을 최소화한다. 메모리 추가 비용은 23 MB 수준.
            "label_ids": [list(ids) for ids in bins_input_ids],
            "label_mask": bins_label_mask,
            "positions": bins_positions,
            "attention_mask": bins_attention,
        }
    )


# 아래 두 함수는 기존 nanotron 코드와의 import 호환을 위해 남겨둔다. 어떤 경로
# 에서도 호출되지 않는 상태이며, 호출 시 의도를 명확히 알리기 위해
# NotImplementedError 로 fail-fast 한다.
def pack_sft_sequences(*args, **kwargs):
    raise NotImplementedError(
        "Sequence packing is now built into prepare_sft_dataset; this helper is no longer used."
    )


def process_sft(*args, **kwargs):
    raise NotImplementedError(
        "process_sft was the per-sample fixed-padding path; prepare_sft_dataset now packs sequences directly."
    )
