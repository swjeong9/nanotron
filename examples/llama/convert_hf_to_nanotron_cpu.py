"""HF → nanotron 변환 (CPU 전용 버전).

기존 ``convert_hf_to_nanotron.py`` 는 ``load_nanotron_model`` 이 default 로 cuda
device 사용해 GPU 필요. 본 스크립트는 device='cpu' 강제 + ``torchrun`` 안 써도
되는 standalone 형태.

Usage:
    cd /home/ubuntu/nanotron
    .venv/bin/python examples/llama/convert_hf_to_nanotron_cpu.py \\
        --checkpoint_path=/path/to/hf_model \\
        --save_path=/path/to/nanotron_model

dist init 은 ``MASTER_ADDR=localhost MASTER_PORT=29500 RANK=0 WORLD_SIZE=1`` 로
환경변수 셋업해 1-rank gloo backend 으로 임시 init.
"""

import dataclasses
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

# 변환 모듈 절대 import 위해 nanotron 의 examples/ 를 sys.path 에 추가.
_THIS_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_EXAMPLES_DIR))

import torch  # noqa: E402

# CPU 노드 호환성: nanotron 의 ``nn/layer_norm.py`` 가 import 시
# ``flash_attn.ops.triton.layer_norm`` 을 부르는데, 그 모듈이 import 시 즉시
# ``torch.cuda.current_device()`` 호출 → CUDA 없는 노드에서 RuntimeError.
# nanotron import 전에 torch.cuda 의 핵심 함수들 stub 처리.
if not torch.cuda.is_available():
    class _FakeProps:
        warp_size = 32
    torch.cuda.current_device = lambda: 0
    torch.cuda.get_device_properties = lambda *a, **kw: _FakeProps()
    torch.cuda.get_device_capability = lambda *a, **kw: (8, 0)
import nanotron  # noqa: E402
from nanotron.models.llama import LlamaForTraining  # noqa: E402
from transformers import LlamaForCausalLM  # noqa: E402

from llama.convert_hf_to_nanotron import (  # noqa: E402
    convert_hf_to_nt,
    get_nanotron_config,
)
from llama.convert_weights import load_nanotron_model  # noqa: E402


def main():
    ap = ArgumentParser()
    ap.add_argument("--checkpoint_path", type=Path, required=True)
    ap.add_argument("--save_path", type=Path, required=True)
    args = ap.parse_args()

    # 1-rank dist init (gloo backend, CPU 전용)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="gloo")

    # HF model — config 가 torch_dtype: bfloat16 이라 from_pretrained 도 BF16 로 로드
    print(f"[convert-cpu] loading HF from {args.checkpoint_path} ...", flush=True)
    hf_model = LlamaForCausalLM.from_pretrained(args.checkpoint_path)
    print(f"[convert-cpu] HF model dtype: {next(hf_model.parameters()).dtype}", flush=True)

    # nanotron model on CPU (default 은 cuda → 우회)
    model_config = get_nanotron_config(hf_model.config)
    print(f"[convert-cpu] creating nanotron model on cpu ...", flush=True)
    nanotron_model = load_nanotron_model(
        model_config=model_config,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    # Weight 복사
    print(f"[convert-cpu] copying weights ...", flush=True)
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1
    )
    convert_hf_to_nt(hf_model, nanotron_model, model_config)

    # 저장
    args.save_path.mkdir(parents=True, exist_ok=True)
    print(f"[convert-cpu] saving to {args.save_path} ...", flush=True)
    nanotron.serialize.save_weights(
        model=nanotron_model, parallel_context=parallel_context, root_folder=args.save_path
    )
    with open(args.save_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(model_config), f)
    print(f"[convert-cpu] done — saved to {args.save_path}", flush=True)


if __name__ == "__main__":
    main()
