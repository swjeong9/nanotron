"""HuggingFace Qwen 3 → nanotron weight conversion.

Qwen 3 vs Qwen 2 architectural deltas (handled here):
  • q_norm / k_norm RMSNorms over head_dim, applied between q/k proj and RoPE.
  • head_dim explicit in HF config (may differ from hidden_size // num_attention_heads,
    e.g. Qwen3-0.6B: 1024/16=64 vs head_dim=128).
  • attention_bias=False (Qwen 2 was True for QKV).
  • RoPE base 1000000 (Qwen 2 default 10000).

Usage:
  uv run torchrun --nproc_per_node=1 -m examples.qwen.convert_hf_qwen3_to_nanotron \
      --hf-model-path /path/to/hf_qwen3_dir \
      --save-path     /path/to/nanotron_out_dir
"""
import argparse
import dataclasses
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import nanotron
from nanotron import logging
from nanotron.config import (
    Config,
    GeneralArgs,
    LoggingArgs,
    ModelArgs,
    ParallelismArgs,
    TokenizerArgs,
)
from nanotron.config.models_config import ExistingCheckpointInit, Qwen2Config
from nanotron.models import build_model
from nanotron.models.qwen import Qwen2ForTraining
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.serialize import TrainingMetadata, save_meta, save_weights
from nanotron.serialize.metadata import DataStageMetadata
from nanotron.trainer import mark_tied_parameters
from nanotron.logging import log_rank, set_ranks_logging_level

logger = logging.get_logger(__name__)

DEVICE = torch.device("cuda")
TORCH_DTYPE = torch.bfloat16


def hf_to_nanotron_config(hf_config) -> Qwen2Config:
    """Map HF Qwen3Config to nanotron Qwen2Config with Qwen 3 extensions."""
    head_dim = getattr(hf_config, "head_dim", None) or (
        hf_config.hidden_size // hf_config.num_attention_heads
    )
    return Qwen2Config(
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=hf_config.eos_token_id,
        hidden_act=hf_config.hidden_act,
        hidden_size=hf_config.hidden_size,
        initializer_range=hf_config.initializer_range,
        intermediate_size=hf_config.intermediate_size,
        is_qwen2_config=True,
        max_position_embeddings=hf_config.max_position_embeddings,
        num_attention_heads=hf_config.num_attention_heads,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=head_dim,
        pad_token_id=getattr(hf_config, "pad_token_id", None),
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_scaling=hf_config.rope_scaling,
        rope_theta=hf_config.rope_theta,
        rope_interleaved=False,
        tie_word_embeddings=hf_config.tie_word_embeddings,
        use_cache=hf_config.use_cache,
        vocab_size=hf_config.vocab_size,
        attention_bias=False,  # Qwen 3
        _use_qk_norm=True,     # Qwen 3
        _attn_implementation="flash_attention_2",
        sliding_window_size=None,
        moe_config=None,
    )


@torch.no_grad()
def copy_weights(hf_model, nanotron_model, nanotron_config: Qwen2Config):
    nt = nanotron_model.model
    hf = hf_model.model

    # Token embeddings
    log_rank("Copying token embeddings", logger=logger, level=logging.INFO, rank=0)
    assert nt.token_position_embeddings.pp_block.token_embedding.weight.shape == hf.embed_tokens.weight.shape
    nt.token_position_embeddings.pp_block.token_embedding.weight.copy_(hf.embed_tokens.weight)

    # Decoder layers
    for i in tqdm(range(nanotron_config.num_hidden_layers), desc="Copying layers"):
        hf_layer = hf.layers[i]
        nt_layer = nt.decoder[i].pp_block

        # Pre-attn / post-attn norms
        nt_layer.input_layernorm.weight.copy_(hf_layer.input_layernorm.weight)
        nt_layer.post_attention_layernorm.weight.copy_(hf_layer.post_attention_layernorm.weight)

        # qkv: HF separate q/k/v → nanotron packed (Q | K | V) along output dim 0
        qkv = torch.cat(
            [
                hf_layer.self_attn.q_proj.weight,
                hf_layer.self_attn.k_proj.weight,
                hf_layer.self_attn.v_proj.weight,
            ],
            dim=0,
        )
        assert qkv.shape == nt_layer.attn.qkv_proj.weight.shape, (
            f"qkv shape mismatch at layer {i}: hf {qkv.shape} vs nt {nt_layer.attn.qkv_proj.weight.shape}"
        )
        nt_layer.attn.qkv_proj.weight.copy_(qkv)
        # Qwen 3: attention_bias=False → no qkv bias

        # o_proj
        nt_layer.attn.o_proj.weight.copy_(hf_layer.self_attn.o_proj.weight)

        # Qwen 3 q_norm / k_norm (head_dim sized)
        nt_layer.attn.q_norm.weight.copy_(hf_layer.self_attn.q_norm.weight)
        nt_layer.attn.k_norm.weight.copy_(hf_layer.self_attn.k_norm.weight)

        # MLP gate_up packed
        gate_up = torch.cat(
            [hf_layer.mlp.gate_proj.weight, hf_layer.mlp.up_proj.weight], dim=0
        )
        nt_layer.mlp.gate_up_proj.weight.copy_(gate_up)
        nt_layer.mlp.down_proj.weight.copy_(hf_layer.mlp.down_proj.weight)

    # Final norm + lm_head
    nt.final_layer_norm.pp_block.weight.copy_(hf.norm.weight)

    # lm_head: HF stores weight (after tie_weights, lm_head.weight aliases embed_tokens.weight when tied)
    assert nt.lm_head.pp_block.weight.shape == hf_model.lm_head.weight.shape
    nt.lm_head.pp_block.weight.copy_(hf_model.lm_head.weight)
    log_rank("Weights copied", logger=logger, level=logging.INFO, rank=0)


def main(args):
    parallel_config = ParallelismArgs(dp=1, pp=1, tp=1)
    parallel_context = ParallelContext(
        data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1
    )
    set_ranks_logging_level(parallel_context=parallel_context, logging_config=LoggingArgs())

    log_rank(f"Loading HF model from {args.hf_model_path}", logger=logger, level=logging.INFO, rank=0)
    # CPU 로드 — 14B 등 큰 모델은 HF + nanotron 둘 다 GPU 올리면 A100 40GB OOM (29.5×2 = 59 GB).
    # ``.to(DEVICE)`` 호출 안 하면 default 로 CPU 에 머무름. ``device_map="cpu"`` 는 accelerate 의존이라 회피.
    # weight copy 는 ``param_nt.copy_(hf_param)`` 가 cross-device (CPU→GPU) 자동 처리.
    # ``attn_implementation`` 도 eager (flash_attn GPU init 의존 회피) — 어차피 forward 안 함.
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path, torch_dtype=TORCH_DTYPE, attn_implementation="eager"
    )
    hf_config = hf_model.config
    arch = getattr(hf_config, "architectures", [None])[0]
    assert arch == "Qwen3ForCausalLM", f"Expected Qwen3ForCausalLM, got {arch}. Use a Qwen3 dense model."

    nanotron_config = hf_to_nanotron_config(hf_config)
    log_rank(
        f"Building nanotron model: hidden={nanotron_config.hidden_size} layers={nanotron_config.num_hidden_layers} "
        f"heads={nanotron_config.num_attention_heads} kv_heads={nanotron_config.num_key_value_heads} "
        f"head_dim={nanotron_config.head_dim} tie_emb={nanotron_config.tie_word_embeddings}",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )

    nanotron_model = build_model(
        model_builder=lambda: Qwen2ForTraining(
            config=nanotron_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
        ),
        parallel_context=parallel_context,
        dtype=TORCH_DTYPE,
        device=DEVICE,
    )
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context, parallel_config=parallel_config)
    sanity_check(root_module=nanotron_model)

    copy_weights(hf_model, nanotron_model, nanotron_config)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=save_path)

    save_meta(
        root_folder=save_path,
        parallel_context=parallel_context,
        training_metadata=TrainingMetadata(
            last_train_step=0,
            consumed_train_samples=0,
            data_stages=[DataStageMetadata(name="Empty", consumed_train_samples=0, start_training_step=0)],
        ),
    )

    # Tokenizer (skip if HF dir didn't include one)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_path)
        tokenizer.save_pretrained(save_path)
    except Exception as e:
        log_rank(f"Tokenizer save skipped: {e}", logger=logger, level=logging.WARNING, rank=0)

    # model_config.json (used by load_nanotron_model and serialize)
    with open(save_path / "model_config.json", "w") as f:
        json.dump(dataclasses.asdict(nanotron_config), f, indent=2)

    log_rank(f"Conversion done → {save_path}", logger=logger, level=logging.INFO, rank=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    main(parser.parse_args())
