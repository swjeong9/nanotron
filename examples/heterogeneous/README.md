# Heterogeneous GPU LLM training

Working directory for the asymmetric pipeline parallelism research on this nanotron fork (A100 + L40S cluster, Llama 3.x family). Plan, hardware, sweep configurations, and measurement methodology live in [`docs/project_background.md`](../../docs/project_background.md).

Anything specific to the heterogeneous-training research — sweep configs, conversion glue, ablation runners, custom scripts — should land here. Keep upstream-relevant changes (general fixes, build wiring) outside this folder so future rebases on HF main stay clean.

## Files

| File | Purpose |
|---|---|
| [`config_tiny_llama_l4_smoke.yaml`](config_tiny_llama_l4_smoke.yaml) | Stage 1 build sanity on a single L4 (g6.xlarge): synthetic ~80M Llama-family model, dummy data, 5 steps, dp=tp=pp=1. Verifies that the fork builds, FlashAttention 2 runs on sm_89, and the training loop completes. |
| [`config_llama32_1b_l4_resume.yaml`](config_llama32_1b_l4_resume.yaml) | Stage 1 conversion sanity: resumes the converted Llama 3.2 1B checkpoint (`s3://swj-nanotron-model/llama-3.2-1b/nanotron/`) on a single L4 with bf16 optimizer state. Confirms the converted weights load, the loss does not match a fresh random init (so the conversion actually transferred information), and the loop steps cleanly within 24 GiB. |

## Related artefacts (outside this folder)

- [`docs/project_background.md`](../../docs/project_background.md) — canonical plan: motivation, hardware, sweep configs (A/A'/B/C/D/E), measurement metrics (throughput, MFU, energy efficiency), Stage 1/2 dev workflow.
- [`docs/dcgm_test_report.md`](../../docs/dcgm_test_report.md) — Stage 1 DCGM verification (per-field results on L4, the `-j` flag absence in DCGM 4.5.2, awk text→JSONL converter pattern, A100/L40S split commands).
- `s3://swj-nanotron-model/` — converted nanotron-format checkpoints.

## Conventions

- All Python via `uv run` (not direct `.venv/bin/python`).
- Single-GPU runs: `CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --nproc_per_node=1 run_train.py --config-file <path>` from the repo root.
- New sweep configs go here as `config_<id>_<hardware>.yaml`. Keep one yaml per (config-id × hardware) pair so logs and checkpoints stay disambiguated.
