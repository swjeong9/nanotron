# Heterogeneous GPU LLM training

Working directory for the asymmetric pipeline parallelism research on this nanotron fork (A100 + L40S cluster, Llama 3.x family). Plan, hardware, sweep configurations, and measurement methodology live in [`docs/project_background.md`](../../docs/project_background.md).

Anything specific to the heterogeneous-training research — sweep configs, conversion glue, ablation runners, custom scripts — should land here. Keep upstream-relevant changes (general fixes, build wiring) outside this folder so future rebases on HF main stay clean.

## Files

| File | Purpose |
|---|---|
| [`config_tiny_llama_l4_smoke.yaml`](config_tiny_llama_l4_smoke.yaml) | Stage 1 build sanity on a single L4 (g6.xlarge): synthetic ~80M Llama-family model, dummy data, 5 steps, dp=tp=pp=1. Verifies that the fork builds, FlashAttention 2 runs on sm_89, and the training loop completes. |
| [`config_llama32_1b_l4_resume.yaml`](config_llama32_1b_l4_resume.yaml) | Stage 1 conversion sanity: resumes the converted Llama 3.2 1B checkpoint (`s3://swj-nanotron-model/llama-3.2-1b/nanotron/`) on a single L4 with bf16 optimizer state. Confirms the converted weights load, the loss does not match a fresh random init (so the conversion actually transferred information), and the loop steps cleanly within 24 GiB. |
| [`config_llama32_1b_alpaca_pp2.yaml`](config_llama32_1b_alpaca_pp2.yaml) | Stage 2 baseline (PP=2 inter-node, layer 8/8 via `parallelism.pp_layer_partition`). Llama 3.2 1B + Alpaca SFT, 1 epoch (~400 step at GBS=128, seq_len=1024). Both nodes resume from the same converted checkpoint. |
| [`prepare_alpaca.py`](prepare_alpaca.py) | One-time: `uv run python examples/heterogeneous/prepare_alpaca.py` downloads `tatsu-lab/alpaca`, maps `(instruction, input, output)` → `(prompt, completion)` (Alpaca standard prompt template), saves a `DatasetDict` to `/tmp/alpaca_sft_local`. Run on each node before launch. |
| [`launch_pp2_node0.sh`](launch_pp2_node0.sh) | Multi-node launcher for **NODE 0** (172.31.31.40 / L4 / master, NODE_RANK=0). Sets `NCCL_IB_DISABLE=1`, `NCCL_SOCKET_IFNAME=ens5`, c10d rendezvous on `:29500`. |
| [`launch_pp2_node1.sh`](launch_pp2_node1.sh) | Multi-node launcher for **NODE 1** (172.31.40.226 / A10G, NODE_RANK=1). `rdzv_endpoint` points at NODE 0. Start within ~30 s of NODE 0. |
| [`nodes.json`](nodes.json) | Active nodes used in the research. Two fields per entry: `private_ip` and `instance_type`. |
| [`add_node.py`](add_node.py) | `uv run python examples/heterogeneous/add_node.py <private_ip>` — SSH into the given private IP, query its IMDS for the instance type, and append/replace the entry in `nodes.json`. Assumes shared SSH keys within the VPC. |

## Related artefacts (outside this folder)

- [`docs/project_background.md`](../../docs/project_background.md) — canonical plan: motivation, hardware, sweep configs (A/A'/B/C/D/E), measurement metrics (throughput, MFU, energy efficiency), Stage 1/2 dev workflow.
- [`docs/dcgm_test_report.md`](../../docs/dcgm_test_report.md) — Stage 1 DCGM verification (per-field results on L4, the `-j` flag absence in DCGM 4.5.2, awk text→JSONL converter pattern, A100/L40S split commands).
- `s3://swj-nanotron-model/` — converted nanotron-format checkpoints.

## Conventions

- All Python via `uv run` (not direct `.venv/bin/python`).
- Single-GPU runs: `CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --nproc_per_node=1 run_train.py --config-file <path>` from the repo root.
- New sweep configs go here as `config_<id>_<hardware>.yaml`. Keep one yaml per (config-id × hardware) pair so logs and checkpoints stay disambiguated.
