"""Download tatsu-lab/alpaca, map (instruction, input, output) -> (prompt, completion),
save as parquet under /opt/dlami/nvme/alpaca_sft_local/ for nanotron SFTDatasetsArgs.

    uv run python examples/heterogeneous/prepare_alpaca.py

We use parquet (not save_to_disk) because nanotron's get_datasets() calls
load_dataset(local_path) which auto-detects parquet but does NOT understand
the arrow-based on-disk format produced by save_to_disk.

Run on each node before launching multi-node training (or run once + rsync the
output dir; ~6.6 M tokens, ~30 MB).
"""

import os
from datasets import load_dataset

OUT_DIR = "/opt/dlami/nvme/alpaca_sft_local"
PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def to_prompt_completion(example):
    instruction = example["instruction"].strip()
    inp = (example.get("input") or "").strip()
    if inp:
        prompt = PROMPT_WITH_INPUT.format(instruction=instruction, input=inp)
    else:
        prompt = PROMPT_NO_INPUT.format(instruction=instruction)
    return {"prompt": prompt, "completion": example["output"]}


def main() -> None:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    ds = ds.map(to_prompt_completion, remove_columns=ds.column_names, num_proc=4).shuffle(seed=42)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "train.parquet")
    ds.to_parquet(out_path)
    print(f"saved {len(ds)} rows to {out_path}")


if __name__ == "__main__":
    main()
