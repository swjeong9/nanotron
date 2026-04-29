"""Download tatsu-lab/alpaca, map (instruction, input, output) -> (prompt, completion),
save as DatasetDict to /tmp/alpaca_sft_local for nanotron SFTDatasetsArgs.

    uv run python examples/heterogeneous/prepare_alpaca.py

Run on each node before launching the multi-node training. The output dir is
local to each node — same content, separate copies (~6.6 M tokens, well under 50 MB).
"""

from datasets import DatasetDict, load_dataset

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
    DatasetDict({"train": ds}).save_to_disk(OUT_DIR)
    print(f"saved {len(ds)} rows to {OUT_DIR}")


if __name__ == "__main__":
    main()
