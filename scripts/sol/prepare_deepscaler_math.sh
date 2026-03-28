#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

DEEPSCALER_DATASET_NAME="${DEEPSCALER_DATASET_NAME:-sungyub/deepscaler-preview-verl}"
DEEPSCALER_DATASET_CONFIG="${DEEPSCALER_DATASET_CONFIG:-}"
DEEPSCALER_PROD_LENGTH_LIMIT_TOKENS="${DEEPSCALER_PROD_LENGTH_LIMIT_TOKENS:-4000}"
DEEPSCALER_DEBUG_LENGTH_LIMIT_TOKENS="${DEEPSCALER_DEBUG_LENGTH_LIMIT_TOKENS:-1024}"
DEEPSCALER_DEBUG_TRAIN_SIZE="${DEEPSCALER_DEBUG_TRAIN_SIZE:-256}"
DEEPSCALER_DEBUG_TEST_SIZE="${DEEPSCALER_DEBUG_TEST_SIZE:-64}"
DEEPSCALER_HOLDOUT_SIZE="${DEEPSCALER_HOLDOUT_SIZE:-512}"

sol_msg "Preparing DeepScaleR-style math parquet files."
sol_msg "Dataset: ${DEEPSCALER_DATASET_NAME}${DEEPSCALER_DATASET_CONFIG:+ (${DEEPSCALER_DATASET_CONFIG})}"
sol_msg "Production output dir: ${DEEPSCALER_DIR}"
sol_msg "Debug output dir: ${DEEPSCALER_DEBUG_DIR}"

"$(sol_python)" - "${DEEPSCALER_DATASET_NAME}" "${DEEPSCALER_DATASET_CONFIG}" "${DEEPSCALER_DIR}" "${DEEPSCALER_DEBUG_DIR}" "${DEEPSCALER_PROD_LENGTH_LIMIT_TOKENS}" "${DEEPSCALER_DEBUG_LENGTH_LIMIT_TOKENS}" "${DEEPSCALER_DEBUG_TRAIN_SIZE}" "${DEEPSCALER_DEBUG_TEST_SIZE}" "${DEEPSCALER_HOLDOUT_SIZE}" <<'PY'
import json
import os
import sys
from typing import Any

from datasets import Dataset, load_dataset

from verl.utils.reward_score.deepscaler_math_length import BOXED_FINAL_ANSWER_PROMPT, canonicalize_boxed_prompt


def maybe_json(value: Any):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
    return value


def get_prompt_text(example: dict) -> str:
    prompt = maybe_json(example.get("prompt"))
    if isinstance(prompt, list) and prompt:
        first = maybe_json(prompt[0])
        if isinstance(first, dict) and "content" in first:
            return str(first["content"]).strip()
    if isinstance(prompt, str):
        return prompt.strip()
    for key in ("question", "problem", "prompt_text", "input"):
        if key in example and example[key]:
            return str(example[key]).strip()
    raise ValueError(f"Could not recover prompt text from example keys: {sorted(example.keys())}")


def get_ground_truth(example: dict) -> str:
    reward_model = maybe_json(example.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"]).strip()
    for key in ("ground_truth", "answer", "solution", "final_answer"):
        if key in example and example[key] is not None:
            return str(example[key]).strip()
    raise ValueError(f"Could not recover ground truth from example keys: {sorted(example.keys())}")


def split_prompt_and_question(prompt_text: str) -> tuple[str, str]:
    prompt_text = prompt_text.strip()
    if BOXED_FINAL_ANSWER_PROMPT in prompt_text:
        question = prompt_text.split(BOXED_FINAL_ANSWER_PROMPT, 1)[0].strip()
        return question, prompt_text
    return prompt_text, canonicalize_boxed_prompt(prompt_text)


def normalize_example(example: dict, *, split: str, idx: int, length_limit_tokens: int, source_dataset: str) -> dict:
    prompt_text = get_prompt_text(example)
    question, canonical_prompt = split_prompt_and_question(prompt_text)
    ground_truth = get_ground_truth(example)
    source_extra = maybe_json(example.get("extra_info", {}))
    if not isinstance(source_extra, dict):
        source_extra = {}

    return {
        "data_source": "deepscaler_math_length",
        "prompt": [{"role": "user", "content": canonical_prompt}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "length_limit_tokens": int(length_limit_tokens),
            "prompt_style": "deepscaler_boxed",
            "source_dataset": source_dataset,
            "source_data_source": example.get("data_source", ""),
            "source_split": source_extra.get("split", split),
        },
    }


def write_split(rows: list[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Dataset.from_list(rows).to_parquet(out_path)


dataset_name, dataset_config, prod_dir, debug_dir, prod_limit, debug_limit, debug_train_size, debug_test_size, holdout_size = sys.argv[1:]
load_kwargs = {}
if dataset_config:
    dataset = load_dataset(dataset_name, dataset_config)
else:
    dataset = load_dataset(dataset_name)

split_names = list(dataset.keys())
train_split_name = "train" if "train" in dataset else split_names[0]
eval_split_name = next((name for name in ("test", "validation", "val") if name in dataset), None)

train_split = dataset[train_split_name]
if eval_split_name is not None:
    eval_split = dataset[eval_split_name]
else:
    holdout = min(int(holdout_size), max(1, len(train_split) // 10))
    eval_split = train_split.select(range(len(train_split) - holdout, len(train_split)))
    train_split = train_split.select(range(0, len(train_split) - holdout))

prod_train_rows = [
    normalize_example(
        example,
        split="train",
        idx=idx,
        length_limit_tokens=int(prod_limit),
        source_dataset=dataset_name,
    )
    for idx, example in enumerate(train_split)
]
prod_test_rows = [
    normalize_example(
        example,
        split="test",
        idx=idx,
        length_limit_tokens=int(prod_limit),
        source_dataset=dataset_name,
    )
    for idx, example in enumerate(eval_split)
]

debug_train_rows = [
    normalize_example(
        example,
        split="train",
        idx=idx,
        length_limit_tokens=int(debug_limit),
        source_dataset=dataset_name,
    )
    for idx, example in enumerate(train_split.select(range(min(int(debug_train_size), len(train_split)))))
]
debug_test_rows = [
    normalize_example(
        example,
        split="test",
        idx=idx,
        length_limit_tokens=int(debug_limit),
        source_dataset=dataset_name,
    )
    for idx, example in enumerate(eval_split.select(range(min(int(debug_test_size), len(eval_split)))))
]

write_split(prod_train_rows, os.path.join(prod_dir, "train.parquet"))
write_split(prod_test_rows, os.path.join(prod_dir, "test.parquet"))
write_split(debug_train_rows, os.path.join(debug_dir, "train.parquet"))
write_split(debug_test_rows, os.path.join(debug_dir, "test.parquet"))

with open(os.path.join(prod_dir, "metadata.json"), "w", encoding="utf-8") as fp:
    json.dump(
        {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config or None,
            "train_split_name": train_split_name,
            "eval_split_name": eval_split_name or "heldout_from_train",
            "train_examples": len(prod_train_rows),
            "test_examples": len(prod_test_rows),
            "length_limit_tokens": int(prod_limit),
        },
        fp,
        indent=2,
        sort_keys=True,
    )

with open(os.path.join(debug_dir, "metadata.json"), "w", encoding="utf-8") as fp:
    json.dump(
        {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config or None,
            "train_examples": len(debug_train_rows),
            "test_examples": len(debug_test_rows),
            "length_limit_tokens": int(debug_limit),
        },
        fp,
        indent=2,
        sort_keys=True,
    )

for out_dir, train_rows, test_rows in ((prod_dir, prod_train_rows, prod_test_rows), (debug_dir, debug_train_rows, debug_test_rows)):
    with open(os.path.join(out_dir, "train_example.json"), "w", encoding="utf-8") as fp:
        json.dump(train_rows[0], fp, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, "test_example.json"), "w", encoding="utf-8") as fp:
        json.dump(test_rows[0], fp, indent=2, sort_keys=True)

print(f"prod_train_parquet={os.path.join(prod_dir, 'train.parquet')}")
print(f"prod_test_parquet={os.path.join(prod_dir, 'test.parquet')}")
print(f"debug_train_parquet={os.path.join(debug_dir, 'train.parquet')}")
print(f"debug_test_parquet={os.path.join(debug_dir, 'test.parquet')}")
PY

[[ -f "${DEEPSCALER_DIR}/train.parquet" ]] || sol_fail "Expected ${DEEPSCALER_DIR}/train.parquet after DeepScaleR preprocessing."
[[ -f "${DEEPSCALER_DIR}/test.parquet" ]] || sol_fail "Expected ${DEEPSCALER_DIR}/test.parquet after DeepScaleR preprocessing."
[[ -f "${DEEPSCALER_DEBUG_DIR}/train.parquet" ]] || sol_fail "Expected ${DEEPSCALER_DEBUG_DIR}/train.parquet after debug preprocessing."
[[ -f "${DEEPSCALER_DEBUG_DIR}/test.parquet" ]] || sol_fail "Expected ${DEEPSCALER_DEBUG_DIR}/test.parquet after debug preprocessing."

sol_msg "Prepared production DeepScaleR-style math data:"
sol_msg "  ${DEEPSCALER_DIR}/train.parquet"
sol_msg "  ${DEEPSCALER_DIR}/test.parquet"
sol_msg "Prepared debug DeepScaleR-style math data:"
sol_msg "  ${DEEPSCALER_DEBUG_DIR}/train.parquet"
sol_msg "  ${DEEPSCALER_DEBUG_DIR}/test.parquet"
