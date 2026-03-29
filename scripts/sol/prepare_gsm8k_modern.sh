#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

[[ -f "${GSM8K_DIR}/train.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/train.parquet. Run scripts/sol/prepare_gsm8k.sh first."
[[ -f "${GSM8K_DIR}/test.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/test.parquet. Run scripts/sol/prepare_gsm8k.sh first."

sol_msg "Preparing modern 2-reward GSM8K parquet files under ${GSM8K_MODERN_DIR}."

"$(sol_python)" - "${GSM8K_DIR}" "${GSM8K_MODERN_DIR}" <<'PY'
import json
import os
import sys
from typing import Any

from datasets import Dataset, load_dataset

from verl.utils.reward_score.gsm8k_modern_two_reward import ALIGNMENT_SPEC, DATA_SOURCE, build_prompt, extract_hash_answer


def maybe_json(value: Any):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
    return value


def extract_question_and_answer(example: dict[str, Any]) -> tuple[str, str]:
    extra_info = maybe_json(example.get("extra_info", {}))
    if not isinstance(extra_info, dict):
        extra_info = {}

    question = extra_info.get("question")
    answer = extra_info.get("answer")
    if question is None or answer is None:
        raise ValueError("Expected upstream GSM8K parquet rows to preserve extra_info.question and extra_info.answer.")
    return str(question), str(answer)


def make_modern_row(example: dict[str, Any], split: str, idx: int) -> dict[str, Any]:
    question, raw_answer = extract_question_and_answer(example)
    ground_truth = extract_hash_answer(raw_answer)
    reward_model = maybe_json(example.get("reward_model", {}))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        upstream_ground_truth = str(reward_model["ground_truth"]).replace(",", "").replace("$", "").strip()
        if upstream_ground_truth != ground_truth:
            raise ValueError(
                f"Upstream ground truth mismatch at split={split} idx={idx}: "
                f"{upstream_ground_truth!r} != {ground_truth!r}"
            )

    return {
        "data_source": DATA_SOURCE,
        "prompt": build_prompt(question),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "answer": raw_answer,
            "source_dataset": "openai/gsm8k",
            "source_subset": "main",
            "baseline_name": ALIGNMENT_SPEC["baseline_name"],
            "alignment_spec_version": ALIGNMENT_SPEC["version"],
        },
    }


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


base_dir, output_dir = sys.argv[1:]
data_files = {
    "train": os.path.join(base_dir, "train.parquet"),
    "test": os.path.join(base_dir, "test.parquet"),
}
dataset = load_dataset("parquet", data_files=data_files)

train_rows = [make_modern_row(example, "train", idx) for idx, example in enumerate(dataset["train"])]
test_rows = [make_modern_row(example, "test", idx) for idx, example in enumerate(dataset["test"])]

os.makedirs(output_dir, exist_ok=True)
Dataset.from_list(train_rows).to_parquet(os.path.join(output_dir, "train.parquet"))
Dataset.from_list(test_rows).to_parquet(os.path.join(output_dir, "test.parquet"))

metadata = {
    "baseline_name": ALIGNMENT_SPEC["baseline_name"],
    "alignment_spec_version": ALIGNMENT_SPEC["version"],
    "data_source": DATA_SOURCE,
    "source_dataset": "openai/gsm8k",
    "source_subset": "main",
    "train_examples": len(train_rows),
    "test_examples": len(test_rows),
    "reward_names": ["correct_reward", "format_reward"],
}
write_json(os.path.join(output_dir, "metadata.json"), metadata)
write_json(os.path.join(output_dir, "alignment_spec.json"), ALIGNMENT_SPEC)
write_json(os.path.join(output_dir, "train_example.json"), train_rows[0])
write_json(os.path.join(output_dir, "test_example.json"), test_rows[0])

print(f"modern_train_parquet={os.path.join(output_dir, 'train.parquet')}")
print(f"modern_test_parquet={os.path.join(output_dir, 'test.parquet')}")
print(f"alignment_spec_json={os.path.join(output_dir, 'alignment_spec.json')}")
print(f"metadata_json={os.path.join(output_dir, 'metadata.json')}")
PY

[[ -f "${GSM8K_MODERN_DIR}/train.parquet" ]] || sol_fail "Expected ${GSM8K_MODERN_DIR}/train.parquet to exist after preprocessing."
[[ -f "${GSM8K_MODERN_DIR}/test.parquet" ]] || sol_fail "Expected ${GSM8K_MODERN_DIR}/test.parquet to exist after preprocessing."
[[ -f "${GSM8K_MODERN_DIR}/alignment_spec.json" ]] || sol_fail "Expected ${GSM8K_MODERN_DIR}/alignment_spec.json to exist after preprocessing."
[[ -f "${GSM8K_MODERN_DIR}/metadata.json" ]] || sol_fail "Expected ${GSM8K_MODERN_DIR}/metadata.json to exist after preprocessing."

sol_msg "Prepared modern GSM8K baseline data:"
sol_msg "  ${GSM8K_MODERN_DIR}/train.parquet"
sol_msg "  ${GSM8K_MODERN_DIR}/test.parquet"
sol_msg "  ${GSM8K_MODERN_DIR}/alignment_spec.json"
sol_msg "  ${GSM8K_MODERN_DIR}/metadata.json"
