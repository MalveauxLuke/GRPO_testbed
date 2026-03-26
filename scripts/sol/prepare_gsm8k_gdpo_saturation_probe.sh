#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_activate_env

[[ -f "${GSM8K_DIR}/train.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/train.parquet. Run scripts/sol/prepare_gsm8k.sh first."
[[ -f "${GSM8K_DIR}/test.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/test.parquet. Run scripts/sol/prepare_gsm8k.sh first."

PROBE_MODEL_NAME="${PROBE_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
BUILD_HARD_FILTER="${BUILD_HARD_FILTER:-0}"
HARD_FILTER_SAMPLES="${HARD_FILTER_SAMPLES:-16}"
HARD_FILTER_MAX_TRAIN_EXAMPLES="${HARD_FILTER_MAX_TRAIN_EXAMPLES:-}"

sol_msg "Preparing GSM8K GDPO saturation-probe parquet files under ${GSM8K_GDPO_PROBE_DIR}."
if [[ "${BUILD_HARD_FILTER}" == "1" ]]; then
  sol_msg "Also building the hard-saturation training split under ${GSM8K_GDPO_PROBE_HARD_DIR} with ${HARD_FILTER_SAMPLES} samples per prompt."
  sol_msg "The hard-filter prepass runs generation and is much faster from a GPU allocation than a CPU-only allocation."
fi

"$(sol_python)" - "${GSM8K_DIR}" "${GSM8K_GDPO_PROBE_DIR}" "${GSM8K_GDPO_PROBE_HARD_DIR}" "${PROBE_MODEL_NAME}" "${BUILD_HARD_FILTER}" "${HARD_FILTER_SAMPLES}" "${HARD_FILTER_MAX_TRAIN_EXAMPLES}" <<'PY'
import json
import os
import sys
from typing import Any

from datasets import Dataset, load_dataset
from transformers import pipeline

from verl.utils.reward_score.gdpo_binary_probe import compute_correct_reward


def maybe_json(value: Any):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
    return value


def extract_question(example: dict) -> str:
    extra_info = maybe_json(example.get("extra_info", {}))
    if isinstance(extra_info, dict) and "question" in extra_info:
        return str(extra_info["question"])

    prompt = maybe_json(example.get("prompt", []))
    if isinstance(prompt, list) and prompt:
        first = maybe_json(prompt[0])
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])

    raise ValueError("Could not recover a GSM8K question from the source row.")


def extract_ground_truth(example: dict) -> str:
    reward_model = maybe_json(example.get("reward_model", {}))
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        return str(reward_model["ground_truth"])
    raise ValueError("Could not recover reward_model.ground_truth from the source row.")


def build_probe_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Solve the problem. Respond using exactly this format and nothing else outside the tags:\n"
        "<think>your reasoning</think>\n"
        "<answer>final answer</answer>\n"
        "The final numeric answer must appear only inside the <answer> tags."
    )


def make_probe_row(example: dict, split: str, idx: int) -> dict:
    question = extract_question(example)
    ground_truth = extract_ground_truth(example)
    source_extra = maybe_json(example.get("extra_info", {}))
    if not isinstance(source_extra, dict):
        source_extra = {}

    return {
        "data_source": "openai/gsm8k_gdpo_saturation_probe",
        "prompt": [{"role": "user", "content": build_probe_prompt(question)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "source_answer": source_extra.get("answer", ""),
            "probe_variant": "gdpo_binary_saturation_probe",
        },
    }


def write_split(rows: list[dict], out_path: str) -> None:
    Dataset.from_list(rows).to_parquet(out_path)


def build_unfiltered_probe_dataset(base_dir: str, probe_dir: str) -> tuple[list[dict], list[dict]]:
    data_files = {
        "train": os.path.join(base_dir, "train.parquet"),
        "test": os.path.join(base_dir, "test.parquet"),
    }
    dataset = load_dataset("parquet", data_files=data_files)

    train_rows = [make_probe_row(example, "train", idx) for idx, example in enumerate(dataset["train"])]
    test_rows = [make_probe_row(example, "test", idx) for idx, example in enumerate(dataset["test"])]

    os.makedirs(probe_dir, exist_ok=True)
    write_split(train_rows, os.path.join(probe_dir, "train.parquet"))
    write_split(test_rows, os.path.join(probe_dir, "test.parquet"))
    return train_rows, test_rows


def keep_hard_saturation_rows(
    train_rows: list[dict],
    hard_dir: str,
    test_rows: list[dict],
    probe_model_name: str,
    hard_filter_samples: int,
    max_train_examples: str,
) -> None:
    max_examples = int(max_train_examples) if max_train_examples else None
    candidate_rows = train_rows[:max_examples] if max_examples is not None else train_rows

    generator = pipeline("text-generation", model=probe_model_name, device_map="auto")
    tokenizer = generator.tokenizer

    kept_rows = []
    total = len(candidate_rows)
    for idx, row in enumerate(candidate_rows, start=1):
        prompt_text = tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        outputs = generator(
            prompt_text,
            max_new_tokens=192,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            num_return_sequences=hard_filter_samples,
            return_full_text=False,
        )
        ground_truth = row["reward_model"]["ground_truth"]
        correct_hits = sum(
            int(
                compute_correct_reward(
                    solution_str=output["generated_text"],
                    ground_truth=ground_truth,
                    extra_info={"experiment_name": probe_model_name.lower()},
                )
            )
            for output in outputs
        )
        if correct_hits == 0:
            kept_rows.append(row)
        if idx % 100 == 0 or idx == total:
            print(f"hard_filter_progress={idx}/{total} kept={len(kept_rows)}")

    os.makedirs(hard_dir, exist_ok=True)
    write_split(kept_rows, os.path.join(hard_dir, "train.parquet"))
    write_split(test_rows, os.path.join(hard_dir, "test.parquet"))
    metadata_path = os.path.join(hard_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "probe_model_name": probe_model_name,
                "hard_filter_samples": hard_filter_samples,
                "source_train_examples": len(candidate_rows),
                "kept_train_examples": len(kept_rows),
            },
            fp,
            indent=2,
            sort_keys=True,
        )


base_dir, probe_dir, hard_dir, probe_model_name, build_hard_filter, hard_filter_samples, max_train_examples = sys.argv[1:]
train_rows, test_rows = build_unfiltered_probe_dataset(base_dir, probe_dir)
print(f"probe_train_parquet={os.path.join(probe_dir, 'train.parquet')}")
print(f"probe_test_parquet={os.path.join(probe_dir, 'test.parquet')}")

if build_hard_filter == "1":
    keep_hard_saturation_rows(
        train_rows=train_rows,
        hard_dir=hard_dir,
        test_rows=test_rows,
        probe_model_name=probe_model_name,
        hard_filter_samples=int(hard_filter_samples),
        max_train_examples=max_train_examples,
    )
    print(f"hard_train_parquet={os.path.join(hard_dir, 'train.parquet')}")
    print(f"hard_test_parquet={os.path.join(hard_dir, 'test.parquet')}")
PY

[[ -f "${GSM8K_GDPO_PROBE_DIR}/train.parquet" ]] || sol_fail "Expected ${GSM8K_GDPO_PROBE_DIR}/train.parquet to exist after probe preprocessing."
[[ -f "${GSM8K_GDPO_PROBE_DIR}/test.parquet" ]] || sol_fail "Expected ${GSM8K_GDPO_PROBE_DIR}/test.parquet to exist after probe preprocessing."

sol_msg "Prepared unfiltered probe data:"
sol_msg "  ${GSM8K_GDPO_PROBE_DIR}/train.parquet"
sol_msg "  ${GSM8K_GDPO_PROBE_DIR}/test.parquet"

if [[ "${BUILD_HARD_FILTER}" == "1" ]]; then
  [[ -f "${GSM8K_GDPO_PROBE_HARD_DIR}/train.parquet" ]] || sol_fail "Expected ${GSM8K_GDPO_PROBE_HARD_DIR}/train.parquet to exist after hard filtering."
  [[ -f "${GSM8K_GDPO_PROBE_HARD_DIR}/test.parquet" ]] || sol_fail "Expected ${GSM8K_GDPO_PROBE_HARD_DIR}/test.parquet to exist after hard filtering."
  sol_msg "Prepared hard-saturation probe data:"
  sol_msg "  ${GSM8K_GDPO_PROBE_HARD_DIR}/train.parquet"
  sol_msg "  ${GSM8K_GDPO_PROBE_HARD_DIR}/test.parquet"
fi
