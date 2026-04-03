# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
REWARD_MODULE_PATH = REPO_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score" / "gsm8k_modern_two_reward.py"
VERIFY_SCRIPT_PATH = REPO_ROOT / "scripts" / "sol" / "verify_gsm8k_modern_baseline.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


reward_module = _load_module("gsm8k_modern_two_reward_verify_test", REWARD_MODULE_PATH)
verify_module = _load_module("verify_gsm8k_modern_baseline_test", VERIFY_SCRIPT_PATH)


def _source_row(question: str = "How many apples are left?", answer: str = "We subtract.\n#### 12") -> dict:
    return {"question": question, "answer": answer}


def _processed_row(split: str = "train", index: int = 0, question: str = "How many apples are left?", answer: str = "We subtract.\n#### 12") -> dict:
    return {
        "data_source": reward_module.DATA_SOURCE,
        "prompt": reward_module.build_prompt(question),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": reward_module.extract_hash_answer(answer)},
        "extra_info": {
            "split": split,
            "index": index,
            "question": question,
            "answer": answer,
            "source_dataset": "openai/gsm8k",
            "source_subset": "main",
            "baseline_name": reward_module.ALIGNMENT_SPEC["baseline_name"],
            "alignment_spec_version": reward_module.ALIGNMENT_SPEC["version"],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def test_dataset_audit_passes_for_matching_rows():
    source_splits = {
        "train": [_source_row()],
        "test": [_source_row(question="What is 3 + 4?", answer="Add them.\n#### 7")],
    }
    processed_splits = {
        "train": [_processed_row()],
        "test": [_processed_row(split="test", question="What is 3 + 4?", answer="Add them.\n#### 7")],
    }

    summary, mismatches = verify_module.audit_dataset_rows(
        source_splits=source_splits,
        processed_splits=processed_splits,
        reward_module=reward_module,
    )

    assert summary["samples_checked"] == 2
    assert summary["mismatch_count"] == 0
    assert mismatches == []


def test_dataset_audit_catches_ground_truth_mismatch():
    source_splits = {"train": [_source_row()], "test": []}
    bad_row = _processed_row()
    bad_row["reward_model"]["ground_truth"] = "13"
    processed_splits = {"train": [bad_row], "test": []}

    summary, mismatches = verify_module.audit_dataset_rows(
        source_splits=source_splits,
        processed_splits=processed_splits,
        reward_module=reward_module,
    )

    assert summary["mismatch_count"] == 1
    assert mismatches[0]["mismatch_fields"] == ["ground_truth"]


def test_reward_audit_passes_full_synthetic_battery():
    processed_splits = {"train": [_processed_row()], "test": []}

    summary, mismatches = verify_module.audit_reward_rows(processed_splits, reward_module)

    assert summary["samples_checked"] == 1
    assert summary["mismatch_count"] == 0
    assert summary["case_counts"]["valid_correct"] == 1
    assert summary["case_counts"]["numeric_equivalence"] == 1
    assert mismatches == []


def test_artifact_audit_detects_logged_reward_mismatch(tmp_path):
    artifact_path = tmp_path / "gsm8k_eval.per_prompt.jsonl"
    _write_jsonl(
        artifact_path,
        [
            {
                "ground_truth": "12",
                "samples": [
                    {
                        "text": "<reasoning>Subtract carefully.</reasoning><answer>12</answer>",
                        "score": 2.0,
                        "correct_reward": 0.0,
                        "format_reward": 1.0,
                    }
                ],
            }
        ],
    )

    samples = verify_module.load_canonical_artifact_samples(str(artifact_path), artifact_type="eval")
    summary, mismatches = verify_module.audit_artifact_samples(samples, reward_module)

    assert summary["total_samples"] == 1
    assert summary["samples_with_any_mismatch"] == 1
    assert summary["field_mismatch_counts"]["correct_reward"] == 1
    assert mismatches[0]["mismatch_fields"] == ["correct_reward"]


def test_reference_audit_passes_with_matching_alignment_and_docs(tmp_path, monkeypatch):
    docs_note_path = tmp_path / "gsm8k_modern_two_reward_baseline.md"
    docs_note_path.write_text(
        (
            "Two rewards.\n"
            "No length reward.\n"
            "Structured format with <reasoning> and <answer> tags.\n"
            "Approximate format credit is blended into format_reward.\n"
            "Correctness uses numeric equivalence independent of strict format parsing.\n"
        ),
        encoding="utf-8",
    )
    alignment_spec_path = tmp_path / "alignment_spec.json"
    alignment_spec_path.write_text(json.dumps(reward_module.ALIGNMENT_SPEC, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(
        verify_module,
        "load_split_rows_from_parquet_dir",
        lambda _path: {"train": [_processed_row()], "test": [_processed_row(split="test", question="What is 3 + 4?", answer="Add them.\n#### 7")]},
    )

    summary, mismatches = verify_module.audit_reference_alignment(
        processed_dir=tmp_path,
        reward_module=reward_module,
        alignment_spec_path=alignment_spec_path,
        docs_note_path=docs_note_path,
    )

    assert summary["checks_passed"] is True
    assert summary["mismatch_count"] == 0
    assert mismatches == []
