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
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
AUDIT_SCRIPT_PATH = REPO_ROOT / "scripts" / "sol" / "audit_math_length_rewards.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_reward_modules():
    if "verl.utils.reward_score.deepscaler_math_length" in sys.modules:
        return

    verl_pkg = types.ModuleType("verl")
    verl_pkg.__path__ = [str(REPO_ROOT / "external" / "verl" / "verl")]
    utils_pkg = types.ModuleType("verl.utils")
    utils_pkg.__path__ = [str(REPO_ROOT / "external" / "verl" / "verl" / "utils")]
    reward_score_pkg = types.ModuleType("verl.utils.reward_score")
    reward_score_pkg.__path__ = [str(REPO_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score")]

    sys.modules.setdefault("verl", verl_pkg)
    sys.modules.setdefault("verl.utils", utils_pkg)
    sys.modules.setdefault("verl.utils.reward_score", reward_score_pkg)

    verl_pkg.utils = utils_pkg
    utils_pkg.reward_score = reward_score_pkg

    math_verify_module = _load_module(
        "verl.utils.reward_score.math_verify",
        REPO_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score" / "math_verify.py",
    )
    math_dapo_module = _load_module(
        "verl.utils.reward_score.math_dapo",
        REPO_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score" / "math_dapo.py",
    )

    reward_score_pkg.math_verify = math_verify_module
    reward_score_pkg.math_dapo = math_dapo_module

    _load_module(
        "verl.utils.reward_score.deepscaler_math_length",
        REPO_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score" / "deepscaler_math_length.py",
    )


_bootstrap_reward_modules()
audit_math_length_rewards = _load_module("audit_math_length_rewards", AUDIT_SCRIPT_PATH)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _build_eval_rows() -> list[dict]:
    return [
        {
            "prompt": "Solve x + 2 = 6.\n\nLet's think step by step and output the final answer within \\boxed{}.",
            "ground_truth": "4",
            "data_source": "deepscaler_math_length",
            "extra_info": {"length_limit_tokens": 64},
            "samples": [
                {
                    "text": "We subtract 2 from both sides and get \\boxed{4}.",
                    "score": 2.0,
                    "correct_reward": 1.0,
                    "length_reward": 1.0,
                    "answer_parse_ok": 1.0,
                    "response_length_tokens": 12,
                    "length_limit_tokens": 64.0,
                },
                {
                    "text": "We subtract 2 from both sides and get \\boxed{5}.",
                    "score": 1.0,
                    "correct_reward": 0.0,
                    "length_reward": 1.0,
                    "answer_parse_ok": 1.0,
                    "response_length_tokens": 12,
                    "length_limit_tokens": 64.0,
                },
            ],
        }
    ]


def _build_rollout_rows() -> list[dict]:
    return [
        {
            "input": "Solve x + 2 = 6.",
            "output": "We subtract 2 from both sides and get \\boxed{4}.",
            "gts": "4",
            "data_source": "deepscaler_math_length",
            "score": 2.0,
            "correct_reward": 1.0,
            "length_reward": 1.0,
            "answer_parse_ok": 1.0,
            "response_length_tokens": 10,
            "length_limit_tokens": 64,
            "step": 7,
        }
    ]


def test_eval_artifact_parsing_and_recompute(tmp_path):
    artifact_path = tmp_path / "eval.per_prompt.jsonl"
    _write_jsonl(artifact_path, _build_eval_rows())

    canonical_samples = audit_math_length_rewards.load_canonical_samples(str(artifact_path), input_type="auto")

    assert len(canonical_samples) == 2
    assert canonical_samples[0]["artifact_type"] == "eval"
    assert canonical_samples[0]["prompt"].startswith("Solve x + 2 = 6.")
    assert canonical_samples[0]["ground_truth"] == "4"
    assert canonical_samples[1]["logged"]["correct_reward"] == 0.0

    recomputed = [audit_math_length_rewards.recompute_sample(sample) for sample in canonical_samples]
    assert recomputed[0]["extracted_boxed_answer"] == "4"
    assert recomputed[1]["extracted_boxed_answer"] == "5"
    assert recomputed[0]["mismatch_fields"] == []
    assert recomputed[1]["mismatch_fields"] == []

    summary, mismatches = audit_math_length_rewards.summarize_recomputed_samples(recomputed)
    assert summary["total_samples"] == 2
    assert summary["samples_with_any_mismatch"] == 0
    assert mismatches == []


def test_rollout_artifact_parsing_and_recompute(tmp_path):
    artifact_path = tmp_path / "7.jsonl"
    _write_jsonl(artifact_path, _build_rollout_rows())

    canonical_samples = audit_math_length_rewards.load_canonical_samples(str(artifact_path), input_type="rollout")

    assert len(canonical_samples) == 1
    assert canonical_samples[0]["artifact_type"] == "rollout"
    assert canonical_samples[0]["step"] == 7
    assert canonical_samples[0]["logged"]["length_limit_tokens"] == 64

    recomputed = audit_math_length_rewards.recompute_sample(canonical_samples[0])
    assert recomputed["mismatch_fields"] == []
    assert recomputed["recomputed"]["score"] == 2.0


def test_recompute_summary_reports_mismatch_and_writes_sidecar(tmp_path):
    rows = _build_eval_rows()
    rows[0]["samples"][1]["correct_reward"] = 1.0

    artifact_path = tmp_path / "eval.per_prompt.jsonl"
    summary_path = tmp_path / "summary.json"
    mismatch_path = tmp_path / "mismatches.jsonl"
    _write_jsonl(artifact_path, rows)

    exit_code = audit_math_length_rewards.main(
        [
            "recompute-summary",
            "--input-path",
            str(artifact_path),
            "--summary-output",
            str(summary_path),
            "--mismatch-output",
            str(mismatch_path),
        ]
    )

    assert exit_code == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_samples"] == 2
    assert summary["samples_with_any_mismatch"] == 1
    assert summary["field_mismatch_counts"]["correct_reward"] == 1

    mismatch_rows = [json.loads(line) for line in mismatch_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(mismatch_rows) == 1
    assert mismatch_rows[0]["mismatch_fields"] == ["correct_reward"]
    assert mismatch_rows[0]["sample_index"] == 1


def test_step_filtering_restricts_mixed_rollout_artifacts(tmp_path):
    artifact_path = tmp_path / "rollout.jsonl"
    rows = [
        {
            "input": "Prompt 1",
            "output": "Answer \\boxed{1}.",
            "gts": "1",
            "score": 2.0,
            "correct_reward": 1.0,
            "length_reward": 1.0,
            "answer_parse_ok": 1.0,
            "response_length_tokens": 6,
            "length_limit_tokens": 32,
            "step": 3,
        },
        {
            "input": "Prompt 2",
            "output": "Answer \\boxed{2}.",
            "gts": "2",
            "score": 2.0,
            "correct_reward": 1.0,
            "length_reward": 1.0,
            "answer_parse_ok": 1.0,
            "response_length_tokens": 6,
            "length_limit_tokens": 32,
            "step": 4,
        },
    ]
    _write_jsonl(artifact_path, rows)

    canonical_samples = audit_math_length_rewards.load_canonical_samples(
        str(artifact_path),
        input_type="rollout",
        step=4,
    )

    assert len(canonical_samples) == 1
    assert canonical_samples[0]["step"] == 4
    assert canonical_samples[0]["prompt"] == "Prompt 2"


def test_sample_audit_writes_selected_rows(tmp_path):
    artifact_path = tmp_path / "eval.per_prompt.jsonl"
    output_jsonl = tmp_path / "sample_audit.jsonl"
    _write_jsonl(artifact_path, _build_eval_rows())

    exit_code = audit_math_length_rewards.main(
        [
            "sample-audit",
            "--input-path",
            str(artifact_path),
            "--sample-count",
            "1",
            "--seed",
            "123",
            "--output-jsonl",
            str(output_jsonl),
        ]
    )

    assert exit_code == 0
    sampled_rows = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(sampled_rows) == 1
    assert sampled_rows[0]["artifact_type"] == "eval"
