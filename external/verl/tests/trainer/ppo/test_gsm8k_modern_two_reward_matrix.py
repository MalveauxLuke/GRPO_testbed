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
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
REWARD_MODULE_PATH = REPO_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score" / "gsm8k_modern_two_reward.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


reward_module = _load_module("gsm8k_modern_two_reward_matrix_test", REWARD_MODULE_PATH)


@dataclass(frozen=True)
class Case:
    name: str
    ground_truth: str
    solution: str
    expected_format_reward: float
    expected_correct_reward: float
    expected_hash_parse_ok: float
    expected_tag_parse_ok: float
    expected_hash_equals_tag: float
    concern: str


CASES = [
    Case(
        name="exact_correct",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 72\n<answer>72</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="baseline sanity",
    ),
    Case(
        name="exact_wrong",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 71\n<answer>71</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=0.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="baseline sanity",
    ),
    Case(
        name="decimal_equivalence",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 72.0\n<answer>72.0</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="checks numeric equivalence",
    ),
    Case(
        name="fractional_equivalence",
        ground_truth="0.5",
        solution="<reasoning>We solve it.</reasoning>\n#### 0.50\n<answer>0.50</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="checks numeric equivalence",
    ),
    Case(
        name="currency_and_commas",
        ground_truth="1234",
        solution="<reasoning>We solve it.</reasoning>\n#### $1,234\n<answer>$1,234</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="normalization coverage",
    ),
    Case(
        name="fraction_hash_supported",
        ground_truth="0.5",
        solution="<reasoning>We solve it.</reasoning>\n#### 1/2\n<answer>1/2</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="fraction support",
    ),
    Case(
        name="scientific_hash_supported",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 7.2e1\n<answer>7.2e1</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="scientific notation support",
    ),
    Case(
        name="correct_tag_but_missing_hash",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning><answer>72</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=0.0,
        expected_hash_parse_ok=0.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=0.0,
        concern="correctness must come from #### only",
    ),
    Case(
        name="plain_text_number_without_hash",
        ground_truth="72",
        solution="The final answer is 72.",
        expected_format_reward=0.0,
        expected_correct_reward=0.0,
        expected_hash_parse_ok=0.0,
        expected_tag_parse_ok=0.0,
        expected_hash_equals_tag=0.0,
        concern="response-wide fallback removed",
    ),
    Case(
        name="hash_only_no_tags",
        ground_truth="72",
        solution="#### 72",
        expected_format_reward=0.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=0.0,
        expected_hash_equals_tag=0.0,
        concern="correctness and format are separated",
    ),
    Case(
        name="hash_with_reasoning_but_no_answer_tag",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 72",
        expected_format_reward=0.25,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=0.0,
        expected_hash_equals_tag=0.0,
        concern="partial format without answer tag",
    ),
    Case(
        name="malformed_order_partial_format",
        ground_truth="72",
        solution="<answer>72</answer>\n#### 72\n<reasoning>We solve it.</reasoning>",
        expected_format_reward=0.5,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="partial format with correct hash",
    ),
    Case(
        name="trailing_junk_partial_format",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 72\n<answer>72</answer> trailing text",
        expected_format_reward=0.5,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="partial format with correct hash",
    ),
    Case(
        name="duplicated_reasoning_tags",
        ground_truth="72",
        solution="<reasoning>One</reasoning><reasoning>Two</reasoning>\n#### 72\n<answer>72</answer>",
        expected_format_reward=0.25,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="partial format with duplicated tags",
    ),
    Case(
        name="answer_outside_tags_no_hash",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>72",
        expected_format_reward=0.25,
        expected_correct_reward=0.0,
        expected_hash_parse_ok=0.0,
        expected_tag_parse_ok=0.0,
        expected_hash_equals_tag=0.0,
        concern="numeric text outside tags no longer counts",
    ),
    Case(
        name="empty_answer_with_correct_hash",
        ground_truth="72",
        solution="<reasoning>We compute 72.</reasoning>\n#### 72\n<answer></answer>",
        expected_format_reward=0.5,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=0.0,
        expected_hash_equals_tag=0.0,
        concern="empty answer tag no longer rescued",
    ),
    Case(
        name="non_numeric_answer_with_correct_hash",
        ground_truth="72",
        solution="<reasoning>We compute 72.</reasoning>\n#### 72\n<answer>final</answer>",
        expected_format_reward=0.5,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=0.0,
        expected_hash_equals_tag=0.0,
        concern="non-numeric tag no longer rescued",
    ),
    Case(
        name="ambiguous_answer_with_correct_hash",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 72\n<answer>71 or 72</answer>",
        expected_format_reward=0.5,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=0.0,
        expected_hash_equals_tag=0.0,
        concern="ambiguous tag answer is rejected",
    ),
    Case(
        name="hash_tag_mismatch",
        ground_truth="72",
        solution="<reasoning>We solve it.</reasoning>\n#### 72\n<answer>71</answer>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=0.0,
        concern="mismatch logged without changing reward keys",
    ),
    Case(
        name="assistant_wrapper_tokens",
        ground_truth="72",
        solution="<|im_start|>assistant\n<reasoning>We solve it.</reasoning>\n#### 72\n<answer>72</answer><|im_end|>",
        expected_format_reward=1.0,
        expected_correct_reward=1.0,
        expected_hash_parse_ok=1.0,
        expected_tag_parse_ok=1.0,
        expected_hash_equals_tag=1.0,
        concern="wrapper compatibility",
    ),
]


def _assert_close(name: str, field: str, observed: float | str, expected: float | str) -> None:
    if isinstance(expected, float):
        assert abs(float(observed) - expected) < 1e-9, (
            f"{name} field {field!r} mismatch: observed={observed!r} expected={expected!r}"
        )
    else:
        assert observed == expected, f"{name} field {field!r} mismatch: observed={observed!r} expected={expected!r}"


def run_case(case: Case) -> dict[str, object]:
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str=case.solution,
        ground_truth=case.ground_truth,
        extra_info={},
    )
    _assert_close(case.name, "format_reward", result["format_reward"], case.expected_format_reward)
    _assert_close(case.name, "correct_reward", result["correct_reward"], case.expected_correct_reward)
    _assert_close(case.name, "hash_parse_ok", result["hash_parse_ok"], case.expected_hash_parse_ok)
    _assert_close(case.name, "tag_answer_parse_ok", result["tag_answer_parse_ok"], case.expected_tag_parse_ok)
    _assert_close(case.name, "hash_answer_equals_tag_answer", result["hash_answer_equals_tag_answer"], case.expected_hash_equals_tag)
    return {
        "name": case.name,
        "ground_truth": case.ground_truth,
        "format_reward": float(result["format_reward"]),
        "correct_reward": float(result["correct_reward"]),
        "score": float(result["score"]),
        "strict_format_reward": float(result["strict_format_reward"]),
        "approx_format_reward": float(result["approx_format_reward"]),
        "hash_parse_ok": float(result["hash_parse_ok"]),
        "tag_answer_parse_ok": float(result["tag_answer_parse_ok"]),
        "hash_answer": result["hash_answer"],
        "tag_answer": result["tag_answer"],
        "hash_answer_equals_tag_answer": float(result["hash_answer_equals_tag_answer"]),
        "concern": case.concern,
    }


def run_matrix() -> dict[str, object]:
    rows = [run_case(case) for case in CASES]
    summary = {
        "total_cases": len(rows),
        "correct_reward_positive_cases": sum(int(row["correct_reward"] == 1.0) for row in rows),
        "unexpected_correct_without_hash": [
            row["name"] for row in rows if row["correct_reward"] == 1.0 and row["hash_parse_ok"] == 0.0
        ],
        "tag_parse_failures": [row["name"] for row in rows if row["tag_answer_parse_ok"] == 0.0],
        "hash_tag_mismatch_cases": [
            row["name"]
            for row in rows
            if row["hash_parse_ok"] == 1.0 and row["tag_answer_parse_ok"] == 1.0 and row["hash_answer_equals_tag_answer"] == 0.0
        ],
        "full_format_without_hash_cases": [
            row["name"] for row in rows if row["format_reward"] == 1.0 and row["hash_parse_ok"] == 0.0
        ],
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    report = run_matrix()
    print(json.dumps(report, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
