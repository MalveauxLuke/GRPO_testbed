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
import sys
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


reward_module = _load_module("gsm8k_modern_two_reward_test", REWARD_MODULE_PATH)


def test_extract_hash_answer_matches_upstream_gsm8k_style():
    raw_answer = "We compute the total carefully.\n#### 1,234"
    assert reward_module.extract_hash_answer(raw_answer) == "1234"


def test_build_prompt_uses_system_and_user_messages():
    prompt = reward_module.build_prompt("What is 2 + 2?")

    assert prompt == [
        {"role": "system", "content": reward_module.SYSTEM_PROMPT},
        {"role": "user", "content": "What is 2 + 2?"},
    ]


def test_compute_score_accepts_valid_correct_structured_answer():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Add the numbers.</reasoning>\n#### 4\n<answer>4</answer>",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
    assert result["hash_parse_ok"] == 1.0
    assert result["tag_answer_parse_ok"] == 1.0
    assert result["hash_answer"] == "4"
    assert result["tag_answer"] == "4"
    assert result["hash_answer_equals_tag_answer"] == 1.0


def test_compute_score_gives_partial_format_credit_for_trailing_junk():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Add the numbers.</reasoning>\n#### 4\n<answer>4</answer> trailing text",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.5
    assert result["correct_reward"] == 1.0
    assert result["score"] == 1.5
    assert result["hash_parse_ok"] == 1.0
    assert result["tag_answer_parse_ok"] == 1.0
    assert result["hash_answer"] == "4"
    assert result["tag_answer"] == "4"


def test_compute_score_normalizes_whitespace_commas_and_currency():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>\n#### $1,234\n<answer> $1,234 </answer>",
        ground_truth="1234",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
    assert result["hash_answer"] == "1234"
    assert result["tag_answer"] == "1234"


def test_compute_score_requires_hash_for_correctness_even_when_tags_are_well_formed():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning><answer>4</answer>",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.5
    assert result["correct_reward"] == 0.0
    assert result["score"] == 0.5
    assert result["hash_parse_ok"] == 0.0
    assert result["tag_answer_parse_ok"] == 1.0
    assert result["tag_answer"] == "4"


def test_compute_score_keeps_correctness_when_only_hash_is_present():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>\n#### 4",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.25
    assert result["correct_reward"] == 1.0
    assert result["score"] == 1.25
    assert result["hash_parse_ok"] == 1.0
    assert result["tag_answer_parse_ok"] == 0.0
    assert result["hash_answer"] == "4"


def test_compute_score_gives_partial_format_credit_for_malformed_tag_order():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<answer>4</answer>\n#### 4\n<reasoning>Compute it.</reasoning>",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.5
    assert result["correct_reward"] == 1.0
    assert result["score"] == 1.5
    assert result["hash_parse_ok"] == 1.0
    assert result["tag_answer_parse_ok"] == 1.0
    assert result["hash_answer"] == "4"
    assert result["tag_answer"] == "4"


def test_compute_score_treats_numeric_equivalence_as_correct_for_integer_answers():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>\n#### 72.0\n<answer>72.0</answer>",
        ground_truth="72",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
    assert result["hash_answer"] == "72.0"
    assert result["tag_answer"] == "72.0"


def test_compute_score_treats_numeric_equivalence_as_correct_for_decimal_answers():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>\n#### 0.50\n<answer>0.50</answer>",
        ground_truth="0.5",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
    assert result["hash_answer"] == "0.50"
    assert result["tag_answer"] == "0.50"


def test_compute_score_supports_fraction_and_scientific_notation_in_hash_answer():
    fraction_result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>\n#### 1/2\n<answer>1/2</answer>",
        ground_truth="0.5",
        extra_info={},
    )
    scientific_result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>\n#### 7.2e1\n<answer>7.2e1</answer>",
        ground_truth="72",
        extra_info={},
    )

    assert fraction_result["correct_reward"] == 1.0
    assert fraction_result["format_reward"] == 1.0
    assert scientific_result["correct_reward"] == 1.0
    assert scientific_result["format_reward"] == 1.0


def test_compute_score_logs_hash_tag_mismatch_without_zeroing_format():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>\n#### 4\n<answer>5</answer>",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["hash_answer"] == "4"
    assert result["tag_answer"] == "5"
    assert result["hash_answer_equals_tag_answer"] == 0.0


def test_compute_score_rejects_non_numeric_tag_without_using_reasoning_fallback():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>We compute 4 in the work.</reasoning>\n#### 4\n<answer>final</answer>",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.5
    assert result["correct_reward"] == 1.0
    assert result["hash_parse_ok"] == 1.0
    assert result["tag_answer_parse_ok"] == 0.0


def test_compute_score_rejects_non_numeric_hash_for_strict_format_and_correctness():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>We compute it.</reasoning>\n#### final\n<answer>4</answer>",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.5
    assert result["correct_reward"] == 0.0
    assert result["score"] == 0.5
    assert result["hash_parse_ok"] == 0.0
    assert result["tag_answer_parse_ok"] == 1.0


def test_compute_score_strips_assistant_wrapper_tokens():
    wrapped_output = "<|im_start|>assistant\n<reasoning>Compute it.</reasoning>\n#### 4\n<answer>4</answer><|im_end|>"
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str=wrapped_output,
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
