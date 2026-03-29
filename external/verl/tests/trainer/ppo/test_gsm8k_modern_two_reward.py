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
        solution_str="<reasoning>Add the numbers.</reasoning><answer>4</answer>",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
    assert result["answer_parse_ok"] == 1.0
    assert result["parsed_answer"] == "4"
    assert result["expected_answer"] == "4"


def test_compute_score_rejects_trailing_junk_even_with_correct_number():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Add the numbers.</reasoning><answer>4</answer> trailing text",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.0
    assert result["correct_reward"] == 0.0
    assert result["score"] == 0.0
    assert result["answer_parse_ok"] == 0.0
    assert result["parsed_answer"] == ""


def test_compute_score_normalizes_whitespace_commas_and_currency():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning><answer> $1,234 </answer>",
        ground_truth="1234",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
    assert result["parsed_answer"] == "1234"


def test_compute_score_requires_answer_inside_tags():
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str="<reasoning>Compute it.</reasoning>4",
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 0.0
    assert result["correct_reward"] == 0.0
    assert result["score"] == 0.0
    assert result["answer_parse_ok"] == 0.0


def test_compute_score_strips_assistant_wrapper_tokens():
    wrapped_output = "<|im_start|>assistant\n<reasoning>Compute it.</reasoning><answer>4</answer><|im_end|>"
    result = reward_module.compute_score(
        data_source=reward_module.DATA_SOURCE,
        solution_str=wrapped_output,
        ground_truth="4",
        extra_info={},
    )

    assert result["format_reward"] == 1.0
    assert result["correct_reward"] == 1.0
    assert result["score"] == 2.0
