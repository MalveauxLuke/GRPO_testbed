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

import re
from typing import Any

from verl.utils.reward_score import math_verify
from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed


BOXED_FINAL_ANSWER_PROMPT = "Let's think step by step and output the final answer within \\boxed{}."
_ASSISTANT_WRAPPER_PATTERNS = (
    re.compile(r"<\|im_start\|>assistant\s*", re.IGNORECASE),
    re.compile(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*", re.IGNORECASE),
    re.compile(r"<\uff5cAssistant\uff5c>\s*", re.IGNORECASE),
)
_ASSISTANT_WRAPPER_SUFFIXES = (
    "<|im_end|>",
    "<|eot_id|>",
    "<｜end▁of▁sentence｜>",
)


def canonicalize_boxed_prompt(question: str) -> str:
    prompt = question.strip()
    if BOXED_FINAL_ANSWER_PROMPT in prompt:
        return prompt
    return f"{prompt}\n\n{BOXED_FINAL_ANSWER_PROMPT}"


def _strip_assistant_wrapper(solution_str: str) -> str:
    text = solution_str.strip()
    for pattern in _ASSISTANT_WRAPPER_PATTERNS:
        if pattern.search(text):
            text = pattern.split(text, maxsplit=1)[-1]
            break
    for suffix in _ASSISTANT_WRAPPER_SUFFIXES:
        if suffix in text:
            text = text.split(suffix, 1)[0]
    return text.strip()


def _normalize_fallback_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.replace("$", " ").strip())


def _normalize_ground_truth_text(ground_truth: str) -> str:
    boxed_ground_truth = last_boxed_only_string(ground_truth.strip())
    if boxed_ground_truth is None:
        return ground_truth.strip()
    try:
        return remove_boxed(boxed_ground_truth).strip()
    except AssertionError:
        return ground_truth.strip()


def extract_last_boxed_answer(solution_str: str) -> str | None:
    stripped = _strip_assistant_wrapper(solution_str)
    boxed_answer = last_boxed_only_string(stripped)
    if boxed_answer is None:
        return None
    try:
        return remove_boxed(boxed_answer).strip()
    except AssertionError:
        return None


def compute_correct_reward(solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> float:
    del extra_info

    extracted_answer = extract_last_boxed_answer(solution_str)
    if not extracted_answer:
        return 0.0

    normalized_ground_truth = _normalize_ground_truth_text(ground_truth)
    symbolic_pred = f"\\boxed{{{extracted_answer}}}"
    symbolic_score = float(math_verify.compute_score(symbolic_pred, normalized_ground_truth))
    if symbolic_score > 0:
        return 1.0

    return float(_normalize_fallback_text(extracted_answer) == _normalize_fallback_text(normalized_ground_truth))


def compute_length_reward(solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> float:
    del solution_str, ground_truth

    info = extra_info or {}
    response_length_tokens = int(info.get("response_length_tokens", 0))
    length_limit_tokens = info.get("length_limit_tokens")
    if length_limit_tokens is None:
        raise ValueError("length_limit_tokens is required in extra_info or reward kwargs for deepscaler math reward.")
    return float(response_length_tokens <= int(length_limit_tokens))


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    del data_source, kwargs

    info = dict(extra_info or {})
    correct_reward = compute_correct_reward(solution_str, ground_truth, info)
    length_reward = compute_length_reward(solution_str, ground_truth, info)
    result = {
        "score": float(correct_reward + length_reward),
        "correct_reward": float(correct_reward),
        "length_reward": float(length_reward),
        "answer_parse_ok": float(extract_last_boxed_answer(solution_str) is not None),
        "response_length_tokens": float(int(info.get("response_length_tokens", 0))),
        "length_limit_tokens": float(int(info.get("length_limit_tokens", 0))),
    }
    return result
