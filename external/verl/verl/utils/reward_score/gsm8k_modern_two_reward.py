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

"""Modern 2-reward GSM8K baseline aligned to public structured-output examples.

This module intentionally implements a minimal modern GSM8K multi-reward setup:
- format_reward: bounded strict+approximate structured-output compliance
- correct_reward: structured-answer-only normalized numeric answer correctness

It is reference-derived, not a novel reward design:
- data origin and gold-answer extraction follow official GSM8K / upstream verl
- the tags-only output contract follows the simpler public GRPO/GSM8K cookbook
  family where answer correctness is extracted from the structured answer span
- the exact/approximate format split is adapted from the Hugging Face advanced
  GRPO cookbook, but folded back into a single public `format_reward` key so
  our GDPO contract stays `correct_reward + format_reward`
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


DATA_SOURCE = "openai/gsm8k_modern_two_reward"
SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Solve the user's problem and respond using exactly "
    "this format with no extra text before or after the tags:\n"
    "<reasoning>your step-by-step reasoning</reasoning>\n"
    "<answer>final numeric answer</answer>"
)

ALIGNMENT_SPEC = {
    "baseline_name": "gsm8k_modern_two_reward",
    "version": "2026-04-03-coupled-soft-format",
    "dataset_source": {
        "name": "openai/gsm8k",
        "subset": "main",
        "precedent": "official_gsm8k_and_upstream_verl_preprocess",
    },
    "structured_output": {
        "schema": "<reasoning>...</reasoning>\n<answer>...</answer>",
        "precedent": "hf_grpo_cookbook_style_structured_gsm8k",
    },
    "rewards": {
        "format_reward": "bounded_blend_of_strict_and_approximate_structured_output_compliance",
        "correct_reward": "structured_answer_only_numeric_equivalence_against_gsm8k_final_answer",
    },
    "simplifications": [
        "two_rewards_not_three_or_four",
        "binary_numeric_equivalence_not_ratio_based_partial_credit",
        "format_reward_bakes_in_strict_plus_approximate_structure_signals",
        "correctness_coupled_to_structured_answer_parse",
        "no_response_wide_fallback_correctness",
        "no_length_reward",
    ],
    "excluded_features": [
        "generated_output_hash_marker",
        "length_reward",
        "ratio_based_partial_credit_correctness",
        "separate_third_numeric_extraction_reward",
        "response_wide_correctness_fallback",
    ],
}

_HASH_ANSWER_PATTERN = re.compile(r"####\s*(?P<answer>[^\n<]+)")
_STRUCTURED_RESPONSE_PATTERN = re.compile(
    r"^\s*<reasoning>(?P<reasoning>.*?)</reasoning>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)
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
_FORBIDDEN_INNER_TAGS = ("<reasoning>", "</reasoning>", "<answer>", "</answer>")
_FORMAT_REWARD_STRICT_WEIGHT = 0.5
_FORMAT_REWARD_APPROX_WEIGHT = 0.5


def build_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]


def extract_hash_answer(text: str) -> str:
    matches = _HASH_ANSWER_PATTERN.findall(text)
    if not matches:
        raise ValueError("Could not extract final numeric #### answer from text.")
    candidate = normalize_numeric_text(matches[-1])
    if _parse_numeric_value(candidate) is None:
        raise ValueError("Could not parse final numeric #### answer from text.")
    return candidate


def normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    normalized = text.strip().replace(",", "").replace("$", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _parse_numeric_value(text: str | None) -> Decimal | None:
    normalized = normalize_numeric_text(text)
    if normalized == "":
        return None
    if normalized.count("/") == 1:
        numerator_text, denominator_text = normalized.split("/", maxsplit=1)
        if numerator_text == "" or denominator_text == "":
            return None
        try:
            numerator = Decimal(numerator_text)
            denominator = Decimal(denominator_text)
        except InvalidOperation:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _is_numeric_equivalent(predicted: str | None, expected: str | None) -> bool:
    predicted_value = _parse_numeric_value(predicted)
    expected_value = _parse_numeric_value(expected)
    if predicted_value is None or expected_value is None:
        return False
    return predicted_value == expected_value


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


def parse_structured_response(solution_str: str) -> tuple[int, str]:
    response = _strip_assistant_wrapper(solution_str)
    match = _STRUCTURED_RESPONSE_PATTERN.match(response)
    if match is None:
        return 0, ""

    reasoning_text = match.group("reasoning")
    answer_text_raw = match.group("answer")
    if any(tag in reasoning_text for tag in _FORBIDDEN_INNER_TAGS):
        return 0, ""
    if any(tag in answer_text_raw for tag in _FORBIDDEN_INNER_TAGS):
        return 0, ""

    answer_text = normalize_numeric_text(answer_text_raw)
    if _parse_numeric_value(answer_text) is None:
        return 0, ""
    return 1, answer_text


def match_format_exactly(solution_str: str) -> float:
    strict_format_reward, _ = parse_structured_response(solution_str)
    return float(strict_format_reward)


def _cookbook_style_approximate_format_raw(solution_str: str) -> float:
    """Return the cookbook-style raw partial-format score in [-2.0, 2.0].

    The HF advanced GRPO cookbook awards +0.5 when each required tag appears
    exactly once and -0.5 otherwise. We normalize that raw score back into
    [0.0, 1.0] before blending it into our single public `format_reward`.
    """

    response = _strip_assistant_wrapper(solution_str)
    score = 0.0
    score += 0.5 if response.count("<reasoning>") == 1 else -0.5
    score += 0.5 if response.count("</reasoning>") == 1 else -0.5
    score += 0.5 if response.count("<answer>") == 1 else -0.5
    score += 0.5 if response.count("</answer>") == 1 else -0.5
    return score


def compute_approx_format_reward(solution_str: str) -> float:
    raw_score = _cookbook_style_approximate_format_raw(solution_str)
    return float((raw_score + 2.0) / 4.0)


def compute_format_reward(solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> float:
    del ground_truth, extra_info
    strict_format_reward = match_format_exactly(solution_str)
    approx_format_reward = compute_approx_format_reward(solution_str)
    return float(
        (_FORMAT_REWARD_STRICT_WEIGHT * strict_format_reward)
        + (_FORMAT_REWARD_APPROX_WEIGHT * approx_format_reward)
    )


def compute_correct_reward(solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> float:
    del extra_info
    _, predicted_answer = parse_structured_response(solution_str)
    expected_answer = normalize_numeric_text(ground_truth)
    return 1.0 if predicted_answer != "" and _is_numeric_equivalent(predicted_answer, expected_answer) else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    del data_source, kwargs

    strict_format_reward = match_format_exactly(solution_str)
    approx_format_reward = compute_approx_format_reward(solution_str)
    format_reward = (
        (_FORMAT_REWARD_STRICT_WEIGHT * strict_format_reward)
        + (_FORMAT_REWARD_APPROX_WEIGHT * approx_format_reward)
    )
    _, parsed_answer = parse_structured_response(solution_str)
    expected_answer = normalize_numeric_text(ground_truth)
    correct_reward = 1.0 if parsed_answer != "" and _is_numeric_equivalent(parsed_answer, expected_answer) else 0.0
    result = {
        "score": float(format_reward + correct_reward),
        "correct_reward": float(correct_reward),
        "format_reward": float(format_reward),
        "strict_format_reward": float(strict_format_reward),
        "approx_format_reward": float(approx_format_reward),
        "answer_parse_ok": float(parsed_answer != ""),
        "parsed_answer": parsed_answer,
        "expected_answer": expected_answer,
    }
    return result
