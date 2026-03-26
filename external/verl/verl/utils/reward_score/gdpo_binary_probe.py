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


_THINK_ANSWER_PATTERN = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)


def _strip_assistant_wrapper(solution_str: str, extra_info: dict | None) -> str:
    experiment_name = (extra_info or {}).get("experiment_name", "")
    text = solution_str

    if "qwen" in experiment_name and "<|im_start|>assistant" in text and "<|im_end|>" in text:
        text = text.split("<|im_start|>assistant", 1)[-1].split("<|im_end|>", 1)[0]
    elif "llama" in experiment_name and "<|start_header_id|>assistant<|end_header_id|>" in text:
        text = text.split("<|start_header_id|>assistant<|end_header_id|>", 1)[-1]
        text = text.split("<|eot_id|>", 1)[0]

    return text.strip()


def _normalize_answer_text(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip().replace(",", "").replace("$", "")


def parse_probe_response(solution_str: str, extra_info: dict | None = None) -> tuple[int, str]:
    response = _strip_assistant_wrapper(solution_str, extra_info)
    match = _THINK_ANSWER_PATTERN.match(response)
    if not match:
        return 0, ""
    return 1, _normalize_answer_text(match.group("answer"))


def compute_correct_reward(solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float:
    _, predicted_answer = parse_probe_response(solution_str, extra_info)
    expected_answer = _normalize_answer_text(ground_truth)
    return 1.0 if predicted_answer != "" and predicted_answer == expected_answer else 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    del data_source, kwargs

    format_reward, _ = parse_probe_response(solution_str, extra_info)
    correct_reward = compute_correct_reward(solution_str, ground_truth, extra_info)

    result = {
        "score": float(format_reward + correct_reward),
        "format_reward": float(format_reward),
        "correct_reward": float(correct_reward),
    }
    return result
