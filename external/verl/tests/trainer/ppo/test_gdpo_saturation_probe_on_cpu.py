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

import numpy as np
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import compute_gdpo_outcome_advantage
from verl.trainer.ppo.ray_trainer import _append_gdpo_saturation_events, _compute_gdpo_saturation_metrics
from verl.utils.reward_score.deepscaler_math_length import (
    compute_correct_reward,
    compute_length_reward,
    compute_score,
    extract_last_boxed_answer,
)


def test_deepscaler_reward_uses_boxed_answer_and_binary_length_reward():
    solution_str = "We reason carefully and end with the final answer \\boxed{4}."
    result = compute_score(
        data_source="deepscaler_math_length",
        solution_str=solution_str,
        ground_truth="4",
        extra_info={"response_length_tokens": 32, "length_limit_tokens": 64},
    )

    assert result["score"] == 2.0
    assert result["correct_reward"] == 1.0
    assert result["length_reward"] == 1.0
    assert result["answer_parse_ok"] == 1.0


def test_deepscaler_reward_uses_the_last_boxed_answer():
    solution_str = "A wrong intermediate result is \\boxed{3}, but the final answer is \\boxed{4}."
    assert extract_last_boxed_answer(solution_str) == "4"
    assert compute_correct_reward(solution_str, "4") == 1.0


def test_deepscaler_reward_accepts_boxed_ground_truth():
    solution_str = "The final answer is \\boxed{4}."
    assert compute_correct_reward(solution_str, "\\boxed{4}") == 1.0


def test_deepscaler_reward_uses_symbolic_verification_before_string_fallback():
    solution_str = "After simplification the answer is \\boxed{\\frac{1}{2}}."
    result = compute_score(
        data_source="deepscaler_math_length",
        solution_str=solution_str,
        ground_truth="0.5",
        extra_info={"response_length_tokens": 80, "length_limit_tokens": 32},
    )

    assert result["correct_reward"] == 1.0
    assert result["length_reward"] == 0.0
    assert result["score"] == 1.0


def test_deepscaler_reward_falls_back_to_exact_string_for_non_symbolic_answers():
    solution_str = "Casework gives roots \\boxed{9 and -7}."
    assert compute_correct_reward(solution_str, "9 and -7") == 1.0


def test_deepscaler_reward_requires_boxed_answer():
    malformed = "The answer is 4."
    assert extract_last_boxed_answer(malformed) is None
    result = compute_score(
        data_source="deepscaler_math_length",
        solution_str=malformed,
        ground_truth="4",
        extra_info={"response_length_tokens": 8, "length_limit_tokens": 16},
    )
    assert result["correct_reward"] == 0.0
    assert result["answer_parse_ok"] == 0.0


def test_deepscaler_reward_fails_closed_on_malformed_boxed_answer():
    malformed = "The answer is \\boxed{4."
    assert extract_last_boxed_answer(malformed) is None
    assert compute_correct_reward(malformed, "4") == 0.0


def test_deepscaler_length_reward_uses_token_count_threshold():
    assert compute_length_reward("unused", "unused", {"response_length_tokens": 16, "length_limit_tokens": 16}) == 1.0
    assert compute_length_reward("unused", "unused", {"response_length_tokens": 17, "length_limit_tokens": 16}) == 0.0


def test_gdpo_saturation_meta_info_distinguishes_all_zero_and_all_one():
    batch_size = 8
    prompt_len = 2
    response_len = 3

    token_level_rewards = torch.zeros(batch_size, response_len, dtype=torch.float32)
    response_mask = torch.ones(batch_size, response_len, dtype=torch.float32)
    index = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    batch = {
        "prompts": torch.ones(batch_size, prompt_len, dtype=torch.int64),
        "attention_mask": torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64),
    }
    non_tensor_batch = {
        "correct_reward": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "length_reward": np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    config = AlgoConfig(
        adv_estimator="gdpo",
        gdpo_baseline_mode="upstream",
        gdpo_reward_keys=["correct_reward", "length_reward"],
    )
    meta_info = {}

    advantages, returns = compute_gdpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
        meta_info=meta_info,
    )

    assert advantages.shape == (batch_size, response_len)
    assert returns.shape == (batch_size, response_len)

    saturation = meta_info["gdpo_saturation"]
    assert saturation["per_reward"]["correct_reward"]["group_fraction"] == 0.5
    assert saturation["per_reward"]["correct_reward"]["all_zero_fraction"] == 0.5
    assert saturation["per_reward"]["correct_reward"]["all_one_fraction"] == 0.0
    assert saturation["per_reward"]["length_reward"]["group_fraction"] == 0.5
    assert saturation["per_reward"]["length_reward"]["all_zero_fraction"] == 0.0
    assert saturation["per_reward"]["length_reward"]["all_one_fraction"] == 0.5
    assert saturation["any_reward_group_fraction"] == 0.5
    assert len(saturation["events"]) == 2


def test_gdpo_saturation_metric_projection_and_event_sidecar(tmp_path):
    gdpo_saturation_info = {
        "per_reward": {
            "correct_reward": {
                "group_fraction": 0.5,
                "all_zero_fraction": 0.5,
                "all_one_fraction": 0.0,
                "any": True,
            },
            "length_reward": {
                "group_fraction": 0.25,
                "all_zero_fraction": 0.0,
                "all_one_fraction": 0.25,
                "any": True,
            },
        },
        "any_reward_group_fraction": 0.75,
        "events": [
            {
                "reward_name": "correct_reward",
                "group_id": "uid-1",
                "group_size": 4,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "is_zero_std": True,
                "is_all_zero": True,
                "is_all_one": False,
            }
        ],
    }

    metrics = _compute_gdpo_saturation_metrics(gdpo_saturation_info)
    assert metrics["gdpo_saturation/correct_reward/group_fraction"] == 0.5
    assert metrics["gdpo_saturation/correct_reward/all_zero_fraction"] == 0.5
    assert metrics["gdpo_saturation/length_reward/all_one_fraction"] == 0.25
    assert metrics["gdpo_saturation/any_reward_group_fraction"] == 0.75

    log_path = tmp_path / "events.jsonl"
    _append_gdpo_saturation_events(str(log_path), step=7, events=gdpo_saturation_info["events"], actor_grad_norm=0.125)

    rows = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert '"step": 7' in rows[0]
    assert '"reward_name": "correct_reward"' in rows[0]
    assert '"group_id": "uid-1"' in rows[0]
    assert '"actor_grad_norm": 0.125' in rows[0]
