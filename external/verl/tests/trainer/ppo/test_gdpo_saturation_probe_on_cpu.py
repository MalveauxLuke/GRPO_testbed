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
from verl.utils.reward_score.gdpo_binary_probe import compute_score, parse_probe_response


def test_binary_probe_reward_requires_exact_think_answer_format_and_exact_answer():
    solution_str = "<|im_start|>assistant\n<think>2 + 2 = 4</think>\n<answer>4</answer><|im_end|>"
    result = compute_score(
        data_source="openai/gsm8k_gdpo_saturation_probe",
        solution_str=solution_str,
        ground_truth="4",
        extra_info={"experiment_name": "qwen2.5-0.5b-gdpo-binary-probe"},
    )

    assert result == {"score": 2.0, "format_reward": 1.0, "correct_reward": 1.0}


def test_binary_probe_reward_gives_format_credit_without_correctness_credit():
    solution_str = "<think>reasoning</think>\n<answer>5</answer>"
    result = compute_score(
        data_source="openai/gsm8k_gdpo_saturation_probe",
        solution_str=solution_str,
        ground_truth="4",
        extra_info={"experiment_name": "qwen2.5-0.5b-gdpo-binary-probe"},
    )

    assert result == {"score": 1.0, "format_reward": 1.0, "correct_reward": 0.0}


def test_binary_probe_reward_rejects_malformed_or_misordered_tags():
    malformed = "<answer>4</answer><think>reasoning</think>"
    format_reward, parsed_answer = parse_probe_response(malformed)

    assert format_reward == 0
    assert parsed_answer == ""


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
        "format_reward": np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    config = AlgoConfig(
        adv_estimator="gdpo",
        gdpo_baseline_mode="upstream",
        gdpo_reward_keys=["correct_reward", "format_reward"],
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
    assert saturation["per_reward"]["format_reward"]["group_fraction"] == 0.5
    assert saturation["per_reward"]["format_reward"]["all_zero_fraction"] == 0.0
    assert saturation["per_reward"]["format_reward"]["all_one_fraction"] == 0.5
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
            "format_reward": {
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
    assert metrics["gdpo_saturation/format_reward/all_one_fraction"] == 0.25
    assert metrics["gdpo_saturation/any_reward_group_fraction"] == 0.75

    log_path = tmp_path / "events.jsonl"
    _append_gdpo_saturation_events(str(log_path), step=7, events=gdpo_saturation_info["events"], actor_grad_norm=0.125)

    rows = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert '"step": 7' in rows[0]
    assert '"reward_name": "correct_reward"' in rows[0]
    assert '"group_id": "uid-1"' in rows[0]
    assert '"actor_grad_norm": 0.125' in rows[0]
