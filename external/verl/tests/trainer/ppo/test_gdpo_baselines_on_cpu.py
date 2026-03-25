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
import pytest
import torch

from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import compute_gdpo_outcome_advantage
from verl.utils.reward_score.rlla import compute_score


def _make_gdpo_inputs():
    batch_size = 4
    prompt_len = 2
    response_len = 3
    token_level_rewards = torch.zeros(batch_size, response_len, dtype=torch.float32)
    response_mask = torch.ones(batch_size, response_len, dtype=torch.float32)
    index = np.asarray([0, 0, 1, 1], dtype=np.int64)
    batch = {
        "prompts": torch.ones(batch_size, prompt_len, dtype=torch.int64),
        "attention_mask": torch.ones(batch_size, prompt_len + response_len, dtype=torch.int64),
    }
    non_tensor_batch = {
        "accuracy_reward": np.asarray([3.0, 1.0, -1.0, 0.5], dtype=np.float32),
        "format_reward": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    return token_level_rewards, response_mask, index, batch, non_tensor_batch


def test_gdpo_baseline_modes_match_for_default_two_reward_case():
    token_level_rewards, response_mask, index, batch, non_tensor_batch = _make_gdpo_inputs()
    upstream_config = AlgoConfig(
        adv_estimator="gdpo",
        gdpo_baseline_mode="upstream",
        gdpo_reward_keys=["accuracy_reward", "format_reward"],
    )
    nvlabs_config = AlgoConfig(
        adv_estimator="gdpo",
        gdpo_baseline_mode="nvlabs_reference",
        gdpo_reward_keys=["accuracy_reward", "format_reward"],
    )

    upstream_adv, upstream_ret = compute_gdpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=upstream_config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )
    nvlabs_adv, nvlabs_ret = compute_gdpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        config=nvlabs_config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )

    assert torch.allclose(upstream_adv, nvlabs_adv)
    assert torch.allclose(upstream_ret, nvlabs_ret)


def test_gdpo_invalid_baseline_mode_raises():
    token_level_rewards, response_mask, index, batch, non_tensor_batch = _make_gdpo_inputs()
    invalid_config = AlgoConfig(
        adv_estimator="gdpo",
        gdpo_baseline_mode="not_a_real_mode",
        gdpo_reward_keys=["accuracy_reward", "format_reward"],
    )

    with pytest.raises(ValueError, match="Unsupported GDPO baseline mode"):
        compute_gdpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            config=invalid_config,
            non_tensor_batch=non_tensor_batch,
            batch=batch,
        )


def test_rlla_compute_score_returns_named_rewards_for_qwen_tool_example():
    solution_str = (
        "<|im_start|>assistant\n"
        "<think>reason</think>\n"
        "<tool_call>\n"
        '{"name":"search","parameters":{"query":"weather"}}\n'
        "</tool_call><|im_end|>"
    )
    ground_truth = '<tool_call>\n{"name":"search","parameters":{"query":"weather"}}\n</tool_call>'

    result = compute_score(
        data_source="toolrl",
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info={
            "experiment_name": "qwen2.5-0.5B-gdpo-debug",
            "gdpo_baseline_mode": "nvlabs_reference",
        },
    )

    assert set(result.keys()) == {"score", "format_reward", "accuracy_reward"}
    assert result["score"] == result["format_reward"] + result["accuracy_reward"]
    assert result["format_reward"] == 1.0
    assert result["accuracy_reward"] == 3.0
