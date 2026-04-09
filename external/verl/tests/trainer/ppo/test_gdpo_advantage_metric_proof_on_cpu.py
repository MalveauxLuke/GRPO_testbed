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

import json

import numpy as np
import pytest
import torch
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import compute_gdpo_outcome_advantage, compute_grpo_outcome_advantage
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import (
    _append_gdpo_saturation_events,
    _compute_gdpo_advantage_diagnostic_metrics,
    _compute_gdpo_saturation_metrics,
)


def _manual_masked_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    masked_values = torch.masked_select(values, mask.bool())
    if masked_values.numel() > 1:
        std = masked_values.std(unbiased=True)
    else:
        std = torch.tensor(0.0, dtype=masked_values.dtype)
    return {
        "mean": float(masked_values.mean().item()),
        "abs_mean": float(masked_values.abs().mean().item()),
        "std": float(std.item()),
        "min": float(masked_values.min().item()),
        "max": float(masked_values.max().item()),
        "zero_fraction": float((masked_values == 0).float().mean().item()),
    }


def _build_case(*, reward_values, index, response_mask, reward_weights=None):
    response_mask_t = torch.tensor(response_mask, dtype=torch.float32)
    batch_size = response_mask_t.shape[0]
    prompt_len = 1
    token_level_rewards = torch.zeros_like(response_mask_t)
    attention_mask = torch.cat(
        [
            torch.ones((batch_size, prompt_len), dtype=torch.int64),
            response_mask_t.to(dtype=torch.int64),
        ],
        dim=1,
    )
    batch = {
        "prompts": torch.ones((batch_size, prompt_len), dtype=torch.int64),
        "attention_mask": attention_mask,
    }
    config_kwargs = {
        "adv_estimator": "gdpo",
        "gdpo_baseline_mode": "upstream",
        "gdpo_reward_keys": list(reward_values.keys()),
    }
    if reward_weights is not None:
        config_kwargs["gdpo_reward_weights"] = [float(reward_weights[key]) for key in reward_values.keys()]
    config = AlgoConfig(**config_kwargs)
    non_tensor_batch = {
        reward_name: np.asarray(values, dtype=np.float32)
        for reward_name, values in reward_values.items()
    }
    meta_info = {}
    advantages, returns = compute_gdpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask_t,
        index=np.asarray(index, dtype=np.int64),
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
        meta_info=meta_info,
    )
    return advantages, returns, response_mask_t, batch, non_tensor_batch, meta_info, config


def _derive_weighted_components(
    *,
    reward_values,
    reward_weights,
    response_mask,
    batch,
    config,
    index,
):
    prompt_len = batch["prompts"].shape[1]
    valid_response_length = batch["attention_mask"][:, prompt_len:].sum(dim=1) - 1
    weights = reward_weights or {reward_name: 1.0 for reward_name in reward_values.keys()}
    components = {}
    for reward_name, values in reward_values.items():
        reward_tensor = torch.tensor(np.asarray(values, dtype=np.float32))
        reward_scores = torch.zeros_like(response_mask)
        reward_scores[torch.arange(reward_scores.shape[0]), valid_response_length] = reward_tensor
        normalized_component, _ = compute_grpo_outcome_advantage(
            token_level_rewards=reward_scores,
            response_mask=response_mask,
            index=np.asarray(index, dtype=np.int64),
            config=config,
        )
        components[reward_name] = float(weights[reward_name]) * normalized_component
    return components


def test_weighted_multi_reward_gdpo_diagnostics_match_manual_derivation():
    reward_values = {
        "correct_reward": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
        "format_reward": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    }
    reward_weights = {"correct_reward": 2.5, "format_reward": 0.25}
    index = [0, 0, 0, 0, 1, 1, 1, 1]
    response_mask = [
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 0],
    ]

    advantages, returns, response_mask_t, batch, _, meta_info, config = _build_case(
        reward_values=reward_values,
        reward_weights=reward_weights,
        index=index,
        response_mask=response_mask,
    )

    weighted_components = _derive_weighted_components(
        reward_values=reward_values,
        reward_weights=reward_weights,
        response_mask=response_mask_t,
        batch=batch,
        config=config,
        index=index,
    )
    pre_whiten_total = weighted_components["correct_reward"] + weighted_components["format_reward"]
    post_whiten_total = verl_F.masked_whiten(pre_whiten_total, response_mask_t) * response_mask_t

    assert torch.allclose(advantages, returns)
    assert torch.allclose(advantages, post_whiten_total)

    diagnostics = meta_info["gdpo_advantage_diagnostics"]
    assert diagnostics["per_reward"]["correct_reward"]["abs_mean"] == pytest.approx(
        _manual_masked_summary(weighted_components["correct_reward"], response_mask_t)["abs_mean"]
    )
    assert diagnostics["per_reward"]["format_reward"]["abs_mean"] == pytest.approx(
        _manual_masked_summary(weighted_components["format_reward"], response_mask_t)["abs_mean"]
    )
    assert diagnostics["pre_whiten_total"]["std"] == pytest.approx(
        _manual_masked_summary(pre_whiten_total, response_mask_t)["std"]
    )
    assert diagnostics["post_whiten_total"]["abs_mean"] == pytest.approx(
        _manual_masked_summary(post_whiten_total, response_mask_t)["abs_mean"]
    )
    assert diagnostics["per_reward"]["correct_reward"]["abs_mean"] > diagnostics["per_reward"]["format_reward"]["abs_mean"]


def test_gdpo_metric_projection_and_sidecar_match_actual_advantages(tmp_path):
    reward_values = {
        "correct_reward": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "format_reward": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    }
    index = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    response_mask = [[1]] * 12

    advantages, returns, response_mask_t, batch, non_tensor_batch, meta_info, _ = _build_case(
        reward_values=reward_values,
        index=index,
        response_mask=response_mask,
    )

    batch_proto = DataProto(
        batch=TensorDict(
            {
                "responses": torch.zeros(response_mask_t.shape, dtype=torch.int64),
                "attention_mask": batch["attention_mask"],
                "response_mask": response_mask_t.to(dtype=torch.int64),
                "token_level_scores": torch.zeros_like(response_mask_t),
                "token_level_rewards": torch.zeros_like(response_mask_t),
                "advantages": advantages,
                "returns": returns,
            },
            batch_size=[response_mask_t.shape[0]],
        ),
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info,
    )

    data_metrics = compute_data_metrics(batch=batch_proto, use_critic=False)
    saturation_metrics = _compute_gdpo_saturation_metrics(meta_info["gdpo_saturation"])
    advantage_metrics = _compute_gdpo_advantage_diagnostic_metrics(meta_info["gdpo_advantage_diagnostics"])

    valid_adv = torch.masked_select(advantages, response_mask_t.bool())
    assert data_metrics["critic/advantages/mean"] == pytest.approx(float(valid_adv.mean().item()))
    assert data_metrics["critic/returns/max"] == pytest.approx(float(valid_adv.max().item()))

    assert saturation_metrics["gdpo_saturation/correct_reward/group_fraction"] == pytest.approx(2.0 / 3.0)
    assert saturation_metrics["gdpo_saturation/format_reward/group_fraction"] == pytest.approx(2.0 / 3.0)
    assert saturation_metrics["gdpo_saturation/any_reward_group_fraction"] == pytest.approx(1.0)
    assert advantage_metrics["gdpo_advantage/post_whiten_total/abs_mean"] == pytest.approx(
        meta_info["gdpo_advantage_diagnostics"]["post_whiten_total"]["abs_mean"]
    )

    log_path = tmp_path / "gdpo-events.jsonl"
    _append_gdpo_saturation_events(
        str(log_path),
        step=11,
        events=meta_info["gdpo_saturation"]["events"],
        actor_grad_norm=0.25,
        advantage_events=meta_info["gdpo_advantage_diagnostics"]["events"],
    )
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 4
    group0_correct = next(row for row in rows if row["reward_name"] == "correct_reward" and row["group_id"] == "0")
    assert group0_correct["step"] == 11
    assert group0_correct["actor_grad_norm"] == 0.25
    assert group0_correct["pre_whiten_component_abs_mean"] == pytest.approx(0.0)
