#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERL_PACKAGE_ROOT = PROJECT_ROOT / "external" / "verl"

import sys

if str(VERL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_PACKAGE_ROOT))

from tensordict import TensorDict

from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import compute_gdpo_outcome_advantage, compute_grpo_outcome_advantage
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import (
    _append_gdpo_saturation_events,
    _compute_gdpo_advantage_diagnostic_metrics,
    _compute_gdpo_saturation_metrics,
)

EPSILON = 1e-6
FLOAT_ATOL = 1e-6
FLOAT_RTOL = 1e-6


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    reward_values: dict[str, list[float]]
    index: list[int]
    response_mask: list[list[int]]
    reward_weights: dict[str, float] | None = None
    expected_relations: dict[str, Any] | None = None


def _manual_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_sum = (values * mask).sum()
    return masked_sum / (mask.sum() + 1e-8)


def _manual_masked_var(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _manual_masked_mean(values, mask)
    centered_squared = (values - mean) ** 2
    variance = _manual_masked_mean(centered_squared, mask)
    mask_sum = mask.sum()
    if mask_sum.item() == 0:
        raise ValueError("Mask must contain at least one valid token.")
    if mask_sum.item() == 1:
        raise ValueError("Mask must contain more than one valid token for unbiased variance.")
    return variance * (mask_sum / (mask_sum - 1))


def _manual_masked_whiten(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = _manual_masked_mean(values, mask)
    var = _manual_masked_var(values, mask)
    return (values - mean) * torch.rsqrt(var + 1e-8)


def _manual_masked_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    masked_values = torch.masked_select(values, mask.bool())
    if masked_values.numel() == 0:
        return {
            "mean": 0.0,
            "abs_mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "zero_fraction": 0.0,
        }

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


def _manual_unmasked_zero_fraction(values: torch.Tensor) -> float:
    return float((values == 0).float().mean().item())


def _manual_reward_stats(reward_values: list[float]) -> dict[str, float]:
    vals = np.asarray(reward_values, dtype=np.float32)
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _manual_saturation_summary(
    reward_values: list[float],
    index: list[int],
    reward_name: str,
    epsilon: float,
) -> tuple[dict[str, float | int | bool], list[dict[str, float | int | bool | str]]]:
    values = np.asarray(reward_values, dtype=np.float32)
    group_index = np.asarray(index, dtype=np.int64)
    unique_groups = list(dict.fromkeys(group_index.tolist()))

    saturated_group_count = 0
    all_zero_group_count = 0
    all_one_group_count = 0
    events: list[dict[str, float | int | bool | str]] = []

    for group_id in unique_groups:
        group_values = values[group_index == group_id]
        reward_mean = float(np.mean(group_values))
        reward_std = float(np.std(group_values))
        is_zero_std = bool(np.isclose(reward_std, 0.0, atol=epsilon))
        is_all_zero = bool(is_zero_std and np.all(np.isclose(group_values, 0.0, atol=epsilon)))
        is_all_one = bool(is_zero_std and np.all(np.isclose(group_values, 1.0, atol=epsilon)))
        if is_zero_std:
            saturated_group_count += 1
            all_zero_group_count += int(is_all_zero)
            all_one_group_count += int(is_all_one)
            events.append(
                {
                    "reward_name": reward_name,
                    "group_id": str(group_id),
                    "group_size": int(group_values.shape[0]),
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "is_zero_std": True,
                    "is_all_zero": is_all_zero,
                    "is_all_one": is_all_one,
                }
            )

    denom = float(len(unique_groups)) if unique_groups else 1.0
    summary = {
        "group_count": len(unique_groups),
        "saturated_group_count": saturated_group_count,
        "all_zero_group_count": all_zero_group_count,
        "all_one_group_count": all_one_group_count,
        "group_fraction": float(saturated_group_count / denom) if unique_groups else 0.0,
        "all_zero_fraction": float(all_zero_group_count / denom) if unique_groups else 0.0,
        "all_one_fraction": float(all_one_group_count / denom) if unique_groups else 0.0,
        "any": bool(saturated_group_count > 0),
    }
    return summary, events


def _build_batch_inputs(scenario: Scenario) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    response_mask = torch.tensor(scenario.response_mask, dtype=torch.float32)
    batch_size, response_length = response_mask.shape
    prompt_length = 1
    prompts = torch.ones((batch_size, prompt_length), dtype=torch.int64)
    attention_mask = torch.cat([prompts, response_mask.to(dtype=torch.int64)], dim=1)
    token_level_rewards = torch.zeros_like(response_mask)
    batch = {"prompts": prompts, "attention_mask": attention_mask}
    return response_mask, {"token_level_rewards": token_level_rewards, **batch}


def _build_config(scenario: Scenario) -> AlgoConfig:
    reward_keys = list(scenario.reward_values.keys())
    config_kwargs: dict[str, Any] = {
        "adv_estimator": "gdpo",
        "gdpo_baseline_mode": "upstream",
        "gdpo_reward_keys": reward_keys,
    }
    if scenario.reward_weights is not None:
        config_kwargs["gdpo_reward_weights"] = [float(scenario.reward_weights[key]) for key in reward_keys]
    return AlgoConfig(**config_kwargs)


def _derive_weighted_components(
    scenario: Scenario,
    response_mask: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: AlgoConfig,
) -> dict[str, torch.Tensor]:
    components: dict[str, torch.Tensor] = {}
    prompt_length = batch["prompts"].shape[1]
    valid_response_length = batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
    reward_keys = list(scenario.reward_values.keys())
    weights = scenario.reward_weights or {reward_key: 1.0 for reward_key in reward_keys}

    for reward_name in reward_keys:
        reward_tensor = torch.tensor(np.asarray(scenario.reward_values[reward_name], dtype=np.float32))
        reward_scores = torch.zeros_like(response_mask)
        reward_scores[torch.arange(reward_scores.shape[0]), valid_response_length] = reward_tensor
        normalized_component, _ = compute_grpo_outcome_advantage(
            token_level_rewards=reward_scores,
            response_mask=response_mask,
            index=np.asarray(scenario.index, dtype=np.int64),
            epsilon=EPSILON,
            norm_adv_by_std_in_grpo=True,
            config=config,
        )
        components[reward_name] = float(weights[reward_name]) * normalized_component
    return components


def _build_manual_advantage_events(
    saturation_events: list[dict[str, Any]],
    per_reward_components: dict[str, torch.Tensor],
    pre_whiten_total: torch.Tensor,
    post_whiten_total: torch.Tensor,
    response_mask: torch.Tensor,
    index: list[int],
) -> list[dict[str, float | int | str]]:
    group_ids = [str(group_id) for group_id in index]
    response_lengths = response_mask.sum(dim=-1)
    rows: list[dict[str, float | int | str]] = []
    for event in saturation_events:
        reward_name = str(event["reward_name"])
        group_id = str(event["group_id"])
        sample_indices = [idx for idx, current_group_id in enumerate(group_ids) if current_group_id == group_id]
        group_mask = response_mask[sample_indices]
        group_component = per_reward_components[reward_name][sample_indices]
        group_pre_total = pre_whiten_total[sample_indices]
        group_post_total = post_whiten_total[sample_indices]
        component_stats = _manual_masked_summary(group_component, group_mask)
        pre_total_stats = _manual_masked_summary(group_pre_total, group_mask)
        post_total_stats = _manual_masked_summary(group_post_total, group_mask)
        rows.append(
            {
                "reward_name": reward_name,
                "group_id": group_id,
                "group_size": int(event["group_size"]),
                "valid_token_count": int(group_mask.sum().item()),
                "response_length_mean": float(response_lengths[sample_indices].mean().item()),
                "pre_whiten_component_mean": component_stats["mean"],
                "pre_whiten_component_abs_mean": component_stats["abs_mean"],
                "pre_whiten_component_std": component_stats["std"],
                "pre_whiten_component_min": component_stats["min"],
                "pre_whiten_component_max": component_stats["max"],
                "pre_whiten_total_mean": pre_total_stats["mean"],
                "pre_whiten_total_abs_mean": pre_total_stats["abs_mean"],
                "pre_whiten_total_std": pre_total_stats["std"],
                "post_whiten_total_mean": post_total_stats["mean"],
                "post_whiten_total_abs_mean": post_total_stats["abs_mean"],
                "post_whiten_total_std": post_total_stats["std"],
            }
        )
    return rows


def _build_manual_saturation_info(scenario: Scenario) -> dict[str, Any]:
    per_reward: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    saturated_group_ids: set[str] = set()
    for reward_name, reward_values in scenario.reward_values.items():
        summary, reward_events = _manual_saturation_summary(reward_values, scenario.index, reward_name, EPSILON)
        per_reward[reward_name] = summary
        events.extend(reward_events)
        saturated_group_ids.update(str(event["group_id"]) for event in reward_events)
    return {
        "per_reward": per_reward,
        "events": events,
        "group_count": len(set(scenario.index)),
        "any_reward_group_fraction": float(len(saturated_group_ids) / float(len(set(scenario.index))))
        if scenario.index
        else 0.0,
    }


def _build_tensorboard_metric_dict(
    scenario: Scenario,
    batch_data: DataProto,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics.update(compute_data_metrics(batch=batch_data, use_critic=False))
    for reward_name, reward_values in scenario.reward_values.items():
        stats = _manual_reward_stats(reward_values)
        metrics[f"gdpo/{reward_name}/mean"] = stats["mean"]
        metrics[f"gdpo/{reward_name}/std"] = stats["std"]
        metrics[f"gdpo/{reward_name}/max"] = stats["max"]
        metrics[f"gdpo/{reward_name}/min"] = stats["min"]
    metrics.update(_compute_gdpo_saturation_metrics(batch_data.meta_info["gdpo_saturation"]))
    metrics.update(_compute_gdpo_advantage_diagnostic_metrics(batch_data.meta_info["gdpo_advantage_diagnostics"]))
    return metrics


def _build_expected_metric_dict(
    scenario: Scenario,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    saturation_info: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    valid_advantages = torch.masked_select(advantages, response_mask.bool())
    valid_returns = torch.masked_select(returns, response_mask.bool())
    response_lengths = response_mask.sum(dim=-1)
    max_response_length = response_mask.shape[1]
    prompt_lengths = torch.ones(response_mask.shape[0], dtype=torch.float32)

    metrics["critic/score/mean"] = 0.0
    metrics["critic/score/max"] = 0.0
    metrics["critic/score/min"] = 0.0
    metrics["critic/rewards/mean"] = 0.0
    metrics["critic/rewards/max"] = 0.0
    metrics["critic/rewards/min"] = 0.0
    metrics["critic/advantages/mean"] = float(valid_advantages.mean().item())
    metrics["critic/advantages/max"] = float(valid_advantages.max().item())
    metrics["critic/advantages/min"] = float(valid_advantages.min().item())
    metrics["critic/returns/mean"] = float(valid_returns.mean().item())
    metrics["critic/returns/max"] = float(valid_returns.max().item())
    metrics["critic/returns/min"] = float(valid_returns.min().item())
    metrics["response_length/mean"] = float(response_lengths.mean().item())
    metrics["response_length/max"] = float(response_lengths.max().item())
    metrics["response_length/min"] = float(response_lengths.min().item())
    metrics["response_length/clip_ratio"] = float((response_lengths == max_response_length).float().mean().item())
    metrics["response_length_non_aborted/mean"] = float(response_lengths.mean().item())
    metrics["response_length_non_aborted/max"] = float(response_lengths.max().item())
    metrics["response_length_non_aborted/min"] = float(response_lengths.min().item())
    metrics["response_length_non_aborted/clip_ratio"] = float(
        (response_lengths == max_response_length).float().mean().item()
    )
    metrics["response/aborted_ratio"] = 0.0
    metrics["prompt_length/mean"] = float(prompt_lengths.mean().item())
    metrics["prompt_length/max"] = float(prompt_lengths.max().item())
    metrics["prompt_length/min"] = float(prompt_lengths.min().item())
    metrics["prompt_length/clip_ratio"] = 1.0

    for reward_name, reward_values in scenario.reward_values.items():
        stats = _manual_reward_stats(reward_values)
        metrics[f"gdpo/{reward_name}/mean"] = stats["mean"]
        metrics[f"gdpo/{reward_name}/std"] = stats["std"]
        metrics[f"gdpo/{reward_name}/max"] = stats["max"]
        metrics[f"gdpo/{reward_name}/min"] = stats["min"]

        summary = saturation_info["per_reward"][reward_name]
        metrics[f"gdpo_saturation/{reward_name}/group_fraction"] = float(summary["group_fraction"])
        metrics[f"gdpo_saturation/{reward_name}/all_zero_fraction"] = float(summary["all_zero_fraction"])
        metrics[f"gdpo_saturation/{reward_name}/all_one_fraction"] = float(summary["all_one_fraction"])
        metrics[f"gdpo_saturation/{reward_name}/any"] = float(bool(summary["any"]))

        component_summary = diagnostics["per_reward"][reward_name]
        prefix = f"gdpo_advantage/pre_whiten/{reward_name}"
        for metric_name, metric_value in component_summary.items():
            metrics[f"{prefix}/{metric_name}"] = float(metric_value)

    metrics["gdpo_saturation/any_reward_group_fraction"] = float(saturation_info["any_reward_group_fraction"])

    for total_name in ("pre_whiten_total", "post_whiten_total"):
        prefix = f"gdpo_advantage/{total_name}"
        for metric_name, metric_value in diagnostics[total_name].items():
            metrics[f"{prefix}/{metric_name}"] = float(metric_value)

    return metrics


def _assert_close(name: str, actual: float, expected: float, failures: list[str]) -> None:
    if not math.isclose(actual, expected, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
        failures.append(f"{name}: actual={actual} expected={expected}")


def _assert_metric_dict_matches(
    actual: dict[str, float],
    expected: dict[str, float],
    failures: list[str],
) -> None:
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    if actual_keys != expected_keys:
        failures.append(
            "metric key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)} extra={sorted(actual_keys - expected_keys)}"
        )
        return
    for key in sorted(expected_keys):
        _assert_close(key, float(actual[key]), float(expected[key]), failures)


def _sort_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row["reward_name"]), str(row["group_id"])))


def _assert_event_rows_match(
    actual_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    failures: list[str],
    *,
    record_name: str,
) -> None:
    actual_sorted = _sort_event_rows(actual_rows)
    expected_sorted = _sort_event_rows(expected_rows)
    if len(actual_sorted) != len(expected_sorted):
        failures.append(
            f"{record_name}: row-count mismatch actual={len(actual_sorted)} expected={len(expected_sorted)}"
        )
        return

    for actual, expected in zip(actual_sorted, expected_sorted, strict=True):
        if set(actual.keys()) != set(expected.keys()):
            failures.append(
                f"{record_name}: key mismatch for {(expected['reward_name'], expected['group_id'])}: "
                f"actual={sorted(actual.keys())} expected={sorted(expected.keys())}"
            )
            continue
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if isinstance(expected_value, float):
                _assert_close(
                    f"{record_name}:{expected['reward_name']}:{expected['group_id']}:{key}",
                    float(actual_value),
                    expected_value,
                    failures,
                )
            else:
                if actual_value != expected_value:
                    failures.append(
                        f"{record_name}:{expected['reward_name']}:{expected['group_id']}:{key}: "
                        f"actual={actual_value} expected={expected_value}"
                    )


def _scenario_sidecar_rows(
    saturation_events: list[dict[str, Any]],
    advantage_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="gdpo-proof-") as tmp_dir:
        log_path = Path(tmp_dir) / "events.jsonl"
        _append_gdpo_saturation_events(
            log_path=str(log_path),
            step=7,
            events=saturation_events,
            actor_grad_norm=0.0,
            advantage_events=advantage_events,
        )
        if not log_path.exists():
            return []
        return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]


def _expected_sidecar_rows(
    saturation_events: list[dict[str, Any]],
    advantage_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    advantage_by_key = {(row["reward_name"], row["group_id"]): row for row in advantage_events}
    rows: list[dict[str, Any]] = []
    for event in saturation_events:
        row = {
            "step": 7,
            "reward_name": event["reward_name"],
            "group_id": event["group_id"],
            "group_size": event["group_size"],
            "reward_mean": event["reward_mean"],
            "reward_std": event["reward_std"],
            "is_zero_std": event["is_zero_std"],
            "is_all_zero": event["is_all_zero"],
            "is_all_one": event["is_all_one"],
            "actor_grad_norm": 0.0,
        }
        advantage_event = advantage_by_key.get((event["reward_name"], event["group_id"]))
        if advantage_event is not None:
            row.update(advantage_event)
        rows.append(row)
    return rows


def _run_scenario(scenario: Scenario) -> dict[str, Any]:
    response_mask, batch_inputs = _build_batch_inputs(scenario)
    config = _build_config(scenario)
    meta_info: dict[str, Any] = {}
    non_tensor_batch = {
        reward_name: np.asarray(reward_values, dtype=np.float32)
        for reward_name, reward_values in scenario.reward_values.items()
    }

    advantages, returns = compute_gdpo_outcome_advantage(
        token_level_rewards=batch_inputs["token_level_rewards"],
        response_mask=response_mask,
        index=np.asarray(scenario.index, dtype=np.int64),
        epsilon=EPSILON,
        norm_adv_by_std_in_grpo=True,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch={"prompts": batch_inputs["prompts"], "attention_mask": batch_inputs["attention_mask"]},
        meta_info=meta_info,
    )

    per_reward_components = _derive_weighted_components(
        scenario=scenario,
        response_mask=response_mask,
        batch={"prompts": batch_inputs["prompts"], "attention_mask": batch_inputs["attention_mask"]},
        config=config,
    )
    pre_whiten_total = sum(per_reward_components.values())
    manual_post_whiten = _manual_masked_whiten(pre_whiten_total, response_mask) * response_mask

    manual_saturation_info = _build_manual_saturation_info(scenario)
    manual_advantage_diagnostics = {
        "per_reward": {
            reward_name: _manual_masked_summary(component, response_mask)
            for reward_name, component in per_reward_components.items()
        },
        "pre_whiten_total": _manual_masked_summary(pre_whiten_total, response_mask),
        "post_whiten_total": _manual_masked_summary(manual_post_whiten, response_mask),
        "events": _build_manual_advantage_events(
            saturation_events=manual_saturation_info["events"],
            per_reward_components=per_reward_components,
            pre_whiten_total=pre_whiten_total,
            post_whiten_total=manual_post_whiten,
            response_mask=response_mask,
            index=scenario.index,
        ),
    }

    token_level_scores = torch.zeros_like(response_mask)
    token_level_rewards = torch.zeros_like(response_mask)
    batch_proto = DataProto(
        batch=TensorDict(
            {
                "responses": torch.zeros(response_mask.shape, dtype=torch.int64),
                "attention_mask": batch_inputs["attention_mask"],
                "response_mask": response_mask.to(dtype=torch.int64),
                "token_level_scores": token_level_scores,
                "token_level_rewards": token_level_rewards,
                "advantages": advantages,
                "returns": returns,
            },
            batch_size=[response_mask.shape[0]],
        ),
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info,
    )

    logged_metrics = _build_tensorboard_metric_dict(scenario, batch_proto)
    expected_metrics = _build_expected_metric_dict(
        scenario=scenario,
        response_mask=response_mask,
        advantages=advantages,
        returns=returns,
        saturation_info=manual_saturation_info,
        diagnostics=manual_advantage_diagnostics,
    )

    actual_sidecar_rows = _scenario_sidecar_rows(
        saturation_events=meta_info["gdpo_saturation"]["events"],
        advantage_events=meta_info["gdpo_advantage_diagnostics"]["events"],
    )
    expected_sidecar_rows = _expected_sidecar_rows(
        saturation_events=manual_saturation_info["events"],
        advantage_events=manual_advantage_diagnostics["events"],
    )

    failures: list[str] = []

    if not torch.allclose(advantages, returns, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
        failures.append("advantages and returns diverged, but GDPO outcome-only should return identical tensors.")

    if not torch.allclose(advantages, manual_post_whiten, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
        failures.append("final advantages did not match manual post-whiten derivation.")

    _assert_metric_dict_matches(logged_metrics, expected_metrics, failures)
    _assert_event_rows_match(
        actual_rows=meta_info["gdpo_advantage_diagnostics"]["events"],
        expected_rows=manual_advantage_diagnostics["events"],
        failures=failures,
        record_name="advantage_events",
    )
    _assert_event_rows_match(
        actual_rows=actual_sidecar_rows,
        expected_rows=expected_sidecar_rows,
        failures=failures,
        record_name="sidecar_rows",
    )

    if scenario.expected_relations:
        for relation_name, expected_value in scenario.expected_relations.items():
            if relation_name == "group0_post_whiten_nonzero":
                group0_event = next(
                    row
                    for row in manual_advantage_diagnostics["events"]
                    if row["reward_name"] == "correct_reward" and row["group_id"] == "0"
                )
                if expected_value and group0_event["post_whiten_total_abs_mean"] <= 0.0:
                    failures.append("expected saturated group 0 to have nonzero post-whiten signal, but it remained zero.")
                if (not expected_value) and group0_event["post_whiten_total_abs_mean"] != 0.0:
                    failures.append("expected saturated group 0 to stay zero post-whiten, but it became nonzero.")
            elif relation_name == "any_reward_group_fraction":
                actual_value = meta_info["gdpo_saturation"]["any_reward_group_fraction"]
                _assert_close("any_reward_group_fraction", float(actual_value), float(expected_value), failures)
            elif relation_name == "correct_weight_dominates_format_abs_mean":
                correct_abs_mean = manual_advantage_diagnostics["per_reward"]["correct_reward"]["abs_mean"]
                format_abs_mean = manual_advantage_diagnostics["per_reward"]["format_reward"]["abs_mean"]
                if bool(expected_value) and not (correct_abs_mean > format_abs_mean):
                    failures.append(
                        "expected weighted correct_reward pre-whiten abs_mean to dominate format_reward, but it did not."
                    )
            elif relation_name == "masked_zero_fraction_differs_from_unmasked":
                reward_name = str(expected_value)
                masked_fraction = manual_advantage_diagnostics["per_reward"][reward_name]["zero_fraction"]
                unmasked_fraction = _manual_unmasked_zero_fraction(per_reward_components[reward_name])
                if math.isclose(masked_fraction, unmasked_fraction, rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
                    failures.append(
                        f"expected masked and unmasked zero_fraction to differ for {reward_name}, but they were equal."
                    )
            else:
                failures.append(f"Unknown scenario relation: {relation_name}")

    highlight_metrics = {
        key: logged_metrics[key]
        for key in sorted(logged_metrics.keys())
        if key.startswith("gdpo_advantage/") or key.startswith("gdpo_saturation/")
    }
    return {
        "name": scenario.name,
        "description": scenario.description,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "reward_keys": list(scenario.reward_values.keys()),
        "sample_count": len(scenario.index),
        "response_length": len(scenario.response_mask[0]),
        "highlight_metrics": highlight_metrics,
        "advantage_events": meta_info["gdpo_advantage_diagnostics"]["events"],
        "sidecar_rows": actual_sidecar_rows,
    }


def _default_scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="equal_lengths_single_reward",
            description=(
                "Single-reward GDPO case with a saturated all-zero group and equal response lengths. "
                "This proves saturated groups are zero before whitening and stay zero after whitening when token lengths match."
            ),
            reward_values={
                "correct_reward": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            },
            index=[0, 0, 0, 0, 1, 1, 1, 1],
            response_mask=[[1], [1], [1], [1], [1], [1], [1], [1]],
            expected_relations={"group0_post_whiten_nonzero": False},
        ),
        Scenario(
            name="uneven_lengths_single_reward",
            description=(
                "Single-reward GDPO case with the same raw rewards, but uneven response lengths. "
                "This proves a saturated group can still receive nonzero final post-whiten optimizer signal."
            ),
            reward_values={
                "correct_reward": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            },
            index=[0, 0, 0, 0, 1, 1, 1, 1],
            response_mask=[
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            expected_relations={
                "group0_post_whiten_nonzero": True,
                "masked_zero_fraction_differs_from_unmasked": "correct_reward",
            },
        ),
        Scenario(
            name="two_reward_disjoint_saturation_union",
            description=(
                "Two-reward case with disjoint saturation patterns. "
                "This proves per-reward saturation fractions and the union-based any_reward_group_fraction are tracked correctly."
            ),
            reward_values={
                "correct_reward": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "format_reward": [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            index=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            response_mask=[[1]] * 12,
            expected_relations={"any_reward_group_fraction": 1.0},
        ),
        Scenario(
            name="weighted_two_reward_masked_case",
            description=(
                "Two-reward case with asymmetric reward weights and uneven response masks. "
                "This proves the pre-whiten per-reward metrics reflect weighted components and ignore masked-out tokens."
            ),
            reward_values={
                "correct_reward": [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                "format_reward": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            },
            index=[0, 0, 0, 0, 1, 1, 1, 1],
            response_mask=[
                [1, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
            ],
            reward_weights={"correct_reward": 2.5, "format_reward": 0.25},
            expected_relations={
                "correct_weight_dominates_format_abs_mean": True,
                "masked_zero_fraction_differs_from_unmasked": "correct_reward",
            },
        ),
    ]


def _build_report(scenarios: list[Scenario]) -> dict[str, Any]:
    scenario_reports = [_run_scenario(scenario) for scenario in scenarios]
    return {
        "purpose": (
            "Verify, from the actual vendored verl code path, that GDPO advantage-related TensorBoard metrics and "
            "saturation sidecar rows match independently derived expectations."
        ),
        "repo_root": str(PROJECT_ROOT),
        "verified_metric_families": [
            "gdpo/<reward>/{mean,std,max,min}",
            "gdpo_saturation/<reward>/{group_fraction,all_zero_fraction,all_one_fraction,any}",
            "gdpo_saturation/any_reward_group_fraction",
            "gdpo_advantage/pre_whiten/<reward>/{mean,abs_mean,std,min,max,zero_fraction}",
            "gdpo_advantage/pre_whiten_total/{mean,abs_mean,std,min,max,zero_fraction}",
            "gdpo_advantage/post_whiten_total/{mean,abs_mean,std,min,max,zero_fraction}",
            "critic/score/{mean,max,min}",
            "critic/rewards/{mean,max,min}",
            "critic/advantages/{mean,max,min}",
            "critic/returns/{mean,max,min}",
            "response_length/*",
            "response_length_non_aborted/*",
            "response/aborted_ratio",
            "prompt_length/*",
        ],
        "scenario_count": len(scenario_reports),
        "passed_scenario_count": sum(1 for scenario in scenario_reports if scenario["passed"]),
        "failed_scenario_count": sum(1 for scenario in scenario_reports if not scenario["passed"]),
        "all_checks_passed": all(scenario["passed"] for scenario in scenario_reports),
        "scenarios": scenario_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exhaustively verify GDPO advantage and saturation TensorBoard metrics against independent derivations."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full JSON proof report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the pretty-printed JSON report on stdout.",
    )
    args = parser.parse_args()

    report = _build_report(_default_scenarios())
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))

    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
