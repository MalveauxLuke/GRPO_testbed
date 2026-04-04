#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from statistics import pstdev


def _parse_reward_arg(arg: str) -> tuple[str, list[str]]:
    if "=" not in arg:
        raise ValueError(f"Expected NAME=pattern,pattern,..., got: {arg}")
    name, pattern_csv = arg.split("=", 1)
    patterns = [pattern.strip() for pattern in pattern_csv.split(",") if pattern.strip()]
    if not name or not patterns:
        raise ValueError(f"Expected NAME=pattern,pattern,..., got: {arg}")
    for pattern in patterns:
        if any(char not in {"0", "1"} for char in pattern):
            raise ValueError(f"Pattern must contain only 0/1 characters, got: {pattern}")
    return name, patterns


def _summarize_patterns(patterns: list[str], reward_name: str) -> tuple[dict, list[dict]]:
    num_groups = len(patterns)
    saturated_group_count = 0
    all_zero_group_count = 0
    all_one_group_count = 0
    events = []

    for group_id, pattern in enumerate(patterns):
        values = [int(char) for char in pattern]
        reward_mean = float(sum(values) / len(values))
        reward_std = float(pstdev(values)) if len(values) > 1 else 0.0
        is_zero_std = reward_std == 0.0
        is_all_zero = is_zero_std and all(value == 0 for value in values)
        is_all_one = is_zero_std and all(value == 1 for value in values)

        if is_zero_std:
            saturated_group_count += 1
            all_zero_group_count += int(is_all_zero)
            all_one_group_count += int(is_all_one)
            events.append(
                {
                    "reward_name": reward_name,
                    "group_id": str(group_id),
                    "pattern": pattern,
                    "group_size": len(pattern),
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "is_zero_std": True,
                    "is_all_zero": is_all_zero,
                    "is_all_one": is_all_one,
                }
            )

    denom = float(num_groups) if num_groups > 0 else 1.0
    return (
        {
            "group_count": num_groups,
            "saturated_group_count": saturated_group_count,
            "all_zero_group_count": all_zero_group_count,
            "all_one_group_count": all_one_group_count,
            "group_fraction": float(saturated_group_count / denom) if num_groups > 0 else 0.0,
            "all_zero_fraction": float(all_zero_group_count / denom) if num_groups > 0 else 0.0,
            "all_one_fraction": float(all_one_group_count / denom) if num_groups > 0 else 0.0,
            "any": bool(saturated_group_count > 0),
        },
        events,
    )


def _build_metrics(per_reward: dict, any_reward_group_fraction: float) -> dict[str, float]:
    metrics = {}
    for reward_name, summary in per_reward.items():
        metrics[f"gdpo_saturation/{reward_name}/group_fraction"] = summary["group_fraction"]
        metrics[f"gdpo_saturation/{reward_name}/all_zero_fraction"] = summary["all_zero_fraction"]
        metrics[f"gdpo_saturation/{reward_name}/all_one_fraction"] = summary["all_one_fraction"]
        metrics[f"gdpo_saturation/{reward_name}/any"] = float(bool(summary["any"]))
    metrics["gdpo_saturation/any_reward_group_fraction"] = any_reward_group_fraction
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GDPO saturation metrics for 0/1 reward-group patterns.")
    parser.add_argument(
        "--reward",
        action="append",
        help="Reward definition in NAME=pattern,pattern,... form. Example: correct_reward=0000,1111,0101",
    )
    args = parser.parse_args()

    reward_patterns = {}
    if args.reward:
        for raw_arg in args.reward:
            name, patterns = _parse_reward_arg(raw_arg)
            reward_patterns[name] = patterns
    else:
        reward_patterns = {
            "correct_reward": ["0000", "1111", "0101"],
            "format_reward": ["0101", "1111", "1111"],
        }

    per_reward = {}
    events = []
    saturated_group_ids = set()
    for reward_name, patterns in reward_patterns.items():
        summary, reward_events = _summarize_patterns(patterns=patterns, reward_name=reward_name)
        per_reward[reward_name] = summary
        events.extend(reward_events)
        saturated_group_ids.update(event["group_id"] for event in reward_events)

    group_count = max((len(patterns) for patterns in reward_patterns.values()), default=0)
    any_reward_group_fraction = float(len(saturated_group_ids) / group_count) if group_count > 0 else 0.0

    payload = {
        "reward_patterns": reward_patterns,
        "per_reward": per_reward,
        "events": events,
        "any_reward_group_fraction": any_reward_group_fraction,
        "metrics": _build_metrics(per_reward=per_reward, any_reward_group_fraction=any_reward_group_fraction),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
