#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from audit_gsm8k_modern_rewards import (  # type: ignore[import-not-found]
    DEFAULT_REWARD_MODULE_PATH,
    load_canonical_samples,
    load_module_from_path,
    recompute_sample,
)


def _stable_key_component(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _preview(text: Any, width: int = 140) -> str:
    if text is None:
        return "<missing>"
    normalized = " ".join(str(text).split())
    if len(normalized) <= width:
        return normalized
    return normalized[: width - 3] + "..."


def _group_key(sample: dict[str, Any]) -> tuple[Any, str, str]:
    return (
        sample.get("step"),
        _stable_key_component(sample.get("prompt")),
        str(sample.get("ground_truth")),
    )


def _count_correct(group: list[dict[str, Any]]) -> tuple[int, int]:
    correct = sum(int(float(sample["recomputed"].get("correct_reward") or 0.0) == 1.0) for sample in group)
    incorrect = len(group) - correct
    return correct, incorrect


def _matching_groups(
    groups: list[list[dict[str, Any]]],
    *,
    group_size: int | None,
    correct_count: int | None,
    incorrect_count: int | None,
) -> list[list[dict[str, Any]]]:
    matches: list[list[dict[str, Any]]] = []
    for group in groups:
        total = len(group)
        correct, incorrect = _count_correct(group)
        if group_size is not None and total != group_size:
            continue
        if correct_count is not None and correct != correct_count:
            continue
        if incorrect_count is not None and incorrect != incorrect_count:
            continue
        if correct_count is None and incorrect_count is None:
            if not (total >= 2 and incorrect == 1 and correct == total - 1):
                continue
        matches.append(group)
    return matches


def _print_group(group: list[dict[str, Any]], *, group_index: int, show_prompt: bool) -> None:
    ordered_group = sorted(
        group,
        key=lambda sample: (
            int(float(sample["recomputed"].get("correct_reward") or 0.0) == 1.0),
            sample.get("row_index", 0),
        ),
    )
    correct_count, incorrect_count = _count_correct(ordered_group)
    first = ordered_group[0]

    print("=" * 120)
    print(
        f"GROUP {group_index}: step={first.get('step')} total={len(ordered_group)} "
        f"correct={correct_count} incorrect={incorrect_count} ground_truth={first.get('ground_truth')!r}"
    )
    print(f"prompt_preview={_preview(first.get('prompt'))}")
    if show_prompt:
        print("\nPROMPT:\n")
        print(first.get("prompt"))

    for rollout_index, sample in enumerate(ordered_group, start=1):
        recomputed = sample["recomputed"]
        print("\n" + "-" * 120)
        print(
            f"ROLLOUT {rollout_index}: correct={recomputed.get('correct_reward')} "
            f"format={recomputed.get('format_reward')} strict_format={recomputed.get('strict_format_reward')} "
            f"answer_parse_ok={recomputed.get('answer_parse_ok')} bucket={sample.get('behavior_bucket')}"
        )
        print(
            f"logged_correct={sample['logged'].get('correct_reward')} "
            f"logged_score={sample['logged'].get('score')} "
            f"source={Path(sample['source_path']).name} row_index={sample.get('row_index')}"
        )
        print(
            f"parsed_answer={recomputed.get('parsed_answer')!r} "
            f"expected_answer={recomputed.get('expected_answer')!r} "
            f"mismatch_fields={sample.get('mismatch_fields')}"
        )
        print("\nOUTPUT:\n")
        print(sample.get("output", ""))

    print("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print full rollout groups from GSM8K-modern dumps, ideally a group with all-correct-but-one-incorrect "
            "under the current reward contract."
        )
    )
    parser.add_argument("--input-path", required=True, help="Rollout JSONL file or rollout dump directory.")
    parser.add_argument("--input-type", choices=("auto", "rollout"), default="rollout")
    parser.add_argument("--reward-module-path", default=str(DEFAULT_REWARD_MODULE_PATH))
    parser.add_argument("--step", type=int, default=None, help="Restrict to a single rollout step.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Require an exact rollout group size. Leave unset to accept any size.",
    )
    parser.add_argument(
        "--correct-count",
        type=int,
        default=None,
        help="Require an exact number of correct rollouts in the printed group.",
    )
    parser.add_argument(
        "--incorrect-count",
        type=int,
        default=None,
        help="Require an exact number of incorrect rollouts in the printed group.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=1,
        help="Print at most this many matching groups.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle matching groups before printing so you can sample different matches reproducibly.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the full prompt above each rollout group instead of only a preview.",
    )
    args = parser.parse_args()

    reward_module = load_module_from_path("gsm8k_modern_two_reward", args.reward_module_path)
    canonical_samples = load_canonical_samples(args.input_path, input_type=args.input_type, step=args.step)
    recomputed_samples = [recompute_sample(sample, reward_module) for sample in canonical_samples]

    grouped_samples: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in recomputed_samples:
        grouped_samples[_group_key(sample)].append(sample)

    groups = list(grouped_samples.values())
    groups.sort(
        key=lambda group: (
            group[0].get("step") if group[0].get("step") is not None else -1,
            _preview(group[0].get("prompt")),
            str(group[0].get("ground_truth")),
        )
    )

    matches = _matching_groups(
        groups,
        group_size=args.group_size,
        correct_count=args.correct_count,
        incorrect_count=args.incorrect_count,
    )
    if not matches:
        raise SystemExit(
            "No matching rollout groups found. Try relaxing --group-size / --correct-count / --incorrect-count."
        )

    rng = random.Random(args.seed)
    rng.shuffle(matches)

    print(
        f"Found {len(matches)} matching groups out of {len(groups)} total groups. "
        f"Printing {min(args.max_groups, len(matches))} group(s)."
    )
    print(
        "Match rule: "
        + (
            f"group_size={args.group_size} correct_count={args.correct_count} incorrect_count={args.incorrect_count}"
            if args.correct_count is not None or args.incorrect_count is not None
            else "all-but-one-correct"
        )
    )

    for group_index, group in enumerate(matches[: args.max_groups], start=1):
        _print_group(group, group_index=group_index, show_prompt=args.show_prompt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
