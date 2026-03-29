#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from verl.utils.reward_score.deepscaler_math_length import compute_score, extract_last_boxed_answer


COMPARE_FIELDS = (
    "score",
    "correct_reward",
    "length_reward",
    "answer_parse_ok",
    "response_length_tokens",
    "length_limit_tokens",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object rows in {path}, got: {type(payload)!r}")
            rows.append(payload)
    return rows


def detect_input_type(record: dict[str, Any]) -> str:
    if isinstance(record.get("samples"), list):
        return "eval"
    if {"input", "output", "gts"} <= set(record.keys()):
        return "rollout"
    raise ValueError(
        "Could not detect artifact type. Expected eval rows with 'samples' or rollout rows with 'input'/'output'/'gts'."
    )


def _resolve_input_paths(input_path: str, input_type: str, step: int | None) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL files found under {path}")
        if step is not None:
            step_file = path / f"{step}.jsonl"
            if input_type == "rollout" or step_file.exists():
                if not step_file.exists():
                    raise FileNotFoundError(f"Could not find rollout dump for step {step}: {step_file}")
                return [step_file]
        return candidates

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if input_type == "rollout" and step is not None and path.stem.isdigit() and int(path.stem) != step:
        raise ValueError(f"Input file {path} is for step {path.stem}, not requested step {step}")
    return [path]


def _normalize_logged_value(field: str, value: Any) -> float | int | None:
    if value is None:
        return None
    if field in {"response_length_tokens", "length_limit_tokens"}:
        return int(value)
    return float(value)


def _build_eval_sample(
    row: dict[str, Any],
    sample: dict[str, Any],
    source_path: Path,
    row_index: int,
    sample_index: int,
) -> dict[str, Any]:
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        extra_info = {}

    return {
        "artifact_type": "eval",
        "source_path": str(source_path),
        "row_index": row_index,
        "sample_index": sample_index,
        "step": row.get("step"),
        "prompt": row.get("prompt"),
        "output": sample.get("text"),
        "ground_truth": row.get("ground_truth"),
        "data_source": row.get("data_source", "deepscaler_math_length"),
        "logged": {
            "score": sample.get("score"),
            "correct_reward": sample.get("correct_reward"),
            "length_reward": sample.get("length_reward"),
            "answer_parse_ok": sample.get("answer_parse_ok"),
            "response_length_tokens": sample.get("response_length_tokens"),
            "length_limit_tokens": sample.get("length_limit_tokens", extra_info.get("length_limit_tokens")),
        },
    }


def _build_rollout_sample(row: dict[str, Any], source_path: Path, row_index: int) -> dict[str, Any]:
    return {
        "artifact_type": "rollout",
        "source_path": str(source_path),
        "row_index": row_index,
        "sample_index": row_index,
        "step": row.get("step"),
        "prompt": row.get("input"),
        "output": row.get("output"),
        "ground_truth": row.get("gts"),
        "data_source": row.get("data_source", "deepscaler_math_length"),
        "logged": {
            "score": row.get("score"),
            "correct_reward": row.get("correct_reward"),
            "length_reward": row.get("length_reward"),
            "answer_parse_ok": row.get("answer_parse_ok"),
            "response_length_tokens": row.get("response_length_tokens"),
            "length_limit_tokens": row.get("length_limit_tokens"),
        },
    }


def load_canonical_samples(input_path: str, input_type: str = "auto", step: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []

    for source_path in _resolve_input_paths(input_path, input_type, step):
        rows = _read_jsonl(source_path)
        if not rows:
            continue

        artifact_type = input_type
        if artifact_type == "auto":
            artifact_type = detect_input_type(rows[0])

        for row_index, row in enumerate(rows):
            if artifact_type == "eval":
                if step is not None and row.get("step") not in (None, step):
                    continue
                row_samples = row.get("samples")
                if not isinstance(row_samples, list):
                    raise ValueError(f"Expected eval row to contain a 'samples' list in {source_path}")
                for sample_index, sample in enumerate(row_samples):
                    samples.append(_build_eval_sample(row, sample, source_path, row_index, sample_index))
            elif artifact_type == "rollout":
                if step is not None and row.get("step") != step:
                    continue
                samples.append(_build_rollout_sample(row, source_path, row_index))
            else:
                raise ValueError(f"Unsupported input type: {artifact_type}")

    if not samples:
        raise ValueError("No samples found for the requested input path / filters.")
    return samples


def recompute_sample(sample: dict[str, Any]) -> dict[str, Any]:
    logged = sample["logged"]
    length_limit_tokens = logged.get("length_limit_tokens")
    response_length_tokens = logged.get("response_length_tokens")
    output_text = sample.get("output")
    ground_truth = sample.get("ground_truth")
    extracted_boxed_answer = extract_last_boxed_answer("" if output_text is None else str(output_text))

    recomputed: dict[str, Any] = {
        "score": None,
        "correct_reward": None,
        "length_reward": None,
        "answer_parse_ok": float(extracted_boxed_answer is not None),
        "response_length_tokens": int(response_length_tokens) if response_length_tokens is not None else None,
        "length_limit_tokens": int(length_limit_tokens) if length_limit_tokens is not None else None,
    }
    recompute_error = None

    if output_text is None or ground_truth is None:
        recompute_error = "missing output or ground_truth"
    elif response_length_tokens is None or length_limit_tokens is None:
        recompute_error = "missing response_length_tokens or length_limit_tokens"
    else:
        reward_info = compute_score(
            data_source=str(sample.get("data_source", "deepscaler_math_length")),
            solution_str=str(output_text),
            ground_truth=str(ground_truth),
            extra_info={
                "response_length_tokens": int(response_length_tokens),
                "length_limit_tokens": int(length_limit_tokens),
            },
        )
        recomputed.update(
            {
                "score": float(reward_info["score"]),
                "correct_reward": float(reward_info["correct_reward"]),
                "length_reward": float(reward_info["length_reward"]),
                "answer_parse_ok": float(reward_info["answer_parse_ok"]),
                "response_length_tokens": int(reward_info["response_length_tokens"]),
                "length_limit_tokens": int(reward_info["length_limit_tokens"]),
            }
        )

    missing_logged_fields: list[str] = []
    mismatch_fields: list[str] = []
    field_matches: dict[str, bool | None] = {}
    for field in COMPARE_FIELDS:
        logged_value = _normalize_logged_value(field, logged.get(field))
        recomputed_value = _normalize_logged_value(field, recomputed.get(field))
        if logged_value is None:
            missing_logged_fields.append(field)
            field_matches[field] = None
            continue
        if recomputed_value is None:
            mismatch_fields.append(field)
            field_matches[field] = False
            continue
        is_match = logged_value == recomputed_value
        field_matches[field] = is_match
        if not is_match:
            mismatch_fields.append(field)

    enriched = dict(sample)
    enriched["extracted_boxed_answer"] = extracted_boxed_answer
    enriched["recomputed"] = recomputed
    enriched["missing_logged_fields"] = missing_logged_fields
    enriched["mismatch_fields"] = mismatch_fields
    enriched["field_matches"] = field_matches
    enriched["recompute_error"] = recompute_error
    return enriched


def summarize_recomputed_samples(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary: dict[str, Any] = {
        "total_samples": len(samples),
        "samples_with_any_mismatch": 0,
        "samples_with_missing_logged_fields": 0,
        "samples_with_recompute_error": 0,
        "field_mismatch_counts": {field: 0 for field in COMPARE_FIELDS},
        "missing_logged_field_counts": {field: 0 for field in COMPARE_FIELDS},
        "by_step": {},
    }
    mismatches: list[dict[str, Any]] = []
    per_step: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_samples": 0,
            "samples_with_any_mismatch": 0,
            "samples_with_missing_logged_fields": 0,
            "samples_with_recompute_error": 0,
            "field_mismatch_counts": {field: 0 for field in COMPARE_FIELDS},
            "missing_logged_field_counts": {field: 0 for field in COMPARE_FIELDS},
        }
    )

    for sample in samples:
        step_key = str(sample.get("step", "unknown"))
        step_summary = per_step[step_key]
        summary["total_samples"] += 0  # explicit for symmetry/readability
        step_summary["total_samples"] += 1

        if sample["missing_logged_fields"]:
            summary["samples_with_missing_logged_fields"] += 1
            step_summary["samples_with_missing_logged_fields"] += 1
        for field in sample["missing_logged_fields"]:
            summary["missing_logged_field_counts"][field] += 1
            step_summary["missing_logged_field_counts"][field] += 1

        if sample["recompute_error"] is not None:
            summary["samples_with_recompute_error"] += 1
            step_summary["samples_with_recompute_error"] += 1

        if sample["mismatch_fields"]:
            summary["samples_with_any_mismatch"] += 1
            step_summary["samples_with_any_mismatch"] += 1
            mismatches.append(sample)
        for field in sample["mismatch_fields"]:
            summary["field_mismatch_counts"][field] += 1
            step_summary["field_mismatch_counts"][field] += 1

    summary["by_step"] = dict(sorted(per_step.items(), key=lambda item: item[0]))
    return summary, mismatches


def _output_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _preview(text: Any, width: int = 100) -> str:
    if text is None:
        return "<missing>"
    normalized = " ".join(str(text).split())
    if len(normalized) <= width:
        return normalized
    return normalized[: width - 3] + "..."


def print_sample_audit(samples: list[dict[str, Any]]) -> None:
    for index, sample in enumerate(samples, start=1):
        print(
            f"[{index}] type={sample['artifact_type']} step={sample.get('step', 'n/a')} "
            f"source={Path(sample['source_path']).name} sample_index={sample['sample_index']}"
        )
        print(f"  boxed_answer={sample['extracted_boxed_answer']!r}")
        if sample["recompute_error"] is not None:
            print(f"  recompute_error={sample['recompute_error']}")
        print(
            "  fields="
            + ", ".join(
                f"{field}:{sample['logged'].get(field)!r}->{sample['recomputed'].get(field)!r}"
                f" ({'match' if sample['field_matches'][field] else 'mismatch' if sample['field_matches'][field] is False else 'missing'})"
                for field in COMPARE_FIELDS
            )
        )
        if sample["mismatch_fields"]:
            print(f"  mismatch_fields={sample['mismatch_fields']}")
        if sample["missing_logged_fields"]:
            print(f"  missing_logged_fields={sample['missing_logged_fields']}")
        print(f"  prompt={_preview(sample.get('prompt'))}")
        print(f"  ground_truth={sample.get('ground_truth')!r}")
        print(f"  output={_preview(sample.get('output'), width=140)}")
        print()


def _print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def _select_samples(
    samples: list[dict[str, Any]],
    *,
    sample_count: int,
    seed: int,
    only_mismatches: bool,
) -> list[dict[str, Any]]:
    filtered = [sample for sample in samples if sample["mismatch_fields"]] if only_mismatches else list(samples)
    if not filtered:
        return []
    rng = random.Random(seed)
    if len(filtered) <= sample_count:
        return filtered
    return rng.sample(filtered, sample_count)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit DeepScaleR-style math reward artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-path", required=True, help="JSONL file or rollout dump directory to audit.")
    common.add_argument("--input-type", choices=("auto", "eval", "rollout"), default="auto")
    common.add_argument("--step", type=int, default=None, help="Restrict audit to a single rollout step.")

    sample_parser = subparsers.add_parser("sample-audit", parents=[common], help="Inspect sampled records by hand.")
    sample_parser.add_argument("--sample-count", type=int, default=10)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--only-mismatches", action="store_true")
    sample_parser.add_argument("--output-jsonl", default=None)

    summary_parser = subparsers.add_parser(
        "recompute-summary",
        parents=[common],
        help="Recompute rewards for all rows and summarize mismatches.",
    )
    summary_parser.add_argument("--summary-output", default=None)
    summary_parser.add_argument("--mismatch-output", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    canonical_samples = load_canonical_samples(args.input_path, input_type=args.input_type, step=args.step)
    recomputed_samples = [recompute_sample(sample) for sample in canonical_samples]

    if args.command == "sample-audit":
        selected = _select_samples(
            recomputed_samples,
            sample_count=args.sample_count,
            seed=args.seed,
            only_mismatches=args.only_mismatches,
        )
        if not selected:
            print("No samples matched the requested filters.")
            return 0
        print_sample_audit(selected)
        if args.output_jsonl is not None:
            _output_jsonl(args.output_jsonl, selected)
        return 0

    summary, mismatches = summarize_recomputed_samples(recomputed_samples)
    _print_summary(summary)
    if args.summary_output is not None:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mismatch_output is not None:
        _output_jsonl(args.mismatch_output, mismatches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
