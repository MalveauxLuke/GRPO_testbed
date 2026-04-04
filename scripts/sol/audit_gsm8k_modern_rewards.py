#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REWARD_MODULE_PATH = PROJECT_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score" / "gsm8k_modern_two_reward.py"
NUMERIC_COMPARE_FIELDS = (
    "score",
    "correct_reward",
    "format_reward",
    "strict_format_reward",
    "approx_format_reward",
    "answer_parse_ok",
)
TEXT_COMPARE_FIELDS = ("parsed_answer", "expected_answer")
ALL_COMPARE_FIELDS = NUMERIC_COMPARE_FIELDS + TEXT_COMPARE_FIELDS


def load_module_from_path(module_name: str, module_path: str | Path):
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _normalize_logged_value(field: str, value: Any) -> float | str | None:
    if value is None:
        return None
    if field in NUMERIC_COMPARE_FIELDS:
        return float(value)
    return str(value)


def _build_eval_sample(
    row: dict[str, Any],
    sample: dict[str, Any],
    source_path: Path,
    row_index: int,
    sample_index: int,
) -> dict[str, Any]:
    return {
        "artifact_type": "eval",
        "source_path": str(source_path),
        "row_index": row_index,
        "sample_index": sample_index,
        "step": row.get("step"),
        "prompt": row.get("prompt"),
        "output": sample.get("text"),
        "ground_truth": row.get("ground_truth"),
        "data_source": row.get("data_source", "openai/gsm8k_modern_two_reward"),
        "logged": {
            "score": sample.get("score"),
            "correct_reward": sample.get("correct_reward"),
            "format_reward": sample.get("format_reward"),
            "strict_format_reward": sample.get("strict_format_reward"),
            "approx_format_reward": sample.get("approx_format_reward"),
            "answer_parse_ok": sample.get("answer_parse_ok"),
            "parsed_answer": sample.get("parsed_answer"),
            "expected_answer": sample.get("expected_answer", row.get("ground_truth")),
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
        "data_source": row.get("data_source", "openai/gsm8k_modern_two_reward"),
        "logged": {
            "score": row.get("score"),
            "correct_reward": row.get("correct_reward"),
            "format_reward": row.get("format_reward"),
            "strict_format_reward": row.get("strict_format_reward"),
            "approx_format_reward": row.get("approx_format_reward"),
            "answer_parse_ok": row.get("answer_parse_ok"),
            "parsed_answer": row.get("parsed_answer"),
            "expected_answer": row.get("expected_answer", row.get("gts")),
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


def _behavior_bucket(output_text: str, recomputed: dict[str, Any]) -> str:
    has_reasoning_tag = "<reasoning>" in output_text and "</reasoning>" in output_text
    has_answer_tag = "<answer>" in output_text and "</answer>" in output_text
    strict_format = float(recomputed.get("strict_format_reward") or 0.0)
    format_reward = float(recomputed.get("format_reward") or 0.0)
    correct_reward = float(recomputed.get("correct_reward") or 0.0)
    answer_parse_ok = float(recomputed.get("answer_parse_ok") or 0.0)

    if strict_format == 1.0 and correct_reward == 1.0:
        return "strict_correct"
    if strict_format == 1.0 and correct_reward == 0.0:
        return "strict_wrong_answer"
    if strict_format == 0.0 and answer_parse_ok == 1.0 and correct_reward == 1.0:
        return "answer_tag_correct_without_strict_format"
    if strict_format == 0.0 and answer_parse_ok == 1.0 and correct_reward == 0.0:
        return "answer_tag_wrong_without_strict_format"
    if has_reasoning_tag and not has_answer_tag:
        return "reasoning_only"
    if format_reward > 0.0 and answer_parse_ok == 0.0:
        return "partial_format_answer_parse_failed"
    if has_answer_tag and not has_reasoning_tag:
        return "answer_only_parse_failed"
    if not has_reasoning_tag and not has_answer_tag:
        return "no_tags"
    return "other"


def recompute_sample(sample: dict[str, Any], reward_module) -> dict[str, Any]:
    logged = sample["logged"]
    output_text = sample.get("output")
    ground_truth = sample.get("ground_truth")

    recomputed: dict[str, Any] = {
        "score": None,
        "correct_reward": None,
        "format_reward": None,
        "strict_format_reward": None,
        "approx_format_reward": None,
        "answer_parse_ok": None,
        "parsed_answer": None,
        "expected_answer": None,
    }
    recompute_error = None

    if output_text is None or ground_truth is None:
        recompute_error = "missing output or ground_truth"
    else:
        reward_info = reward_module.compute_score(
            data_source=str(sample.get("data_source", reward_module.DATA_SOURCE)),
            solution_str=str(output_text),
            ground_truth=str(ground_truth),
            extra_info={},
        )
        recomputed.update(
            {
                "score": float(reward_info["score"]),
                "correct_reward": float(reward_info["correct_reward"]),
                "format_reward": float(reward_info["format_reward"]),
                "strict_format_reward": float(reward_info["strict_format_reward"]),
                "approx_format_reward": float(reward_info["approx_format_reward"]),
                "answer_parse_ok": float(reward_info["answer_parse_ok"]),
                "parsed_answer": str(reward_info["parsed_answer"]),
                "expected_answer": str(reward_info["expected_answer"]),
            }
        )

    missing_logged_fields: list[str] = []
    mismatch_fields: list[str] = []
    field_matches: dict[str, bool | None] = {}
    for field in ALL_COMPARE_FIELDS:
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

    output_text_str = "" if output_text is None else str(output_text)
    tag_presence = {
        "has_reasoning_open": "<reasoning>" in output_text_str,
        "has_reasoning_close": "</reasoning>" in output_text_str,
        "has_answer_open": "<answer>" in output_text_str,
        "has_answer_close": "</answer>" in output_text_str,
    }

    enriched = dict(sample)
    enriched["recomputed"] = recomputed
    enriched["missing_logged_fields"] = missing_logged_fields
    enriched["mismatch_fields"] = mismatch_fields
    enriched["field_matches"] = field_matches
    enriched["recompute_error"] = recompute_error
    enriched["tag_presence"] = tag_presence
    enriched["behavior_bucket"] = _behavior_bucket(output_text_str, recomputed)
    return enriched


def _init_summary() -> dict[str, Any]:
    return {
        "total_samples": 0,
        "samples_with_any_mismatch": 0,
        "samples_with_missing_logged_fields": 0,
        "samples_with_recompute_error": 0,
        "field_mismatch_counts": {field: 0 for field in ALL_COMPARE_FIELDS},
        "missing_logged_field_counts": {field: 0 for field in ALL_COMPARE_FIELDS},
        "field_match_counts": {field: 0 for field in ALL_COMPARE_FIELDS},
        "field_comparable_counts": {field: 0 for field in ALL_COMPARE_FIELDS},
        "logged_metric_sums": {field: 0.0 for field in NUMERIC_COMPARE_FIELDS},
        "logged_metric_counts": {field: 0 for field in NUMERIC_COMPARE_FIELDS},
        "recomputed_metric_sums": {field: 0.0 for field in NUMERIC_COMPARE_FIELDS},
        "recomputed_metric_counts": {field: 0 for field in NUMERIC_COMPARE_FIELDS},
        "behavior_bucket_counts": {},
        "tag_presence_counts": {
            "has_reasoning_open": 0,
            "has_reasoning_close": 0,
            "has_answer_open": 0,
            "has_answer_close": 0,
        },
    }


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    field_match_rates: dict[str, float | None] = {}
    for field in ALL_COMPARE_FIELDS:
        comparable = summary["field_comparable_counts"][field]
        if comparable == 0:
            field_match_rates[field] = None
        else:
            field_match_rates[field] = summary["field_match_counts"][field] / comparable

    logged_metric_means: dict[str, float | None] = {}
    recomputed_metric_means: dict[str, float | None] = {}
    for field in NUMERIC_COMPARE_FIELDS:
        logged_count = summary["logged_metric_counts"][field]
        recomputed_count = summary["recomputed_metric_counts"][field]
        logged_metric_means[field] = None if logged_count == 0 else summary["logged_metric_sums"][field] / logged_count
        recomputed_metric_means[field] = (
            None if recomputed_count == 0 else summary["recomputed_metric_sums"][field] / recomputed_count
        )

    summary["field_match_rates"] = field_match_rates
    summary["logged_metric_means"] = logged_metric_means
    summary["recomputed_metric_means"] = recomputed_metric_means
    summary["behavior_bucket_counts"] = dict(sorted(summary["behavior_bucket_counts"].items()))
    return summary


def summarize_recomputed_samples(samples: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _init_summary()
    mismatches: list[dict[str, Any]] = []
    per_step: dict[str, dict[str, Any]] = defaultdict(_init_summary)

    for sample in samples:
        step_key = str(sample.get("step", "unknown"))
        step_summary = per_step[step_key]

        for target in (summary, step_summary):
            target["total_samples"] += 1
            if sample["missing_logged_fields"]:
                target["samples_with_missing_logged_fields"] += 1
            if sample["recompute_error"] is not None:
                target["samples_with_recompute_error"] += 1
            if sample["mismatch_fields"]:
                target["samples_with_any_mismatch"] += 1

            for field in sample["missing_logged_fields"]:
                target["missing_logged_field_counts"][field] += 1
            for field in sample["mismatch_fields"]:
                target["field_mismatch_counts"][field] += 1
            for field, match_state in sample["field_matches"].items():
                if match_state is not None:
                    target["field_comparable_counts"][field] += 1
                    if match_state:
                        target["field_match_counts"][field] += 1

            for field in NUMERIC_COMPARE_FIELDS:
                logged_value = sample["logged"].get(field)
                recomputed_value = sample["recomputed"].get(field)
                if logged_value is not None:
                    target["logged_metric_sums"][field] += float(logged_value)
                    target["logged_metric_counts"][field] += 1
                if recomputed_value is not None:
                    target["recomputed_metric_sums"][field] += float(recomputed_value)
                    target["recomputed_metric_counts"][field] += 1

            target["behavior_bucket_counts"][sample["behavior_bucket"]] = (
                target["behavior_bucket_counts"].get(sample["behavior_bucket"], 0) + 1
            )
            for field, present in sample["tag_presence"].items():
                target["tag_presence_counts"][field] += int(bool(present))

        if sample["mismatch_fields"] or sample["missing_logged_fields"] or sample["recompute_error"] is not None:
            mismatches.append(sample)

    summary = _finalize_summary(summary)
    summary["by_step"] = {step: _finalize_summary(step_summary) for step, step_summary in sorted(per_step.items())}
    return summary, mismatches


def _output_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _preview(text: Any, width: int = 120) -> str:
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
            f"bucket={sample['behavior_bucket']} source={Path(sample['source_path']).name} sample_index={sample['sample_index']}"
        )
        if sample["recompute_error"] is not None:
            print(f"  recompute_error={sample['recompute_error']}")
        print(
            "  fields="
            + ", ".join(
                f"{field}:{sample['logged'].get(field)!r}->{sample['recomputed'].get(field)!r}"
                f" ({'match' if sample['field_matches'][field] else 'mismatch' if sample['field_matches'][field] is False else 'missing'})"
                for field in ALL_COMPARE_FIELDS
            )
        )
        if sample["mismatch_fields"]:
            print(f"  mismatch_fields={sample['mismatch_fields']}")
        if sample["missing_logged_fields"]:
            print(f"  missing_logged_fields={sample['missing_logged_fields']}")
        print(f"  ground_truth={sample.get('ground_truth')!r}")
        print(f"  parsed_answer={sample['recomputed'].get('parsed_answer')!r}")
        print(f"  prompt={_preview(sample.get('prompt'))}")
        print(f"  output={_preview(sample.get('output'), width=180)}")
        print()


def _select_samples(
    samples: list[dict[str, Any]],
    *,
    sample_count: int,
    seed: int,
    only_mismatches: bool,
    behavior_bucket: str | None,
) -> list[dict[str, Any]]:
    filtered = list(samples)
    if only_mismatches:
        filtered = [
            sample
            for sample in filtered
            if sample["mismatch_fields"] or sample["missing_logged_fields"] or sample["recompute_error"] is not None
        ]
    if behavior_bucket is not None:
        filtered = [sample for sample in filtered if sample["behavior_bucket"] == behavior_bucket]
    if not filtered:
        return []
    rng = random.Random(seed)
    if len(filtered) <= sample_count:
        return filtered
    return rng.sample(filtered, sample_count)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit GSM8K modern rollout and eval rewards against ground truth.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-path", required=True, help="JSONL file or rollout dump directory to audit.")
    common.add_argument("--input-type", choices=("auto", "eval", "rollout"), default="auto")
    common.add_argument("--step", type=int, default=None, help="Restrict audit to a single rollout step.")
    common.add_argument("--reward-module-path", default=str(DEFAULT_REWARD_MODULE_PATH))

    sample_parser = subparsers.add_parser("sample-audit", parents=[common], help="Inspect sampled records by hand.")
    sample_parser.add_argument("--sample-count", type=int, default=10)
    sample_parser.add_argument("--seed", type=int, default=0)
    sample_parser.add_argument("--only-mismatches", action="store_true")
    sample_parser.add_argument("--behavior-bucket", default=None)
    sample_parser.add_argument("--output-jsonl", default=None)

    summary_parser = subparsers.add_parser(
        "recompute-summary",
        parents=[common],
        help="Recompute rewards for all rows and summarize mismatches and model behavior.",
    )
    summary_parser.add_argument("--summary-output", default=None)
    summary_parser.add_argument("--mismatch-output", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    reward_module = load_module_from_path("gsm8k_modern_two_reward_audit", args.reward_module_path)

    canonical_samples = load_canonical_samples(args.input_path, input_type=args.input_type, step=args.step)
    recomputed_samples = [recompute_sample(sample, reward_module) for sample in canonical_samples]

    if args.command == "sample-audit":
        selected = _select_samples(
            recomputed_samples,
            sample_count=args.sample_count,
            seed=args.seed,
            only_mismatches=args.only_mismatches,
            behavior_bucket=args.behavior_bucket,
        )
        if not selected:
            print("No samples matched the requested filters.")
            return 0
        print_sample_audit(selected)
        if args.output_jsonl is not None:
            _output_jsonl(args.output_jsonl, selected)
        return 0

    summary, mismatches = summarize_recomputed_samples(recomputed_samples)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.summary_output is not None:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mismatch_output is not None:
        _output_jsonl(args.mismatch_output, mismatches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
