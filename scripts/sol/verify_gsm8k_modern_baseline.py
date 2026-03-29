#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REWARD_MODULE_PATH = PROJECT_ROOT / "external" / "verl" / "verl" / "utils" / "reward_score" / "gsm8k_modern_two_reward.py"
DEFAULT_DOCS_NOTE_PATH = PROJECT_ROOT / "docs" / "gsm8k_modern_two_reward_baseline.md"
ARTIFACT_COMPARE_FIELDS = ("score", "correct_reward", "format_reward")


def maybe_json(value: Any):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
    return value


def load_module_from_path(module_name: str, module_path: str | Path):
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def output_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def normalize_logged_value(field: str, value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def load_split_rows_from_parquet_dir(dataset_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    from datasets import load_dataset

    dataset_dir = Path(dataset_dir)
    data_files = {}
    for split in ("train", "test"):
        split_path = dataset_dir / f"{split}.parquet"
        if split_path.exists():
            data_files[split] = str(split_path)
    if not data_files:
        raise FileNotFoundError(f"No train/test parquet files found under {dataset_dir}")

    dataset = load_dataset("parquet", data_files=data_files)
    return {split: [dict(row) for row in dataset[split]] for split in data_files}


def load_source_splits(source_mode: str, base_dir: str | None) -> dict[str, list[dict[str, Any]]]:
    if source_mode == "preprocessed":
        if base_dir is None:
            raise ValueError("--base-dir is required when --source-mode=preprocessed")
        return load_split_rows_from_parquet_dir(base_dir)

    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main")
    return {
        "train": [dict(row) for row in dataset["train"]],
        "test": [dict(row) for row in dataset["test"]],
    }


def extract_source_question_answer(row: dict[str, Any]) -> tuple[str, str]:
    if "question" in row and "answer" in row:
        return str(row["question"]), str(row["answer"])

    extra_info = maybe_json(row.get("extra_info", {}))
    if isinstance(extra_info, dict) and extra_info.get("question") is not None and extra_info.get("answer") is not None:
        return str(extra_info["question"]), str(extra_info["answer"])

    raise ValueError("Could not recover source question/answer from row.")


def extract_processed_question_system(row: dict[str, Any]) -> tuple[str, str]:
    prompt = maybe_json(row.get("prompt", []))
    if not isinstance(prompt, list) or len(prompt) < 2:
        raise ValueError("Processed row is missing the expected system+user prompt format.")

    system_message = maybe_json(prompt[0])
    user_message = maybe_json(prompt[1])
    if not isinstance(system_message, dict) or not isinstance(user_message, dict):
        raise ValueError("Processed prompt messages are malformed.")
    return str(user_message["content"]), str(system_message["content"])


def audit_dataset_rows(
    *,
    source_splits: dict[str, list[dict[str, Any]]],
    processed_splits: dict[str, list[dict[str, Any]]],
    reward_module,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    summary = {
        "row_count_mismatches": {},
        "split_counts": {},
        "samples_checked": 0,
        "mismatch_count": 0,
    }

    for split in ("train", "test"):
        source_rows = source_splits.get(split, [])
        processed_rows = processed_splits.get(split, [])
        summary["split_counts"][split] = {"source": len(source_rows), "processed": len(processed_rows)}
        if len(source_rows) != len(processed_rows):
            summary["row_count_mismatches"][split] = {"source": len(source_rows), "processed": len(processed_rows)}

        for index, (source_row, processed_row) in enumerate(zip(source_rows, processed_rows, strict=False)):
            summary["samples_checked"] += 1
            mismatch_fields: list[str] = []

            source_question, source_answer = extract_source_question_answer(source_row)
            expected_ground_truth = reward_module.extract_hash_answer(source_answer)
            processed_question, processed_system_prompt = extract_processed_question_system(processed_row)
            processed_reward_model = maybe_json(processed_row.get("reward_model", {}))
            processed_extra_info = maybe_json(processed_row.get("extra_info", {}))

            if processed_row.get("data_source") != reward_module.DATA_SOURCE:
                mismatch_fields.append("data_source")
            if processed_system_prompt != reward_module.SYSTEM_PROMPT:
                mismatch_fields.append("system_prompt")
            if processed_question != source_question:
                mismatch_fields.append("question")
            if not isinstance(processed_reward_model, dict) or str(processed_reward_model.get("ground_truth")) != expected_ground_truth:
                mismatch_fields.append("ground_truth")
            if not isinstance(processed_extra_info, dict) or str(processed_extra_info.get("question")) != source_question:
                mismatch_fields.append("extra_info.question")
            if not isinstance(processed_extra_info, dict) or str(processed_extra_info.get("answer")) != source_answer:
                mismatch_fields.append("extra_info.answer")
            if not isinstance(processed_extra_info, dict) or str(processed_extra_info.get("split")) != split:
                mismatch_fields.append("extra_info.split")
            if not isinstance(processed_extra_info, dict) or int(processed_extra_info.get("index", -1)) != index:
                mismatch_fields.append("extra_info.index")
            if not isinstance(processed_extra_info, dict) or str(processed_extra_info.get("source_dataset")) != "openai/gsm8k":
                mismatch_fields.append("extra_info.source_dataset")
            if not isinstance(processed_extra_info, dict) or str(processed_extra_info.get("source_subset")) != "main":
                mismatch_fields.append("extra_info.source_subset")
            if (
                not isinstance(processed_extra_info, dict)
                or str(processed_extra_info.get("baseline_name")) != str(reward_module.ALIGNMENT_SPEC.get("baseline_name"))
            ):
                mismatch_fields.append("extra_info.baseline_name")
            if not isinstance(processed_extra_info, dict) or str(processed_extra_info.get("alignment_spec_version")) != reward_module.ALIGNMENT_SPEC["version"]:
                mismatch_fields.append("extra_info.alignment_spec_version")

            if mismatch_fields:
                mismatches.append(
                    {
                        "audit": "dataset-audit",
                        "split": split,
                        "index": index,
                        "mismatch_fields": mismatch_fields,
                        "expected_ground_truth": expected_ground_truth,
                        "processed_ground_truth": (
                            None if not isinstance(processed_reward_model, dict) else processed_reward_model.get("ground_truth")
                        ),
                    }
                )

    summary["mismatch_count"] = len(mismatches)
    return summary, mismatches


def build_wrong_answer(gold_answer: str) -> str:
    try:
        numeric = float(gold_answer)
        if numeric == 0:
            return "1"
        if gold_answer.isdigit() or (gold_answer.startswith("-") and gold_answer[1:].isdigit()):
            return str(int(numeric) + 1)
        return str(numeric + 1.0)
    except Exception:
        return "999999"


def build_reward_test_cases(gold_answer: str) -> list[dict[str, Any]]:
    cases = [
        {
            "name": "valid_correct",
            "solution": f"<reasoning>We solve the problem carefully.</reasoning><answer>{gold_answer}</answer>",
            "expected": {"format_reward": 1.0, "correct_reward": 1.0, "score": 2.0},
        },
        {
            "name": "valid_wrong",
            "solution": f"<reasoning>We solve the problem carefully.</reasoning><answer>{build_wrong_answer(gold_answer)}</answer>",
            "expected": {"format_reward": 1.0, "correct_reward": 0.0, "score": 1.0},
        },
        {
            "name": "missing_tags",
            "solution": f"The answer is {gold_answer}.",
            "expected": {"format_reward": 0.0, "correct_reward": 0.0, "score": 0.0},
        },
        {
            "name": "malformed_order",
            "solution": f"<answer>{gold_answer}</answer><reasoning>Reasoning first is missing.</reasoning>",
            "expected": {"format_reward": 0.0, "correct_reward": 0.0, "score": 0.0},
        },
        {
            "name": "duplicated_tags",
            "solution": f"<reasoning>One</reasoning><reasoning>Two</reasoning><answer>{gold_answer}</answer>",
            "expected": {"format_reward": 0.0, "correct_reward": 0.0, "score": 0.0},
        },
        {
            "name": "answer_outside_tags",
            "solution": f"<reasoning>We solve it.</reasoning>{gold_answer}",
            "expected": {"format_reward": 0.0, "correct_reward": 0.0, "score": 0.0},
        },
        {
            "name": "trailing_junk",
            "solution": f"<reasoning>We solve it.</reasoning><answer>{gold_answer}</answer> trailing",
            "expected": {"format_reward": 0.0, "correct_reward": 0.0, "score": 0.0},
        },
        {
            "name": "whitespace_normalization",
            "solution": f"<reasoning>We solve it.</reasoning><answer>  {gold_answer}  </answer>",
            "expected": {"format_reward": 1.0, "correct_reward": 1.0, "score": 2.0},
        },
        {
            "name": "dollar_normalization",
            "solution": f"<reasoning>We solve it.</reasoning><answer>${gold_answer}</answer>",
            "expected": {"format_reward": 1.0, "correct_reward": 1.0, "score": 2.0},
        },
    ]

    if gold_answer.lstrip("-").isdigit() and abs(int(gold_answer)) >= 1000:
        cases.append(
            {
                "name": "comma_normalization",
                "solution": f"<reasoning>We solve it.</reasoning><answer>{int(gold_answer):,}</answer>",
                "expected": {"format_reward": 1.0, "correct_reward": 1.0, "score": 2.0},
            }
        )

    return cases


def audit_reward_rows(processed_splits: dict[str, list[dict[str, Any]]], reward_module) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    case_counts = Counter()

    for split, rows in processed_splits.items():
        for index, row in enumerate(rows):
            reward_model = maybe_json(row.get("reward_model", {}))
            if not isinstance(reward_model, dict) or reward_model.get("ground_truth") is None:
                mismatches.append(
                    {
                        "audit": "reward-audit",
                        "split": split,
                        "index": index,
                        "case": "missing_ground_truth",
                        "mismatch_fields": ["ground_truth"],
                    }
                )
                continue

            gold_answer = str(reward_model["ground_truth"])
            for case in build_reward_test_cases(gold_answer):
                case_counts[case["name"]] += 1
                result = reward_module.compute_score(
                    data_source=reward_module.DATA_SOURCE,
                    solution_str=case["solution"],
                    ground_truth=gold_answer,
                    extra_info=row.get("extra_info", {}),
                )
                mismatch_fields = [
                    field
                    for field in ARTIFACT_COMPARE_FIELDS
                    if float(result[field]) != float(case["expected"][field])
                ]
                if mismatch_fields:
                    mismatches.append(
                        {
                            "audit": "reward-audit",
                            "split": split,
                            "index": index,
                            "case": case["name"],
                            "mismatch_fields": mismatch_fields,
                            "expected": case["expected"],
                            "observed": {field: float(result[field]) for field in ARTIFACT_COMPARE_FIELDS},
                        }
                    )

    summary = {
        "samples_checked": sum(len(rows) for rows in processed_splits.values()),
        "case_counts": dict(sorted(case_counts.items())),
        "mismatch_count": len(mismatches),
    }
    return summary, mismatches


def detect_artifact_type(record: dict[str, Any]) -> str:
    if isinstance(record.get("samples"), list):
        return "eval"
    if {"input", "output", "gts"} <= set(record.keys()):
        return "rollout"
    raise ValueError("Unsupported artifact row shape.")


def resolve_artifact_paths(input_path: str, artifact_type: str, step: int | None) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL files found under {path}")
        if step is not None:
            step_file = path / f"{step}.jsonl"
            if artifact_type == "rollout" or step_file.exists():
                if not step_file.exists():
                    raise FileNotFoundError(f"Could not find rollout dump for step {step}: {step_file}")
                return [step_file]
        return candidates
    if not path.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {path}")
    return [path]


def load_canonical_artifact_samples(input_path: str, artifact_type: str = "auto", step: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in resolve_artifact_paths(input_path, artifact_type, step):
        rows = load_jsonl(path)
        if not rows:
            continue
        current_type = artifact_type if artifact_type != "auto" else detect_artifact_type(rows[0])
        for row_index, row in enumerate(rows):
            if current_type == "eval":
                row_samples = row.get("samples")
                if not isinstance(row_samples, list):
                    raise ValueError(f"Expected eval row with samples list in {path}")
                for sample_index, sample in enumerate(row_samples):
                    samples.append(
                        {
                            "artifact_type": "eval",
                            "source_path": str(path),
                            "row_index": row_index,
                            "sample_index": sample_index,
                            "step": row.get("step"),
                            "output": sample.get("text"),
                            "ground_truth": row.get("ground_truth"),
                            "logged": {
                                "score": sample.get("score"),
                                "correct_reward": sample.get("correct_reward"),
                                "format_reward": sample.get("format_reward"),
                            },
                        }
                    )
            elif current_type == "rollout":
                if step is not None and row.get("step") != step:
                    continue
                samples.append(
                    {
                        "artifact_type": "rollout",
                        "source_path": str(path),
                        "row_index": row_index,
                        "sample_index": row_index,
                        "step": row.get("step"),
                        "output": row.get("output"),
                        "ground_truth": row.get("gts"),
                        "logged": {
                            "score": row.get("score"),
                            "correct_reward": row.get("correct_reward"),
                            "format_reward": row.get("format_reward"),
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported artifact type: {current_type}")
    if not samples:
        raise ValueError("No artifact samples found.")
    return samples


def audit_artifact_samples(artifact_samples: list[dict[str, Any]], reward_module) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = {
        "total_samples": len(artifact_samples),
        "samples_with_any_mismatch": 0,
        "field_mismatch_counts": {field: 0 for field in ARTIFACT_COMPARE_FIELDS},
        "by_step": {},
    }
    mismatches: list[dict[str, Any]] = []
    by_step: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total_samples": 0, "samples_with_any_mismatch": 0, "field_mismatch_counts": {field: 0 for field in ARTIFACT_COMPARE_FIELDS}}
    )

    for sample in artifact_samples:
        result = reward_module.compute_score(
            data_source=reward_module.DATA_SOURCE,
            solution_str=str(sample.get("output", "")),
            ground_truth=str(sample.get("ground_truth", "")),
            extra_info={},
        )
        mismatch_fields = []
        for field in ARTIFACT_COMPARE_FIELDS:
            logged_value = normalize_logged_value(field, sample["logged"].get(field))
            recomputed_value = normalize_logged_value(field, result.get(field))
            if logged_value is None or recomputed_value is None or logged_value != recomputed_value:
                mismatch_fields.append(field)

        step_key = str(sample.get("step", "unknown"))
        by_step[step_key]["total_samples"] += 1
        if mismatch_fields:
            summary["samples_with_any_mismatch"] += 1
            by_step[step_key]["samples_with_any_mismatch"] += 1
            mismatches.append(
                {
                    "audit": "artifact-audit",
                    "artifact_type": sample["artifact_type"],
                    "source_path": sample["source_path"],
                    "row_index": sample["row_index"],
                    "sample_index": sample["sample_index"],
                    "step": sample.get("step"),
                    "mismatch_fields": mismatch_fields,
                    "logged": sample["logged"],
                    "recomputed": {field: float(result[field]) for field in ARTIFACT_COMPARE_FIELDS},
                }
            )
        for field in mismatch_fields:
            summary["field_mismatch_counts"][field] += 1
            by_step[step_key]["field_mismatch_counts"][field] += 1

    summary["by_step"] = dict(sorted(by_step.items(), key=lambda item: item[0]))
    return summary, mismatches


def audit_reference_alignment(
    *,
    processed_dir: str | Path,
    reward_module,
    alignment_spec_path: str | Path | None,
    docs_note_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    processed_dir = Path(processed_dir)
    alignment_spec_path = Path(alignment_spec_path) if alignment_spec_path is not None else processed_dir / "alignment_spec.json"
    mismatches: list[dict[str, Any]] = []

    if not alignment_spec_path.exists():
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["alignment_spec.json_missing"]})
        alignment_spec = None
    else:
        alignment_spec = json.loads(alignment_spec_path.read_text(encoding="utf-8"))
        if alignment_spec != reward_module.ALIGNMENT_SPEC:
            mismatches.append({"audit": "reference-audit", "mismatch_fields": ["alignment_spec_content"]})

    docs_note = Path(docs_note_path)
    if not docs_note.exists():
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["docs_note_missing"]})
    else:
        docs_text = docs_note.read_text(encoding="utf-8")
        docs_text_lower = docs_text.lower()
        required_strings = (
            "two rewards",
            "no length reward",
            "<reasoning>",
            "<answer>",
            "structured",
        )
        missing_strings = [value for value in required_strings if value not in docs_text_lower]
        if missing_strings:
            mismatches.append(
                {
                    "audit": "reference-audit",
                    "mismatch_fields": ["docs_note_content"],
                    "missing_strings": missing_strings,
                }
            )

    processed_splits = load_split_rows_from_parquet_dir(processed_dir)
    sample_row = processed_splits["train"][0]
    processed_question, processed_system_prompt = extract_processed_question_system(sample_row)
    del processed_question
    if processed_system_prompt != reward_module.SYSTEM_PROMPT:
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["system_prompt"]})
    if sample_row.get("data_source") != reward_module.DATA_SOURCE:
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["data_source"]})
    sample_extra_info = maybe_json(sample_row.get("extra_info", {}))
    if not isinstance(sample_extra_info, dict) or sample_extra_info.get("source_dataset") != "openai/gsm8k":
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["extra_info.source_dataset"]})
    if not isinstance(sample_extra_info, dict) or sample_extra_info.get("source_subset") != "main":
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["extra_info.source_subset"]})

    simplifications = set(reward_module.ALIGNMENT_SPEC.get("simplifications", []))
    required_simplifications = {
        "two_rewards_not_three_or_four",
        "binary_exact_correctness_not_approximate_matching",
        "single_strict_format_reward_not_strict_plus_soft",
        "no_length_reward",
    }
    if simplifications != required_simplifications:
        mismatches.append(
            {
                "audit": "reference-audit",
                "mismatch_fields": ["simplifications"],
                "expected": sorted(required_simplifications),
                "observed": sorted(simplifications),
            }
        )
    structured_output = reward_module.ALIGNMENT_SPEC.get("structured_output", {})
    if not isinstance(structured_output, dict) or structured_output.get("schema") != "<reasoning>...</reasoning><answer>...</answer>":
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["structured_output.schema"]})
    rewards = reward_module.ALIGNMENT_SPEC.get("rewards", {})
    if not isinstance(rewards, dict) or set(rewards.keys()) != {"format_reward", "correct_reward"}:
        mismatches.append({"audit": "reference-audit", "mismatch_fields": ["rewards.keys"]})
    excluded_features = set(reward_module.ALIGNMENT_SPEC.get("excluded_features", []))
    forbidden_absences = {"numeric_extractability_reward", "soft_format_reward", "length_reward", "partial_credit_correctness"}
    if excluded_features != forbidden_absences:
        mismatches.append(
            {
                "audit": "reference-audit",
                "mismatch_fields": ["excluded_features"],
                "expected": sorted(forbidden_absences),
                "observed": sorted(excluded_features),
            }
        )

    summary = {
        "alignment_spec_path": str(alignment_spec_path),
        "docs_note_path": str(docs_note),
        "checks_passed": len(mismatches) == 0,
        "mismatch_count": len(mismatches),
    }
    return summary, mismatches


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the modern GSM8K two-reward baseline end to end.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_outputs = argparse.ArgumentParser(add_help=False)
    common_outputs.add_argument("--summary-output", default=None)
    common_outputs.add_argument("--mismatch-output", default=None)

    dataset_parser = subparsers.add_parser("dataset-audit", parents=[common_outputs])
    dataset_parser.add_argument("--processed-dir", required=True)
    dataset_parser.add_argument("--source-mode", choices=("preprocessed", "hf"), default="preprocessed")
    dataset_parser.add_argument("--base-dir", default=None)
    dataset_parser.add_argument("--reward-module-path", default=str(DEFAULT_REWARD_MODULE_PATH))

    reward_parser = subparsers.add_parser("reward-audit", parents=[common_outputs])
    reward_parser.add_argument("--processed-dir", required=True)
    reward_parser.add_argument("--reward-module-path", default=str(DEFAULT_REWARD_MODULE_PATH))

    artifact_parser = subparsers.add_parser("artifact-audit", parents=[common_outputs])
    artifact_parser.add_argument("--input-path", required=True)
    artifact_parser.add_argument("--artifact-type", choices=("auto", "eval", "rollout"), default="auto")
    artifact_parser.add_argument("--step", type=int, default=None)
    artifact_parser.add_argument("--reward-module-path", default=str(DEFAULT_REWARD_MODULE_PATH))

    reference_parser = subparsers.add_parser("reference-audit", parents=[common_outputs])
    reference_parser.add_argument("--processed-dir", required=True)
    reference_parser.add_argument("--reward-module-path", default=str(DEFAULT_REWARD_MODULE_PATH))
    reference_parser.add_argument("--alignment-spec", default=None)
    reference_parser.add_argument("--docs-note-path", default=str(DEFAULT_DOCS_NOTE_PATH))

    all_parser = subparsers.add_parser("all", parents=[common_outputs])
    all_parser.add_argument("--processed-dir", required=True)
    all_parser.add_argument("--source-mode", choices=("preprocessed", "hf"), default="preprocessed")
    all_parser.add_argument("--base-dir", default=None)
    all_parser.add_argument("--artifact-path", default=None)
    all_parser.add_argument("--artifact-type", choices=("auto", "eval", "rollout"), default="auto")
    all_parser.add_argument("--step", type=int, default=None)
    all_parser.add_argument("--reward-module-path", default=str(DEFAULT_REWARD_MODULE_PATH))
    all_parser.add_argument("--alignment-spec", default=None)
    all_parser.add_argument("--docs-note-path", default=str(DEFAULT_DOCS_NOTE_PATH))

    return parser


def write_outputs(summary: dict[str, Any], mismatches: list[dict[str, Any]], summary_output: str | None, mismatch_output: str | None) -> None:
    if summary_output is not None:
        path = Path(summary_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mismatch_output is not None:
        output_jsonl(Path(mismatch_output), mismatches)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reward_module = load_module_from_path("gsm8k_modern_two_reward", args.reward_module_path)

    if args.command == "dataset-audit":
        source_splits = load_source_splits(args.source_mode, args.base_dir)
        processed_splits = load_split_rows_from_parquet_dir(args.processed_dir)
        summary, mismatches = audit_dataset_rows(
            source_splits=source_splits,
            processed_splits=processed_splits,
            reward_module=reward_module,
        )
    elif args.command == "reward-audit":
        processed_splits = load_split_rows_from_parquet_dir(args.processed_dir)
        summary, mismatches = audit_reward_rows(processed_splits, reward_module)
    elif args.command == "artifact-audit":
        artifact_samples = load_canonical_artifact_samples(args.input_path, args.artifact_type, args.step)
        summary, mismatches = audit_artifact_samples(artifact_samples, reward_module)
    elif args.command == "reference-audit":
        summary, mismatches = audit_reference_alignment(
            processed_dir=args.processed_dir,
            reward_module=reward_module,
            alignment_spec_path=args.alignment_spec,
            docs_note_path=args.docs_note_path,
        )
    else:
        source_splits = load_source_splits(args.source_mode, args.base_dir)
        processed_splits = load_split_rows_from_parquet_dir(args.processed_dir)
        dataset_summary, dataset_mismatches = audit_dataset_rows(
            source_splits=source_splits,
            processed_splits=processed_splits,
            reward_module=reward_module,
        )
        reward_summary, reward_mismatches = audit_reward_rows(processed_splits, reward_module)
        reference_summary, reference_mismatches = audit_reference_alignment(
            processed_dir=args.processed_dir,
            reward_module=reward_module,
            alignment_spec_path=args.alignment_spec,
            docs_note_path=args.docs_note_path,
        )
        artifact_summary = None
        artifact_mismatches: list[dict[str, Any]] = []
        if args.artifact_path is not None:
            artifact_samples = load_canonical_artifact_samples(args.artifact_path, args.artifact_type, args.step)
            artifact_summary, artifact_mismatches = audit_artifact_samples(artifact_samples, reward_module)

        summary = {
            "dataset_audit": dataset_summary,
            "reward_audit": reward_summary,
            "reference_audit": reference_summary,
            "artifact_audit": artifact_summary,
            "checks_passed": not (dataset_mismatches or reward_mismatches or reference_mismatches or artifact_mismatches),
        }
        mismatches = dataset_mismatches + reward_mismatches + reference_mismatches + artifact_mismatches

    print_summary(summary)
    write_outputs(summary, mismatches, args.summary_output, args.mismatch_output)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
