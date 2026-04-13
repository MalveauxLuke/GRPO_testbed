#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import re
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from aime_divergence.answer_utils import check_answer, ground_truth_sanity

if TYPE_CHECKING:
    from aime_divergence.data_loader import AIMEProblem


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_LOGPROBS = 50
CHAT_TEMPLATE = "User: \n {question} \n Please reason step by step, and put your final answer within \\boxed{{}}. \n \n Assistant:"


class LogprobsExtractionError(RuntimeError):
    """Raised when vLLM did not return token-level logprobs in a supported shape."""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_tensorboard_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _render_prompt(problem: AIMEProblem) -> str:
    return CHAT_TEMPLATE.format(question=problem.problem_text)


def _default_output_dir() -> Path:
    env_dir = os.environ.get("AIME_DIVERGENCE_OUTPUT_DIR")
    if env_dir:
        return Path(env_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("aime_divergence") / "outputs" / f"logged_{timestamp}"


def _correctness_status(is_correct: bool | None) -> int:
    if is_correct is True:
        return 1
    if is_correct is False:
        return 0
    return -1


def _string_array(values: list[Any]) -> np.ndarray:
    return np.asarray(["" if value is None else str(value) for value in values], dtype=np.str_)


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise LogprobsExtractionError(f"Could not parse {field_name} as int from {value!r}") from exc


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise LogprobsExtractionError(f"Could not parse {field_name} as float from {value!r}") from exc


def approx_entropy(top_k_logprobs: np.ndarray) -> float:
    probs = np.exp(top_k_logprobs.astype(np.float64))
    top_k_h = -np.sum(probs * top_k_logprobs.astype(np.float64))
    remaining = max(0.0, 1.0 - float(np.sum(probs)))
    if remaining > 1e-8:
        top_k_h -= remaining * np.log(remaining)
    return float(top_k_h)


def _normalize_entry_rank(entry_rank: int | None, min_rank: int | None) -> int | None:
    if entry_rank is None or min_rank is None:
        return None
    return entry_rank - min_rank


def _candidate_token_id(raw_key: Any, raw_value: Any) -> int:
    for candidate in (raw_key, getattr(raw_value, "token_id", None), getattr(raw_value, "id", None)):
        if candidate is None:
            continue
        try:
            return int(candidate)
        except Exception:
            continue
    raise LogprobsExtractionError(f"Could not determine token id from logprobs entry key={raw_key!r} value={raw_value!r}")


def _candidate_logprob(raw_value: Any) -> float:
    if hasattr(raw_value, "logprob"):
        return _coerce_float(raw_value.logprob, "logprob")
    return _coerce_float(raw_value, "logprob")


def _candidate_rank(raw_value: Any) -> int | None:
    rank = getattr(raw_value, "rank", None)
    if rank is None:
        return None
    return _coerce_int(rank, "rank")


def _first_entry_summary(entry: Any) -> tuple[str, str | None, str | None]:
    if not isinstance(entry, Mapping):
        return (type(entry).__name__, None, None)
    try:
        first_key, first_value = next(iter(entry.items()))
    except StopIteration:
        return (type(entry).__name__, None, None)
    return (type(entry).__name__, repr(first_key), type(first_value).__name__)


def summarize_logprobs_probe(output: Any) -> dict[str, Any]:
    logprobs = getattr(output, "logprobs", None)
    token_ids = getattr(output, "token_ids", None)
    has_logprobs = logprobs is not None and len(logprobs) > 0
    has_token_ids = token_ids is not None and len(token_ids) > 0
    first_entry_type, first_entry_key_sample, first_entry_value_type = (
        _first_entry_summary(logprobs[0]) if has_logprobs else (None, None, None)
    )
    return {
        "output_type": type(output).__name__,
        "output_attrs": sorted(name for name in dir(output) if not name.startswith("_"))[:50],
        "logprobs_type": type(logprobs).__name__ if logprobs is not None else None,
        "logprobs_length": len(logprobs) if logprobs is not None else None,
        "first_entry_type": first_entry_type,
        "first_entry_key_sample": first_entry_key_sample,
        "first_entry_value_type": first_entry_value_type,
        "sampled_token_id": int(token_ids[0]) if has_token_ids else None,
    }


def extract_token_data(output: Any, top_k: int = DEFAULT_LOGPROBS) -> dict[str, np.ndarray]:
    token_ids = getattr(output, "token_ids", None)
    logprobs = getattr(output, "logprobs", None)
    if token_ids is None:
        raise LogprobsExtractionError("Completion output is missing token_ids.")
    if logprobs is None:
        raise LogprobsExtractionError("Completion output did not return logprobs. Make sure SamplingParams(logprobs=K) is set.")
    if len(logprobs) == 0:
        raise LogprobsExtractionError("Completion output returned an empty logprobs list.")
    if len(token_ids) != len(logprobs):
        raise LogprobsExtractionError(
            f"Completion output returned mismatched token_ids/logprobs lengths: {len(token_ids)} vs {len(logprobs)}."
        )

    seq_len = len(token_ids)
    sampled_logprobs = np.zeros(seq_len, dtype=np.float64)
    sampled_ranks = np.full(seq_len, -1, dtype=np.int32)
    entropies = np.zeros(seq_len, dtype=np.float64)
    top_k_ids = np.full((seq_len, top_k), -1, dtype=np.int32)
    top_k_probs = np.zeros((seq_len, top_k), dtype=np.float64)

    for position, (token_id, entry) in enumerate(zip(token_ids, logprobs, strict=True)):
        if not isinstance(entry, Mapping):
            raise LogprobsExtractionError(
                f"Unsupported logprobs payload at position {position}: expected Mapping, got {type(entry).__name__}."
            )

        candidates: list[dict[str, Any]] = []
        for raw_key, raw_value in entry.items():
            candidates.append(
                {
                    "token_id": _candidate_token_id(raw_key, raw_value),
                    "logprob": _candidate_logprob(raw_value),
                    "rank": _candidate_rank(raw_value),
                }
            )

        if not candidates:
            raise LogprobsExtractionError(f"Logprobs payload at position {position} was empty.")

        raw_ranks = [candidate["rank"] for candidate in candidates if candidate["rank"] is not None]
        min_rank = min(raw_ranks) if raw_ranks else None

        if raw_ranks:
            for candidate in candidates:
                candidate["normalized_rank"] = _normalize_entry_rank(candidate["rank"], min_rank)
            ordered_candidates = sorted(
                candidates,
                key=lambda candidate: (
                    candidate["normalized_rank"] if candidate["normalized_rank"] is not None else 10**9,
                    -candidate["logprob"],
                    candidate["token_id"],
                ),
            )
        else:
            ordered_candidates = sorted(
                candidates,
                key=lambda candidate: (-candidate["logprob"], candidate["token_id"]),
            )
            for rank, candidate in enumerate(ordered_candidates):
                candidate["normalized_rank"] = rank

        sampled_candidate = next((candidate for candidate in ordered_candidates if candidate["token_id"] == int(token_id)), None)
        if sampled_candidate is None:
            raise LogprobsExtractionError(
                f"Sampled token id {int(token_id)} was not present in logprobs payload at position {position}."
            )

        sampled_logprobs[position] = sampled_candidate["logprob"]
        sampled_ranks[position] = int(sampled_candidate["normalized_rank"])

        limited_candidates = ordered_candidates[:top_k]
        limited_logprobs = np.asarray([candidate["logprob"] for candidate in limited_candidates], dtype=np.float64)
        if limited_candidates:
            top_k_ids[position, : len(limited_candidates)] = np.asarray(
                [candidate["token_id"] for candidate in limited_candidates], dtype=np.int32
            )
            top_k_probs[position, : len(limited_candidates)] = np.exp(limited_logprobs)
            entropies[position] = approx_entropy(limited_logprobs)

    return {
        "token_ids": np.asarray(token_ids, dtype=np.int32),
        "token_probs": np.exp(sampled_logprobs).astype(np.float32),
        "token_ranks": sampled_ranks.astype(np.int32),
        "token_entropies": entropies.astype(np.float32),
        "cumulative_logprobs": np.cumsum(sampled_logprobs, dtype=np.float64).astype(np.float32),
        "top_k_ids": top_k_ids.astype(np.int32),
        "top_k_probs": top_k_probs.astype(np.float32),
    }


def build_problem_token_payload(
    problem: AIMEProblem,
    rollouts: list[dict[str, Any]],
    token_data_per_rollout: list[dict[str, np.ndarray]],
    top_k: int,
) -> dict[str, np.ndarray]:
    rollout_count = len(rollouts)
    max_len = max((int(token_data["token_ids"].shape[0]) for token_data in token_data_per_rollout), default=0)

    token_ids = np.full((rollout_count, max_len), -1, dtype=np.int32)
    token_probs = np.zeros((rollout_count, max_len), dtype=np.float32)
    token_ranks = np.full((rollout_count, max_len), -1, dtype=np.int32)
    token_entropies = np.zeros((rollout_count, max_len), dtype=np.float32)
    cumulative_logprobs = np.zeros((rollout_count, max_len), dtype=np.float32)
    top_k_ids = np.full((rollout_count, max_len, top_k), -1, dtype=np.int32)
    top_k_probs = np.zeros((rollout_count, max_len, top_k), dtype=np.float32)
    num_tokens = np.zeros(rollout_count, dtype=np.int32)

    for rollout_idx, token_data in enumerate(token_data_per_rollout):
        seq_len = int(token_data["token_ids"].shape[0])
        num_tokens[rollout_idx] = seq_len
        token_ids[rollout_idx, :seq_len] = token_data["token_ids"]
        token_probs[rollout_idx, :seq_len] = token_data["token_probs"]
        token_ranks[rollout_idx, :seq_len] = token_data["token_ranks"]
        token_entropies[rollout_idx, :seq_len] = token_data["token_entropies"]
        cumulative_logprobs[rollout_idx, :seq_len] = token_data["cumulative_logprobs"]
        top_k_ids[rollout_idx, :seq_len, :] = token_data["top_k_ids"]
        top_k_probs[rollout_idx, :seq_len, :] = token_data["top_k_probs"]

    return {
        "problem_id": np.asarray(problem.problem_id, dtype=np.str_),
        "problem_text": np.asarray(problem.problem_text, dtype=np.str_),
        "ground_truth": np.asarray(problem.ground_truth_answer, dtype=np.str_),
        "dataset": np.asarray(problem.dataset, dtype=np.str_),
        "correctness_status": np.asarray(
            [_correctness_status(rollout["is_correct"]) for rollout in rollouts],
            dtype=np.int8,
        ),
        "extracted_answers": _string_array([rollout["extracted_answer"] for rollout in rollouts]),
        "extraction_methods": _string_array([rollout["extraction_method"] for rollout in rollouts]),
        "num_tokens": num_tokens,
        "token_ids": token_ids,
        "token_probs": token_probs,
        "token_ranks": token_ranks,
        "token_entropies": token_entropies,
        "cumulative_logprobs": cumulative_logprobs,
        "top_k_ids": top_k_ids,
        "top_k_probs": top_k_probs,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, path)


def _load_partial_problem_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    rows: list[dict[str, Any]] = []
    lines = text.splitlines()
    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError:
            if line_idx == len(lines):
                print(
                    f"[aime-logged] Ignoring malformed trailing checkpoint row at {path}:{line_idx}; "
                    "the problem will be regenerated."
                )
                break
            raise
    return rows


def append_partial_problem_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False))
        fp.write("\n")


def should_skip_problem(problem_id: str, token_data_path: Path, rows_by_id: dict[str, dict[str, Any]]) -> bool:
    return token_data_path.exists() and problem_id in rows_by_id


def _rollout_rows_in_dataset_order(
    problems: list[AIMEProblem],
    rows_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [rows_by_id[problem.problem_id] for problem in problems if problem.problem_id in rows_by_id]


def _build_summary(problems_json: list[dict[str, Any]], ground_truth_warnings: list[dict[str, Any]]) -> dict[str, Any]:
    split_distribution = Counter(problem["split_ratio"] for problem in problems_json)
    extraction_method_counts = Counter(
        rollout["extraction_method"] for problem in problems_json for rollout in problem["rollouts"]
    )
    response_lengths = [rollout["num_tokens"] for problem in problems_json for rollout in problem["rollouts"]]
    total_rollouts = sum(len(problem["rollouts"]) for problem in problems_json)
    total_correct = sum(problem["num_correct"] for problem in problems_json)
    total_incorrect = sum(problem["num_incorrect"] for problem in problems_json)
    total_unknown = sum(problem["num_unknown"] for problem in problems_json)
    total_known = total_correct + total_incorrect
    extraction_failures = [
        {
            "problem_id": problem["problem_id"],
            "rollout_idx": rollout["rollout_idx"],
            "extraction_method": rollout["extraction_method"],
        }
        for problem in problems_json
        for rollout in problem["rollouts"]
        if rollout["extraction_method"] == "extraction_failed"
    ]
    all_correct_problems = sum(
        1
        for problem in problems_json
        if problem["num_unknown"] == 0 and problem["num_correct"] > 0 and problem["num_incorrect"] == 0
    )
    all_incorrect_problems = sum(
        1
        for problem in problems_json
        if problem["num_unknown"] == 0 and problem["num_incorrect"] > 0 and problem["num_correct"] == 0
    )
    return {
        "total_problems": len(problems_json),
        "total_rollouts": total_rollouts,
        "split_distribution": dict(sorted(split_distribution.items())),
        "usable_problems": sum(
            1
            for problem in problems_json
            if problem["num_unknown"] == 0 and problem["num_correct"] > 0 and problem["num_incorrect"] > 0
        ),
        "mixed_known_with_unknown_problems": sum(
            1
            for problem in problems_json
            if problem["num_unknown"] > 0 and problem["num_correct"] > 0 and problem["num_incorrect"] > 0
        ),
        "all_correct_problems": all_correct_problems,
        "all_incorrect_problems": all_incorrect_problems,
        "total_correct": total_correct,
        "total_incorrect": total_incorrect,
        "total_unknown": total_unknown,
        "avg_response_length_tokens": mean(response_lengths) if response_lengths else 0.0,
        "extraction_method_counts": dict(sorted(extraction_method_counts.items())),
        "overall_accuracy": (total_correct / total_known) if total_known else 0.0,
        "overall_accuracy_including_unknown_as_incorrect": (total_correct / total_rollouts) if total_rollouts else 0.0,
        "extraction_failed_rollouts": extraction_failures,
        "ground_truth_warnings": ground_truth_warnings,
    }


def _write_tensorboard(problems_json: list[dict[str, Any]], summary: dict[str, Any], log_dir: Path) -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[aime-logged] TensorBoard disabled: failed to import SummaryWriter: {exc}")
        return False

    log_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        writer.add_scalar("aime/summary/total_problems", summary["total_problems"], 0)
        writer.add_scalar("aime/summary/total_rollouts", summary["total_rollouts"], 0)
        writer.add_scalar("aime/summary/usable_problems", summary["usable_problems"], 0)
        writer.add_scalar("aime/summary/all_correct_problems", summary["all_correct_problems"], 0)
        writer.add_scalar("aime/summary/all_incorrect_problems", summary["all_incorrect_problems"], 0)
        writer.add_scalar("aime/summary/total_correct", summary["total_correct"], 0)
        writer.add_scalar("aime/summary/total_incorrect", summary["total_incorrect"], 0)
        writer.add_scalar("aime/summary/total_unknown", summary["total_unknown"], 0)
        writer.add_scalar("aime/summary/overall_accuracy_known", summary["overall_accuracy"], 0)
        writer.add_scalar(
            "aime/summary/overall_accuracy_unknown_as_incorrect",
            summary["overall_accuracy_including_unknown_as_incorrect"],
            0,
        )
        writer.add_scalar("aime/summary/avg_response_length_tokens", summary["avg_response_length_tokens"], 0)
        writer.add_scalar("aime/summary/extraction_failed_rollouts", len(summary["extraction_failed_rollouts"]), 0)

        for split_ratio, count in summary["split_distribution"].items():
            writer.add_scalar(f"aime/split_distribution/{_sanitize_tensorboard_tag(split_ratio)}", count, 0)
        for method, count in summary["extraction_method_counts"].items():
            writer.add_scalar(f"aime/extraction_method/{_sanitize_tensorboard_tag(method)}", count, 0)

        for problem_idx, problem in enumerate(problems_json):
            rollouts = problem["rollouts"]
            known = problem["num_correct"] + problem["num_incorrect"]
            total = len(rollouts)
            writer.add_scalar("aime/problem/num_correct", problem["num_correct"], problem_idx)
            writer.add_scalar("aime/problem/num_incorrect", problem["num_incorrect"], problem_idx)
            writer.add_scalar("aime/problem/num_unknown", problem["num_unknown"], problem_idx)
            writer.add_scalar("aime/problem/accuracy_known", problem["num_correct"] / known if known else 0.0, problem_idx)
            writer.add_scalar("aime/problem/unknown_fraction", problem["num_unknown"] / total if total else 0.0, problem_idx)
            writer.add_scalar(
                "aime/problem/avg_response_length_tokens",
                mean(rollout["num_tokens"] for rollout in rollouts) if rollouts else 0.0,
                problem_idx,
            )
            writer.add_text("aime/problem/id", problem["problem_id"], problem_idx)

        writer.flush()
    print(f"[aime-logged] TensorBoard log dir: {log_dir}")
    return True


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in ("vllm", "torch", "datasets", "transformers", "math-verify", "numpy", "tensorboard"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = "<missing>"
    return versions


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    try:
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        )
    except Exception:
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def build_rollout_results_payload(
    metadata: dict[str, Any],
    problems_json: list[dict[str, Any]],
    ground_truth_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "problems": problems_json,
        "summary": _build_summary(problems_json, ground_truth_warnings),
    }


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _token_data_stats(token_data_dir: Path) -> dict[str, Any]:
    files = sorted(token_data_dir.glob("*.npz"))
    total_size = sum(path.stat().st_size for path in files)
    return {"npz_count": len(files), "total_size_bytes": total_size}


def _npz_sanity_text(token_data_dir: Path, problems_json: list[dict[str, Any]]) -> str:
    if not problems_json:
        return "none"
    selected_problem = next(
        (
            problem
            for problem in problems_json
            if problem["num_unknown"] == 0 and problem["num_correct"] > 0 and problem["num_incorrect"] > 0
        ),
        problems_json[0],
    )
    npz_path = token_data_dir / f"{selected_problem['problem_id']}.npz"
    if not npz_path.exists():
        return f"missing: {npz_path}"
    lines = [f"{npz_path.name}:"]
    with np.load(npz_path, allow_pickle=False) as data:
        for key in sorted(data.files):
            lines.append(f"  {key}: shape={data[key].shape} dtype={data[key].dtype}")
    return "\n".join(lines)


def _render_summary_text(
    summary: dict[str, Any],
    token_stats: dict[str, Any],
    logprobs_failures: list[str],
    npz_sanity: str,
) -> str:
    lines = [
        "",
        "=" * 80,
        "AIME logged rollout summary",
        "=" * 80,
        f"1. Total problems processed: {summary['total_problems']}",
        f"2. Total rollouts generated: {summary['total_rollouts']}",
        f"3. Mixed correct/incorrect usable problems: {summary['usable_problems']}",
        f"4. All-correct problems: {summary['all_correct_problems']}",
        f"5. All-incorrect problems: {summary['all_incorrect_problems']}",
        f"6. Average response length tokens: {summary['avg_response_length_tokens']:.2f}",
        f"7. Extraction failures: {len(summary['extraction_failed_rollouts'])}",
        (
            "Known/unknown rollout counts: "
            f"correct={summary['total_correct']} incorrect={summary['total_incorrect']} unknown={summary['total_unknown']}"
        ),
        f"Mixed known problems with unknown rollouts: {summary['mixed_known_with_unknown_problems']}",
        "8. Split ratio distribution:",
    ]
    lines.extend(f"   {split_ratio}: {count}" for split_ratio, count in sorted(summary["split_distribution"].items()))
    lines.append("9. Extraction method counts:")
    lines.extend(f"   {method}: {count}" for method, count in sorted(summary["extraction_method_counts"].items()))
    lines.append("10. Ground-truth sanity warnings:")
    if summary["ground_truth_warnings"]:
        lines.extend(f"   {warning}" for warning in summary["ground_truth_warnings"])
    else:
        lines.append("   none")
    lines.append(f"Overall accuracy on known rollouts: {summary['overall_accuracy']:.4f}")
    lines.append(
        "Overall accuracy if unknowns are counted as incorrect: "
        f"{summary['overall_accuracy_including_unknown_as_incorrect']:.4f}"
    )
    lines.append(f"11. Token data files created: {token_stats['npz_count']}")
    lines.append(f"12. Token data directory size: {_format_bytes(token_stats['total_size_bytes'])}")
    lines.append(f"13. Average tokens per rollout: {summary['avg_response_length_tokens']:.2f}")
    lines.append("14. Logprobs extraction failures:")
    if logprobs_failures:
        lines.extend(f"   {failure}" for failure in logprobs_failures)
    else:
        lines.append("   none")
    lines.append("15. NPZ shape sanity check:")
    if npz_sanity == "none":
        lines.append("   none")
    else:
        lines.extend(f"   {line}" for line in npz_sanity.splitlines())
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate AIME rollouts with per-token logprob logging.")
    parser.add_argument("--model", default=os.environ.get("AIME_DIVERGENCE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=int(os.environ.get("AIME_DIVERGENCE_SAMPLES", "8")))
    parser.add_argument("--logprobs", type=int, default=int(os.environ.get("AIME_DIVERGENCE_LOGPROBS", str(DEFAULT_LOGPROBS))))
    parser.add_argument(
        "--max-problems",
        type=int,
        default=int(os.environ.get("AIME_DIVERGENCE_MAX_PROBLEMS", "0")),
        help="Limit to the first N loaded problems for smoke tests. Use 0 for all problems.",
    )
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("AIME_DIVERGENCE_TEMPERATURE", "1.0")))
    parser.add_argument("--top-p", type=float, default=float(os.environ.get("AIME_DIVERGENCE_TOP_P", "0.95")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("AIME_DIVERGENCE_MAX_TOKENS", "16384")))
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("AIME_DIVERGENCE_MAX_MODEL_LEN", "16384")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("AIME_DIVERGENCE_TP_SIZE", "1")))
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("AIME_DIVERGENCE_GPU_MEMORY_UTILIZATION", "0.90")),
    )
    parser.add_argument("--dtype", default=os.environ.get("AIME_DIVERGENCE_DTYPE", "bfloat16"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("AIME_DIVERGENCE_SEED", "0")))
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=_env_flag("AIME_DIVERGENCE_FAIL_FAST", False))
    parser.add_argument("--enable-tensorboard", action=argparse.BooleanOptionalAction, default=_env_flag("AIME_DIVERGENCE_TENSORBOARD", False))
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path(os.environ["AIME_DIVERGENCE_TENSORBOARD_DIR"])
        if os.environ.get("AIME_DIVERGENCE_TENSORBOARD_DIR")
        else None,
    )
    return parser


def _build_sampling_params(args: argparse.Namespace) -> Any:
    try:
        from vllm import SamplingParams
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("vLLM is required to run aime_divergence.run_logged") from exc

    kwargs: dict[str, Any] = {
        "n": args.samples_per_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "logprobs": args.logprobs,
    }
    if "seed" in inspect.signature(SamplingParams).parameters:
        kwargs["seed"] = args.seed
    return SamplingParams(**kwargs)


def _write_progress_artifacts(
    output_dir: Path,
    metadata: dict[str, Any],
    problems: list[AIMEProblem],
    rows_by_id: dict[str, dict[str, Any]],
    ground_truth_warnings: list[dict[str, Any]],
    enable_tensorboard: bool,
    tensorboard_dir: Path | None,
    logprobs_failures: list[str],
) -> dict[str, Any]:
    ordered_rows = _rollout_rows_in_dataset_order(problems, rows_by_id)
    rollout_payload = build_rollout_results_payload(metadata, ordered_rows, ground_truth_warnings)
    token_stats = _token_data_stats(output_dir / "token_data")
    npz_sanity = _npz_sanity_text(output_dir / "token_data", ordered_rows)
    summary_text = _render_summary_text(rollout_payload["summary"], token_stats, logprobs_failures, npz_sanity)

    _write_json_atomic(output_dir / "rollout_results.json", rollout_payload)
    _write_text_atomic(output_dir / "run_summary.txt", summary_text)
    if enable_tensorboard:
        _write_tensorboard(ordered_rows, rollout_payload["summary"], tensorboard_dir or (output_dir / "tensorboard"))
    return rollout_payload


def main() -> int:
    from aime_divergence.data_loader import load_aime_2024_2025, print_dataset_confirmation

    args = _build_arg_parser().parse_args()
    if args.max_problems < 0:
        raise ValueError("--max-problems must be non-negative")
    if args.logprobs <= 0:
        raise ValueError("--logprobs must be positive")

    output_dir = args.output_dir or _default_output_dir()
    token_data_dir = output_dir / "token_data"
    tensorboard_dir = args.tensorboard_dir or (output_dir / "tensorboard")
    partial_results_path = output_dir / "_rollout_results.partial.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    token_data_dir.mkdir(parents=True, exist_ok=True)

    problems, dataset_sources = load_aime_2024_2025()
    if args.max_problems:
        problems = problems[: args.max_problems]
        print(f"[aime-logged] smoke/problem limit active: using first {len(problems)} problems")
    print_dataset_confirmation(problems, dataset_sources)

    ground_truth_warnings = [
        warning
        for warning in (ground_truth_sanity(problem.problem_id, problem.ground_truth_answer) for problem in problems)
        if warning is not None
    ]

    metadata = {
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_config": {
            "samples_per_prompt": args.samples_per_prompt,
            "logprobs": args.logprobs,
            "max_problems": args.max_problems,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": args.seed,
        },
        "execution_config": {
            "fail_fast": args.fail_fast,
        },
        "dataset_sources": dataset_sources,
        "chat_template": CHAT_TEMPLATE,
        "padding_conventions": {
            "int_pad": -1,
            "float_pad": 0.0,
            "correctness_status": {"correct": 1, "incorrect": 0, "unknown": -1},
        },
        "dtypes": {
            "token_ids": "int32",
            "token_probs": "float32",
            "token_ranks": "int32",
            "token_entropies": "float32",
            "cumulative_logprobs": "float32",
            "top_k_ids": "int32",
            "top_k_probs": "float32",
            "correctness_status": "int8",
            "num_tokens": "int32",
        },
        "env_overrides": {
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        },
        "package_versions": _package_versions(),
        **_git_metadata(),
    }
    _write_json_atomic(output_dir / "config.json", metadata)

    rows_by_id = {row["problem_id"]: row for row in _load_partial_problem_rows(partial_results_path)}
    logprobs_failures: list[str] = []

    try:
        from vllm import LLM
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("vLLM is required to run aime_divergence.run_logged") from exc

    sampling_params = _build_sampling_params(args)

    print(f"[aime-logged] loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    probe_logged = False

    for problem in problems:
        token_data_path = token_data_dir / f"{problem.problem_id}.npz"
        if should_skip_problem(problem.problem_id, token_data_path, rows_by_id):
            print(f"[aime-logged] skipping completed problem {problem.problem_id}")
            continue

        if token_data_path.exists() and problem.problem_id not in rows_by_id:
            print(f"[aime-logged] found stale token data without checkpoint row for {problem.problem_id}; regenerating")

        rendered_prompt = _render_prompt(problem)
        print(f"[aime-logged] generating {args.samples_per_prompt} logged rollouts for {problem.problem_id}")
        try:
            request_output = llm.generate([rendered_prompt], sampling_params=sampling_params)[0]
            rollouts: list[dict[str, Any]] = []
            token_data_per_rollout: list[dict[str, np.ndarray]] = []

            for rollout_idx, output in enumerate(request_output.outputs):
                if not probe_logged:
                    probe = summarize_logprobs_probe(output)
                    print(f"[aime-logged] logprobs probe: {json.dumps(probe, indent=2, ensure_ascii=False)}")

                token_data = extract_token_data(output, top_k=args.logprobs)
                probe_logged = True
                token_data_per_rollout.append(token_data)

                answer_info = check_answer(output.text, problem.ground_truth_answer)
                rollouts.append(
                    {
                        "rollout_idx": rollout_idx,
                        "generated_text": output.text,
                        "num_tokens": len(output.token_ids),
                        "finish_reason": getattr(output, "finish_reason", None),
                        "extracted_answer": answer_info["extracted_answer"],
                        "is_correct": answer_info["is_correct"],
                        "extraction_method": answer_info["extraction_method"],
                    }
                )

            num_correct = sum(1 for rollout in rollouts if rollout["is_correct"] is True)
            num_incorrect = sum(1 for rollout in rollouts if rollout["is_correct"] is False)
            num_unknown = sum(1 for rollout in rollouts if rollout["is_correct"] is None)
            split_ratio = f"{num_correct}/{num_incorrect}" if num_unknown == 0 else f"{num_correct}/{num_incorrect}/{num_unknown}"

            problem_row = {
                "problem_id": problem.problem_id,
                "problem_text": problem.problem_text,
                "ground_truth": problem.ground_truth_answer,
                "dataset": problem.dataset,
                "source": problem.source,
                "source_row_index": problem.source_row_index,
                "rendered_prompt": rendered_prompt,
                "rollouts": rollouts,
                "num_correct": num_correct,
                "num_incorrect": num_incorrect,
                "num_unknown": num_unknown,
                "split_ratio": split_ratio,
                "token_data_file": f"token_data/{problem.problem_id}.npz",
            }

            token_payload = build_problem_token_payload(problem, rollouts, token_data_per_rollout, top_k=args.logprobs)
            write_npz_atomic(token_data_path, token_payload)
            append_partial_problem_row(partial_results_path, problem_row)
            rows_by_id[problem.problem_id] = problem_row

            print(f"[aime-logged] {problem.problem_id}: split={split_ratio} ground_truth={problem.ground_truth_answer}")
            _write_progress_artifacts(
                output_dir=output_dir,
                metadata=metadata,
                problems=problems,
                rows_by_id=rows_by_id,
                ground_truth_warnings=ground_truth_warnings,
                enable_tensorboard=args.enable_tensorboard,
                tensorboard_dir=tensorboard_dir,
                logprobs_failures=logprobs_failures,
            )
        except Exception as exc:
            print(f"[aime-logged] ERROR on {problem.problem_id}: {type(exc).__name__}: {exc}")
            print(traceback.format_exc(), end="")
            logprobs_failures.append(f"{problem.problem_id}: {type(exc).__name__}: {exc}")
            _write_progress_artifacts(
                output_dir=output_dir,
                metadata=metadata,
                problems=problems,
                rows_by_id=rows_by_id,
                ground_truth_warnings=ground_truth_warnings,
                enable_tensorboard=args.enable_tensorboard,
                tensorboard_dir=tensorboard_dir,
                logprobs_failures=logprobs_failures,
            )
            if args.fail_fast:
                raise
            print(f"[aime-logged] continuing after failure on {problem.problem_id}")
            continue

    final_payload = _write_progress_artifacts(
        output_dir=output_dir,
        metadata=metadata,
        problems=problems,
        rows_by_id=rows_by_id,
        ground_truth_warnings=ground_truth_warnings,
        enable_tensorboard=args.enable_tensorboard,
        tensorboard_dir=tensorboard_dir,
        logprobs_failures=logprobs_failures,
    )

    if len(final_payload["problems"]) == len(problems) and partial_results_path.exists():
        partial_results_path.unlink()

    print(f"[aime-logged] wrote {output_dir / 'rollout_results.json'}")
    print((output_dir / "run_summary.txt").read_text(encoding="utf-8"), end="")
    if logprobs_failures:
        print(f"[aime-logged] completed with {len(logprobs_failures)} problem-level failures.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
