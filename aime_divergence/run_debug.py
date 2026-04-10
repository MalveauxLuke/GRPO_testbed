#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from vllm import LLM, SamplingParams

from aime_divergence.answer_utils import check_answer, ground_truth_sanity
from aime_divergence.data_loader import AIMEProblem, load_aime_2024_2025, print_dataset_confirmation


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_OUTPUT_PATH = Path("aime_divergence") / "outputs" / "aime_rollouts_debug.json"
CHAT_TEMPLATE = "User: \n {question} \n Please reason step by step, and put your final answer within \\boxed{{}}. \n \n Assistant:"


def _render_prompt(problem: AIMEProblem) -> str:
    return CHAT_TEMPLATE.format(question=problem.problem_text)


def _output_path_from_env() -> Path:
    env_path = os.environ.get("AIME_DIVERGENCE_OUTPUT_PATH")
    if env_path:
        return Path(env_path)
    output_dir = os.environ.get("AIME_DIVERGENCE_OUTPUT_DIR")
    if output_dir:
        return Path(output_dir) / "aime_rollouts_debug.json"
    return DEFAULT_OUTPUT_PATH


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_tensorboard_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


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


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("AIME divergence debug summary")
    print("=" * 80)
    print(f"1. Total problems processed: {summary['total_problems']}")
    print(f"2. Total rollouts generated: {summary['total_rollouts']}")
    print(f"3. Mixed correct/incorrect usable problems: {summary['usable_problems']}")
    print(f"4. All-correct problems: {summary['all_correct_problems']}")
    print(f"5. All-incorrect problems: {summary['all_incorrect_problems']}")
    print(f"6. Average response length tokens: {summary['avg_response_length_tokens']:.2f}")
    print(f"7. Extraction failures: {len(summary['extraction_failed_rollouts'])}")
    print(
        "Known/unknown rollout counts: "
        f"correct={summary['total_correct']} incorrect={summary['total_incorrect']} unknown={summary['total_unknown']}"
    )
    print(f"Mixed known problems with unknown rollouts: {summary['mixed_known_with_unknown_problems']}")
    print("8. Split ratio distribution:")
    for split_ratio, count in sorted(summary["split_distribution"].items()):
        print(f"   {split_ratio}: {count}")
    print("9. Extraction method counts:")
    for method, count in sorted(summary["extraction_method_counts"].items()):
        print(f"   {method}: {count}")
    print("10. Ground-truth sanity warnings:")
    if summary["ground_truth_warnings"]:
        for warning in summary["ground_truth_warnings"]:
            print(f"   {warning}")
    else:
        print("   none")
    print(f"Overall accuracy on known rollouts: {summary['overall_accuracy']:.4f}")
    print(
        "Overall accuracy if unknowns are counted as incorrect: "
        f"{summary['overall_accuracy_including_unknown_as_incorrect']:.4f}"
    )


def _write_tensorboard(problems_json: list[dict[str, Any]], summary: dict[str, Any], log_dir: Path) -> bool:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[aime-debug] TensorBoard disabled: failed to import SummaryWriter: {exc}")
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
    print(f"[aime-debug] TensorBoard log dir: {log_dir}")
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and verify 8 AIME rollouts per problem with vLLM.")
    parser.add_argument("--model", default=os.environ.get("AIME_DIVERGENCE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-path", type=Path, default=_output_path_from_env())
    parser.add_argument("--samples-per-prompt", type=int, default=int(os.environ.get("AIME_DIVERGENCE_SAMPLES", "8")))
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
    parser.add_argument("--enable-tensorboard", action=argparse.BooleanOptionalAction, default=_env_flag("AIME_DIVERGENCE_TENSORBOARD", False))
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path(os.environ["AIME_DIVERGENCE_TENSORBOARD_DIR"])
        if os.environ.get("AIME_DIVERGENCE_TENSORBOARD_DIR")
        else None,
    )
    return parser


def _build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    kwargs: dict[str, Any] = {
        "n": args.samples_per_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if "seed" in inspect.signature(SamplingParams).parameters:
        kwargs["seed"] = args.seed
    return SamplingParams(**kwargs)


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.max_problems < 0:
        raise ValueError("--max-problems must be non-negative")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    problems, dataset_sources = load_aime_2024_2025()
    if args.max_problems:
        problems = problems[: args.max_problems]
        print(f"[aime-debug] smoke/problem limit active: using first {len(problems)} problems")
    print_dataset_confirmation(problems, dataset_sources)

    ground_truth_warnings = [
        warning
        for warning in (ground_truth_sanity(problem.problem_id, problem.ground_truth_answer) for problem in problems)
        if warning is not None
    ]

    rendered_prompts = [_render_prompt(problem) for problem in problems]
    sampling_params = _build_sampling_params(args)

    print(f"[aime-debug] loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"[aime-debug] generating {args.samples_per_prompt} rollouts for {len(rendered_prompts)} prompts")
    request_outputs = llm.generate(rendered_prompts, sampling_params=sampling_params)

    problems_json: list[dict[str, Any]] = []
    for problem, rendered_prompt, request_output in zip(problems, rendered_prompts, request_outputs, strict=True):
        rollouts: list[dict[str, Any]] = []
        for rollout_idx, output in enumerate(request_output.outputs):
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
        problems_json.append(
            {
                "problem_id": problem.problem_id,
                "problem_text": problem.problem_text,
                "ground_truth": problem.ground_truth_answer,
                "dataset": problem.dataset,
                "source": problem.source,
                "rendered_prompt": rendered_prompt,
                "rollouts": rollouts,
                "num_correct": num_correct,
                "num_incorrect": num_incorrect,
                "num_unknown": num_unknown,
                "split_ratio": split_ratio,
            }
        )
        print(
            f"[aime-debug] {problem.problem_id}: split={split_ratio} "
            f"ground_truth={problem.ground_truth_answer}"
        )

    summary = _build_summary(problems_json, ground_truth_warnings)
    payload = {
        "metadata": {
            "model": args.model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_config": {
                "samples_per_prompt": args.samples_per_prompt,
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
            "dataset_sources": dataset_sources,
            "chat_template": CHAT_TEMPLATE,
        },
        "problems": problems_json,
        "summary": summary,
    }

    with args.output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(f"[aime-debug] wrote {args.output_path}")
    if args.enable_tensorboard:
        tensorboard_dir = args.tensorboard_dir or (args.output_path.parent / "tensorboard")
        _write_tensorboard(problems_json, summary, tensorboard_dir)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
