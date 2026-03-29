#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from statistics import mean
from typing import Any

from datasets import load_dataset
from vllm import LLM, SamplingParams

from verl.utils.reward_score.gsm8k_modern_two_reward import compute_score


def maybe_json(value: Any):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(stripped)
    return value


def get_messages(example: dict) -> list[dict[str, str]]:
    prompt = maybe_json(example.get("prompt"))
    if isinstance(prompt, list) and prompt:
        return [{"role": str(item["role"]), "content": str(item["content"])} for item in prompt]
    raise ValueError("Expected a VERL-format prompt field.")


def get_ground_truth(example: dict) -> str:
    reward_model = maybe_json(example.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    raise ValueError("Missing reward_model.ground_truth")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the modern 2-reward GSM8K baseline.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    per_prompt_path = f"{args.output_path}.per_prompt.jsonl"

    dataset = load_dataset("parquet", data_files={"eval": args.dataset_path})["eval"]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    rendered_prompts = [
        tokenizer.apply_chat_template(get_messages(example), tokenize=False, add_generation_prompt=True) for example in dataset
    ]
    sampling_params = SamplingParams(
        n=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(rendered_prompts, sampling_params=sampling_params)

    prompt_pass_at_1 = []
    prompt_pass_at_n = []
    sample_correctness = []
    sample_format = []

    with open(per_prompt_path, "w", encoding="utf-8") as per_prompt_fp:
        for rendered_prompt, example, request_output in zip(rendered_prompts, dataset, outputs, strict=True):
            ground_truth = get_ground_truth(example)
            extra_info = maybe_json(example.get("extra_info", {}))
            if not isinstance(extra_info, dict):
                extra_info = {}

            prompt_results = []
            for output in request_output.outputs:
                reward_info = compute_score(
                    data_source=str(example.get("data_source", "openai/gsm8k_modern_two_reward")),
                    solution_str=output.text,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                correct_reward = float(reward_info["correct_reward"])
                format_reward = float(reward_info["format_reward"])

                prompt_results.append(
                    {
                        "text": output.text,
                        "score": float(reward_info["score"]),
                        "correct_reward": correct_reward,
                        "format_reward": format_reward,
                        "answer_parse_ok": float(reward_info["answer_parse_ok"]),
                        "parsed_answer": str(reward_info["parsed_answer"]),
                        "expected_answer": str(reward_info["expected_answer"]),
                    }
                )
                sample_correctness.append(correct_reward)
                sample_format.append(format_reward)

            prompt_correctness = [item["correct_reward"] for item in prompt_results]
            prompt_pass_at_1.append(mean(prompt_correctness))
            prompt_pass_at_n.append(float(any(value > 0 for value in prompt_correctness)))
            per_prompt_fp.write(
                json.dumps(
                    {
                        "prompt": rendered_prompt,
                        "data_source": str(example.get("data_source", "openai/gsm8k_modern_two_reward")),
                        "ground_truth": ground_truth,
                        "extra_info": extra_info,
                        "samples": prompt_results,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "num_prompts": len(dataset),
        "samples_per_prompt": args.samples_per_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "pass_at_1": mean(prompt_pass_at_1) if prompt_pass_at_1 else 0.0,
        f"pass_at_{args.samples_per_prompt}_any": mean(prompt_pass_at_n) if prompt_pass_at_n else 0.0,
        "correct_reward_mean": mean(sample_correctness) if sample_correctness else 0.0,
        "format_reward_mean": mean(sample_format) if sample_format else 0.0,
        "per_prompt_path": per_prompt_path,
    }

    with open(args.output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
