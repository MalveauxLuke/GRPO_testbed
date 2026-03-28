#!/usr/bin/env python3

import argparse
import json
import os
from statistics import mean
from typing import Any

from datasets import load_dataset
from vllm import LLM, SamplingParams

from verl.utils.reward_score.deepscaler_math_length import compute_score


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
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise ValueError("Expected a VERL-format prompt field.")


def get_ground_truth(example: dict) -> str:
    reward_model = maybe_json(example.get("reward_model"))
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    raise ValueError("Missing reward_model.ground_truth")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a DeepScaleR-style math model with the boxed-answer reward.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--length-limit-tokens", type=int, default=4000)
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
    sample_exceed = []
    sample_lengths = []
    sample_length_reward = []
    max_response_length = 0

    with open(per_prompt_path, "w", encoding="utf-8") as per_prompt_fp:
        for example, request_output in zip(dataset, outputs, strict=True):
            ground_truth = get_ground_truth(example)
            extra_info = maybe_json(example.get("extra_info", {}))
            if not isinstance(extra_info, dict):
                extra_info = {}
            extra_info.setdefault("length_limit_tokens", args.length_limit_tokens)

            prompt_results = []
            for output in request_output.outputs:
                reward_info = compute_score(
                    data_source=str(example.get("data_source", "deepscaler_math_length")),
                    solution_str=output.text,
                    ground_truth=ground_truth,
                    extra_info={
                        **extra_info,
                        "response_length_tokens": len(output.token_ids),
                    },
                )
                correct = float(reward_info["correct_reward"])
                length_reward = float(reward_info["length_reward"])
                exceed = float(length_reward == 0.0)
                response_len = len(output.token_ids)

                prompt_results.append(
                    {
                        "text": output.text,
                        "response_length_tokens": response_len,
                        "correct_reward": correct,
                        "length_reward": length_reward,
                        "exceed": exceed,
                    }
                )
                sample_correctness.append(correct)
                sample_exceed.append(exceed)
                sample_lengths.append(response_len)
                sample_length_reward.append(length_reward)
                max_response_length = max(max_response_length, response_len)

            prompt_correctness = [item["correct_reward"] for item in prompt_results]
            prompt_pass_at_1.append(mean(prompt_correctness))
            prompt_pass_at_n.append(float(any(value > 0 for value in prompt_correctness)))
            per_prompt_fp.write(
                json.dumps(
                    {
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
        "length_limit_tokens": args.length_limit_tokens,
        "pass_at_1": mean(prompt_pass_at_1) if prompt_pass_at_1 else 0.0,
        f"pass_at_{args.samples_per_prompt}_any": mean(prompt_pass_at_n) if prompt_pass_at_n else 0.0,
        "correct_reward_mean": mean(sample_correctness) if sample_correctness else 0.0,
        "exceed": mean(sample_exceed) if sample_exceed else 0.0,
        "length_reward_mean": mean(sample_length_reward) if sample_length_reward else 0.0,
        "response_length_mean": mean(sample_lengths) if sample_lengths else 0.0,
        "response_length_max": max_response_length,
        "per_prompt_path": per_prompt_path,
    }

    with open(args.output_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
