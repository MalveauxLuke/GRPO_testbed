# GSM8K Modern Realistic GDPO Training Lineage

This run configuration is intended to be a realistic, public-example-like training environment for measuring GDPO saturation on the already-verified modern GSM8K two-reward baseline.

It is now the `reference_public` profile for this workflow, not the recommended default 2-GPU SOL path.

For the recommended constrained-hardware path, use:

- [run_gdpo_gsm8k_modern_fit_2gpu.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_2gpu.sh)
- [gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch)
- [gdpo_gsm8k_modern_fit_2gpu.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch)

Future GSM8K variants must cite:

- one upstream anchor
- one closest prior working GSM8K config
- one prior failure or caution from [sol_rl_fit_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md)

Use [sol_rl_fit_config_creation_checklist.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md) before creating a new GSM8K wrapper or sbatch.

Shared profile guidance lives in:

- [sol_rl_fit_guide.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_guide.md)

## Source lineage

Primary environment anchor:
- Official `verl` Qwen2.5-3B GSM8K GRPO-LoRA script:
  - [run_qwen2_5-3b_gsm8k_grpo_lora.sh](/Users/god/Documents/VERL_GRPO/external/verl/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh)

GDPO behavior anchor:
- Official `verl` GDPO example:
  - [run_qwen1_5b_gdpo.sh](/Users/god/Documents/VERL_GRPO/external/verl/examples/gdpo_trainer/run_qwen1_5b_gdpo.sh)

Local verified baseline reused without semantic changes:
- dataset prep:
  - [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh)
- reward module:
  - [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)
- verifier:
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py)

Closest prior working GSM8K local config:
- [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh)
  - this is the verified smaller-scale GSM8K run path whose dataset, reward behavior, and runtime reward logging were already audited in [gsm8k_modern_two_reward_verification_proof.md](/Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_two_reward_verification_proof.md)

## Inherited Public Defaults

- `Qwen/Qwen2.5-3B-Instruct`
- LoRA training with `lora_rank=64`, `lora_alpha=32`, `target_modules=all-linear`
- `data.train_batch_size=16`
- `data.max_prompt_length=512`
- `data.max_response_length=1024`
- `actor.optim.lr=3e-6`
- `actor.use_kl_loss=True`
- `actor.kl_loss_coef=0.001`
- `rollout.tensor_model_parallel_size=2`
- `rollout.gpu_memory_utilization=0.6`
- `trainer.total_epochs=15`
- `trainer.val_before_train=False`

These values come from the official public Qwen2.5-3B GSM8K GRPO-LoRA script, not from the earlier local debug wrapper.

## SOL-Required Compatibility Overrides

- forces HF actor attention to `eager` on SOL
- forces HF ref attention to `eager` on SOL
- keeps the hardened launcher/env flow from [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh)
- preserves the optional rollout-dump hook so failed or surprising runs can still be audited offline

These are compatibility choices learned from prior SOL failures, not algorithmic changes to the GSM8K baseline.

## Intentional GSM8K/GDPO Deviations

- switches `algorithm.adv_estimator` from `grpo` to `gdpo`
- uses `algorithm.gdpo_reward_keys=["correct_reward","format_reward"]`
- keeps the already-verified GSM8K modern structured-output dataset and reward function
- sets `rollout.n=4` instead of `5` to align with the grouped multi-reward GDPO-style setup
- derives actor / rollout / ref micro-batches from the normalized actor mini-batch instead of hardcoding the official script's `40`, so GPU-count or rollout-count changes do not silently repeat the prior divisibility failure
- keeps the existing GDPO saturation event logging and optional rollout-dump audit hook

## What this script does not change

- no dataset drift
- no reward-semantic changes
- no length reward
- no additional reward shaping
- no removal of the existing verification workflow

## Known Historical Failure Modes For This Path

The GSM8K modern realistic path must be read together with [sol_rl_fit_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md). The failure classes that have already mattered here are:

- wrong env/interpreter or missing package
- Slurm QoS and walltime mismatch
- FlashAttention / `GLIBC_2.32` incompatibility on SOL
- actor mini-batch normalization vs micro-batch divisibility
- smoke/full memory-shape drift
- worker-init memory pressure in the realistic 3B colocated topology
- older SOL launcher regressions around project-root discovery and GPU-visibility normalization

These are considered historical constraints on this path, not hypothetical edge cases.

## Why This Config Differs From Prior Failed Attempts

- It preserves the verified GSM8K dataset/reward stack from the debug wrapper instead of introducing new reward semantics.
- It keeps the SOL-compatible eager-attention override that the smaller debug path already needed.
- It computes the normalized actor mini-batch and derives default micro-batches from it, instead of copying the official public values unchanged.
- It is explicitly documented as the `reference_public` profile so it is no longer confused with the new 2-GPU SOL fit-first path.
- Its smoke wrapper is intended to be a feasibility-smoke entrypoint, meaning it should preserve the realistic memory shape and only shorten duration and cadence.
- It keeps rollout dumping opt-in so surprising runtime behavior can be checked with `artifact-audit` instead of interpreted from curves alone.

## Operational split

Two Slurm entrypoints are provided:

- [gdpo_gsm8k_modern_smoke.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_smoke.sbatch)
  - feasibility-smoke run with short duration limits
  - should preserve the realistic memory shape unless it is explicitly relabeled as integration-only in the checklist

- [gdpo_gsm8k_modern_realistic.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_realistic.sbatch)
  - full realistic run with `15` epochs and `1` day walltime

This split exists so the realistic setup can be verified end-to-end without reusing the old `15` minute debug walltime.
