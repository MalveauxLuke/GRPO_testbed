# GSM8K Modern Realistic GDPO Training Lineage

This run configuration is intended to be a realistic, public-example-like training environment for measuring GDPO saturation on the already-verified modern GSM8K two-reward baseline.

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

## What this realistic script preserves from the public 3B Qwen GSM8K setup

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

## What this realistic script changes intentionally

- switches `algorithm.adv_estimator` from `grpo` to `gdpo`
- uses `algorithm.gdpo_reward_keys=["correct_reward","format_reward"]`
- keeps the already-verified GSM8K modern structured-output dataset and reward function
- sets `rollout.n=4` instead of `5` to align with the grouped multi-reward GDPO-style setup
- sets actor / rollout / ref micro-batches to `32` instead of the official script's `40`, because with `rollout.n=4` and `2` GPUs the normalized actor mini-batch is `32`
- forces HF actor/ref attention to `eager` on SOL to avoid the local `flash_attn` binary incompatibility already seen in the smaller debug path
- keeps the existing GDPO saturation event logging and optional rollout-dump audit hook

## What this script does not change

- no dataset drift
- no reward-semantic changes
- no length reward
- no additional reward shaping
- no removal of the existing verification workflow

## Operational split

Two Slurm entrypoints are provided:

- [gdpo_gsm8k_modern_smoke.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_smoke.sbatch)
  - short integration run with `5` training steps and `45` minutes walltime

- [gdpo_gsm8k_modern_realistic.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_realistic.sbatch)
  - full realistic run with `15` epochs and `1` day walltime

This split exists so the realistic setup can be verified end-to-end without reusing the old `15` minute debug walltime.
