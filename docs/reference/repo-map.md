# Repo Map

Use this page when you need to know where a piece of the workflow lives.

## Datasets

- GSM8K modern prep: [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh)
- math prep: [prepare_deepscaler_math.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_deepscaler_math.sh)
- scratch data root: `/scratch/$USER/verl-grpo/data`

## Reward logic

- GSM8K reward: [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)
- math reward: [deepscaler_math_length.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/deepscaler_math_length.py)

## Launchers

- shared GSM8K builder: [run_gdpo_gsm8k_modern_fit_2gpu.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_2gpu.sh)
- shared math builder: [run_math_length_common.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh)
- config-driven front door: [submit_experiment.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/submit_experiment.sh)
- human-editable configs: [configs/experiments](/Users/god/Documents/VERL_GRPO/configs/experiments)
- knob reference: [config-knobs.md](/Users/god/Documents/VERL_GRPO/docs/reference/config-knobs.md)

## Slurm entrypoints

Active top-level entrypoints stay in [slurm](/Users/god/Documents/VERL_GRPO/slurm).

Commented examples live in:
- [slurm/examples](/Users/god/Documents/VERL_GRPO/slurm/examples)

## Audits

- GSM8K reward audit: [audit_gsm8k_modern_rewards.py](/Users/god/Documents/VERL_GRPO/scripts/sol/audit_gsm8k_modern_rewards.py)
- GSM8K recompute wrapper: [recompute_gsm8k_modern_rollout_rewards.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/recompute_gsm8k_modern_rollout_rewards.sh)
- math audit: [audit_math_length_rewards.py](/Users/god/Documents/VERL_GRPO/scripts/sol/audit_math_length_rewards.py)

## Logs and artifacts

- slurm logs: `/scratch/$USER/verl-grpo/logs`
- checkpoints: `/scratch/$USER/verl-grpo/checkpoints`
- outputs: `/scratch/$USER/verl-grpo/outputs`
- tensorboard: `/scratch/$USER/verl-grpo/tensorboard`
- metrics jsonl: `/scratch/$USER/verl-grpo/metrics`
- rollout dumps: `/scratch/$USER/verl-grpo/rollout_dumps`
