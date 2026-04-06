# GSM8K Workflow

This is the active GSM8K path in the repo.

## Current contract

- dataset prep: [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh)
- reward code: [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)
- reward keys:
  - `correct_reward`
  - `format_reward`
- generation format:
  - `<reasoning>...</reasoning>`
  - `<answer>final_numeric_answer</answer>`

`correct_reward` uses the numeric `<answer>` tag. `format_reward` stays separate.

## Data prep

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
./scripts/sol/prepare_gsm8k_modern.sh
```

## Current runs

Debug run:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/debug_answer_tag_1p5b_instruct.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/debug_answer_tag_1p5b_instruct.env --submit
```

75-step saturation run:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env --submit
```

Full run:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/full_answer_tag_1p5b_instruct.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/full_answer_tag_1p5b_instruct.env --submit
```

The config files are the intended place to edit:
- model path
- experiment/project names
- step caps
- save/test cadence

## Old active entrypoints

These still work if you already know them:

```bash
sbatch slurm/gdpo_gsm8k_modern_fit_debug_hybrid_hash.sbatch
sbatch slurm/gdpo_gsm8k_modern_fit_2gpu_hybrid_hash_saturation_check.sbatch
sbatch slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch
```

## TensorBoard

Current 1.5B GSM8K helper:

```bash
cd ~/GRPO_testbed
bash scripts/sol/start_tensorboard_gsm8k_modern_current.sh both 6006
```

Focused guide for the current capped/full GSM8K runs:
- [tensorboard-gsm8k.md](/Users/god/Documents/VERL_GRPO/docs/reference/tensorboard-gsm8k.md)

If you want the raw command:

```bash
tensorboard --logdir_spec debug:/scratch/$USER/verl-grpo/tensorboard/asu_sol_upstream_verl_grpo_gdpo_gsm8k_modern_fit_debug_answer_tag_1_5b_instruct,capped:/scratch/$USER/verl-grpo/tensorboard/asu_sol_upstream_verl_grpo_gdpo_gsm8k_modern_answer_tag_1_5b_instruct_saturation_check --host 0.0.0.0 --port 6006
```

## Rollout audit

Find the rollout dump:

```bash
ROLLDIR=$(latest_log=$(ls -t /scratch/$USER/verl-grpo/logs/slurm-*.log | head -n 1); grep -m1 "Rollout data dir:" "$latest_log" | sed 's/.*Rollout data dir: //')
echo "$ROLLDIR"
```

Recompute rewards:

```bash
cd ~/GRPO_testbed
bash scripts/sol/recompute_gsm8k_modern_rollout_rewards.sh "$ROLLDIR"
```

Sample rollouts:

```bash
cd ~/GRPO_testbed
bash scripts/sol/sample_gsm8k_modern_rollouts.sh "$ROLLDIR"
```

Only mismatches:

```bash
cd ~/GRPO_testbed
GSM8K_ROLLOUT_ONLY_MISMATCHES=1 bash scripts/sol/sample_gsm8k_modern_rollouts.sh "$ROLLDIR"
```

## Creating a new GSM8K experiment safely

1. Copy one of the files in [configs/experiments/gsm8k](/Users/god/Documents/VERL_GRPO/configs/experiments/gsm8k).
2. Change only the exported variables in that file.
3. Run `scripts/sol/submit_experiment.sh <new-config> --dry-run`.
4. Check the resolved Hydra command and rollout audit command.
5. Submit only after the dry-run matches what you intended.
