# Math Workflow

This is the active DeepScaleR-style math path in the repo.

## Current contract

- dataset prep: [prepare_deepscaler_math.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_deepscaler_math.sh)
- reward code: [deepscaler_math_length.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/deepscaler_math_length.py)
- reward keys:
  - `correct_reward`
  - `length_reward`

## Data prep

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
./scripts/sol/prepare_deepscaler_math.sh
```

## Current runs

GRPO debug:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/math/grpo_debug.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/math/grpo_debug.env --submit
```

GDPO debug:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/math/gdpo_debug.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/math/gdpo_debug.env --submit
```

GRPO production:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/math/grpo_production.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/math/grpo_production.env --submit
```

GDPO production:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/math/gdpo_production.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/math/gdpo_production.env --submit
```

## Old active entrypoints

These still work:

```bash
sbatch slurm/grpo_math_length_debug.sbatch
sbatch slurm/gdpo_math_length_debug.sbatch
sbatch slurm/grpo_math_length_production.sbatch
sbatch slurm/gdpo_math_length_production.sbatch
```

## Rollout audit

```bash
cd ~/GRPO_testbed
python scripts/sol/audit_math_length_rewards.py recompute-summary \
  --input-path "$MATH_LENGTH_ROLLOUT_DATA_DIR" \
  --input-type rollout
```

## Creating a new math experiment safely

1. Copy one of the files in [configs/experiments/math](/Users/god/Documents/VERL_GRPO/configs/experiments/math).
2. Edit only the exported variables.
3. Run `scripts/sol/submit_experiment.sh <new-config> --dry-run`.
4. Check the batch math and resolved trainer command before you submit.
