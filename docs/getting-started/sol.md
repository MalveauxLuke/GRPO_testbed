# SOL Getting Started

This is the shortest current path for using this repo on SOL.

## 1. Reopen a session

```bash
ssh <asurite>@sol.asu.edu
cd ~/GRPO_testbed
git pull
source scripts/sol/common_env.sh
```

If `sol.asu.edu` is not resolving from your network, `login.sol.rc.asu.edu` is the practical fallback.

## 2. First-time setup only

Request a light setup allocation:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
```

Then build the env and the current datasets:

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
./scripts/sol/bootstrap_lightwork.sh
./scripts/sol/prepare_gsm8k_modern.sh
./scripts/sol/prepare_deepscaler_math.sh
```

You usually do this once per fresh checkout or scratch reset.

## 3. Inspect before you submit

GSM8K debug:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/debug_answer_tag_1p5b_instruct.env --dry-run
```

GSM8K 75-step saturation run:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env --dry-run
```

Math GDPO debug:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/math/gdpo_debug.env --dry-run
```

Dry-run prints:
- the config you loaded
- the Slurm entrypoint
- the resolved key environment variables
- the exact `verl.trainer.main_ppo` command
- the post-run audit command

## 4. Submit

After the dry-run looks right, switch `--dry-run` to `--submit`:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env --submit
```

The old `sbatch slurm/...` entrypoints still work. The config-driven front door is just the easier human-facing layer.

## 5. Logs and TensorBoard

Check jobs:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName%25,State,ExitCode,Elapsed,NodeList
```

Open the shared TensorBoard tree:

```bash
cd ~/GRPO_testbed
./scripts/sol/start_tensorboard.sh
```

The current GSM8K 1.5B view helper is:

```bash
cd ~/GRPO_testbed
bash scripts/sol/start_tensorboard_gsm8k_modern_current.sh both 6006
```

Primary runtime locations:
- logs: `/scratch/$USER/verl-grpo/logs`
- tensorboard: `/scratch/$USER/verl-grpo/tensorboard`
- metrics jsonl: `/scratch/$USER/verl-grpo/metrics`
- checkpoints: `/scratch/$USER/verl-grpo/checkpoints`

## 6. Rollout audits

GSM8K rollout recompute:

```bash
cd ~/GRPO_testbed
ROLLDIR=$(latest_log=$(ls -t /scratch/$USER/verl-grpo/logs/slurm-*.log | head -n 1); grep -m1 "Rollout data dir:" "$latest_log" | sed 's/.*Rollout data dir: //')
bash scripts/sol/recompute_gsm8k_modern_rollout_rewards.sh "$ROLLDIR"
```

Math rollout recompute:

```bash
cd ~/GRPO_testbed
python scripts/sol/audit_math_length_rewards.py recompute-summary \
  --input-path "$MATH_LENGTH_ROLLOUT_DATA_DIR" \
  --input-type rollout
```
