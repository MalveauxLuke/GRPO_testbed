# Math-Length Reward Audit Follow-Up

This is a reminder to run the manual reward audit and offline recomputation audit later.

## Goal

Verify that:

- logged `correct_reward` matches the intended boxed-answer correctness logic
- logged `length_reward` matches the intended token-threshold logic
- logged `score` matches `correct_reward + length_reward`
- logged `answer_parse_ok` and token-count metadata are consistent with offline recomputation
- there are no hidden reward/logging bugs contaminating the training conclusions

## Files Involved

- Audit CLI:
  - `/Users/god/Documents/VERL_GRPO/scripts/sol/audit_math_length_rewards.py`
- Eval artifact producer:
  - `/Users/god/Documents/VERL_GRPO/scripts/sol/eval_math_length.py`
- Training wrapper with optional rollout dump hook:
  - `/Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh`

## Environment Setup

Run this first on SOL:

```bash
cd /Users/god/Documents/VERL_GRPO
source scripts/sol/common_env.sh
sol_activate_env
```

## Check 1: Manual Reward Audit On Eval Artifacts

Use an existing eval `.per_prompt.jsonl` artifact and inspect a sampled subset by hand.

```bash
python scripts/sol/audit_math_length_rewards.py sample-audit \
  --input-path /path/to/eval.json.per_prompt.jsonl \
  --sample-count 25 \
  --seed 0 \
  --output-jsonl /tmp/math_length_sample_audit.jsonl
```

What to look for:

- the extracted boxed answer looks correct
- `correct_reward` agrees with the final boxed answer vs ground truth
- `length_reward` agrees with `response_length_tokens <= length_limit_tokens`
- `answer_parse_ok` is `1` only when a valid final boxed answer is actually present
- the prompt and output previews look sane

## Check 2: Offline Recompute Audit On Eval Artifacts

Recompute rewards for the full eval artifact and compare them to the logged values.

```bash
python scripts/sol/audit_math_length_rewards.py recompute-summary \
  --input-path /path/to/eval.json.per_prompt.jsonl \
  --summary-output /tmp/math_length_recompute_summary.json \
  --mismatch-output /tmp/math_length_recompute_mismatches.jsonl
```

What success looks like:

- zero mismatches for:
  - `correct_reward`
  - `length_reward`
  - `score`
  - `answer_parse_ok`
  - `response_length_tokens`
  - `length_limit_tokens`
- mismatch sidecar is empty

## Training-Time Rollout Dump Audit

If deeper verification is needed, rerun training with rollout dumping enabled.

Set this before launching the job:

```bash
export MATH_LENGTH_ROLLOUT_DATA_DIR=/scratch/$USER/verl-grpo/rollout_dumps/gdpo_math_length_probe
```

Then launch the run normally, for example:

```bash
sbatch slurm/gdpo_saturation_probe_math.sbatch
```

The wrapper will forward that env var into:

- `trainer.rollout_data_dir=/scratch/$USER/verl-grpo/rollout_dumps/gdpo_math_length_probe`

## Offline Recompute Audit On Rollout Dumps

Audit a whole rollout dump directory:

```bash
python scripts/sol/audit_math_length_rewards.py recompute-summary \
  --input-path /scratch/$USER/verl-grpo/rollout_dumps/gdpo_math_length_probe \
  --input-type rollout \
  --summary-output /tmp/rollout_recompute_summary.json \
  --mismatch-output /tmp/rollout_recompute_mismatches.jsonl
```

Audit a specific dumped step:

```bash
python scripts/sol/audit_math_length_rewards.py recompute-summary \
  --input-path /scratch/$USER/verl-grpo/rollout_dumps/gdpo_math_length_probe \
  --input-type rollout \
  --step 14 \
  --summary-output /tmp/rollout_step14_summary.json \
  --mismatch-output /tmp/rollout_step14_mismatches.jsonl
```

Optional sampled manual audit on rollout dumps:

```bash
python scripts/sol/audit_math_length_rewards.py sample-audit \
  --input-path /scratch/$USER/verl-grpo/rollout_dumps/gdpo_math_length_probe \
  --input-type rollout \
  --step 14 \
  --sample-count 20 \
  --seed 0 \
  --output-jsonl /tmp/rollout_step14_sample_audit.jsonl
```

## Acceptance Criteria

Treat the audit as passed only if all of the following are true:

- sampled manual rows are readable and make sense
- boxed-answer extraction matches what a human would expect
- recomputed rewards exactly match logged rewards
- recomputed token metadata exactly matches logged token metadata
- mismatch outputs are empty, or every mismatch is understood and intentionally explained

## What The Script Supports

The audit CLI supports:

- input types:
  - eval `.per_prompt.jsonl`
  - rollout dump JSONL or rollout dump directories
- subcommands:
  - `sample-audit`
  - `recompute-summary`
- filters:
  - `--step` for rollout artifacts
- outputs:
  - sampled JSONL sidecars
  - summary JSON
  - mismatch JSONL sidecars

## Final Reminder

Do not trust TensorBoard curves alone if something looks surprising.

Use this audit workflow to verify:

- the reward criteria are actually correct
- the logged reward fields are faithful
- the saturation story is real and not a bug from parsing, token counts, or reward plumbing
