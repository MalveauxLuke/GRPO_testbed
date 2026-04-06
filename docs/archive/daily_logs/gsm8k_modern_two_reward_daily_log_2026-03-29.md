# Daily Log: GSM8K Modern Two-Reward Baseline

Date: 2026-03-29

## Summary

Today we finished building and validating a modern 2-reward GSM8K baseline for saturation measurement. The immediate goal was to move from the earlier intentionally extreme saturation probe to a more normal multi-reward reasoning setup that we can trust technically before running a more standard Qwen2.5 training job.

The baseline we prepared is:

- dataset: official GSM8K `main`, repackaged into a structured chat format
- model family target: `Qwen/Qwen2.5-0.5B-Instruct`
- rewards:
  - `correct_reward`
  - `format_reward`

The plan for tomorrow is to use this validated baseline for a more normal training run and track GDPO saturation behavior in a realistic setting.

## What We Built

We created a new modern structured-output GSM8K baseline with exactly two rewards.

Core implementation:

- reward module:
  - [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)
- dataset preparation:
  - [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh)
- debug training wrapper:
  - [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh)
- verifier:
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py)

Supporting docs:

- baseline alignment note:
  - [gsm8k_modern_two_reward_baseline.md](/Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_two_reward_baseline.md)
- baseline spec:
  - [gsm8k_modern_two_reward_baseline_spec.md](/Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_two_reward_baseline_spec.md)
- verification proof:
  - [gsm8k_modern_two_reward_verification_proof.md](/Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_two_reward_verification_proof.md)

## Baseline Definition

The processed dataset uses:

- source data from GSM8K `question` / `answer`
- gold answer extracted from the final `#### number`
- structured prompting with:
  - `<reasoning>...</reasoning><answer>...</answer>`

The rewards are:

- `format_reward`
  - binary strict compliance with the required structured output
- `correct_reward`
  - binary exact normalized numeric match against the GSM8K gold answer

This baseline intentionally does not include:

- length reward
- numeric extractability as a separate reward
- soft format reward
- partial credit

That keeps the setup close to a normal modern multi-reward GSM8K run while staying analytically clean for saturation logging.

## What We Verified

We did two major layers of verification.

### 1. Pre-run verifier

Command run:

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py all \
  --processed-dir "$GSM8K_MODERN_DIR" \
  --base-dir "$GSM8K_DIR" \
  --source-mode preprocessed
```

Observed result:

- `checks_passed = true`
- `dataset_audit.mismatch_count = 0`
- `reward_audit.mismatch_count = 0`
- `reference_audit.mismatch_count = 0`

Important counts:

- total processed rows checked:
  - `8792`
- split sizes matched:
  - train `7473`
  - test `1319`
- reward audit synthetic cases:
  - all standard cases passed with `0` mismatches
  - comma normalization also passed where applicable

This established:

- the processed dataset is a faithful transform of the GSM8K source data
- the reward functions behave exactly as intended
- the implementation still matches the intended modern 2-reward baseline spec

### 2. Runtime artifact verifier

We then ran the debug job with rollout dumping enabled and audited the real runtime artifacts.

Run setup:

```bash
RUN_TAG=$(date +%Y%m%d-%H%M%S)
export GSM8K_MODERN_ROLLOUT_DATA_DIR=/scratch/$USER/verl-grpo/rollout_dumps/gsm8k_modern_${RUN_TAG}
sbatch slurm/gdpo_gsm8k_modern_debug.sbatch
```

Submitted job:

- `49579281`

Artifact audit command:

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py artifact-audit \
  --input-path "$GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  --artifact-type rollout
```

Observed result:

- `total_samples = 80`
- `samples_with_any_mismatch = 0`
- `score` mismatches = `0`
- `correct_reward` mismatches = `0`
- `format_reward` mismatches = `0`

Per-step result:

- steps `1` through `5`
- `16` samples per step
- zero mismatches at every step

This established:

- the training run logged reward values that exactly match offline recomputation
- there is no evidence of a runtime reward-logging mismatch
- the dataset and reward path are not only correct in principle, but correct in the actual debug run

## Credibility Status

As of today, we have high confidence in the authenticity and credibility of this baseline.

What is now technically verified:

- dataset creation correctness
- source-answer extraction correctness
- reward semantics correctness
- reference alignment with the intended modern GSM8K baseline
- runtime reward logging correctness on real rollout samples

What this means:

- if we observe saturation behavior on this baseline, it is much less likely to be explained by preprocessing bugs, reward bugs, or logging bugs

## Tomorrow’s Next Step

Tomorrow’s goal is to use this verified baseline for a more normal Qwen2.5 training run and track saturation in a realistic setting.

The working interpretation is:

- the extreme probe helped expose a clear saturation regime
- the modern GSM8K two-reward baseline now gives us a technically credible environment for measuring how much saturation happens in more ordinary training

Tomorrow’s focus should be:

- run the normal training configuration on this baseline
- monitor saturation metrics during training
- compare the resulting behavior to the earlier extreme probe

## Bottom Line

Today’s work was primarily about preparation and trust-building, not about claiming final scientific conclusions.

We now have:

- a clean modern GSM8K two-reward baseline
- a verifier that checks dataset fidelity, reward behavior, and reference alignment
- runtime evidence that the logged rewards match offline recomputation

That puts us in a strong position to run a normal Qwen2.5 GSM8K training job tomorrow and interpret its saturation measurements with much higher confidence.
