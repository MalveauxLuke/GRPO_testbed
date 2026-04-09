# GDPO Advantage Metric Verification

This page has two jobs:

1. show how to run a proof-style check on SOL against the actual vendored `verl` code path
2. explain exactly what the relevant TensorBoard metrics mean for the current GSM8K GDPO experiment

The current target experiment is:

- dataset: `openai/gsm8k_modern_two_reward`
- reward keys:
  - `correct_reward`
  - `format_reward`
- model family currently used in the active workflow:
  - `Qwen/Qwen2.5-1.5B-Instruct`

## What this proof harness verifies

The SOL proof harness checks the real code path used in training:

- [compute_gdpo_outcome_advantage](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/core_algos.py)
- [compute_data_metrics](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/metric_utils.py)
- [_compute_gdpo_saturation_metrics](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/ray_trainer.py)
- [_compute_gdpo_advantage_diagnostic_metrics](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/ray_trainer.py)
- [_append_gdpo_saturation_events](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/ray_trainer.py)

It does not rely only on those helper outputs. It also independently derives the expected values from:

- the raw per-reward scalar rewards
- the response mask
- the group ids
- the weighted pre-whiten per-reward components
- the summed pre-whiten total
- the final post-whiten total

It then checks that the emitted metrics and event rows match those independently derived values.

## What scenarios it covers

The default proof harness runs four scenarios:

1. equal-length single-reward saturation
   - proves a saturated group is zero before whitening and stays zero after whitening when response lengths match
2. uneven-length single-reward saturation
   - proves a saturated group can still get nonzero final optimizer signal after batch whitening
3. two-reward disjoint saturation union
   - proves `any_reward_group_fraction` is the union over saturated group ids across reward dimensions
4. weighted two-reward masked case
   - proves pre-whiten per-reward metrics are weighted components and are computed only on valid response tokens

## Run on SOL

Recommended command:

```bash
cd ~/GRPO_testbed
bash scripts/sol/verify_gdpo_advantage_metrics.sh
```

That wrapper:

- activates the SOL env
- runs the proof harness against the current repo checkout
- writes a JSON report under `/scratch/$USER/verl-grpo/outputs/gdpo_advantage_metric_proof/`

If you want the raw Python command:

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
sol_activate_env
python scripts/sol/verify_gdpo_advantage_metrics.py
```

If you want to save the JSON somewhere explicit:

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
sol_activate_env
python scripts/sol/verify_gdpo_advantage_metrics.py \
  --output-json /scratch/$USER/verl-grpo/outputs/gdpo_advantage_metric_proof/manual_run.json
```

## How to read the proof output

The JSON report contains:

- `all_checks_passed`
  - global pass/fail for the whole proof suite
- `verified_metric_families`
  - every TensorBoard metric family that was checked
- `scenarios`
  - one entry per constructed scenario
- `failures`
  - explicit mismatch descriptions if something drifted

The report is meant to answer:

- are the TensorBoard metrics numerically correct?
- are the saturation events numerically correct?
- do the pre-whiten diagnostics really describe the pre-whiten tensors?
- does `post_whiten_total` really describe the final optimizer-facing `advantages` tensor?

## The exact GDPO tensors

For the current code path, GDPO works like this:

1. build one scalar reward per sample for each reward key
2. convert each scalar reward into a GRPO-style normalized component
3. multiply each component by its configured reward weight
4. sum those weighted components into `new_advantage`
5. apply final batch whitening:
   - `advantages = masked_whiten(new_advantage, response_mask) * response_mask`

So there are three distinct things to keep straight:

- per-reward weighted pre-whiten components
- the summed pre-whiten total
- the final post-whiten total

That is why the current TensorBoard metrics are:

- per reward only before whitening
- total before whitening
- total after whitening

There is no true per-reward post-whiten metric in the current algorithm because whitening happens only after the reward components have already been summed.

## TensorBoard regex for this experiment

Use this regex for the current GSM8K answer-tag experiment:

```regex
^gdpo/(correct_reward|format_reward)/(mean|std|max|min)$|^gdpo_saturation/|^gdpo_advantage/|^critic/(score|rewards|advantages|returns)/(mean|max|min)$|^response_length(_non_aborted)?/(mean|max|min|clip_ratio)$|^response/aborted_ratio$|^prompt_length/(mean|max|min|clip_ratio)$|^actor/grad_norm$|^perf/(time_per_step|throughput)$|^val-core/openai/gsm8k_modern_two_reward/reward/(mean@1|best@4/mean)$|^val-aux/openai/gsm8k_modern_two_reward/(correct_reward|format_reward|answer_parse_ok|strict_format_reward|approx_format_reward)/(mean@1|best@4/mean)$
```

For the run-specific start command, see:

- [tensorboard-gsm8k.md](/Users/god/Documents/VERL_GRPO/docs/reference/tensorboard-gsm8k.md)

## Metric glossary

### `gdpo/<reward_name>/mean`

Definition:

- batch mean of the raw scalar per-sample reward in `batch.non_tensor_batch[reward_name]`

Current examples:

- `gdpo/correct_reward/mean`
- `gdpo/format_reward/mean`

What it tells you:

- whether the sampled batch is, on average, getting that reward more often

What it does not tell you:

- whether GDPO still has within-group variation for that reward
- whether the optimizer-facing advantage is still nonzero

### `gdpo/<reward_name>/std`

Definition:

- batch standard deviation of the raw scalar per-sample reward in `batch.non_tensor_batch[reward_name]`

What it tells you:

- whether the batch still has raw reward variation at all

Important distinction:

- this is batch-level variation, not within-group saturation

### `gdpo/<reward_name>/max`
### `gdpo/<reward_name>/min`

Definition:

- batch extrema of the raw scalar reward values

Use:

- quick sanity check that the reward is in the range you expect

## Saturation metrics

These come from group-level analysis of the raw per-sample reward scalars.

### `gdpo_saturation/<reward_name>/group_fraction`

Definition:

- fraction of prompt groups whose reward values have zero within-group standard deviation

Interpretation:

- the reward dimension has no within-group variation for those groups
- before final whitening, that reward dimension contributes zero GRPO-style signal for those groups

### `gdpo_saturation/<reward_name>/all_zero_fraction`

Definition:

- fraction of groups where that reward dimension is both saturated and entirely zero

Interpretation:

- these are groups where every sampled response failed that reward

### `gdpo_saturation/<reward_name>/all_one_fraction`

Definition:

- fraction of groups where that reward dimension is both saturated and entirely one

Interpretation:

- these are groups where every sampled response succeeded on that reward

### `gdpo_saturation/<reward_name>/any`

Definition:

- `1.0` if at least one group was saturated for that reward in the current batch, else `0.0`

Use:

- coarse “did saturation happen at all this step?” flag

### `gdpo_saturation/any_reward_group_fraction`

Definition:

- fraction of groups that are saturated in at least one reward dimension

Important detail:

- this is a union over group ids across all reward dimensions, not a mean of the per-reward fractions

## Advantage diagnostics

These are the most important metrics for understanding what GDPO is actually optimizing.

### `gdpo_advantage/pre_whiten/<reward_name>/mean`

Definition:

- masked mean over valid response tokens of the weighted pre-whiten component for that reward

Formula:

- `w_k * A_k`
- where `A_k` is the GRPO-normalized component for reward `k`

Interpretation:

- because the component is group-centered, this often hovers near zero
- `abs_mean` and `std` are usually more informative

### `gdpo_advantage/pre_whiten/<reward_name>/abs_mean`

Definition:

- masked mean absolute value of the weighted pre-whiten component for that reward

Interpretation:

- best single summary for whether that reward still contributes real pre-whiten signal

If this collapses while `gdpo_saturation/<reward>/group_fraction` rises:

- that reward dimension is becoming dead before final whitening

### `gdpo_advantage/pre_whiten/<reward_name>/std`

Definition:

- masked standard deviation of the weighted pre-whiten component for that reward

Interpretation:

- how much that reward component varies across valid tokens in the batch

### `gdpo_advantage/pre_whiten/<reward_name>/min`
### `gdpo_advantage/pre_whiten/<reward_name>/max`

Definition:

- masked extrema of that weighted pre-whiten component

Use:

- quick sign and range sanity check

### `gdpo_advantage/pre_whiten/<reward_name>/zero_fraction`

Definition:

- fraction of valid response tokens whose weighted pre-whiten component is exactly zero

Important detail:

- this is computed only over masked-in tokens
- padded tokens do not count

Interpretation:

- high values mean much of the batch carries no signal from that reward dimension

## Total pre-whiten metrics

These summarize the summed pre-whiten tensor:

- `new_advantage = Σ_k (w_k * A_k)`

### `gdpo_advantage/pre_whiten_total/mean`
### `gdpo_advantage/pre_whiten_total/abs_mean`
### `gdpo_advantage/pre_whiten_total/std`
### `gdpo_advantage/pre_whiten_total/min`
### `gdpo_advantage/pre_whiten_total/max`
### `gdpo_advantage/pre_whiten_total/zero_fraction`

Interpretation:

- this is the true GDPO signal before final batch whitening

If:

- `pre_whiten/correct_reward/abs_mean` is near zero
- but `pre_whiten_total/abs_mean` is still healthy

then:

- some other reward dimension is still carrying the pre-whiten learning signal

## Final post-whiten metrics

These summarize the final `advantages` tensor after:

- `masked_whiten(new_advantage, response_mask) * response_mask`

### `gdpo_advantage/post_whiten_total/mean`
### `gdpo_advantage/post_whiten_total/abs_mean`
### `gdpo_advantage/post_whiten_total/std`
### `gdpo_advantage/post_whiten_total/min`
### `gdpo_advantage/post_whiten_total/max`
### `gdpo_advantage/post_whiten_total/zero_fraction`

Interpretation:

- this is the closest TensorBoard view of what the optimizer actually uses

Most important reading rule:

- a saturated reward component can have zero pre-whiten signal
- while `post_whiten_total` is still nonzero because final batch whitening re-centers the summed tensor

So:

- saturation does not automatically imply zero final optimizer signal

## Generic batch metrics

These are emitted by [compute_data_metrics](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/metric_utils.py).

### `critic/advantages/mean`
### `critic/advantages/max`
### `critic/advantages/min`

Definition:

- masked summaries over valid response tokens of the final `advantages` tensor

For outcome-only GDPO:

- these describe the same final tensor as `gdpo_advantage/post_whiten_total/*`

Difference from the `gdpo_advantage` namespace:

- `critic/advantages/*` only exposes mean/max/min
- `gdpo_advantage/post_whiten_total/*` also exposes `abs_mean`, `std`, and `zero_fraction`

### `critic/returns/mean`
### `critic/returns/max`
### `critic/returns/min`

Definition:

- masked summaries over valid response tokens of the `returns` tensor

For outcome-only GDPO:

- `returns == advantages`

So in this experiment:

- `critic/returns/*` should match the corresponding `critic/advantages/*` values

### `critic/score/*`
### `critic/rewards/*`

Definition:

- sequence-level summaries of the batch token-level scores and rewards

In the proof harness:

- these are zeroed on purpose because the harness focuses on advantage correctness, not reward-model generation

In real training:

- these summarize the sequence-level token rewards coming out of the rollout/reward path

## Response-shape metrics

### `response_length/mean`
### `response_length/max`
### `response_length/min`

Definition:

- statistics over the per-sample response token counts

### `response_length/clip_ratio`

Definition:

- fraction of samples whose response length equals the configured response-length cap

Interpretation:

- high values can mean generation is repeatedly hitting the cap and being truncated

### `response_length_non_aborted/*`

Definition:

- the same response-length statistics, but computed only on non-aborted samples

### `response/aborted_ratio`

Definition:

- fraction of samples with zero-length responses

### `prompt_length/*`

Definition:

- prompt token-count statistics over the batch

## Validation metrics

Validation metrics come from:

- [process_validation_metrics](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/metric_utils.py)
- [_val_metrics_update](/Users/god/Documents/VERL_GRPO/external/verl/verl/trainer/ppo/ray_trainer.py)
- [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)

### `val-core/openai/gsm8k_modern_two_reward/reward/mean@1`

Definition:

- held-out mean of the reward module’s top-level `score` field using one sample per prompt

For the current reward module:

- `score = correct_reward + format_reward`

Interpretation:

- single-number held-out quality summary for the exact current reward contract

### `val-core/openai/gsm8k_modern_two_reward/reward/best@4/mean`

Definition:

- bootstrap estimate of the mean best-of-4 total reward across held-out prompts

Use:

- measures headroom from sampling multiple candidates

### `val-aux/openai/gsm8k_modern_two_reward/correct_reward/mean@1`

Definition:

- held-out mean of the reward module’s `correct_reward` field

Interpretation:

- held-out answer correctness under the current answer-tag numeric parser

### `val-aux/openai/gsm8k_modern_two_reward/format_reward/mean@1`

Definition:

- held-out mean of the reward module’s blended format reward

### `val-aux/openai/gsm8k_modern_two_reward/strict_format_reward/mean@1`

Definition:

- held-out mean of the exact strict-format subscore

Interpretation:

- how often the output exactly matches the full `<reasoning>...</reasoning><answer>...</answer>` contract

### `val-aux/openai/gsm8k_modern_two_reward/approx_format_reward/mean@1`

Definition:

- held-out mean of the partial-format subscore

Interpretation:

- softer structural compliance signal

### `val-aux/openai/gsm8k_modern_two_reward/answer_parse_ok/mean@1`

Definition:

- held-out mean of the binary “clean answer tag parse succeeded” flag

Interpretation:

- separates parser success from actual correctness

## Recommended reading order in TensorBoard

For this experiment, read the charts in this order:

1. `val-core/.../reward/mean@1`
2. `val-aux/.../correct_reward/mean@1`
3. `gdpo/correct_reward/mean`
4. `gdpo_saturation/correct_reward/*`
5. `gdpo_advantage/pre_whiten/correct_reward/abs_mean`
6. `gdpo_advantage/pre_whiten_total/abs_mean`
7. `gdpo_advantage/post_whiten_total/abs_mean`
8. `response_length_non_aborted/mean`
9. `response_length_non_aborted/clip_ratio`
10. `actor/grad_norm`

That sequence lets you answer:

- is held-out quality improving?
- is the raw reward improving?
- is that reward still informative within groups?
- is the pre-whiten signal dying?
- is the final optimizer signal still alive anyway?
- are response-length pathologies distorting the picture?

## Bottom line

When you are analyzing GDPO saturation for this repo, the safe interpretation is:

- `gdpo_saturation/*` tells you whether a reward dimension has died as a within-group signal
- `gdpo_advantage/pre_whiten/*` tells you whether that reward still contributes real GDPO signal before batch whitening
- `gdpo_advantage/post_whiten_total/*` tells you whether the final optimizer-facing advantage is still nonzero

All three matter. Looking at only one of them is incomplete.
