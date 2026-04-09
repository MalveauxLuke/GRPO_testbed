# TensorBoard Guide For Current GSM8K 1.5B Runs

This page is for the current GSM8K answer-tag experiment family:

- debug run
- capped 75-step saturation run
- uncapped full run

The main use case is the current 75-step saturation-check run with:

- model: `Qwen/Qwen2.5-1.5B-Instruct`
- reward contract:
  - `correct_reward`
  - `format_reward`

Exact metric glossary and proof harness:

- [gdpo-advantage-verification.md](/Users/god/Documents/VERL_GRPO/docs/reference/gdpo-advantage-verification.md)

## Start TensorBoard

For the current 75-step run:

```bash
cd ~/GRPO_testbed
bash scripts/sol/start_tensorboard_gsm8k_modern_current.sh capped 6006
```

Raw command:

```bash
tensorboard --logdir_spec capped:/scratch/$USER/verl-grpo/tensorboard/asu_sol_upstream_verl_grpo_gdpo_gsm8k_modern_answer_tag_1_5b_instruct_saturation_check --host 127.0.0.1 --port 6006
```

Other scopes:

```bash
bash scripts/sol/start_tensorboard_gsm8k_modern_current.sh debug 6006
bash scripts/sol/start_tensorboard_gsm8k_modern_current.sh full 6006
bash scripts/sol/start_tensorboard_gsm8k_modern_current.sh both 6006
bash scripts/sol/start_tensorboard_gsm8k_modern_current.sh all 6006
```

## Regex For This Experiment

Use this regex to focus on the metrics that matter for the current GSM8K GDPO saturation run:

```regex
^gdpo/(correct_reward|format_reward)/(mean|std|max|min)$|^gdpo_saturation/|^gdpo_advantage/|^response_length(_non_aborted)?/(mean|clip_ratio)$|^response/aborted_ratio$|^actor/grad_norm$|^perf/(time_per_step|throughput)$|^val-core/openai/gsm8k_modern_two_reward/reward/(mean@1|best@4/mean)$|^val-aux/openai/gsm8k_modern_two_reward/(correct_reward|format_reward|answer_parse_ok|strict_format_reward|approx_format_reward)/(mean@1|best@4/mean)$
```

Notes:

- `mean@1` is the cleanest held-out metric for single-sample quality.
- `best@4/mean` is useful when the run is generating 4 candidates per prompt and you want to see “best-of-4” headroom.
- If a `best@4/mean` series is missing, your validation path may not currently be producing 4 samples per prompt. In that case, focus on the `mean@1` series.

## What Each Metric Means

## Training reward metrics

### `gdpo/correct_reward/mean`

What it shows:
- the average binary correctness reward on the training rollouts for that step

What to look for:
- this is the main “is the model getting the math right?” training signal
- if it rises while reward audits still match recomputation, that is real improvement
- if it saturates very early, the correctness channel may be reaching the ceiling

### `gdpo/format_reward/mean`

What it shows:
- the average format reward on training rollouts

What to look for:
- this tracks structured-output compliance
- if `format_reward` trails `correct_reward`, the model is often right before it is perfectly well-formed
- if `format_reward` rises while `correct_reward` stays flat, you are mostly polishing tags rather than improving math

### `gdpo/.../std`

What it shows:
- variation in each reward across the training batch

What to look for:
- if the std collapses toward zero, the reward dimension may be losing useful variation

## GDPO saturation metrics

### `gdpo_saturation/correct_reward/group_fraction`

What it shows:
- fraction of groups whose `correct_reward` has zero within-group variance

What to look for:
- rising values mean correctness is becoming less informative within groups
- high values are a direct warning that GDPO is losing the pre-whiten correctness signal

### `gdpo_saturation/correct_reward/all_zero_fraction`

What it shows:
- fraction of groups where every sample in the group got `0` correctness

What to look for:
- high early values mean the model is still broadly failing on those groups
- if this falls while `all_one_fraction` rises, the model is moving from all-wrong groups toward all-correct groups

### `gdpo_saturation/correct_reward/all_one_fraction`

What it shows:
- fraction of groups where every sample in the group got `1` correctness

What to look for:
- rising values indicate true saturation at the top
- if this gets large quickly, the model may be too strong for a long correctness-learning curve

### `gdpo_saturation/format_reward/...`

What it shows:
- the same saturation story, but for structure/format instead of answer correctness

What to look for:
- if format saturation stays lower than correctness saturation, formatting still has room to learn

### `gdpo_saturation/any_reward_group_fraction`

What it shows:
- fraction of groups saturated in at least one reward dimension

What to look for:
- this is the broadest “how much group-level reward variation is dying?” summary

## GDPO advantage diagnostics

These are the new metrics we added specifically to see the difference between:

- the true per-reward pre-whiten signal
- the summed pre-whiten signal
- the final post-whiten optimizer signal

## Per-reward pre-whiten metrics

### `gdpo_advantage/pre_whiten/correct_reward/mean`
### `gdpo_advantage/pre_whiten/format_reward/mean`

What they show:
- the mean of the weighted per-reward component before final batch whitening

What to look for:
- these often hover near zero because GDPO group-centers the signal
- the more informative summaries are usually `abs_mean` and `std`

### `gdpo_advantage/pre_whiten/<reward>/abs_mean`

What it shows:
- average absolute magnitude of the reward component before final whitening

What to look for:
- this is one of the best “is this reward dimension still doing work?” metrics
- if `correct_reward` saturation rises and `pre_whiten/correct_reward/abs_mean` collapses, correctness is becoming dead before the final whitening step

### `gdpo_advantage/pre_whiten/<reward>/std`

What it shows:
- variation in the pre-whiten component

What to look for:
- falling toward zero means the reward dimension is losing within-batch signal

### `gdpo_advantage/pre_whiten/<reward>/zero_fraction`

What it shows:
- fraction of valid response tokens whose component is exactly zero

What to look for:
- high values mean a large share of tokens are carrying no signal from that reward dimension
- if this gets very high for `correct_reward`, that is consistent with saturation-driven signal collapse

## Summed pre-whiten total

### `gdpo_advantage/pre_whiten_total/abs_mean`
### `gdpo_advantage/pre_whiten_total/std`

What they show:
- the total GDPO signal before final batch whitening, after summing the reward components

What to look for:
- if per-reward correctness fades but the total remains healthy, some other reward dimension may still be carrying learning signal

## Final post-whiten total

### `gdpo_advantage/post_whiten_total/abs_mean`
### `gdpo_advantage/post_whiten_total/std`

What they show:
- the actual final optimizer-facing advantage magnitude after `masked_whiten`

What to look for:
- this is the closest TensorBoard view of what the optimizer is really using
- if `pre_whiten/correct_reward/abs_mean` is near zero but `post_whiten_total/abs_mean` stays clearly nonzero, final whitening is still producing optimizer signal even though the correctness component itself is saturated

That distinction is exactly why these metrics were added.

## Response-shape metrics

### `response_length/mean`

What it shows:
- average response length including aborted samples

What to look for:
- if it pins near the configured max length, generation may be over-running or failing to stop cleanly

### `response_length_non_aborted/mean`

What it shows:
- average response length among non-aborted samples only

What to look for:
- this is usually the cleaner version of response length for actual model behavior

### `response_length/clip_ratio`
### `response_length_non_aborted/clip_ratio`

What they show:
- fraction of samples hitting the response-length cap

What to look for:
- high values are a warning that max-length clipping may be distorting training

### `response/aborted_ratio`

What it shows:
- fraction of samples with zero-length response

What to look for:
- this should stay near zero

## Optimization and system metrics

### `actor/grad_norm`

What it shows:
- size of the actor gradient update

What to look for:
- if it collapses to tiny values while all saturation metrics rise, learning may be stalling
- if it spikes wildly, training may be unstable

### `perf/time_per_step`
### `perf/throughput`

What they show:
- wall-clock speed and token throughput

What to look for:
- these are mostly for runtime regressions, not reward quality

## Validation metrics

### `val-core/openai/gsm8k_modern_two_reward/reward/mean@1`

What it shows:
- the main held-out single-sample reward summary

What to look for:
- this is the cleanest “is the model actually getting better on validation?” metric
- if training `gdpo/correct_reward/mean` rises but this stays flat, training improvements may not be generalizing

### `val-core/openai/gsm8k_modern_two_reward/reward/best@4/mean`

What it shows:
- best-of-4 held-out reward, when available

What to look for:
- useful for measuring headroom and reranking potential

### `val-aux/openai/gsm8k_modern_two_reward/correct_reward/mean@1`

What it shows:
- held-out answer correctness, separated from total reward

What to look for:
- this is the best held-out counterpart to `gdpo/correct_reward/mean`

### `val-aux/openai/gsm8k_modern_two_reward/format_reward/mean@1`

What it shows:
- held-out structure/format quality

What to look for:
- compare it directly with held-out correctness to see whether the model is learning math or just learning tags

### `val-aux/openai/gsm8k_modern_two_reward/answer_parse_ok/mean@1`

What it shows:
- fraction of held-out responses that produced a valid numeric `<answer>` parse

What to look for:
- if this rises while correctness stays flat, parsing is improving faster than actual math
- if both rise, the model is getting cleaner and more correct

### `val-aux/openai/gsm8k_modern_two_reward/strict_format_reward/mean@1`

What it shows:
- held-out strict structured-format success

What to look for:
- if this stays well below `correct_reward`, the model often knows the answer before it masters strict reasoning-tag formatting

### `val-aux/openai/gsm8k_modern_two_reward/approx_format_reward/mean@1`

What it shows:
- held-out partial-format compliance

What to look for:
- useful for separating “some tags are present” from “the full strict format is correct”

## Recommended reading order in TensorBoard

For this experiment, I would usually read the charts in this order:

1. `val-core/.../reward/mean@1`
2. `gdpo/correct_reward/mean`
3. `gdpo/format_reward/mean`
4. `gdpo_saturation/correct_reward/group_fraction`
5. `gdpo_advantage/pre_whiten/correct_reward/abs_mean`
6. `gdpo_advantage/post_whiten_total/abs_mean`
7. `response_length_non_aborted/mean`
8. `response/aborted_ratio`

That ordering gives you:
- generalization
- training reward quality
- saturation
- true pre-whiten signal
- actual optimizer-facing signal
- generation health
