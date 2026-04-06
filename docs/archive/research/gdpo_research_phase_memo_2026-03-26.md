# GDPO Research Phase Memo

Date: 2026-03-26  
Repo: `/Users/god/Documents/VERL_GRPO`

## Current status

The infrastructure question is mostly settled.

- The SOL-safe GRPO debug path is already working.
- The ToolRL-backed GDPO upstream baseline now runs end to end on SOL.
- The ToolRL-backed GDPO NVLabs-reference baseline also runs end to end on SOL.
- Both GDPO baselines were validated on the real `rlla_4k` dataset, not GSM8K.
- The current debug target is a short 1-GPU, 5-step validation profile, which is appropriate for bring-up and quick ablations.

Internal evidence from the recent SOL runs:

- Upstream baseline completed: job `49443492`
- NVLabs-reference baseline completed: job `49443826`

This means the next step should not be more infrastructure work. It should be a clean, research-legible experimental patch.

---

## What is the most research-standard route from here?

The most research-standard route is:

1. Keep the current vendored `verl` tree as the implementation base.
2. Preserve the two working baselines:
   - `upstream`
   - `nvlabs_reference`
3. Introduce exactly one small, config-gated algorithmic change.
4. Compare that change against both baselines under the same runtime conditions.
5. Once the experiment is stable, repeat it across multiple seeds and summarize with uncertainty, not just single-run point estimates.

Why this is the most standard route:

- It preserves attribution: you know what changed.
- It matches the ablation style used in most ML/RL papers.
- It avoids the common trap of mixing infrastructure changes with algorithm changes.
- It lines up with the RL evaluation guidance from [Agarwal et al. / `rliable`](https://github.com/google-research/rliable) and the accompanying paper ["Deep Reinforcement Learning at the Edge of the Statistical Precipice"](https://arxiv.org/abs/2108.13264), which argues against relying on single point estimates alone.

Practical interpretation:

- Do not swap to the NVLabs fork as the runtime codebase.
- Do not change multiple algorithm components at once.
- Do not scale model size or GPU count before the first ablation is understood.

---

## Recommended first experiment

### Recommendation

The easiest real code change that still looks like a legitimate research ablation is:

**Make the final batch-level whitening step in GDPO optional, then compare `with` vs `without` whitening.**

In the current code, the most natural place is:

- `external/verl/verl/trainer/ppo/core_algos.py`
- specifically the `compute_gdpo_outcome_advantage(...)` path

### Why this is the best first experiment

It is:

- local: the change lives in one function
- cheap: no reward-function rewrite, no dataset rewrite, no rollout rewrite
- conceptually clear: it directly changes how advantages are scaled before PPO updates
- research-legible: advantage normalization / whitening is a known PPO implementation detail and a reasonable ablation target

The current GDPO implementation already does:

1. per-reward-dimension normalization within groups
2. weighted aggregation across reward dimensions
3. final batch-level whitening

That final step is easy to isolate experimentally.

### Why this is still "real research" and not a toy tweak

The GDPO paper itself argues that decoupled normalization matters for multi-reward optimization:

- [GDPO paper](https://arxiv.org/abs/2601.05242)

Separately, PPO literature and implementation notes treat advantage normalization as an important but somewhat under-theorized implementation choice:

- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

That blog notes that some forms of advantage normalization do not strongly affect performance in some settings, which makes it a good target for a lightweight ablation:

- if it does not matter here, that is useful to know
- if it does matter here, then it becomes a meaningful result for tool-use RL with multi-reward optimization

### Why I am not recommending other first changes

#### Not first: reward redesign

Changing `rlla.py` reward semantics is tempting, but it mixes:

- algorithm
- task definition
- reward engineering

That is harder to interpret cleanly.

#### Not first: changing rollout or model scale

That mostly changes resource usage and stability, not GDPO semantics.

#### Not first: large merge from NVLabs

That would make attribution muddy and slow you down.

### Suggested naming for the first ablation

If/when you code it, the clean naming would be something like:

- `algorithm.gdpo_post_aggregate_whiten = true|false`

or

- `algorithm.gdpo_adv_norm_mode = final_whiten | no_final_whiten`

The key point is to keep it config-gated and reversible.

---

## Second-best experimental changes

If you do not want to start with whitening, the next two easiest code-level experiments are:

### 1. Per-reward clipping before aggregation

Idea:

- clip each reward-dimension advantage after decoupled normalization but before summing

Why it is reasonable:

- it is still local to the advantage function
- it targets stability
- it is easy to compare to baseline

Why it is not my first pick:

- clipping thresholds introduce another hyperparameter immediately

### 2. Reward-dimension temperature scaling

Idea:

- scale `accuracy_reward` and `format_reward` by separate temperatures before or during aggregation

Why it is reasonable:

- multi-reward optimization often depends heavily on relative scale

Why it is not my first pick:

- it is partly an algorithm change and partly a reward-design change
- it can quickly turn into tuning rather than a clean ablation

---

## Why the current console output is so hard to read

The ugly output is not just "because Ray is noisy." There is a concrete reason in the vendored code.

The console backend in `verl` currently flattens all numeric metrics into one long line:

- `external/verl/verl/utils/logger/aggregate_logger.py`
- `concat_dict_to_str(...)`

That function essentially emits:

- `step:<n> - metric_a:<value> - metric_b:<value> - metric_c:<value> ...`

As the number of logged metrics grows, the line becomes unreadable.

And in your runs, the problem is amplified because:

- PPO/GDPO already logs many metrics
- GDPO adds per-reward metrics such as `gdpo/accuracy_reward/*` and `gdpo/format_reward/*`
- Ray and vLLM also emit their own warnings and lifecycle messages

So the current "disgusting" log feel is structurally expected if you rely on `trainer.logger=["console"]`.

---

## Best non-code fixes for output readability

These are listed in recommended order for your situation.

### Option A: `console + file`

Use the built-in file logger in vendored `verl`.

Why it is good:

- no external service required
- cluster-friendly
- preserves raw structured metrics in JSONL
- excellent source for later plotting and paper figures

What exists in the current code:

- `external/verl/verl/utils/tracking.py`
- `FileLogger`
- environment knobs:
  - `VERL_FILE_LOGGER_PATH`
  - `VERL_FILE_LOGGER_ROOT`

Recommendation:

- for serious experiments, this should be your minimum logging setup

### Option B: `console + tensorboard`

This is the best cluster-native dashboard option.

Why it is good:

- no public cloud account required
- logs stay local
- very standard in ML research
- easy to browse scalar curves and compare runs

What vendored `verl` already supports:

- `external/verl/verl/utils/tracking.py`
- `_TensorboardAdapter`
- environment knob:
  - `TENSORBOARD_DIR`

What TensorBoard officially supports:

- scalar metrics
- histograms
- text
- images
- profiling

Sources:

- [TensorBoard overview](https://www.tensorflow.org/tensorboard)
- [TensorBoard HParams dashboard](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams)

Recommendation:

- this is the easiest "research-standard and readable" upgrade if you want to stay fully local on SOL

### Option C: `console + wandb`

This is the best interactive UI if network/auth are acceptable.

Why it is good:

- best default visual readability
- easiest run comparison UI
- built-in smoothing
- strong support for tables and custom charts

What vendored `verl` already supports:

- `external/verl/verl/utils/tracking.py`
- `wandb` backend

Useful official W&B capabilities:

- line smoothing with:
  - TWEMA
  - Gaussian
  - running average
  - EMA
- custom chart panels
- tables for structured inspection

Sources:

- [W&B line smoothing](https://docs.wandb.ai/models/app/features/panels/line-plot/smoothing)
- [W&B custom charts](https://docs.wandb.ai/models/app/features/custom-charts)
- [W&B tables](https://docs.wandb.ai/models/tables)

Recommendation:

- best if you want fast human-readable monitoring
- especially strong once you begin comparing multiple experimental runs

### Option D: `console + mlflow`

This is a good middle ground for local or self-hosted experiment tracking.

Why it is good:

- experiment comparison UI
- parameter/metric search
- local server mode is easy

What vendored `verl` already supports:

- `external/verl/verl/utils/tracking.py`
- `mlflow` backend
- env knob:
  - `MLFLOW_TRACKING_URI`

Source:

- [MLflow Tracking UI](https://www.mlflow.org/docs/latest/ml/tracking)

Recommendation:

- sensible if you want structured offline experiment tracking but do not want W&B

---

## My recommendation for readable training tracking

If the goal is to actually understand what is happening during training, the best practical stack is:

### Bring-up / debugging

- `console + tensorboard`

Why:

- local, easy, standard, readable

### Serious ablations

- `console + file + tensorboard`

Why:

- TensorBoard for live reading
- file logger for exact post-hoc analysis and plotting

### Best interactive experience

- `console + file + wandb`

Why:

- W&B is the best UI
- file logger gives raw offline ownership of the data

---

## Which metrics are actually worth watching?

Do not watch every metric. That is part of why the logs feel impossible.

For GDPO bring-up and first ablations, I would focus on these:

### Reward quality

- `gdpo/accuracy_reward/mean`
- `gdpo/format_reward/mean`
- `critic/score/mean`

### Policy behavior

- `actor/entropy`
- `actor/pg_loss`
- `actor/ppo_kl`

### Sequence behavior

- `prompt_length/mean`
- `response_length/mean`
- `response/aborted_ratio`

### Performance / systems

- `perf/time_per_step`
- `perf/throughput`
- `perf/max_memory_allocated_gb`
- `perf/max_memory_reserved_gb`
- `perf/cpu_memory_used_gb`

If I were designing a default dashboard, those would be the first pane.

---

## Plotting methods available now

There are really three layers of plotting available to you.

### 1. Live dashboard plotting

Best for:

- watching runs while they train
- spotting divergence or saturation
- quick side-by-side comparisons

Methods:

- TensorBoard scalar curves
- W&B line plots
- MLflow run charts

### 2. Structured exploratory analysis

Best for:

- comparing runs after they finish
- slicing metrics by condition
- investigating strange behavior

Methods:

- W&B tables and custom charts
- MLflow run comparison
- local JSONL logs loaded into pandas

### 3. Paper-style statistical plots

Best for:

- final figures
- multi-seed comparisons
- uncertainty-aware reporting

Methods:

- pandas + matplotlib
- pandas + seaborn
- `rliable` for RL aggregate plots with confidence intervals

---

## Plotting methods I recommend by maturity level

### Level 1: single-run debug plots

Use:

- TensorBoard or W&B line plots

What to plot:

- step vs reward metrics
- step vs entropy
- step vs throughput
- step vs memory

### Level 2: small ablation comparisons

Use:

- overlaid line plots for each metric
- one line per run
- very light smoothing only

W&B is strongest here because of built-in smoothing and comparisons.

### Level 3: multi-seed research figures

Use:

- mean or IQM across seeds
- confidence intervals
- performance profiles when comparing methods

This is where `rliable` becomes the most research-standard option:

- [rliable GitHub](https://github.com/google-research/rliable)
- it explicitly supports:
  - interquartile mean (IQM)
  - probability of improvement
  - optimality gap
  - performance profiles
  - bootstrap confidence intervals

This is the strongest answer to "how do I make my evaluation look like serious RL research instead of just screenshots of loss curves?"

---

## What I would choose today

If I were making the next move in this repo, I would do the following:

### Recommendation 1

Treat the current state as your frozen baseline checkpoint.

### Recommendation 2

Your first real experimental patch should be:

- **optional final-whitening ablation in GDPO**

### Recommendation 3

For readability, the next operational improvement should be:

- **move away from console-only logging**
- specifically:
  - first choice: `console + tensorboard + file`
  - second choice: `console + wandb + file`

### Recommendation 4

For figures, use a two-stage process:

1. live monitoring in TensorBoard or W&B
2. final analysis from structured logs using pandas/seaborn and, once you have multiple seeds, `rliable`

---

## Concrete next-step recommendation

If time is limited and only one path should be prioritized, I would choose:

1. keep the two GDPO baselines frozen
2. make a single code patch for optional post-aggregation whitening in GDPO
3. stop relying on console-only logs for interpretation
4. begin logging to TensorBoard or W&B plus the local file logger
5. once the ablation runs, compare:
   - reward metrics
   - entropy
   - throughput
   - memory
   - run-to-run variability

That is the smallest next move that is:

- actually a code change
- scientifically legible
- easy to explain
- and likely publishable as part of a broader algorithm exploration if it turns into something interesting

---

## Sources

### External research / tooling

- GDPO paper: [Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization](https://arxiv.org/abs/2601.05242)
- ToolRL repository: [qiancheng0/ToolRL](https://github.com/qiancheng0/ToolRL)
- PPO implementation details: [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
- RL evaluation library: [`rliable`](https://github.com/google-research/rliable)
- RL evaluation paper: [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)
- TensorBoard overview: [TensorBoard](https://www.tensorflow.org/tensorboard)
- TensorBoard HParams dashboard: [HParams](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams)
- W&B smoothing docs: [Smooth line plots](https://docs.wandb.ai/models/app/features/panels/line-plot/smoothing)
- W&B custom charts: [Custom charts overview](https://docs.wandb.ai/models/app/features/custom-charts)
- W&B tables: [Tables overview](https://docs.wandb.ai/models/tables)
- MLflow tracking UI: [MLflow Tracking](https://www.mlflow.org/docs/latest/ml/tracking)

### Internal repo references

- `external/verl/verl/trainer/ppo/core_algos.py`
- `external/verl/verl/utils/logger/aggregate_logger.py`
- `external/verl/verl/utils/tracking.py`
- `scripts/sol/run_gdpo_debug_common.sh`
- `slurm/gdpo_debug_upstream.sbatch`
- `slurm/gdpo_debug_nvlabs_reference.sbatch`
