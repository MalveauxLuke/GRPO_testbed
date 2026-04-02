# ASU SOL Upstream `verl` GRPO and GDPO

This repo provides a SOL-compliant wrapper around official upstream `verl` GRPO and GDPO, with the working `v0.7.1` source vendored under `external/verl`.

Start with [docs/asu_sol_upstream_verl_grpo.md](/Users/god/Documents/VERL_GRPO/docs/asu_sol_upstream_verl_grpo.md). If you want the shared launcher/fit rules across GSM8K and math, use [docs/sol_rl_fit_guide.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_guide.md). If you want the shortest possible "fresh SSH session" command order, use [docs/sol_fresh_terminal_checklist.md](/Users/god/Documents/VERL_GRPO/docs/sol_fresh_terminal_checklist.md).

Then use the helper scripts in [scripts/sol](/Users/god/Documents/VERL_GRPO/scripts/sol) and the Slurm entrypoints in [slurm](/Users/god/Documents/VERL_GRPO/slurm).

The checked-in debug validation path is now designed to work with a plain:

```bash
sbatch slurm/grpo_debug_validation.sbatch
```

For fast preflight checks before you spend time on longer smoke/full jobs, use:

```bash
sbatch slurm/gdpo_gsm8k_modern_fit_debug.sbatch
sbatch slurm/grpo_math_length_debug.sbatch
sbatch slurm/gdpo_math_length_debug.sbatch
```

For GSM8K, the intended stage order is now:

```bash
sbatch slurm/gdpo_gsm8k_modern_fit_debug.sbatch
sbatch slurm/gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch
sbatch slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch
```

That means:

- `fit_debug`: 15-minute preflight to catch immediate init/load failures
- `fit_2gpu_smoke`: 40-minute extended preflight
- `fit_2gpu`: the main run

All checked-in run wrappers now emit:

- console logs in the Slurm output
- TensorBoard event files under `/scratch/$USER/verl-grpo/tensorboard/...`
- structured JSONL metric logs under `/scratch/$USER/verl-grpo/metrics/...`

To browse the TensorBoard logs from SOL, use:

```bash
./scripts/sol/start_tensorboard.sh
```

The repo also includes two GDPO debug baselines on the same vendored tree:

```bash
sbatch slurm/gdpo_debug_upstream.sbatch
sbatch slurm/gdpo_debug_nvlabs_reference.sbatch
```

Those depend on the ToolRL `rlla_4k` dataset staged by:

```bash
./scripts/sol/prepare_rlla_toolrl.sh
```

For the recommended 2-GPU modern GSM8K path on SOL, use:

```bash
sbatch slurm/gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch
sbatch slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch
```

The older "realistic" GSM8K 3B wrapper remains available as the `reference_public` profile for public-example comparison, but it is no longer the default 2-GPU recommendation.

For the primary DeepScaleR-style math reasoning path, first build the boxed-answer math parquet files:

```bash
./scripts/sol/prepare_deepscaler_math.sh
```

By default this uses the public VERL-format DeepScaleR dataset `sungyub/deepscaler-preview-verl` and rewrites it into the local boxed-answer training/eval layout used by the new math wrappers.

Then submit the new debug wrappers:

```bash
sbatch slurm/grpo_math_length_debug.sbatch
sbatch slurm/gdpo_math_length_debug.sbatch
```

The production-tier single-node entrypoints are also available:

```bash
sbatch slurm/grpo_math_length_production.sbatch
sbatch slurm/gdpo_math_length_production.sbatch
```

The new math path uses:

- DeepScaleR-style prompts ending with `\boxed{}`
- binary `correct_reward` from boxed-answer extraction plus `math_verify`
- binary `length_reward` from token count
- the existing GDPO saturation logs, now keyed on `correct_reward` and `length_reward`

You can evaluate a model on the same reward contract with:

```bash
sbatch slurm/math_length_eval.sbatch
```

The old GSM8K `<think>/<answer>` saturation probe remains in-tree temporarily as an archived side path, but it is no longer the primary math workflow.

The standard 7B wrapper intentionally stays closer to official upstream `verl` and is documented as pending broader SOL compatibility validation.
By default it also logs to `console + tensorboard + file`, and it only adds W&B when `ENABLE_WANDB=1`.

For the vendoring record, upstream base metadata, and future sync guidance, see [docs/repo_state_and_vendoring_plan.md](/Users/god/Documents/VERL_GRPO/docs/repo_state_and_vendoring_plan.md).
