# ASU SOL Upstream `verl` GRPO and GDPO

This repo provides a SOL-compliant wrapper around official upstream `verl` GRPO and GDPO, with the working `v0.7.1` source vendored under `external/verl`.

Start with [docs/asu_sol_upstream_verl_grpo.md](/Users/god/Documents/VERL_GRPO/docs/asu_sol_upstream_verl_grpo.md). If you want the shortest possible "fresh SSH session" command order, use [docs/sol_fresh_terminal_checklist.md](/Users/god/Documents/VERL_GRPO/docs/sol_fresh_terminal_checklist.md).

Then use the helper scripts in [scripts/sol](/Users/god/Documents/VERL_GRPO/scripts/sol) and the Slurm entrypoints in [slurm](/Users/god/Documents/VERL_GRPO/slurm).

The checked-in debug validation path is now designed to work with a plain:

```bash
sbatch slurm/grpo_debug_validation.sbatch
```

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

For the separate GSM8K-based saturation probe path, first build the probe parquet files:

```bash
./scripts/sol/prepare_gsm8k_gdpo_saturation_probe.sh
```

Then submit the separate probe wrapper:

```bash
sbatch slurm/gdpo_binary_saturation_probe.sbatch
```

The GDPO paths now also emit per-reward saturation metrics to TensorBoard / JSONL and append raw saturation events to a sidecar JSONL file next to the main metrics log.

The standard 7B wrapper intentionally stays closer to official upstream `verl` and is documented as pending broader SOL compatibility validation.
By default it also logs to `console + tensorboard + file`, and it only adds W&B when `ENABLE_WANDB=1`.

For the vendoring record, upstream base metadata, and future sync guidance, see [docs/repo_state_and_vendoring_plan.md](/Users/god/Documents/VERL_GRPO/docs/repo_state_and_vendoring_plan.md).
