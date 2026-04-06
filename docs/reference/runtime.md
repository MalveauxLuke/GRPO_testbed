# Runtime Reference

This page is the active runtime contract for the repo.

## Front door

Normal use is:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh <config> --dry-run
scripts/sol/submit_experiment.sh <config> --submit
```

Config files live under:
- [configs/experiments/gsm8k](/Users/god/Documents/VERL_GRPO/configs/experiments/gsm8k)
- [configs/experiments/math](/Users/god/Documents/VERL_GRPO/configs/experiments/math)

The old `sbatch slurm/...` paths are still supported.

For the actual supported env knobs and edit process, use [config-knobs.md](/Users/god/Documents/VERL_GRPO/docs/reference/config-knobs.md).

## What a config file is allowed to do

A config file is just a bash env fragment.

It should contain:
- human-readable metadata like `EXPERIMENT_TITLE`
- the Slurm entrypoint path
- the dry-run builder path
- optional rollout-dump defaults
- exported env vars that tune the run

It should not contain:
- launch logic
- shell loops
- extra command execution

## Current shared builders

These are the two main behavior-preserving launch builders:
- [run_gdpo_gsm8k_modern_fit_2gpu.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_2gpu.sh)
- [run_math_length_common.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh)

The new front door and the old active sbatches both resolve to these builders.

If you want readable examples first, use:
- [slurm/examples/gsm8k_modern_example.sbatch](/Users/god/Documents/VERL_GRPO/slurm/examples/gsm8k_modern_example.sbatch)
- [slurm/examples/math_length_example.sbatch](/Users/god/Documents/VERL_GRPO/slurm/examples/math_length_example.sbatch)

## Dry-run behavior

`--dry-run` does not launch training. It prints:
- the resolved environment variables
- the run root and tracking paths
- the exact `verl.trainer.main_ppo` command
- the post-run audit command

This is the preferred parity check before editing a config.

## Shared runtime invariants

Keep these unless you are intentionally revalidating the runtime:
- source [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh)
- use `sol_activate_env` and the resolved env python
- keep actor/ref `attn_implementation=eager` on SOL
- keep the validated batch-math checks
- keep rollout-dump hooks for runs you plan to audit
- keep the current reward functions and dataset prep paths fixed when you are only changing runtime shape

## How to create a new experiment safely

1. Copy the nearest config in `configs/experiments/...`.
2. Rename the experiment and project.
3. Change only the env vars you actually intend to change.
4. Run `scripts/sol/submit_experiment.sh <config> --dry-run`.
5. Confirm:
   - dataset path
   - model path
   - reward path
   - batch math
   - save/test cadence
   - post-run audit command
6. Submit only after the dry-run is clean.
