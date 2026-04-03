# SOL RL Fit Guide

This guide is the shared runtime-fit contract for the active SOL RL workflows in this repo:

- modern GSM8K GDPO
- DeepScaleR-style math-length GRPO/GDPO

It exists to satisfy two constraints at the same time:

- stay close to official `verl` patterns instead of inventing local folklore
- preserve the SOL-hardening learned from real failures in this repo

Use this guide together with:

- [sol_rl_fit_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md)
- [sol_rl_fit_config_creation_checklist.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md)

## What stays semantic vs what changes by profile

Semantic baselines stay fixed within a workflow:

- modern GSM8K:
  - processed dataset under [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh)
  - reward contract in [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)
- math-length:
  - processed dataset under [prepare_deepscaler_math.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_deepscaler_math.sh)
  - reward contract in [deepscaler_math_length.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/deepscaler_math_length.py)

Runtime profiles are allowed to change only fit and launcher behavior:

- `reference_public`
  - closest to the public high-throughput `verl` examples
- `sol_fit_2gpu`
  - recommended constrained-hardware profile for SOL
- `debug_safe`
  - reduced-risk debug profile used to validate wiring on SOL

## Shared SOL invariants

These are not optional per-wrapper decisions anymore:

- use the activated env interpreter through [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh) and `sol_python`
- force actor and ref HF attention to `eager` on SOL
- declare a runtime profile and a run classification in every active wrapper
- validate normalized actor mini-batch math before launch
- declare the post-run audit command in the wrapper/docs before submission
- preserve rollout-dump hooks for runs that need runtime auditing

For memory-constrained fit profiles, also keep:

- rollout eager mode
- low-risk rollout worker settings like `skip_tokenizer_init` and `agent.num_workers=1` when the workflow already proved they matter on SOL

## Recommended wrappers

For 2-GPU SOL use:

- GSM8K recommended path:
  - [run_gdpo_gsm8k_modern_fit_2gpu.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_2gpu.sh)
  - [gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch)
  - [gdpo_gsm8k_modern_fit_2gpu.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch)

For fast first-stage GSM8K preflight on the debug queue:

- [run_gdpo_gsm8k_modern_fit_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh)
- [gdpo_gsm8k_modern_fit_debug.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_debug.sbatch)

This preserves the real `sol_fit_2gpu` memory shape and is now the short first-stage `feasibility-smoke` used to catch immediate init/load failures inside the 15-minute debug window.

For the longer second-stage GSM8K preflight and official pre-full-run gate:

- [gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu_smoke.sbatch)

This is the 40-minute `integration-smoke` gate before the main run at [gdpo_gsm8k_modern_fit_2gpu.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch). It intentionally bakes in the proven SOL-safe compatibility shape for the current Qwen2.5-3B fit path: `use_shm=False`, `gpu_memory_utilization=0.35`, `max_num_seqs=64`, `skip_tokenizer_init=True`, and `val_max_samples=64`.

If the goal is saturation measurement rather than a long full run, use the capped 75-step check at [gdpo_gsm8k_modern_fit_2gpu_saturation_check.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu_saturation_check.sbatch). It keeps the same proven SOL-safe compatibility shape, sets `test_freq=10`, `save_freq=25`, `val_max_samples=64`, and requests 6 hours of walltime.

Reference-public GSM8K path:

- [run_gdpo_gsm8k_modern_realistic.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_realistic.sh)
- [gdpo_gsm8k_modern_smoke.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_smoke.sbatch)
- [gdpo_gsm8k_modern_realistic.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_realistic.sbatch)

Math wrappers stay on the shared common path:

- [run_math_length_common.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh)
- [grpo_math_length_debug.sbatch](/Users/god/Documents/VERL_GRPO/slurm/grpo_math_length_debug.sbatch)
- [gdpo_math_length_debug.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_math_length_debug.sbatch)
- [grpo_math_length_production.sbatch](/Users/god/Documents/VERL_GRPO/slurm/grpo_math_length_production.sbatch)
- [gdpo_math_length_production.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_math_length_production.sbatch)

## Post-run audits

GSM8K rollout audit:

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py artifact-audit \
  --input-path "$GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  --artifact-type rollout
```

Math rollout audit:

```bash
python scripts/sol/audit_math_length_rewards.py recompute-summary \
  --input-path "$MATH_LENGTH_ROLLOUT_DATA_DIR" \
  --input-type rollout
```

## If the 2-GPU fit profile still fails

Do not blindly keep changing colocated GSM8K knobs.

The next upstream-aligned move is:

- split placement / disaggregated resources, following the pattern already present in the vendored `verl` tree at [grpo_3b_gsm8k_fsdp2_2_6.sh](/Users/god/Documents/VERL_GRPO/external/verl/verl/experimental/one_step_off_policy/shell/grpo_3b_gsm8k_fsdp2_2_6.sh)
