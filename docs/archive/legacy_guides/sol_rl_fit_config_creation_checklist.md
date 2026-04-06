# SOL RL Fit Config Creation Checklist

Use this checklist before creating or modifying any active SOL RL wrapper or sbatch for:

- modern GSM8K GDPO
- DeepScaleR-style math-length GRPO/GDPO

Do not create a new config from scratch. Start from an existing repo path, fill this out, and keep the answers with the wrapper provenance comments or the implementation note.

## 1. Workflow and profile

- [ ] Workflow:
  - `gsm8k_modern`
  - `math_length`
- [ ] Semantic baseline:
  - name the dataset + reward contract that must remain unchanged
- [ ] Runtime profile:
  - `reference_public`
  - `sol_fit_2gpu`
  - `debug_safe`
- [ ] Run classification:
  - `feasibility-smoke`
  - `integration-smoke`
  - `full-run`

## 2. Config lineage

- [ ] Upstream/public anchor:
  - must be a real path or official example
- [ ] Closest prior working in-repo config:
  - must be an existing path
- [ ] Closest prior failed config or caution:
  - cite a row from [sol_rl_fit_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md)
- [ ] Why this is not being created from scratch:
  - one short sentence

## 3. Semantic vs runtime changes

- [ ] Semantic changes:
  - list only if the dataset, prompt, reward logic, or algorithm semantics are intentionally changing
- [ ] Runtime-profile changes:
  - list only the launcher / fit / topology knobs that are intentionally changing
- [ ] SOL compatibility overrides preserved:
  - [ ] env activation and `sol_python`
  - [ ] actor eager attention override
  - [ ] ref eager attention override
  - [ ] post-run audit path declared
  - [ ] rollout dump hook preserved if a runtime audit is expected

## 4. Batch math

- [ ] Total GPUs:
  - `total_gpus = n_gpus_per_node * nnodes = ______`
- [ ] Rollout count:
  - `rollout.n = ______`
- [ ] Actor PPO mini-batch:
  - `ppo_mini_batch_size = ______`
- [ ] Normalized actor mini-batch:
  - `ppo_mini_batch_size * rollout.n / total_gpus = ______`
- [ ] Actor micro-batch per GPU:
  - `______`
- [ ] Actor micro-batch divides the normalized mini-batch:
  - `yes / no`
- [ ] Rollout log-prob micro-batch per GPU:
  - `______`
- [ ] Rollout log-prob micro-batch divides the normalized mini-batch:
  - `yes / no`
- [ ] Ref log-prob micro-batch per GPU:
  - `______`
- [ ] Ref log-prob micro-batch divides the normalized mini-batch:
  - `yes / no`

Do not submit until the normalized mini-batch and all chosen micro-batches are written down.

## 5. Smoke honesty

- [ ] If this is `feasibility-smoke`, only these knobs differ from the full run:
  - duration
  - save/test cadence
  - optional val-before-train
- [ ] If this is `integration-smoke`, list every memory/topology difference from the full run:
  - train batch
  - val batch
  - response length
  - GPU count
  - TP size
  - rollout count
  - GPU memory utilization
  - actor/rollout/ref micro-batches

Do not call a smoke run “realistic” unless it preserves the real memory shape.

## 6. Post-run audit declared before launch

- [ ] GSM8K rollout audit, if this is the GSM8K workflow:

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py artifact-audit \
  --input-path "$GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  --artifact-type rollout
```

- [ ] Math rollout audit, if this is the math workflow:

```bash
python scripts/sol/audit_math_length_rewards.py recompute-summary \
  --input-path "$MATH_LENGTH_ROLLOUT_DATA_DIR" \
  --input-type rollout
```

- [ ] Live log command chosen before launch:

```bash
tail -f /scratch/$USER/verl-grpo/logs/slurm-<job-name>-<jobid>.log
```

## 7. Launch readiness gate

- [ ] I cited one upstream anchor.
- [ ] I cited one prior working local config.
- [ ] I cited one relevant failure row from [sol_rl_fit_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md).
- [ ] I wrote down the normalized batch math.
- [ ] I classified the run honestly.
- [ ] I wrote the post-run audit command before submission.
- [ ] I can explain why a failure would be launcher/fit related rather than dataset/reward related.

If any box above is unchecked, the config is not ready to be copied into a new wrapper or sbatch.
