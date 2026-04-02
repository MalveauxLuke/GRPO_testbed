# GSM8K Modern Config Creation Checklist

Use this checklist before creating or modifying any modern GSM8K SOL wrapper or sbatch.

Do not create a new GSM8K config from scratch. Start from an existing repo path, fill this out, and keep the answers with the wrapper provenance comments or the associated implementation note.

## 1. Config Lineage

- [ ] Upstream/public anchor:
  - Example: [run_qwen2_5-3b_gsm8k_grpo_lora.sh](/Users/god/Documents/VERL_GRPO/external/verl/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh)
- [ ] Closest prior working in-repo config:
  - Must be an existing path such as [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh)
- [ ] Closest prior failed config or caution:
  - Cite a row from [gsm8k_modern_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_error_catalog.md) and the related wrapper or commit
- [ ] Why this change is not starting from scratch:
  - One short sentence naming the parent config and what is being changed

## 2. Intended Divergences

- [ ] Exact intentional divergences from the parent config:
  - List only the knobs that are intentionally different
- [ ] SOL compatibility choices preserved:
  - [ ] env activation and `sol_python` flow from [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh)
  - [ ] eager attention override on actor
  - [ ] eager attention override on ref
  - [ ] rollout dump hook preserved if runtime audit is expected
- [ ] Dataset/reward authenticity source:
  - Must cite [gsm8k_modern_two_reward_verification_proof.md](/Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_two_reward_verification_proof.md) and [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py)

## 3. Batch Math

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
- [ ] Ref log-prob micro-batch per GPU:
  - `______`
- [ ] Any manual override away from the normalized mini-batch default is justified in writing:
  - `yes / no`

Do not submit until the normalized mini-batch and all chosen micro-batches are explicitly written down.

## 4. Smoke Classification

- [ ] This run is:
  - `feasibility-smoke` or `integration-smoke`
- [ ] If `feasibility-smoke`, confirm only these knobs differ from the full run:
  - duration
  - save/test cadence
  - optional val-before-train
- [ ] If `integration-smoke`, list every memory/topology difference from the full run:
  - train batch
  - val batch
  - response length
  - GPU count
  - TP size
  - rollout count
  - GPU memory utilization
  - actor/rollout/ref micro-batches

Do not call a smoke run “realistic” unless it preserves the real memory shape.

## 5. Pre-Run And Post-Run Audit

- [ ] Pre-run authenticity baseline command:

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py all \
  --processed-dir "$GSM8K_MODERN_DIR" \
  --base-dir "$GSM8K_DIR" \
  --source-mode preprocessed
```

- [ ] Expected rollout dump directory:
  - `/scratch/$USER/verl-grpo/rollout_dumps/<run_name>`
- [ ] Post-run runtime audit command written before launch:

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py artifact-audit \
  --input-path "$GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  --artifact-type rollout
```

- [ ] Live log command chosen before launch:

```bash
tail -f /scratch/$USER/verl-grpo/logs/slurm-<job-name>-<jobid>.log
```

## 6. Launch Readiness Gate

- [ ] I cited one upstream anchor.
- [ ] I cited one prior working GSM8K config.
- [ ] I cited one prior failure row from [gsm8k_modern_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_error_catalog.md).
- [ ] I wrote down the normalized batch math.
- [ ] I classified the smoke honestly.
- [ ] I wrote the post-run audit command before submission.
- [ ] I can explain why a failure would be config/launcher related rather than dataset/reward related.

If any box above is unchecked, the config is not ready to be copied into a new wrapper or sbatch.
