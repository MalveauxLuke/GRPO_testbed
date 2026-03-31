# GSM8K Modern Two-Reward Verification Proof

This note records what was verified for the modern 2-reward GSM8K baseline, which commands were run, and what those commands prove.

## Scope

The baseline under verification is the structured GSM8K setup implemented in:

- [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)
- [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh)
- [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh)
- [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py)

Its reward definition is:

- `format_reward`: strict `<reasoning>...</reasoning><answer>...</answer>` compliance
- `correct_reward`: exact normalized numeric match against the GSM8K final answer

The reward module computes these through:

- GSM8K gold-answer extraction: [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L96)
- structured parsing: [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L123)
- format reward: [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L142)
- correctness reward: [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L148)
- combined `compute_score`: [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L155)

## Verification Commands Actually Run

### 1. Pre-run verifier

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py all \
  --processed-dir "$GSM8K_MODERN_DIR" \
  --base-dir "$GSM8K_DIR" \
  --source-mode preprocessed
```

### 2. Post-run runtime artifact verifier

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py artifact-audit \
  --input-path "$GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  --artifact-type rollout
```

The rollout dump directory existed because the debug wrapper adds `trainer.rollout_data_dir=...` only when `GSM8K_MODERN_ROLLOUT_DATA_DIR` is set, at [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L102).

## What The First Verifier Checked

The `all` subcommand is defined in [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L585). In `main()`, it runs:

- `dataset-audit`
- `reward-audit`
- `reference-audit`
- `artifact-audit` only if `--artifact-path` is provided

That flow is implemented at [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L635).

### Dataset audit

The dataset audit is implemented in [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L115).

It loads:

- the source split rows from `--base-dir` via [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L77)
- the processed rows from `--processed-dir` via [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L61)

For every train and test row, it checks:

- split counts match: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L129)
- processed `data_source` matches the reward module: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L146)
- processed system prompt matches `SYSTEM_PROMPT`: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L148)
- processed user question matches the source question: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L150)
- processed `reward_model.ground_truth` matches the source `####` answer parsed by `extract_hash_answer`: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L140) and [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L152)
- preserved metadata in `extra_info`:
  - question: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L154)
  - answer: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L156)
  - split: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L158)
  - index: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L160)
  - source dataset/subset: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L162)
  - baseline name/version: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L166)

This is a direct audit of the dataset creation logic in [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh#L49), which is the code that builds the processed rows.

Observed result:

- `samples_checked = 8792`
- `mismatch_count = 0`
- split counts:
  - train `7473`
  - test `1319`

This proves the processed dataset is a faithful row-by-row transform of the source GSM8K parquet, not a partial or corrupted derivative.

### Reward audit

The reward audit is implemented in [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L265).

For each processed example, it builds a fixed synthetic battery of completions in [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L204), including:

- valid correct structured answer
- valid wrong structured answer
- missing tags
- malformed tag order
- duplicated tags
- answer outside tags
- trailing junk
- whitespace normalization
- dollar-sign normalization
- comma normalization when the gold answer is a large enough integer

Then for every case it calls the real runtime reward function:

- [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L287)

and compares the returned `score`, `correct_reward`, and `format_reward` against exact expected values:

- [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L293)

Observed result:

- `samples_checked = 8792`
- `mismatch_count = 0`
- case counts:
  - `8792` for all standard cases
  - `928` for comma normalization, which only applies to a subset of answers

This proves the reward function behaves as specified over the full processed dataset, not just on a small hand-picked test set.

### Reference audit

The reference audit is implemented in [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L452).

It verifies that the implemented baseline still matches the intended public modern-GSM8K reference pattern by checking:

- `alignment_spec.json` exists and matches `ALIGNMENT_SPEC` from the reward module:
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L463)
- the documentation note exists and contains the key baseline claims:
  - `two rewards`
  - `no length reward`
  - `<reasoning>`
  - `<answer>`
  - `structured`
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L471)
- the processed dataset sample still uses:
  - the expected system prompt: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L498)
  - the expected `data_source`: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L500)
  - the expected source dataset/subset lineage: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L502)
- the reward module’s declared simplifications exactly match the intended baseline:
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L508)
- the structured schema is still exactly `<reasoning>...</reasoning><answer>...</answer>`:
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L524)
- the reward keys are still exactly `format_reward` and `correct_reward`:
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L527)
- excluded features still include numeric extractability, soft format, length reward, and partial credit:
  - [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L530)

Observed result:

- `checks_passed = true`
- `mismatch_count = 0`

This proves the implementation did not drift away from the intended baseline definition.

### What the first verifier did not check

In the first command, `artifact_audit` was `null` because `--artifact-path` was not passed. That behavior is explicitly coded at [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L649).

So the first verifier proved:

- dataset fidelity
- reward correctness on synthetic cases
- reference alignment

but it did not yet prove:

- that the training job’s logged reward values match offline recomputation on real generated samples

## What The Second Verifier Checked

The second command ran the dedicated `artifact-audit` subcommand:

```bash
python scripts/sol/verify_gsm8k_modern_baseline.py artifact-audit \
  --input-path "$GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  --artifact-type rollout
```

The rollout dump directory existed because:

- the debug wrapper uses `train_batch_size = 4`: [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L51)
- the debug wrapper uses `rollout.n = 4`: [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L81)
- the debug wrapper uses `total_training_steps = 5`: [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L97)
- and it only enables rollout dumping when `GSM8K_MODERN_ROLLOUT_DATA_DIR` is set: [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L102)

### Artifact loading

The verifier first resolves the JSONL files under the rollout dump directory:

- [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L327)

It then interprets each row as a rollout artifact if it has:

- `input`
- `output`
- `gts`

That detection is in [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L319).

For rollout rows, it extracts the canonical fields:

- generated text from `output`
- gold answer from `gts`
- logged `score`
- logged `correct_reward`
- logged `format_reward`
- `step`

That happens at [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L374).

### Offline recomputation

For each dumped rollout sample, the verifier recomputes rewards by calling the real reward module again:

- [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L412)

It compares:

- logged `score`
- logged `correct_reward`
- logged `format_reward`

against the recomputed values field-by-field:

- [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L419)

It aggregates mismatches:

- globally: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L401)
- by training step: [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py#L426)

### Observed runtime result

Observed output:

- `total_samples = 80`
- `samples_with_any_mismatch = 0`
- all field mismatch counts were zero
- each step `1` through `5` had:
  - `total_samples = 16`
  - zero mismatches

Those counts are exactly what the debug wrapper implies:

- batch size `4`
- rollout `n = 4`
- so `16` rollout samples per step
- over `5` steps gives `80` samples total

This proves:

- the runtime training job used the expected reward function
- the logged reward values match offline recomputation from the dumped generated outputs
- there is no evidence of a logging bug or reward-path mismatch in the training run

## What The Two Verifiers Prove Together

Taken together, the two commands establish four separate facts:

1. The processed modern GSM8K dataset is a faithful transform of the source GSM8K data.
2. The reward function behaves exactly as intended on an exhaustive synthetic battery over the whole dataset.
3. The implemented baseline still matches the intended modern 2-reward reference-derived spec.
4. The actual runtime training job logged rewards that exactly match offline recomputation on real generated outputs.

## Remaining Limits

These verifiers prove technical correctness and runtime consistency. They do not prove:

- that the reward design is optimal
- that the model is strong
- that saturation conclusions will generalize across all scales or hyperparameters

What they do prove is that the observed saturation behavior is highly unlikely to be explained by:

- bad dataset preprocessing
- bad ground-truth extraction
- bad reward implementation
- reward-path drift
- broken runtime logging

## Bottom Line

This verification stack is strong enough to treat the GSM8K modern 2-reward debug run as a technically credible experiment.

The pre-run verifier establishes correctness of the dataset and reward system before training.

The post-run artifact verifier establishes that the training-time logged rewards match the real reward function on actual generated outputs.
