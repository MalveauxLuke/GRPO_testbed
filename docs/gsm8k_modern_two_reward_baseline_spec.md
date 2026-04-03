# GSM8K Modern Two-Reward Baseline Spec

This note explains what the modern GSM8K baseline actually is in concrete terms:

- what the processed dataset looks like
- how the model is prompted
- what `format_reward` and `correct_reward` mean
- what extra fields are carried for auditing

All claims below are tied back to the implementation.

## Source Dataset

The processed baseline is built from the official GSM8K `main` split, but it is built on top of the already-prepared upstream GSM8K parquet created by:

- [prepare_gsm8k.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k.sh#L1)
- upstream preprocess script [gsm8k.py](/Users/god/Documents/VERL_GRPO/external/verl/examples/data_preprocess/gsm8k.py#L1)

The modern dataset transform itself is implemented in:

- [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh#L1)

It reads the upstream parquet rows, recovers the original raw GSM8K `question` and `answer` from `extra_info`, and then rebuilds each example into the structured modern format at [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh#L37).

## Dataset Row Shape

Each processed row is created in [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh#L49) and has this shape:

```json
{
  "data_source": "openai/gsm8k_modern_two_reward",
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "..."},
  "extra_info": {
    "split": "train|test",
    "index": 0,
    "question": "...",
    "answer": "...",
    "source_dataset": "openai/gsm8k",
    "source_subset": "main",
    "baseline_name": "gsm8k_modern_two_reward",
    "alignment_spec_version": "2026-04-03-hybrid-hash"
  }
}
```

The exact field construction happens at:

- [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh#L61)

### Important fields

`data_source`
- fixed to `openai/gsm8k_modern_two_reward`
- defined in [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L33)

`prompt`
- a two-message chat prompt built by [build_prompt](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L89)
- contains one `system` message and one `user` message

`reward_model.ground_truth`
- the final numeric GSM8K answer
- extracted from the original worked solution using [extract_hash_answer](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L96)
- the transform cross-checks that this matches the upstream parquet ground truth at [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh#L52)

`extra_info`
- preserves the original raw question and answer for auditability
- also stores split/index and baseline metadata

## How The Model Is Prompted

The prompt is built by [build_prompt](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L89).

The exact system prompt is defined in [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L34):

```text
You are a mathematical reasoning assistant. Solve the user's problem and respond using exactly this format with no extra text before or after the structured answer:
<reasoning>your step-by-step reasoning</reasoning>
#### final numeric answer
<answer>final numeric answer</answer>
```

So the model sees:

- `system`: the structured-output instruction above
- `user`: the original GSM8K question text

This means the baseline is using a hybrid contract:

- modern structured chat tags for formatting
- an explicit `#### final answer` marker for correctness
- the original GSM8K `#### number` convention for gold-answer extraction

During evaluation, the stored chat prompt is rendered with the tokenizer chat template in:

- [eval_gsm8k_modern.py](/Users/god/Documents/VERL_GRPO/scripts/sol/eval_gsm8k_modern.py#L65)

During training, the same `prompt` field is what verl consumes from the parquet configured in:

- [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L49)

## Reward Criteria

The reward module is:

- [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L15)

It exposes:

- `format_reward`
- `correct_reward`
- `score = format_reward + correct_reward`

The combined reward output is returned by [compute_score](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L155).

### Structured parsing requirement

The exact-format sub-signal depends on successful structured parsing by:

- [parse_structured_response](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L123)

This parser does all of the following:

1. strips common assistant wrapper tokens first via:
   - [_strip_assistant_wrapper](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L111)
2. requires the response to match:
   - one `<reasoning>...</reasoning>` block
   - optionally followed by one `#### ...` line
   - followed by one `<answer>...</answer>` block
   - no extra text before or after the structured response
3. rejects responses where nested or duplicated `<reasoning>` / `<answer>` tags appear inside the parsed blocks:
   - [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L129)
4. normalizes the answer text with:
   - whitespace removal
   - comma removal
   - dollar-sign removal
   - [normalize_numeric_text](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L103)
5. requires the normalized `<answer>` text to be numerically parseable

Strict parsing only controls the exact-format sub-signal. Correctness is scored
separately from the final `#### ...` marker.

### `format_reward`

`format_reward` is defined in:

- [compute_format_reward](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L142)

It is a bounded blend of:

- exact structured-output compliance
- approximate tag-level format credit

So `format_reward = 1` means all of the following were true:

- the response used the exact `<reasoning>...</reasoning><answer>...</answer>` structure
- the answer block was present
- the answer block was numeric-looking
- there was no trailing junk
- there were no duplicated or nested tags
- `format_reward` does not depend on whether the `####` marker is present or numerically correct

Values between `0` and `1` mean the response had some of the expected tags but
did not satisfy the full exact structure. It does **not** mean the numeric
answer was correct. It only means the structured output schema was obeyed fully
or partially.

### `correct_reward`

`correct_reward` is defined in:

- [compute_correct_reward](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L148)

It is binary:

- `1.0` if a valid numeric answer can be extracted from the final `#### ...`
  marker and that numeric value is equivalent to the gold GSM8K answer
- `0.0` otherwise

The gold answer comes from the original GSM8K `answer` field by extracting the final `#### number` at:

- [extract_hash_answer](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L96)

Extraction is intentionally narrow:

- only the final `#### ...` marker is used for correctness
- reasoning text is ignored for correctness
- malformed, empty, non-numeric, or ambiguous `<answer>` contents do not rescue correctness

Normalization is intentionally light:

- trim spaces
- remove commas
- remove `$`
- collapse internal whitespace

That behavior is implemented in:

- [normalize_numeric_text](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L103)

This means examples like these are treated as equivalent:

- `1234`
- `1,234`
- ` $1,234 `
- `72`
- `72.0`
- `0.5`
- `0.50`
- `1/2`
- `7.2e1`

### `score`

The combined `score` is defined in:

- [compute_score](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L155)

It is simply:

```text
score = format_reward + correct_reward
```

Representative values are:

- `0.0`: no useful format signal and no correct numeric answer
- `1.0`: either valid-format-but-wrong-answer, or wrong-format-but-correct-answer with no format credit
- `1.25` / `1.5`: wrong-format-but-correct-answer with partial format credit
- `2.0`: valid exact format and correct answer

### Extra diagnostic fields

`compute_score` also returns:

- `hash_answer`
- `tag_answer`
- `hash_parse_ok`
- `tag_answer_parse_ok`
- `hash_answer_equals_tag_answer`

These are included in [compute_score](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L217) and are useful for audits and debugging, even though the training objective only uses `score`, `correct_reward`, and `format_reward`.

## Training Configuration Hooks

The debug GDPO wrapper points training at this baseline here:

- dataset files: [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L49)
- reward keys: [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L47)
- custom reward module path: [run_gdpo_gsm8k_modern_debug.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_debug.sh#L86)

So the intended GDPO multi-reward setup is explicitly:

- `algorithm.gdpo_reward_keys=["correct_reward","format_reward"]`

## What This Baseline Is Not

This baseline intentionally does **not** include:

- length reward
- numeric extractability as a separate third reward key
- ratio-based partial-credit correctness

Those exclusions are declared in the alignment spec in:

- [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py#L41)

## Bottom Line

This baseline is:

- official GSM8K answers and questions underneath
- repackaged into a modern structured chat prompt
- rewarded with exactly two public channels:
  - a blended strict-plus-approximate `format_reward`
  - an independent binary numeric `correct_reward`

That makes it a clean, minimal modern multi-reward GSM8K setup for saturation measurement.
