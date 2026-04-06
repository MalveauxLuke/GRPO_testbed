# GSM8K Modern Two-Reward Baseline Spec

This note explains what the active GSM8K modern baseline actually is in concrete
terms:

- what the processed dataset looks like
- how the model is prompted
- what `format_reward` and `correct_reward` mean
- what extra fields are carried for auditing

All claims below are tied back to the implementation in:

- [gsm8k_modern_two_reward.py](/Users/god/Documents/VERL_GRPO/external/verl/verl/utils/reward_score/gsm8k_modern_two_reward.py)
- [prepare_gsm8k_modern.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k_modern.sh)
- [verify_gsm8k_modern_baseline.py](/Users/god/Documents/VERL_GRPO/scripts/sol/verify_gsm8k_modern_baseline.py)

## Source Dataset

The processed baseline is built from the official GSM8K `main` split on top of
the already-prepared upstream GSM8K parquet. The modern transform:

- recovers the original raw GSM8K `question` and `answer`
- extracts the gold answer from the source solution’s final `#### number`
- rebuilds each example into the repo-native structured chat format

The generated-output schema is tags-only. The source-side `####` convention is
still used only for gold-answer extraction.

## Dataset Row Shape

Each processed row has this shape:

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
    "alignment_spec_version": "2026-04-04-answer-tag-correctness"
  }
}
```

Important fields:

- `prompt`
  - built from one `system` message and one `user` message
- `reward_model.ground_truth`
  - the numeric GSM8K gold answer extracted from the source raw `answer` field’s final `#### number`
- `extra_info`
  - preserves the raw source question/answer plus split/index and baseline metadata

The prep metadata also records:

- `structured_output_schema`
- `gold_answer_extraction`

so the generated-output contract and source-answer convention are visible in the
output directory itself.

## How The Model Is Prompted

The model is instructed to answer with exactly:

```text
<reasoning>your step-by-step reasoning</reasoning>
<answer>final numeric answer</answer>
```

So the active contract is:

- generated outputs use repo-native `<reasoning>/<answer>` tags only
- source GSM8K gold answers still come from the original source `#### number`

This is the answer-tag correctness baseline used for trainability.

## Reward Criteria

The reward module exposes:

- `format_reward`
- `correct_reward`
- `score = format_reward + correct_reward`

### Structured parsing requirement

The strict parser:

- strips common assistant wrapper tokens first
- requires exactly one `<reasoning>...</reasoning>` block followed by one `<answer>...</answer>` block
- allows only surrounding whitespace outside the tags
- rejects nested `<reasoning>` / `<answer>` tags inside the parsed blocks
- requires the `<answer>` contents to be numerically parseable

Strict parsing is used only for `strict_format_reward`. Correctness uses a
separate answer-tag-only parse.

### `format_reward`

`format_reward` is intentionally non-binary:

```text
format_reward = 0.5 * strict_format_reward + 0.5 * approx_format_reward
```

Where:

- `strict_format_reward`
  - `1.0` only when the exact tags-only structure parses cleanly and `<answer>` is numeric
  - otherwise `0.0`
- `approx_format_reward`
  - uses the cookbook-style tag-count heuristic
  - each required tag contributes `+0.5` if it appears exactly once and `-0.5` otherwise
  - the raw score in `[-2, 2]` is normalized back to `[0, 1]`

That yields representative values like:

- `1.0`
  - exact valid `<reasoning>` + numeric `<answer>`
- `0.5`
  - all four tags appear once, but strict parsing fails
  - examples: wrong order, trailing junk, empty `<answer>`, non-numeric `<answer>`
- `0.25`
  - only one tag pair is present or tag counts are duplicated badly
  - examples: reasoning-only, answer-only, duplicated reasoning tags
- `0.0`
  - plain text with no usable tag scaffold

`format_reward` measures structure only. It does not mean the numeric answer is
correct.

### `correct_reward`

`correct_reward` is binary and uses a single clean answer tag:

- strip assistant wrappers first
- require exactly one `<answer>` open tag and one `</answer>` close tag
- extract only the content inside that one answer span
- reject duplicate or nested answer/reasoning tags inside the answer span
- require the extracted answer to be numerically parseable
- compare that parsed numeric answer against the GSM8K gold answer
- do not require the reasoning block to be valid for correctness

Numeric equivalence is intentionally forgiving for formatting, while remaining
binary:

- `72` and `72.0`
- `0.5` and `0.50`
- `1,234` and `$1,234`
- `1/2` and `0.5`
- `7.2e1` and `72`

There is no response-wide fallback, no reasoning-text rescue, and no separate
extractability reward.

### `score`

The combined score is simply:

```text
score = format_reward + correct_reward
```

Representative totals:

- `0.0`
  - no usable format and no correct answer
- `0.25`
  - partial tag scaffold only
- `0.5`
  - good tag scaffold but failed strict parse and no clean answer-tag correctness
- `1.25`
  - partial format but one clean correct `<answer>` tag
- `1.5`
  - all four tags appear once, strict format failed, but the clean `<answer>` tag is correct
- `1.0`
  - valid format but wrong answer
- `2.0`
  - valid exact format and numerically correct answer

## Extra Diagnostic Fields

`compute_score` also returns:

- `strict_format_reward`
- `approx_format_reward`
- `answer_parse_ok`
- `parsed_answer`
- `expected_answer`

These are useful for audits and rollout inspection, even though training only
uses `score`, `correct_reward`, and `format_reward`.

## What This Baseline Is Not

This baseline intentionally does **not** include:

- generated-output `####` markers
- response-wide correctness fallback
- numeric extractability as a separate reward key
- ratio-based partial-credit correctness
- length reward

## Bottom Line

This baseline is:

- official GSM8K questions and source answers underneath
- repackaged into a tags-only structured chat prompt
- rewarded with exactly two public channels:
  - a soft blended `format_reward`
- a binary `correct_reward` derived from a single clean `<answer>` parse

That makes it a trainable, minimal modern two-reward GSM8K baseline for the
next saturation check.
