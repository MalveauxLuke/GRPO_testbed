# GSM8K Modern Two-Reward Baseline

This note records the public precedents and intentional simplifications for the
modern GSM8K saturation baseline.

## Goal

Run a normal-looking GSM8K reasoning setup for the `Qwen/Qwen2.5-Instruct`
that uses modern structured outputs and exactly two public reward keys:

- `correct_reward`
- `format_reward`

The baseline is for saturation measurement, not reward-design novelty.

## Public Precedents

- Official GSM8K / upstream verl data flow:
  - raw `question` / `answer`
  - final numeric answer recovered from the trailing `#### number`
- Modern structured GSM8K multi-reward family:
  - structured reasoning/answer outputs
  - separate format and answer-quality reward channels
  - reflected in the Hugging Face GRPO cookbook and ReDit-style GSM8K examples

## Chosen Structured Format

The model is instructed to answer using:

```text
<reasoning> ... </reasoning>
#### final_numeric_answer
<answer> final_numeric_answer </answer>
```

This is intentionally modern and structured, but simpler than archived
`<think>/<answer>` probe-only setups.

## Rewards

### `format_reward`

A bounded blend of two public precedents:

- exact structured-output compliance
- approximate tag-level format credit

Concretely:

- exact format requires one clean `<reasoning>...</reasoning>`, one parseable numeric `#### ...` line, and one parseable numeric `<answer>...</answer>` block in the right order
- the `<answer>` field must be numeric-looking for full strict format credit
- approximate format gives partial credit when the expected tags appear even if the overall structure is imperfect
- missing `####` therefore drops the response from full format credit down to partial tag-only credit
- approximate format still ignores `####` and only scores the tag-level scaffold
- the two are blended back into a single `format_reward` so the GDPO contract stays two-dimensional

### `correct_reward`

Binary `####`-only numeric correctness:

- extract the predicted answer from the final `#### ...` marker only
- compare numerically against the GSM8K gold answer parsed from the original `#### number`
- treat numeric equivalents like `72` and `72.0` as correct
- support decimals, negatives, commas, `$`, fractions, and scientific notation
- do not recover correctness from reasoning text or malformed `<answer>` fields

## Intentional Simplifications

- Two rewards, not three or four
- Binary numeric equivalence, not ratio-based partial credit
- One public `format_reward`, even though it internally blends strict and approximate format signals
- Correctness is not format-gated
- Correctness uses `####` only
- No response-wide correctness fallback
- No length reward
- No separate third numeric-extraction reward

## Explicit Exclusions

Not part of this baseline:

- `numeric_extractability` as a separate reward key
- ratio-based partial-credit correctness
- length-aware rewards
- response-wide correctness fallback
- efficiency-focused reward shaping

Those are valid follow-up experiments, but they would no longer represent the
cleanest minimal modern multi-reward GSM8K baseline.
