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
  - correctness extracted from the structured answer span
  - separate exact and approximate format shaping
  - reflected in the Hugging Face GRPO cookbook family and the upstream verl GDPO example style

## Chosen Structured Format

The model is instructed to answer using:

```text
<reasoning> ... </reasoning>
<answer> final_numeric_answer </answer>
```

This keeps the repo-native tags-only contract for generated outputs. The source
GSM8K gold answer still comes from the original worked solution’s trailing
`#### number`.

## Rewards

### `format_reward`

A bounded blend of two public precedents:

- exact structured-output compliance
- approximate tag-level format credit

Concretely:

- exact format requires one clean `<reasoning>...</reasoning>` block and one parseable numeric `<answer>...</answer>` block in the right order, with no extra text before or after
- the `<answer>` field must be numeric-looking for full strict format credit
- approximate format gives partial credit when the expected tags appear even if the overall structure is imperfect
- the final `format_reward` is `0.5 * strict_format_reward + 0.5 * approx_format_reward`
- the two are blended back into a single `format_reward` so the GDPO contract stays two-dimensional

### `correct_reward`

Binary coupled numeric correctness:

- parse the predicted answer only from the exact structured `<answer>...</answer>` span
- if strict structured parsing fails, correctness is `0`
- compare numerically against the GSM8K gold answer parsed from the original source `#### number`
- treat numeric equivalents like `72` and `72.0` as correct
- support decimals, negatives, commas, `$`, fractions, and scientific notation
- do not recover correctness from reasoning text, malformed `<answer>` fields, or plain-text fallbacks

## Intentional Simplifications

- Two rewards, not three or four
- Binary numeric equivalence, not ratio-based partial credit
- One public `format_reward`, even though it internally blends strict and approximate format signals
- Correctness is coupled to the structured answer parse
- No response-wide correctness fallback
- No length reward
- No separate third numeric-extraction reward

## Explicit Exclusions

Not part of this baseline:

- generated-output `####` markers
- `numeric_extractability` as a separate reward key
- ratio-based partial-credit correctness
- length-aware rewards
- response-wide correctness fallback
- efficiency-focused reward shaping

Those are valid follow-up experiments, but they would no longer represent the
cleanest minimal modern multi-reward GSM8K baseline.
