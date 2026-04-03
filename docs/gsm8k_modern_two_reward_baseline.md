# GSM8K Modern Two-Reward Baseline

This note records the public precedents and intentional simplifications for the
modern GSM8K saturation baseline.

## Goal

Run a normal-looking GSM8K reasoning setup for `Qwen/Qwen2.5-0.5B-Instruct`
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
<reasoning> ... </reasoning><answer> ... </answer>
```

This is intentionally modern and structured, but simpler than archived
`<think>/<answer>` probe-only setups.

## Rewards

### `format_reward`

A bounded blend of two public precedents:

- exact structured-output compliance
- approximate tag-level format credit

Concretely:

- exact format still requires one clean `<reasoning>...</reasoning><answer>...</answer>` response
- approximate format gives partial credit when the expected tags appear even if the overall structure is imperfect
- the two are blended back into a single `format_reward` so the GDPO contract stays two-dimensional

### `correct_reward`

Binary independent numeric correctness:

- first try to extract a number from the `<answer>...</answer>` section
- if that fails, fall back to extracting the last numeric candidate from the response
- compare numerically against the GSM8K gold answer parsed from the original `#### number`
- treat numeric equivalents like `72` and `72.0` as correct

## Intentional Simplifications

- Two rewards, not three or four
- Binary numeric equivalence, not ratio-based partial credit
- One public `format_reward`, even though it internally blends strict and approximate format signals
- Correctness is not format-gated
- No length reward
- No separate third numeric-extraction reward

## Explicit Exclusions

Not part of this baseline:

- `numeric_extractability` as a separate reward key
- ratio-based partial-credit correctness
- length-aware rewards
- efficiency-focused reward shaping

Those are valid follow-up experiments, but they would no longer represent the
cleanest minimal modern multi-reward GSM8K baseline.
