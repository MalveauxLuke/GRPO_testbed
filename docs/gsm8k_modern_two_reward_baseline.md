# GSM8K Modern Two-Reward Baseline

This note records the public precedents and intentional simplifications for the
modern GSM8K saturation baseline.

## Goal

Run a normal-looking GSM8K reasoning setup for `Qwen/Qwen2.5-0.5B-Instruct`
that uses modern structured outputs and exactly two binary rewards:

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

Binary strict structured-output compliance:

- exactly one `<reasoning>...</reasoning>` block
- followed by exactly one `<answer>...</answer>` block
- no extra trailing text
- non-empty answer block

### `correct_reward`

Binary exact normalized numeric correctness:

- parse the content inside `<answer>...</answer>`
- normalize light numeric formatting only
- compare to the GSM8K gold answer parsed from the original `#### number`

## Intentional Simplifications

- Two rewards, not three or four
- Exact correctness, not approximate matching
- One strict format reward, not strict plus soft format
- No length reward
- No partial-credit answer reward

## Explicit Exclusions

Not part of this baseline:

- `numeric_extractability` as a separate reward
- soft / approximate format reward
- length-aware rewards
- efficiency-focused reward shaping

Those are valid follow-up experiments, but they would no longer represent the
cleanest minimal modern multi-reward GSM8K baseline.
