# GSM8K Modern Two-Reward Baseline

This compatibility note stays at its original path because audit tooling points
to it by default.

The active runnable workflow is now [docs/workflows/gsm8k.md](/Users/god/Documents/VERL_GRPO/docs/workflows/gsm8k.md).

## Current contract

- reward keys:
  - `correct_reward`
  - `format_reward`
- output format:
  - `<reasoning>...</reasoning>`
  - `<answer>final_numeric_answer</answer>`
- gold answer source:
  - original GSM8K trailing `#### number`

## Current reward meaning

`correct_reward`:
- requires exactly one clean numeric `<answer>...</answer>` span
- does not require strict reasoning structure
- compares numerically against the GSM8K gold answer

`format_reward`:
- blends strict structured compliance with approximate tag credit
- remains a separate shaping dimension

## Historical notes

Archived supporting material now lives under:
- [docs/archive/gsm8k](/Users/god/Documents/VERL_GRPO/docs/archive/gsm8k)
- [docs/archive/daily_logs](/Users/god/Documents/VERL_GRPO/docs/archive/daily_logs)
