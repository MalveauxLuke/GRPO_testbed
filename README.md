# ASU SOL `verl` GRPO and GDPO

This repo keeps the working SOL runtime around vendored upstream `verl`, but the human-facing path is now much smaller:

1. start with [docs/getting-started/sol.md](/Users/god/Documents/VERL_GRPO/docs/getting-started/sol.md)
2. pick a workflow:
   - [docs/workflows/gsm8k.md](/Users/god/Documents/VERL_GRPO/docs/workflows/gsm8k.md)
   - [docs/workflows/math.md](/Users/god/Documents/VERL_GRPO/docs/workflows/math.md)
3. inspect and submit through the config-driven front door:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env --dry-run
scripts/sol/submit_experiment.sh configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env --submit
```

The main reference docs are:
- [docs/getting-started/sol.md](/Users/god/Documents/VERL_GRPO/docs/getting-started/sol.md)
- [docs/workflows/gsm8k.md](/Users/god/Documents/VERL_GRPO/docs/workflows/gsm8k.md)
- [docs/workflows/math.md](/Users/god/Documents/VERL_GRPO/docs/workflows/math.md)
- [docs/reference/runtime.md](/Users/god/Documents/VERL_GRPO/docs/reference/runtime.md)
- [docs/reference/config-knobs.md](/Users/god/Documents/VERL_GRPO/docs/reference/config-knobs.md)
- [docs/reference/troubleshooting.md](/Users/god/Documents/VERL_GRPO/docs/reference/troubleshooting.md)
- [docs/reference/repo-map.md](/Users/god/Documents/VERL_GRPO/docs/reference/repo-map.md)

Human-editable experiment configs live in:
- [configs/experiments/gsm8k](/Users/god/Documents/VERL_GRPO/configs/experiments/gsm8k)
- [configs/experiments/math](/Users/god/Documents/VERL_GRPO/configs/experiments/math)

The old `sbatch slurm/...` entrypoints still work. They remain the compatibility layer under the new front door.
