# Troubleshooting

This is the active short error catalog.

## Common failures

| Problem | Symptom | Canonical fix |
| --- | --- | --- |
| Wrong env or interpreter | `ModuleNotFoundError: verl`, `No module named 'datasets'` | `source scripts/sol/common_env.sh && sol_activate_env` before running Python or dry-run commands |
| QoS or walltime mismatch | `QOSMaxWallDurationPerJobLimit` | Reuse the current debug/public sbatch shapes instead of inventing a new queue/time pair |
| FlashAttention mismatch on SOL | `GLIBC_2.32 not found`, `flash_attn_2_cuda` import errors | Keep actor and ref on eager attention |
| Batch arithmetic mismatch | normalized mini-batch divisibility assertion | Re-derive `ppo_mini_batch_size * rollout.n / total_gpus` and ensure every micro-batch divides it |
| Load-time worker death | `ActorDiedError`, `SYSTEM_ERROR`, failure during `Loading weights` | Treat it as fit/init pressure first: compare against the current fit-safe wrappers before changing reward code |
| Debug/full drift | smoke run passes but full run still fails | Keep smoke runs honest about which memory-shaping knobs differ from the real run |

## First triage order

1. Check the Slurm log in `/scratch/$USER/verl-grpo/logs`.
2. Re-run the front-door dry-run for the config you submitted.
3. Compare the resolved command against the nearest working config.
4. Only after the run reached real rollout steps should you start debugging reward behavior.

## Detailed historical notes

The longer historical versions are archived here:
- [sol_rl_fit_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md)
- [docs/archive/legacy_guides/sol_rl_fit_error_catalog.md](/Users/god/Documents/VERL_GRPO/docs/archive/legacy_guides/sol_rl_fit_error_catalog.md)
