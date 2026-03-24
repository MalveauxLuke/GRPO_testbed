# ASU SOL Upstream `verl` GRPO

This guide sets up **official upstream `verl` GRPO** on ASU SOL without Docker and without a custom bridge layer.

## Scope

This repo wraps official upstream `verl` in a SOL-safe way:

- Official `verl` source is cloned into `external/verl` and pinned to release `v0.7.1`, published on **March 16, 2026**.
- Setup uses a dedicated **Mamba** environment, not the `base` environment.
- Runtime caches, checkpoints, logs, Ray temp files, and Hugging Face downloads go under `/scratch/$USER/verl-grpo`.
- Real training runs go through `sbatch`.

This repo intentionally does **not** implement a custom runtime bridge, multinode orchestration, or Docker-based workflow.

## Official References

### ASU SOL

- [ASU RC docs home](https://docs.rc.asu.edu/)
- [Sol Open OnDemand portal](https://sol.asu.edu)
- [ASU Python](https://docs.rc.asu.edu/python/)
- [ASU Python Common Issues](https://docs.rc.asu.edu/python-common-issues/)
- [ASU Building Software](https://docs.rc.asu.edu/building-software/)
- [ASU Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos)
- [ASU Transitioning Interactive Jobs to Batch Jobs](https://docs.rc.asu.edu/transition-interactive-to-sbatch/)
- [ASU Scratch File System](https://docs.rc.asu.edu/scratch-file-system/)
- [ASU Resource Limits](https://docs.rc.asu.edu/resource-limits/)
- [ASU Get Help](https://docs.rc.asu.edu/contact-us/)

### Official `verl`

- [Official `verl` docs home](https://verl.readthedocs.io/)
- [Official installation docs](https://verl.readthedocs.io/en/latest/start/install.html)
- [Official quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [Official multinode docs](https://verl.readthedocs.io/en/latest/start/multinode.html)
- [Official GRPO docs](https://verl.readthedocs.io/en/latest/algo/grpo.html)
- [Official Reward Loop docs](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)
- [Official Trainer Interface docs](https://verl.readthedocs.io/en/latest/api/trainer.html)
- [Official `verl` GitHub repo](https://github.com/volcengine/verl)
- [Official GRPO example script](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2-7b.sh)
- [Official examples directory](https://github.com/volcengine/verl/tree/main/examples)

## ASU Policy vs Engineering Choice

> **ASU policy**
> [ASU Building Software](https://docs.rc.asu.edu/building-software/) says Docker is not supported on the cluster, `sudo` is reserved for admins, and `apt`-style system installs are not the right model on SOL. [ASU Python](https://docs.rc.asu.edu/python/) and [ASU Python Common Issues](https://docs.rc.asu.edu/python-common-issues/) say not to install Python packages on login nodes or inside Jupyter and to avoid the `base` environment. [ASU Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos) says `lightwork` is appropriate for creating Mamba environments and other light setup tasks, while [ASU Transitioning Interactive Jobs to Batch Jobs](https://docs.rc.asu.edu/transition-interactive-to-sbatch/) says real runs should move to `sbatch`. [ASU Resource Limits](https://docs.rc.asu.edu/resource-limits/) says home has a 100 GiB quota and `/scratch/$USER` is the right place for active compute data.

> **Engineering choice**
> Official [verl installation docs](https://verl.readthedocs.io/en/latest/start/install.html) recommend Docker for convenience, but they also document a custom Python environment path. This repo chooses that documented custom-environment route because it matches ASU policy. We pin upstream to `v0.7.1`, disable Megatron and SGLang in the official install script because the selected GRPO path uses vLLM + FSDP, use a 1-GPU debug run derived from the official [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) plus official [GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html) knobs, and keep the longer run as a thin wrapper around the official [`run_qwen2-7b.sh`](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2-7b.sh).

## Repo Layout

- [scripts/sol/common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh): shared environment contract for paths, caches, and helper functions
- [scripts/sol/bootstrap_lightwork.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/bootstrap_lightwork.sh): convenience wrapper for env creation, upstream clone, install, and import verification
- [scripts/sol/prepare_gsm8k.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prepare_gsm8k.sh): runs the official upstream GSM8K preprocessing script
- [scripts/sol/prewarm_model.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/prewarm_model.sh): pre-downloads the debug model with the same `transformers.pipeline(...)` pattern shown in the official quickstart
- [scripts/sol/run_grpo_debug_validation.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_grpo_debug_validation.sh): short 1-GPU validation run
- [scripts/sol/run_grpo_standard.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_grpo_standard.sh): standard single-node upstream 7B run
- [scripts/sol/cleanup_reset.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/cleanup_reset.sh): safe cleanup for repo-managed scratch data, env, and optional upstream checkout
- [slurm/grpo_debug_validation.sbatch](/Users/god/Documents/VERL_GRPO/slurm/grpo_debug_validation.sbatch): debug QoS batch job
- [slurm/grpo_standard.sbatch](/Users/god/Documents/VERL_GRPO/slurm/grpo_standard.sbatch): standard 7B batch job

## Phase 1: Lightwork Setup

Start from the repo root on SOL. If you use SSH, the login step is:

```bash
ssh <asurite>@sol.asu.edu
cd /path/to/this/repo
```

Request a short `lightwork` allocation for setup. This follows [ASU Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos).

```bash
salloc -p lightwork -q public -t 02:00:00 -c 8
```

Run the bootstrap helper:

```bash
./scripts/sol/bootstrap_lightwork.sh
```

That wrapper runs:

```bash
./scripts/sol/create_env.sh
./scripts/sol/clone_upstream_verl.sh
./scripts/sol/install_upstream_verl.sh
./scripts/sol/verify_install.sh
```

### What the bootstrap does

- Creates a dedicated Mamba env named `verl-grpo-sol`
- Clones [official upstream `verl`](https://github.com/volcengine/verl) into `external/verl`
- Checks out tag `v0.7.1`
- Runs the official `scripts/install_vllm_sglang_mcore.sh` with `USE_MEGATRON=0 USE_SGLANG=0`
- Installs upstream `verl` in editable mode with `pip install --no-deps -e .`
- Confirms `import verl`, `import ray`, and `import vllm`

## Phase 2: Prepare Data and Debug-Model Cache

Still from a compute allocation, build the official GSM8K parquet files using the upstream script shown in the official [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html):

```bash
./scripts/sol/prepare_gsm8k.sh
```

Prewarm the small debug model so the 15-minute debug job spends its time on GRPO startup rather than downloading artifacts:

```bash
./scripts/sol/prewarm_model.sh
```

By default, that downloads `Qwen/Qwen2.5-0.5B-Instruct`. To prewarm a different model explicitly:

```bash
./scripts/sol/prewarm_model.sh Qwen/Qwen2.5-0.5B-Instruct
```

### Scratch layout

The shared runtime contract in [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh) sends active compute data to scratch:

```text
/scratch/$USER/verl-grpo/
  data/gsm8k/
  hf/
  vllm/
  ray/
  tmp/
  outputs/
  checkpoints/
  logs/
  wandb/
```

This is deliberate because [ASU Resource Limits](https://docs.rc.asu.edu/resource-limits/) says home has a 100 GiB quota and `/scratch/$USER` is the right place for active compute data.

## Phase 3: Submit the Short Debug Validation Job

The short debug job uses `public` + `debug` QoS and a 1-GPU GRPO command based on the official [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) model plus the official [GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html) switches:

- `algorithm.adv_estimator=grpo`
- `actor_rollout_ref.rollout.n=4`
- `actor_rollout_ref.actor.use_kl_loss=True`
- `algorithm.use_kl_in_reward=False`
- `trainer.critic_warmup=0`

Submit it like this:

```bash
cd /path/to/this/repo
export SOL_ACCOUNT=grp_yourgroup
sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/grpo_debug_validation.sbatch
```

If your account does not require `-A`, leave `SOL_ACCOUNT` unset:

```bash
unset SOL_ACCOUNT
sbatch slurm/grpo_debug_validation.sbatch
```

### What this job is for

- Validate that the env activates cleanly in batch mode
- Validate that upstream `verl.trainer.main_ppo` starts successfully on SOL
- Validate that the official GRPO knobs work with scratch-backed paths
- Validate that logs and checkpoints land under scratch

This is a **pipeline validation job**, not a performance run.

## Phase 4: Submit the Standard Upstream 7B Run

The standard path stays very close to the official [GRPO example script](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen2-7b.sh). The wrapper in [run_grpo_standard.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_grpo_standard.sh) calls that script directly and only overrides:

- data paths
- logger backend
- project/experiment names
- checkpoint location

Submit the standard run like this:

```bash
cd /path/to/this/repo
export SOL_ACCOUNT=grp_yourgroup
sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/grpo_standard.sbatch
```

To re-enable Weights & Biases, set `ENABLE_WANDB=1` before submission:

```bash
cd /path/to/this/repo
export SOL_ACCOUNT=grp_yourgroup
ENABLE_WANDB=1 sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/grpo_standard.sbatch
```

If you enable W&B, provide credentials the same way you normally would on SOL. By default this repo uses console logging only.

## Monitoring Jobs and Logs

ASU’s [Transitioning Interactive Jobs to Batch Jobs](https://docs.rc.asu.edu/transition-interactive-to-sbatch/) guide uses standard Slurm monitoring commands. The main ones you need here are:

```bash
squeue -u "$USER"
```

```bash
sacct -j <jobid> --format=JobID,JobName%25,State,ExitCode,Elapsed,MaxRSS
```

The Slurm scripts tee job output into scratch-backed logs:

```bash
tail -f /scratch/$USER/verl-grpo/logs/slurm-verl-grpo-debug-<jobid>.log
```

```bash
tail -f /scratch/$USER/verl-grpo/logs/slurm-verl-grpo-7b-<jobid>.log
```

Checkpoints land under the scratch checkpoint root:

```bash
find /scratch/$USER/verl-grpo/checkpoints -maxdepth 4 -type d | sort
```

The fallback `slurm-*.out` and `slurm-*.err` files in the repo root are just safety copies from Slurm itself; the repo-managed runtime logs are the ones under scratch.

## Cleanup and Reset

The cleanup helper only removes repo-managed paths.

Remove the scratch runtime tree only:

```bash
./scripts/sol/cleanup_reset.sh
```

Also remove the dedicated Mamba env:

```bash
./scripts/sol/cleanup_reset.sh --remove-env
```

Also remove the pinned upstream checkout:

```bash
./scripts/sol/cleanup_reset.sh --remove-upstream
```

Remove everything this repo created, including fallback Slurm logs in the repo root:

```bash
./scripts/sol/cleanup_reset.sh --all
```

This script deliberately refuses to delete:

- `/scratch/$USER` as a whole
- any upstream checkout outside `external/`
- unrelated files in home, scratch, or project storage

## Future Customization: Reward Logic Without a Runtime Bridge

When you want to customize reward logic later, stay inside official upstream extension points documented in [Reward Loop](https://verl.readthedocs.io/en/latest/advance/reward_loop.html) and [Trainer Interface](https://verl.readthedocs.io/en/latest/api/trainer.html):

- Use `reward.custom_reward_function` for a custom function-based reward
- Use `reward.reward_manager.name=<your_manager>` when you need a custom reward manager
- Implement a custom manager by subclassing `RewardManagerBase`
- Keep the runtime entrypoint on `python3 -m verl.trainer.main_ppo`

That keeps the architecture aligned with official upstream `verl` instead of introducing an external compatibility layer.

## What This Repo Does Not Do Yet

- No Ray-on-Slurm multinode orchestration
- No Apptainer image flow
- No custom reward manager implementation

If you later need multinode, start from the official [Multinode Training](https://verl.readthedocs.io/en/latest/start/multinode.html) docs and adapt them carefully to ASU policy rather than adding a parallel runtime stack.
