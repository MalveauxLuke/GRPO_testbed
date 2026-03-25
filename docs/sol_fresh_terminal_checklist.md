# SOL Fresh Terminal Checklist

This is the practical checklist to use after opening a **new terminal** and SSHing back into SOL.

Use this file for the real command order. The longer background guide is [docs/asu_sol_upstream_verl_grpo.md](/Users/god/Documents/VERL_GRPO/docs/asu_sol_upstream_verl_grpo.md).

## If You Closed Your Terminal

- If you had submitted a job with `sbatch`, that job **keeps running** after your laptop terminal closes.
- If you were inside an interactive `salloc` shell, that interactive shell is gone. Just request a new `salloc` if you still need one.

## 1. SSH Back Into SOL

ASU's current SSH client page shows:

```bash
ssh <asurite>@sol.asu.edu
```

During live bring-up, `login.sol.rc.asu.edu` also worked when `sol.asu.edu` had name-resolution trouble:

```bash
ssh <asurite>@login.sol.rc.asu.edu
```

If you are off campus, connect through the ASU VPN first.

Replace `<asurite>` with your ASURITE username.

## 2. Go To The Repo

If you cloned the repo in your home directory with the current repo name:

```bash
cd ~/GRPO_testbed
```

If you cloned it elsewhere, use that path instead.

## 3. Sync To The Latest Repo State

```bash
git pull
```

This matters because the repo now includes the Slurm path fix, the CUDA/ROCR visibility cleanup, the env-Python fix, the vendored upstream source, and the SOL-compatible debug defaults for both GRPO and GDPO.

## 4. Load The Repo Shell Helpers

```bash
source scripts/sol/common_env.sh
```

What this does:

- restores the shared scratch/cache paths for this shell
- defines helper functions like `sol_activate_env`
- does **not** by itself activate the Mamba env

If you need the Python env interactively, then run:

```bash
sol_activate_env
```

### VS Code Remote Terminals

This workspace now includes a repo-local VS Code setting in [.vscode/settings.json](/Users/god/Documents/VERL_GRPO/.vscode/settings.json) that makes new Linux terminals open as login `bash`.

That matters because [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh) is a Bash-oriented helper file. In normal daily VS Code use on SOL, opening a fresh integrated terminal in this workspace should therefore let you run:

```bash
source scripts/sol/common_env.sh
```

without first doing `exec bash -l`.

## 5. Decide Which Case You Are In

### Case A: First-Time Setup Or Missing Env

Request a light setup allocation:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
```

Once the allocation starts and you land on a compute node:

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
./scripts/sol/bootstrap_lightwork.sh
./scripts/sol/prepare_gsm8k.sh
./scripts/sol/prepare_rlla_toolrl.sh
./scripts/sol/prewarm_model.sh
```

You do **not** run bootstrap every session. You only rerun it if the env is missing, the vendored `external/verl` source is missing from your checkout, or you intentionally want to reinstall.

### Case B: Normal Later Session, Just Submit A Debug Job

From the repo root:

```bash
source scripts/sol/common_env.sh
sbatch slurm/grpo_debug_validation.sbatch
```

Or for GDPO baselines on the ToolRL `rlla_4k` dataset:

```bash
source scripts/sol/common_env.sh
sbatch slurm/gdpo_debug_upstream.sbatch
```

```bash
source scripts/sol/common_env.sh
sbatch slurm/gdpo_debug_nvlabs_reference.sbatch
```

Why this simple command now works:

- the checked-in debug wrapper now contains the validated SOL-compatible fallback profile
- that profile includes the non-FlashAttention path, the tokenizer-skip workaround, lower-memory rollout settings, and a 5-step cap so the debug job can finish inside the debug QoS window
- the GDPO wrappers use the same SOL-safe runtime profile and differ only by baseline mode (`upstream` vs `nvlabs_reference`)

If Slurm ever tells you an account is required, submit with `-A`:

```bash
sbatch -A grp_yourgroup slurm/grpo_debug_validation.sbatch
```

### Case C: Check Whether A Job Is Still Running

Show your current jobs:

```bash
squeue -u "$USER"
```

Show one job:

```bash
squeue -j <jobid>
```

If a job has already finished, inspect final status:

```bash
sacct -j <jobid> --format=JobID,JobName%25,State,ExitCode,Elapsed,NodeList
```

### Case D: Watch The Debug Log

Inspect the latest chunk once:

```bash
tail -n 120 /scratch/$USER/verl-grpo/logs/slurm-verl-grpo-debug-<jobid>.log
```

For GDPO logs:

```bash
tail -n 120 /scratch/$USER/verl-grpo/logs/slurm-verl-gdpo-upstream-<jobid>.log
```

```bash
tail -n 120 /scratch/$USER/verl-grpo/logs/slurm-verl-gdpo-nvlabs-<jobid>.log
```

If you want live-following without risking your interactive shell, prefer:

```bash
less +F /scratch/$USER/verl-grpo/logs/slurm-verl-grpo-debug-<jobid>.log
```

Or use a second SSH terminal for log following.

### Case E: Cancel A Running Batch Job

```bash
scancel <jobid>
```

## 6. Standard 7B Run

Do this only after the debug path is green.

The standard job script is still:

```bash
sbatch slurm/grpo_standard.sbatch
```

But as of the current SOL bring-up, the tested node image showed a `flash-attn` / `GLIBC_2.32` incompatibility. So the standard run should be treated as **not yet fully validated** until we either:

- build a SOL-compatible FlashAttention path
- or adopt a broader non-FlashAttention override for the larger run too

## 7. Tiny Mental Model

- `source scripts/sol/common_env.sh`
  restores repo helper functions and scratch path variables
- `sol_activate_env`
  activates the dedicated Mamba env in the current shell
- `./scripts/sol/bootstrap_lightwork.sh`
  first-time setup or reinstall step
- `sbatch ...`
  submits a detached batch job that keeps running even if your laptop terminal closes
