# Repo State And Vendoring Record

This file is the durable note for the overall `VERL_GRPO` repo state after vendoring `external/verl` into this repository.

## Current State

The repo now contains two tracked parts:

- the SOL wrapper:
  - setup scripts
  - Slurm entrypoints
  - docs
  - scratch/cache policy
- the vendored `verl` source under `external/verl`

This means local edits to `external/verl` can now be versioned in the same Git history as the SOL wrapper, pushed from a workstation, and pulled on SOL without relying on a separate unmanaged checkout.

## Vendored Base Metadata

- upstream repo URL: `https://github.com/volcengine/verl.git`
- upstream tag: `v0.7.1`
- upstream commit: `bec9ef74`
- upstream package version file: `0.7.1`
- vendoring date: `2026-03-24`

## Important Vendoring Notes

- `external/verl` was vendored from the exact checkout that had already been stabilized for SOL bring-up.
- upstream `.gitmodules` is preserved as historical metadata.
- upstream `recipe/` was recorded as a submodule in `.gitmodules`, but it was uninitialized at vendoring time and is **not** part of the validated SOL path in this repo.
- this pass does **not** vendor `recipe/`; the current validated SOL bootstrap and debug workflow do not depend on it.

## What Changed In Repo Behavior

### Bootstrap

Bootstrap no longer clones upstream `verl`.

It now performs:

1. env creation
2. install from vendored `external/verl`
3. import verification

### Source Validation

The shared env helpers no longer treat `external/verl/.git` as the source-of-truth.

They now validate the vendored source using expected files such as:

- `external/verl/setup.py`
- `external/verl/scripts/install_vllm_sglang_mcore.sh`
- `external/verl/examples/grpo_trainer/run_qwen2-7b.sh`

### `clone_upstream_verl.sh`

`scripts/sol/clone_upstream_verl.sh` is now only a compatibility validator.

It no longer clones, fetches, checks out tags, or mutates `external/verl`. Normal updates now happen through `git pull` on this repo.

### Cleanup

`scripts/sol/cleanup_reset.sh --remove-upstream` now refuses because `external/verl` is tracked source.

`--all` still removes:

- scratch runtime data
- the dedicated Mamba env
- repo-local fallback Slurm logs

but it deliberately preserves vendored source.

## Why This Migration Happened

The original split layout was useful for early bring-up because it kept the repo close to official upstream `verl` while we validated SOL compatibility.

That goal is now complete enough that the better tradeoff is reproducibility:

- one repo contains both wrapper logic and `verl` source
- SOL and local workstations can stay in sync through normal Git operations
- future algorithm work, architecture changes, and local patches to `verl` can be reviewed and reproduced cleanly

## Future Upstream Sync Guidance

The current default assumption is:

- keep `v0.7.1` / `bec9ef74` as the recorded upstream base
- make downstream changes in this repo
- handle future upstream syncs manually and explicitly

Any future sync should record:

- the upstream target tag or commit
- the merge/update date
- notable local patches that were preserved or rebased

## Decision Record

As of `2026-03-24`, the repo is no longer in the temporary split-layout phase.

The decision is now:

- keep `external/verl` tracked in this repo
- use normal Git operations to propagate both wrapper and `verl` source changes
- preserve the current SOL-compatible debug validation path as the operational baseline

Any future thread working in this repo should assume vendored `external/verl` unless the user explicitly decides to re-architect that choice.
