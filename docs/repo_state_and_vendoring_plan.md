# Repo State And Future Vendoring Plan

This file is the durable note for the overall `VERL_GRPO` repo state.

## Current State

Right now the repo is intentionally in a **split** layout because the immediate goal is to get official upstream `verl` GRPO running on ASU SOL with the smallest possible divergence from upstream:

- This GitHub repo tracks the **SOL wrapper**:
  - SOL setup scripts
  - Slurm scripts
  - docs
  - scratch/cache path policy
- The actual `verl` source is currently expected to live in `external/verl`
- `external/verl` is currently treated as a **separate upstream checkout**
- During bootstrap, the repo clones official upstream `verl`, pins it to `v0.7.1`, and installs from that source tree

This is a temporary architecture choice for validation and bring-up.

## Important Intent

This split layout is **not** the intended long-term architecture.

The next major step, after we confirm the current setup works on SOL, is:

> Move the actual `verl` source into this GitHub repo so that architecture-level changes to `verl` itself can be versioned, edited locally, pushed to GitHub, and then pulled on SOL without relying on a separate unmanaged checkout.

In other words:

- **Current phase**: wrapper repo + upstream clone, just to validate setup
- **Next phase**: vendor the `verl` source into this repo cleanly and make this repo the working source tree for downstream architectural changes

## Why The Split Exists Right Now

The current split was chosen because:

- it keeps the first pass close to official upstream `verl`
- it reduces initial repo size and initial Git complexity
- it makes it easier to verify that any setup issue is not caused by a custom fork too early

That is useful for bring-up, but it is not ideal if we plan to modify real `verl` internals.

## Required Changes When We Vendor `verl` Into This Repo

When we decide to stop treating `external/verl` as a separate clone and start tracking it in GitHub, the following changes must be made cleanly.

### 1. Stop ignoring `external/verl`

- Remove `external/verl/` from `.gitignore`
- Ensure no unwanted build/cache files under `external/verl` are committed

### 2. Convert `external/verl` from nested checkout to tracked source

- Record the pinned upstream source revision before conversion
- Remove `external/verl/.git`
- Commit the actual source files under `external/verl`

Recommended metadata to preserve in docs or a tracked note:

- upstream repo URL
- upstream tag
- upstream commit hash
- date of vendoring

### 3. Patch repo scripts that currently assume `external/verl/.git`

These scripts currently assume `external/verl` is an external checkout and must be updated:

- `scripts/sol/common_env.sh`
  - `sol_ensure_upstream_checkout` should stop checking for `external/verl/.git`
  - it should instead validate expected source files such as:
    - `external/verl/setup.py`
    - `external/verl/examples/grpo_trainer/run_qwen2-7b.sh`
    - `external/verl/scripts/install_vllm_sglang_mcore.sh`

- `scripts/sol/clone_upstream_verl.sh`
  - should no longer clone by default once source is vendored
  - should become one of:
    - a no-op validator
    - a manually-invoked sync helper
    - or be removed from bootstrap entirely

- `scripts/sol/bootstrap_lightwork.sh`
  - should stop depending on clone behavior
  - if vendored source exists, bootstrap should just:
    - create env
    - install vendored source
    - verify imports

- `scripts/sol/cleanup_reset.sh`
  - `--remove-upstream` should not delete vendored tracked source
  - for vendored mode it should either:
    - refuse
    - or only clean generated files, not source

### 4. Update docs to reflect vendored architecture

The following docs should be updated:

- `README.md`
- `docs/asu_sol_upstream_verl_grpo.md`

These docs should stop describing `external/verl` as a cloned external dependency and start describing it as:

- tracked vendored source
- pinned to upstream `v0.7.1` as the base import point
- locally modifiable and GitHub-backed

### 5. Decide how upstream sync will work after vendoring

This decision must be explicit.

Possible approaches:

- manual upstream sync by copying/merging from official `verl`
- keep a recorded upstream base tag and merge forward manually
- later switch to a fork-based workflow

At minimum, the repo should keep a tracked note of:

- current upstream base tag
- current upstream base commit
- any local modifications relative to that base

### 6. Make architectural editing reproducible across PC and SOL

After vendoring:

- edits to `external/verl` on a local machine can be committed and pushed
- SOL can `git pull`
- the source used by the install/run scripts will match GitHub

This is the main reason for the migration.

## Recommended Future Implementation Shape

When the migration happens, the clean target state should be:

- `external/verl` is tracked in Git
- bootstrap does **not** clone official upstream `verl`
- bootstrap only:
  - creates env
  - installs vendored `external/verl`
  - verifies imports
- docs clearly say this repo now contains:
  - SOL wrapper logic
  - vendored `verl` source

## Suggested Migration Checklist

When we are ready to do the migration, follow this checklist:

1. Confirm current SOL validation workflow works end-to-end.
2. Record the exact upstream `verl` tag and commit in use.
3. Remove `external/verl/` from `.gitignore`.
4. Remove nested Git metadata from `external/verl`.
5. Commit vendored source into this repo.
6. Patch `common_env.sh`, `clone_upstream_verl.sh`, `bootstrap_lightwork.sh`, and `cleanup_reset.sh`.
7. Update README and SOL docs to match the new architecture.
8. Re-test bootstrap on SOL.
9. Re-test debug batch run on SOL.
10. Only after that start changing `verl` internals in-repo.

## Decision Record

As of now, the decision is:

- keep the split architecture temporarily
- use it only to validate the current SOL setup
- then migrate to vendored tracked `verl` source for real architecture work

Any future thread working in this repo should preserve that intent unless the user explicitly changes direction.
