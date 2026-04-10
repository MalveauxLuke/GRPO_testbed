#!/usr/bin/env bash
# Fast SOL smoke test for the AIME divergence pipeline.
#
# This intentionally reuses run_debug.sbatch after setting smaller env defaults.
# That keeps the smoke test on the same model/env/logging/TensorBoard code path
# as the real debug run, while reducing the work to a tiny 2-problem x 2-rollout
# check.
#
# Submit from the repo root with:
#
#   sbatch aime_divergence/run_smoke.sbatch
#
# Expected output:
# - 4 total rollouts
# - JSON under /scratch/$USER/verl-grpo/outputs/aime_divergence/smoke_<jobid>/
# - TensorBoard logs under that same output directory
# - scratch log under /scratch/$USER/verl-grpo/logs/
#SBATCH --job-name=aime-divergence-smoke
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/aime_divergence/run_debug.sbatch" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SBATCH_DIR}/.." && pwd)}"
fi

export PROJECT_ROOT
export AIME_DIVERGENCE_RUN_TAG="${AIME_DIVERGENCE_RUN_TAG:-smoke_${SLURM_JOB_ID:-manual}}"
export AIME_DIVERGENCE_MAX_PROBLEMS="${AIME_DIVERGENCE_MAX_PROBLEMS:-2}"
export AIME_DIVERGENCE_SAMPLES="${AIME_DIVERGENCE_SAMPLES:-2}"
export AIME_DIVERGENCE_MAX_TOKENS="${AIME_DIVERGENCE_MAX_TOKENS:-2048}"
export AIME_DIVERGENCE_MAX_MODEL_LEN="${AIME_DIVERGENCE_MAX_MODEL_LEN:-4096}"
export AIME_DIVERGENCE_TENSORBOARD="${AIME_DIVERGENCE_TENSORBOARD:-1}"

exec bash "${PROJECT_ROOT}/aime_divergence/run_debug.sbatch"
