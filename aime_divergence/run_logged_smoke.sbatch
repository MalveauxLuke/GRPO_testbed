#!/usr/bin/env bash
# Fast SOL smoke test for the AIME logged rollout pipeline.
#
# This intentionally reuses run_logged.sbatch after setting smaller env defaults
# so the smoke test exercises the same logprobs extraction and checkpointing path
# as the full logged run.
#SBATCH --job-name=aime-divergence-logged-smoke
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/aime_divergence/run_logged.sbatch" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SBATCH_DIR}/.." && pwd)}"
fi

export PROJECT_ROOT
export AIME_DIVERGENCE_RUN_TAG="${AIME_DIVERGENCE_RUN_TAG:-logged_smoke_${SLURM_JOB_ID:-manual}}"
export AIME_DIVERGENCE_MAX_PROBLEMS="${AIME_DIVERGENCE_MAX_PROBLEMS:-2}"
export AIME_DIVERGENCE_SAMPLES="${AIME_DIVERGENCE_SAMPLES:-2}"
export AIME_DIVERGENCE_MAX_TOKENS="${AIME_DIVERGENCE_MAX_TOKENS:-2048}"
export AIME_DIVERGENCE_MAX_MODEL_LEN="${AIME_DIVERGENCE_MAX_MODEL_LEN:-4096}"
export AIME_DIVERGENCE_LOGPROBS="${AIME_DIVERGENCE_LOGPROBS:-20}"
export AIME_DIVERGENCE_TENSORBOARD="${AIME_DIVERGENCE_TENSORBOARD:-1}"

# The exported env vars above stay in effect because run_logged.sbatch only
# applies defaults via ${VAR:-default} before executing the Python entrypoint.
exec bash "${PROJECT_ROOT}/aime_divergence/run_logged.sbatch"
