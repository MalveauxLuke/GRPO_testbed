#!/usr/bin/env bash
# Submit with an optional account override:
#   sbatch -A grp_<your_group> slurm/gdpo_gsm8k_modern_debug.sbatch
# Or:
#   SOL_ACCOUNT=grp_<your_group> sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/gdpo_gsm8k_modern_debug.sbatch
# Optional rollout dump capture:
#   GSM8K_MODERN_ROLLOUT_DATA_DIR=/scratch/$USER/verl-grpo/rollout_dumps/gsm8k_modern sbatch slurm/gdpo_gsm8k_modern_debug.sbatch
# Legacy compatibility path retained for historical verification notes.
# Prefer: slurm/gdpo_gsm8k_modern_fit_debug.sbatch
#SBATCH --job-name=legacy-verl-gdpo-gsm8k-modern-debug
#SBATCH --partition=htc
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SBATCH_DIR}/.." && pwd)}"
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/sol/common_env.sh"

sol_ensure_runtime_dirs

JOB_LOG="${LOG_ROOT}/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

sol_msg "Scratch log: ${JOB_LOG}"
sol_msg "Submitting LEGACY modern 2-reward GSM8K GDPO debug wrapper from ${PROJECT_ROOT}."
sol_msg "Preferred active GSM8K debug/preflight path: slurm/gdpo_gsm8k_modern_fit_debug.sbatch"

"${PROJECT_ROOT}/scripts/sol/run_gdpo_gsm8k_modern_debug.sh" "$@"
