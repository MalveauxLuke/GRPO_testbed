#!/usr/bin/env bash
# Config provenance:
# - Entry type: full realistic GSM8K modern GDPO entrypoint.
# - Upstream anchor inherited through: /Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_realistic.sh
# - Local parent config: /Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_realistic.sh
# - Historical failure reference: /Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_error_catalog.md
# - Required manual gate before edits: /Users/god/Documents/VERL_GRPO/docs/gsm8k_modern_config_creation_checklist.md
# Submit with an optional account override:
#   sbatch -A grp_<your_group> slurm/gdpo_gsm8k_modern_realistic.sbatch
# Or:
#   SOL_ACCOUNT=grp_<your_group> sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/gdpo_gsm8k_modern_realistic.sbatch
# Optional rollout dump capture:
#   GSM8K_MODERN_ROLLOUT_DATA_DIR=/scratch/$USER/verl-grpo/rollout_dumps/gsm8k_modern_realistic_$(date +%Y%m%d-%H%M%S) sbatch slurm/gdpo_gsm8k_modern_realistic.sbatch
# This wrapper keeps the verified GSM8K modern two-reward baseline but runs it in a realistic 3B/2-GPU GDPO setup.
#SBATCH --job-name=verl-gdpo-gsm8k-modern-realistic
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=1-00:00:00
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
sol_msg "Submitting realistic modern 2-reward GSM8K GDPO wrapper from ${PROJECT_ROOT}."

export GSM8K_MODERN_N_GPUS_PER_NODE="${GSM8K_MODERN_N_GPUS_PER_NODE:-2}"

"${PROJECT_ROOT}/scripts/sol/run_gdpo_gsm8k_modern_realistic.sh" "$@"
