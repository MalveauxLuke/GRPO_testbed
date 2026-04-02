#!/usr/bin/env bash
# Config provenance:
# - Semantic baseline: modern GSM8K two-reward GDPO (`correct_reward + format_reward`).
# - Runtime profile: sol_fit_2gpu.
# - Entry type: 15-minute feasibility preflight for the GSM8K fit-contract path.
# - Upstream anchor inherited through: /Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh
# - Local parent config: /Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh
# - Historical failure reference: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md
# - Required manual gate before edits: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md
# Submit with an optional account override:
#   sbatch -A grp_<your_group> slurm/gdpo_gsm8k_modern_fit_debug.sbatch
# Or:
#   SOL_ACCOUNT=grp_<your_group> sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/gdpo_gsm8k_modern_fit_debug.sbatch
# Optional rollout dump capture:
#   GSM8K_MODERN_ROLLOUT_DATA_DIR=/scratch/$USER/verl-grpo/rollout_dumps/gsm8k_modern_fit_debug_$(date +%Y%m%d-%H%M%S) sbatch slurm/gdpo_gsm8k_modern_fit_debug.sbatch
#SBATCH --job-name=verl-gdpo-gsm8k-modern-fit-debug
#SBATCH --partition=htc
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
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
sol_msg "Submitting 15-minute GSM8K fit-contract feasibility preflight from ${PROJECT_ROOT}."

export GSM8K_MODERN_N_GPUS_PER_NODE="${GSM8K_MODERN_N_GPUS_PER_NODE:-2}"
export GSM8K_MODERN_TOTAL_TRAINING_STEPS="${GSM8K_MODERN_TOTAL_TRAINING_STEPS:-1}"
export GSM8K_MODERN_TOTAL_EPOCHS="${GSM8K_MODERN_TOTAL_EPOCHS:-1}"
export GSM8K_MODERN_SAVE_FREQ="${GSM8K_MODERN_SAVE_FREQ:-1}"
export GSM8K_MODERN_TEST_FREQ="${GSM8K_MODERN_TEST_FREQ:-1}"
export GSM8K_MODERN_VAL_BEFORE_TRAIN="${GSM8K_MODERN_VAL_BEFORE_TRAIN:-False}"
export GSM8K_MODERN_RUN_CLASSIFICATION="${GSM8K_MODERN_RUN_CLASSIFICATION:-feasibility-smoke}"
export GSM8K_MODERN_RUNTIME_PROFILE="${GSM8K_MODERN_RUNTIME_PROFILE:-sol_fit_2gpu}"

"${PROJECT_ROOT}/scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh" "$@"
