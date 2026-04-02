#!/usr/bin/env bash
# Config provenance:
# - Semantic baseline: DeepScaleR-style boxed-answer math reward (`correct_reward + length_reward`).
# - Runtime profile: debug_safe.
# - Entry type: integration-smoke Slurm entrypoint for math-length GDPO debug.
# - Upstream anchor inherited through: /Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_math_length_debug.sh
# - Local parent config: /Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh
# - Historical failure reference: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md
# - Required manual gate before edits: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md
# Submit with an optional account override:
#   sbatch -A grp_<your_group> slurm/gdpo_math_length_debug.sbatch
# Or:
#   SOL_ACCOUNT=grp_<your_group> sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/gdpo_math_length_debug.sbatch
# This submits the DeepScaleR-style correctness+length GDPO debug wrapper.
#SBATCH --job-name=verl-gdpo-math-length-debug
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
sol_msg "Submitting DeepScaleR-style GDPO math-length debug wrapper from ${PROJECT_ROOT}."

"${PROJECT_ROOT}/scripts/sol/run_gdpo_math_length_debug.sh" "$@"
