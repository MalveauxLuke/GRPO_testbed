#!/usr/bin/env bash
# Config provenance:
# - Semantic baseline: DeepScaleR-style boxed-answer math reward (`correct_reward + length_reward`).
# - Runtime profile: reference_public.
# - Entry type: full-run Slurm entrypoint for math-length GRPO production.
# - Upstream anchor inherited through: /Users/god/Documents/VERL_GRPO/scripts/sol/run_grpo_math_length_production.sh
# - Local parent config: /Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh
# - Historical failure reference: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md
# - Required manual gate before edits: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md
# Submit with an optional account override:
#   sbatch -A grp_<your_group> slurm/grpo_math_length_production.sbatch
# Or:
#   SOL_ACCOUNT=grp_<your_group> sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/grpo_math_length_production.sbatch
# This submits the production-tier DeepScaleR-style correctness+length GRPO wrapper.
# By default it requests 8 GPUs and tells the wrapper to use all 8 on one node.
# Override MATH_LENGTH_N_GPUS_PER_NODE or the Slurm resources if your account needs a smaller run.
#SBATCH --job-name=verl-grpo-math-length
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
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
sol_msg "Submitting production DeepScaleR-style GRPO math-length wrapper from ${PROJECT_ROOT}."

export MATH_LENGTH_N_GPUS_PER_NODE="${MATH_LENGTH_N_GPUS_PER_NODE:-8}"

"${PROJECT_ROOT}/scripts/sol/run_grpo_math_length_production.sh" "$@"
