#!/usr/bin/env bash
# Submit with an optional account override:
#   sbatch -A grp_<your_group> slurm/gdpo_saturation_probe_math.sbatch
# Or:
#   SOL_ACCOUNT=grp_<your_group> sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/gdpo_saturation_probe_math.sbatch
# This runs the GDPO saturation probe on DeepScaleR-derived math data using the
# existing boxed-answer correctness+length reward path under a longer 200-step debug profile.
#SBATCH --job-name=verl-gdpo-saturation-probe-math
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
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
sol_msg "Submitting GDPO saturation probe on DeepScaleR math from ${PROJECT_ROOT}."

"${PROJECT_ROOT}/scripts/sol/run_gdpo_saturation_probe_math.sh" "$@"
