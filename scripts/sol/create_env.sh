#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_load_mamba
sol_deactivate_base_if_needed

if mamba env list | awk 'NR > 2 {print $1}' | tr -d '*' | grep -Fxq "${SOL_ENV_NAME}"; then
  sol_msg "Mamba env '${SOL_ENV_NAME}' already exists."
else
  sol_msg "Creating Mamba env '${SOL_ENV_NAME}' with Python 3.12."
  mamba create -y -n "${SOL_ENV_NAME}" python=3.12
fi

sol_activate_env
sol_msg "Python version inside '${SOL_ENV_NAME}': $(python3 --version)"
