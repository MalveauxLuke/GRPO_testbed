#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

REMOVE_ENV=0
REMOVE_UPSTREAM=0
REMOVE_REPO_LOGS=0

usage() {
  cat <<'EOF'
Usage:
  scripts/sol/cleanup_reset.sh [--remove-env] [--remove-upstream] [--all]

Defaults:
  Removes only the repo-managed scratch runtime tree under /scratch/$USER/verl-grpo.

Options:
  --remove-env       Also remove the dedicated Mamba environment.
  --remove-upstream  Also remove external/verl.
  --all              Remove scratch runtime, env, upstream checkout, and repo-local slurm output files.
  -h, --help         Show this help text.
EOF
}

while (($#)); do
  case "$1" in
    --remove-env)
      REMOVE_ENV=1
      ;;
    --remove-upstream)
      REMOVE_UPSTREAM=1
      ;;
    --all)
      REMOVE_ENV=1
      REMOVE_UPSTREAM=1
      REMOVE_REPO_LOGS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      sol_fail "Unknown option: $1"
      ;;
  esac
  shift
done

if [[ "${SCRATCH_ROOT}" != "/scratch/${USER}/"* ]]; then
  sol_fail "Refusing to delete SCRATCH_ROOT outside /scratch/${USER}: ${SCRATCH_ROOT}"
fi

if [[ "${SCRATCH_ROOT}" == "/scratch/${USER}" ]]; then
  sol_fail "Refusing to delete /scratch/${USER} directly."
fi

sol_msg "Removing repo-managed scratch runtime at ${SCRATCH_ROOT}."
rm -rf "${SCRATCH_ROOT}"

if (( REMOVE_UPSTREAM )); then
  if [[ "${UPSTREAM_VERL_DIR}" != "${PROJECT_ROOT}/external/"* ]]; then
    sol_fail "Refusing to remove upstream checkout outside ${PROJECT_ROOT}/external: ${UPSTREAM_VERL_DIR}"
  fi
  sol_msg "Removing upstream checkout at ${UPSTREAM_VERL_DIR}."
  rm -rf "${UPSTREAM_VERL_DIR}"
fi

if (( REMOVE_ENV )); then
  sol_load_mamba
  sol_deactivate_base_if_needed
  if mamba env list | awk 'NR > 2 {print $1}' | tr -d '*' | grep -Fxq "${SOL_ENV_NAME}"; then
    sol_msg "Removing Mamba env ${SOL_ENV_NAME}."
    mamba remove -y -n "${SOL_ENV_NAME}" --all
  else
    sol_msg "Mamba env ${SOL_ENV_NAME} does not exist; nothing to remove."
  fi
fi

if (( REMOVE_REPO_LOGS )); then
  sol_msg "Removing repo-local slurm fallback logs."
  find "${PROJECT_ROOT}" -maxdepth 1 -type f \( -name 'slurm-*.out' -o -name 'slurm-*.err' \) -delete
fi

sol_msg "Cleanup complete."
