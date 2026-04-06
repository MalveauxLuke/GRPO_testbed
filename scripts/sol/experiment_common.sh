#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_resolve_experiment_config_path() {
  local requested_path="${1:-}"
  [[ -n "${requested_path}" ]] || sol_fail "Expected an experiment config path."

  if [[ -f "${requested_path}" ]]; then
    printf '%s\n' "${requested_path}"
    return 0
  fi

  if [[ -f "${PROJECT_ROOT}/${requested_path}" ]]; then
    printf '%s\n' "${PROJECT_ROOT}/${requested_path}"
    return 0
  fi

  sol_fail "Experiment config not found: ${requested_path}"
}

sol_load_experiment_config() {
  local config_path="${1:-}"
  [[ -f "${config_path}" ]] || sol_fail "Experiment config does not exist: ${config_path}"

  # shellcheck disable=SC1090
  source "${config_path}"

  export EXPERIMENT_CONFIG_PATH="${config_path}"
  export EXPERIMENT_TITLE="${EXPERIMENT_TITLE:-$(basename "${config_path}" .env)}"
  export EXPERIMENT_WORKFLOW="${EXPERIMENT_WORKFLOW:-unknown}"
  export EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-}"
  export EXPERIMENT_SUBMIT_TARGET="${EXPERIMENT_SUBMIT_TARGET:-}"
  export EXPERIMENT_DRY_RUN_TARGET="${EXPERIMENT_DRY_RUN_TARGET:-}"
  export EXPERIMENT_POST_RUN_AUDIT="${EXPERIMENT_POST_RUN_AUDIT:-}"
  export EXPERIMENT_ROLLOUT_ENV_VAR="${EXPERIMENT_ROLLOUT_ENV_VAR:-}"
  export EXPERIMENT_ROLLOUT_PREFIX="${EXPERIMENT_ROLLOUT_PREFIX:-}"
  export EXPERIMENT_TENSORBOARD_HINT="${EXPERIMENT_TENSORBOARD_HINT:-}"
  export EXPERIMENT_PRINT_ENV_VARS="${EXPERIMENT_PRINT_ENV_VARS:-}"

  [[ -n "${EXPERIMENT_SUBMIT_TARGET}" ]] || sol_fail "Config ${config_path} did not set EXPERIMENT_SUBMIT_TARGET."
  [[ -n "${EXPERIMENT_DRY_RUN_TARGET}" ]] || sol_fail "Config ${config_path} did not set EXPERIMENT_DRY_RUN_TARGET."
}

sol_apply_experiment_rollout_default() {
  local rollout_env_var="${EXPERIMENT_ROLLOUT_ENV_VAR:-}"
  local rollout_prefix="${EXPERIMENT_ROLLOUT_PREFIX:-}"
  [[ -n "${rollout_env_var}" && -n "${rollout_prefix}" ]] || return 0

  if [[ -n "${!rollout_env_var:-}" ]]; then
    return 0
  fi

  local rollout_dir="${SCRATCH_ROOT}/rollout_dumps/${rollout_prefix}_$(sol_timestamp)"
  printf -v "${rollout_env_var}" '%s' "${rollout_dir}"
  export "${rollout_env_var}"
}

sol_print_experiment_overview() {
  sol_msg "Experiment config: ${EXPERIMENT_CONFIG_PATH}"
  sol_msg "Title: ${EXPERIMENT_TITLE}"
  if [[ -n "${EXPERIMENT_DESCRIPTION:-}" ]]; then
    sol_msg "Description: ${EXPERIMENT_DESCRIPTION}"
  fi
  sol_msg "Workflow: ${EXPERIMENT_WORKFLOW}"
  sol_msg "Slurm entrypoint: ${EXPERIMENT_SUBMIT_TARGET}"
  sol_msg "Builder: ${EXPERIMENT_DRY_RUN_TARGET}"
  if [[ -n "${EXPERIMENT_POST_RUN_AUDIT:-}" ]]; then
    sol_msg "Recommended post-run audit: ${EXPERIMENT_POST_RUN_AUDIT}"
  fi
  if [[ -n "${EXPERIMENT_TENSORBOARD_HINT:-}" ]]; then
    sol_msg "TensorBoard hint: ${EXPERIMENT_TENSORBOARD_HINT}"
  fi
  if [[ -n "${EXPERIMENT_PRINT_ENV_VARS:-}" ]]; then
    # shellcheck disable=SC2086
    sol_print_named_variables "Key experiment environment variables" ${EXPERIMENT_PRINT_ENV_VARS}
  fi
}
