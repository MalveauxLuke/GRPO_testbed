#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_ensure_runtime_dirs
sol_activate_env

TB_SCOPE="${1:-both}"
TB_PORT="${2:-6006}"
TB_HOST="${TB_HOST:-127.0.0.1}"

DEBUG_PROJECT="${TENSORBOARD_ROOT}/${SOL_PROJECT_NAME}_gdpo_gsm8k_modern_fit_debug_answer_tag_1_5b_instruct"
CAPPED_PROJECT="${TENSORBOARD_ROOT}/${SOL_PROJECT_NAME}_gdpo_gsm8k_modern_answer_tag_1_5b_instruct_saturation_check"
FULL_PROJECT="${TENSORBOARD_ROOT}/${SOL_PROJECT_NAME}_gdpo_gsm8k_modern_fit_2gpu_answer_tag_1_5b_instruct"

case "${TB_SCOPE}" in
  debug)
    LOGDIR_SPEC="debug:${DEBUG_PROJECT}"
    ;;
  capped|main|saturation)
    LOGDIR_SPEC="capped:${CAPPED_PROJECT}"
    ;;
  full)
    LOGDIR_SPEC="full:${FULL_PROJECT}"
    ;;
  both|current)
    LOGDIR_SPEC="debug:${DEBUG_PROJECT},capped:${CAPPED_PROJECT}"
    ;;
  all)
    LOGDIR_SPEC="debug:${DEBUG_PROJECT},capped:${CAPPED_PROJECT},full:${FULL_PROJECT}"
    ;;
  *)
    sol_fail "Unknown scope '${TB_SCOPE}'. Use one of: debug, capped, full, both, all."
    ;;
esac

sol_msg "Starting TensorBoard for current GSM8K-modern 1.5B-Instruct runs."
sol_msg "Scope: ${TB_SCOPE}"
sol_msg "Host: ${TB_HOST}"
sol_msg "Port: ${TB_PORT}"
sol_msg "Logdir spec: ${LOGDIR_SPEC}"
sol_msg "URL: http://${TB_HOST}:${TB_PORT}"

"$(sol_python)" -m tensorboard.main --logdir_spec "${LOGDIR_SPEC}" --host "${TB_HOST}" --port "${TB_PORT}"
