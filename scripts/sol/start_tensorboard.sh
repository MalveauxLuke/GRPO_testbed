#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_ensure_runtime_dirs
sol_activate_env

TB_LOGDIR="${1:-${TENSORBOARD_ROOT}}"
TB_PORT="${2:-6006}"

mkdir -p "${TB_LOGDIR}"

sol_msg "Starting TensorBoard."
sol_msg "Log dir: ${TB_LOGDIR}"
sol_msg "URL: http://127.0.0.1:${TB_PORT}"

"$(sol_python)" -m tensorboard.main --logdir "${TB_LOGDIR}" --host 127.0.0.1 --port "${TB_PORT}"
