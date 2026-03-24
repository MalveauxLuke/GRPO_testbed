#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_ensure_runtime_dirs

if [[ ! -d "${UPSTREAM_VERL_DIR}/.git" ]]; then
  sol_msg "Cloning official upstream verl into ${UPSTREAM_VERL_DIR}."
  git clone https://github.com/volcengine/verl.git "${UPSTREAM_VERL_DIR}"
else
  sol_msg "Upstream checkout already exists at ${UPSTREAM_VERL_DIR}; fetching tags."
fi

git -C "${UPSTREAM_VERL_DIR}" fetch --tags origin
git -C "${UPSTREAM_VERL_DIR}" checkout "${VERL_REF}"
git -C "${UPSTREAM_VERL_DIR}" submodule update --init --recursive

sol_msg "Pinned upstream verl to $(git -C "${UPSTREAM_VERL_DIR}" rev-parse --short HEAD) (${VERL_REF})."
