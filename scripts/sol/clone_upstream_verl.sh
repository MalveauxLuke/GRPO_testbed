#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_ensure_runtime_dirs
sol_ensure_upstream_checkout

if [[ -d "${UPSTREAM_VERL_DIR}/.git" ]]; then
  sol_fail "Vendored source at ${UPSTREAM_VERL_DIR} still contains nested Git metadata. Remove external/verl/.git as part of the vendoring flow."
fi

sol_msg "Vendored verl source is present at ${UPSTREAM_VERL_DIR}."
sol_msg "scripts/sol/clone_upstream_verl.sh is now a compatibility validator only."
sol_msg "Bootstrap no longer clones upstream; use git pull on this repo to update both wrapper logic and vendored verl source."
