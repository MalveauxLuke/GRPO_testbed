#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/create_env.sh"
"${SCRIPT_DIR}/install_upstream_verl.sh"
"${SCRIPT_DIR}/verify_install.sh"

printf '[sol-setup] Bootstrap complete with vendored external/verl source. Next steps: prepare data with scripts/sol/prepare_gsm8k.sh and prewarm the debug model with scripts/sol/prewarm_model.sh.\n'
