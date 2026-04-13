#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

cd "${UPSTREAM_VERL_DIR}"

[[ -f scripts/install_vllm_sglang_mcore.sh ]] || sol_fail "Official install script not found under ${UPSTREAM_VERL_DIR}."

sol_msg "Installing official upstream verl dependencies with vLLM-only settings."
PYTHON_BIN="$(sol_python)" USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh

sol_msg "Installing AIME/debug analysis helpers."
"$(sol_python)" -m pip install "math-verify" "tensorboard"

sol_msg "Pinning Transformers below 5.x for vLLM tokenizer compatibility."
"$(sol_python)" -m pip install "transformers[hf_xet]>=4.51.0,<5.0"

sol_msg "Repairing NumPy to the upstream verl-supported range for SOL."
"$(sol_python)" -m pip install "numpy>=1.26,<2.0"

sol_msg "Reinstalling upstream verl editable checkout after dependency repair."
"$(sol_python)" -m pip install --no-deps -e .

sol_msg "Install finished."
