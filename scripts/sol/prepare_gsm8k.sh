#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

sol_msg "Preparing GSM8K parquet files under ${GSM8K_DIR}."
"$(sol_python)" "${UPSTREAM_VERL_DIR}/examples/data_preprocess/gsm8k.py" --local_save_dir "${GSM8K_DIR}"

[[ -f "${GSM8K_DIR}/train.parquet" ]] || sol_fail "Expected ${GSM8K_DIR}/train.parquet to exist after preprocessing."
[[ -f "${GSM8K_DIR}/test.parquet" ]] || sol_fail "Expected ${GSM8K_DIR}/test.parquet to exist after preprocessing."

sol_msg "Prepared:"
sol_msg "  ${GSM8K_DIR}/train.parquet"
sol_msg "  ${GSM8K_DIR}/test.parquet"
