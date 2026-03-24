#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

[[ -f "${GSM8K_DIR}/train.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/train.parquet. Run scripts/sol/prepare_gsm8k.sh first."
[[ -f "${GSM8K_DIR}/test.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/test.parquet. Run scripts/sol/prepare_gsm8k.sh first."

RUN_TAG="${RUN_TAG:-$(sol_timestamp)}"
EXPERIMENT_NAME="${STANDARD_EXPERIMENT_NAME:-qwen2_7b_function_rm_sol}"
RUN_ROOT="${OUTPUT_ROOT}/standard/${RUN_TAG}"
LOCAL_CKPT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/${RUN_TAG}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"

if [[ "${ENABLE_WANDB}" == "1" ]]; then
  LOGGER='["console","wandb"]'
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    sol_msg "ENABLE_WANDB=1 but WANDB_API_KEY is not set; relying on an existing wandb login."
  fi
else
  LOGGER='["console"]'
fi

mkdir -p "${RUN_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${RUN_ROOT}"

sol_msg "Starting standard single-node upstream GRPO run."
sol_msg "Run root: ${RUN_ROOT}"
sol_msg "Checkpoint dir: ${LOCAL_CKPT_DIR}"

bash "${UPSTREAM_VERL_DIR}/examples/grpo_trainer/run_qwen2-7b.sh" \
  data.train_files="${GSM8K_DIR}/train.parquet" \
  data.val_files="${GSM8K_DIR}/test.parquet" \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${SOL_PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${LOCAL_CKPT_DIR}" \
  "$@"
