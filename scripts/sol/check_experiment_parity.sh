#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

extract_builder_block() {
  local input_path="$1"
  awk '/^\[sol-setup\] Dry-run only\./,0' "${input_path}"
}

check_case() {
  local case_name="$1"
  local config_path="$2"
  local sbatch_path="$3"
  local rollout_env_var="$4"
  local rollout_value="$5"

  local front_output="${TMP_DIR}/${case_name}.front.txt"
  local legacy_output="${TMP_DIR}/${case_name}.legacy.txt"
  local front_block="${TMP_DIR}/${case_name}.front.block.txt"
  local legacy_block="${TMP_DIR}/${case_name}.legacy.block.txt"

  env RUN_TAG="parity-check" "${rollout_env_var}=${rollout_value}" \
    "${PROJECT_ROOT}/scripts/sol/submit_experiment.sh" "${config_path}" --dry-run > "${front_output}"
  env RUN_TAG="parity-check" SLURM_JOB_NAME="parity-check" SLURM_JOB_ID="0000" "${rollout_env_var}=${rollout_value}" \
    bash "${PROJECT_ROOT}/${sbatch_path}" --dry-run > "${legacy_output}"

  extract_builder_block "${front_output}" > "${front_block}"
  extract_builder_block "${legacy_output}" > "${legacy_block}"

  if ! diff -u "${front_block}" "${legacy_block}"; then
    printf 'Parity mismatch for %s\n' "${case_name}" >&2
    return 1
  fi

  printf 'Parity OK: %s\n' "${case_name}"
}

check_case \
  "gsm8k_debug" \
  "${PROJECT_ROOT}/configs/experiments/gsm8k/debug_answer_tag_1p5b_instruct.env" \
  "slurm/gdpo_gsm8k_modern_fit_debug_hybrid_hash.sbatch" \
  "GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  "/scratch/${USER}/verl-grpo/rollout_dumps/parity-gsm8k-debug"

check_case \
  "gsm8k_saturation_75step" \
  "${PROJECT_ROOT}/configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env" \
  "slurm/gdpo_gsm8k_modern_fit_2gpu_hybrid_hash_saturation_check.sbatch" \
  "GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  "/scratch/${USER}/verl-grpo/rollout_dumps/parity-gsm8k-75step"

check_case \
  "gsm8k_full" \
  "${PROJECT_ROOT}/configs/experiments/gsm8k/full_answer_tag_1p5b_instruct.env" \
  "slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch" \
  "GSM8K_MODERN_ROLLOUT_DATA_DIR" \
  "/scratch/${USER}/verl-grpo/rollout_dumps/parity-gsm8k-full"

check_case \
  "math_grpo_debug" \
  "${PROJECT_ROOT}/configs/experiments/math/grpo_debug.env" \
  "slurm/grpo_math_length_debug.sbatch" \
  "MATH_LENGTH_ROLLOUT_DATA_DIR" \
  "/scratch/${USER}/verl-grpo/rollout_dumps/parity-math-grpo-debug"

check_case \
  "math_gdpo_debug" \
  "${PROJECT_ROOT}/configs/experiments/math/gdpo_debug.env" \
  "slurm/gdpo_math_length_debug.sbatch" \
  "MATH_LENGTH_ROLLOUT_DATA_DIR" \
  "/scratch/${USER}/verl-grpo/rollout_dumps/parity-math-gdpo-debug"

check_case \
  "math_grpo_production" \
  "${PROJECT_ROOT}/configs/experiments/math/grpo_production.env" \
  "slurm/grpo_math_length_production.sbatch" \
  "MATH_LENGTH_ROLLOUT_DATA_DIR" \
  "/scratch/${USER}/verl-grpo/rollout_dumps/parity-math-grpo-production"

check_case \
  "math_gdpo_production" \
  "${PROJECT_ROOT}/configs/experiments/math/gdpo_production.env" \
  "slurm/gdpo_math_length_production.sbatch" \
  "MATH_LENGTH_ROLLOUT_DATA_DIR" \
  "/scratch/${USER}/verl-grpo/rollout_dumps/parity-math-gdpo-production"
