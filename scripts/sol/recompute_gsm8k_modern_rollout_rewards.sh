#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_ensure_runtime_dirs
sol_activate_env

INPUT_PATH="${1:-${GSM8K_MODERN_ROLLOUT_DATA_DIR:-}}"
[[ -n "${INPUT_PATH}" ]] || sol_fail "Pass a rollout dump path as the first argument or set GSM8K_MODERN_ROLLOUT_DATA_DIR."

RUN_LABEL="${GSM8K_ROLLOUT_AUDIT_LABEL:-$(basename "${INPUT_PATH}")}"
OUTPUT_DIR="${OUTPUT_ROOT}/gsm8k_modern/rollout_audits/${RUN_LABEL}"
SUMMARY_PATH="${OUTPUT_DIR}/summary.json"
MISMATCH_PATH="${OUTPUT_DIR}/mismatches.jsonl"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "$(sol_python)"
  "${PROJECT_ROOT}/scripts/sol/audit_gsm8k_modern_rewards.py"
  recompute-summary
  --input-path "${INPUT_PATH}"
  --input-type rollout
  --summary-output "${SUMMARY_PATH}"
  --mismatch-output "${MISMATCH_PATH}"
)

if [[ -n "${GSM8K_ROLLOUT_AUDIT_STEP:-}" ]]; then
  CMD+=(--step "${GSM8K_ROLLOUT_AUDIT_STEP}")
fi

sol_msg "Recomputing GSM8K modern rewards from ${INPUT_PATH}"
"${CMD[@]}"
sol_msg "Summary: ${SUMMARY_PATH}"
sol_msg "Mismatches: ${MISMATCH_PATH}"
