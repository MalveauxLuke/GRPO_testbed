#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_ensure_runtime_dirs
sol_activate_env

INPUT_PATH="${1:-${GSM8K_MODERN_ROLLOUT_DATA_DIR:-}}"
[[ -n "${INPUT_PATH}" ]] || sol_fail "Pass a rollout dump path as the first argument or set GSM8K_MODERN_ROLLOUT_DATA_DIR."

CMD=(
  "$(sol_python)"
  "${PROJECT_ROOT}/scripts/sol/audit_gsm8k_modern_rewards.py"
  sample-audit
  --input-path "${INPUT_PATH}"
  --input-type rollout
  --sample-count "${GSM8K_ROLLOUT_SAMPLE_COUNT:-10}"
  --seed "${GSM8K_ROLLOUT_SAMPLE_SEED:-0}"
)

if [[ "${GSM8K_ROLLOUT_ONLY_MISMATCHES:-0}" == "1" ]]; then
  CMD+=(--only-mismatches)
fi
if [[ -n "${GSM8K_ROLLOUT_BEHAVIOR_BUCKET:-}" ]]; then
  CMD+=(--behavior-bucket "${GSM8K_ROLLOUT_BEHAVIOR_BUCKET}")
fi
if [[ -n "${GSM8K_ROLLOUT_AUDIT_STEP:-}" ]]; then
  CMD+=(--step "${GSM8K_ROLLOUT_AUDIT_STEP}")
fi
if [[ -n "${GSM8K_ROLLOUT_SAMPLE_OUTPUT_JSONL:-}" ]]; then
  CMD+=(--output-jsonl "${GSM8K_ROLLOUT_SAMPLE_OUTPUT_JSONL}")
fi

"${CMD[@]}"
