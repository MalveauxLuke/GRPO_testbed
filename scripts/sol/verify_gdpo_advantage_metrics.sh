#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_ensure_runtime_dirs
sol_activate_env

RUN_LABEL="${GDPO_ADVANTAGE_PROOF_LABEL:-gdpo_advantage_metric_proof_$(sol_timestamp)}"
OUTPUT_DIR="${OUTPUT_ROOT}/gdpo_advantage_metric_proof"
OUTPUT_JSON="${GDPO_ADVANTAGE_PROOF_OUTPUT_JSON:-${OUTPUT_DIR}/${RUN_LABEL}.json}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  "$(sol_python)"
  "${PROJECT_ROOT}/scripts/sol/verify_gdpo_advantage_metrics.py"
  --output-json "${OUTPUT_JSON}"
)

if [[ "${GDPO_ADVANTAGE_PROOF_QUIET:-0}" == "1" ]]; then
  CMD+=(--quiet)
fi

sol_msg "Running GDPO advantage metric proof harness from the actual vendored verl code path."
sol_msg "Output JSON: ${OUTPUT_JSON}"
"${CMD[@]}"
sol_msg "Proof report: ${OUTPUT_JSON}"
