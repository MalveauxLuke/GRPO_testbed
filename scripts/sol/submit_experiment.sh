#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/experiment_common.sh"

usage() {
  cat <<'EOF'
Usage:
  scripts/sol/submit_experiment.sh <config> [--dry-run|--submit]

Examples:
  scripts/sol/submit_experiment.sh configs/experiments/gsm8k/debug_answer_tag_1p5b_instruct.env --dry-run
  scripts/sol/submit_experiment.sh configs/experiments/gsm8k/saturation_75step_answer_tag_1p5b_instruct.env --submit
EOF
}

CONFIG_ARG="${1:-}"
MODE="${2:---dry-run}"

if [[ -z "${CONFIG_ARG}" || "${CONFIG_ARG}" == "-h" || "${CONFIG_ARG}" == "--help" ]]; then
  usage
  exit 0
fi

case "${MODE}" in
  --dry-run|--submit) ;;
  *)
    sol_fail "Unsupported mode ${MODE}. Expected --dry-run or --submit."
    ;;
esac

CONFIG_PATH="$(sol_resolve_experiment_config_path "${CONFIG_ARG}")"
sol_load_experiment_config "${CONFIG_PATH}"
sol_apply_experiment_rollout_default
sol_print_experiment_overview

if [[ "${MODE}" == "--dry-run" ]]; then
  exec "${PROJECT_ROOT}/${EXPERIMENT_DRY_RUN_TARGET}" --dry-run
fi

sol_msg "Submitting via sbatch."
exec sbatch "${PROJECT_ROOT}/${EXPERIMENT_SUBMIT_TARGET}"
