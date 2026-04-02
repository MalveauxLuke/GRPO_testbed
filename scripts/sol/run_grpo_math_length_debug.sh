#!/usr/bin/env bash
# Config provenance:
# - Semantic baseline: DeepScaleR-style boxed-answer math reward (`correct_reward + length_reward`).
# - Runtime profile: debug_safe.
# - Entry type: GRPO debug-safe dispatcher for the shared math-length wrapper.
# - Upstream anchor inherited through: /Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh
# - Local parent config: /Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh
# - Historical failure reference: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md
# - Required manual gate before edits: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_math_length_common.sh" grpo debug "$@"
