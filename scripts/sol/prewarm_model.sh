#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:-Qwen/Qwen2.5-0.5B-Instruct}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_activate_env

sol_msg "Prewarming Hugging Face caches for ${MODEL_NAME}."
"$(sol_python)" - "${MODEL_NAME}" <<'PY'
import sys
import transformers

model_name = sys.argv[1]
transformers.pipeline("text-generation", model=model_name)
print(f"prewarmed={model_name}")
PY
