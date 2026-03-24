#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_activate_env

"$(sol_python)" - <<'PY'
import importlib
import importlib.metadata
import sys

print(f"python={sys.version.split()[0]}")
for module_name, dist_name in [("verl", "verl"), ("ray", "ray"), ("vllm", "vllm")]:
    module = importlib.import_module(module_name)
    version = importlib.metadata.version(dist_name)
    print(f"{module_name}={version} ({getattr(module, '__file__', '<namespace>')})")
PY
