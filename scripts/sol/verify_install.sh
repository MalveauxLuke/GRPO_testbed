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
import os
import sys

print(f"python={sys.version.split()[0]}")
expected_verl_root = os.environ.get("UPSTREAM_VERL_DIR")
expected_env_root = os.path.realpath(os.environ.get("CONDA_PREFIX", ""))

for module_name, dist_name in [
    ("verl", "verl"),
    ("ray", "ray"),
    ("vllm", "vllm"),
    ("datasets", "datasets"),
    ("numpy", "numpy"),
    ("numba", "numba"),
    ("tensorboard", "tensorboard"),
]:
    module = importlib.import_module(module_name)
    version = importlib.metadata.version(dist_name)
    print(f"{module_name}={version} ({getattr(module, '__file__', '<namespace>')})")

    module_path = os.path.realpath(getattr(module, "__file__", ""))
    if module_name == "verl" and expected_verl_root:
        expected_prefix = os.path.realpath(expected_verl_root)
        if not module_path.startswith(expected_prefix + os.sep):
            raise SystemExit(
                f"verl must import from the editable checkout under {expected_prefix}, "
                f"but imported from {module_path}"
            )
    elif expected_env_root and module_path and not module_path.startswith(expected_env_root + os.sep):
        raise SystemExit(
            f"{module_name} must import from the active env under {expected_env_root}, "
            f"but imported from {module_path}"
        )

numpy_version = importlib.metadata.version("numpy")
numpy_major = int(numpy_version.split(".", 1)[0])
if numpy_major >= 2:
    raise SystemExit(f"numpy must stay below 2.0.0 for upstream verl compatibility; found {numpy_version}")

tb_writer = importlib.import_module("torch.utils.tensorboard")
print(f"torch.utils.tensorboard={getattr(tb_writer, '__file__', '<namespace>')}")

math_verify_metric = importlib.import_module("math_verify.metric")
print(f"math_verify.metric={getattr(math_verify_metric, '__file__', '<namespace>')}")

deepscaler_reward = importlib.import_module("verl.utils.reward_score.deepscaler_math_length")
print(f"deepscaler_math_length={getattr(deepscaler_reward, '__file__', '<namespace>')}")
PY
