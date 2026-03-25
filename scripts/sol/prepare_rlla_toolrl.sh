#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_activate_env

TMP_CLONE_DIR="$(mktemp -d "${TMPDIR}/toolrl-rlla-XXXXXX")"
cleanup() {
  rm -rf "${TMP_CLONE_DIR}"
}
trap cleanup EXIT

sol_msg "Fetching ToolRL rlla_4k parquet files from ${TOOLRL_REPO_URL}."
git clone --depth 1 --filter=blob:none --sparse "${TOOLRL_REPO_URL}" "${TMP_CLONE_DIR}/ToolRL" >/dev/null
(
  cd "${TMP_CLONE_DIR}/ToolRL"
  git sparse-checkout set dataset/rlla_4k >/dev/null
)

mkdir -p "${RLLA_4K_DIR}"
cp "${TMP_CLONE_DIR}/ToolRL/dataset/rlla_4k/train.parquet" "${RLLA_4K_DIR}/train.parquet"
cp "${TMP_CLONE_DIR}/ToolRL/dataset/rlla_4k/test.parquet" "${RLLA_4K_DIR}/test.parquet"

"$(sol_python)" - "${RLLA_4K_DIR}" <<'PY'
import importlib.util
import json
import os
import sys


def load_first_row(path):
    if importlib.util.find_spec("pyarrow") is not None:
        import pyarrow.parquet as pq

        row = pq.read_table(path).slice(0, 1).to_pylist()[0]
        return row

    if importlib.util.find_spec("pandas") is not None:
        import pandas as pd

        row = pd.read_parquet(path).iloc[0].to_dict()
        return row

    raise SystemExit("Need either pyarrow or pandas installed in the active env to validate parquet schema.")


def ensure_reward_model_ground_truth(row, path):
    reward_model = row.get("reward_model")
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)

    if not isinstance(reward_model, dict):
        raise SystemExit(f"{path} is missing a dict-like reward_model field in the first row: {type(reward_model)!r}")
    if "ground_truth" not in reward_model:
        raise SystemExit(f"{path} is missing reward_model.ground_truth in the first row.")


root = sys.argv[1]
for split in ("train", "test"):
    path = os.path.join(root, f"{split}.parquet")
    row = load_first_row(path)
    ensure_reward_model_ground_truth(row, path)
    keys = sorted(row.keys())
    print(f"{split}_parquet={path}")
    print(f"{split}_keys={keys}")
PY

sol_msg "Prepared:"
sol_msg "  ${RLLA_4K_DIR}/train.parquet"
sol_msg "  ${RLLA_4K_DIR}/test.parquet"
