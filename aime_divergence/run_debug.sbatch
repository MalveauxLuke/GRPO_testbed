#!/usr/bin/env bash
# Inference-only AIME divergence debug run.
#
# Reuses this repo's established SOL conventions:
# - source scripts/sol/common_env.sh
# - activate the shared verl-grpo-sol mamba env
# - use /scratch/$USER/verl-grpo for HF/vLLM/tmp/output caches
# - tee stdout/stderr to the scratch Slurm log root
#
# TODO(user): verify whether SOL exposes an A100-80GB-specific constraint. The
# existing repo sbatches use --gpus-per-node without GPU-type constraints, so
# this debug job follows that established pattern.
#SBATCH --job-name=aime-divergence-debug
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR}}"
else
  SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SBATCH_DIR}/.." && pwd)}"
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/sol/common_env.sh"

sol_ensure_runtime_dirs

JOB_LOG="${LOG_ROOT}/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"
exec > >(tee -a "${JOB_LOG}") 2>&1

sol_activate_env

RUN_TAG="${AIME_DIVERGENCE_RUN_TAG:-debug_$(sol_timestamp)}"
export AIME_DIVERGENCE_OUTPUT_DIR="${AIME_DIVERGENCE_OUTPUT_DIR:-${OUTPUT_ROOT}/aime_divergence/${RUN_TAG}}"
export AIME_DIVERGENCE_OUTPUT_PATH="${AIME_DIVERGENCE_OUTPUT_PATH:-${AIME_DIVERGENCE_OUTPUT_DIR}/aime_rollouts_debug.json}"
export AIME_DIVERGENCE_TENSORBOARD="${AIME_DIVERGENCE_TENSORBOARD:-1}"
export AIME_DIVERGENCE_TENSORBOARD_DIR="${AIME_DIVERGENCE_TENSORBOARD_DIR:-${AIME_DIVERGENCE_OUTPUT_DIR}/tensorboard}"

mkdir -p "${AIME_DIVERGENCE_OUTPUT_DIR}"

sol_msg "Scratch log: ${JOB_LOG}"
sol_msg "AIME divergence output: ${AIME_DIVERGENCE_OUTPUT_PATH}"
sol_msg "AIME divergence TensorBoard: ${AIME_DIVERGENCE_TENSORBOARD_DIR}"
sol_msg "Running inference-only AIME divergence debug pipeline from ${PROJECT_ROOT}."

"$(sol_python)" - <<'PY'
import importlib.metadata

for package_name in ("vllm", "torch", "datasets", "transformers", "math-verify", "tensorboard"):
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "<missing>"
    print(f"[sol-setup] package {package_name}={version}")
PY

cd "${PROJECT_ROOT}"

"$(sol_python)" -m aime_divergence.run_debug \
  --model "${AIME_DIVERGENCE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}" \
  --output-path "${AIME_DIVERGENCE_OUTPUT_PATH}" \
  --samples-per-prompt "${AIME_DIVERGENCE_SAMPLES:-8}" \
  --max-problems "${AIME_DIVERGENCE_MAX_PROBLEMS:-0}" \
  --temperature "${AIME_DIVERGENCE_TEMPERATURE:-1.0}" \
  --top-p "${AIME_DIVERGENCE_TOP_P:-0.95}" \
  --max-tokens "${AIME_DIVERGENCE_MAX_TOKENS:-16384}" \
  --max-model-len "${AIME_DIVERGENCE_MAX_MODEL_LEN:-16384}" \
  --tensor-parallel-size "${AIME_DIVERGENCE_TP_SIZE:-1}" \
  --gpu-memory-utilization "${AIME_DIVERGENCE_GPU_MEMORY_UTILIZATION:-0.90}" \
  --dtype "${AIME_DIVERGENCE_DTYPE:-bfloat16}" \
  --seed "${AIME_DIVERGENCE_SEED:-0}"

sol_msg "AIME divergence debug run finished."
sol_msg "Output JSON: ${AIME_DIVERGENCE_OUTPUT_PATH}"
