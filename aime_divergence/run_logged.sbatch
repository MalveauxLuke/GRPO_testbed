#!/usr/bin/env bash
# Inference-only AIME divergence logged run.
#
# Reuses this repo's established SOL conventions:
# - source scripts/sol/common_env.sh
# - activate the shared verl-grpo-sol mamba env
# - use /scratch/$USER/verl-grpo for HF/vLLM/tmp/output caches
# - tee stdout/stderr to the scratch Slurm log root
# - disable FlashInfer paths that are unstable on SOL for this workflow
#SBATCH --job-name=aime-divergence-logged
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
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

RUN_TAG="${AIME_DIVERGENCE_RUN_TAG:-logged_$(sol_timestamp)}"
export AIME_DIVERGENCE_OUTPUT_DIR="${AIME_DIVERGENCE_OUTPUT_DIR:-${OUTPUT_ROOT}/aime_divergence/${RUN_TAG}}"
export AIME_DIVERGENCE_TENSORBOARD="${AIME_DIVERGENCE_TENSORBOARD:-1}"
export AIME_DIVERGENCE_TENSORBOARD_DIR="${AIME_DIVERGENCE_TENSORBOARD_DIR:-${AIME_DIVERGENCE_OUTPUT_DIR}/tensorboard}"
export AIME_DIVERGENCE_LOGPROBS="${AIME_DIVERGENCE_LOGPROBS:-50}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

mkdir -p "${AIME_DIVERGENCE_OUTPUT_DIR}"

sol_msg "Scratch log: ${JOB_LOG}"
sol_msg "AIME logged output dir: ${AIME_DIVERGENCE_OUTPUT_DIR}"
sol_msg "AIME logged TensorBoard: ${AIME_DIVERGENCE_TENSORBOARD_DIR}"
sol_msg "vLLM attention backend: ${VLLM_ATTENTION_BACKEND}"
sol_msg "vLLM flashinfer sampler enabled: ${VLLM_USE_FLASHINFER_SAMPLER}"
sol_msg "Running inference-only AIME logged pipeline from ${PROJECT_ROOT}."

"$(sol_python)" - <<'PY'
import importlib.metadata

for package_name in ("vllm", "torch", "datasets", "transformers", "math-verify", "numpy", "tensorboard"):
    try:
        version = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        version = "<missing>"
    print(f"[sol-setup] package {package_name}={version}")
PY

cd "${PROJECT_ROOT}"

"$(sol_python)" -m aime_divergence.run_logged \
  --model "${AIME_DIVERGENCE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}" \
  --output-dir "${AIME_DIVERGENCE_OUTPUT_DIR}" \
  --samples-per-prompt "${AIME_DIVERGENCE_SAMPLES:-8}" \
  --logprobs "${AIME_DIVERGENCE_LOGPROBS:-50}" \
  --max-problems "${AIME_DIVERGENCE_MAX_PROBLEMS:-0}" \
  --temperature "${AIME_DIVERGENCE_TEMPERATURE:-1.0}" \
  --top-p "${AIME_DIVERGENCE_TOP_P:-0.95}" \
  --max-tokens "${AIME_DIVERGENCE_MAX_TOKENS:-16384}" \
  --max-model-len "${AIME_DIVERGENCE_MAX_MODEL_LEN:-16384}" \
  --tensor-parallel-size "${AIME_DIVERGENCE_TP_SIZE:-1}" \
  --gpu-memory-utilization "${AIME_DIVERGENCE_GPU_MEMORY_UTILIZATION:-0.90}" \
  --dtype "${AIME_DIVERGENCE_DTYPE:-bfloat16}" \
  --seed "${AIME_DIVERGENCE_SEED:-0}"

sol_msg "AIME logged run finished."
sol_msg "Output dir: ${AIME_DIVERGENCE_OUTPUT_DIR}"
