#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

DATASET_TIER="${MATH_LENGTH_EVAL_DATASET_TIER:-production}"
case "${DATASET_TIER}" in
  debug)
    DEFAULT_DATASET_PATH="${DEEPSCALER_DEBUG_DIR}/test.parquet"
    DEFAULT_LENGTH_LIMIT="${MATH_LENGTH_EVAL_LENGTH_LIMIT_TOKENS:-1024}"
    ;;
  production)
    DEFAULT_DATASET_PATH="${DEEPSCALER_DIR}/test.parquet"
    DEFAULT_LENGTH_LIMIT="${MATH_LENGTH_EVAL_LENGTH_LIMIT_TOKENS:-4000}"
    ;;
  *)
    sol_fail "Unsupported MATH_LENGTH_EVAL_DATASET_TIER='${DATASET_TIER}'. Expected debug or production."
    ;;
esac

DATASET_PATH="${MATH_LENGTH_EVAL_DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
[[ -f "${DATASET_PATH}" ]] || sol_fail "Missing evaluation parquet at ${DATASET_PATH}. Run scripts/sol/prepare_deepscaler_math.sh first."

RUN_TAG="${RUN_TAG:-$(sol_timestamp)}"
MODEL_PATH="${MATH_LENGTH_EVAL_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
OUTPUT_PATH="${MATH_LENGTH_EVAL_OUTPUT_PATH:-${OUTPUT_ROOT}/eval/math_length/${RUN_TAG}.json}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

sol_msg "Evaluating DeepScaleR-style math-length model."
sol_msg "Dataset path: ${DATASET_PATH}"
sol_msg "Model path: ${MODEL_PATH}"
sol_msg "Output path: ${OUTPUT_PATH}"

"$(sol_python)" "${SCRIPT_DIR}/eval_math_length.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --samples-per-prompt "${MATH_LENGTH_EVAL_SAMPLES_PER_PROMPT:-16}" \
  --temperature "${MATH_LENGTH_EVAL_TEMPERATURE:-0.6}" \
  --top-p "${MATH_LENGTH_EVAL_TOP_P:-0.95}" \
  --max-tokens "${MATH_LENGTH_EVAL_MAX_TOKENS:-32768}" \
  --length-limit-tokens "${DEFAULT_LENGTH_LIMIT}" \
  --tensor-parallel-size "${MATH_LENGTH_EVAL_TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${MATH_LENGTH_EVAL_GPU_MEMORY_UTILIZATION:-0.75}" \
  "$@"
