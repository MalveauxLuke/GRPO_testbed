#!/usr/bin/env bash
# Example only. This file is intentionally verbose so someone can understand the
# current GSM8K shape without reverse-engineering the active wrappers.
#
# Safe rule:
# - learn from this file
# - for real use, prefer configs/experiments/gsm8k/*.env plus
#   scripts/sol/submit_experiment.sh
#
# This example mirrors the current 75-step GSM8K saturation run.
#
# Resource choices:
# - partition/qos public: this is not a 15-minute debug job
# - 2 GPUs: matches the active constrained-hardware GSM8K path
# - 16 CPUs: enough for dataloader, Ray, and logging overhead without being
#   materially different from the current supported entrypoint
# - 128G RAM: deliberate; this path already proved it needs explicit host memory
# - 6 hours: enough for the capped 75-step saturation check
#SBATCH --job-name=example-gsm8k-75step
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${PROJECT_ROOT}/scripts/sol/common_env.sh"

# This is the current model and reward contract we are using for GSM8K.
export GSM8K_MODERN_MODEL_PATH="${GSM8K_MODERN_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
export GSM8K_MODERN_EXPERIMENT_NAME="${GSM8K_MODERN_EXPERIMENT_NAME:-qwen2_5_1_5b_instruct_gdpo_gsm8k_modern_fit_2gpu_answer_tag_saturation_check_v20260404}"
export GSM8K_MODERN_PROJECT_NAME="${GSM8K_MODERN_PROJECT_NAME:-${SOL_PROJECT_NAME}_gdpo_gsm8k_modern_answer_tag_1_5b_instruct_saturation_check}"

# These are the cap-only knobs that make this a saturation probe rather than a
# full run.
export GSM8K_MODERN_TOTAL_TRAINING_STEPS="${GSM8K_MODERN_TOTAL_TRAINING_STEPS:-75}"
export GSM8K_MODERN_TEST_FREQ="${GSM8K_MODERN_TEST_FREQ:-10}"
export GSM8K_MODERN_SAVE_FREQ="${GSM8K_MODERN_SAVE_FREQ:-25}"

# These are the proven SOL-safe runtime knobs for this path. Leave them alone
# unless you are intentionally revalidating runtime fit.
export GSM8K_MODERN_N_GPUS_PER_NODE="${GSM8K_MODERN_N_GPUS_PER_NODE:-2}"
export GSM8K_MODERN_USE_SHM="${GSM8K_MODERN_USE_SHM:-False}"
export GSM8K_MODERN_GPU_MEMORY_UTILIZATION="${GSM8K_MODERN_GPU_MEMORY_UTILIZATION:-0.35}"
export GSM8K_MODERN_MAX_NUM_SEQS="${GSM8K_MODERN_MAX_NUM_SEQS:-64}"
export GSM8K_MODERN_SKIP_TOKENIZER_INIT="${GSM8K_MODERN_SKIP_TOKENIZER_INIT:-True}"
export GSM8K_MODERN_ROLLOUT_ENFORCE_EAGER="${GSM8K_MODERN_ROLLOUT_ENFORCE_EAGER:-True}"
export GSM8K_MODERN_MAX_MODEL_LEN="${GSM8K_MODERN_MAX_MODEL_LEN:-1536}"
export GSM8K_MODERN_MAX_NUM_BATCHED_TOKENS="${GSM8K_MODERN_MAX_NUM_BATCHED_TOKENS:-1536}"
export GSM8K_MODERN_ENABLE_CHUNKED_PREFILL="${GSM8K_MODERN_ENABLE_CHUNKED_PREFILL:-False}"
export GSM8K_MODERN_MAX_PARALLEL_LOADING_WORKERS="${GSM8K_MODERN_MAX_PARALLEL_LOADING_WORKERS:-1}"
export GSM8K_MODERN_AGENT_NUM_WORKERS="${GSM8K_MODERN_AGENT_NUM_WORKERS:-1}"
export GSM8K_MODERN_TP_SIZE="${GSM8K_MODERN_TP_SIZE:-1}"
export GSM8K_MODERN_ACTOR_PARAM_OFFLOAD="${GSM8K_MODERN_ACTOR_PARAM_OFFLOAD:-True}"
export GSM8K_MODERN_ACTOR_OPTIMIZER_OFFLOAD="${GSM8K_MODERN_ACTOR_OPTIMIZER_OFFLOAD:-True}"
export GSM8K_MODERN_REF_PARAM_OFFLOAD="${GSM8K_MODERN_REF_PARAM_OFFLOAD:-True}"
export GSM8K_MODERN_VAL_MAX_SAMPLES="${GSM8K_MODERN_VAL_MAX_SAMPLES:-64}"

# This is optional but extremely useful because it makes rollout audit easy.
export GSM8K_MODERN_ROLLOUT_DATA_DIR="${GSM8K_MODERN_ROLLOUT_DATA_DIR:-/scratch/$USER/verl-grpo/rollout_dumps/example_gsm8k_75step_$(date +%Y%m%d-%H%M%S)}"

"${PROJECT_ROOT}/scripts/sol/run_gdpo_gsm8k_modern_fit_2gpu.sh" "$@"
