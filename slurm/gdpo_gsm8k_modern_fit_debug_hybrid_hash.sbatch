#!/usr/bin/env bash
# Config provenance:
# - Semantic baseline: hybrid-hash GSM8K two-reward GDPO (`correct_reward + format_reward`).
# - Runtime profile: sol_fit_2gpu.
# - Entry type: short multi-step feasibility preflight for the hybrid `####` + tags fit-contract path.
# - Upstream anchor inherited through: /Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh
# - Local parent config: /Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh
# - Historical failure reference: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md
# - Required manual gate before edits: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md
# Submit with an optional account override:
#   sbatch -A grp_<your_group> slurm/gdpo_gsm8k_modern_fit_debug_hybrid_hash.sbatch
# Or:
#   SOL_ACCOUNT=grp_<your_group> sbatch ${SOL_ACCOUNT:+-A "$SOL_ACCOUNT"} slurm/gdpo_gsm8k_modern_fit_debug_hybrid_hash.sbatch
# Optional rollout dump capture:
#   GSM8K_MODERN_ROLLOUT_DATA_DIR=/scratch/$USER/verl-grpo/rollout_dumps/gsm8k_modern_fit_debug_hybrid_hash_$(date +%Y%m%d-%H%M%S) sbatch slurm/gdpo_gsm8k_modern_fit_debug_hybrid_hash.sbatch
# Expected data refresh before first run after reward changes:
#   scripts/sol/prepare_gsm8k_modern.sh
#SBATCH --job-name=verl-gdpo-gsm8k-modern-fit-debug-hybrid-hash
#SBATCH --partition=htc
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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

sol_msg "Scratch log: ${JOB_LOG}"
sol_msg "Submitting short multi-step hybrid-hash GSM8K fit debug preflight from ${PROJECT_ROOT}."
sol_msg "This entrypoint keeps the proven SOL-safe rollout knobs explicit to avoid the HF-path, tokenizer-init, and vLLM cache regressions we already hit."

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
export GSM8K_MODERN_TOTAL_TRAINING_STEPS="${GSM8K_MODERN_TOTAL_TRAINING_STEPS:-3}"
export GSM8K_MODERN_TOTAL_EPOCHS="${GSM8K_MODERN_TOTAL_EPOCHS:-1}"
export GSM8K_MODERN_SAVE_FREQ="${GSM8K_MODERN_SAVE_FREQ:-3}"
export GSM8K_MODERN_TEST_FREQ="${GSM8K_MODERN_TEST_FREQ:-1}"
export GSM8K_MODERN_VAL_BEFORE_TRAIN="${GSM8K_MODERN_VAL_BEFORE_TRAIN:-False}"
export GSM8K_MODERN_RUN_CLASSIFICATION="${GSM8K_MODERN_RUN_CLASSIFICATION:-feasibility-smoke}"
export GSM8K_MODERN_RUNTIME_PROFILE="${GSM8K_MODERN_RUNTIME_PROFILE:-sol_fit_2gpu}"
export GSM8K_MODERN_EXPERIMENT_NAME="${GSM8K_MODERN_EXPERIMENT_NAME:-qwen2_5_3b_gdpo_gsm8k_modern_fit_debug_hybrid_hash_v20260403}"
export GSM8K_MODERN_PROJECT_NAME="${GSM8K_MODERN_PROJECT_NAME:-${SOL_PROJECT_NAME}_gdpo_gsm8k_modern_fit_debug_hybrid_hash}"

"${PROJECT_ROOT}/scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh" "$@"
