#!/usr/bin/env bash
# Config provenance:
# - Semantic baseline: modern GSM8K two-reward GDPO (`correct_reward + format_reward`).
# - Runtime profile: debug_safe.
# - Entry type: LEGACY debug-safe GSM8K wrapper retained for historical verification reproductions.
# - Public status: archived compatibility path; prefer `run_gdpo_gsm8k_modern_fit_debug.sh` for the active GSM8K debug/preflight flow.
# - Upstream anchors: external/verl/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh and external/verl/examples/gdpo_trainer/run_qwen1_5b_gdpo.sh
# - Closest prior working local config: /Users/god/Documents/VERL_GRPO/scripts/sol/run_grpo_debug_validation.sh
# - Historical failure reference: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_error_catalog.md
# - Required manual gate before edits: /Users/god/Documents/VERL_GRPO/docs/sol_rl_fit_config_creation_checklist.md
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

[[ -f "${GSM8K_MODERN_DIR}/train.parquet" ]] || sol_fail "Missing ${GSM8K_MODERN_DIR}/train.parquet. Run scripts/sol/prepare_gsm8k_modern.sh first."
[[ -f "${GSM8K_MODERN_DIR}/test.parquet" ]] || sol_fail "Missing ${GSM8K_MODERN_DIR}/test.parquet. Run scripts/sol/prepare_gsm8k_modern.sh first."

RUN_TAG="${RUN_TAG:-$(sol_timestamp)}"
EXPERIMENT_NAME="${GSM8K_MODERN_EXPERIMENT_NAME:-qwen2_5_0_5b_gdpo_gsm8k_modern_debug}"
RUN_ROOT="${OUTPUT_ROOT}/gsm8k_modern/debug/gdpo/${RUN_TAG}"
LOCAL_CKPT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/${RUN_TAG}"
PROJECT_NAME="${GSM8K_MODERN_PROJECT_NAME:-${SOL_PROJECT_NAME}_gdpo_gsm8k_modern}"
MODEL_PATH="${GSM8K_MODERN_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
ROLLOUT_DATA_DIR="${GSM8K_MODERN_ROLLOUT_DATA_DIR:-}"
N_GPUS_PER_NODE="${GSM8K_MODERN_N_GPUS_PER_NODE:-1}"
NNODES="${GSM8K_MODERN_NNODES:-1}"
TOTAL_GPUS=$((N_GPUS_PER_NODE * NNODES))
ROLLOUT_N="${GSM8K_MODERN_ROLLOUT_N:-4}"
PPO_MINI_BATCH_SIZE="${GSM8K_MODERN_PPO_MINI_BATCH_SIZE:-4}"
RUNTIME_PROFILE="${GSM8K_MODERN_RUNTIME_PROFILE:-debug_safe}"
RUN_CLASSIFICATION="${GSM8K_MODERN_RUN_CLASSIFICATION:-integration-smoke}"
POST_RUN_AUDIT_CMD='python scripts/sol/verify_gsm8k_modern_baseline.py artifact-audit --input-path "$GSM8K_MODERN_ROLLOUT_DATA_DIR" --artifact-type rollout'

NORMALIZED_PPO_MINI_BATCH_SIZE="$(sol_resolve_normalized_actor_mini_batch "${TOTAL_GPUS}" "${ROLLOUT_N}" "${PPO_MINI_BATCH_SIZE}")"
ACTOR_PPO_MICRO_BATCH_SIZE="${GSM8K_MODERN_PPO_MICRO_BATCH_SIZE:-1}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE="${GSM8K_MODERN_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE:-1}"
REF_LOGPROB_MICRO_BATCH_SIZE="${GSM8K_MODERN_REF_LOGPROB_MICRO_BATCH_SIZE:-1}"

sol_validate_micro_batch_divides_normalized \
  "${NORMALIZED_PPO_MINI_BATCH_SIZE}" "${ACTOR_PPO_MICRO_BATCH_SIZE}" "actor.ppo_micro_batch_size_per_gpu"
sol_validate_micro_batch_divides_normalized \
  "${NORMALIZED_PPO_MINI_BATCH_SIZE}" "${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}" "rollout.log_prob_micro_batch_size_per_gpu"
sol_validate_micro_batch_divides_normalized \
  "${NORMALIZED_PPO_MINI_BATCH_SIZE}" "${REF_LOGPROB_MICRO_BATCH_SIZE}" "ref.log_prob_micro_batch_size_per_gpu"

sol_prepare_tracking_paths "${PROJECT_NAME}" "${EXPERIMENT_NAME}" "${RUN_TAG}"
sol_prepare_gdpo_saturation_event_log_path

mkdir -p "${RUN_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${RUN_ROOT}"

sol_msg "Starting LEGACY modern 2-reward GSM8K GDPO debug run."
sol_msg "Preferred active GSM8K flow: fit_debug -> fit_2gpu_smoke -> fit_2gpu."
sol_msg "Use scripts/sol/run_gdpo_gsm8k_modern_fit_debug.sh for the active fast preflight path."
sol_msg "Dataset dir: ${GSM8K_MODERN_DIR}"
sol_msg "Model path: ${MODEL_PATH}"
sol_msg "Run root: ${RUN_ROOT}"
sol_msg "Checkpoint dir: ${LOCAL_CKPT_DIR}"
sol_msg "TensorBoard dir: ${TENSORBOARD_DIR}"
sol_msg "File logger path: ${VERL_FILE_LOGGER_PATH}"
sol_msg "GDPO saturation event log: ${GDPO_SATURATION_EVENT_LOG_PATH}"
sol_log_run_contract \
  "gsm8k_modern_two_reward_gdpo" \
  "${RUNTIME_PROFILE}" \
  "${RUN_CLASSIFICATION}" \
  "external/verl/examples/grpo_trainer/run_qwen2_5-3b_gsm8k_grpo_lora.sh + external/verl/examples/gdpo_trainer/run_qwen1_5b_gdpo.sh" \
  "/Users/god/Documents/VERL_GRPO/scripts/sol/run_grpo_debug_validation.sh" \
  "${POST_RUN_AUDIT_CMD}"
sol_log_batch_math \
  "${TOTAL_GPUS}" \
  "${ROLLOUT_N}" \
  "${PPO_MINI_BATCH_SIZE}" \
  "${NORMALIZED_PPO_MINI_BATCH_SIZE}" \
  "${ACTOR_PPO_MICRO_BATCH_SIZE}" \
  "${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}" \
  "${REF_LOGPROB_MICRO_BATCH_SIZE}"
if [[ -n "${ROLLOUT_DATA_DIR}" ]]; then
  mkdir -p "${ROLLOUT_DATA_DIR}"
  sol_msg "Rollout data dir: ${ROLLOUT_DATA_DIR}"
fi

cmd=(
  "$(sol_python)" -m verl.trainer.main_ppo
  "algorithm.adv_estimator=gdpo"
  "algorithm.gdpo_baseline_mode=${GSM8K_MODERN_GDPO_BASELINE_MODE:-upstream}"
  "algorithm.gdpo_reward_keys=[\"correct_reward\",\"format_reward\"]"
  "trainer.val_before_train=False"
  "data.train_files=${GSM8K_MODERN_DIR}/train.parquet"
  "data.val_files=${GSM8K_MODERN_DIR}/test.parquet"
  "data.train_batch_size=${GSM8K_MODERN_TRAIN_BATCH_SIZE:-4}"
  "data.val_batch_size=${GSM8K_MODERN_VAL_BATCH_SIZE:-4}"
  "data.dataloader_num_workers=0"
  "data.max_prompt_length=${GSM8K_MODERN_MAX_PROMPT_LENGTH:-512}"
  "data.max_response_length=${GSM8K_MODERN_MAX_RESPONSE_LENGTH:-256}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "+actor_rollout_ref.model.override_config.attn_implementation=eager"
  "actor_rollout_ref.actor.optim.lr=${GSM8K_MODERN_LR:-1e-6}"
  "actor_rollout_ref.model.use_remove_padding=False"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.prompt_length=${GSM8K_MODERN_MAX_PROMPT_LENGTH:-512}"
  "actor_rollout_ref.rollout.response_length=${GSM8K_MODERN_MAX_RESPONSE_LENGTH:-256}"
  "actor_rollout_ref.rollout.enforce_eager=${GSM8K_MODERN_ROLLOUT_ENFORCE_EAGER:-True}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GSM8K_MODERN_GPU_MEMORY_UTILIZATION:-0.15}"
  "actor_rollout_ref.rollout.max_num_seqs=${GSM8K_MODERN_MAX_NUM_SEQS:-8}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${GSM8K_MODERN_MAX_NUM_BATCHED_TOKENS:-4096}"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.skip_tokenizer_init=True"
  "actor_rollout_ref.rollout.agent.num_workers=1"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MICRO_BATCH_SIZE}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "+actor_rollout_ref.ref.model.override_config.attn_implementation=eager"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "reward.custom_reward_function.path=${UPSTREAM_VERL_DIR}/verl/utils/reward_score/gsm8k_modern_two_reward.py"
  "reward.custom_reward_function.name=compute_score"
  "reward.reward_manager.name=gdpo"
  "trainer.critic_warmup=0"
  "trainer.logger=[\"console\",\"tensorboard\",\"file\"]"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}"
  "trainer.save_freq=${GSM8K_MODERN_SAVE_FREQ:-1}"
  "trainer.test_freq=${GSM8K_MODERN_TEST_FREQ:--1}"
  "trainer.total_training_steps=${GSM8K_MODERN_TOTAL_TRAINING_STEPS:-5}"
  "trainer.total_epochs=${GSM8K_MODERN_TOTAL_EPOCHS:-1}"
  "trainer.default_local_dir=${LOCAL_CKPT_DIR}"
)

if [[ -n "${ROLLOUT_DATA_DIR}" ]]; then
  cmd+=("trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}")
fi

"${cmd[@]}" "$@"
