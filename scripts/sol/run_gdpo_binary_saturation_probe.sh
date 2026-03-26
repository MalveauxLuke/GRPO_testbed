#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

PROBE_DATASET_MODE="${GDPO_SATURATION_PROBE_DATASET_MODE:-unfiltered}"
case "${PROBE_DATASET_MODE}" in
  unfiltered)
    PROBE_DATASET_DIR="${GSM8K_GDPO_PROBE_DIR}"
    ;;
  hard)
    PROBE_DATASET_DIR="${GSM8K_GDPO_PROBE_HARD_DIR}"
    ;;
  *)
    sol_fail "Unsupported GDPO saturation probe dataset mode '${PROBE_DATASET_MODE}'. Expected 'unfiltered' or 'hard'."
    ;;
esac

[[ -f "${PROBE_DATASET_DIR}/train.parquet" ]] || sol_fail "Missing ${PROBE_DATASET_DIR}/train.parquet. Run scripts/sol/prepare_gsm8k_gdpo_saturation_probe.sh first."
[[ -f "${PROBE_DATASET_DIR}/test.parquet" ]] || sol_fail "Missing ${PROBE_DATASET_DIR}/test.parquet. Run scripts/sol/prepare_gsm8k_gdpo_saturation_probe.sh first."

RUN_TAG="${RUN_TAG:-$(sol_timestamp)}"
EXPERIMENT_NAME="${GDPO_BINARY_PROBE_EXPERIMENT_NAME:-qwen2_5_0_5b_gdpo_binary_saturation_probe}"
RUN_ROOT="${OUTPUT_ROOT}/gdpo_binary_probe/${PROBE_DATASET_MODE}/${RUN_TAG}"
LOCAL_CKPT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/${RUN_TAG}"
GDPO_BINARY_PROBE_PROJECT_NAME="${GDPO_BINARY_PROBE_PROJECT_NAME:-${SOL_PROJECT_NAME}_gdpo_binary_probe}"
PROBE_MODEL_PATH="${GDPO_BINARY_PROBE_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

sol_prepare_tracking_paths "${GDPO_BINARY_PROBE_PROJECT_NAME}" "${EXPERIMENT_NAME}" "${RUN_TAG}"
sol_prepare_gdpo_saturation_event_log_path

mkdir -p "${RUN_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${RUN_ROOT}"

sol_msg "Starting short 1-GPU GDPO binary saturation probe."
sol_msg "Dataset mode: ${PROBE_DATASET_MODE}"
sol_msg "Probe dataset dir: ${PROBE_DATASET_DIR}"
sol_msg "Run root: ${RUN_ROOT}"
sol_msg "Checkpoint dir: ${LOCAL_CKPT_DIR}"
sol_msg "TensorBoard dir: ${TENSORBOARD_DIR}"
sol_msg "File logger path: ${VERL_FILE_LOGGER_PATH}"
sol_msg "GDPO saturation event log: ${GDPO_SATURATION_EVENT_LOG_PATH}"

"$(sol_python)" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gdpo \
  algorithm.gdpo_baseline_mode=upstream \
  algorithm.gdpo_reward_keys='["correct_reward","format_reward"]' \
  trainer.val_before_train=False \
  data.train_files="${PROBE_DATASET_DIR}/train.parquet" \
  data.val_files="${PROBE_DATASET_DIR}/test.parquet" \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=256 \
  data.max_response_length=192 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path="${PROBE_MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.prompt_length=256 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.10 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.skip_tokenizer_init=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.ref.model.override_config.attn_implementation=eager \
  algorithm.kl_ctrl.kl_coef=0.001 \
  reward.custom_reward_function.path="${UPSTREAM_VERL_DIR}/verl/utils/reward_score/gdpo_binary_probe.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=gdpo \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard","file"]' \
  trainer.project_name="${GDPO_BINARY_PROBE_PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=5 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${LOCAL_CKPT_DIR}" \
  "$@"
