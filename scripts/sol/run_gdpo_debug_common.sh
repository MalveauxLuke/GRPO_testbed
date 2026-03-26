#!/usr/bin/env bash
set -euo pipefail

GDPO_BASELINE_MODE="${1:-}"
if [[ -z "${GDPO_BASELINE_MODE}" ]]; then
  printf '[sol-setup] ERROR: Missing GDPO baseline mode. Expected one of: upstream, nvlabs_reference.\n' >&2
  exit 1
fi
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

case "${GDPO_BASELINE_MODE}" in
  upstream|nvlabs_reference) ;;
  *)
    sol_fail "Unsupported GDPO baseline mode '${GDPO_BASELINE_MODE}'. Expected 'upstream' or 'nvlabs_reference'."
    ;;
esac

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

[[ -f "${RLLA_4K_DIR}/train.parquet" ]] || sol_fail "Missing ${RLLA_4K_DIR}/train.parquet. Run scripts/sol/prepare_rlla_toolrl.sh first."
[[ -f "${RLLA_4K_DIR}/test.parquet" ]] || sol_fail "Missing ${RLLA_4K_DIR}/test.parquet. Run scripts/sol/prepare_rlla_toolrl.sh first."

RUN_TAG="${RUN_TAG:-$(sol_timestamp)}"
EXPERIMENT_NAME="${GDPO_EXPERIMENT_NAME:-qwen2_5_0_5b_gdpo_${GDPO_BASELINE_MODE}_sol_debug}"
RUN_ROOT="${OUTPUT_ROOT}/gdpo/${GDPO_BASELINE_MODE}/${RUN_TAG}"
LOCAL_CKPT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/${RUN_TAG}"
GDPO_PROJECT_NAME="${GDPO_PROJECT_NAME:-${SOL_PROJECT_NAME}_gdpo}"

sol_prepare_tracking_paths "${GDPO_PROJECT_NAME}" "${EXPERIMENT_NAME}" "${RUN_TAG}"

mkdir -p "${RUN_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${RUN_ROOT}"

sol_msg "Starting short 1-GPU GDPO validation run."
sol_msg "Baseline mode: ${GDPO_BASELINE_MODE}"
sol_msg "Using the SOL-compatible debug validation profile with reduced memory pressure and non-FlashAttention fallbacks."
sol_msg "Using a ToolRL-compatible prompt cap so the rlla_4k prompts are not filtered out at dataset load time."
sol_msg "Run root: ${RUN_ROOT}"
sol_msg "Checkpoint dir: ${LOCAL_CKPT_DIR}"
sol_msg "TensorBoard dir: ${TENSORBOARD_DIR}"
sol_msg "File logger path: ${VERL_FILE_LOGGER_PATH}"

"$(sol_python)" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gdpo \
  algorithm.gdpo_baseline_mode="${GDPO_BASELINE_MODE}" \
  algorithm.gdpo_reward_keys='["accuracy_reward","format_reward"]' \
  trainer.val_before_train=False \
  data.train_files="${RLLA_4K_DIR}/train.parquet" \
  data.val_files="${RLLA_4K_DIR}/test.parquet" \
  data.train_batch_size=2 \
  data.val_batch_size=2 \
  data.dataloader_num_workers=0 \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.prompt_length=2048 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.10 \
  actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.skip_tokenizer_init=True \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.ref.model.override_config.attn_implementation=eager \
  algorithm.kl_ctrl.kl_coef=0.001 \
  reward.custom_reward_function.path="${UPSTREAM_VERL_DIR}/verl/utils/reward_score/rlla.py" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.name=gdpo \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard","file"]' \
  trainer.project_name="${GDPO_PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=5 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${LOCAL_CKPT_DIR}" \
  "$@"
