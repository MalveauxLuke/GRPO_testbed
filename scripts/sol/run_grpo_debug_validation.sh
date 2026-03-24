#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

[[ -f "${GSM8K_DIR}/train.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/train.parquet. Run scripts/sol/prepare_gsm8k.sh first."
[[ -f "${GSM8K_DIR}/test.parquet" ]] || sol_fail "Missing ${GSM8K_DIR}/test.parquet. Run scripts/sol/prepare_gsm8k.sh first."

RUN_TAG="${RUN_TAG:-$(sol_timestamp)}"
EXPERIMENT_NAME="${DEBUG_EXPERIMENT_NAME:-qwen2_5_0_5b_grpo_sol_debug}"
RUN_ROOT="${OUTPUT_ROOT}/debug/${RUN_TAG}"
LOCAL_CKPT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/${RUN_TAG}"

mkdir -p "${RUN_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${RUN_ROOT}"

sol_msg "Starting short 1-GPU GRPO validation run."
sol_msg "Run root: ${RUN_ROOT}"
sol_msg "Checkpoint dir: ${LOCAL_CKPT_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.val_before_train=False \
  data.train_files="${GSM8K_DIR}/train.parquet" \
  data.val_files="${GSM8K_DIR}/test.parquet" \
  data.train_batch_size=8 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="${SOL_PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${LOCAL_CKPT_DIR}" \
  "$@"
