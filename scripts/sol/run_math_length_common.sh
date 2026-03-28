#!/usr/bin/env bash
set -euo pipefail

ADV_ESTIMATOR="${1:-}"
PROFILE="${2:-}"
if [[ -z "${ADV_ESTIMATOR}" || -z "${PROFILE}" ]]; then
  printf '[sol-setup] ERROR: Expected usage: run_math_length_common.sh <grpo|gdpo> <debug|production> [extra overrides...]\n' >&2
  exit 1
fi
shift 2

case "${ADV_ESTIMATOR}" in
  grpo|gdpo) ;;
  *)
    printf '[sol-setup] ERROR: Unsupported estimator %s. Expected grpo or gdpo.\n' "${ADV_ESTIMATOR}" >&2
    exit 1
    ;;
esac

case "${PROFILE}" in
  debug|production) ;;
  *)
    printf '[sol-setup] ERROR: Unsupported profile %s. Expected debug or production.\n' "${PROFILE}" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

sol_require_slurm_allocation
sol_ensure_runtime_dirs
sol_ensure_upstream_checkout
sol_activate_env

case "${PROFILE}" in
  debug)
    DATASET_DIR="${DEEPSCALER_DEBUG_DIR}"
    MODEL_PATH_DEFAULT="Qwen/Qwen2.5-0.5B-Instruct"
    LENGTH_LIMIT_TOKENS="${MATH_LENGTH_LIMIT_TOKENS:-1024}"
    MAX_PROMPT_LENGTH="${MATH_LENGTH_MAX_PROMPT_LENGTH:-1024}"
    MAX_RESPONSE_LENGTH="${MATH_LENGTH_MAX_RESPONSE_LENGTH:-2048}"
    TRAIN_BATCH_SIZE="${MATH_LENGTH_TRAIN_BATCH_SIZE:-1}"
    VAL_BATCH_SIZE="${MATH_LENGTH_VAL_BATCH_SIZE:-1}"
    PPO_MINI_BATCH_SIZE="${MATH_LENGTH_PPO_MINI_BATCH_SIZE:-1}"
    PPO_MICRO_BATCH_SIZE="${MATH_LENGTH_PPO_MICRO_BATCH_SIZE:-1}"
    ROLLOUT_N="${MATH_LENGTH_ROLLOUT_N:-4}"
    MAX_NUM_SEQS="${MATH_LENGTH_MAX_NUM_SEQS:-4}"
    MAX_NUM_BATCHED_TOKENS="${MATH_LENGTH_MAX_NUM_BATCHED_TOKENS:-3072}"
    GPU_MEMORY_UTILIZATION="${MATH_LENGTH_GPU_MEMORY_UTILIZATION:-0.12}"
    TOTAL_TRAINING_STEPS="${MATH_LENGTH_TOTAL_TRAINING_STEPS:-5}"
    TOTAL_EPOCHS="${MATH_LENGTH_TOTAL_EPOCHS:-1}"
    SAVE_FREQ="${MATH_LENGTH_SAVE_FREQ:-1}"
    TEST_FREQ="${MATH_LENGTH_TEST_FREQ:--1}"
    USE_DYNAMIC_BSZ="${MATH_LENGTH_USE_DYNAMIC_BSZ:-False}"
    ENABLE_FILTER_GROUPS="${MATH_LENGTH_ENABLE_FILTER_GROUPS:-False}"
    FILTER_GROUPS_METRIC="${MATH_LENGTH_FILTER_GROUPS_METRIC:-correct_reward}"
    MAX_NUM_GEN_BATCHES="${MATH_LENGTH_MAX_NUM_GEN_BATCHES:-10}"
    MODEL_PATH="${MATH_LENGTH_MODEL_PATH:-${MODEL_PATH_DEFAULT}}"
    EXPERIMENT_NAME="${MATH_LENGTH_EXPERIMENT_NAME:-${MODEL_PATH##*/}_${ADV_ESTIMATOR}_math_length_sol_debug}"
    ;;
  production)
    DATASET_DIR="${DEEPSCALER_DIR}"
    MODEL_PATH_DEFAULT="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    LENGTH_LIMIT_TOKENS="${MATH_LENGTH_LIMIT_TOKENS:-4000}"
    MAX_PROMPT_LENGTH="${MATH_LENGTH_MAX_PROMPT_LENGTH:-2048}"
    MAX_RESPONSE_LENGTH="${MATH_LENGTH_MAX_RESPONSE_LENGTH:-8000}"
    TRAIN_BATCH_SIZE="${MATH_LENGTH_TRAIN_BATCH_SIZE:-512}"
    VAL_BATCH_SIZE="${MATH_LENGTH_VAL_BATCH_SIZE:-64}"
    PPO_MINI_BATCH_SIZE="${MATH_LENGTH_PPO_MINI_BATCH_SIZE:-32}"
    PPO_MICRO_BATCH_SIZE="${MATH_LENGTH_PPO_MICRO_BATCH_SIZE:-1}"
    ROLLOUT_N="${MATH_LENGTH_ROLLOUT_N:-16}"
    MAX_NUM_SEQS="${MATH_LENGTH_MAX_NUM_SEQS:-32}"
    MAX_NUM_BATCHED_TOKENS="${MATH_LENGTH_MAX_NUM_BATCHED_TOKENS:-10000}"
    GPU_MEMORY_UTILIZATION="${MATH_LENGTH_GPU_MEMORY_UTILIZATION:-0.50}"
    TOTAL_TRAINING_STEPS="${MATH_LENGTH_TOTAL_TRAINING_STEPS:-500}"
    TOTAL_EPOCHS="${MATH_LENGTH_TOTAL_EPOCHS:-1}"
    SAVE_FREQ="${MATH_LENGTH_SAVE_FREQ:-25}"
    TEST_FREQ="${MATH_LENGTH_TEST_FREQ:-25}"
    USE_DYNAMIC_BSZ="${MATH_LENGTH_USE_DYNAMIC_BSZ:-True}"
    ENABLE_FILTER_GROUPS="${MATH_LENGTH_ENABLE_FILTER_GROUPS:-True}"
    FILTER_GROUPS_METRIC="${MATH_LENGTH_FILTER_GROUPS_METRIC:-correct_reward}"
    MAX_NUM_GEN_BATCHES="${MATH_LENGTH_MAX_NUM_GEN_BATCHES:-10}"
    MODEL_PATH="${MATH_LENGTH_MODEL_PATH:-${MODEL_PATH_DEFAULT}}"
    EXPERIMENT_NAME="${MATH_LENGTH_EXPERIMENT_NAME:-${MODEL_PATH##*/}_${ADV_ESTIMATOR}_math_length_sol}"
    ;;
esac

[[ -f "${DATASET_DIR}/train.parquet" ]] || sol_fail "Missing ${DATASET_DIR}/train.parquet. Run scripts/sol/prepare_deepscaler_math.sh first."
[[ -f "${DATASET_DIR}/test.parquet" ]] || sol_fail "Missing ${DATASET_DIR}/test.parquet. Run scripts/sol/prepare_deepscaler_math.sh first."

RUN_TAG="${RUN_TAG:-$(sol_timestamp)}"
RUN_ROOT="${OUTPUT_ROOT}/math_length/${PROFILE}/${ADV_ESTIMATOR}/${RUN_TAG}"
LOCAL_CKPT_DIR="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/${RUN_TAG}"
PROJECT_NAME="${MATH_LENGTH_PROJECT_NAME:-${SOL_PROJECT_NAME}_${ADV_ESTIMATOR}_math_length}"

sol_prepare_tracking_paths "${PROJECT_NAME}" "${EXPERIMENT_NAME}" "${RUN_TAG}"
if [[ "${ADV_ESTIMATOR}" == "gdpo" ]]; then
  sol_prepare_gdpo_saturation_event_log_path
fi

mkdir -p "${RUN_ROOT}" "${LOCAL_CKPT_DIR}"
cd "${RUN_ROOT}"

sol_msg "Starting ${PROFILE} ${ADV_ESTIMATOR^^} DeepScaleR-style math-length run."
sol_msg "Dataset dir: ${DATASET_DIR}"
sol_msg "Model path: ${MODEL_PATH}"
sol_msg "Length limit tokens: ${LENGTH_LIMIT_TOKENS}"
sol_msg "Run root: ${RUN_ROOT}"
sol_msg "Checkpoint dir: ${LOCAL_CKPT_DIR}"
sol_msg "TensorBoard dir: ${TENSORBOARD_DIR}"
sol_msg "File logger path: ${VERL_FILE_LOGGER_PATH}"
if [[ "${ADV_ESTIMATOR}" == "gdpo" ]]; then
  sol_msg "GDPO saturation event log: ${GDPO_SATURATION_EVENT_LOG_PATH}"
fi

cmd=(
  "$(sol_python)" -m verl.trainer.main_ppo
  "algorithm.adv_estimator=${ADV_ESTIMATOR}"
  "trainer.val_before_train=False"
  "data.train_files=${DATASET_DIR}/train.parquet"
  "data.val_files=${DATASET_DIR}/test.parquet"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.val_batch_size=${VAL_BATCH_SIZE}"
  "data.dataloader_num_workers=0"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "+actor_rollout_ref.model.override_config.attn_implementation=eager"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.optim.weight_decay=0.1"
  "actor_rollout_ref.model.use_remove_padding=False"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}"
  "actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.skip_tokenizer_init=True"
  "actor_rollout_ref.rollout.agent.num_workers=1"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "+actor_rollout_ref.ref.model.override_config.attn_implementation=eager"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "reward.custom_reward_function.path=${UPSTREAM_VERL_DIR}/verl/utils/reward_score/deepscaler_math_length.py"
  "reward.custom_reward_function.name=compute_score"
  "+reward.custom_reward_function.reward_kwargs.length_limit_tokens=${LENGTH_LIMIT_TOKENS}"
  "trainer.critic_warmup=0"
  "trainer.logger=[\"console\",\"tensorboard\",\"file\"]"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=${MATH_LENGTH_N_GPUS_PER_NODE:-1}"
  "trainer.nnodes=${MATH_LENGTH_NNODES:-1}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.default_local_dir=${LOCAL_CKPT_DIR}"
)

if [[ "${ADV_ESTIMATOR}" == "gdpo" ]]; then
  cmd+=(
    "algorithm.gdpo_baseline_mode=${GDPO_BASELINE_MODE:-upstream}"
    "algorithm.gdpo_reward_keys=[\"correct_reward\",\"length_reward\"]"
    "reward.reward_manager.name=gdpo"
  )
else
  cmd+=("reward.reward_manager.name=naive")
fi

if [[ "${ENABLE_FILTER_GROUPS}" == "True" ]]; then
  cmd+=(
    "+data.gen_batch_size=${MATH_LENGTH_GEN_BATCH_SIZE:-1536}"
    "algorithm.filter_groups.enable=True"
    "algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}"
    "algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}"
  )
fi

"${cmd[@]}" "$@"
