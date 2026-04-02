#!/usr/bin/env bash

if [[ -z "${BASH_VERSION:-}" ]]; then
  printf '[sol-setup] ERROR: scripts/sol/common_env.sh must be sourced from bash, not the current shell.\n' >&2
  printf '[sol-setup] Run `exec bash -l` first, then `source scripts/sol/common_env.sh`.\n' >&2
  return 1 2>/dev/null || exit 1
fi

SOL_COMMON_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SOL_COMMON_ENV_DIR}/../.." && pwd)}"

export PROJECT_ROOT
export VERL_REF="${VERL_REF:-v0.7.1}"
export SOL_ENV_NAME="${SOL_ENV_NAME:-verl-grpo-sol}"
export SOL_PROJECT_NAME="${SOL_PROJECT_NAME:-asu_sol_upstream_verl_grpo}"
export SOL_ACCOUNT="${SOL_ACCOUNT:-}"

export EXTERNAL_ROOT="${EXTERNAL_ROOT:-${PROJECT_ROOT}/external}"
export UPSTREAM_VERL_DIR="${UPSTREAM_VERL_DIR:-${EXTERNAL_ROOT}/verl}"

export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/verl-grpo}"
export DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/data}"
export GSM8K_DIR="${GSM8K_DIR:-${DATA_ROOT}/gsm8k}"
export GSM8K_MODERN_DIR="${GSM8K_MODERN_DIR:-${DATA_ROOT}/gsm8k_modern_two_reward}"
export GSM8K_GDPO_PROBE_DIR="${GSM8K_GDPO_PROBE_DIR:-${DATA_ROOT}/gsm8k_gdpo_saturation_probe}"
export GSM8K_GDPO_PROBE_HARD_DIR="${GSM8K_GDPO_PROBE_HARD_DIR:-${DATA_ROOT}/gsm8k_gdpo_saturation_probe_hard}"
export DEEPSCALER_DIR="${DEEPSCALER_DIR:-${DATA_ROOT}/deepscaler_math_length}"
export DEEPSCALER_DEBUG_DIR="${DEEPSCALER_DEBUG_DIR:-${DATA_ROOT}/deepscaler_math_length_debug}"
export DEEPSCALER_SATURATION_PROBE_DIR="${DEEPSCALER_SATURATION_PROBE_DIR:-${DATA_ROOT}/deepscaler_math_length_saturation_probe}"
export MATH_EVAL_DIR="${MATH_EVAL_DIR:-${DATA_ROOT}/math_eval}"
export RLLA_4K_DIR="${RLLA_4K_DIR:-${DATA_ROOT}/rlla_4k}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/outputs}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SCRATCH_ROOT}/checkpoints}"
export LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/logs}"
export TENSORBOARD_ROOT="${TENSORBOARD_ROOT:-${SCRATCH_ROOT}/tensorboard}"
export FILE_LOG_ROOT="${FILE_LOG_ROOT:-${SCRATCH_ROOT}/metrics}"
export RAY_TMPDIR="${RAY_TMPDIR:-${SCRATCH_ROOT}/ray}"
export TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_ROOT}/hf/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${SCRATCH_ROOT}/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SCRATCH_ROOT}/hf/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${SCRATCH_ROOT}/vllm}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/wandb}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/pip-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg-cache}"
export MAMBA_PKGS_DIRS="${MAMBA_PKGS_DIRS:-${SCRATCH_ROOT}/mamba-pkgs}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export TOOLRL_REPO_URL="${TOOLRL_REPO_URL:-https://github.com/qiancheng0/ToolRL.git}"
export VERL_FILE_LOGGER_ROOT="${VERL_FILE_LOGGER_ROOT:-${FILE_LOG_ROOT}}"

sol_msg() {
  printf '[sol-setup] %s\n' "$*"
}

sol_fail() {
  printf '[sol-setup] ERROR: %s\n' "$*" >&2
  return 1 2>/dev/null || exit 1
}

sol_timestamp() {
  date +"%Y%m%d-%H%M%S"
}

sol_init_modules() {
  if command -v module >/dev/null 2>&1; then
    return 0
  fi

  local init_file
  for init_file in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
    if [[ -f "${init_file}" ]]; then
      # shellcheck disable=SC1090
      source "${init_file}"
      break
    fi
  done

  command -v module >/dev/null 2>&1 || sol_fail "Could not initialize the module command. Load modules manually before rerunning."
}

sol_load_mamba() {
  sol_init_modules
  module load mamba/latest
}

sol_deactivate_base_if_needed() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "base" ]]; then
    # shellcheck disable=SC1091
    source deactivate >/dev/null 2>&1 || true
  fi
}

sol_normalize_gpu_visibility_env() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
      sol_msg "Unsetting ROCR_VISIBLE_DEVICES for CUDA-based verl execution."
      unset ROCR_VISIBLE_DEVICES
    fi
    if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
      sol_msg "Unsetting HIP_VISIBLE_DEVICES for CUDA-based verl execution."
      unset HIP_VISIBLE_DEVICES
    fi
  fi
}

sol_prepend_env_bin_to_path() {
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/bin" ]]; then
    case ":${PATH}:" in
      *":${CONDA_PREFIX}/bin:"*) ;;
      *) export PATH="${CONDA_PREFIX}/bin:${PATH}" ;;
    esac
  fi
}

sol_activate_env() {
  sol_load_mamba
  sol_deactivate_base_if_needed
  # shellcheck disable=SC1091
  source activate "${SOL_ENV_NAME}" || sol_fail "Could not activate Mamba env '${SOL_ENV_NAME}'. Run scripts/sol/create_env.sh first."
  sol_prepend_env_bin_to_path
  sol_normalize_gpu_visibility_env
}

sol_python() {
  if [[ -z "${CONDA_PREFIX:-}" ]]; then
    sol_fail "No active Mamba env detected. Run sol_activate_env first."
  fi

  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "${SOL_ENV_NAME}" ]]; then
    sol_fail "Active env '${CONDA_DEFAULT_ENV}' does not match expected env '${SOL_ENV_NAME}'. Run sol_activate_env first."
  fi

  if [[ ! -x "${CONDA_PREFIX}/bin/python" ]]; then
    sol_fail "Expected active environment python at ${CONDA_PREFIX}/bin/python, but it was not found."
  fi

  printf '%s\n' "${CONDA_PREFIX}/bin/python"
}

sol_require_slurm_allocation() {
  [[ -n "${SLURM_JOB_ID:-}" ]] || sol_fail "Run this from a Slurm compute allocation, for example: salloc -p lightwork -q public -t 02:00:00 -c 4"
}

sol_ensure_upstream_checkout() {
  local required_paths=(
    "${UPSTREAM_VERL_DIR}/setup.py"
    "${UPSTREAM_VERL_DIR}/scripts/install_vllm_sglang_mcore.sh"
    "${UPSTREAM_VERL_DIR}/examples/grpo_trainer/run_qwen2-7b.sh"
  )
  local missing=()
  local required_path
  for required_path in "${required_paths[@]}"; do
    [[ -e "${required_path}" ]] || missing+=("${required_path}")
  done

  if (( ${#missing[@]} > 0 )); then
    sol_fail "Vendored verl source is incomplete under ${UPSTREAM_VERL_DIR}. Missing: ${missing[*]}. Restore the repo checkout or run git pull."
  fi
}

sol_ensure_runtime_dirs() {
  mkdir -p \
    "${EXTERNAL_ROOT}" \
    "${DATA_ROOT}" \
    "${GSM8K_DIR}" \
    "${GSM8K_MODERN_DIR}" \
    "${GSM8K_GDPO_PROBE_DIR}" \
    "${GSM8K_GDPO_PROBE_HARD_DIR}" \
    "${DEEPSCALER_DIR}" \
    "${DEEPSCALER_DEBUG_DIR}" \
    "${DEEPSCALER_SATURATION_PROBE_DIR}" \
    "${MATH_EVAL_DIR}" \
    "${RLLA_4K_DIR}" \
    "${OUTPUT_ROOT}" \
    "${CHECKPOINT_ROOT}" \
    "${LOG_ROOT}" \
    "${TENSORBOARD_ROOT}" \
    "${FILE_LOG_ROOT}" \
    "${RAY_TMPDIR}" \
    "${TMPDIR}" \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${VLLM_CACHE_ROOT}" \
    "${WANDB_DIR}" \
    "${PIP_CACHE_DIR}" \
    "${XDG_CACHE_HOME}" \
    "${MAMBA_PKGS_DIRS}"
}

sol_prepare_tracking_paths() {
  local project_name="$1"
  local experiment_name="$2"
  local run_tag="$3"

  export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${TENSORBOARD_ROOT}/${project_name}/${experiment_name}/${run_tag}}"
  export VERL_FILE_LOGGER_PATH="${VERL_FILE_LOGGER_PATH:-${FILE_LOG_ROOT}/${project_name}/${experiment_name}/${run_tag}.jsonl}"

  mkdir -p "${TENSORBOARD_DIR}" "$(dirname "${VERL_FILE_LOGGER_PATH}")"
}

sol_prepare_gdpo_saturation_event_log_path() {
  export GDPO_SATURATION_EVENT_LOG_PATH="${GDPO_SATURATION_EVENT_LOG_PATH:-${VERL_FILE_LOGGER_PATH%.jsonl}.gdpo_saturation_events.jsonl}"
  mkdir -p "$(dirname "${GDPO_SATURATION_EVENT_LOG_PATH}")"
}

sol_require_runtime_profile() {
  local runtime_profile="${1:-}"
  case "${runtime_profile}" in
    reference_public|sol_fit_2gpu|debug_safe) ;;
    *)
      sol_fail \
        "Unsupported runtime profile '${runtime_profile}'. Expected one of: reference_public, sol_fit_2gpu, debug_safe."
      ;;
  esac
}

sol_require_run_classification() {
  local run_classification="${1:-}"
  case "${run_classification}" in
    feasibility-smoke|integration-smoke|full-run) ;;
    *)
      sol_fail \
        "Unsupported run classification '${run_classification}'. Expected one of: feasibility-smoke, integration-smoke, full-run."
      ;;
  esac
}

sol_resolve_normalized_actor_mini_batch() {
  local total_gpus="${1:-}"
  local rollout_n="${2:-}"
  local ppo_mini_batch_size="${3:-}"

  [[ -n "${total_gpus}" && -n "${rollout_n}" && -n "${ppo_mini_batch_size}" ]] || sol_fail \
    "sol_resolve_normalized_actor_mini_batch expects: <total_gpus> <rollout_n> <ppo_mini_batch_size>."
  (( total_gpus > 0 )) || sol_fail "Expected total_gpus > 0, got ${total_gpus}."
  (( rollout_n > 0 )) || sol_fail "Expected rollout_n > 0, got ${rollout_n}."
  (( ppo_mini_batch_size > 0 )) || sol_fail "Expected ppo_mini_batch_size > 0, got ${ppo_mini_batch_size}."
  (( (ppo_mini_batch_size * rollout_n) % total_gpus == 0 )) || sol_fail \
    "Invalid config: ppo_mini_batch_size (${ppo_mini_batch_size}) * rollout.n (${rollout_n}) must be divisible by total_gpus (${total_gpus})."

  printf '%s\n' "$((ppo_mini_batch_size * rollout_n / total_gpus))"
}

sol_validate_micro_batch_divides_normalized() {
  local normalized_mini_batch="${1:-}"
  local micro_batch_value="${2:-}"
  local micro_batch_label="${3:-micro_batch}"

  [[ -n "${normalized_mini_batch}" ]] || sol_fail "Expected normalized_mini_batch to be set."
  if [[ -z "${micro_batch_value}" || "${micro_batch_value}" == "None" ]]; then
    return 0
  fi

  (( micro_batch_value > 0 )) || sol_fail \
    "${micro_batch_label} must be > 0 when provided, got ${micro_batch_value}."
  (( normalized_mini_batch % micro_batch_value == 0 )) || sol_fail \
    "Invalid config: normalized actor mini-batch (${normalized_mini_batch}) must be divisible by ${micro_batch_label} (${micro_batch_value})."
}

sol_require_post_run_audit_command() {
  local post_run_audit_cmd="${1:-}"
  [[ -n "${post_run_audit_cmd}" ]] || sol_fail \
    "Each active wrapper must declare a post-run audit command."
}

sol_log_run_contract() {
  local semantic_baseline="${1:-}"
  local runtime_profile="${2:-}"
  local run_classification="${3:-}"
  local upstream_anchor="${4:-}"
  local local_parent="${5:-}"
  local post_run_audit_cmd="${6:-}"

  sol_require_runtime_profile "${runtime_profile}"
  sol_require_run_classification "${run_classification}"
  sol_require_post_run_audit_command "${post_run_audit_cmd}"

  sol_msg "Semantic baseline: ${semantic_baseline}"
  sol_msg "Runtime profile: ${runtime_profile}"
  sol_msg "Run classification: ${run_classification}"
  sol_msg "Upstream anchor: ${upstream_anchor}"
  sol_msg "Closest prior local config: ${local_parent}"
  sol_msg "Post-run audit: ${post_run_audit_cmd}"
}

sol_log_batch_math() {
  local total_gpus="${1:-}"
  local rollout_n="${2:-}"
  local ppo_mini_batch_size="${3:-}"
  local normalized_mini_batch="${4:-}"
  local actor_micro_batch="${5:-}"
  local rollout_micro_batch="${6:-}"
  local ref_micro_batch="${7:-}"

  sol_msg "Batch math: total_gpus=${total_gpus}, rollout.n=${rollout_n}, actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
  sol_msg "Normalized actor mini-batch after rollout/GPU split: ${normalized_mini_batch}"
  sol_msg "Per-GPU micro-batches: actor=${actor_micro_batch}, rollout_logprob=${rollout_micro_batch}, ref_logprob=${ref_micro_batch}"
}

sol_start_resource_sampler() {
  local log_path="${1:-}"
  local interval_seconds="${2:-5}"
  [[ -n "${log_path}" ]] || sol_fail "sol_start_resource_sampler expects a log path."

  mkdir -p "$(dirname "${log_path}")"
  (
    while true; do
      {
        printf '===== %s =====\n' "$(date +"%Y-%m-%d %H:%M:%S %Z")"
        if command -v nvidia-smi >/dev/null 2>&1; then
          nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
        else
          printf 'nvidia-smi not available on PATH\n'
        fi
        if command -v free >/dev/null 2>&1; then
          free -h || true
        else
          printf 'free not available on PATH\n'
        fi
        ps -eo pid,ppid,rss,comm,args | awk 'NR == 1 || /ray|python|vllm/' | head -n 40 || true
        printf '\n'
      } >> "${log_path}" 2>&1
      sleep "${interval_seconds}"
    done
  ) &
  export SOL_RESOURCE_SAMPLER_PID="$!"
}
