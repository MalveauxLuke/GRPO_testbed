#!/usr/bin/env bash
# Example only. This file is the readable version of the current DeepScaleR-style
# math GDPO debug job.
#
# It is meant for learning:
# - how the math path is wired
# - which knobs are semantic vs runtime-only
# - which defaults are intentionally conservative on SOL
#
#SBATCH --job-name=example-math-gdpo-debug
#SBATCH --partition=htc
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${PROJECT_ROOT}/scripts/sol/common_env.sh"

# Semantic knobs:
# - model path
# - reward limit in tokens
export MATH_LENGTH_MODEL_PATH="${MATH_LENGTH_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export MATH_LENGTH_LIMIT_TOKENS="${MATH_LENGTH_LIMIT_TOKENS:-1024}"

# Debug-job caps:
export MATH_LENGTH_TOTAL_TRAINING_STEPS="${MATH_LENGTH_TOTAL_TRAINING_STEPS:-5}"
export MATH_LENGTH_SAVE_FREQ="${MATH_LENGTH_SAVE_FREQ:-1}"
export MATH_LENGTH_TEST_FREQ="${MATH_LENGTH_TEST_FREQ:--1}"

# Runtime-fit defaults. These are intentionally conservative because this job is
# for wiring validation, not for high throughput.
export MATH_LENGTH_GPU_MEMORY_UTILIZATION="${MATH_LENGTH_GPU_MEMORY_UTILIZATION:-0.12}"
export MATH_LENGTH_MAX_NUM_SEQS="${MATH_LENGTH_MAX_NUM_SEQS:-4}"
export MATH_LENGTH_MAX_NUM_BATCHED_TOKENS="${MATH_LENGTH_MAX_NUM_BATCHED_TOKENS:-3072}"
export MATH_LENGTH_ROLLOUT_N="${MATH_LENGTH_ROLLOUT_N:-4}"

# Rollout dumps are optional but useful for reward audits.
export MATH_LENGTH_ROLLOUT_DATA_DIR="${MATH_LENGTH_ROLLOUT_DATA_DIR:-/scratch/$USER/verl-grpo/rollout_dumps/example_math_gdpo_debug_$(date +%Y%m%d-%H%M%S)}"

"${PROJECT_ROOT}/scripts/sol/run_gdpo_math_length_debug.sh" "$@"
