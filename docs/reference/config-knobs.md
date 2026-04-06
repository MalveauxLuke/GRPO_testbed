# Configuration Knobs

This is the single place to look when you want to change configuration on purpose.

## How config resolution works

The repo now resolves runs in this order:

1. a front-door config in [configs/experiments](/Users/god/Documents/VERL_GRPO/configs/experiments)
2. a Slurm entrypoint in [slurm](/Users/god/Documents/VERL_GRPO/slurm)
3. a shared builder:
   - [run_gdpo_gsm8k_modern_fit_2gpu.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_2gpu.sh)
   - [run_math_length_common.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh)
4. the final `verl.trainer.main_ppo` Hydra command

That means there are three safe places to edit, depending on what you want to change.

## Which file to edit

### 1. Front-door config

Edit a file in:
- [configs/experiments/gsm8k](/Users/god/Documents/VERL_GRPO/configs/experiments/gsm8k)
- [configs/experiments/math](/Users/god/Documents/VERL_GRPO/configs/experiments/math)

Use this for:
- model choice
- experiment and project names
- step caps
- save and test cadence
- rollout dump path
- any env var already supported by the shared builder

These config files are just bash env fragments. You are allowed to add more `export ...` lines to them.

After editing, inspect with:

```bash
cd ~/GRPO_testbed
scripts/sol/submit_experiment.sh <config> --dry-run
```

### 2. Shared builder

Edit:
- [run_gdpo_gsm8k_modern_fit_2gpu.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_gdpo_gsm8k_modern_fit_2gpu.sh)
- [run_math_length_common.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/run_math_length_common.sh)

Use this when the thing you want is not already parameterized through an env var.

Examples:
- adding a new Hydra argument
- changing reward path wiring
- changing algorithm semantics
- changing logger configuration
- exposing a currently hardcoded rollout or actor arg

### 3. Slurm entrypoint

Edit a `.sbatch` file in [slurm](/Users/god/Documents/VERL_GRPO/slurm).

Use this for:
- partition
- qos
- walltime
- CPU count
- GPU count
- host RAM

## Where the heavily commented files are

These are the readable teaching files:

- GSM8K example sbatch:
  [slurm/examples/gsm8k_modern_example.sbatch](/Users/god/Documents/VERL_GRPO/slurm/examples/gsm8k_modern_example.sbatch)
- math example sbatch:
  [slurm/examples/math_length_example.sbatch](/Users/god/Documents/VERL_GRPO/slurm/examples/math_length_example.sbatch)

These are intentionally verbose and explain:
- which knobs are semantic
- which knobs are runtime-only
- which settings are there because SOL already proved they matter

The real active behavior still comes from the shared builders.

## GSM8K knobs

These are the current env vars consumed by the shared GSM8K path.

### Identity and tracking

- `GSM8K_MODERN_MODEL_PATH`
- `GSM8K_MODERN_EXPERIMENT_NAME`
- `GSM8K_MODERN_PROJECT_NAME`
- `GSM8K_MODERN_ROLLOUT_DATA_DIR`
- `GSM8K_MODERN_RUNTIME_PROFILE`
- `GSM8K_MODERN_RUN_CLASSIFICATION`

### Dataset and validation

- `GSM8K_MODERN_DIR`
- `GSM8K_MODERN_TRAIN_BATCH_SIZE`
- `GSM8K_MODERN_VAL_BATCH_SIZE`
- `GSM8K_MODERN_VAL_MAX_SAMPLES`
- `GSM8K_MODERN_VAL_BEFORE_TRAIN`
- `GSM8K_MODERN_DATALOADER_NUM_WORKERS`

### Length limits

- `GSM8K_MODERN_MAX_PROMPT_LENGTH`
- `GSM8K_MODERN_MAX_RESPONSE_LENGTH`
- `GSM8K_MODERN_MAX_MODEL_LEN`
- `GSM8K_MODERN_MAX_NUM_BATCHED_TOKENS`
- `GSM8K_MODERN_MAX_NUM_SEQS`

### Optimization and LoRA

- `GSM8K_MODERN_LR`
- `GSM8K_MODERN_KL_CTRL_COEF`
- `GSM8K_MODERN_ACTOR_KL_LOSS_COEF`
- `GSM8K_MODERN_LORA_RANK`
- `GSM8K_MODERN_LORA_ALPHA`
- `GSM8K_MODERN_TARGET_MODULES`

### Batch math

- `GSM8K_MODERN_N_GPUS_PER_NODE`
- `GSM8K_MODERN_NNODES`
- `GSM8K_MODERN_ROLLOUT_N`
- `GSM8K_MODERN_PPO_MINI_BATCH_SIZE`
- `GSM8K_MODERN_PPO_MICRO_BATCH_SIZE`
- `GSM8K_MODERN_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE`
- `GSM8K_MODERN_REF_LOGPROB_MICRO_BATCH_SIZE`

### Rollout and fit safety

- `GSM8K_MODERN_TP_SIZE`
- `GSM8K_MODERN_GPU_MEMORY_UTILIZATION`
- `GSM8K_MODERN_USE_SHM`
- `GSM8K_MODERN_SKIP_TOKENIZER_INIT`
- `GSM8K_MODERN_ROLLOUT_ENFORCE_EAGER`
- `GSM8K_MODERN_ENABLE_CHUNKED_PREFILL`
- `GSM8K_MODERN_MAX_PARALLEL_LOADING_WORKERS`
- `GSM8K_MODERN_AGENT_NUM_WORKERS`
- `GSM8K_MODERN_USE_REMOVE_PADDING`
- `GSM8K_MODERN_ACTOR_PARAM_OFFLOAD`
- `GSM8K_MODERN_ACTOR_OPTIMIZER_OFFLOAD`
- `GSM8K_MODERN_REF_PARAM_OFFLOAD`

### Schedule and diagnostics

- `GSM8K_MODERN_TOTAL_TRAINING_STEPS`
- `GSM8K_MODERN_TOTAL_EPOCHS`
- `GSM8K_MODERN_SAVE_FREQ`
- `GSM8K_MODERN_TEST_FREQ`
- `GSM8K_MODERN_DIAGNOSTICS_INTERVAL_SECONDS`
- `GSM8K_MODERN_GDPO_BASELINE_MODE`

### Practical example

If you want to make the 75-step config more aggressive, you can add lines like:

```bash
export GSM8K_MODERN_TRAIN_BATCH_SIZE=32
export GSM8K_MODERN_VAL_BATCH_SIZE=32
export GSM8K_MODERN_PPO_MINI_BATCH_SIZE=32
export GSM8K_MODERN_PPO_MICRO_BATCH_SIZE=2
export GSM8K_MODERN_ROLLOUT_N=8
export GSM8K_MODERN_LR=1e-6
export GSM8K_MODERN_GPU_MEMORY_UTILIZATION=0.30
```

## Math knobs

These are the current env vars consumed by the shared math path.

### Identity and tracking

- `MATH_LENGTH_MODEL_PATH`
- `MATH_LENGTH_EXPERIMENT_NAME`
- `MATH_LENGTH_PROJECT_NAME`
- `MATH_LENGTH_ROLLOUT_DATA_DIR`
- `MATH_LENGTH_RUNTIME_PROFILE`
- `MATH_LENGTH_RUN_CLASSIFICATION`

### Dataset and reward shape

- `MATH_LENGTH_LIMIT_TOKENS`
- `MATH_LENGTH_TRAIN_BATCH_SIZE`
- `MATH_LENGTH_VAL_BATCH_SIZE`
- `MATH_LENGTH_MAX_PROMPT_LENGTH`
- `MATH_LENGTH_MAX_RESPONSE_LENGTH`

### Batch math

- `MATH_LENGTH_N_GPUS_PER_NODE`
- `MATH_LENGTH_NNODES`
- `MATH_LENGTH_ROLLOUT_N`
- `MATH_LENGTH_PPO_MINI_BATCH_SIZE`
- `MATH_LENGTH_PPO_MICRO_BATCH_SIZE`
- `MATH_LENGTH_ROLLOUT_LOGPROB_MICRO_BATCH_SIZE`
- `MATH_LENGTH_REF_LOGPROB_MICRO_BATCH_SIZE`

### Rollout and runtime shape

- `MATH_LENGTH_GPU_MEMORY_UTILIZATION`
- `MATH_LENGTH_MAX_NUM_SEQS`
- `MATH_LENGTH_MAX_NUM_BATCHED_TOKENS`
- `MATH_LENGTH_USE_DYNAMIC_BSZ`

### Filtering and generation controls

- `MATH_LENGTH_ENABLE_FILTER_GROUPS`
- `MATH_LENGTH_FILTER_GROUPS_METRIC`
- `MATH_LENGTH_MAX_NUM_GEN_BATCHES`
- `MATH_LENGTH_GEN_BATCH_SIZE`

### Schedule

- `MATH_LENGTH_TOTAL_TRAINING_STEPS`
- `MATH_LENGTH_TOTAL_EPOCHS`
- `MATH_LENGTH_SAVE_FREQ`
- `MATH_LENGTH_TEST_FREQ`

### Extra math note

The GDPO math path also reads:

- `GDPO_BASELINE_MODE`

That one is not `MATH_LENGTH_`-prefixed today because it is passed directly into the GDPO-specific builder logic.

### Practical example

If you want to change the debug math shape, you can add lines like:

```bash
export MATH_LENGTH_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
export MATH_LENGTH_LIMIT_TOKENS=1536
export MATH_LENGTH_ROLLOUT_N=8
export MATH_LENGTH_GPU_MEMORY_UTILIZATION=0.18
export MATH_LENGTH_TOTAL_TRAINING_STEPS=10
```

## Repo-wide path and env knobs

These live in [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh) and affect every workflow:

- `SOL_ENV_NAME`
- `SOL_PROJECT_NAME`
- `SCRATCH_ROOT`
- `DATA_ROOT`
- `OUTPUT_ROOT`
- `CHECKPOINT_ROOT`
- `LOG_ROOT`
- `TENSORBOARD_ROOT`
- `FILE_LOG_ROOT`
- `HF_HOME`
- `HF_DATASETS_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`
- `VLLM_CACHE_ROOT`
- `WANDB_DIR`

Only change these if you are intentionally moving shared storage locations or runtime roots.

## Safe edit process

1. Copy the nearest config in [configs/experiments](/Users/god/Documents/VERL_GRPO/configs/experiments).
2. Add or change the `export ...` lines you actually want.
3. Run `scripts/sol/submit_experiment.sh <config> --dry-run`.
4. Check the resolved command and batch math.
5. If the thing you need is not exposed, add a new env-backed knob to the shared builder.
6. Only edit `.sbatch` files when you are changing cluster resources.
