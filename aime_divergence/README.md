# AIME Divergence Debug

This is an inference-only debug pipeline for AIME rollout divergence analysis.
It generates 8 rollouts per problem for AIME 2024 and AIME 2025, checks each
answer, and writes a single JSON report.

There is no training in this workflow.

Reference papers:

- Tang et al., "Rethinking Sample Polarity in RLVR": https://arxiv.org/abs/2512.21625
- Qwen Pilot Team: https://arxiv.org/abs/2603.22446

## Model

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

## Generation

The prompt template intentionally does not use the model chat template:

```text
User:
 {question}
 Please reason step by step, and put your final answer within \boxed{}.

 Assistant:
```

Default generation settings:

- samples per prompt: `8`
- max problems: `0`, meaning all loaded problems
- temperature: `1.0`
- top-p: `0.95`
- max tokens: `16384`
- dtype: `bfloat16`
- tensor parallel size: `1`
- TensorBoard: enabled in the SOL sbatches

## Run on SOL

From the repo root on SOL:

```bash
cd ~/GRPO_testbed
sbatch aime_divergence/run_debug.sbatch
```

For a short debug smoke first, use the same debug launcher with smaller limits:

```bash
cd ~/GRPO_testbed
AIME_DIVERGENCE_MAX_PROBLEMS=2 \
AIME_DIVERGENCE_SAMPLES=2 \
AIME_DIVERGENCE_MAX_TOKENS=2048 \
AIME_DIVERGENCE_MAX_MODEL_LEN=4096 \
  sbatch aime_divergence/run_debug.sbatch
```

For the token-logged pipeline smoke test:

```bash
cd ~/GRPO_testbed
sbatch aime_divergence/run_logged_smoke.sbatch
```

The token-logged smoke job uses the same code path as the full logged run, but defaults to:

- first `2` loaded AIME problems
- `2` rollouts per problem
- `2048` max generated tokens
- `4096` max model length
- `1` hour Slurm limit
- TensorBoard enabled
- manual resume testing is still recommended if you want to exercise checkpoint recovery explicitly

The sbatch reuses the repo SOL setup:

- [common_env.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/common_env.sh)
- scratch cache env vars
- `verl-grpo-sol` mamba environment
- scratch Slurm logs under `/scratch/$USER/verl-grpo/logs`
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` to avoid FlashInfer attention backend issues on SOL
- `VLLM_USE_FLASHINFER_SAMPLER=0` because vLLM V1 can still route top-k/top-p sampling through FlashInfer unless explicitly disabled

The default output path is:

```text
/scratch/$USER/verl-grpo/outputs/aime_divergence/debug_<timestamp>/aime_rollouts_debug.json
```

Override it if needed:

```bash
AIME_DIVERGENCE_OUTPUT_PATH=/scratch/$USER/verl-grpo/outputs/aime_divergence/manual/aime_rollouts_debug.json \
  sbatch aime_divergence/run_debug.sbatch
```

## Logged Rollouts

The logged pipeline keeps the same prompting, datasets, extraction, and summary
behavior as the debug run, but also stores one compressed `.npz` per problem
with token ids, sampled-token probabilities, ranks, top-k probabilities, and
approximate entropy.

Run the full logged pipeline on SOL with:

```bash
cd ~/GRPO_testbed
sbatch aime_divergence/run_logged.sbatch
```

Or locally/in an active env:

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
sol_activate_env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m aime_divergence.run_logged --output-dir aime_divergence/outputs/logged_manual
```

If you want development-time strictness instead of the default continue-on-error
behavior, add:

```bash
python -m aime_divergence.run_logged --output-dir aime_divergence/outputs/logged_manual --fail-fast
```

Logged output layout:

```text
outputs/aime_divergence/logged_<timestamp>/
├── config.json
├── rollout_results.json
├── run_summary.txt
├── tensorboard/
└── token_data/
    ├── aime24_01.npz
    ├── ...
    └── aime25_30.npz
```

Checkpointing/resume rules:

- generation runs one problem at a time
- `token_data/<problem_id>.npz` plus the internal checkpoint manifest determine completion
- if a problem has token data but no checkpoint row, the run regenerates it
- rerunning the same output dir resumes from completed problems
- token-level analysis is intentionally out of scope for this pipeline; the run only logs and stores artifacts

## TensorBoard

The SOL sbatches write TensorBoard logs next to the JSON output:

```text
/scratch/$USER/verl-grpo/outputs/aime_divergence/<run_tag>/tensorboard
```

After submitting a smoke or debug job, launch TensorBoard from SOL with:

```bash
tensorboard --logdir /scratch/$USER/verl-grpo/outputs/aime_divergence --host 0.0.0.0 --port 6006
```

Useful tags:

- `aime/summary/*`: final run-level counts and accuracy.
- `aime/problem/*`: per-problem rollout counts, unknown fraction, and response length.
- `aime/split_distribution/*`: how many problems landed in each correct/incorrect/unknown split.
- `aime/extraction_method/*`: how often each extraction path was used.

## Run inside an active SOL env

```bash
cd ~/GRPO_testbed
source scripts/sol/common_env.sh
sol_activate_env
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m aime_divergence.run_debug
```

## Dependencies

`math-verify` is required for the primary extraction path. It is already
installed by the repo's upstream environment installer:

```bash
scripts/sol/install_upstream_verl.sh
```

If the import fails on SOL, rerun that installer from a Slurm allocation.

## Output

The output JSON has:

- `metadata`
- `problems`
- `summary`

Each problem contains 8 rollouts with:

- full generated text
- token count
- finish reason
- extracted answer
- correctness flag
- extraction method

The stdout summary prints the debug checklist:

- total problems
- total rollouts
- mixed correct/incorrect problems
- all-correct and all-incorrect counts
- unknown extraction count
- average response length
- extraction failures
- split distribution
- extraction method counts
- ground-truth sanity warnings
