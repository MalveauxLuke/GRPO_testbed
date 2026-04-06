# Slurm Entry Points

Top-level files in [slurm](/Users/god/Documents/VERL_GRPO/slurm) are the compatibility entrypoints that existing commands still use.

Current active runs:
- GSM8K debug:
  - [gdpo_gsm8k_modern_fit_debug_hybrid_hash.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_debug_hybrid_hash.sbatch)
- GSM8K 75-step saturation run:
  - [gdpo_gsm8k_modern_fit_2gpu_hybrid_hash_saturation_check.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu_hybrid_hash_saturation_check.sbatch)
- GSM8K full run:
  - [gdpo_gsm8k_modern_fit_2gpu.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_gsm8k_modern_fit_2gpu.sbatch)
- math debug:
  - [grpo_math_length_debug.sbatch](/Users/god/Documents/VERL_GRPO/slurm/grpo_math_length_debug.sbatch)
  - [gdpo_math_length_debug.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_math_length_debug.sbatch)
- math production:
  - [grpo_math_length_production.sbatch](/Users/god/Documents/VERL_GRPO/slurm/grpo_math_length_production.sbatch)
  - [gdpo_math_length_production.sbatch](/Users/god/Documents/VERL_GRPO/slurm/gdpo_math_length_production.sbatch)

For learning and hand-editing examples, use:
- [slurm/examples](/Users/god/Documents/VERL_GRPO/slurm/examples)

For the new human-facing path, prefer:
- [submit_experiment.sh](/Users/god/Documents/VERL_GRPO/scripts/sol/submit_experiment.sh)
- [configs/experiments](/Users/god/Documents/VERL_GRPO/configs/experiments)
