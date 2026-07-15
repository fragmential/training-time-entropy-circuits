#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/entropy_own_gen_%j.out
export HF_HOME="/projects/prjs1815/hf_cache"
cd "$SLURM_SUBMIT_DIR"
uv run python oneoff_scripts/entropy_own_gen.py --tag d12
