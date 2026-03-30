#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=profile_sweep
#SBATCH --output=slurm/logs/profile_sweep_%j.out
#SBATCH --error=slurm/logs/profile_sweep_%j.err

module purge
export HF_HOME="/projects/prjs1815/hf_cache"
cd "$HOME/Tracing-representation-geometry-reproduction" || exit 1

uv run scripts/profile_sweep.py "$@"
