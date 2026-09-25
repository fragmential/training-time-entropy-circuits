#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/plain_retune_%j.out
cd "$SLURM_SUBMIT_DIR"
uv run python /gpfs/scratch1/nodespecific/int4/75231/claude-75231/-gpfs-home2-dcampregher1-Tracing-representation-geometry-reproduction/68e6cc7a-f1be-468a-a4b7-1f3028c2f129/scratchpad/plain_retune.py
