#!/bin/bash
# TriviaQA answer-likelihood sweep over the block_representations_samples model set.
# Usage: ./slurm/memorization.sh [sbatch overrides...]
# Runs from THIS worktree (code not yet in the native checkout).
set -euo pipefail

WORKTREE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# model : batch_size (short sequences, so batch generously; smaller for 7B)
MODELS=(
    "EleutherAI/pythia-1b-deduped"
    "EleutherAI/pythia-6.9b-deduped"
    "allenai/OLMo-2-0425-1B"
    "allenai/OLMo-2-1124-7B"
)
BATCH=(128 64 128 64)

MODELS_STR="${MODELS[*]}"
BATCH_STR="${BATCH[*]}"

mkdir -p "${WORKTREE}/slurm/logs"

sbatch --array=0-$((${#MODELS[@]} - 1)) "$@" <<EOF
#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=memorization
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/memorization_%A_%a.out
#SBATCH --error=slurm/logs/memorization_%A_%a.err

MODELS=(${MODELS_STR})
BATCH=(${BATCH_STR})
MODEL="\${MODELS[\$SLURM_ARRAY_TASK_ID]}"
BS="\${BATCH[\$SLURM_ARRAY_TASK_ID]}"

module purge
export HF_HOME="/projects/prjs1815/hf_cache"
cd "${WORKTREE}" || exit 1
time uv run scripts/compute_memorization.py \
    --model_name "\${MODEL}" --batch_size "\${BS}" \
    --max_checkpoints 50 --checkpoint_spacing log
EOF
