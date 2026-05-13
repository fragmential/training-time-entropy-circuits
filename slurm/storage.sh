#!/bin/bash
# SLURM wrapper for python -m utils.accessor — one job per model subdir (or single job if --input points to a model dir).
# Usage: ./slurm/storage.sh <command> [flags...] --input <path>
# Examples:
#   ./slurm/storage.sh convert --to cov_svd --input inferences/full_limited
#   ./slurm/storage.sh project --onto both --input inferences/full_limited/pythia-31m-deduped
#   ./slurm/storage.sh set-filter --token-selection last --input inferences/full_limited

set -euo pipefail

mkdir -p slurm/logs

# Split --input from remaining args
INPUT_PATH=""
CMD_ARGS=()
i=0; ARGS=("$@")
while [ $i -lt ${#ARGS[@]} ]; do
    if [ "${ARGS[$i]}" = "--input" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        INPUT_PATH="${ARGS[$((i+1))]}"; i=$((i+2))
    else
        CMD_ARGS+=("${ARGS[$i]}"); i=$((i+1))
    fi
done

if [ -z "$INPUT_PATH" ]; then
    echo "Error: --input <path> is required" >&2; exit 1
fi

# Model dir (.pt files) or dataset dir (subdirs)
if compgen -G "${INPUT_PATH}/*.pt" > /dev/null 2>&1; then
    MODEL_DIRS=("$INPUT_PATH")
else
    mapfile -t MODEL_DIRS < <(find "$INPUT_PATH" -maxdepth 1 -mindepth 1 -type d | sort)
    [[ ${#MODEL_DIRS[@]} -eq 0 ]] && { echo "Error: no model subdirs in $INPUT_PATH" >&2; exit 1; }
fi

N=${#MODEL_DIRS[@]}
echo "Input: $INPUT_PATH | $N model dir(s)"

DIRS_STR=$(printf '%q ' "${MODEL_DIRS[@]}")
QUOTED_ARGS=$(printf ' %q' "${CMD_ARGS[@]}")

sbatch --array=0-$((N-1)) <<EOF
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --job-name=storage_${CMD_ARGS[0]:-misc}
#SBATCH --output=slurm/logs/storage_%A_%a.out

MODEL_DIRS=(${DIRS_STR})
MODEL_DIR="\${MODEL_DIRS[\$SLURM_ARRAY_TASK_ID]}"
export HF_HOME="/projects/prjs1815/hf_cache"

cd "\$HOME/Tracing-representation-geometry-reproduction" || exit 1
export OMP_NUM_THREADS=1
echo "Processing: \$MODEL_DIR"
time uv run python -m utils.accessor${QUOTED_ARGS} --input "\$MODEL_DIR" --workers 16
EOF
