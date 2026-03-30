#!/bin/bash
# SLURM wrapper for python -m utils.storage
#
# Usage: ./slurm/storage.sh <command> [flags...] --input <path>
#
# If <path> is a dataset directory (contains model subdirs):
#   submits a SLURM array, one job per model subdir.
# If <path> is a model directory (contains .pt files):
#   submits a single job.
#
# Each job uses the staging partition with 16 cores; the Python script
# pools over checkpoints within its assigned directory.
#
# Examples:
#   ./slurm/storage.sh convert --to cov_svd --input activations/fineweb
#   ./slurm/storage.sh project --onto both --input activations/fineweb/pythia-31m-deduped
#   ./slurm/storage.sh set-filter --token-selection last --input collected/native/pythia-70m

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

mkdir -p slurm/logs

# --- Extract --input value and rebuild args without it ---
ARGS=("$@")
INPUT_PATH=""
CMD_ARGS=()
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    if [ "${ARGS[$i]}" = "--input" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        INPUT_PATH="${ARGS[$((i+1))]}"
        i=$((i+2))
    else
        CMD_ARGS+=("${ARGS[$i]}")
        i=$((i+1))
    fi
done

if [ -z "$INPUT_PATH" ]; then
    echo "Error: --input <path> is required" >&2
    exit 1
fi

# --- Detect: model dir (contains .pt files) vs dataset dir (contains subdirs) ---
if compgen -G "${INPUT_PATH}/*.pt" > /dev/null 2>&1; then
    # Model directory
    MODEL_DIRS=("$INPUT_PATH")
else
    # Dataset directory — find model subdirs
    mapfile -t MODEL_DIRS < <(find "$INPUT_PATH" -maxdepth 1 -mindepth 1 -type d | sort)
    if [ ${#MODEL_DIRS[@]} -eq 0 ]; then
        echo "Error: no model subdirectories found in $INPUT_PATH" >&2
        exit 1
    fi
fi

N=${#MODEL_DIRS[@]}
echo "Input: $INPUT_PATH | $N model director(ies)"

# Write model list to temp file for array indexing
MODELS_TXT=$(mktemp /tmp/storage_models_XXXX.txt)
printf '%s\n' "${MODEL_DIRS[@]}" > "$MODELS_TXT"

# --- Quote args for embedding in the job ---
QUOTED_ARGS=$(printf ' %q' "${CMD_ARGS[@]}")

# --- Build array spec ---
ARRAY_ARG=""
[ $N -gt 1 ] && ARRAY_ARG="--array=0-$((N-1))"

JOB_NAME="storage_${CMD_ARGS[0]:-misc}"

sbatch $ARRAY_ARG \
    --partition=staging \
    --ntasks=1 \
    --cpus-per-task=16 \
    --time=04:00:00 \
    --job-name="$JOB_NAME" \
    --output="$REPO_DIR/slurm/logs/storage_%j_%a.out" \
    --wrap="
set -euo pipefail
cd '$REPO_DIR'
export HF_HOME='/projects/prjs1815/hf_cache'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}
MODEL_DIR=\$(awk \"NR==\$((TASK_ID+1))\" '$MODELS_TXT')
echo \"Processing: \$MODEL_DIR\"
python -m utils.storage $QUOTED_ARGS --input \"\$MODEL_DIR\" --workers 16
echo \"Done: \$MODEL_DIR\"
"

echo "Submitted $N job(s). Model list: $MODELS_TXT"
