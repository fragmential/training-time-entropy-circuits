#!/bin/bash
# SLURM wrapper for python -m utils.accessor — one job per model subdir (or single job if --input points to a model dir).
# Usage: ./slurm/storage.sh <command> [flags...] --input <path> [--output-dir <path>]
# In-place by default; --output-dir writes results to <output-dir>/<model_name>/ instead.
# Examples:
#   ./slurm/storage.sh convert --to cov_svd --input data/inferences/full_limited
#   ./slurm/storage.sh convert --to eigenvalues --input data/inferences/full_limited --output-dir data/inferences/full_limited_eig
#   ./slurm/storage.sh project --onto-file data/inferences/full_limited/pythia-31m-deduped/step143000.pt --input data/inferences/full_limited/pythia-31m-deduped

set -euo pipefail

mkdir -p slurm/logs

# Split --input and --output-dir from remaining args
INPUT_PATH=""
OUTPUT_BASE=""
CMD_ARGS=()
i=0; ARGS=("$@")
while [ $i -lt ${#ARGS[@]} ]; do
    if [ "${ARGS[$i]}" = "--input" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        INPUT_PATH="${ARGS[$((i+1))]}"; i=$((i+2))
    elif [ "${ARGS[$i]}" = "--output-dir" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        OUTPUT_BASE="${ARGS[$((i+1))]}"; i=$((i+2))
    else
        CMD_ARGS+=("${ARGS[$i]}"); i=$((i+1))
    fi
done

if [ -z "$INPUT_PATH" ]; then
    echo "Error: --input <path> is required" >&2; exit 1
fi

# Loud warnings when a `convert` discards data (irreversible if in-place)
CMD="${CMD_ARGS[0]:-}"
TO_FMT=""; MATERIALIZE=""
for ((j=0; j<${#CMD_ARGS[@]}; j++)); do
    [ "${CMD_ARGS[$j]}" = "--to" ] && TO_FMT="${CMD_ARGS[$((j+1))]:-}"
    [ "${CMD_ARGS[$j]}" = "--materialize" ] && MATERIALIZE="${CMD_ARGS[$((j+1))]:-}"
done
if [ "$CMD" = "convert" ]; then
    WARNED=0

    # eigenvalues keeps only eigenvalues -> covariances + eigenvectors are gone
    if [ "$TO_FMT" = "eigenvalues" ] || [ "$TO_FMT" = "eigvals" ]; then
        cat >&2 <<'EOF'

WARNING:  --to eigenvalues  DISCARDS the COVARIANCES and EIGENVECTORS
Only eigenvalues are kept  ->  no reconstruction, no reprojection onto other bases.
EOF
        WARNED=1
    fi

    # negative --materialize overrides drop derived families (B = MLP-out acts, O = head contrib)
    if [[ "$MATERIALIZE" == *"-b"* ]] || [[ "$MATERIALIZE" == *"-o"* ]]; then
        cat >&2 <<'EOF'

WARNING:  --materialize -b / -o drops the derived B (MLP-out) / O (head-contrib)
families where they would otherwise be preserved.
EOF
        WARNED=1
    fi

    if [ "$WARNED" = "1" ]; then
        if [ -n "$OUTPUT_BASE" ]; then
            echo ">>> Mode: writing to $OUTPUT_BASE — source files are PRESERVED." >&2
        else
            echo ">>> Mode: IN-PLACE — the above are PERMANENTLY REMOVED from the source files." >&2
        fi
        if [ -t 0 ]; then
            printf '>>> Type "proceed" to continue (anything else aborts): ' >&2
            read -r CONFIRM
            if [ "$CONFIRM" != "proceed" ]; then
                echo ">>> Aborted." >&2; exit 1
            fi
        else
            echo ">>> No TTY to confirm a lossy convert — aborting." >&2
            echo ">>> Re-run interactively, or use --output-dir while testing." >&2
            exit 1
        fi
    fi
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

WORKERS_ARG=""; [ "$CMD" != "info" ] && WORKERS_ARG="--workers 16"

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
OUTPUT_BASE="${OUTPUT_BASE}"
export HF_HOME="/projects/prjs1815/hf_cache"

cd "\$HOME/Tracing-representation-geometry-reproduction" || exit 1
export OMP_NUM_THREADS=1
echo "Processing: \$MODEL_DIR"
if [ -n "\$OUTPUT_BASE" ]; then
    OUT_DIR="\$OUTPUT_BASE/\$(basename "\$MODEL_DIR")"
    echo "Output: \$OUT_DIR"
    time uv run python -m utils.accessor${QUOTED_ARGS} --input "\$MODEL_DIR" --output-dir "\$OUT_DIR" ${WORKERS_ARG}
else
    time uv run python -m utils.accessor${QUOTED_ARGS} --input "\$MODEL_DIR" ${WORKERS_ARG}
fi
EOF
