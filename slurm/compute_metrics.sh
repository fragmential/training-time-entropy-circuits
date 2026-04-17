#!/bin/bash
# Usage:
#   ./slurm/compute_metrics.sh <config_dir | path/to/config_dir> [model_name ...] [extra args...]
#
# Examples:
#   ./slurm/compute_metrics.sh rankme_alpha_packed
#   ./slurm/compute_metrics.sh rankme_alpha_packed --recompute
#   ./slurm/compute_metrics.sh rankme_alpha_packed pythia-1b-deduped
#   ./slurm/compute_metrics.sh rankme_alpha_packed pythia-1b-deduped pythia-6.9b-deduped
#   ./slurm/compute_metrics.sh activations/rankme_alpha_packed
#   ./slurm/compute_metrics.sh /home/data/activations/custom_runs pythia-31m-deduped --recompute

set -euo pipefail

INPUT="${1:?Usage: $0 <config_dir | path/to/config_dir> [model_name ...] [extra args...]}"
shift

# Consume all non-flag arguments as model names.
MODEL_NAMES=()
while [[ $# -gt 0 && "$1" != -* ]]; do
    MODEL_NAMES+=("$1")
    shift
done

EXTRA_ARGS=("$@")

mkdir -p slurm/logs

CONFIG_DIR="$INPUT"
SCAN_DIR="$CONFIG_DIR"
[[ "$CONFIG_DIR" != */* ]] && SCAN_DIR="inferences/$CONFIG_DIR"

# Build model list.
MODELS=()
if [[ ${#MODEL_NAMES[@]} -gt 0 ]]; then
    MODELS=("${MODEL_NAMES[@]}")
else
    for D in "$SCAN_DIR"/*; do
        [[ -d "$D" ]] || continue
        MODELS+=("$(basename "$D")")
    done
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: no models selected." >&2
    exit 1
fi

MODELS_STR=$(printf '%q ' "${MODELS[@]}")
EXTRA_STR=""
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    EXTRA_STR=$(printf ' %q' "${EXTRA_ARGS[@]}")
fi
sbatch --array=0-$((${#MODELS[@]} - 1)) <<EOF
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/metrics_%A_%a.out

MODELS=(${MODELS_STR})
MODEL_NAME="\${MODELS[\$SLURM_ARRAY_TASK_ID]}"
export HF_HOME="/projects/prjs1815/hf_cache"

cd "\$HOME/Tracing-representation-geometry-reproduction" || exit 1

echo "Config directory: ${CONFIG_DIR}"
echo "Model: \$MODEL_NAME"

time uv run scripts/compute_metrics.py \
    "${CONFIG_DIR}" \
    "\$MODEL_NAME"${EXTRA_STR}
EOF