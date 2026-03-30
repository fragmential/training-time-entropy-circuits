#!/bin/bash
# Usage:
#   ./slurm/compute_metrics.sh <config_dir | path/to/config_dir> [model_name] [extra args...]
#
# Examples:
#   ./slurm/compute_metrics.sh rankme_alpha_packed
#   ./slurm/compute_metrics.sh rankme_alpha_packed --recompute
#   ./slurm/compute_metrics.sh rankme_alpha_packed pythia-1b-deduped
#   ./slurm/compute_metrics.sh activations/rankme_alpha_packed
#   ./slurm/compute_metrics.sh /home/data/activations/custom_runs pythia-31m-deduped --recompute

set -euo pipefail

INPUT="${1:?Usage: $0 <config_dir | path/to/config_dir> [model_name] [extra args...]}"
shift

# If the next argument does not start with -, treat it as model_name.
MODEL_NAME=""
if [[ $# -gt 0 && "$1" != -* ]]; then
    MODEL_NAME="$1"
    shift
fi

EXTRA_ARGS=("$@")

mkdir -p slurm/logs

CONFIG_DIR="$INPUT"
SCAN_DIR="$CONFIG_DIR"
[[ "$CONFIG_DIR" != */* ]] && SCAN_DIR="inferences/$CONFIG_DIR"

# Build model list.
MODELS=()
if [[ -n "$MODEL_NAME" ]]; then
    MODELS=("$MODEL_NAME")
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

cd "\$HOME/Tracing-representation-geometry-reproduction" || exit 1

echo "Config directory: ${CONFIG_DIR}"
echo "Model: \$MODEL_NAME"

time uv run scripts/compute_metrics.py \
    "${CONFIG_DIR}" \
    "\$MODEL_NAME"${EXTRA_STR}
EOF