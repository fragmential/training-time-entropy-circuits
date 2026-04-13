#!/bin/bash
# Usage: ./slurm/collect.sh <config.yaml> [sbatch overrides...]
#        ./slurm/collect.sh <config.yaml> --models "pythia-14m pythia-70m" [sbatch overrides...]
set -euo pipefail
CONFIG="${1:?Usage: $0 <config.yaml> [--models \"name1 name2\"] [sbatch overrides...]}"
shift

# Comment out models to exclude them from this run.
MODELS=(
    "EleutherAI/pythia-14m"
    "EleutherAI/pythia-14m-deduped"
    "EleutherAI/pythia-31m"
    "EleutherAI/pythia-31m-deduped"
    "EleutherAI/pythia-70m"
    "EleutherAI/pythia-70m-deduped"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-160m-deduped"
    "EleutherAI/pythia-410m"
    "EleutherAI/pythia-410m-deduped"
    "EleutherAI/pythia-1b"
    "EleutherAI/pythia-1b-deduped"
    "EleutherAI/pythia-1.4b"
    "EleutherAI/pythia-1.4b-deduped"
    "EleutherAI/pythia-2.8b"
    "EleutherAI/pythia-2.8b-deduped"
    "EleutherAI/pythia-6.9b"
    "EleutherAI/pythia-6.9b-deduped"
    "EleutherAI/pythia-12b"
    "EleutherAI/pythia-12b-deduped"
    "allenai/OLMo-2-0425-1B"
    "allenai/OLMo-2-1124-7B"
)

# Override model list from CLI: --models "pythia-14m pythia-70m" (exact short-name match)
if [[ "${1:-}" == "--models" ]]; then
    shift; FILTER="$1"; shift
    FILTERED=()
    for M in "${MODELS[@]}"; do
        SHORT="${M##*/}"
        for F in $FILTER; do
            [[ "$SHORT" == "$F" ]] && FILTERED+=("$M") && break
        done
    done
    MODELS=("${FILTERED[@]}")
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "Error: no models selected." >&2
    exit 1
fi

# Resolve each model's index in the config's model_name list
CONFIG_INDICES=()
for MODEL in "${MODELS[@]}"; do
    CONFIG_INDICES+=("$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['model_name'].index('${MODEL}'))")")
done

INDICES_STR=$(printf "%s " "${CONFIG_INDICES[@]}")

sbatch --array=0-$((${#MODELS[@]} - 1)) "$@" <<EOF
#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/collect_%A_%a.out
#SBATCH --error=slurm/logs/collect_%A_%a.err

INDICES=(${INDICES_STR})
module purge
export HF_HOME="/projects/prjs1815/hf_cache"
cd "\$HOME/Tracing-representation-geometry-reproduction" || exit 1
time uv run scripts/collect.py --config "${CONFIG}" --array_id "\${INDICES[\$SLURM_ARRAY_TASK_ID]}"
EOF
