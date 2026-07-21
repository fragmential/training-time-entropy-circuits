#!/bin/bash
# Usage: ./slurm/ablate.sh [--models "name1 name2"] [sbatch overrides...]
# Layer-ablation sweep over configs/ablate_samples.yaml: ONE job per model; each
# checkpoint is loaded once and every intervention (the four L/4 chunks, the middle
# half, and the pythia carriers) runs as its own forward sweep, written to
# data/inferences/ablate_<blk-tag>/<model>.
set -euo pipefail
CONFIG="${CONFIG:-configs/ablate_samples.yaml}"
OUT="${OUT:-data/inferences/ablate}"

MODELS=("OLMo-2-0425-1B" "OLMo-2-1124-7B" "pythia-1b-deduped" "pythia-6.9b-deduped"
        "pythia-160m-deduped" "pythia-410m-deduped" "nanochat-d12")
declare -A L=([OLMo-2-0425-1B]=16 [OLMo-2-1124-7B]=32 [pythia-1b-deduped]=16
              [pythia-6.9b-deduped]=32 [pythia-160m-deduped]=12 [pythia-410m-deduped]=24
              [nanochat-d12]=12)
declare -A IDX=([OLMo-2-0425-1B]=0 [OLMo-2-1124-7B]=1 [pythia-1b-deduped]=2
                [pythia-6.9b-deduped]=3 [pythia-160m-deduped]=4 [pythia-410m-deduped]=5
                [nanochat-d12]=6)
declare -A CARRIER=([pythia-1b-deduped]="[3]" [pythia-6.9b-deduped]="[3,4]")
declare -A HOURS=([OLMo-2-0425-1B]=24 [OLMo-2-1124-7B]=72 [pythia-1b-deduped]=24
                  [pythia-6.9b-deduped]=72 [pythia-160m-deduped]=12
                  [pythia-410m-deduped]=16 [nanochat-d12]=12)

if [[ "${1:-}" == "--models" ]]; then
    shift; FILTER="$1"; shift
    SELECTED=()
    for M in "${MODELS[@]}"; do
        for F in $FILTER; do [[ "$M" == "$F" ]] && SELECTED+=("$M") && break; done
    done
    MODELS=("${SELECTED[@]}")
fi

for MODEL in "${MODELS[@]}"; do
    Q=$(( L[$MODEL] / 4 ))
    chunk() { seq -s, "$1" "$2" | sed 's/^/[/;s/$/]/'; }
    ABL="[$(chunk 0 $((Q-1))),$(chunk $Q $((2*Q-1))),$(chunk $((2*Q)) $((3*Q-1))),$(chunk $((3*Q)) $((L[$MODEL]-1))),$(chunk $Q $((3*Q-1)))${CARRIER[$MODEL]:+,${CARRIER[$MODEL]}}]"
    JID=$(sbatch --parsable --time="${HOURS[$MODEL]}:00:00" "$@" <<EOF
#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=ablate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/logs/ablate_%j.out
#SBATCH --error=slurm/logs/ablate_%j.err
module purge
export HF_HOME="/projects/prjs1815/hf_cache"
export MALLOC_ARENA_MAX=2
cd "\$SLURM_SUBMIT_DIR" || exit 1
time uv run scripts/collect.py --config "${CONFIG}" --array_id ${IDX[$MODEL]} \
    --ablate "${ABL}" --output_dir "${OUT}"
EOF
    )
    echo "$MODEL ${ABL} -> job $JID (${HOURS[$MODEL]}h)"
done
