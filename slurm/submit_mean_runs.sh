#!/bin/bash
# Submit the mean-ablation twins (loo_mean + ablate_mean, packed & padded), staged so no
# two jobs touch the same model's HF checkpoint cache at once (keep_cached:False). Run this
# ONLY after the zero-ablation chain is fully done (squeue clear of its jobs) — M1 starts
# immediately, M2-M4 chain via afterany. Packed = 7 models (incl nanochat); padded = 6.
set -euo pipefail
cd "$(dirname "$0")/.."
SENTINEL=data/results/.mean_runs_submitted
if [[ -e "$SENTINEL" ]]; then
    echo "mean runs already submitted ($(cat "$SENTINEL")) — skipping."; exit 0
fi
echo "submitted $(date -u +%Y-%m-%dT%H:%MZ)" > "$SENTINEL"    # claim before submitting (idempotency)
M7="OLMo-2-0425-1B OLMo-2-1124-7B pythia-1b-deduped pythia-6.9b-deduped pythia-160m-deduped pythia-410m-deduped nanochat-d12"
M6="OLMo-2-0425-1B OLMo-2-1124-7B pythia-1b-deduped pythia-6.9b-deduped pythia-160m-deduped pythia-410m-deduped"

jid() { grep -oE '[0-9]+' | tail -1; }             # last integer = job id (collect.sh: "Submitted batch job N")
jids() { grep -oE 'job [0-9]+' | grep -oE '[0-9]+' | paste -sd:; }   # ablate.sh: "... -> job N"

M1DEP="${M1_DEP:+--dependency=$M1_DEP}"    # optional gate so M1 waits until the zero chain is done
M1=$(./slurm/collect.sh configs/loo_mean_samples.yaml --models "$M7" $M1DEP 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid)
echo ">> M1 loo_mean packed = $M1 ${M1DEP}"

M2=$(./slurm/collect.sh configs/loo_mean_samples_padded.yaml --models "$M6" --dependency=afterany:$M1 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid)
echo ">> M2 loo_mean padded = $M2 (afterany:$M1)"

M3=$(CONFIG=configs/ablate_mean_samples.yaml OUT=data/inferences/ablate_mean \
     ./slurm/ablate.sh --models "$M7" --time=24:00:00 --dependency=afterany:$M2 2>&1 | tee /dev/stderr | jids)
echo ">> M3 ablate_mean packed = $M3 (afterany:$M2)"

M4=$(CONFIG=configs/ablate_mean_samples_padded.yaml OUT=data/inferences/ablate_mean_padded \
     ./slurm/ablate.sh --models "$M6" --time=24:00:00 --dependency=afterany:$M3 2>&1 | tee /dev/stderr | jids)
echo ">> M4 ablate_mean padded = $M4 (afterany:$M3)"
echo "MEAN_RUNS_SUBMITTED M1=$M1 M2=$M2 M3=$M3 M4=$M4"
