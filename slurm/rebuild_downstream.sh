#!/bin/bash
# Rebuild the downstream chain after dropping the py6.9b padded full ablation.
# New order: fsvd(24784776) → packed[3,4](24786175) → packed-rename → MEAN FLEET
# (loo_mean packed+padded, ablate_mean packed[7], ablate_mean padded[5 — OL7B DROPPED])
# → py6.9b padded [3,4] ONLY (dead last) → padded-rename.
# Keeps existing fsvd 24784776 + packed correction 24786175 (do not resubmit those).
set -euo pipefail
cd "$(dirname "$0")/.."
M7="OLMo-2-0425-1B OLMo-2-1124-7B pythia-1b-deduped pythia-6.9b-deduped pythia-160m-deduped pythia-410m-deduped nanochat-d12"
M6="OLMo-2-0425-1B OLMo-2-1124-7B pythia-1b-deduped pythia-6.9b-deduped pythia-160m-deduped pythia-410m-deduped"
M5="OLMo-2-0425-1B pythia-1b-deduped pythia-6.9b-deduped pythia-160m-deduped pythia-410m-deduped"   # padded ablate_mean: OL7B dropped
jid() { grep -oE '[0-9]+' | tail -1; }
jids() { grep -oE 'job [0-9]+' | grep -oE '[0-9]+' | paste -sd:; }

renjob() {  # $1=dep  $2=inf_dir  $3=res_dir  -> emits a rename job id
  sbatch --parsable --dependency="$1" <<EOF
#!/bin/bash
#SBATCH --partition=staging
#SBATCH --ntasks=1 --cpus-per-task=1 --time=00:10:00 --job-name=carrier_rename
#SBATCH --output=slurm/logs/carrier_rename_%j.out
cd "\$SLURM_SUBMIT_DIR" || exit 1
for pair in "data/inferences/$2:data/inferences/${2}_oldcarrier" "data/results/$3:data/results/${3}_oldcarrier"; do
  src="\${pair%%:*}"; dst="\${pair##*:}"
  if [[ -e "\$src" && ! -e "\$dst" ]]; then mv "\$src" "\$dst" && echo "renamed \$src -> \$dst"; else echo "skip \$src"; fi
done
EOF
}

# 1) packed rename after packed [3,4] correction (24786175)
PREN=$(renjob afterok:24786175 ablate_blk4-5 ablate_blk4-5); echo ">> packed-rename=$PREN (afterok:24786175)"
# 2) mean fleet, gated to start after packed-rename (staging barrier)
M1=$(./slurm/collect.sh configs/loo_mean_samples.yaml --models "$M7" --dependency=afterany:$PREN 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid); echo ">> M1 loo_mean packed=$M1"
M2=$(./slurm/collect.sh configs/loo_mean_samples_padded.yaml --models "$M6" --dependency=afterany:$M1 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid); echo ">> M2 loo_mean padded=$M2"
M3=$(CONFIG=configs/ablate_mean_samples.yaml OUT=data/inferences/ablate_mean ./slurm/ablate.sh --models "$M7" --time=24:00:00 --dependency=afterany:$M2 2>&1 | tee /dev/stderr | jids); echo ">> M3 ablate_mean packed=$M3"
M4=$(CONFIG=configs/ablate_mean_samples_padded.yaml OUT=data/inferences/ablate_mean_padded ./slurm/ablate.sh --models "$M5" --time=24:00:00 --dependency=afterany:$M3 2>&1 | tee /dev/stderr | jids); echo ">> M4 ablate_mean padded (no OL7B)=$M4"
# 3) py6.9b padded [3,4] ONLY — dead last, after the entire mean fleet
PY=$(sbatch --parsable --dependency=afterany:$M4 --time=12:00:00 <<'EOF'
#!/bin/bash
#SBATCH --partition=gpu_h100 --gpus=1 --ntasks=1 --cpus-per-task=16 --job-name=ablate34p_last
#SBATCH --output=slurm/logs/ablate34p_last_%j.out
#SBATCH --error=slurm/logs/ablate34p_last_%j.err
module purge; export HF_HOME="/projects/prjs1815/hf_cache"; export MALLOC_ARENA_MAX=2
cd "$SLURM_SUBMIT_DIR" || exit 1
time uv run scripts/collect.py --config configs/ablate_samples_padded.yaml --array_id 3 \
    --ablate "[[3,4]]" --output_dir data/inferences/ablate_padded
EOF
); echo ">> py6.9b padded [3,4] LAST=$PY (afterany:$M4)"
# 4) padded rename after the last-place py6.9b [3,4]
DREN=$(renjob afterok:$PY ablate_padded_blk4-5 ablate_padded_blk4-5); echo ">> padded-rename=$DREN (afterok:$PY)"
echo "REBUILD_DONE packed-rename=$PREN M1=$M1 M2=$M2 M3=$M3 M4=$M4 py6.9b34=$PY padded-rename=$DREN"
