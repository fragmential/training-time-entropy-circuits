#!/bin/bash
# Full chain rebuild with TIGHT walltimes so jobs fit before the 08:00 maintenance
# reservation (SLURM won't start a job whose walltime crosses into it). Realistic times,
# generous enough not to time out. Order: fsvd → packed[3,4] → packed-rename → loo_mean
# (packed, padded) → ablate_mean packed(7) → ablate_mean padded(4 small) → py6.9b padded
# [3,4] (last) → padded-rename. Both 7B padded mean-ablations dropped.
set -euo pipefail
cd "$(dirname "$0")/.."
M7="OLMo-2-0425-1B OLMo-2-1124-7B pythia-1b-deduped pythia-6.9b-deduped pythia-160m-deduped pythia-410m-deduped nanochat-d12"
M6="OLMo-2-0425-1B OLMo-2-1124-7B pythia-1b-deduped pythia-6.9b-deduped pythia-160m-deduped pythia-410m-deduped"
M4S="OLMo-2-0425-1B pythia-1b-deduped pythia-160m-deduped pythia-410m-deduped"   # padded ablate_mean: 4 small (both 7B dropped)
jid() { grep -oE '[0-9]+' | tail -1; }
jids() { grep -oE 'job [0-9]+' | grep -oE '[0-9]+' | paste -sd:; }
renjob() {  # $1=dep $2=inf/res basename
  sbatch --parsable --dependency="$1" <<EOF
#!/bin/bash
#SBATCH --partition=staging --ntasks=1 --cpus-per-task=1 --time=00:10:00 --job-name=carrier_rename
#SBATCH --output=slurm/logs/carrier_rename_%j.out
cd "\$SLURM_SUBMIT_DIR" || exit 1
for pair in "data/inferences/$2:data/inferences/${2}_oldcarrier" "data/results/$2:data/results/${2}_oldcarrier"; do
  s="\${pair%%:*}"; d="\${pair##*:}"; [[ -e "\$s" && ! -e "\$d" ]] && mv "\$s" "\$d" && echo "renamed \$s" || echo "skip \$s"
done
EOF
}
corrjob() {  # $1=dep $2=config $3=outdir $4=time  -> py6.9b [3,4] single-intervention
  sbatch --parsable --dependency="$1" --time="$4" <<EOF
#!/bin/bash
#SBATCH --partition=gpu_h100 --gpus=1 --ntasks=1 --cpus-per-task=16 --job-name=ablate34
#SBATCH --output=slurm/logs/ablate34_%j.out
#SBATCH --error=slurm/logs/ablate34_%j.err
module purge; export HF_HOME="/projects/prjs1815/hf_cache"; export MALLOC_ARENA_MAX=2
cd "\$SLURM_SUBMIT_DIR" || exit 1
time uv run scripts/collect.py --config $2 --array_id 3 --ablate "[[3,4]]" --output_dir $3
EOF
}

FSVD=$(./slurm/collect.sh configs/final_stream_svd_padded.yaml --models "$M6" --time=05:00:00 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid); echo ">> fsvd=$FSVD (5h, no dep)"
PC=$(corrjob afterany:$FSVD configs/ablate_samples.yaml data/inferences/ablate 05:00:00); echo ">> packed[3,4]=$PC (afterany:$FSVD,5h)"
PREN=$(renjob afterok:$PC ablate_blk4-5); echo ">> packed-rename=$PREN"
M1=$(./slurm/collect.sh configs/loo_mean_samples.yaml --models "$M7" --time=06:00:00 --dependency=afterany:$PREN 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid); echo ">> M1 loo_mean packed=$M1 (6h)"
M2=$(./slurm/collect.sh configs/loo_mean_samples_padded.yaml --models "$M6" --time=09:00:00 --dependency=afterany:$M1 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid); echo ">> M2 loo_mean padded=$M2 (9h)"
M3=$(CONFIG=configs/ablate_mean_samples.yaml OUT=data/inferences/ablate_mean ./slurm/ablate.sh --models "$M7" --time=12:00:00 --dependency=afterany:$M2 2>&1 | tee /dev/stderr | jids); echo ">> M3 ablate_mean packed=$M3 (12h)"
M4=$(CONFIG=configs/ablate_mean_samples_padded.yaml OUT=data/inferences/ablate_mean_padded ./slurm/ablate.sh --models "$M4S" --time=08:00:00 --dependency=afterany:$M3 2>&1 | tee /dev/stderr | jids); echo ">> M4 ablate_mean padded(4 small)=$M4 (8h)"
PY=$(corrjob afterany:$M4 configs/ablate_samples_padded.yaml data/inferences/ablate_padded 07:00:00); echo ">> py6.9b padded[3,4] LAST=$PY (7h)"
DREN=$(renjob afterok:$PY ablate_padded_blk4-5); echo ">> padded-rename=$DREN"
echo "REBUILD_ALL_DONE fsvd=$FSVD pc=$PC pren=$PREN M1=$M1 M2=$M2 M3=$M3 M4=$M4 py=$PY dren=$DREN"
