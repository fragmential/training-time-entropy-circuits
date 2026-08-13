#!/bin/bash
# content_tokens run: bfn + afn + the EXACT MIDDLE residual (blk{L/2}.attn.in, per model), on
# interior content tokens (exclude first-of-window / last / delimiters). Per-model hooks because
# the middle block index differs by depth; array order matches configs/content_tokens.yaml
# model_name = [410m(24), 1b(16), 6.9b(32), nanochat-d12(12), OL1B(16), OL7B(32)] -> L/2 below.
set -euo pipefail
cd "$(dirname "$0")/.."
sbatch --array=0-5 <<'EOF'
#!/bin/bash
#SBATCH --partition=gpu_h100 --gpus=1 --ntasks=1 --cpus-per-task=16 --time=04:00:00 --job-name=content_tok
#SBATCH --output=slurm/logs/content_%A_%a.out --error=slurm/logs/content_%A_%a.err
module purge; export HF_HOME="/projects/prjs1815/hf_cache"; export MALLOC_ARENA_MAX=2
cd "$SLURM_SUBMIT_DIR" || exit 1
MID=(12 8 16 6 8 16)                     # L/2 per array index (config model order)
i=$SLURM_ARRAY_TASK_ID
time uv run scripts/collect.py --config configs/content_tokens.yaml --array_id $i \
  --hooks "[\"before_final_norm\",\"after_final_norm\",\"blk${MID[$i]}.attn.in\"]"
EOF
