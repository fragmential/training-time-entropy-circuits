#!/bin/bash
# Prefetch the e2e models, then run the all-hooks e2e suite on a GPU node.
# These are real forward+backward passes; OLMo-2-1B is bf16 and unusably slow
# on CPU, so the suite needs a GPU.
# Usage: ./slurm/e2e.sh [extra pytest args...]
set -euo pipefail
mkdir -p slurm/logs

sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:30:00
#SBATCH --job-name=e2e_hooks
#SBATCH --output=slurm/logs/e2e_%j.out
#SBATCH --error=slurm/logs/e2e_%j.err

export HF_HOME="/projects/prjs1815/hf_cache"
export OMP_NUM_THREADS=4
cd "\$HOME/Tracing-representation-geometry-reproduction" || exit 1

echo "=== prefetch models ==="
uv run python -m tests.prefetch_e2e_models
echo "=== run e2e all-hooks suite ==="
uv run pytest tests/test_e2e_all_hooks.py -v -m e2e ${*:+$*}
EOF
