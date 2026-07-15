#!/bin/bash
# Benchmark /scratch-shared vs /projects/prjs1815 for the HF-cache access pattern:
# bulk write/read (dd, O_DIRECT = page-cache-free), many-small-file metadata, and a real
# snapshot_download. Both bench dirs are created and removed by this script.
set -uo pipefail
export HF_HOME_BASE_A="/scratch-shared/$USER/iobench"
export HF_HOME_BASE_B="/projects/prjs1815/iobench"

bench_fs() {
    local dir="$1"; mkdir -p "$dir"
    echo "=== $dir ==="
    echo "-- bulk write 4GiB (O_DIRECT):"
    dd if=/dev/zero of="$dir/big.bin" bs=16M count=256 oflag=direct 2>&1 | tail -1
    echo "-- bulk read 4GiB (O_DIRECT):"
    dd if="$dir/big.bin" of=/dev/null bs=16M iflag=direct 2>&1 | tail -1
    echo "-- 2000 x 8KiB small files: create / stat / read (seconds):"
    mkdir -p "$dir/small"
    /usr/bin/time -f "create: %e s" bash -c 'for i in $(seq 1 2000); do head -c 8192 /dev/zero > '"$dir"'/small/f$i; done'
    /usr/bin/time -f "stat:   %e s" bash -c 'stat '"$dir"'/small/f* > /dev/null'
    /usr/bin/time -f "read:   %e s" bash -c 'cat '"$dir"'/small/f* > /dev/null'
}

bench_hf() {
    local dir="$1" label="$2"
    echo "-- snapshot_download pythia-1b-deduped ($label):"
    HF_HOME="$dir/hf" uv run python -c "
import time
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download('EleutherAI/pythia-1b-deduped')
print(f'download: {time.time()-t:.1f} s')"
}

cd "$SLURM_SUBMIT_DIR" || exit 1
bench_fs "$HF_HOME_BASE_A"
bench_fs "$HF_HOME_BASE_B"
# alternate download order across the two rounds to average out network variability
bench_hf "$HF_HOME_BASE_A" "scratch, round 1"
bench_hf "$HF_HOME_BASE_B" "projects, round 1"
rm -rf "$HF_HOME_BASE_A/hf"
rm -rf "$HF_HOME_BASE_B/hf"
bench_hf "$HF_HOME_BASE_B" "projects, round 2"
bench_hf "$HF_HOME_BASE_A" "scratch, round 2"

rm -rf "$HF_HOME_BASE_A"
rm -rf "$HF_HOME_BASE_B"
echo "done (bench dirs removed)"
