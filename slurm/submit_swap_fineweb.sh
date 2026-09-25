#!/bin/bash
# Cheap bfn/afn-only runs to extend the padded-vs-packed figure (experiments_padded):
#   1) packed fineweb  (py1b, py6.9b, OL1B, OL7B) — pairs with existing padded fineweb
#   2) padded swap     (py1b, OL1B)               — pairs with existing packed swap
# py1b & OL1B appear in BOTH; with keep_cached:False concurrent same-model jobs corrupt each
# other's HF checkpoint cache, so (2) is gated afterany on the WHOLE (1) array. Idempotent via
# a sentinel so a re-run (post-crash) does not double-submit.
set -euo pipefail
cd "$(dirname "$0")/.."
SENTINEL="data/results/.swap_fineweb_submitted"
if [[ -e "$SENTINEL" ]]; then echo "Already submitted:"; cat "$SENTINEL"; exit 0; fi
jid() { grep -oE '[0-9]+' | tail -1; }

FW=$(./slurm/collect.sh configs/block_representations_fineweb_packed.yaml \
        --models "pythia-1b-deduped pythia-6.9b-deduped OLMo-2-0425-1B OLMo-2-1124-7B" \
        --time=03:00:00 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid)
echo ">> packed fineweb array = $FW"

SW=$(./slurm/collect.sh configs/block_representations_samples_swap_padded.yaml \
        --models "pythia-1b-deduped OLMo-2-0425-1B" \
        --time=02:00:00 --dependency="afterany:$FW" 2>&1 | tee /dev/stderr | grep "Submitted batch job" | jid)
echo ">> padded swap array = $SW (afterany:$FW)"

printf "packed_fineweb=%s\npadded_swap=%s (afterany:%s)\n" "$FW" "$SW" "$FW" | tee "$SENTINEL"
echo "SWAP_FINEWEB_SUBMIT_DONE fineweb=$FW swap=$SW"
