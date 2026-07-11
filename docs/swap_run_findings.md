# Dataset-swap control — block_representations_samples_swap

**Question:** are the family differences in docs/dig_findings.md model-dependent or just
the datasets? Each 1B model was re-run on the *other* family's pretraining mix:
pythia-1b-deduped on olmo-mix text, OLMo-2-0425-1B on Pile text.

**Setup.** Config `configs/block_representations_samples_swap.yaml`, mirroring
`block_representations_samples.yaml` (packed, all-token, 262,144 tokens, 512 seq_len,
same hooks, ~50-58 log-spaced checkpoints, eigenvalues storage, inline metrics; no drift).
Because the tokenizers differ (GPT-NeoX ~50k vs OLMo-2 cl100k ~100k), the pre-packed
mixes' token ids are not portable: cross-mixes were built by decoding the first 1,500
chunks of each original mix with its source tokenizer and re-tokenizing with the target
model's tokenizer (`data/mixes/olmo_mix_pythiatok_512.pt`, `data/mixes/pile_olmotok_512.pt`).
Text content is therefore identical to the originals' head; only the model (and its
tokenizer) changes. All numbers below computed by the same script applied to both the
main-run and swap results files (baselines reproduce docs/dig_findings.md exactly).

## Verdict table

| # | Finding (dig_findings) | pythia-1b original (Pile) | pythia-1b **swap** (olmo-mix) | OLMo-1B original (olmo-mix) | OLMo-1B **swap** (Pile) | Verdict |
|---|---|---|---|---|---|---|
| 1 | Compression driver: Σquality trajectory | −0.54 → min **−6.34** (final −6.34) | −0.48 → min **−6.35** (final −6.34) | dip −3.96@5k → recovers to **−0.95** | dip −4.05@5k → recovers to **−1.07** | **MODEL** |
| 1 | Σinterference trajectory | −3.48 → **+2.33** (crosses 0) | −3.54 → **+2.49** (crosses 0) | → **−4.16** (falling) | → **−4.36** (falling) | **MODEL** |
| 1 | Quality locus | blk3 carries it (−6.26 of −6.34) | blk3 carries it (−6.25 of −6.34) | diffuse (worst −0.31) | diffuse (worst −0.40) | **MODEL** |
| 2 | Rogue write: min final write RankMe | blk3: **1.2** | blk3: **1.2** | blk14: 352 (no block < 100) | blk14: 284 (no block < 100) | **MODEL** |
| 2 | Rogue trace ratio (write/stream, final) | blk3: **22.6×** | blk3: **24.4×** | max 0.50 (blk14) | max 0.50 (blk14) | **MODEL** |
| 2 | Collapse timing | 46→2.5 over 2k→5k (RankMe peak) | 46→2.4 over 2k→5k (RankMe peak) | — | — | **MODEL** |
| 3 | Head phenomenon: k=0 RankMe (before_final_norm) | 555@4k → 194 (−65%) | 559@6k → 192 (−66%) | 822 → 447 (−46%) | 823 → 389 (−53%) | **MODEL** (magnitudes) |
| 3 | k=32-removed RankMe over same span | 1226 → 1361 (**+11%**, bulk expands) | 1313 → 1339 (**+2%**, bulk ~flat) | 1058 → 1417 (**+34%**) | 1049 → 1389 (**+32%**) | **MODEL** (universal head-confinement) |
| 4 | Last-two-mlp-writes signed_trace @final | blk14~15: **+0.52** | blk14~15: **+0.56** | blk14~15: **+0.32** | blk14~15: **+0.32** | **MODEL** |
| 4 | Rogue cancellation: blk3.mlp ~ blk15.mlp | **−0.64** (min −0.69) | **−0.67** (min −0.72) | n/a (no rogue) | n/a (no rogue) | **MODEL** |

## Verdict: everything survives the swap — the family split is MODEL-dependent

Every headline finding replicates on the other family's data, to within a few percent:

1. **Ledger driver (MODEL).** Pythia-on-olmo-mix is still purely quality-driven
   (Σquality −0.48 → −6.35, Σinterference crossing zero to +2.49 near the RankMe peak,
   blk3 carrying ~98% of the final quality deficit — vs 22.6×/−6.26 on Pile). OLMo-on-Pile
   is still interference-driven with recovering quality (dip −4.05@5k → −1.07 final;
   interference falls to −4.36, concentrated in late blocks 15/9/11). The
   quality-vs-interference family split is a property of the models, not the corpora.

2. **Rogue write (MODEL).** Pythia grows the *same* blk3 rank-one write on olmo-mix data:
   final write RankMe 1.2, trace ratio 24.4× (vs 22.6× on Pile), with the identical
   collapse trajectory — healthy (~440) until step 1k, 46 @ 2k, ~2.4 @ 5k, exactly at the
   RankMe peak, then monotone energy growth. Even the secondary late-block low-rank writes
   (blk14 ≈ 15, blk15 ≈ 11) reappear at the same blocks with the same ranks. OLMo on Pile
   develops **no** rogue write: all write RankMe ≥ 284, max trace ratio 0.50. Note the
   newline-direction identification (dig_findings) is consistent with this: olmo-mix text also
   contains newlines, so a data-side account would need a corpus *without* the trigger
   token — but the exact replication of onset step and magnitude on a very different mix
   (94.9% dclm web text vs Pile) already rules out Pile-specific content as the cause.

3. **Head-confined compression (MODEL, and universal).** In both swap legs the measured
   RankMe compression stays confined to the spectrum head: pythia k=0 falls −66% from peak
   while k=32-removed moves +2% (original: −65% / +11%); OLMo k=0 −53% while k=32 +32%
   (original: −46% / +34%). The "bulk never compresses" claim survives the swap in both
   directions. (Pythia's bulk *rise* is somewhat flatter on olmo-mix, +2% vs +11% — the
   only readout where the dataset visibly modulates a magnitude; the sign/structure is
   unchanged.)

4. **Late-write alignment (MODEL).** OLMo's mutually aligned final writes replicate
   exactly (blk14~blk15 signed_trace +0.321 on Pile vs +0.320 on olmo-mix). Pythia's
   anti-aligned rogue cancellation also replicates (blk3~blk15 −0.665 on olmo-mix vs
   −0.637 on Pile; trajectory 0 → min −0.72).

**What these verdicts support.** The dig_findings family split — rogue-write
quality-compression (Pythia) vs distributed aligned-late-write interference-compression
(OLMo-2) — is attributable to the models (architecture + optimization + tokenizer), not to
Pile-vs-olmo-mix data content. Specifically: H1.1's family split, the blk3 newline-activated
mechanism and its late-block cancellation, the absence of a rogue write in OLMo-2, and the
head-confinement universal all survive. **Limits:** the swap keeps everything about the
model fixed (including tokenizer and its interaction with the text), so it separates
*data content* from *model*, not architecture from tokenizer from training recipe; and
the checkpoints are the released pretrained ones — the swap changes the *evaluation*
distribution, not the training data. It answers "would these findings have appeared had we
merely probed with the other corpus?" — no reading of the findings depends on which mix
was used to probe.

**Nothing here is DATA-dependent or mixed.** The largest dataset effects observed are
~±8% on the rogue trace ratio (22.6 → 24.4×), −13% on OLMo's final k=0 RankMe (447 → 389),
and pythia's bulk-rise flattening (+11% → +2%) — all sign-preserving modulations of
magnitude, none of them alter any claim.

## Run inventory / provenance

- Config: `configs/block_representations_samples_swap.yaml` (packed_data_path per family).
- Cross-mixes: `data/mixes/olmo_mix_pythiatok_512.pt` (1572×512), `data/mixes/pile_olmotok_512.pt`
  (1398×512); both runs use the first 512 chunks (262,144 tokens).
- Collections: `data/inferences/block_representations_samples_swap/{pythia-1b-deduped,OLMo-2-0425-1B}/`
  (58 / 62 checkpoints, eigenvalues format).
- Metrics: `data/results/block_representations_samples_swap/results_*.npy` (58 / 62 steps).
- Jobs: 24474092 (died at ~ckpt 17 on disk quota — `keep_cached: True` hoarded checkpoint
  revisions; config flipped to `keep_cached: False`), 24557576 (resumed, completed).
  OLMo steps 30000/70000/80000 were truncated by the quota crash and recollected in
  job 24563516; they land smoothly on the surrounding trajectories (blk14 write RankMe
  492 / 377 / 370) and change no table entry.
