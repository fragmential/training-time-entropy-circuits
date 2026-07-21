# Plan (living document)

The only forward-looking doc: in-flight work, ranked next steps, deferred decisions.
Results NEVER live here — they go to final_report.md (headline) and the topic docs
(evidence). Collection-specific additions for the next sweep:
collection/next_run_additions.md.

## In flight (SLURM, Jul 20 — chained: each waits on the previous)

- LOO resume 24758742 (first pass completed for small models; big models were
  quota-starved and resume here after the 129G HF-cache cleanup).
- final_stream_svd resume 24758743.
- Ablation fleet 24758749–24758755: ONE job per model, interventions looped per
  checkpoint (4 quarter-chunks + middle half + carriers blk3/blk4_5), outputs
  data/inferences/ablate_<blk-tag>/.
- unembed 24758744 (after the fleet; now deletes each snapshot + cached head after use).
- gdrive rq2 backup 24758745 (staging): all rq2_* inference dirs →
  gdrive:msc-backup-data/<name>.
- unembed vectors+alignment 24771983 (gpu_h100): head right-singular-vectors re-fetch
  (data/results/unembedding_svd) then the alignment products →
  data/results/unembedding_alignment.pt.

## Next (ranked)

1. UNEMBEDDING PROGRAM (added Jul 19; commanded — runnable without further sign-off).
   Test the conditional-final-layers interpretation (final_report §2.3, RQ1c) from the
   unembedding side:
   (a) unembedding singular spectra across checkpoints, all models — weights-only via
   data/weight_cache, no forward passes;
   (b) final-stream ↔ unembedding row-space alignment across training: small dedicated
   collection storing cov_svd at before/after_final_norm only (eigvecs needed; the samples
   sweeps persist eigenvalues only) — COLLECTED (data/inferences/final_stream_svd).
   Remaining: the head's right singular VECTORS were not saved (only values; cached heads
   deleted) → one re-fetch pass saving top-k right singular vectors, then the alignment as
   the share of centered stream variance in the head's top-k subspace, tr(P·Σc)/tr(Σc) —
   Σc reconstructs from the stored eigvecs/eigvals/mean, so matrix products only, no new
   eigendecompositions.
   Prediction if RQ1c's leading interpretation holds: alignment tightens through the compression phase.
2. COUNTERFACTUAL PROGRAM (added Jul 19; design set, big runs need job sign-off). Two
   rungs of counterfactualness for layer attribution:
   (a) WEAK — leave-one-out final entropy. CORRECTION (Jul 20): the single-write version
   already exists — `ablation_contribution` (compute_metrics.py) computes
   S(final − c_k) per attn/mlp write in covariance space and is in the past sweep
   results; the earlier "confirmed absent" note was wrong. The new piece is the GROUP
   version (`loo` metric, sample-space): every block jointly + the four L/4 chunks +
   the middle half, groups passed down from the caller (loo_groups). Runs as
   configs/loo_samples.yaml — all 7 models, 20 log checkpoints, drift off, cross-term
   metrics skipped.
   (b) TRUE — ablated inference: zero-ablate writes during the forward pass, full run per
   intervention per checkpoint. Interventions per model: the model-specific carrier
   (blk3 at pythia-1b; blk4+blk5 at 6.9b), each of the four contiguous L/4 blocks, and
   the middle two quarters together (half the model) as the extreme = 5 block
   interventions + 1 carrier. Collect ALL metrics on the ablated runs (ledger terms under
   ablation are the interesting part), max_checkpoints ≈ 30 (log schedule now correct).
   Expensive: job design + resources need user sign-off before submission.
3. The padded-pythia puzzle (added Jul 19, user): pythia-1b's padded last-token pre-norm
   stream RankMe is SINGLE-DIGIT at every sample count (showcase_appendix §E: 7.1–8.6 —
   a representation property, not an estimator artifact), and how sink-direction variance
   reaches padded last tokens is only partially accounted for (mid-stack residue ~3.6%,
   quantitative match unchecked — final_report §6). Sub-items:
   (a) a PADDED dataset-swap leg — the existing swap control is packed-only, so a
   dataset-side explanation of the padded collapse is currently unruled;
   (b) the quantitative residue accounting (does ~3.6% mid-stack residue explain the
   padded last-token measurement?);
   (c) settle whether the padded head-crush is the same object as the packed head event
   or a last-token-geometry phenomenon of its own.
4. Separation-clean H2.1 follow-ups (after the redo).
5. Rerun the architecture-knob grid under the constructed-timing setup (the current grid
   runs on the no-pattern 32-class task — its trajectory-shape readings are void, only the
   suppression and carrier attributions stand; toy.md §6 scope note). Add a QK-norm cell
   in the same rerun (the sink literature says QK-norm is the STRONGER spike-suppressing
   lever; the old grid only tested write-norm).
6. Toy: add an attention-like ingredient (token-addressed mixing) to test whether anything
   produces OLMo's interference mode — the missing half of the architecture-knob result.
7. Drift re-derivation under the gap-corrected rate (`cka_drift_rate`; settle per-dex vs
   per-gtok normalization): the raw-CKA "two reorganization events" claim is
   checkpoint-spacing-confounded and a first corrected look contradicts it (pythia peaks
   mid-training, OLMo-1B rises late) — flagged in final_report corrections record.
8. Deferred pending infgrams (other session): TriviaQA distributional-memorization —
   memorization-peak vs spectral-event alignment, checkpoint-wise head-ablation cost, a
   behaviorally-grounded memorized population for RQ2.
9. pythia-6.9b RQ2 collection (SCRAPPED for now, Jul 13 — user call; OLMo-7B data suffices):
   all 11 configs OOM'd at the standard 180G host-RAM share (173G used vs OLMo-7B's 107G —
   smells like a pythia-specific pipeline problem, e.g. checkpoint loading holding duplicate
   copies). Before any rerun: diagnose the footprint on the standard share (RSS logging),
   fix it, THEN collect. Job design + resource choices need user sign-off.
10. RQ2 REDO (PARKED by user, Jul 13): the documented excess-mass aggregates mixed acts and
   grads quantities; acts-only inverts the quotes/memorized ordering (memorized 926 > null
   338 > quotes 274, padded, pythia-1b) and weakens math (1.8x vs claimed 2.4x). Re-derive
   every H2.2 verdict with acts and grads SEPARATE; re-check split-half for the same
   conflation; VERIFY the ref pairing (stored gen_vs_ref results don't record which G file
   was used — re-run the --ref passes geometry-matched and record the ref path in results);
   re-render figures with each population against its own geometry- and N-matched null
   (baseline at 1). DESIGN CONSTRAINT (user, Jul 13): comparisons are valid WITHIN one
   geometry/token-selection only — packed all-token and padded last-token covariances are
   different measured objects, so cross-population rankings that straddle geometries (e.g.
   "quotes padded > math packed") are invalid regardless of the acts/grads fix; any
   cross-population ranking requires a shared-geometry collection (e.g. add a math
   padded/last arm; note packing short populations adds separator tokens = sink slots).
   final_report SS4 carries a do-not-cite banner.


## Done (moved to the report/topic docs; kept here briefly for continuity)

- Rogue follow-ups (Jul 13): EOT-twin (newline slot NOT absorbed) + mid-stack depth profile
  (deposit → ride → late scrub; ordinary tokens 27%→4%) → dig_findings "Sink-slot
  follow-ups", showcase §4 depth figure. Remaining thread: quantitative match of the ~4%
  final-stream residue to the padded last-token measurement (final_report §6).
- Own-generations entropy sweep → default view in vocab_entropy.ipynb.
- Fair-shot plain-stack retune (lr 0.02 solves, still no phases) → final_report §3,
  multi_plain spec updated to lr 0.02 and figure regenerated.
- RQ2 large-model collections: OLMo-7B configs collected (metrics passes parked with the
  RQ2 redo, item 1); pythia-6.9b scrapped (item 0).
- Dataset-swap control → final_report §2 + dig_findings "Controls: the dataset swap".
- Toy write-norm grid → final_report §0.3/§3 + toy.md §6.
- Rogue-write interventions (the old "next #1") → H1.5 triptych, dig_findings.
