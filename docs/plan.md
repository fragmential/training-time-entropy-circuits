# Plan (living document)

The only forward-looking doc: in-flight work, ranked next steps, deferred decisions.
Results NEVER live here — they go to final_report.md (headline) and the topic docs
(evidence). Collection-specific additions for the next sweep: next_run_additions.md.

## In flight (SLURM, Jul 13)

- (nothing — the Jul 13 batch has landed; see Done)

## Next (ranked)

0. pythia-6.9b RQ2 collection (SCRAPPED for now, Jul 13 — user call; OLMo-7B data suffices):
   all 11 configs OOM'd at the standard 180G host-RAM share (173G used vs OLMo-7B's 107G —
   smells like a pythia-specific pipeline problem, e.g. checkpoint loading holding duplicate
   copies). Before any rerun: diagnose the footprint on the standard share (RSS logging),
   fix it, THEN collect. Job design + resource choices need user sign-off.
1. RQ2 REDO (PARKED by user, Jul 13): the documented excess-mass aggregates mixed acts and
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
2. Separation-clean H2.1 follow-ups (after the redo).
2. QK-norm in the toy grid (the sink literature says it is the STRONGER spike-suppressing
   lever; our grid only tested write-norm).
3. Toy: add an attention-like ingredient (token-addressed mixing) to test whether anything
   produces OLMo's interference mode — the missing half of the architecture-knob result.
4. Drift re-derivation under the gap-corrected rate (`cka_drift_rate`; settle per-dex vs
   per-gtok normalization): the raw-CKA "two reorganization events" claim is
   checkpoint-spacing-confounded and a first corrected look contradicts it (pythia peaks
   mid-training, OLMo-1B rises late) — flagged ⚠️ in final_report appendix, Jul 13.
5. Deferred pending infgrams (other session): TriviaQA distributional-memorization —
   memorization-peak vs spectral-event alignment, checkpoint-wise head-ablation cost, a
   behaviorally-grounded memorized population for RQ2.

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
- Dataset-swap control → final_report §2 + swap_run_findings.md.
- Toy write-norm grid → final_report §0.3/§3 + toy_model_report.md.
- Rogue-write interventions (the old "next #1") → H1.5 triptych, dig_findings.
