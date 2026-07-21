# Ablations & validity checks

Every control run and assumption check made on top of the main results: what assumption it
guards, what was done, the outcome, and where the evidence lives. Companion to
[final_report.md](final_report.md).

## Measurement geometry

| check | guards against | done | outcome | where |
|---|---|---|---|---|
| Padded last-token twin | "findings are packed-all-token artifacts" | full 4-model sweep, 16,384 rows, sqrt/30 ckpts | every family-level claim survives; only the rogue write's *strength* is geometry-sensitive | dig_findings "Padded-vs-packed"; experiments_padded.ipynb |
| Before vs after final norm | "the head event is a pre-norm artifact" | head-removed RankMe at both leaves, padded | same structure post-norm (Li et al's exact measurement point); the norm attenuates the head's dominance but doesn't change it | dig_findings; showcase §2 |
| Fineweb vs training mix | "curve shape depends on eval corpus" | Li-setting (padded fineweb) vs trainset, all models | phases reproduce in both; trainset peaks higher (expected: max expressive space) | experiments.ipynb part 1; showcase §1 |
| Identity-head vs hook capture | "post-norm capture method biases the spectrum" | `reproduce_rankme_alpha` vs `_hook` config (old verification) | agree | experiments.ipynb data_sources_4 |
| 13.9% newline last-tokens check | "padded rogue persistence contradicts the newline attribution" | tokenized the padded text cache, counted final-token identities | slot-inclusion account REJECTED: only 5/2,331 newline-bearing last tokens are their document's FIRST newline (bulk-ordinary, not slots); current partial account is mid-stack residue (~27% of ordinary-row centered norm along the direction at blk4, ~3.6% at the final stream); quantitative match to the padded measurement still open | dig_findings "Sink-slot follow-ups" |

## Data / sample size

| check | guards against | done | outcome | where |
|---|---|---|---|---|
| Dataset swap (cross-tokenized) | "the family split is the datasets, not the models" | both 1Bs on the OTHER family's mix, decoded+retokenized from the SAME texts, exact main config minus drift | **everything survives — MODEL-dependent** (rogue write replicates on olmo-mix at 24.4×; OLMo stays rogue-free on Pile) | dig_findings.md (dataset-swap controls section) |
| 262k vs 1.2M token budget | "samples-mode budgets too small for stable RankMe" | ablation on the 1Bs | RankMe within ~1%, corr ≥ 0.993 | early-session check (block_representations_all_262k) |
| Shuffled text sampling | "padded populations = head-of-dataset bias" | seeded shard-shuffle; disjoint opening docs verified | bias removed; cache keys carry the seed | data_utils `text_shuffle_seed` |
| N-matched nulls (RQ2) | "geneig excess inflates as task-side N shrinks" | split-half nulls at matched N per geometry; small-N packed nulls (73k/1M) for the memorized populations | padded conclusions N-matched by design; packed mem numbers held until matched nulls land ⏳ | rq2.md §2 step 5 |
| MP-broadening caveat | "deep-tail levels comparable across different-N runs" | trends-at-fixed-N only; deep-band windows kept ≲ low hundreds padded | stated wherever bands are read | dig_findings; final_report §5 |

## Statistics / nulls

| check | guards against | done | outcome | where |
|---|---|---|---|---|
| Split-half G null (per geometry) | "excess over 1 = structure" (it isn't — sampling noise) | G′-vs-G with identical estimator/N | the null IS the bar (61–88 packed, ~368 padded); all H2.2 claims read against it | rq2.md §2 |
| Independent whiteners | shared-reference error correlating both sides of an overlap | 4 G samples per geometry; null collapsed 0.43–0.60 → ≤0.10 | required for any split-half-style comparison | rq2.md §6 |
| Raw-space subspace comparison | whitened coordinates are whitener-specific | un-whiten + QR before overlaps | cross-whitener comparisons meaningless without it (everything → chance) | rq2.md §6 |
| Synthetic selection-bias reference | "our band curves would look the same under Li et al's theory" | dσ_i ∝ σ_i dynamic evolved from the measured peak spectrum | their mechanism tilts EVERY band; measured deep band does the opposite | showcase §2 synthetic cell |
| tail_centroid whitened-profile choice | raw-space profile skews tailward tautologically (1/λ_G weighting) | whitened-coordinate profile used; power caveat noted | mid-spectrum verdict robust; low power for weak tail preferences | rq2.md §2 step 6 |
| All-token type ranking (rogue) | post-hoc token grouping | mean projection per token type, no prior grouping | newline claim survives; small-n tail flagged | showcase §4 `rogue_ranking` |
| Statistical preimage (rogue) | "the write just echoes the newline embedding" | regress score on block INPUT, read vs embedding matrix | raw `\n` embedding unaligned (cos −0.013) → MLP-constructed | dig_findings addendum |

## Implementation equivalences

| check | guards against | done | outcome | where |
|---|---|---|---|---|
| alpha_window vs stock alpha | notebook stats diverging from pipeline stats | 1,668 real spectra compared | max dev 6.4e-5 (fp32-vs-fp64 only) | experiments_lib `_alpha` |
| geneig identity/scale tests | wrong generalized solver | gen(A,A)=1s, gen(cA,A)=c unit tests | pass | tests/test_compute_metrics.py |
| Toy dup-invariance | sample duplication distorting toy spectra | dup=k verified bit-identical to dup=1 | exact | toy.md §7 (deviations) |
| e2e on real models | hook bugs invisible in unit tests | full forward/backward e2e on Pythia + OLMo-1B | caught the kwarg grad_input bug class | tests/ -m e2e |

## Replication axes (robustness by repetition)

- **Scale**: every RQ1 family claim at two scales per family (1b/6.9b, 1B/7B) — the "2+2".
- **Seeds**: toy signatures reported per-cell over ≥3 seeds (never single-seed); single-toy
  compression robust 4/5 jitter seeds under the paper's constructed init (the earlier ~25%
  seed-dependence was an iid-init artifact — toy.md §3).
- **Third architecture (DONE)**: nanochat-d12 shows the full Pythia signature —
  quality-carried compression, carried by a split between the early and final blocks — so QK-norm
  without write-norm does not prevent the mechanism (H1.4 out-of-family confirmation).
- **Negative controls (toy)**: uniform labels / no bottleneck / MSE ×2 all remove compression,
  as the paper claims; H3.3 representativeness guard (max w_k) checked for every variant.
