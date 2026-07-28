# Next samples-run: metrics to add / fix

Everything here is **inline-only** (needs raw samples at metric time; samples are never
persisted), so existing results can't be patched — these land on the next collection sweep.

0. Cosine of centered samples against mean
0. Cosine of centered class samples against their GT unembed row. / + I guess all the things from "Linguistic Collapse: Neural Collapse in (Large) Language Models" (wu2024linguistic)

1. **`cos_cr` also against the incoming stream** (`cos_c_rin`): the per-token mean cosine in
   `block_residual_coupling` is currently measured only against the *final* stream $r$;
   add the same against $r_{<k}$ (the views are already fetched — `_c_r`'s `v[1]`).

2. **Centered variants of the per-token cosines**: `cos_cr` and `block_block_coupling.mean_cos`
   run on raw (uncentered) samples; subtract the stored means to emit centered + uncentered
   side by side (`(X - μ)` before the cosine — trivial at metric time). Uncentered is inflated
   wherever means co-align, which they do (Exp 1.1).

3. **Joint top-k ablated stream spectrum** (`joint_ablation`, root metric): rank writes by
   single-write $|\Delta\mathrm{RankMe}|$ at the current checkpoint, subtract the top-3 writes'
   samples from the stream **per token** (exact — all cross terms included). Note: "most
   influential averaged over training" is decided post-hoc in the notebook; inline we store
   per-checkpoint top-3 (set is stable in practice — blk3 + late blocks).
   **Interface (Exp 4.5 already consumes it):** results node `''`, metric name
   `joint_ablation`, keys: `eigenspectrum` (normalized, descending — feeds the tail_rankme /
   alpha_window virtual hooks), `trace`, `rankme`, `alpha`, `leaves` (the 3 removed paths).
   The additive estimate from single-write deltas is already plotted and demonstrably breaks
   where writes interact (pythia-1b: goes negative — the interaction term made visible).

4. **Previous checkpoint's eigenpairs in the drift spill** (1B drift runs): at spill time the
   prev checkpoint's eigvals/eigvecs sit in its resolver cache; persisting them (~+1–2 GB on
   the spill) saves the ~1–2 min/ckpt re-decomposition on the next checkpoint.

5. **Gap-aware drift** (Jul 13): stored `cka_drift` compares consecutive checkpoints without
   normalizing for the gap between them, and the samples schedule was ~5× denser in
   log-tokens near the end (the `_subsample` linear-refill bug, fixed Jul 13) — so late-run
   drift levels read artificially low. A plotting-side correction exists
   (`cka_drift_rate` virtual hook, `per='dex'|'gtok'`), but the clean fix is at collection
   time: either store the per-checkpoint token gap alongside the drift value, or compute the
   drift against a fixed-Δ reference (e.g. the checkpoint nearest half a dex back) instead of
   "previous checkpoint". Note the fixed `_subsample` also means the NEXT sweep's step set
   differs from the old one (36 shared + 14 shifted) — incremental reruns won't line up.

Decided against: drift at 7B scale (not needed — user call, Jul 3 2026).

## Historical ops note (moved from final_report — not science, kept for the record)

A 2×-RSS bug in collection (`captured()` retained per-batch chunks) caused the padded-7B
OOMs and explains historical memory peaks; fixed — memory budgets since are honest.
