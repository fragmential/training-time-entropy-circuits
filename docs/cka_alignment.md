# CKA alignment statistic (stream ↔ unembedding)

The alignment measure used in unembedding.ipynb §2 (toy) and proposed for the LLM runs.

## Definition

    s(Σ, G) = ⟨Σ, G⟩_F / (‖Σ‖_F · ‖G‖_F)

- Σ = centered feature/stream covariance (d × d).
- G = stream-side head Gram: WᵀW for the LLM head (V × d), W̃W̃ᵀ for the toy classifier
  (d × C). Corrections (freq-centering, mean-deflation) are applied to W before forming G.
- tr(ΣG) = mean squared logit norm of the centered stream pushed through the head — the
  signal energy the head reads.
- s = 1 iff Σ ∝ G (Cauchy–Schwarz): gain profile matches variance profile, direction for
  direction, weight for weight. k-free, both spectra enter with their natural weights.

## Invariances

- Textbook linear CKA compares two representations via their sample × sample Grams; that
  erases each side's feature basis → invariant to orthogonal transforms of either side
  separately.
- This statistic is a matrix cosine in one shared space. Invariant to a simultaneous
  rotation of both (⟨RΣRᵀ, RGRᵀ⟩ = ⟨Σ, G⟩ — basis choice of the stream space drops out).
  NOT invariant to rotating one side alone; that sensitivity is the alignment content.
- Correspondence: linear CKA between the stream X and the logits XWᵀ depends on W only
  through WᵀW = G. So vocab-side rotations (U) drop out entirely; the stream side stays
  pinned. Insensitive to which-token bookkeeping, sensitive to where the head listens.

## Rotated-W control

- W → QW, Q a fixed seeded random orthogonal matrix. Preserves every spectral property
  (singular values, norms, RankMe); scrambles only the stream-side directions.
- Answers: how much would the statistic report for a head with the identical spectrum but
  random orientation? A spectrum-matched chance floor — stronger than the analytic k/d
  line, since weighted statistics sit above naive chance from spectrum shapes alone.
- The quantity to read is the gap s(standard) − s(rotated), not the absolute level.

## Blind spots (seen in the toy, d = 2)

- If either side is near-isotropic, orientation stops mattering: G ≈ cI ⇒ QGQᵀ = G, and
  the control equals the real measurement by construction. Toy recovery phase: feature
  RankMe → 2 (isotropy in d = 2) ⇒ control converges to 1. That is loss of discriminative
  power, not late alignment.
- Direction sensitivity is maximal when both spectra are dominated by few directions; a
  random rotation then scores E[cos²] ≈ 1/d (0.5 at d = 2). So the gap peaks at the
  RankMe trough — the statistic is informative exactly in the compression window.
- Reading rules: don't interpret s where RankMe is near its dimensional ceiling; at LLM
  widths the isotropy blind spot is negligible, but spike domination reduces s to a
  one-direction question — run it on the corrected head variants.
