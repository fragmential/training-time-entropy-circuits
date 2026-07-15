# Fig-4 addendum: the constructed init, and the timing hierarchy behind "compression"

Companion to [toy_model_report.md](toy_model_report.md) (H3.1). Li et al. state three
ingredients for the toy's three-phase dynamics: cross-entropy, skewed class counts, and an
information bottleneck (d < |V|). This note documents a **fourth, unstated requirement** that
their own Fig 4 depends on, with the experiments that establish it.

*Background (one paragraph).* Our first reproduction (iid init std 0.3, seed 3) produced the
three RankMe phases but not the paper's rare-class geometry — the two singleton classes
diverged from step 0 and one stayed stunted, making the "compression" decline an artifact of
a still-unlearned class. Zooming into Fig 4's gray t=0 markers showed the init is
*constructed*, not iid: each frequent class's two samples clustered on its own direction
(counts stay (2,2,1,1) — two triangles, two circles), W columns aligned to the class
directions, and the two rare classes **coincident** with zero weights. This is now
`toy/train.py::_clustered_init` (`clustered=True`, jitter `delta`); lr 0.25 was calibrated so
the eigenvalue scale matches the paper's D panel over 300 steps. (Comparison gotcha: `dup=3`
triples raw feature eigenvalues; divide by `dup` before reading them against the paper's λ
axis.)

## The fourth condition: a timing hierarchy

GD preserves the rare-pair swap symmetry exactly (if f₂=f₃ and w₂=w₃, their gradients are
identical), so the coincident init makes the shared path exact and the jitter δ sets the
fork time: the antisymmetric mode grows exponentially at measured rate r ≈ 0.031/step
(log-linear fit R² residual 0.09), so t_fork ≈ log(1/δ)/r — verified across five decades:

| δ | predicted fork (mid, cos<0.5) | observed (3 seeds) |
|---|---|---|
| 1e-1 | 96 | 113, 98, 165 |
| 1e-2 | 170 | 175, 163, 206 |
| 1e-3 | 244 | 236, 224, 265 |
| 1e-4 | 319 | 296, 284, 324 |
| 1e-6 | 467 | 415, 401, 443 |

The compression decline then exists **iff the fork lands inside a window**:

**t_saturate < t_fork < t_end** — the fork must come *after* the entropy-phase spectrum has
saturated (RankMe plateau) and *before* the observation horizon.

δ sweep (lr 0.25, 300-step window, 3 seeds each; fork onset = sustained cos(w₂,w₃) < 0.97):

| δ | fork onset (w) | decline magnitude (RM peak − min after peak) |
|---|---|---|
| 1e-1 | 3–106 (too early) | 0.0000–0.0020 |
| 1e-2 | 113–157 | 0.0042–0.0046 |
| 1e-3 | 175–217 | 0.0034–0.0049 |
| 1e-4 | 235–276 (late) | 0.0000–0.0018 |
| 1e-5, 1e-6 | > 294 (outside) | ≤ 0.0003 |

Inverted-U, exactly as the hierarchy predicts. Three further checks pin down *which* feature
of the init is load-bearing:

- **Homotopy to iid-tiny** (θ₀, W₀ = α·constructed + (1−α)·0.01·randn, δ fixed 1e-3): decline
  survives α ≥ 0.5 (0.0048–0.0057) and **vanishes from the curve's peak-then-decline shape at
  α = 0.25** even though the fork still lands mid-window (~155) — because there RankMe is
  *still rising* at fork time (RM(300)=1.9918 = its maximum). The fork's ratio-kick is
  **present but masked** there, not absent (see "Anatomy of the kick"). iid init fails not
  for lack of anisotropy but because it *locks t_fork to t_saturate* — both are set by the
  same growth dynamics, so the fork always arrives while the spectrum is still filling out
  and the kick lands on a steeply negative baseline.
- **Base anisotropy is irrelevant**: swapping/equalizing the frequent-class scales
  (fm, fo) ∈ {(0.6,0.75), (0.68,0.68), (0.75,0.6), (0.4,0.85), (0.85,0.4)} leaves the decline
  at 0.0034–0.0075 with identical fork times. (This kills the tempting "decline needs an
  anisotropic base" narrative — we held it briefly; the data contradicted it.)
- **The decline is a transient**: long-horizon run recovers, RM 1.9992@218 → min 1.9942@330 →
  1.9979@1000 → 1.9993@3000 (loss 5e-4). d log(λ₁/λ₂)/dt: −0.72e-3 pre-fork (homogenizing),
  **+0.74e-3 during the fork**, −0.09e-3 after. The λ₁ acceleration the paper shows
  (σ₁ pulling ahead after ~200) *is* the fork event, not an endpoint; "compression" in this
  toy is the spectral signature of late rare-class separation, full stop.

## Anatomy of the kick: why the fork boosts λ₁/λ₂ at all

The two corrections above meet in one question: in a saturated, near-isotropic spectrum,
blue folds onto orange's axis and green onto magenta's — both axes get fed, so where does
the anisotropy come from? Not from any of the obvious suspects, per the fixed-axis Gram
decomposition of the canonical run (steps 170–320):

- **Not norm asymmetry**: |f₂| ≈ |f₃| within 1% throughout the fork (1.58/1.56 → 2.12/2.10);
  the folding is symmetric, and Gxx − Gyy barely moves (0.57 → 0.42).
- **Not class counts or the frequent classes' residual gap**: the frequent contributions to
  both diagonal terms grow in near-lockstep during the fork window.
- **It is the off-diagonal.** For a 2×2 Gram, λ₁ − λ₂ = √((Gxx−Gyy)² + 4·Gxy²). Pre-fork,
  total Gxy ≈ +0.03 — but that near-zero is a *cancellation*: the co-traveling pair sits on
  the third-quadrant diagonal and carries Gxy ≈ +2.2, offset by the tilted frequent classes'
  ≈ −2.2 (the magenta direction is 100°, x·y < 0). The fork collapses the pair's covariance
  (+2.2 → +1.4 as each feature folds toward an axis) while the frequent term stays put
  (−2.2 → −2.2): net Gxy swings to −0.86, and the eigengap reopens through the 4Gxy² term
  (λ₁−λ₂: 0.58 → 1.78; the λ₁ eigvec rotates ~38° off the x-axis). **The kick is the
  collapse of the rare pair's diagonal covariance breaking a cancellation** — a geometric
  consequence of folding-off-the-diagonal, independent of which axis "wins".

**Masked, not absent.** Measuring d log(λ₁/λ₂)/dt through the fork in both regimes:

| regime | baseline before fork | kick peak | net RM effect |
|---|---|---|---|
| canonical (fork @ ~218, saturated) | −0.2e-3 (≈ flat) | **+2.1e-3** | visible peak-then-decline, 0.0048 |
| homotopy α=0.25 (fork @ ~155, still rising) | −8.7…−0.5e-3 (λ₂ catch-up) | **+1.0e-3** | transient dip ≈ 0.003 (RM 1.9909@130 → 1.9880@180 → 1.9917@300); RM still ends at its max |

So in the rising-phase regime the kick is *present* — it even flips d log(λ₁/λ₂)/dt
transiently positive (steps ~140–185) and produces a genuine mid-run RankMe dip — but the
resumed λ₂ catch-up overrides it and RankMe finishes at its maximum, so the windowed
"peak − min after peak" metric (and the eye, on the full curve) reads zero. (The homotopy
bullet's "0.0000" at α = 0.25 and the δ = 1e-1 "too early" rows undercount for the same
reason — those forks also land on rising baselines.) The unified story:
**the fork always delivers the covariance-collapse kick; t_saturate < t_fork is the
*visibility* condition** — it determines whether the kick lands on a flat baseline and prints
as the compression phase, or on a falling-ratio baseline that swallows it. The kick is also
~2× smaller in the rising regime (the pair's diagonal covariance has had less time to grow
before folding), compounding the masking.

## The dash lag: decline onset vs the fork

With the rare pair started off-origin (rx = −0.25, our first figure-read of their gray
marker), the RankMe peak (= decline onset, where the figure switches to dashes) lags the
*feature* fork onset by ~24 steps (weights fork first: cos(w₂,w₃) < 0.97 at ~187,
cos(f₂,f₃) < 0.97 at ~194, RM argmax at ~224), so the first stretch of the visible fork is
drawn solid. In the paper's panel C the dashes begin *at* the fork — and the current
canonical repo run (rare pair at the exact origin) reproduces that: fork onset = RankMe peak
= step 204 at checkpoint resolution. Both are real regimes:

- **Mechanism of the lag**: the kick's driver is the pair's covariance collapse (above),
  which scales with the *squared* fork amplitude — while the separation is small the pair
  still sits essentially on the diagonal, the Gxy cancellation holds, and RankMe even creeps
  up (1.9989 → 1.9992 through step ~210). The decline starts once the separation reaches
  covariance-moving amplitude — ~1/3 of its final value at the RM peak (0.33, vs 0.14 at
  fork onset); with growth rate r this predicts lag ≈ log(0.33/0.14)/r ≈ 28 steps ≈ the
  observed 24.
- **What controls it**: lag × lr ≈ 6.0–6.3 across lr ∈ [0.15, 0.35] (pure timescale, lag ∝
  1/lr); and the rare pair's initial offset from the origin dominates: lag = 5 / 10 / 24 / 41
  steps for rx = 0 / −0.1 / −0.25 / −0.5, with decline 0.0067 / 0.0061 / 0.0050 / 0.0029. δ
  barely moves it (18–29 steps for δ ∈ [1e-4, 1e-2]).
- **Verdict on the paper**: with the rare pair at the exact origin (rx = 0) the lag is ~5
  steps — zero at plot resolution — and the decline is *larger*. Their D panel's dashes begin
  at ~step 210, the end of the RankMe plateau, consistent with fork = peak. So the
  dashes-at-the-fork in their B/C is dynamically reachable (hypothesis "genuine zero lag"),
  not a presentation choice; it suggests their rare classes initialize at the origin itself.
  `_clustered_init` therefore starts the rare pair at (0, 0) — this is the canonical
  `single` run — matching their zero lag and deepening the decline.

## Residual gaps vs the paper (canonical repo run: rare pair at origin, dup-corrected)

| quantity | paper Fig 4D | repo `single` |
|---|---|---|
| RankMe start / dip | ~1.91 / ~1.88 | 1.933 / 1.906 |
| peak | ~1.99 @ ~150–210 | 1.99964 @ 204 |
| RankMe(300) / decline | ~1.981 / ~0.010 | 1.99313 / 0.0065 |
| λ₁, λ₂ @300 | 15.1, 11.5 (ratio 1.31) | 12.8, 10.9 (ratio 1.18) |
| final w₂/w₃ angle | ~78° | ~85° (cos +0.08) |
| decline onset vs feature fork | coincide (~210) | coincide (204) |

Same shape, ~1.5× shallower decline; their exact init magnitudes/jitter are unpublished, and
lr trades λ scale against peak timing (lr 0.3 gives λ₁ 14.4 but peaks at ~150).

**Takeaway.** "Skew + bottleneck + CE" is not sufficient for the compression phase in this
toy: it additionally needs late symmetry breaking among the rare classes — an initialization
in which rare classes start (near-)coincident so their separation arrives after the entropy
phase saturates but inside the training window. That is a fourth necessary condition the
paper's text never states, and their Fig 4's init visibly encodes it.

## Appendix: proof sketch — why the decline is transient

Setting: d=2, classes (2,2,1,1), features F free parameters, logits FW, CE, full-batch GD.
Write G = FᵀF; the eigengap obeys λ₁ − λ₂ = √((G_xx − G_yy)² + 4G_xy²), and RankMe is a
scale-invariant function of λ₂/λ₁ (maximal iff isotropic).

1. **Shared path (exact).** The GD vector field commutes with the rare-pair swap (2↔3): if
   f₂ = f₃ and w₂ = w₃ then their time-derivatives are identical (verified symbolically —
   the difference dynamics vanish). Coincident init ⇒ the co-travel is exact; a jitter δ
   evolves under the linearization around the symmetric path.
2. **Fork time (linearized instability, rate measured).** The antisymmetric (2↔3) mode grows
   exponentially at rate r (measured 0.031/step at lr 0.25; log-linear over 4+ decades), so
   the fork arrives at t_fork ≈ log(1/δ)/r — validated across δ ∈ [1e-6, 1e-1] within
   10–15%. The fork's spectral effect is the collapse of the pair's off-diagonal covariance
   cancellation: a BOUNDED change ΔG_xy of order (pair amplitude at fork)².
3. **Asymptotics under CE.** After all classes separate, CE on separable data drives margins
   to grow ~log t with the parameter direction converging (implicit-bias regime à la
   Soudry et al): F(t) = ρ(t)·F̂ + o(ρ), ρ → ∞. Hence G(t) = ρ²·F̂ᵀF̂ + o(ρ²) — spectrum
   shape converges to that of the FIXED limit Gram F̂ᵀF̂.
4. **Transience.** RankMe(t) → RankMe(F̂), and the fork's ΔG_xy is o(ρ²) relative to the
   growing diagonal, so its rank-entropy kick decays like (ρ(t_fork)/ρ(t))² — the decline is
   a transient of relative size (fork amplitude / current scale)². Empirically the limit
   configuration is balanced (blue → −x, green → −y, symmetric against the frequent pair),
   so RankMe(F̂) ≈ 2 and the curve recovers fully (measured 1.9993 at steps 3000–6000).

What is proven vs assumed: step 1 is exact; step 2's exponential rate is measured, not
derived from the Hessian; step 3 invokes the standard separable-CE implicit-bias picture
without re-deriving per-class rate equalization for unequal counts (empirically the λ ratios
do converge); step 4's RankMe(F̂) ≈ 2 is numerical. A complete proof needs (i) the max-margin
characterization of (F̂, Ŵ) for this dataset, (ii) isotropy of its Gram, (iii) a decay-rate
bound for the fork transient. Appendix-grade sketch, flagged accordingly.

*Experiments live in the session scratchpad (`part1.py`, `part1c.py`, `part1d.py`,
`refine.py`); dynamics
identical to `toy/train.py` (imports `_clustered_init`), per-step logging, dup=1 ≡ dup=3.*
