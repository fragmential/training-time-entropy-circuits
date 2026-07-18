# The multi-layer toy: what turns the phase curve on

*Companion to [toy_model_report.md](toy_model_report.md) (the single-layer reproduction and
the architecture grid) and [toy_fig4_addendum.md](toy_fig4_addendum.md) (the single-layer
fork/kick mechanism and the transience proof sketch). Written Jul 13 2026, after the
depth investigation this doc summarizes. All runs referenced by tag live in
`data/results/toy_<tag>/` (spectral metrics) and `data/results/toy/trajectories_<tag>.pt`
(per-checkpoint weights/features); every one is reproducible from `toy/train.py` — named
variants directly, others via `replace()` on a named spec as listed in §7.*

## 1. The question

Li et al's phase curve (dip → entropy-seeking rise → peak → compression decline) is
measured at the END of multi-layer residual LLMs, and our LLM sweeps reproduce it there.
Yet the repo's multi-layer toys (`multi_residual*`, `multi_plain`) never showed it: at
every depth their final stream does plateau → collapse → partial recovery, with no
post-saturation decline. Since the theory is supposed to explain multi-layer residual
models, that discrepancy had to be resolved: which ingredient do the LLMs (and Li et al's
single-layer toy) have that the multi-layer toy lacked?

Three early readings failed verification and are retracted (history kept because each
failure shaped the answer): "the pattern fades with depth" (an artifact of UNCENTERED
per-depth RankMe — the nonlinear writes' bias means hold ~50% of the uncentered trace;
showcase_appendix §C), "deep streams start at their maximum, an init artifact of depth"
(wrong — centered deep streams start near-full-rank), and "no rise because init rank
exceeds solution rank" (its positive test failed: a rank-1 init produced no clean rise
either).

## 2. The mechanism, stated generally

Let X(t) ∈ R^{N×d} be the measured representation (Li et al: the uncentered features; our
LLM runs: the centered output stream), λ_i(t) its covariance spectrum, p_i = λ_i/Σλ_j, and
S = −Σ p_i log p_i the spectral entropy, so RankMe = e^S. Writing g_i = d log λ_i / dt for
each direction's growth rate, differentiating S gives (using ṗ_i = p_i(g_i − ḡ) with
ḡ = Σ p_j g_j):

**dS/dt = −Cov_p(g, log p)**

— exact, model-agnostic. RankMe falls if and only if the already-heavy directions are the
faster-growing ones (positive p-weighted covariance between growth rate and log spectral
share). Every phase is a regime of this one quantity:

- **Warmup collapse:** the dominant structure (frequent classes / frequent patterns) is
  learned first, so heavy directions grow fastest → Cov > 0 → RankMe falls.
- **Entropy-seeking rise:** progressively rarer structure is learned, so light directions
  grow fastest → Cov < 0 → RankMe rises.
- **Saturation:** all structure learned; cross-entropy margin growth is asymptotically
  uniform (~log t per class) → g_i ≈ ḡ → Cov ≈ 0 → RankMe flat.
- **Compression event:** any mechanism that makes top directions outgrow the bulk again →
  Cov > 0 → decline. In the toys this is the rare-pair **fork kick** — one-shot, fixed
  size; the addendum derives it: the co-traveling pair's off-diagonal covariance cancels
  the frequent tilt, the fork collapses the cancellation and reopens the eigengap. In the
  LLMs (RQ1) it is a **sustained** concentration mechanism: Pythia's rogue sink-write
  growing enormous relative to its local input, or OLMo-2's mutually aligned late writes.

Three conditions decide whether a decline **prints** in the measured curve:

- **(T) Timing/visibility.** The printed slope is the event's positive Cov contribution
  *net of* the baseline's. If structure is still being learned (baseline Cov < 0), a
  fixed-size event is swallowed — "masked, not absent" (single-layer B3 homotopy). Printing
  requires the event to land after saturation, OR to be large/sustained relative to the
  baseline. In LLMs the visibility is per-band: the event lives in the spectral head while
  the bulk keeps entropy-seeking — which is precisely the "compression is a head event on
  top of continuing expansion" finding (final_report §2.1); full-spectrum RankMe still
  declines because the head dominates p.
- **(K) Kick size.** For one-shot events the decline magnitude is set by the event's
  eigengap change (addendum: ∝ squared fork amplitude); too small and it drowns in drift.
- **(I) Transmission.** The event happens in the learnable feature geometry; the
  measurement happens at the network output. The output inherits the feature spectrum iff
  the map between them is **near-identity at initialization**. Residual stack:
  X_out = X_feat + E (E = summed writes), and Weyl's inequality bounds every singular
  value's displacement by ‖E‖_op — small-at-init writes ⇒ spectrum (and its dynamics)
  carries through, at any depth. Plain stack: X_out = Φ(X_feat); the spectrum is distorted
  by Φ's conditioning, and training Φ adds spectral dynamics of its own — transmission
  holds only if Φ starts near identity. **The residual architecture provides near-identity
  by default; a plain stack must have it arranged by hand.** This is the precise content of
  the earlier slogan "the residual stream is constitutive for the phases".

These conditions cover all three systems: the single layer satisfies (I) trivially and
uses the constructed init for (T); the deep toys below satisfy (I) via small blocks /
identity init and (T) via the constructed init's delayed fork; the LLMs satisfy (I) by
being residual and (T)+(K) via sustained head-concentrated mechanisms on a
slowly-expanding baseline. Honest scope note: the LLM mapping of the "event" is our RQ1
empirical finding, not derived; and LLM saturation is only partial (per-band), which the
band-α data supports.

## 3. Ablation table (all runs, uncentered final-stream RankMe unless noted)

| # | run (tags) | setup | result |
|---|---|---|---|
| 1 | `multi_residual_nonlinear` + `_d1/_d2/_d3` | MULTI task (32 classes, counts 128…2×26, d=16), iid init, depths 1/2/3/6 | no phases at ANY depth: plateau → collapse at learning onset → recovery to ~2.1–2.5 (centered) |
| 2 | `multi_residual_nonlinear_flat_s0/1` | as 1, rank-1 init (`flat_init`) | no clean rise (init 1.75, wanders to ~2.1) — killed the "init-rank vs solution-rank" account |
| 3 | per-class timing (stored trajectories of 1) | — | classes learned in strict frequency order; collapse = frequent-class solution forming; recovery = entropy-seeking (rank rises exactly while mid/rare classes learn, ~0.9k–7.4k); the 26 rare forks smear across the rise → masked per (T) |
| 4 | `single_d6_bs_s0/1/2` (= variant `single_deep`) | Fig-4 constructed init, depth 6 residual, linear, `block_scale=0.05` | **FULL PRINT 3/3**: dip → rise → peak@34–38 = fork (cos(w_rare) +1.00 → fork; rare CE < 0.1 by 42–48) → decline 0.093–0.098 → transient recovery |
| 5 | `single_d6_s0/1/2` | as 4, default block init | **no print 3/3** (negative control for (I)/(T)): fork lands on a perturbed/rising baseline; one seed collapses to rank 1 |
| 6 | `single_d6_bs0p3/0p1/0p01_s0` | block_scale sweep | prints at 0.01/0.05/0.1 (decline ≈ 0.10, fork@34); DEAD at 0.3 — (I) is a threshold condition |
| 7 | `single_d2_bs_s0`, `single_d12_bs_s0` | depth 2 / 12 | prints at both; kick grows with depth (0.046 / ~0.10 / 0.128) and fork accelerates (76 / 34 / 24) — observed, not explained (§6) |
| 8 | `single_d6_id_plain_s0/1/2` | **PLAIN** (no residual), blocks init I + 0.05·randn (`block_identity`) | **FULL PRINT 3/3** (decline 0.092–0.108, fork = peak @34–42): the phases do not require the residual, only near-identity transmission |
| 9 | `single_d6_bs_plain_s0/1/2` | plain, `block_scale=0.05` | never learns (loss = ln 4 exactly): scaling a plain stack toward zero starves signal and gradient — why `block_scale` is the wrong plain analog and near-identity is the right one |
| 10 | `single_d6_plain` | plain, default init | partial learning (loss 0.44), stream pinned at rank 1.0 — ill-conditioned composite, (I) fails |
| 11 | `single_d6_bs_nl_s0/1/2` | as 4, nonlinear blocks | ⚠️ decline (≈0.51) robustly PRECEDES the fork (peak@17, fork@108–137), 3/3 — a second, unexplained mechanism (§6) |
| 12 | `skewpair_deep_s0/1/2` (+`deep2` ortho-dirs twin) | generalized constructed init (§4), δ=1e-3 | no print: fork@132–174 lands mid-rise → masked, exactly (T) |
| 13 | `skewpair_deep_d8_s0/1/2` | as 12, δ=1e-8 | **FULL PRINT 3/3** (§4) — same task, only the fork delay moved: the within-task demonstration of (T) |

## 4. The conditions carried to a skewed multi-class task (skewpair_deep)

`skewpair_deep` (named variant): a MULTI-style task — 6 classes with skewed counts
32:16:8:4:2:2 through a genuine bottleneck (d=4 < 6) — with the constructed init
generalized (`_clustered_init_general`): each non-rare class clustered on an orthonormal
direction with matched W column, the LAST TWO classes a coincident rare pair at the origin
with zero W columns (exact swap symmetry, broken only by the global jitter δ), depth-6
residual stack, `block_scale=0.05`.

At δ=1e-8 (fork delayed past saturation), all 3 seeds print the full sequence, with exact
event alignment at checkpoint resolution (seed 0 shown; others identical in shape):

| steps | 101 | 152 | 228 | 299 | 342 | 392 | 514 | 3000 |
|---|---|---|---|---|---|---|---|---|
| RankMe | 2.31 | 2.77 | 2.92 | **2.96 (plateau)** | **2.58 (drop)** | 2.47 | 2.43 | 2.44 |
| cos(w₅,w₆) | +1.00 | +1.00 | +1.00 | +0.98 | **+0.33 (fork)** | +0.17 | +0.09 | — |
| rare-class CE | 1.34 | 0.73 | 0.70 | 0.67 | **0.05** | 0.01 | 0.00 | — |

The same task at δ=1e-3 (fork@174, mid-rise) prints nothing — the multi-layer, multi-class
reproduction of the single-layer B1 timing window. Note the construction is NOT
near-equilibrium for the frequent classes (all seeds show a warmup collapse ~3 → 1.4
before the rise) — the print only needs saturation before the fork, not a quiet start.

## 5. What this resolves

- **Depth is innocent.** The phase curve passes through 2, 6, and 12 blocks, residual or
  plain, once (I) holds. The MULTI toys miss the phases for task-structure reasons: their
  26 rare classes fork all across the entropy-seeking rise ((T) violated by construction
  of the task, not by depth).
- **"Residual is constitutive" — corrected form.** The residual path is one way (the
  automatic way, and the LLMs' way) to satisfy near-identity transmission; an identity-
  initialized plain stack satisfies it too and prints the phases. What plain stacks cannot
  do is satisfy it generically (default init: ill-conditioned composite, row 10; shrunken
  init: no signal path at all, row 9).
- **The earlier "fair-shot retune" conclusion stands but means less than claimed:** the
  plain MULTI stack solving the task without phases is over-determined — it fails (I)
  (default init) AND its task fails (T); it is not evidence that "replacement
  architectures cannot have phases" (row 8 shows they can).
- **The LLM connection is now one mechanism, three instantiations:** frequency-ordered
  learning produces collapse-then-rise everywhere (toy per-class table; LLM warmup + rise);
  the decline needs a concentration event that outruns the baseline — arranged by hand in
  the toys (delayed fork), supplied architecturally in LLMs (rogue write / aligned writes,
  RQ1), with per-band visibility explaining why LLM compression is a head event.

## 6. Observed but NOT explained (do not cite as understood)

- **Kick scaling with depth:** decline 0.046 → ~0.10 → 0.128 and fork time 76 → 34 → 24
  across depths 2/6/12. Plausibly the blocks amplify the effective growth rate r (fork ~
  log(1/δ)/r) and the kick; no derivation.
- **The nonlinear pre-fork decline (row 11):** 3/3 nonlinear deep runs decline ~0.5
  starting well before the fork. By the §2 identity something makes heavy directions
  outgrow the bulk there — candidate: the tanh blocks' own growth/saturation acting as a
  concentration mechanism independent of the fork (which would make it a second instance
  of (C3), interestingly LLM-flavored) — SPECULATION, unverified.
- The generalized construction's warmup collapse (§4) differs from the single-layer case
  (which has none); harmless to the print but unmodeled.

## 7. Reproduction

Named variants (toy/train.py): `single_deep` (row 4), `skewpair_deep` (rows 12–13; δ via
`replace(..., delta=1e-8)`). Other rows via `replace()` on `VARIANTS['single']`:
depth/`block_scale`/`block_identity`/`residual`/`nonlinear` as in the table; MULTI rows on
`VARIANTS['multi_residual_nonlinear']` with `depth`/`flat_init`. New Spec fields introduced
for this study, all documented in train.py: `flat_init`, `block_scale`, `block_identity`;
plus `_clustered_init_general` (constructed init for any d/class count, rare pair last).
