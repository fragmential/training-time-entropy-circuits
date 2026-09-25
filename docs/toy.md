# The toy platform (RQ3)

One doc for the whole toy arc: single-layer reproduction of Li et al's Fig 4 (§2), the
mechanism behind its compression phase (§3), the depth conditions (§4), the decomposition
signature (§5), the architecture knobs (§6). Everything lives in the isolated `toy/`
directory; activations are dumped in DataAccessor `acts` format so the **unmodified**
`scripts/compute_metrics.py` computes all metrics — including the ledger — exactly as for
real models. Figures: showcase §6 + analysis/toy_multilayer.ipynb (every depth ablation) +
showcase_appendix §B–C. Superseded readings: final_report corrections record.

## 1. Setup

The paper's model: features are free parameters ($f_0 = \theta$ rows, since inputs $S = I$
orthonormal), logits $= f\,W$, cross-entropy, full-batch GD. One `Spec` dataclass
parameterizes every variant (class counts, $d$, depth, residual/nonlinear, loss, lr, init).
Multi-layer variants add a stream: $f_{k+1} = f_k + g_k(f_k)$ (residual; $g_k$ linear or
one hidden tanh layer) or $f_{k+1} = g_k(f_k)$ (plain — He et al's term for no skip
connections).

| variant | classes (counts) | d | depth | loss | steps |
|---|---|---|---|---|---|
| single | (2,2,1,1) — paper's skew, clustered init | 2 | — | CE | 300 |
| uniform | (2,2,2,2) | 2 | — | CE | 1000 |
| nobottleneck | (2,2,1,1) | 3 | — | CE | 1000 |
| mse_skew / mse_uniform | skew / uniform | 2 | — | MSE | 1000 |
| single_deep | (2,2,1,1), clustered, block_scale 0.05 | 2 | 6 | CE | 1000 |
| skewpair_deep | 6 classes 32:16:8:4:2:2, constructed-general | 4 | 6 | CE | 3000 |
| multi_residual (+nonlinear) | 32 classes, counts 128↘2 | 16 | 6 | CE | 30k |
| multi_plain | same | 16 | 6 (plain) | CE | 100k |

Initialisations: `_clustered_init` (the paper's constructed geometry, §3; jitter `delta`);
`_clustered_init_general` (same for any d/class count, rare pair last); iid; `flat_init`
(rank-1 start). Layer inits: default Kaiming-uniform, `block_scale` (×0.05 shrink),
`block_identity` (I + 0.05·randn, plain only).

## 2. Single-layer reproduction (H3.1)

**Hypothesis:** the paper's model (skew + bottleneck + CE) reproduces warmup →
entropy-seeking → compression in RankMe; the four controls (uniform; d≈|V|; MSE ×2) each
remove compression.

**Verdict: CONFIRMED — with the paper's constructed init, a fourth unstated condition (§3).**

`single` reproduces all four Fig-4 panels: warmup RankMe 1.933 → 1.906 (~15 steps),
entropy-seeking → 1.99964 (peak at step 204), compression → 1.99313 at step 300 (decline
0.0065), λ₁ accelerating past λ₂ (12.8 vs 10.9, dup-corrected) — and the rare-class
geometry iid init can never give: the two singleton classes travel one **exact shared
path** and fork late (fork onset = the RankMe peak, matching the paper's
dashes-at-the-fork), ending ~85° apart on the reused frequent-class axes. Robust 4/5
jitter seeds at δ=1e-3.

| control | RankMe trajectory | compression |
|---|---|---|
| uniform | monotone rise → 1.9995 | none (0.0000) |
| nobottleneck (d=3) | monotone rise → 2.879 | none (0.0000) |
| mse_skew | monotone rise → 1.990 | none (0.0000) |
| mse_uniform | rise → 1.979, flat | none (0.002 saturation wobble) |

mse_skew cannot fit its task, provably: with logits = FW at d=2, MSE's global optimum
is the best rank-2 approximation of the one-hot targets (YᵀY = diag(6,6,3,3)), which
keeps the two frequent classes and maps every rare row to ZERO — per-rare-row MSE 0.25
and rare weight columns → 0, exactly where the sweep seeds AND the control run converge
(appendix B4; rare columns 0.42 → ≤0.016). Abandoning the rare classes is the optimum,
not a failed optimization: no fork is possible, hence no compression — the loss, not the
optimizer, removes the mechanism, which makes this a principled negative control. (CE
solves the same task at the same rank because it needs argmax margins, not one-hot
values.) Li et al's own statement is a single supplementary caption sentence — "only
information about the most frequently occurring classes are learned" — with no
mechanism; "gradient starvation" was our label, not theirs, and is retired. The dip is
not interpreted; observationally it aligns with the rare weight columns' peak, the
onset of their decay.

Standing caveats when citing the toy as "explaining" the phases: its compression is a
**transient** (recovers to 1.9993 by step 3000), it is the spectral signature of late
rare-class separation, and it rests on an initialization condition the paper never states.
(The earlier "~25% of seeds" caveat was an artifact of our iid init — corrections record.)

## 3. The fourth condition and the kick (single-layer mechanism)

Li et al's Fig 4 initializes from a *constructed* geometry, visible in its gray t=0
markers: frequent-class samples clustered per class with W columns aligned, the two rare
classes **coincident** with zero weights, split only by a jitter δ
(`toy/train.py::_clustered_init`; lr 0.25 calibrated to the paper's eigenvalue scale.
Comparison gotcha: `dup=3` triples raw feature eigenvalues — divide by dup before reading
against the paper's λ axis).

### 3.1 Timing hierarchy

GD preserves the rare-pair swap symmetry exactly, so the coincident init makes the shared
path exact and δ sets the fork time: the antisymmetric mode grows exponentially at measured
rate r ≈ 0.031/step, so t_fork ≈ log(1/δ)/r — verified across five decades:

| δ | predicted fork (mid, cos<0.5) | observed (3 seeds) |
|---|---|---|
| 1e-1 | 96 | 113, 98, 165 |
| 1e-2 | 170 | 175, 163, 206 |
| 1e-3 | 244 | 236, 224, 265 |
| 1e-4 | 319 | 296, 284, 324 |
| 1e-6† | 467 | 415, 401, 443 |

† from the original session sweep; `toy/appendix_sweeps.py` regenerates δ ∈ {1e-1 … 1e-5}.

The FULL compression decline shows iff **t_saturate < t_fork < t_end** — the fork must
land after the entropy-phase spectrum saturates and before the horizon; an early fork
leaves only a reduced transient dip on the rising curve (§3.2). δ sweep (3 seeds; fork
onset = sustained cos(w₂,w₃) < 0.97; decline = largest post-warmup drawdown; figures:
showcase_appendix §B1):

| δ | fork onset | decline (largest post-warmup drawdown) |
|---|---|---|
| 1e-1 | 4–132 (early, mid-rise) | 0.0016–0.0019 |
| 1e-2 | 119–172 | 0.0055–0.0061 |
| 1e-3 | 182–230 | 0.0037–0.0066 |
| 1e-4 | 242–289 (late) | 0.0000–0.0023 |
| 1e-5, 1e-6† | ≈ 300 (at/outside the window) | ≤ 0.0003 |

Two auxiliary checks: base anisotropy is irrelevant
(swapping/equalizing frequent-class scales leaves the decline at 0.0034–0.0075, identical
fork times); the homotopy to iid-tiny shows iid init fails not for lack of anisotropy but
because it *locks t_fork to t_saturate* — the fork always arrives while the spectrum is
still filling and the kick lands on a steeply negative baseline.

### 3.2 Anatomy of the kick

In the saturated near-isotropic spectrum, where does the fork's anisotropy come from?
Fixed-axis Gram decomposition of the canonical run (steps 170–320): not norm asymmetry
(|f₂| ≈ |f₃| within 1%), not class counts. **It is the off-diagonal.** For a 2×2 Gram,
λ₁ − λ₂ = √((Gxx−Gyy)² + 4Gxy²). Pre-fork, total Gxy ≈ +0.03 — a *cancellation*: the
co-traveling pair carries Gxy ≈ +2.2, offset by the tilted frequent classes' ≈ −2.2. The
fork collapses the pair's covariance (+2.2 → +1.4) while the frequent term stays put: net
Gxy swings to −0.86 and the eigengap reopens through 4Gxy² (λ₁−λ₂: 0.58 → 1.78). **The
kick is the collapse of the rare pair's off-diagonal covariance breaking a cancellation**,
independent of which axis "wins"; it scales with the squared fork amplitude.

**Masked, not absent.** d log(λ₁/λ₂)/dt through the fork in both regimes:

| regime | baseline before fork | kick peak | net RM effect |
|---|---|---|---|
| canonical (fork @ ~218, saturated) | −0.2e-3 (≈ flat) | **+2.1e-3** | visible peak-then-decline, 0.0048 |
| homotopy α=0.25 (fork @ ~155, still rising) | −8.7…−0.5e-3 | **+1.0e-3** | transient dip ≈ 0.003; RM still ends at its max |

The fork **always** delivers the kick; saturation-before-fork is the *visibility*
condition — it decides whether the kick shows as a compression phase or is swallowed by
the rising baseline (and the kick is ~2× smaller in the rising regime, compounding the
masking).

### 3.3 The dash lag (fork → decline onset)

The kick scales with squared fork amplitude, so decline onset lags fork onset by however
long the separation needs to reach covariance-moving amplitude:

- rare pair at the exact origin (rx = 0): lag ~5 steps ≈ zero at plot resolution, decline
  0.0067 — the paper's regime (their dashes begin at the fork; their rare classes evidently
  initialize at the origin, which is what `_clustered_init` does);

- off-origin: lag = 10 / 24 / 41 steps at rx = −0.1 / −0.25 / −0.5, decline 0.0061 /
  0.0050 / 0.0029. lag × lr ≈ 6.0–6.3 across lr ∈ [0.15, 0.35] (pure timescale); δ barely
  moves it.

### 3.4 Fidelity vs the paper (canonical run, dup-corrected)

| quantity | paper Fig 4D | repo `single` |
|---|---|---|
| RankMe start / dip | ~1.91 / ~1.88 | 1.933 / 1.906 |
| peak | ~1.99 @ ~150–210 | 1.99964 @ 204 |
| RankMe(300) / decline | ~1.981 / ~0.010 | 1.99313 / 0.0065 |
| λ₁, λ₂ @300 | 15.1, 11.5 (ratio 1.31) | 12.8, 10.9 (ratio 1.18) |
| final w₂/w₃ angle | ~78° | ~85° (cos +0.08) |
| decline onset vs feature fork | coincide (~210) | coincide (204) |

Same shape, ~1.5× shallower decline; their exact init magnitudes/jitter are unpublished.

**Takeaway.** "Skew + bottleneck + CE" is not sufficient: the compression phase needs late
symmetry breaking among the rare classes — rare classes starting (near-)coincident so
their separation arrives after saturation but inside the window. A fourth necessary
condition the paper's text never states, and their Fig 4's init visibly encodes.

## 4. Depth: when the phase pattern survives a stack

**Question.** The phase curve is measured at the END of multi-layer residual LLMs, yet the
32-class multi-layer toys never showed it at any depth or init. Which ingredient was
missing?

**The general identity.** With λ_i(t) the measured covariance spectrum, p_i = λ_i/Σλ_j,
g_i = d log λ_i/dt:

**dS/dt = −Cov_p(g, log p)** — exact, model-agnostic. RankMe falls iff the already-heavy
directions are the faster-growing ones. Warmup collapse = frequent structure learned first
(Cov > 0); entropy-seeking rise = progressively rarer structure (Cov < 0); saturation =
uniform margin growth (Cov ≈ 0); compression event = anything making top directions
outgrow the bulk again — the toys' one-shot fork kick, or the LLMs' *sustained*
concentration mechanisms (sink write / aligned writes).

**Three conditions decide whether a decline shows in the measured curve:**

- **(T) Timing/visibility.** The visible slope is the event's positive Cov net of the
  baseline's; an event landing on a still-rising baseline is masked (§3.2). In LLMs
  visibility is per-band — the event lives in the spectral head while the bulk keeps
  entropy-seeking (final_report §2.1).

- **(K) Kick size.** One-shot events show in proportion to their eigengap change
  (∝ squared fork amplitude); too small drowns in drift.

- **(I) Transmission.** The event lives in the feature geometry; the measurement at the
  output. Residual stack: X_out = X_feat + E, and Weyl bounds every singular value's
  displacement by ‖E‖_op — small-at-init writes carry the spectrum through at any depth.
  Plain stack: X_out = Φ(X_feat) — transmission holds only if Φ starts near identity.
  **The residual provides near-identity by default; a plain stack must have it arranged.**
  This is the corrected content of the retired slogan "the residual stream is
  constitutive".

### 4.1 Ablation table (uncentered final-stream RankMe unless noted)

| # | run (tags) | setup | result |
|---|---|---|---|
| 1 | `multi_residual_nonlinear` + `_d1/_d2/_d3` | MULTI task (32 classes, d=16), iid init, depths 1/2/3/6 | no phases at ANY depth: plateau → collapse at learning onset → recovery to ~2.1–2.5 (centered) |
| 2 | `multi_residual_nonlinear_flat_s0/1` | as 1, rank-1 init (`flat_init`) | no clean rise — killed the "init-rank vs solution-rank" account |
| 3 | per-class timing (trajectories of 1) | — | classes learned in strict frequency order; collapse = frequent-class solution; recovery IS the entropy-seeking phase; the 26 rare forks smear across the rise → masked per (T) |
| 4 | `single_d6_bs_s0/1/2` (= `single_deep`) | constructed init, depth-6 residual, linear, block_scale 0.05 | **FULL PATTERN 3/3**: dip → rise → peak@34–38 = fork → decline 0.093–0.098 → transient recovery |
| 5 | `single_d6_s0/1/2` | as 4, default block init | **no pattern 3/3** (negative control for (I)/(T)); one seed collapses to rank 1 |
| 6 | `single_d6_bs0p3/0p1/0p01_s0` | block_scale sweep | pattern at 0.01/0.05/0.1; DEAD at 0.3 — (I) is a threshold condition |
| 7 | `single_d2_bs_s0`, `single_d12_bs_s0` | depth 2 / 12 | pattern at both; kick grows with depth (0.046 / ~0.10 / 0.128), fork accelerates (76 / 34 / 24) — observed, not explained (§4.3) |
| 8 | `single_d6_id_plain_s0/1/2` | **PLAIN**, blocks I + 0.05·randn | **FULL PATTERN 3/3** (decline 0.092–0.108, fork = peak): phases need near-identity transmission, not the residual |
| 9 | `single_d6_bs_plain_s0/1/2` | plain, block_scale 0.05 | never learns (loss = ln 4): shrinking a plain stack starves signal — why block_scale is the wrong plain analog |
| 10 | `single_d6_plain` | plain, default init | partial learning (loss 0.44), stream pinned at rank 1.0 — (I) fails |
| 11 | `single_d6_bs_nl_s0/1/2` | as 4, tanh blocks | ⚠️ decline (≈0.51) robustly PRECEDES the fork (peak@17, fork@108–137), 3/3 — second, unexplained mechanism (§4.3) |
| 12 | `skewpair_deep_s0/1/2` | constructed-general init, δ=1e-3 | no pattern: fork@132–174 lands mid-rise → masked, exactly (T) |
| 13 | `skewpair_deep_d8_s0/1/2` | as 12, δ=1e-8 | **FULL PATTERN 3/3** — same task, only the fork delay moved: the within-task demonstration of (T) |

### 4.2 The δ toggle on a skewed multi-class task (skewpair_deep)

6 classes 32:16:8:4:2:2 through a real bottleneck (d=4 < 6), constructed-general init,
depth-6 residual. At δ=1e-8, all 3 seeds show the full pattern with exact event alignment (seed 0):

| steps | 101 | 152 | 228 | 299 | 342 | 392 | 514 | 3000 |
|---|---|---|---|---|---|---|---|---|
| RankMe | 2.31 | 2.77 | 2.92 | **2.96 (plateau)** | **2.58 (drop)** | 2.47 | 2.43 | 2.44 |
| cos(w₅,w₆) | +1.00 | +1.00 | +1.00 | +0.98 | **+0.33 (fork)** | +0.17 | +0.09 | — |
| rare-class CE | 1.34 | 0.73 | 0.70 | 0.67 | **0.05** | 0.01 | 0.00 | — |

Wart: the construction is not near-equilibrium for the frequent classes (warmup collapse
~3 → 1.4 in every seed) — the pattern needs saturation before the fork, not a quiet start.

### 4.3 Observed but NOT explained (do not cite as understood)

- Kick scaling with depth (row 7): plausibly the blocks amplify the effective growth rate
  r; no derivation.

- The tanh pre-fork decline (row 11): by the §4 identity something makes heavy directions
  outgrow the bulk before the fork — candidate: tanh-block growth/saturation as a second
  concentration mechanism (LLM-flavored if true) — SPECULATION, unverified.

- The generalized construction's warmup collapse (§4.2); the depth-0 vs depth-1 jump on
  the MULTI task (flat near full rank + underfit vs collapse-and-recover).

- The tanh activation was never motivated or ablated; one lr per task; the layer-scale
  threshold only bracketed (0.1–0.3).


## 5. Decomposition signature (H3.2) and the guard (H3.3)

**H3.2 verdict: SPLIT — ledger signature YES; phases governed by §4's pattern conditions;
the residual-vs-plain contrast dissolves.** The ledger fired unmodified on the multi-layer
dumps (`block_ledger` per blk, `overlap_chi`/`block_block_coupling` at root).

- `multi_residual(_nonlinear)`: no phase trajectory (task fails (T)), but the ledger shows
  the transformer's Pythia signature — **quality declining (−0.5 → −1.8) while χ stays
  positive and stable** — in a toy with no attention, no norms, no tokens.
  Trajectory-independent; also holds 6/6 in the no-norm grid cells (§6).

- `multi_plain`: solves the task (fair-shot lr 0.02, loss 3e-5) without phases —
  over-determined (fails (I) AND (T)), so not evidence about the residual (row 8 is).

- Depth × centering: the "deep streams start collapsed" reading was an uncentered-mean
  artifact (bias means ~50% of uncentered trace at depth ≥ 1; centered, every stream
  starts near-full-rank and the FINAL stream compresses hardest 8.91 → 2.49) —
  showcase_appendix §C.

**H3.3 guard — WORKED AS DESIGNED.** max_k w_k (does one write dominate): multi_residual
(linear) 0.87 = borderline non-representative, excluded from H3.2; multi_residual_nonlinear
0.55 = healthy; all grid cells 0.09–0.19.

## 6. Architecture knobs — is the concentration mode architecturally selected?

> **Scope note.** This grid runs on the 32-class MULTI task, which can NEVER show the
> phase pattern (violates (T) by task construction) — any trajectory-shape reading is
> void. What stands: the sink-write suppression and ledger-carrier attributions (centered,
> trajectory-independent). A rerun under the constructed-timing setup is planned
> (docs/plan.md).

Knobs on the `multi_residual_nonlinear` base: `norm` ∈ {none, **prenorm** (write reads
rms(f_k) — the read-side norm Pythia itself has), **writenorm** (write output per-sample
rms-normalized before the residual add — the OLMo-2 reordered-norm analog), **bothnorm**};
`writes: 2` with parallel vs sequential wiring. Grid {4 norms} × {2 wirings} × 3 seeds;
writes dumped as `blk{k}.{attn,mlp}.out` so the 3-component ledger fires. Carrier = most
negative Δ(summed term) from stream-RankMe peak to final; "min write" = lowest final
centered write RankMe.

| variant × seed | RankMe peak→final | carrier | Δq / ΔI / Δχ | min write RankMe (ratio) | last-2-mlp signed tr | max_k w_k |
|---|---|---|---|---|---|---|
| arch_par s0/s1/s2 | 7.5→2.8, 6.7→2.6, 7.1→2.1 | **quality ×3** | −1.0…−1.4 / −0.2…−0.3 / +0.3…+0.6 | 1.8–2.6 (0.7–1.7×) | +0.82…+0.85 | 0.17–0.19 |
| arch_seq s0/s1/s2 | 7.2→3.1, 6.2→2.6, 7.1→2.5 | **quality ×3** | −1.1…−1.6 / +0.1…+0.3 / +0.1…+0.5 | 1.7–2.4 (1.2–1.8×) | +0.76…+0.88 | 0.13–0.16 |
| arch_pre_par s0/s1/s2 | 7.0→4.8, 7.0→4.4, 6.8→4.0 | quality, chi, chi | −0.1…−0.3 / −0.03…−0.16 / +0.04…−0.27 | 3.2–3.9 (0.03–0.13×) | +0.77…+0.83 | 0.14–0.18 |
| arch_pre_seq s0/s1/s2 | 6.4→4.5, 6.8→4.4, 6.5→4.6 | quality, chi, quality | −0.2…−0.7 / −0.02…+0.4 / −0.1…−0.2 | 3.3–3.8 (0.02–0.08×) | +0.78…+0.87 | 0.12–0.16 |
| arch_wn_par s0/s1/s2 | 8.7→8.7, 6.9→6.7, 7.2→7.2 | none, chi, none | 0…+0.04 / 0…+0.42 / −0.48…0 | 5.6–7.5 (0.30–0.36×) | +0.45…+0.48 | 0.09–0.10 |
| arch_wn_seq s0/s1/s2 | 7.7→7.7, 6.9→6.6, 7.7→7.7 | none, chi, none | 0…+0.08 / 0…+0.36 / −0.48…0 | 6.4–6.7 (0.03–81×)† | +0.42…+0.55 | 0.09 |
| arch_bn_par s0/s1/s2 | 8.5→7.8, 8.0→7.0, 7.1→7.0 | **chi ×3** | −0.1…−0.3 / +0.4…+0.7 / −0.3…−0.5 | 5.1–5.3 (0.41–0.78×) | +0.69…+0.77 | 0.09–0.10 |
| arch_bn_seq s0/s1/s2 | 7.6→7.4, 7.7→7.1, 7.2→7.2 | chi, chi, none | −0.17…0 / 0…+0.57 / −0.56…0 | 5.4–5.8 (0.04–0.42×) | +0.74…+0.78 | 0.10 |

† `blk0.mlp.out` vs the near-init tiny block-0 stream — a scale artifact, not a sink write
(RankMe 6.7).

- **No-norm: Pythia signature 6/6 seeds**, wiring-robust — quality carries the drop, one
  write collapses to RankMe 1.7–2.6 at O(1) energy (rogue-like rank collapse, though never
  Pythia's 20× energy growth).

- **Prenorm suppresses 6/6** — but prenorm is the READ-side norm real Pythia has while
  developing a sink write anyway, so this is a **toy-vs-LLM discrepancy** (the toy's
  mechanism needs raw reads), not evidence about OLMo-2.

- **Writenorm — the actual OLMo-2 analog — suppresses completely, 6/6, both wirings**: 4/6
  runs never decline from peak at all, Δq ≥ 0, min write RankMe 5.6–7.5 — per-sample rms
  forbids energy concentration by construction.

- **Bothnorm** ≈ writenorm with slightly larger residual decline (≤ 1.0), χ-attributed.

- **Interference never becomes the carrier** in any of the 24 runs — where it moves under
  write-side norms it *rises*. Wiring selects nothing. Late writes stay mutually aligned
  in every cell (+0.42…+0.88, never negative): the toy never reproduces Pythia's
  anti-aligned late cancellation.

**Verdict:** write-side normalization removes the sink-write/quality mechanism by
construction, consistent with OLMo-2 never developing one — but no knob produces OLMo's
interference-*carried* decline (clean bounded negative), and the concentration mode is
architecturally suppressible, not re-routable. Remaining candidate knob: a QK-norm analog
(plan.md).

## 7. Deviations from the paper (all forced or documented)

1. **13-eigenvalue floor**: `spectral_metrics`' power-law fit needs ≥13 positive
   eigenvalues → fixed isometric embedding into 16 dims + per-leaf noise floor (3e-3
   relative, spectrally negligible); N lifted by exact sample duplication (`dup`:
   RankMe-invariant, verified bit-identical).

2. **Unstated hyperparameters**: the paper gives no init/lr anywhere. `single`'s init read
   off Fig 4's t=0 markers; lr 0.25 calibrated to their eigenvalue scale. Other variants
   iid std 0.3, lr 0.3.

3. **Rare-pair offset**: `_clustered_init` puts the pair at the exact origin (their
   zero-lag regime, §3.3).

## 8. Files and reproduction

- `toy/train.py` (Spec/VARIANTS/train; new fields `flat_init`, `block_scale`,
  `block_identity`; `_clustered_init_general`), `toy/plots.py`, `toy/appendix_sweeps.py`
  (regenerates the §3 sweeps → `data/results/toy/appendix_sweeps.pt`).

- Named variants cover rows 4 (`single_deep`) and 12–13 (`skewpair_deep`, δ via
  `replace(..., delta=1e-8)`); other §4.1 rows via `replace()` on `VARIANTS['single']`
  (depth/block_scale/block_identity/residual/nonlinear) or
  `VARIANTS['multi_residual_nonlinear']` (depth/flat_init).

- Dumps: `data/inferences/toy_<variant>/`; metrics: `data/results/toy_<variant>/`; raw
  trajectories: `data/results/toy/trajectories_<variant>.pt`; figures: `toy/figures/`.

- Test: `tests/test_toy_format.py` (dump loads via DataAccessor, ledger fires on a 2-block
  micro-run).

## Appendix: proof sketch — why the single-layer decline is transient

Setting: d=2, classes (2,2,1,1), features F free, logits FW, CE, full-batch GD; G = FᵀF.

1. **Shared path (exact).** The GD field commutes with the rare-pair swap: coincident init
   ⇒ exact co-travel; jitter δ evolves under the linearization around the symmetric path.

2. **Fork time (linearized instability, rate measured).** Antisymmetric mode grows at
   measured rate r = 0.031/step ⇒ t_fork ≈ log(1/δ)/r (validated over δ ∈ [1e-6, 1e-1]
   within 10–15%). The fork's spectral effect is a BOUNDED ΔG_xy of order (pair amplitude
   at fork)².

3. **Asymptotics under CE.** After separation, margins grow ~log t with converging
   parameter direction (Soudry-style implicit bias): G(t) = ρ²·F̂ᵀF̂ + o(ρ²) — spectrum
   shape converges to the fixed limit Gram.

4. **Transience.** The fork's ΔG_xy is o(ρ²) against the growing diagonal, so its kick
   decays like (ρ(t_fork)/ρ(t))². Empirically the limit is balanced, RankMe(F̂) ≈ 2, full
   recovery measured (1.9993 at steps 3000–6000).

Proven vs assumed: step 1 exact; step 2's rate measured, not derived; step 3 invokes the
standard separable-CE picture without per-class re-derivation; step 4's RankMe(F̂) ≈ 2 is
numerical. Appendix-grade sketch, flagged accordingly.
