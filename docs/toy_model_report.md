# Toy-model reproduction & residual extension (RQ3)

Reproduction of Li et al.'s analytically-tractable classifier (their Fig 4 + supplementary
controls) plus the residual / no-residual multi-layer extensions from
[research_questions.md](research_questions.md) RQ3. Everything lives in the isolated `toy/`
directory; activations are dumped in DataAccessor `acts` format so the **unmodified**
`scripts/compute_metrics.py` computes all metrics — including the rank ledger — exactly as it
does for real models. Figures: `toy/figures/`.

## Setup

The paper's model: features are free parameters ($f_0 = \theta$ rows, since inputs $S = I$
orthonormal), logits $= f\,W$, cross-entropy, full-batch gradient descent. One `Spec` dataclass
parameterizes every variant homogeneously (class counts, $d$, depth, residual/nonlinear, loss,
lr, init); multi-layer variants extend the same model with a stream:
$f_{k+1} = f_k + g_k(f_k)$ (residual; $g_k$ linear, or one hidden tanh layer for the
`nonlinear` variant) or $f_{k+1} = g_k(f_k)$ (plain).

| variant | classes (counts) | d | depth | loss | steps |
|---|---|---|---|---|---|
| single | (2,2,1,1) — paper's skew, clustered init | 2 | — | CE | 300 |
| uniform | (2,2,2,2) | 2 | — | CE | 1000 |
| nobottleneck | (2,2,1,1) | 3 | — | CE | 1000 |
| mse_skew / mse_uniform | skew / uniform | 2 | — | MSE | 1000 |
| multi_residual (+nonlinear) | 32 classes, counts 128↘2 (Zipf-ish) | 16 | 6 | CE | 30k |
| multi_plain | same | 16 | 6 (no residual) | CE | 100k |

## H3.1 — Replication: CONFIRMED (with the paper's constructed init)

`single` (clustered init, rare pair at the exact origin, lr 0.25, 300 steps — see below)
reproduces all four Fig-4 panels: **warmup** RankMe 1.933 → 1.906 (first ~15 steps),
**entropy-seeking** → 1.99964 (peak at step 204), **compression** → 1.99313 at step 300
(decline 0.0065) with $\lambda_1$ accelerating past $\lambda_2$ (12.8 vs 10.9 at 300,
dup-corrected) — and, the part iid init could never give, the paper's rare-class geometry:
the two singleton classes travel one **exact shared path** and fork late (fork onset = the
RankMe peak, step 204, matching the paper's dashes-at-the-fork), ending ~85° apart on the
two reused frequent-class axes (w₂ → −x, w₃ → −y), with the fork driving the RankMe decline.

The key discovery: the paper's Fig 4 initializes from a *constructed* geometry, visible in
its gray t=0 markers (frequent-class samples clustered per class, rare pair **coincident**
with zero weights) — not iid. Reproducing it (`toy/train.py::_clustered_init`,
`clustered=True`) is what makes the shared-path-then-fork exact (GD preserves the rare-pair
swap symmetry; the `delta`-jitter sets the fork time ≈ log(1/δ)/0.031 steps). The compression
decline requires the fork to land after the entropy-phase spectrum saturates but inside the
window — a **fourth necessary condition** beyond the paper's stated skew + bottleneck + CE.
Full evidence, mechanism, and the decline-onset-lag analysis:
[toy_fig4_addendum.md](toy_fig4_addendum.md).

**All four controls removed compression**, as the paper claims:

| control | RankMe trajectory | compression |
|---|---|---|
| uniform | monotone rise → 1.9995 | none (0.0000) |
| nobottleneck (d=3) | monotone rise → 2.879 | none (0.0000) |
| mse_skew | monotone rise → 1.990 | none (0.0000) |
| mse_uniform | rise → 1.979, flat | none (0.002 saturation wobble) |

**Honesty caveat, revised:** an earlier version of this section claimed the toy's compression
was seed-dependent (~25% of seeds) and insinuated the paper's 300-step window was doing the
same cherry-picking. That is **retracted**: under the paper's own (constructed) init the
three phases plus shared-path-fork are robust — 4/5 jitter seeds at δ=1e-3 — and the window
isn't a trick but part of the mechanism. What *does* stand, sharpened by the addendum's
experiments: the toy's compression is a **transient** (RankMe recovers to 1.9993 by step
3000), it is the spectral signature of late rare-class separation, and it needs the
fork-after-saturation timing that the constructed init provides — iid init cannot produce it
at any scale. Worth remembering when citing the toy as "explaining" the phases: in the toy,
compression is neither inevitable nor an endpoint, and it rests on an initialization
condition the paper never states.

## H3.2 — Residual toy's ledger signature: PARTIAL, Pythia-flavored

The rank ledger fired unmodified on the multi-layer dumps (`block_ledger` at every `blk{k}`,
`overlap_chi` / `block_block_coupling` at root, `before_final_norm` spectral family).

- **multi_residual**: warmup collapse (RankMe 5.7 → 1.1) then entropy re-expansion (→ 1.85);
  **no compression phase in the stream** over the horizon. But the ledger shows compression
  *pressure* with the transformer's Pythia signature: **quality declining (−0.5 → −1.8) while
  χ stays positive and stable** — the same term-attribution as pythia-1b/6.9b, present in a
  6-layer linear toy with no attention, no norms, no tokens.
- **multi_residual_nonlinear**: same qualitative shape.
- **multi_plain (no residual): loses the phases entirely** — monotone rank decay into total
  collapse (RankMe → 1.0, loss stuck at 0.97). The stream is not incidental to the phase
  structure; without it the entropy-seeking expansion never happens.

## H3.3 — Representativeness guard

Measured $\max_k w_k$ (ledger trace weight — "does one write dominate the mix"):

| variant | max_k w_k | verdict |
|---|---|---|
| multi_residual (linear) | 0.87 late | **borderline non-representative** — one write dominates; vary depth/width/init before drawing H3.2 conclusions from it |
| multi_residual_nonlinear | 0.55 | healthy — H3.2 evaluable |

Exactly the failure mode the guard was designed for, and it discriminates between the two
variants: the nonlinear residual toy is the legitimate H3.2 test bed.

## Deviations from the paper (all forced or documented)

1. **13-eigenvalue floor**: `spectral_metrics`' power-law fit needs ≥13 positive eigenvalues,
   so dumps go through a fixed isometric embedding into 16 ambient dims + a fixed per-leaf
   noise floor (3e-3 relative — spectrally negligible, above fp32 eigh error), and $N$ is
   lifted by exact sample duplication (`dup`: identical init ⇒ identical dynamics, RankMe
   -invariant; verified bit-identical to dup=1).
2. **Unstated hyperparameters**: the paper gives no init/lr anywhere (tex + supplementary
   checked exhaustively). For `single`, the init geometry was read off Fig 4's gray t=0
   markers (`_clustered_init`) and lr 0.25 calibrated to the paper's eigenvalue scale over
   300 steps ([toy_fig4_addendum.md](toy_fig4_addendum.md)); the remaining variants keep
   iid init std 0.3, lr 0.3. Uniform control uses 2/class as in the paper.
3. **Rare-pair offset**: `_clustered_init` places the coincident rare pair at the exact
   origin, which makes the decline onset coincide with the fork (the paper's zero dash-lag)
   and deepens the decline; an off-origin start (e.g. the (−0.25, 0) first read of their
   gray marker) opens a ~24-step lag instead (addendum, "dash lag").

## Files

- `toy/train.py` (Spec/VARIANTS/Toy/dump/train; jsonargparse CLI), `toy/plots.py`
- Dumps: `data/inferences/toy_<variant>/toy-<variant>/step{t}.pt` (44–57 log-spaced ckpts × 8
  variants); metrics: `data/results/toy_<variant>/results_toy-<variant>.npy`; raw trajectories
  for the Fig-4 panels: `data/results/toy/trajectories_<variant>.pt`
- Figures: `toy/figures/fig4_single.png`, `fig_controls.png`, `fig_multi_*.png`, `fig_arch_grid.png`
- Test: `tests/test_toy_format.py` (dump loads via DataAccessor, no weight derivation needed,
  ledger fires on a 2-block micro-run)

## Architecture knobs — is the concentration mode architecturally selected?

Two `Spec` knobs on the `multi_residual_nonlinear` base test whether the LLM family split
(Pythia rogue-write/quality vs OLMo aligned-writes/interference, docs/dig_findings.md) is
architecturally selected. `norm` places an rms normalization: **prenorm** — each write reads
$\mathrm{rms}(f_k)$, i.e. the read-side norm that *Pythia itself* (like every pre-norm
transformer) already has; **writenorm** — each write's *output* is per-sample rms-normalized
before the residual add, $f_{k+1} = f_k + \mathrm{rms}(g_k(f_k))$, the analog of OLMo-2's
reordered norm (RMSNorm on the sublayer output), which forbids massive write flags by
construction; **bothnorm** — both. The stream itself always stays raw. `writes: 2` with
`parallel` (both writes read the block input) vs sequential (the second reads input + first
write; identical parameter count). Grid {none, prenorm, writenorm, bothnorm} × {parallel,
sequential} × seeds {0,1,2} — variants `arch[_pre|_wn|_bn]_{par,seq}`, the two writes dumped
as `blk{k}.{attn,mlp}.out` so the unmodified 3-component `block_ledger` fires. Carrier = the
most negative Δ(term summed over blocks) from the stream-RankMe peak (centered,
`before_final_norm`) to the final step ("none" when the stream never declines / every |Δ| <
0.05); trace ratio = final ledger-`w` write/stream share; "min write" = the write with the
lowest final centered RankMe. Signature figure: `toy/figures/fig_arch_grid.png` (per-run
stream RankMe + summed quality term, and the final min-write RankMe vs energy-share rogue
check, colored by norm cell).

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

† the 81× is `blk0.mlp.out` against the near-init (tiny) block-0 stream — a scale artifact of
adding unit-rms writes to a small early stream, at RankMe 6.7; not a rogue.

Guard: $\max_k w_k$ = 0.12–0.19 (no-norm / prenorm) and 0.09–0.10 (writenorm / bothnorm; the
per-sample norm equalizes shares almost exactly) — all eight cells are comfortably
representative; H3.3-evaluable.

**Correction (this round).** The previous round read the prenorm result as "direct toy
support for OLMo-2's norm changes removing the Pythia mechanism". That was mislabeled:
prenorm is a *read*-side norm, which is Pythia's own arrangement — real Pythia reads a
normalized stream *and still* develops the rogue write. So prenorm's suppression here is a
**toy-vs-LLM discrepancy** (the toy's rogue mechanism depends on raw reads in a way the
LLM's does not), not evidence about OLMo-2. The architecturally distinctive OLMo-2 feature
is the *reordered* norm — RMSNorm on the sublayer **output**, i.e. the write itself is
per-token normalized before joining the stream — which is the `writenorm` cell (OLMo-2, like
the toy's writenorm, reads the raw stream).

- **No-norm cells reproduce the Pythia signature 6/6 seeds**, robust to wiring: quality
  carries the post-peak decline, and one write collapses to RankMe ≈ 1.7–2.6 at O(1) energy
  vs the stream — rogue-like rank collapse (though never Pythia's 20× energy growth).
- **Prenorm suppresses the mechanism in both wirings, 6/6 seeds**: compression halves (final
  RankMe 4.0–4.8 vs 2.1–3.1 from similar peaks), the quality decline shrinks 3–7×, the worst
  write's RankMe roughly doubles (3.2–3.9) and its energy share collapses to 0.02–0.13× the
  stream — no rogue-like write survives. Per the correction above this is a toy/LLM
  discrepancy, not an OLMo statement.
- **Writenorm — the actual OLMo-2 analog — suppresses the rogue/quality mechanism completely,
  6/6 seeds, both wirings**: in 4/6 runs the stream RankMe never declines from its peak at
  all (6.9–8.7 held to the final step), the summed quality term stops falling entirely
  (Δq ≥ 0), and the lowest-rank write finishes at RankMe 5.6–7.5 — no rogue write exists,
  as the per-sample rms forbids energy concentration by construction. The residual decline
  in 2/6 runs is small (≤ 0.3) and χ-attributed with ΔI *positive*.
- **Bothnorm** behaves like writenorm with a slightly larger residual decline (≤ 1.0),
  χ-attributed 5/6 (none 1/6); ΔI is again positive (+0.4…+0.7).
- **Interference never becomes the carrier** in any of the 24 runs across the 8-cell grid —
  where it moves at all under write-side norms it *rises*, the opposite of OLMo's falling I.
- **Parallel vs sequential selects nothing** — same carrier, same rogue behavior, only ΔI's
  sign wobbles under no-norm/prenorm.
- Late writes stay mutually **aligned** in every cell — +0.76…+0.88 (no-norm/prenorm/bothnorm),
  lowered to +0.42…+0.55 by writenorm but never negative: the toy never reproduces Pythia's
  anti-aligned late cancellation, knobs or not.

**Verdict: a half-flip, now with the right label on it.** Write-side normalization — the
OLMo-2 reordered-norm analog — removes the rogue/quality-collapse mechanism by construction
(more completely than read-norm: mostly no compression phase at all), consistent with OLMo-2
never developing a massive-flag rogue write. But no knob produces OLMo's
interference-*driven* compression (a clean bounded negative for this grid), and write wiring
is irrelevant. The concentration *mode* is architecturally suppressible here, not
architecturally re-routable. Separately, the prenorm cell shows the toy's mechanism needs raw
*reads*, which real pre-norm Pythia does not have — a caveat on how literally the toy's rogue
maps onto Pythia's.

## What this buys RQ1/RQ3 (next steps, deprioritised)

The toy is now the **component-ablation platform** the LLM scale forbids: the
quality-vs-interference family split (Pythia's rogue write vs OLMo's aligned late writes,
docs/dig_findings.md) is untestable by retraining LLMs, but norm-placement / QK-norm-analog /
parallel-vs-sequential-block variants are one `Spec` field each here. The norm-placement
(read-norm, write-norm, both) and parallel/sequential knobs are done (see "Architecture
knobs" above: write-norm — the OLMo-2 reordered-norm analog — removes the quality-collapse
by construction, nothing re-routes it to interference); the remaining candidate is a QK-norm
analog.
