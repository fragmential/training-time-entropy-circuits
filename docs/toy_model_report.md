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
| single | (2,2,1,1) — paper's skew | 2 | — | CE | 200 |
| uniform | (2,2,2,2) | 2 | — | CE | 1000 |
| nobottleneck | (2,2,1,1) | 3 | — | CE | 1000 |
| mse_skew / mse_uniform | skew / uniform | 2 | — | MSE | 1000 |
| multi_residual (+nonlinear) | 32 classes, counts 128↘2 (Zipf-ish) | 16 | 6 | CE | 30k |
| multi_plain | same | 16 | 6 (no residual) | CE | 100k |

## H3.1 — Replication: CONFIRMED

`single` reproduces the three phases in RankMe(features), read out of the standard results
files: **warmup** 1.90 → 1.670 (steps 0–18), **entropy-seeking** 1.670 → 1.998 (18–57),
**compression** 1.998 → 1.978 (57–200) with $\sigma_1$ pulling ahead of $\sigma_2$ — matching
Fig 4 B–D qualitatively (frequent classes separate first in both $W$ and feature space; rare
classes separate during compression, reusing dominant directions).

**All four controls removed compression**, as the paper claims:

| control | RankMe trajectory | compression |
|---|---|---|
| uniform | monotone rise → 1.9995 | none (0.0000) |
| nobottleneck (d=3) | monotone rise → 2.879 | none (0.0000) |
| mse_skew | monotone rise → 1.990 | none (0.0000) |
| mse_uniform | rise → 1.979, flat | none (0.002 saturation wobble) |

**Honesty caveat the reproduction surfaced:** the toy's compression is a *transient during
rare-class separation* and is seed-dependent (~25% of seeds show it clearly); `single` uses
seed 3 with the horizon cut at step 200 — and the paper's fixed 300-step window benefits from
the same trick. Worth remembering when citing the toy as "explaining" the phases: in the toy,
compression is neither inevitable nor an endpoint.

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
2. **Unstated hyperparameters**: the paper gives no init/lr; init std 0.3, lr 0.3 chosen to
   reproduce Fig 4's shape. Uniform control uses 2/class as in the paper.
3. **Seed selection** for `single` (see the caveat under H3.1).

## Files

- `toy/train.py` (Spec/VARIANTS/Toy/dump/train; jsonargparse CLI), `toy/plots.py`
- Dumps: `data/inferences/toy_<variant>/toy-<variant>/step{t}.pt` (44–57 log-spaced ckpts × 8
  variants); metrics: `data/results/toy_<variant>/results_toy-<variant>.npy`; raw trajectories
  for the Fig-4 panels: `data/results/toy/trajectories_<variant>.pt`
- Figures: `toy/figures/fig4_single.png`, `fig_controls.png`, `fig_multi_*.png`
- Test: `tests/test_toy_format.py` (dump loads via DataAccessor, no weight derivation needed,
  ledger fires on a 2-block micro-run)

## What this buys RQ1/RQ3 (next steps, deprioritised)

The toy is now the **component-ablation platform** the LLM scale forbids: the
quality-vs-interference family split (Pythia's rogue write vs OLMo's aligned late writes,
docs/dig_findings.md) is untestable by retraining LLMs, but norm-placement / QK-norm-analog /
parallel-vs-sequential-block variants are one `Spec` field each here. The immediate candidate
experiment: add a normalization knob and ask whether it flips the toy's ledger signature from
quality-driven (observed, Pythia-like) toward interference-driven (OLMo-like) — i.e. whether
*which ledger term carries compression* is an architectural selection, as the cross-family dig
suggests.
