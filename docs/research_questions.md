# Research questions & hypotheses

Priority: **RQ1 > RQ2 ≫ RQ3** (RQ3 is explicitly extra — a bridge experiment, not a pillar).

Notation: $S(\cdot)$ = spectral entropy of the trace-normalized **centered** covariance;
$\mathrm{RankMe} = e^S$ (the $\lambda$-entropy variant, for which the ledger is exact —
`matrix_entropy`/`rankme` in results; `true_rankme` tracks it). Ledger per block $k$
(docs/rank_ledger_notes.md): $\Delta S_k = \chi_k + \mathrm{quality}_k + I_k$, telescoping to
$\log \mathrm{RankMe}(\text{final}) - \log \mathrm{RankMe}(\text{emb})$. Phases (warmup →
entropy-seeking → compression) are **training-time** phases; depth is the axis along which each
training-time phase is carried.

---

## RQ1 (main): Through what layerwise mechanism does the entropy-seeking → compression-seeking rank trajectory arise in residual-stream transformers?

An earlier phrasing was *"Do the phases exist independently in each layer, or are they emergent
from the combination of layers?"* Verdict: keep the mechanism phrasing as the RQ and demote the
old phrasing to the leading hypothesis dichotomy (H1.0 below). Reasons: (i) the old phrasing is
ambiguous about what "layer" means — Li et al already show the *stream at every depth* has the
phases, but the stream at depth $\ell$ is cumulative (it contains all writes $\le \ell$), so
that observation cannot distinguish the two readings; the sharp version of the question is
about the *writes* $c_k$, which only the ledger separates. (ii) as a dichotomy it is answerable
with a yes/no that would end the investigation without explaining anything; "mechanism" asks
for the attribution itself, of which independent-vs-emergent is the first bit. The dichotomy
survives as H1.0 because it is precisely decidable by the instrument.

Why the toy model can't answer it: a one-layer classifier has no stream to write into — every
term except quality is identically zero there. The transformer curve could arise the same way
(one dominant write behaving like the toy) or by a genuinely different route (interference
between writes, stream-mediated coordination); RQ1 is which.

### H1.0 — Independent vs emergent (the old RQ, as a hypothesis)
**Statement:** The training-time phases are *emergent from combination*: the trajectory's
turning points are driven by the interaction terms ($\chi_k$, $I_k$, and the stream-relative
part of quality), not by each write's own spectrum $S(c_k)$ turning over in phase.
**Confidence:** medium.
**Decide by:** (a) per-write spectra: do the $S(c_k)$ curves individually show the three phases
with shared timing? (`{q}_centered.matrix_entropy` at every `blk*.{attn,mlp}.out`); (b) ledger
totals: do $\sum_k \chi_k$, $\sum_k \mathrm{quality}_k$, $\sum_k I_k$ over training explain the
$\Delta$RankMe turning points, and which dominates each phase?
*Independent* iff (a) yes and (b) interaction terms are flat/negligible. *Emergent* iff the
turning points coincide with dominance/sign changes in (b) that (a) does not show. Mixed
outcomes are reportable as such — the ledger makes "how much of each" quantitative.
**Current evidence (pythia-1b, packed):** leans emergent-with-a-twist — the compression-phase
decline is carried by the quality term (−0.5 → −6.3), which is *relational* (each write's
entropy relative to the stream, weighted by energy share), and $\sum I_k$ flips sign
(−3.4 → +2.4); but the quality collapse is concentrated in one write (blk3.mlp, below), which
is closer to "one rogue layer" than to distributed emergence. Cross-family check pending.

### H1.1 — Mechanism of the transition
**Statement (original):** entropy-seeking is $\chi$-driven (blocks write novel directions);
compression is $I$-driven (blocks increasingly write into occupied directions; reinforcement
$\operatorname{tr}P_k > 0$ growing, $I_k$ going negative). The transition point is where the
dominant ledger term flips.
**Confidence:** medium → the $I$-driven half is **already partially refuted** on pythia-1b:
$\sum \chi_k$ stays positive and roughly stable, and the compression decline is
quality-driven, with $\sum I_k$ becoming *positive* (rank-restoring) late. Revised statement to
test cross-family: *compression is quality-driven — training increasingly concentrates each
write's own spectrum (writes become sharper/lower-entropy relative to the stream) — while late
interference partially offsets it.*
**Decide by:** ledger totals vs the RankMe peak: confirmed if the peak checkpoint coincides
(within checkpoint resolution) with the crossover where $\sum \mathrm{quality}_k$'s decline
overtakes $\sum \chi_k$'s growth, consistently in ≥3 of the 4 models. Plot: Exp 4.2 grids +
totals.

### H1.2 — Locus
**Statement (original):** the compression phase is carried by the late layers' terms,
specifically their interference with the top eigendirections of the final spectrum.
**Confidence:** low-medium → **partially refuted** on pythia-1b: the quality collapse is
carried by blk3 (an early-mid block whose rank-one MLP write grows over training to 22× the
stream's trace), while late blocks are where the *positive* interference appears. Revised:
*the compression phase is carried by a small number of identifiable writes (not necessarily
late), and their energy lands on the head of the final spectrum.*
**Decide by:** depth profiles of $\mathrm{quality}_k$ and $I_k$ at/after the transition
(Exp 4.2), plus `eigendirection_attribution` head-mass (share of $v_i^T R_k v_i$ in the top-32
final directions) for the identified writes. Confirmed if ≤3 writes carry ≥70% of the total
quality decline and their attribution is head-concentrated; cross-family bonus if the *type* of
write (MLP, early-mid) recurs. The pythia-1b blk3 anomaly stops being a stray observation
here: its ledger row (huge $w$, low $\chi$, RankMe ≈ 1.2 → amplifier, not novel subspace) is
the prototype.

### H1.3 — Toy-model correspondence (transformer side)
**Statement:** the *last* write's terms qualitatively reproduce the toy-model story —
selection bias ($\Delta\sigma_i \propto \sigma_i$: head of the spectrum grows fastest during
compression) and collapse toward dominant directions — while mid-stack writes do not. This is
the precise sense in which the toy model explains the transformer curve incompletely.
**Confidence:** medium.
**Decide by:** for the last block's write vs mid-stack writes: (a) `eigendirection_attribution`
head-mass over training (last write increasingly head-concentrated during compression);
(b) `gen_block_vs_residual` spectrum (last write's geneigs concentrating: it emphasizes what
the stream already has); (c) growth-rate test on the final spectrum: rank-correlate
$\dot\lambda_i$ with $\lambda_i$ across the compression phase (selection bias, directly).
Confirmed if (a)–(c) hold for the final write and fail for mid-stack ones. The toy side of
this correspondence is RQ3.

**Data:** `block_representations_samples` (packed, log/50, all four models — running) + padded
twin (independent rows, last-token); drift metrics on the 1Bs. Analysis: Exp 4.1–4.4.

---

## RQ2: Does maths/numerical data have more shared geometric structure than quotes/memorised data?

Populations, per model at the **final checkpoint only**: $G$ = general (pile / olmo-mix — the
existing mixes), $T_{\text{math}}$ = maths/numerical text, $T_{\text{mem}}$ = memorised/quotes
data. All statements are about structure *beyond* $G$'s, so every comparison is made after
removing $G$'s geometry.

**The canonical "subtract the general population" operator** is whitening by $G$, i.e. the
generalized eigenproblem $\Sigma_T v = \lambda \Sigma_G v$ (equivalently the spectrum of
$\Sigma_G^{-1/2} \Sigma_T \Sigma_G^{-1/2}$): basis-aware, already implemented
(`generalized_eigenvalues`), and null-calibratable. Literal eigen*spectrum* subtraction is
basis-blind (two populations can share every eigenvalue and no eigenvector); where the
"subtract then resort" experiment is kept, define it as **deflation**: project out $G$'s top-$m$
eigendirections from $\Sigma_T$, eigendecompose the remainder, resort. Both reported;
geneig is primary.

**Null model (essential):** split $G$ in half; geneig of $G_1$ vs $G_2$ gives the noise floor
for every property below (same $N$, same estimator). A size-matched random subset of $G$ posing
as a task is the second null. "Excess structure" = above this floor, not above 1.

**Properties** (proposed; to be audited): for a geneig spectrum $\{\lambda_i\}$ of $T$ vs $G$ —
1. `excess_count` / `excess_mass`: number of $\lambda_i$ above the null's 95th-percentile
   envelope, and $\sum (\log\lambda_i)_+$ over those — "how many directions does the task
   overexpress, and by how much".
2. Spectral family of the geneig spectrum (`spectral_metrics`): entropy/RankMe (flat ≈ task is
   a rescaled $G$; spread = genuine reshaping), $\alpha$.
3. `tail_locality`: for the top excess generalized eigenvectors $v_i$, the energy profile
   $|V_G^\top v_i|^2$ against $G$'s eigenrank → centroid index (the `mean_metrics` profile
   machinery, applied to geneig vectors). High centroid = the shared task structure lives in
   $G$'s tail. *(Needs geneig eigenvectors — small extension: `eigh` instead of `eigvalsh` in
   `generalized_eigenvalues`.)*

### H2.1 — Gradient–activation eigenstructure coupling
**Statement:** in memorised data, gradient eigenstructure is uncorrelated with activation
eigenstructure; in maths data it is more correlated. Rationale: storing unstructured items can
use any free direction; learning a shared mechanism must write into directions already used by
other examples of the same task.
**Confidence:** medium (user's own rating; I concur — see feedback).
**Prior step:** whiten both $\Sigma_{\text{acts}}^T$ and $\Sigma_{\text{grads}}^T$ by their $G$
counterparts (removes trivial, non-task-specific structure).
**Measure, per leaf:** (a) entropy/RankMe of the geneig spectrum of grads w.r.t. acts (the
existing `gen` metric on the whitened pair): $\Sigma_g \propto \Sigma_a$ ⇒ flat spectrum ⇒
high entropy = coupled; (b) CKA$(\Sigma_a, \Sigma_g)$ and top-$k$ subspace overlap
$\|V_a^{(k)\top} V_g^{(k)}\|_F^2 / k$ — small metric additions.
**Decide by:** alignment(math) > alignment(mem) beyond the split-half noise floor, consistent
across leaves (majority of blocks) and across ≥3 of 4 models. Refuted if no consistent
ordering; the confound check (below) must pass for a confirmation to count.

### H2.2 — Excess shared geometry lives in the tail
**Statement:** memorised data has little shared structure beyond $G$'s; niche-but-logical
(maths) data has common directions that are negligible in $G$ — i.e. its excess structure lies
in $G$'s spectral tail.
**Confidence:** high (user's rating).
**Measure:** properties 1–3 on the acts geneig spectrum $T$ vs $G$, plus the deflation variant
(spectral mass surviving removal of $G$'s top-$m$, as a function of $m$).
**Decide by:** `excess_mass`(math) ≫ `excess_mass`(mem) (≥ the gap between mem and the null),
AND `tail_locality` centroid for math's excess directions in $G$'s tail (centroid index beyond
the median rank). Refuted if mem shows comparable excess — with the caveat that verbatim-quote
data shares surface format (see confounds), so a mem excess localized in *head* directions
would still be consistent with H2.2's spirit; report the locality either way.

### Experiment plan (cheap: final checkpoint only)
1. Three collections per model (G / math / mem), same config apart from `dataset_name`:
   `checkpoints: "final"`, boundary leaves + `blk*.mlp.up.{in,out}` with `:both` (grads via
   next-token CE), packed for math/G, **padded for mem** (memorised items must not be packed
   across boundaries), matched token budgets, `storage_format: cov_svd` (geneigs need
   eigvecs — `eigenvalues` is insufficient here).
2. Per-population metrics runs (existing pipeline, unchanged).
3. Cross-population runs via the new `--ref` mechanism in compute_metrics: geneig of every
   (leaf, quantity) against the same leaf of the reference file (`gen_vs_ref` metric). Run
   math-vs-G, mem-vs-G, G₁-vs-G₂ (null), and for H2.1 the whitened acts-grads pass.
4. Grads caveat: backward passes cost 2–3× forward VRAM; still trivial at final-checkpoint-only.

---

## RQ3 (extra, deprioritised): Where along {1-layer linear, deep no-residual, deep residual, transformer} does the three-phase trajectory first arise — and does the minimal model that shows it share the transformer's ledger signature?

This is the toy side of H1.3: Li et al's toy explains the curve in a model with no stream; the
bridge experiment asks what the *minimal* stream-bearing model does, and whether "having the
phases" and "having them by the transformer's mechanism" come apart. Cheap (toy models train in
seconds), high explanatory payoff, but strictly after RQ1/RQ2 work.

### H3.1 — Replication (single-layer)
**Statement:** the paper's analytically-tractable model (linear $f_\theta(S) = S\theta$, logits
$f W$, CE by full-batch GD, skewed class frequencies, bottleneck $d < |\mathcal{V}|$)
reproduces warmup → entropy-seeking → compression in RankMe$(f)$, and the four controls
(uniform labels; $d \approx |\mathcal{V}|$; MSE ×2) each remove compression.
**Confidence:** high — this is replication of published results (their Fig 4 + supplementary
controls).
**Decide by:** qualitative match of Fig 4 B–D (weight/feature trajectories; RankMe and
$\sigma_1, \sigma_2$ curves) and the controls' monotone-expansion signature.

### H3.2 — The residual toy and its ledger signature
**Statement:** a multi-layer *residual* toy ($f_{k+1} = f_k + g_k(f_k)$, $g_k$ linear or
one-hidden-layer, same CE/skew/bottleneck conditions) still shows the three phases in the
stream, and its ledger decomposition qualitatively matches the transformer's (compression
carried by the quality term of few dominant writes) — while the *no-residual* stack
($f_{k+1} = g_k(f_k)$) either loses the phases or shows a different ledger signature.
**Confidence:** low (genuinely open — this is the experiment's point).
**Decide by:** run the real compute_metrics on toy activations dumped in accessor format;
compare ledger-term dominance ordering (quality vs χ vs I over training) and locus
concentration against the transformer runs. Match on both ⇒ the mechanism is
optimization-driven and architecture-generic; phases-without-matching-ledger ⇒ the trajectory
is generic but the transformer's mechanism is architectural; no phases ⇒ the stream changes
the story entirely.

### H3.3 — Representativeness guard (precondition, not hypothesis)
The multi-layer toy may effectively use one layer, making H3.2 vacuous (the stated fear,
formalized): **measure** the ledger trace weights $w_k$ — if $\max_k w_k \to 1$ the variant is
non-representative and H3.2 is *not evaluable* on it (report as such; try depth/width/init
variations before concluding). The ledger is itself the diagnostic for "does it really use its
layers".

**Implementation:** isolated `toy/` directory (architecturally incompatible with the collection
pipeline, so no hooks/model_registry reuse), but dumping per-step activations in DataAccessor
storage format so `scripts/compute_metrics.py` and the analysis notebooks read toy results
exactly like model results. Paper-figure plots (Fig 4 panels + controls) alongside.

---

## Feedback (Claude)

- **RQ1 phrasing:** resolved above — mechanism phrasing as the RQ, your old phrasing as H1.0.
  The strongest reason to keep yours *somewhere* is that it's the version a committee
  immediately understands; the strongest reason not to make it *the* RQ is that Li et al's
  layerwise figure already looks like an answer to its literal reading, and you'd spend the
  defense re-explaining that it isn't.
- **RQ2 is well-posed and cheap, with one load-bearing weakness: the operational definition of
  "memorised".** Verbatim quotes, distributionally-memorised n-grams, and
  extraction-memorised sequences are different populations. Recommendation: use the Merullo
  et al. memorised-sequence sets (the reference repo is already in-tree, and it's Pythia-native)
  as $T_{\text{mem}}$, plus a quotes corpus as a second mem population; if the two mem
  populations disagree, that's a finding, not a failure.
- **Confounds to control in RQ2:** math text has distinctive surface statistics (digits,
  symbols, formatting) that create shared directions for trivial reasons. Controls: match
  populations on sequence length and token-count; add a *shuffled-math* control (same tokens,
  order destroyed within sequences) — structure surviving shuffling is lexical, not
  mechanism-like. This is the difference between H2.2 confirming "maths has shared mechanisms"
  and "maths has digits".
- **H2.1's mechanism story is plausible but the effect could hide in layers you don't expect**
  (memorisation lore says early-mid MLPs). Compute per-leaf and report the depth profile
  rather than one aggregate — the depth profile is also the more thesis-coherent result, since
  it reuses RQ1's where-does-it-live framing. Note the arc: your K-FAC dead end was
  *marginal* acts/grads spectra being trivial; H2.1 asks the *relational* question those
  marginals couldn't see — it retroactively justifies the K-FAC machinery.
- **Eigenspectrum subtraction:** intended as rotate-then-subtract
  ($\operatorname{diag}(V_G^\top \Sigma_T V_G) - \Lambda_G$), which is basis-aware. Whitening is
  its multiplicative sibling — same rotation, per-direction *ratios* instead of differences,
  plus re-diagonalization so off-diagonal task structure in $G$'s basis isn't lost. Ratios stay
  primary for H2.2 because tail excess (tiny $\lambda_G$) is huge in ratio and invisible in
  difference; the additive version is the complementary view.
- **RQ3:** the single-layer replication is near-zero-risk; the residual variant is the only
  genuinely novel piece and the ledger-on-toy comparison is (to my knowledge) not in the
  literature. Worth doing *because* it's cheap — but H3.3's guard should be checked first
  thing, and a failed guard is a fine one-paragraph negative result.
- **Thesis shape:** RQ1 + RQ3 are one arc (mechanism in the wild + minimal model that has it);
  RQ2 is a second arc reusing the same instrument on a different contrast (task geometry
  instead of training time). That's a coherent thesis: *one instrument — covariance geometry
  of residual writes — answering both a dynamics question and a content question.*
