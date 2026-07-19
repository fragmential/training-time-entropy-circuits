# Final report: what the compression phase is made of

*The synthesis doc: current belief only, stated once. Evidence lives in the topic docs
(dig_findings.md, toy.md, ledger.md, rq2.md, sink_literature.md); the question registry is
research_questions.md; figures are analysis/showcase.ipynb. Superseded readings are
collected in the "Corrections record" appendix, not patched inline.*

---

## 0. Executive summary

Li et al describe pretraining as warmup → entropy-seeking → compression phases of the
representation's RankMe. This work asked **what mechanism produces that trajectory in
residual-stream transformers** (RQ1), built an exact instrument to answer it (the per-block
decomposition ΔS = χ + quality + interference; repo name: the rank ledger), and reached
three headline results, each replicated across two model families (Pythia, OLMo-2), two
scales each (~1B, ~7B), and two data geometries (packed all-token, padded last-token), with
nanochat-d12 as a third-family confirmation.

### 0.1 The phases are spectrally local, not global

The measured "compression phase" is a concentration event in the top ~10–30 of 2048/4096
eigendirections. Below that, compression behaves as a *front* that propagates down-spectrum
with decaying amplitude and never reaches the deep spectrum: the deepest band flattens
monotonically through *all* of training — entropy-seeking never ends there. RankMe
conflates the head event with the bulk; windowed metrics (band alpha, head-removed RankMe)
separate them. [showcase §2]

### 0.2 The concentration mechanism is architecture-dependent — and in Pythia it has a name

Pythia compresses via the *quality* term: a single early write (blk3.mlp at 1b/6.9b)
collapses to rank ≈ 1 exactly at the RankMe peak and then grows enormous relative to its
own incoming stream (a block-local, input-relative trace ratio — definitional numbers in
dig_findings; it says nothing about the write's share of the final stream's energy, which
is small). Its dominant variance direction (top centered eigenvector, 94–99% of the write's
variance) is a **sink direction** with two carriers per packed window (position-split
analysis, dig_findings): every window-start position fires unconditionally regardless of
token identity (+1657, 510/512 windows — the packed-stream form of the attention-sinks
literature's "BOS" slot), and each window's FIRST newline fires hardest (+2135; 96% of
newlines are bulk-ordinary). The final blocks learn to *cancel* it before the output
(signed trace → −0.65).

OLMo-2 shows none of this: it compresses via the *interference* term — distributed,
mutually **aligned** late writes — with shallower total compression.

Stated plainly: a substantial part of Pythia's measured "compression" is the model devoting
one enormous direction to its attention-sink slots. [showcase §3–§5]

### 0.3 Norm placement selects the mechanism

Pythia is itself pre-norm (writes READ a normalized stream) and develops the sink write
anyway; OLMo-2's distinguishing feature is the *reordered norm* — RMSNorm on the sublayer
**output**, i.e. the WRITE is per-token normalized before entering the stream, which
forbids an outsized token-activated write by construction. The evidence chain:

- **Toy knob (write-norm, the reordered-norm analog):** suppresses the sink-write
  mechanism completely — 6/6 seeds, both wirings; min write RankMe 5.6–7.5 vs 1.7–2.6
  un-normed. (Read-norm also suppresses the toy's sink write, but real Pythia has
  read-norm and a sink write anyway — a toy/LLM discrepancy, not support.)

- **Data ruled out:** every finding replicates on swapped corpora within a few percent
  (dig_findings.md (dataset-swap controls section)).

- **Pre-registered prediction confirmed (nanochat-d12):** standard pre-norm, OLMo-style
  sequential wiring, no write-norm → it shows the complete Pythia signature
  (quality-carried, Σquality −0.07 → −6.43, Σχ stable ~2.8, interference crossing zero;
  rank-collapsed early-mid writes blk3.mlp RankMe 2.7 / blk4 RankMe 1.7; blk10 rank
  restoration; blk11 head-mass 0.95). Locus note: nanochat's largest single quality
  contributor is the FINAL block (blk11: −2.74 of −6.56, 42%; blk3+blk4 53% together) —
  the confirmed prediction is the carrier TERM, not a blk3 locus.

Robust negative: interference never becomes the carrier under any toy knob (24 runs,
8 cells), and parallel-vs-sequential wiring selects nothing — OLMo's own mode needs an
ingredient the toy lacks (attention / token structure are the candidates).

Caveats: token attribution not run for nanochat (no raw samples yet); endpoint magnitudes
need the token-matched pythia-410m comparison; three families is correlation — the causal
weight rests on toy + prediction.

### 0.4 Depth-axis unification

The sink write's depth shadow is an extreme compression valley (stream RankMe ≈ 2 across
Pythia's mid-stack, entered exactly at blk4, recovered by the late-block cancellation; same
in nanochat) while OLMo-2 has no valley at all — connecting the training-time account to
the compression-valleys literature (2510.06477, which did not test OLMo-2) and extending
the write-norm prediction to a third axis. The massive activation (feature axis), the
valley (depth axis), and the compression phase (training axis) are one object seen three
ways. [showcase §8; dig_findings "Compression valleys"]

### Hypothesis scoreboard

Details: research_questions.md (registry) and §2.3.

| hypothesis | verdict |
|---|---|
| H1.0 independent-vs-emergent | **emergent** in both families (never per-write phase turnover), by different routes |
| H1.1 transition = χ→I dominance flip | **refuted as stated**; replaced: compression = concentration terms (quality in Pythia/nanochat, interference in OLMo-2), χ stable everywhere |
| H1.2 late-layer locus | **holds for OLMo-2, refuted for Pythia** (early sink write + late cancellation) |
| H1.3 toy-like last write | **holds for Pythia** (head-mass 0.26→0.92), weaker for OLMo-2 (0.63, falling) |
| H1.4 write-norm selects the mode | **CONFIRMED, all 3 deciders** (toy suppression; data swap; nanochat prediction) |
| H1.5 sink direction inert at output | **REFUTED**; reconciled with Sun et al via the intervention triptych (zero +1.04 / global-mean +0.74 / token-conditional +0.43): their bias account holds at newline positions (+0.09), but the within-class spike variance is READ BY THE NEXT POSITION (+1.15) — a functional role their constant-bias account does not describe. LayerNorm perturbation remains a recorded confound. Details: dig_findings "interventions" |
| H2.1 grads↔acts coupling: math > memorised | **refuted in all three variants** (incl. the clean subspace test) |
| H2.2 math's excess in G's tail | **PARKED with all of RQ2** (§4) |
| H3.1 toy replication | **confirmed**, incl. panels B/C — requires the paper's CONSTRUCTED init (unstated fourth condition); 4/5 jitter seeds |
| H3.2 deep toy shares the decomposition signature | **resolved: signature-without-phases** — Pythia's signature yes (6/6); phases governed by the pattern conditions (§3); OLMo's mode, no |
| H3.3 representativeness guard | worked as designed |

---

## 1. The instrument (what was built)

- **The exact decomposition** (repo name: the rank ledger): for each block step
  $r_{next} = r + \sum_i c_i$, the identity ΔS = χ + quality + I — overlap (do writes
  occupy new subspaces), block-intrinsic entropy (are writes flatter/sharper than the
  stream, energy-weighted), interference (the pure cross-covariance effect). Exact, per
  block, per checkpoint; telescopes to log RankMe(final) − log RankMe(emb).
  Derivation/symbols: ledger.md. Implementation: in-memory
  cross-covariances (`View.cross`), samples-mode collection with tiny eigenvalues
  persistence, drift metrics via prev-checkpoint spill.

- **Windowed spectral metrics**: head-removed RankMe and band alpha (`tail_rankme`,
  `alpha_window`), validated against the stock implementation to 6e-5 over 1,668 spectra.

- **Cross-run generalized eigenanalysis** (`--ref` / `gen_vs_ref`): any run's leaves
  whitened by a reference run's — the RQ2 primitive.

- **The toy platform**: Li et al's classifier + multi-layer/architecture-knob extensions,
  dumping activations in the repo's storage format so the *unmodified* pipeline
  (decomposition included) runs on toys — the component-ablation platform LLM scale
  forbids.

---

## 2. RQ1: the mechanism of the rank trajectory

### 2.1 The universal: a concentration front, not a contraction

Full tables: dig_findings.md. Key rows, centered final-stream spectra (packed):

| | RankMe (k=0) | top-32 removed | band α 32–128 | band α 128–512 | band α 512+ |
|---|---|---|---|---|---|
| pythia-1b | 540@6k → 190 | 1327@6k → **1459@31k** → 1372 | turns up late (0.70→0.94) | barely (0.68→0.74) | **falls monotonically** (2.42→0.92) |
| pythia-6.9b | 494@6k → 255 | 1576@6k → **2328@70k** → 2209 | late (0.79→0.96) | 0.78→0.83 | **falls monotonically** (2.05→0.95) |
| OLMo-2-7B | 931@8k → 646 | plateau ~2400–2500 | late, then re-drops (re-entropy) | mild | **falls monotonically** |

Falling band-α = flattening = entropy-seeking. The front reaches ranks ~30–100 late,
touches 128–512 barely, never arrives at 512+. In padded last-token geometry the head event
is *stronger* (pre-norm full RankMe collapses to single digits) and the same structure
holds after the final norm — Li et al's exact measurement. Both band metrics are invariant
to head growth; no tail fall-off anywhere.

### 2.2 Family mechanisms

**Pythia — the sink write.** blk3.mlp's write starts ordinary (RankMe ≈ 470/830, ~0.27×
stream energy) and collapses to rank ≈ 1–2 *exactly at the RankMe peak* (1b: steps 2k→5k,
peak 4k; 6.9b: 10k→26k, peak 13k), then grows in energy for the rest of training. It
carries ~90–100% of the total quality decline; the dominant quality block is
scale-dependent within the family (blk3 at 1b/6.9b, blk5 at 410m, split early+final at
160m). Token attribution (final checkpoint): the two sink slots per window (§0.2); top
eigenvalue 94–99% of write variance; mean_frac ≈ 0.003 (pure variance, not a bias —
distinct from the "constant massive activation" reading). The last blocks increasingly
write *against* it (signed trace → −0.65 in both models) — the late positive interference
is rank *restoration*, seen directly as a depth-wise scrub in the mid-stack profile (the
spike rides the stream ~unchanged blk4–blk12, nearly gone at before_final_norm).
Robustness: an explicit EOT does not absorb the first-newline slot (both slots
structural); ordinary tokens carry a decaying mid-stack residue of the direction (~27% at
blk4 → ~3.6% at the final stream). [dig_findings "Sink-slot follow-ups"; showcase §4]

**OLMo-2 — distributed aligned interference.** The anti-Pythia in token space: no
token-concentrated structure (top eigenvalue ~10% of write variance vs 94–99%; scores ±20
vs ±2200), diffuse weakly thematic content, top directions moderately *shared* across late
blocks (cosines 0.47–0.58). No write below RankMe ≈ 200 anywhere; quality *recovers* over
late training (1B: −3.9 → −1.0; 7B: −6.0 → −1.3) while interference falls steadily (→ −4.1
/ −5.5), concentrated in the last 2–3 blocks whose writes are mutually *aligned*
(+0.32…+0.39 — reinforcement, the opposite of Pythia's cancellation). Both scales, both
geometries. OLMo also shows the late *re-entropy* rise Li et al note. [showcase §5]

**Unifying statement:** in every model measured, compression is concentration of write
energy into few shared directions — never a per-write phase turnover (Σχ stable
everywhere). A rank-one write and several mutually-aligned writes are the same geometry
landing in different decomposition bins; the family difference is *where the concentration
lives*, not whether the law holds.

### 2.3 Relation to Li et al (what is theirs, what is new)

Theirs: the phase phenomenology; compression = "anisotropic concentration"; the toy showing
selection bias pushes information into dominant directions. New here: (i) the phases
*coexist spectrally* — the bulk never stops entropy-seeking, which their sequential-phases
framing does not contain; (ii) composition: which writes, which decomposition term, and
that two architectures produce the same curve by different mechanisms; (iii) the mechanism
identity in Pythia (sink direction + late cancellation) and the resulting metric caveat —
an all-token RankMe on a Pythia-lineage model substantially measures sink-slot direction
energy; (iv) the toy's unstated fourth condition — their Fig 4 needs its constructed init,
and its compression is a covariance-collapse transient whose visibility depends on split
timing (§3).

---

## 3. RQ3: the minimal model (arc closed)

Full detail: toy.md.

- **H3.1 confirmed:** the paper's phases + all four negative controls + the Fig-4 B/C
  trajectory geometry reproduce through the unmodified pipeline — but only under the
  paper's CONSTRUCTED init (clustered frequent classes, coincident rare pair + tiny jitter
  δ), an unstated fourth condition. iid init at any scale either breaks the rare-pair
  symmetry at order 1 or welds the fork to spectrum saturation so the decline never
  shows.

- **Mechanism (addendum):** the fork always delivers an eigengap kick — the co-traveling
  pair's off-diagonal covariance cancels the frequent classes' tilt, and the fork collapses
  that cancellation (kick ∝ squared fork amplitude; with the pair at the exact origin,
  decline onset coincides with the fork). Saturation-before-fork is the VISIBILITY
  condition. Toy compression is a transient (RankMe back to ~2.0 by step ~3000).

- **Depth — RESOLVED (toy.md §4):** deep stacks show the phase pattern iff
  (T) the compression event lands after saturation, (K) the kick is large enough, and
  (I) near-identity transmission from features to measurement point. Residual stacks with
  small-init writes show it (2/6/12 blocks, 3/3 seeds, decline onset = the fork); so does
  a PLAIN stack with identity-initialized blocks (3/3); the 6-class bottlenecked task
  toggles the pattern by the fork delay δ alone. The residual is NOT constitutive — it
  provides (I) by default, which is how LLMs inherit it. The 32-class task lacks the
  phases for task-structure reasons (frequency-ordered learning smears its 26 forks across
  the rise).

- **Decomposition signature:** the deep residual toy reproduces *Pythia's* quality-driven
  signature (6/6 no-norm seeds, wiring-independent), including a rank-collapsed dominant
  write — independent of the (missing) phase trajectory.

- **Architecture knobs:** write-norm (the OLMo-2 reordered-norm analog) suppresses the
  sink-write/quality mechanism completely (6/6, both wirings); nothing re-routes
  compression to interference; wiring selects nothing. OLMo's mode needs an ingredient the
  toy lacks. ⚠️ The grid predates the pattern conditions and runs on the no-pattern task —
  trajectory-shape readings void pending the planned rerun (plan.md).

- **H3.3 guard** worked: flagged the linear variant (max w_k = 0.87), passed the nonlinear
  base and all grid cells (0.12–0.19).

---

## 4. RQ2: task geometry (maths vs memorised) — PARKED, DO NOT CITE (Jul 13)

**Why parked:** the documented excess-mass aggregates averaged over leaf×QUANTITY entries —
mixing GRADIENT excess into activation-geometry claims. Acts-only recomputation
(pythia-1b) inverts the quotes/memorized ordering (quotes 274 vs padded null 338;
memorized 926 ≈ 2.7× null) and weakens math (1.8× vs claimed 2.4×). The tail_centroid
conclusion is contaminated the same way (acts-only, math's excess sits HEAD-ward of the
null). Additionally the stored gen_vs_ref results do not record which reference file was
used, and several cross-population comparisons straddled incompatible geometries (packed
all-token vs padded last-token are different measured objects). Redo constraints: plan.md
item 1.

**What survives now:**

- H2.1 (grads↔acts coupling: math > memorised) is **refuted in all three variants**,
  including the clean subspace test (excess-acts vs excess-grads overlap ≈ chance for
  every population) — independent of the contamination.

- The split-half machinery (independent whiteners, raw-space subspaces; rq2.md §6)
  is methodologically sound; its results should be re-checked for the same conflation
  before reuse.

- Design and methods (populations, nulls, excess-mass/tail_centroid definitions):
  rq2.md — still the redo's blueprint.

The original (contaminated) verdict tables remain in git history and rq2.md;
they are deliberately not reproduced here.

---

## 5. Caveats

- Decomposition exactness holds for the λ-entropy RankMe (`matrix_entropy`);
  `true_rankme` tracks it but the identity is not exact there. Both stored.

- Packed rows are within-sequence correlated (standard for this repo's packed runs);
  padded runs at N/d ≈ 4–8 carry Marchenko–Pastur broadening in the deep tail (trends
  valid at fixed N; deep-band absolute levels not comparable across different-N runs).

- Head/bulk cut (k = 32) and band edges are conventions; the front picture is robust to
  them (checked at k ∈ {1,2,8,32,128,512}); the deep-band claims rest on 512+, where the
  cut choice is not load-bearing.

- The sink-write token attribution is final-checkpoint; the direction's token identity
  across training is unmeasured (samples aren't persisted historically).

---

## 6. Open questions

Statements of ignorance, not plans — plans live in plan.md.

- **The origin of OLMo-2's interference mode.** No toy knob (norm placement × wiring, 24
  runs) produces interference-carried compression; the responsible ingredient is missing
  from the toy (attention / token structure are the candidates).

- **How sink-direction variance reaches padded last tokens — narrowed, not closed.** Not
  slot inclusion (checked, dead) and not a modulated final-stream deposit (deposit test);
  ordinary tokens carry the direction mid-stack (~27% at blk4, ~3.6% residue at the final
  stream); unchecked whether that residue quantitatively accounts for the padded
  measurement.

- **Whether QK-norm alone suppresses the sink-write mechanism.** The sink literature finds
  QK-norm the stronger lever at 7B scale; our toy grid only tested write-norm, and OLMo-2
  has both. (nanochat has QK-norm and develops the mechanism — QK-norm alone is
  insufficient — but the toy-side test is missing.)

- **Whether the norm→mechanism link is a cross-family regularity** or a three-family
  correlation plus one confirmed prediction.

- **Why layer output representations align at warmup** (the interference-driven early drop
  at LLM scale) — weights and theory unchecked.

---

## Appendix: corrections record

Superseded readings, kept in one place so no live section carries retraction layers:

- **"Mean anomalies" (mean_frac family split)** — retracted (Jul 13) as a
  misinterpretation; the panels contradicted its cross-model form. Nothing
  mean_frac-related is citable until re-derived. Panels removed from showcase §8.

- **Drift ("two reorganization events then stabilization")** — not confirmed under the
  gap-corrected rate ((1−CKA)/Δ): pythia-1b's fastest per-dex drift sits mid-training and
  OLMo-1B's rates rise late. Stored `cka_drift` additionally subsamples inconsistently at
  late checkpoints (the `_subsample` bug; caveat preserved in dig_findings). Re-derivation:
  plan.md item 4.

- **"38/40 top-spiking tokens are newlines"** — the type-level shadow of the first-newline
  slot; superseded by the position-split two-slot account (§0.2).

- **"The residual stream is constitutive for the phases"** — corrected: the residual
  provides near-identity transmission by default; identity-init plain stacks show the
  pattern (§3).

- **"~25% of seeds" replication caveat on H3.1** — an artifact of our old iid init;
  retracted with the constructed-init discovery.

- **RQ2 original verdicts** — parked wholesale (§4).

- **Slot-inclusion account of padded sink variance** — checked and rejected (5/2,331);
  replaced by the mid-stack-residue account (§6).

- **The toy decline metric (windowed post-global-peak)** — corrected (Jul 19) to the
  largest post-warmup drawdown: the old definition was horizon-dependent (a recovered
  transient pushed the "peak" to the window edge) and undercounted masked dips on some
  seeds. δ-sweep values re-derived in toy.md §3.1.
