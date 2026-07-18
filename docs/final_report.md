# Final report: what the compression phase is made of

*Status: living document (built overnight Jul 5–6). Sections marked ⏳ fill in as tonight's
runs land. Detailed evidence lives in [dig_findings.md](dig_findings.md),
[toy_model_report.md](toy_model_report.md), [rank_ledger_notes.md](rank_ledger_notes.md);
this report is the synthesis.*

---

## 0. Executive summary

Li et al. describe pretraining as warmup → entropy-seeking → compression phases of the
representation's RankMe. This work asked **what mechanism produces that trajectory in
residual-stream transformers** (RQ1), built an exact instrument to answer it (the rank
ledger), and reached three headline results, each replicated across two model families
(Pythia, OLMo-2), two scales each (~1B, ~7B), and two data geometries (packed all-token,
padded last-token):

1. **The phases are spectrally local, not global.** The measured "compression phase" is a
   concentration event in the top ~10–30 of 2048/4096 eigendirections. Below that, compression
   behaves as a *front* that propagates down-spectrum with decaying amplitude and **never
   reaches the deep spectrum**: ranks ≳512 flatten monotonically through *all* of training —
   entropy-seeking never ends there. RankMe conflates the head event with the bulk; windowed
   metrics (band alphaReQ, head-removed RankMe) separate them.

2. **The concentration mechanism is architecture-dependent — and in Pythia it has a name.**
   Pythia compresses via the ledger's *quality* term: a single early write (blk3.mlp, same
   absolute index at 1b and 6.9b) collapses to rank ≈ 1 exactly at the RankMe peak and then
   grows enormous relative to its own incoming stream (a block-local, input-relative trace
   ratio — definitional statement and numbers in dig_findings; it says nothing about the
   write's share of the final stream's energy, which is small). Its dominant variance
   direction (the top centered
   eigenvector of the write's output, carrying 94–99% of its variance — "the rogue direction")
   is a **sink direction** with two carriers per packed window (position-split analysis,
   dig_findings): every window-start position fires unconditionally regardless of token
   identity (+1657, 510/512 windows — the packed-stream form of the attention-sinks
   literature's "BOS" slot), and each window's FIRST newline fires hardest (+2135; 96% of
   newlines are bulk-ordinary — the earlier "newline-activated, 38/40 top rows" framing was
   this mixture read per token type). The final blocks learn to *cancel* it before the output
   (signed trace → −0.65). OLMo-2 shows none of this: it compresses via the *interference*
   term — distributed, mutually **aligned** late writes — with shallower total compression.
   Stated plainly: a substantial part of Pythia's measured "compression" is the model devoting
   one enormous direction to its attention-sink slots.

3. **Norm placement and the rogue mechanism — corrected reading.** Pythia is itself pre-norm
   (writes READ a normalized stream) and develops the rogue write anyway; OLMo-2's
   distinguishing feature is the *reordered norm* — RMSNorm on the sublayer **output**, i.e.
   the WRITE itself is per-token normalized before entering the stream, which forbids a
   outsized token-activated write by construction (plus QK-norm, irrelevant to an MLP write). The toy
   grid tested read-norm (`prenorm` = normalize what the write reads): it suppressed the
   rogue 6/6 seeds — but since real Pythia has read-norm and a rogue anyway, that is a
   toy-vs-LLM discrepancy, NOT support for the OLMo story. The architecturally relevant knob —
   **write-norm, the OLMo-2 reordered-norm analog — is now tested and suppresses the rogue
   mechanism completely** (6/6 seeds, both wirings; 4/6 runs lose the post-peak decline
   entirely; min write RankMe 5.6–7.5 vs 1.7–2.6 un-normed). Robust toy results: interference
   never becomes the carrier under any tested knob (24 runs, 8 cells), and
   parallel-vs-sequential wiring selects nothing — OLMo's own interference mode needs an
   ingredient the toy lacks (attention/token structure).

**Late addition — the H1.4 nanochat prediction CONFIRMED** (analysis of the third-family
data): nanochat-d12 (standard pre-norm, OLMo-style sequential wiring, **no write-norm**)
shows the complete Pythia signature despite the OLMo wiring — quality-carried compression
(Σquality −0.07 → −6.43 dominating; Σχ stable ~2.8), interference *rising and crossing zero*
(−1.3 → +0.98), **rank-collapsed early-mid writes carrying much of the collapse** (blk3.mlp
RankMe 2.7 at 1.3× its incoming stream, blk4 RankMe 1.7). ⚠️ Locus correction (Jul 17,
per-block quality decomposition, dig_findings): nanochat is NOT blk3-dominated the way the
pythias are — its largest single quality contributor is the FINAL block (blk11: −2.74 of
Σ −6.56, 42%), with blk3+blk4 at 53% together; the confirmed prediction is the carrier
TERM (quality; interference crossing zero), not a blk3 locus. Continuing:
late-block rank restoration (blk10 I = +1.6) and head-targeting (blk11 head-mass 0.95). The
architecture story now stands on: 2+2 observation → data ruled out (swap) → knob identified
(toy write-norm) → prediction confirmed (nanochat). Caveats: token attribution not run for
nanochat (no raw samples yet); endpoint magnitudes need the token-matched pythia-410m
comparison; three families is correlation — the causal weight rests on toy + prediction.

**Depth-axis unification (late addition):** the rogue write's depth shadow is an extreme
compression valley (stream RankMe ≈2 across pythia's mid-stack, entered exactly at blk4,
recovered by the late-block cancellation; same in nanochat) while **OLMo-2 has no valley at
all** — connecting our training-time account to the attention-sinks/compression-valleys
literature (2510.06477, which did not test OLMo-2) and extending the write-norm prediction to
a third axis. Details: dig_findings "Compression valleys".

RQ2 (does maths have more shared geometry than memorised data, beyond the general
population's?) — **⚠️ the paragraph below is UNDER REVISION (see §4 banner): its numbers
mixed gradient excess into the activation claims; acts-only inverts the quotes/memorized
ordering.** As originally written: **maths text does carry genuine shared structure beyond the
general population** (2.3–2.4× the split-half null's excess, both families) though
mid-spectrum rather than in the tail; the two "memorised" populations split sharply
(high-entropy verbatim strings add almost nothing beyond early-layer lexical novelty; famous
quotes are the most structured population measured); and H2.1's coupling prediction comes out
*inverted* — memorised data has the most acts-aligned gradients, not the least. Details §4.

**Hypothesis scoreboard** (details in §2.3, §3, §4):

| hypothesis | verdict |
|---|---|
| H1.0 independent-vs-emergent | **emergent** in both families (never per-write phase turnover), by different routes |
| H1.1 transition = χ→I dominance flip | **refuted as stated**; replaced: compression = concentration terms (quality in Pythia / interference in OLMo), χ stable everywhere |
| H1.2 late-layer locus | **holds for OLMo, refuted for Pythia** (blk3 early rogue + late cancellation) |
| H1.3 toy-like last write | **holds for Pythia** (head-mass 0.26→0.92), weaker for OLMo (0.63, falling) |
| H1.4 write-norm selects the concentration mode | **CONFIRMED, all 3 deciders**: toy write-norm suppresses the rogue (6/6, both wirings); dataset swap rules the data out; and the nanochat PRE-REGISTERED prediction hit — nanochat-d12 (pre-norm, sequential, no write-norm) shows the full Pythia signature |
| H1.5 rogue direction functionally inert at output | **REFUTED**, and reconciled with Sun et al via the intervention triptych (zero +1.04 / global-mean +0.74 / token-conditional +0.43): the apparent divergence was a protocol artifact (global mean ≈ zeroing for spike tokens); with the aligned token-conditional protocol their bias account holds at newline positions (+0.09) — but the within-class spike variance is READ BY THE NEXT POSITION (+1.15 after newlines, worse than zeroing), a functional role their constant-bias account does not describe — though the recorded alternative reading (mean-substitution also perturbs LayerNorm statistics, a known confound of massive-activation ablations; research_questions) is not excluded. Details: dig_findings 'interventions' section |
| H2.1 grads↔acts coupling: math > memorised | **refuted in all three variants** — incl. the clean subspace test: excess-acts and excess-grads subspaces overlap at ≈ chance for every population (max ~2× null, OLMo math-web) |
| H2.2 math's excess structure in G's tail | ⚠️ UNDER REVISION (§4 banner — acts/grads conflation). Original verdict: **half-supported**: math has real excess (2.3–2.4× null, both families) but it lives mid-spectrum, not the tail; mem populations split (strings ≈ nothing, quotes = largest excess of all) |
| H3.1 toy replication | **confirmed**, incl. panels B/C — requires the paper's CONSTRUCTED init (an unstated fourth condition beyond skew+bottleneck+CE; read off their figure's t=0 markers). Robust 4/5 jitter seeds under that init; the earlier seed-dependence caveat was an artifact of our iid init and is retracted. Details: toy_fig4_addendum.md |
| H3.2 residual toy shares transformer's ledger signature | **partial**: Pythia's signature, yes (6/6 no-norm seeds); OLMo's, no |
| H3.3 representativeness guard | worked as designed (flagged the linear variant, passed the nonlinear + all grid cells) |

---

## 1. The instrument (what was built)

- **The exact rank ledger**: for each block step $r_{next} = r + \sum_i c_i$, the identity
  $\Delta S = \chi + \text{quality} + I$ — overlap (do writes occupy new subspaces), spectral
  quality (are writes flatter/sharper than the stream, energy-weighted), interference (the
  pure cross-covariance effect). Exact, per block, per checkpoint; telescopes to
  $\log \mathrm{RankMe}(\text{final}) - \log \mathrm{RankMe}(\text{emb})$. Symbol-by-symbol:
  rank_ledger_notes.md / ledger_symbols.md (both in docs/). Implementation: in-memory cross-covariances
  through the accessor resolver (`View.cross`), samples-mode collection with tiny
  (eigenvalues) persistence, drift metrics via prev-checkpoint spill.
- **Windowed spectral metrics**: head-removed RankMe and band alphaReQ (`tail_rankme`,
  `alpha_window` virtual hooks; the stock alpha's 11–100 window overlaps the head), validated
  against the stock implementation to 6e-5 over 1,668 spectra.
- **Cross-run generalized eigenanalysis** (`--ref` / `gen_vs_ref`): any run's leaves whitened
  by a reference run's — the RQ2 primitive (+ `tail_centroid` for where excess structure lives
  in the reference spectrum).
- **The toy platform**: Li et al's classifier + residual/multi-layer/architecture-knob
  extensions, dumping activations in the repo's storage format so the *unmodified* pipeline
  (ledger included) runs on toy models — the component-ablation platform LLM scale forbids.
- Notebook surface: Exp 4.1–4.8 in experiments.ipynb + the padded twin (coupling matrices,
  ledger grids, block↔final coupling, ablation/overlap/drift, band spectra, sub-block ledger
  (OLMo), write-mean panels, ledger contribution stack).

## 2. RQ1: the mechanism of the rank trajectory

### 2.1 The universal: a concentration front, not a contraction

Full tables: dig_findings.md ("head phenomenon" + "windowed alphaReQ" + band data). Key rows,
centered final-stream spectra (packed):

| | RankMe (k=0) | top-32 removed | band α 32–128 | band α 128–512 | band α 512+ |
|---|---|---|---|---|---|
| pythia-1b | 540@6k → 190 | 1327@6k → **1459@31k** → 1372 | turns up late (0.70→0.94) | barely (0.68→0.74) | **falls monotonically** (2.42→0.92) |
| pythia-6.9b | 494@6k → 255 | 1576@6k → **2328@70k** → 2209 | late (0.79→0.96) | 0.78→0.83 | **falls monotonically** (2.05→0.95) |
| OLMo-2-7B | 931@8k → 646 | plateau ~2400–2500 | late, then re-drops (re-entropy) | mild | **falls monotonically** |

Falling band-α = flattening = entropy-seeking. The front reaches ranks ~30–100 late in
training, touches 128–512 barely, and never arrives at 512+. In padded last-token geometry the
head event is *stronger* (pre-norm full RankMe collapses to single digits; the top-1 direction
dominates) and the same structure holds after the final norm — Li et al's exact measurement.
Both band metrics are invariant to head growth (slope is scale-free; band RankMe renormalizes
within the band), so these are genuine shape changes, not normalization artifacts. No
tail fall-off anywhere (band RankMes at 512+/1024+ are monotone non-decreasing).

### 2.2 Family mechanisms

**Pythia — the sink-direction rogue write.** blk3.mlp's write starts ordinary (RankMe ≈ 470/830,
~0.27× stream energy) and collapses to rank ≈ 1–2 *exactly at the RankMe peak* (1b: over steps
2k→5k, peak 4k; 6.9b: 10k→26k, peak 13k), then grows in energy for the rest of training
(local trace ratios: dig_findings' definitional sentence). It carries ~90–100% of the
total quality decline (per-block table in dig_findings). Token attribution (raw
samples, final checkpoint; position-split Jul 13): the direction is a **sink direction with
two slots per window** — position 0 unconditionally (+1657 on arbitrary content tokens) and
the window's first newline (+2135; the remaining 96% of newlines are bulk-ordinary, so the
earlier "38/40 top-spiking tokens are `\n`" reading was the type-level shadow of the
first-newline slot); the top eigenvalue holds 94–99% of the write's variance; mean_frac ≈
0.003 (pure variance, not a bias — distinct from the "constant massive activation" reading). The last blocks increasingly write *against* it (signed trace vs blk3 →
−0.65 in both models) — the late positive interference is rank *restoration*, seen directly
as a depth-wise scrub in the mid-stack profile (the spike rides the stream ~unchanged
blk4–blk12, nearly gone at before_final_norm). Robustness follow-ups (Jul 13: an explicit
EOT does not absorb the first-newline slot — both slots structural; ordinary tokens carry a
decaying mid-stack residue of the direction): dig_findings "Sink-slot follow-ups" +
showcase §4 depth figure; how that residue relates to the padded last-token measurement is
an open question (§6). Figures:
experiments.ipynb Exp 4.9 (write trajectory + newline-vs-other projection histogram) and
Exp 4.10 (cancellation trajectory, head-mass, per-write concentration bars).

**OLMo-2 — distributed aligned interference.** Token attribution of the late writes (raw
samples, 7B) finds the anti-Pythia: no token-concentrated structure (top eigenvalue ~10% of write variance
vs Pythia's 94–99%; top 0.1% of tokens <1% vs ~33%; scores ±20 vs ±2200), diffuse weakly
thematic content (dates/metadata at blk31, math/punctuation at blk30), and top directions
moderately *shared* across late blocks (cosines 0.47–0.58) — many writes reinforcing common
ordinary-token subspaces rather than one write flagging a token class. No write below RankMe
≈ 200 anywhere; quality
*recovers* over late training (1B: −3.9 → −1.0; 7B: −6.0 → −1.3) while interference falls
steadily (→ −4.1 / −5.5), concentrated in the last 2–3 blocks whose writes are mutually
*aligned* (+0.32…+0.39 — reinforcement, the opposite of Pythia's cancellation). OLMo's late
writes were also claimed to be "the mean-heavy ones (mean_frac 0.15–0.6), where Pythia's
rogue is mean-free" — ⚠️ that mean_frac family split is FLAGGED as unverified/faulty (see
appendix: the figures contradict its cross-model form). Both scales, both geometries (for
the non-flagged claims). OLMo also shows the late *re-entropy* rise Li et al note.

**Unifying statement:** in every model measured, compression is concentration of write energy
into few shared directions — never a per-write phase turnover ($\sum\chi$ stable everywhere).
A rank-one write and several mutually-aligned writes are the same geometry landing in
different ledger bins; the family difference is *where the concentration lives*, not whether
the law holds.

### 2.3 Hypotheses (scoreboard reasoning)

- **H1.0** — emergent, both families: the trajectory is carried by relational terms (quality
  is stream-relative; interference is cross-write), and no per-write spectrum turns over in
  phase. But "emergent" splits: Pythia = one write + ensemble cancellation; OLMo = ensemble
  alignment.
- **H1.1** — the original (entropy-seeking χ-driven, compression I-driven) is refuted as a
  universal: χ is roughly stable through both phases everywhere. Compression is quality-driven
  (Pythia) or interference-driven (OLMo); in pythia-1b the interference total crosses zero at
  ≈ the RankMe peak — suggestive timing, but the driver is quality.
- **H1.2** — locus: late-blocks holds for OLMo; Pythia's locus is early (blk3) with late-block
  *cancellation* — same blocks, opposite role.
- **H1.3** — Pythia's last write becomes strongly head-targeted (eigendirection head-mass
  0.26 → 0.92 at 6.9b), matching the toy's selection-bias story; OLMo's is 0.63 and falling.
  The toy correspondence is itself family-dependent.

### 2.4 Relation to Li et al (what is theirs, what is new)

Theirs: the phase phenomenology; compression = "anisotropic concentration" (their takeaway);
the toy showing selection bias pushes information into dominant directions. New here: (i) the
phases *coexist spectrally* — the bulk never stops entropy-seeking, so "compression phase" is
a head event superimposed on continuing expansion, which their sequential-phases framing does
not contain; (ii) composition: which writes, which ledger term, and that two architectures
produce the same curve by different mechanisms; (iii) the mechanism identity in Pythia
(sink direction — window-start + first-newline slots — plus late cancellation) and the
resulting metric caveat — an all-token RankMe on a Pythia-lineage model is substantially
measuring sink-slot direction energy; (iv) the
toy's unstated fourth condition — their Fig 4 needs its constructed init, and its compression
is a covariance-collapse transient whose visibility depends on split timing (§3, addendum).

## 3. RQ3: the toy bridge (deprioritised arc, now closed)

Full detail: toy_model_report.md + toy_fig4_addendum.md. (a) **H3.1 confirmed**: paper's
phases + all four negative controls + the Fig-4 B/C trajectory geometry (rare-class shared
path → late split) reproduce through the unmodified pipeline — but only under the paper's
CONSTRUCTED init (clustered frequent classes, coincident rare pair + tiny jitter δ; read off
their figure's gray t=0 markers), an unstated fourth condition: iid init at any scale either
breaks the rare-pair symmetry at order 1 (old spec — the "decline" it showed was a
stunted-class artifact, and our earlier "~25% of seeds" caveat is retracted accordingly) or
welds the split time to spectrum saturation so the decline never prints. Mechanism (addendum):
the split always delivers an eigengap kick — the co-traveling pair's off-diagonal covariance
cancels the frequent classes' tilt, and the fork collapses that cancellation (the kick scales
with squared fork amplitude — with the rare pair at the exact origin, as in the paper and the
canonical run, decline onset coincides with the fork) — and
saturation-before-split is the VISIBILITY condition determining whether the kick prints as a
compression phase or is swallowed by the rising baseline. Toy compression remains a transient
(RankMe recovers to ~2.0 by step ~3000).
(b) **Residual vs plain — RESOLVED (Jul 13, full study: toy_multilayer.md):** the phase
curve prints through deep stacks whenever three conditions hold — a compression event
timed after spectrum saturation, a large-enough kick, and near-identity transmission from
the feature layer to the measurement point. Residual stacks with small-at-init writes
print it (2/6/12 blocks, decline onset = the rare-pair fork, 3/3 seeds), and so does a
NO-residual stack with identity-initialized blocks (3/3) — so the corrected form of "the
residual stream is constitutive" is: the residual provides near-identity transmission BY
DEFAULT (LLMs inherit it architecturally), while plain stacks only have it if arranged.
The MULTI task lacks the phases for task-structure reasons (frequency-ordered learning;
its 26 rare forks smear across the entropy-seeking rise — masked; the collapse is the
stream snapping onto the frequent-class solution and the recovery IS the entropy-seeking
phase). Supporting facts folded in: the plain MULTI stack solving the task (fair-shot lr,
final loss 3e-5) without phases is over-determined — it fails transmission (default init)
AND its task fails the timing condition — so it is not evidence that replacement
architectures cannot have phases; and the earlier per-depth readings were
uncentered-measurement artifacts (accumulated write-bias means, ~50% of uncentered trace
at depth; showcase_appendix §C). Every ablation is plotted in
analysis/toy_multilayer.ipynb; open items (nonlinear pre-fork decline; kick-vs-depth
scaling) in toy_multilayer.md §6.
(c) **Ledger signature**: the
residual toy reproduces *Pythia's* quality-driven signature (6/6 no-norm seeds, wiring-
independent), including a rogue-like rank-collapsed write (though never Pythia's 20× energy
growth). (d) **Architecture knobs**: pre-norm suppresses the rogue/quality mechanism 6/6
(compression halves; rogue energy share ×10–50 down) — supporting the claim that OLMo-2's
normalization removes the Pythia mechanism — but nothing re-routes compression to
interference, and parallel-vs-sequential selects nothing: OLMo's mode needs an ingredient the
toy lacks (attention / token structure are the candidates). (e) **H3.3 guard** worked: flagged
the linear multi-layer variant (max w_k = 0.87), passed the nonlinear base and all grid cells
(0.12–0.19).

## 4. RQ2: task geometry (maths vs memorised) — ⚠️ UNDER REVISION, DO NOT CITE (Jul 13)

**This section's headline numbers are contaminated and its verdicts are suspect.** The
"excess mass" aggregates below averaged over leaf×QUANTITY entries — i.e. they mixed
GRADIENT excess into an activation-geometry claim. Recomputed on activations only
(pythia-1b): quotes 274 vs padded null 338 (BELOW the null — "quotes = largest excess of
any population" was carried by quotes' gradient excess, 1385), memorized 926 (the actual
largest padded population on acts, ~2.7× null), math-web 209 vs packed null 117 (~1.8×,
not 2.4×). Every H2.2 verdict needs re-derivation with acts and grads reported separately.
The tail_centroid conclusion ("mid-spectrum, matches the null") is contaminated the same
way: the doc numbers (math 510 vs null 555) are the acts+grads mixtures; acts-only, math's
excess sits HEAD-ward of the null (816 vs 1012 of 2048) — the tail prediction fails even
more clearly, but there is a real locality signal the mixture erased.
Parked by user decision (plan.md); the split-half coherence results use a different
(subspace-overlap) machinery and are not automatically implicated, but should be re-checked
for the same conflation before reuse.

**Design** (final checkpoint only; pythia-1b-deduped + OLMo-2-0425-1B; grads via `:both`
hooks at blk\*.attn.in / blk\*.mlp.up.in / before_final_norm; storage cov_svd):

| population | dataset | geometry |
|---|---|---|
| G, G′ (null pair) | native mix (pile / olmo-mix), seeds 42/43 | packed AND padded/last |
| T_math (text) | open-web-math | packed |
| T_math (answer slot) | GSM8K question+answer, padded/last = the answer position | padded |
| T_mem (verbatim) | EleutherAI pythia-memorized-evals (deduped.1b; skews to high-entropy strings — precisely H2.2's "memorised spurious data") | padded (pythia only) |
| T_mem (quotes) | quotes-500k | padded |

Analysis: per-population metrics, then cross-population `gen_vs_ref` (T-vs-G geneig per leaf
per quantity, geometry-matched), with G-vs-G′ as the null envelope. Properties: excess
mass/count of geneigs above the null, geneig-spectrum entropy/α, and `tail_centroid` (where
excess directions live in G's eigenrank — H2.2's tail-locality). H2.1 = acts↔grads coupling
(the `gen` metric) compared math-vs-memorised after G-whitening. Deferred to a second pass:
shuffled-math lexical control; multi-checkpoint sweep.

**Methods:** the full step-by-step derivation — whitening, reading a geneig spectrum,
excess mass / tail_centroid, both experiments (A: task-vs-G geneig; B: acts↔grads
coupling), and the not-run additive variant — lives in docs/rq2_methods.md; the split-half
estimator (independent whiteners, raw-space subspace overlap) in docs/splithalf_note.md.

**Results** (both models at 1B scale; geometry-matched nulls; "excess mass" =
$\sum (\log\lambda)_+$ over geneig $\lambda > 1$ vs G, averaged over 66 leaf×quantity entries):

| population (vs its G, geometry) | pythia-1b | OLMo-2-1B | null (G′ vs G) |
|---|---|---|---|
| math web text (packed) | **148** (2.4× null), n(λ>2)=50 | **201** (2.3×), n=95 | 61 / 88, n=4/8 |
| math answer-slot (padded) | 377 (≈1.0× null) | 599 (1.6×) | 368 / 365, n≈200 |
| quotes (padded) | **830 (2.3×)** | **1524 (4.2×)** | " |
| verbatim-memorized (padded) | 465 (1.3×), early-leaf spikes ≈1400 | — (pythia-only) | " |

- **H2.2, math side: SUPPORTED in packed geometry** — maths text carries genuine shared
  structure beyond the general population, 2.3–2.4× the null's excess mass and 6–12× its
  λ>2 count, consistent across both families. The answer-slot variant is ≈ null at pythia
  (GSM8K's ~7.5k rows + number-token last positions are too weak a probe; OLMo shows 1.6×).
- **H2.2, tail-locality: NOT confirmed** — the excess directions' centroid in G's eigenrank
  (`tail_centroid` ≈ 510–717 of 2048) matches the null's (555–673); math's extra structure
  lives mid-spectrum, not in the tail.
- **H2.2, memorised side: the two mem populations disagree — which is a finding.**
  Verbatim-memorized (high-entropy strings) has weak overall excess (1.3×) concentrated
  enormously at the *earliest* leaves (blk1–2.attn.in ≈ 1400: lexically-anomalous tokens
  encoded distinctly at input-adjacent layers — surface, not mechanism). Quotes — memorized
  but *coherent* — have the **largest excess of any population** (2.3× / 4.2×). "Memorised"
  is not one geometry: spurious-string memorization adds almost nothing beyond lexical
  novelty; famous-text memorization is heavily structured.
- **Depth profile splits by family once more**: pythia's math excess sits at the *late*
  stream leaves (blk14–15, before_final_norm), OLMo's at the *earliest* (blk0–1).
- **Split-half coherence (the sharp form of H2.2, added after review):** do disjoint halves
  of a task share the SAME excess-over-G directions? Measured as raw-space subspace overlap
  between the halves' top-k excess directions, each half whitened by an *independent* G
  sample (four G samples per geometry; the null = two more independent G pairs). Null: 0.005–
  0.10 (≈ chance). Results (k=8/32): **math 0.68–0.83** (both families), **quotes 0.53–0.62**,
  **memorized 0.45–0.51**. So the strong form of the memorised prediction ("no shared
  memorization direction — items are stored independently") is **refuted**: even disjoint
  halves of the verbatim-string set share ~half their excess subspace (plausibly the lexical
  geometry of high-entropy text — consistent with its early-layer excess). The *graded*
  form survives cleanly: **math's shared structure > quotes' > memorized's**, in both
  families where measurable, with math's coherence strongest (0.83 at OLMo).
  **Depth-resolved** (analysis/rq2_results.ipynb, data/results/rq2_splithalf.npy), the
  strong-form prediction partially returns: memorized coherence *rises monotonically with
  depth* (0.16 at blk0 → 0.64 at the final stream) and is LOWEST exactly where memorized
  excess mass is biggest (the early blocks) — at the storage layers, the halves do NOT share
  directions (0.16–0.4 vs math's 0.6–0.7 there); the shared memorized component is an
  output-side phenomenon (a common high-entropy-text prediction mode, not shared storage).
  Math's coherence depth-shape splits by family like its excess does (pythia mid/late
  plateau ~0.72; OLMo strongest early, 0.88–0.94). Nulls are ≤0.12 at every depth.
  The methodological findings this measurement forced (shared-whitener inflation of the
  null; cross-whitener comparisons must return to raw activation space): splithalf_note.md.
- **Packed memorized populations (added later, N-matched nulls):** the Merullo et al OLMo-2
  memorized set (650 seqs, packed all-token, ~73k tokens) shows **excess ≈ 954 vs its
  N-matched null of 61 (~16×)**, concentrated at MID blocks (blk8–10.attn.in) with
  tail_centroid 878 vs null 674 — the first population whose excess skews *tailward* of the
  null. The pythia verbatim-string set in the same packed geometry: 637 vs its 1M-token null
  282 (~2.3×, vs 1.3× in padded/last — the packed all-token view exposes more of its lexical
  structure). Caveat: the two mem sets differ in *kind* (Merullo's = memorized natural text;
  EleutherAI's = high-entropy strings), so the 16× vs 2.3× gap confounds family with
  population content. Also noteworthy: the null excess is content-difference-dominated rather
  than sampling-noise-dominated (the 73k null ≈ the 2M null), so N-matching mattered less
  than feared here — but it's what makes the comparison defensible.
- **H2.1: preliminarily REFUTED, with an inversion.** Unwhitened acts↔grads coupling
  (gen-spectrum RankMe; higher = grads more proportional to acts): memorized 177 >
  math-answers 123 > G 104 > quotes 92 (pythia, padded); math-web is slightly *below* G in
  packed for both families. Memorized data shows the *most* coupled gradients — opposite to
  the storage-in-any-free-direction intuition; a plausible reading is that well-memorized
  items have small gradients confined to already-represented directions.
  **G-whitened variant (the design's "prior step") — run; H2.1 refuted in both variants.**
  Whitening task acts by G-acts and task grads by G-grads, then taking the gen spectrum: a
  no-excess population (G′) sets the ceiling (~630 padded / ~1960 packed — both whitened
  matrices ≈ identity, trivially proportional), every real task falls below it, and **math
  falls furthest** (answers 103 vs quotes 291 vs memorized 183, pythia padded; math-web
  1625 vs ceiling 1960 packed). The scalar conflates "amount of genuine excess" with
  "acts-excess vs grads-excess mismatch", so its cleanest reading is as independent
  gradient-side corroboration of H2.2's ordering (math has the most real structure) rather
  than a coupling verdict. A separation-clean H2.1 test would compare the top-k excess-acts
  and excess-grads subspaces directly (raw-space overlap, as in the split-half machinery) —
  noted as future work.

## 5. Caveats

- Ledger exactness holds for the λ-entropy RankMe (`matrix_entropy`); `true_rankme` tracks it
  but the identity is not exact there. Both stored.
- Packed rows are within-sequence correlated (standard for this repo's packed runs); padded
  runs at N/d ≈ 4–8 carry Marchenko–Pastur broadening in the deep tail (trends valid at fixed
  N; deep-band absolute levels not comparable across different-N runs).
- Head/bulk cut (k = 32) and band edges are conventions; the front picture is robust to them
  (checked at k ∈ {1,2,8,32,128,512}), and the deep-band claims rest on 512+, where the cut
  choice is not load-bearing.
- **Unexplained: the origin of OLMo's interference mode.** The toy grid shows write-norm
  removes Pythia's mechanism but nothing in the toy (any norm, any wiring, 24 runs) produces
  interference-carried compression — OLMo's mode needs an ingredient the toy lacks
  (attention / token structure are the candidates). Known gap, deliberately not pursued yet.
- The rogue-write token attribution is final-checkpoint; the direction's token identity across
  training is unmeasured (samples aren't persisted historically).
- GSM8K's train split is small (~7.5k rows); the answer-slot population is correspondingly
  lower-N than the others.

## 6. Open questions

What we do not know (statements of ignorance, not plans — plans live in docs/plan.md):

- **The origin of OLMo-2's interference mode.** No toy knob (norm placement × wiring, 24
  runs) produces interference-carried compression; the responsible ingredient is missing
  from the toy (attention / token structure are the candidates).
- **How rogue-direction variance reaches padded last tokens — now narrowed, not closed.**
  Not slot inclusion (only 5/2,331 newline-bearing last tokens are their document's first
  newline) and not carried along v₁ at the final stream. The mid-stack run (Jul 13) shows
  ordinary tokens DO carry the direction mid-stack (~27% of centered row norm at blk4,
  ~3.6% residue at the final stream); what remains unchecked is whether that residue
  quantitatively accounts for the padded last-token measurement.
- **Whether QK-norm alone suppresses the rogue mechanism.** The sink literature finds
  QK-norm the stronger lever at 7B scale; our toy grid only tested write-norm, and OLMo-2
  has both.
- **Whether the norm→mechanism link is a cross-family regularity** or a three-family
  correlation plus one confirmed prediction.

## Appendix: side findings (real results, not load-bearing for RQ1–RQ3)

- **"Mean anomalies" — ⚠️ FLAGGED, NO VERIFIED FINDING, DO NOT CITE (Jul 13).** The claim
  formerly here ("mean anomalies split by family: Pythia's rogue variance-only, OLMo's late
  writes mean-heavy 0.15–0.6") is retracted as a misinterpretation: the showcase §8 panels
  contradict its cross-model form (OLMo-1B's late writes have LOWER mean_frac — the
  write's constant per-token offset ‖μ‖² relative to its variance trace — than pythia's
  ordinary writes; the 0.15–0.6 range was OLMo-7B alone), and the "mean anomaly"
  terminology drifted between analyses. Everything mean_frac/mean_migration-related is
  unverified until re-derived from scratch.
- **Drift — ⚠️ NOT re-verified under the gap-corrected metric.** The original claim (raw
  consecutive-checkpoint CKA: violent early reorganization, second dip at the RankMe peak,
  then monotone stabilization) conflates checkpoint spacing with drift speed. A first look
  at the corrected rate ((1−CKA)/Δ, `cka_drift_rate`) does NOT reproduce it: pythia-1b's
  largest per-dex rates sit mid-training (blk8 ≈ steps 14k–70k) and OLMo-1B's rates rise
  toward the END of training rather than stabilizing. Needs a proper pass (incl. the
  per-dex vs per-gtok normalization choice) before the claim is used anywhere.
- **mlp×attn coupling**: near-diagonal negative signed traces show MLPs partially consuming
  their neighbouring attention outputs (the GELU-4L "memory management" pattern) — showcase
  §8; independent of the family mechanisms.
- **Data-mode robustness**: every family-level claim survived packed→padded; the only
  geometry-sensitive quantity found is the rogue write's *strength* (weaker on last tokens).
- **Rogue follow-up detail (Jul 13, oneoff_scripts/rogue_id_followups.py, full numbers in
  dig_findings "Sink-slot follow-ups"):** EOT-twin — with `<|endoftext|>` prepended, the
  first newline still spikes (2178 vs 2132; the EOT takes the position-0 slot at 1583 vs
  1657). Mid-stack depth profile — sink rows ≈99% along v₁ from blk4 through blk12, halved
  at blk15, ≈0.17–0.20 at before_final_norm; ordinary tokens carry ~27% of centered row
  norm along v₁ at blk4, decaying to ~3.6% at the final stream.
