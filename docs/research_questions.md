# Research questions — registry

This doc is the question registry: every RQ and hypothesis, its one-line verdict, and where
the evidence lives. No narrative — evidence is in the topic docs, synthesis in
final_report.md, figures in analysis/showcase.ipynb. Priority: RQ1 > RQ1b > RQ1c > RQ3 > RQ2
(RQ1c is the active open front; RQ2 parked).

Notation: $S(\cdot)$ = spectral entropy of the trace-normalized **centered** covariance;
$\mathrm{RankMe} = e^S$ (the $\lambda$-entropy variant, for which the decomposition is
exact). The decomposition (repo name: the rank ledger), per block $k$:
$\Delta S_k = \chi_k + \mathrm{quality}_k + I_k$ (overlap + block-intrinsic entropy +
interference), telescoping to $\log \mathrm{RankMe}(\text{final}) -
\log \mathrm{RankMe}(\text{emb})$. Derivation and symbols: ledger.md. Phases (warmup →
entropy-seeking → compression) are training-time phases; depth is the axis each phase is
carried along.

---

## RQ1 (main): Through what layerwise mechanism does the entropy-seeking → compression-seeking rank trajectory arise in residual-stream transformers?

**Status: answered.** Emergent, with the mechanism split by architecture (norm placement):
Pythia/nanochat compress via the quality term of a sink write; OLMo-2 via interference from
distributed aligned late writes. Synthesis: final_report §0/§2; evidence: dig_findings.md;
figures: showcase §3–§5.

Sub-questions and hypotheses:

- **H1.0 — independent vs emergent.** *Emergent in both families* — no per-write spectrum
  turns over in phase; the trajectory is carried by relational terms. But "emergent"
  splits: Pythia = one write + ensemble cancellation; OLMo-2 = ensemble alignment.
  [showcase §3]

- **H1.1 — transition mechanism (original: entropy-seeking χ-driven, compression
  I-driven).** *Refuted as stated*: χ is stable through both phases everywhere. Revised and
  confirmed: compression = concentration terms — quality (Pythia/nanochat) or interference
  (OLMo-2). [showcase §3; dig_findings headline table]

- **H1.2 — carried by the late layers.** *Holds for OLMo-2, refuted for Pythia*: Pythia's carrier is an
  early-mid sink write (blk3 at 1b/6.9b; blk5 at 410m; split early+final at 160m/nanochat)
  with late-block cancellation — same blocks, opposite role. The family invariant is the
  carrier TERM, not a block index. [showcase §3 per-block quality + ΔS figures]

- **H1.3 — toy-like last write (selection bias).** *Holds for Pythia* (last-write head-mass
  0.26 → 0.92), weaker for OLMo-2 (0.63, falling). The toy correspondence is itself
  family-dependent. [dig_findings]

- **H1.4 — write-norm placement selects the concentration mode.** *CONFIRMED, all three
  deciders*: toy write-norm suppresses the sink-write mechanism (6/6 seeds, both wirings);
  the dataset swap rules data out; and the pre-registered nanochat prediction hit (no
  write-norm ⇒ full Pythia signature despite OLMo wiring). [showcase §6.4;
  dig_findings.md (dataset-swap controls section); final_report §0.3]

- **H1.5 — the sink direction is functionally inert at the output.** *REFUTED*: projecting
  it out costs +1.04 nats, diffusely; with the protocol aligned to Sun et al, their bias
  account holds at newline positions but the within-class spike variance is read by the
  NEXT position (+1.15) — a functional role their account does not describe. LayerNorm
  perturbation remains a recorded confound. [dig_findings "interventions"]

Open (RQ1): why layer output representations align at warmup (weights + theory unchecked);
the origin of OLMo-2's interference mode (no toy realisation — see RQ3); whether QK-norm
alone suppresses the mechanism.

---

## RQ1b (late discovery): Do compressions and expansions operate on the same parts of the spectrum?

**Status: answered — no.** The measured compression is a top-of-spectrum concentration
front (top ~10–30 eigendirections) that reaches ranks 32–128 late, 128–512 barely, and
never the deep band, which flattens (entropy-seeks) through all of training. RankMe is
head-biased. The Li et al selection-bias theory only explains growth proportional to
eigenvalue size, so more theory is needed for the band structure (tilt-baseline contrast).
[showcase §2; dig_findings head/bulk + windowed-alpha sections]

---

## RQ1c (open): What determines the final layers' entropy contributions — fixed layer roles, or conforming the stream to the unembedding?

**Status: open — leading interpretation documented, deciders planned.** The interpretation
(final_report §2.3): a layer's entropy effect is conditional on the stream it receives and
on what the output must become — the final layers move the representation into
unembedding-suitable form, conforming to expected output entropy. *Evidence so far*: the
packed↔padded sign flip of the final layers' per-block ΔS (showcase §3 per-block ΔS pair;
dig_findings padded-vs-packed). Everything attributable today is compositional (the exact
decomposition); the sub-questions below add the counterfactual and unembedding-side tests.

### RQ1c.0 — Is a block's entropy-change sign a fixed property of the block?

*Answered — no*: pythia's final layers add entropy in packed geometry and remove it in
padded last-token geometry. The founding evidence for this RQ. (Whether the change is
depth-disproportionate at all is H1.2's question, same figures.)
[showcase §3 per-block ΔS pair, packed + padded]

### RQ1c.1 — How much did a write matter to the final representation, holding everything else fixed?

(leave-one-out: S(final − c) − S(final)). Separates "wrote a lot" from
"mattered at the end", which the trace-share metrics conflate, without rerunning
inference — the in-between rung of counterfactualness. *In flight*: the single-write
version (`ablation_contribution`) already exists in the past sweeps; the group version
(`loo` metric: blocks, L/4 chunks, middle half) is collected for all 7 models. Blind
dig: showcase_appendix §G, pending cross-check [plan.md item 2a].

### RQ1c.2 — Does removing a layer remove its attributed effect, or does the network compensate?

(true ablation: zero-ablated inference runs, full metrics incl. the
decomposition on ablated runs). The only experiment that licenses causal language, and
compensation-vs-disappearance directly discriminates fixed roles from conditional
behaviour. *Collected*: the ablation fleet (quarter-chunks + middle half + carriers)
landed for all 7 models with full metrics. Blind dig: showcase_appendix §H, pending
cross-check [plan.md item 2b].

### RQ1c.3 — Does the unembedding's geometry explain what the final stream must become?

#### exp1. Unembedding SVD properties.

In the base toy model, The classifier heads' rows perfectly resemble the input space classes. And thus the rankme of its square singular values seems to imitate the rankme of $f_\theta(x)$. Since their findings hint at the bottleneck creating the compression, we can theorize that perhaps the final layers are handling the transition into a space suitable for the output embedding. Ideally, we can find a way to probe what the rankme curve position will be based on the embedding.

- Naïve attempt (just looking at embedding svd rankme) failed. Fingings include:

  - seemingly inconsistent initialisations causing very different curves (olmo 7b: low -> high -> low -> high -> low; nanochat-d12: low -> monotonic increase to high. olmo 7b and nanochat d12 are the only ones starting with low rank. pythia 6.9b is different from the rest.)

  - Timings don't match consistently in general.

  - We should get a basic understanding of what these *do* come from.

  - Recommend running on padded / last token only.

  - Maybe centering is also problematic (mean). Not sure. But it doesn't seem like that's it when in general centering does affect rankme a lot but regardless of centering or not, both centered and uncentered tend to show a curve on their own, so its unlikely to be this I think. (reason about this)

#### exp2. Representation space alignment with the unembedding row space.
Li et al. show this alignment visually, but we need to find a way to confirm that using a metric that can be generalised to many dimensions.

- Based on previous point (exp1 naive attempt), it seems this is unlikely. Instead, I want to know: "why does it match in the toy model?". I suppose it's because if they match, features are at their peak when in the same direction. So essentially the dot product for that row is maximised when the direction aligns -> highest logit for that row.
  - (relevant back to the previous point) Maybe class frequency is the thing interfering with our comparison. Would that distort the eigenvalue space relative to the inputs? (reason about this). If this is the case, we can obtain rough class frequency by just looking at each training set's tokens. It's possible this isn't it. I can also imagine that maybe there's just a massive singular direction that's just kind of orthogonal to the actual tokens that pass through, this doesn't match the toy model though. Another thing is: I've heard that word embedding spaces are actually cone shaped, rather than evenly spread (is this true?). If true, then how would that change our path to finding a mechanism?

---

## RQ2: Does maths/numerical data have more shared geometric structure than quotes/memorised data?

**Status: PARKED (Jul 13) — do not cite.** The documented excess-mass results mixed acts
and grads quantities; ref pairing unverified; some comparisons straddled incompatible
geometries. Redo constraints: plan.md item 10; diagnosis: final_report §4; methods (still
valid): rq2.md.

- **H2.1 — grads↔acts coupling: math > memorised.** *Refuted in all three variants*
  (including the clean subspace test: excess-acts vs excess-grads overlap ≈ chance for
  every population). This negative stands independently of the contamination.

- **H2.2 — math's excess structure lives in G's tail.** *Under revision* — original
  verdict (half-supported: real math excess, mid-spectrum not tail; mem populations split)
  awaits the acts/grads-separated redo.

---

## RQ3: What is the minimal model that shows the trajectory — and does it share the transformer's decomposition signature?

**Status: answered, with named open items.** Evidence: toy.md; figures: showcase §6,
analysis/toy_multilayer.ipynb, showcase_appendix §B–C.

- **H3.1 — single-layer replication.** *Confirmed*, incl. panels B/C — but requires the
  paper's CONSTRUCTED init (an unstated fourth condition; read off their figure's t=0
  markers). Mechanism of the decline: covariance-cancellation collapse at the rare-pair
  fork; the decline is a transient. [toy.md §3; showcase §6.1]

- **Fork→decline lag.** *Answered*: set by the rare pair's starting offset from the origin
  (zero lag at the origin — the paper's geometry; grows with offset).
  [toy.md §3.3; showcase_appendix §B2]

- **H3.2 — deep toys and the decomposition signature.** *Resolved with the inverse of the
  anticipated split*: signature-without-phases. The deep residual toy reproduces Pythia's
  quality-carried signature (6/6 no-norm seeds) while its 32-class task can never show
  the pattern (task structure). Deep stacks DO show the pattern iff (T) the fork
  lands after saturation, (K) the kick is large enough, (I) near-identity transmission —
  and the residual is NOT constitutive (identity-init plain networks show the full pattern 3/3; the
  residual provides (I) by default). [toy.md §4; showcase §6.2–6.3]

- **H3.3 — representativeness guard.** *Worked as designed* (flagged the linear variant,
  passed the nonlinear base and all grid cells).

- **Architecture knobs.** Write-norm suppresses the sink-write mechanism completely (6/6,
  both wirings); no knob produces OLMo's interference mode; wiring selects nothing. ⚠️ The
  grid predates the pattern conditions and runs on the no-pattern task — trajectory-shape
  readings void pending the rerun. [showcase §6.4; toy.md §6]

Open (RQ3): the tanh-block pre-fork compression (a second, unexplained mechanism, 3/3
seeds); kick growth with depth and earlier forks at depth (observed, not modeled); the
depth-0 → depth-1 jump on the 32-class task; activation function never ablated; single lr
per task; layer-scale threshold only bracketed (0.1–0.3); OLMo's interference mode has no
toy realisation (the program's largest missing piece).
