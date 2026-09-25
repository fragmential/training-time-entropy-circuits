# Unembedding RankMe corrections + RQ1c.3 reasoning (chat transcript)

Verbatim transcript of two assistant responses (Jul 21, 2026 session). Context: the first
answers "could you recover what the 160m head *would* have as a RankMe by removing the
rank-1 mean it's absorbing, and would that make the unembedding matrices comparable?" plus a
requested evaluation of the RQ1c.3 draft; the second is the claim-by-claim reasoning pass
over the same RQ1c.3 text. Corrections to these responses live in the chat that follows
them, not here.

---

## Response 1

Both threads are complete. Here's everything.

### 1. Yes — and it makes the curves dramatically more comparable

Two versions, both computable from the already-stored head SVDs (`unembedding_svd`) + stream
means (`final_stream_svd`):

- **Mean-deflated**: project the stream-mean direction out of the head, W(I − μ̂μ̂ᵀ), then
  RankMe. The principled "remove the absorbed bias" correction.

- **Drop-top-1** (tail RankMe of sv[1:]): cheap, data-free, removes whatever single direction
  dominates.

Final-checkpoint numbers:

| model | raw | drop-top1 | mean-deflated | cos(v₀, μ̂) |
|---|---|---|---|---|
| pythia-160m | 2.3 | 463 | 373 | **0.996** |
| pythia-410m | 668 | 945 | 679 | 0.19 |
| pythia-1b | 1341 | 1877 | 1343 | 0.05 |
| pythia-6.9b | 2897 | 3652 | 2918 | 0.16 |
| OLMo-2-1B | 120 | 1911 | 213 | 0.53 |
| OLMo-2-7B | 187 | **3708** | 239 | 0.34 |
| nanochat-d12 | 627 | 721 | 638 | 0.30 |

Three distinct regimes fall out. **160m** is the only model whose dominant direction *is* the
stream-mean reader (cos 0.996) — deflation recovers 2.3 → 373, most of the way back to its
pre-collapse plateau (~560); the residual gap means a second direction also grew late. **The
other Pythias + nanochat**: their top direction is *not* the stream mean (cos 0.05–0.30), so
mean-deflation barely moves them — their giant direction is something else (plausibly the
token-frequency direction, whose token-space side is fixed but whose stream-space side
needn't equal μ). **OLMo (both)**: the raw "collapse to 120/187" is almost entirely one
colossal direction sitting on a *nearly untouched flat bulk* — drop-top1 gives 1911/3708
(≈ 0.9·d, close to init flatness), and the tail trajectory is a clean monotone rise
373 → 3708. So my earlier "long-run collapse to 187" for OLMo-7B was itself a top-1
artifact — worth knowing before trusting any raw head-RankMe curve. Recommendation: plot
raw, drop-top1, and mean-deflated side by side; drop-top1 is the best single cross-model
comparator.

### 2. Evaluation of RQ1c.3

**What's sound.** The exp2 explanation of why the toy aligns (CE pushes head rows toward
class features because that maximizes the correct logit) is correct — it's the standard
argument, neural-collapse-adjacent, and it correctly predicts where the LLM case should
*break*: the toy has balanced classes; tokens are Zipfian. Three of your hunches were right
and are now confirmed: (i) "we should understand what these init curves come from" — done,
`docs/model_init_details.md`; (ii) "maybe class frequency is interfering" — yes, this is the
first-order confound: every model's raw head RankMe is dominated by one or two
frequency/mean-linked directions (the table above); (iii) questioning centering — the
resolution is that "centering" a weight matrix is ill-defined, and the meaningful analog is
exactly the deflation above.

**What's now answered.** The "seemingly inconsistent initialisations" line dissolves
completely: OLMo-7B's low start is the fake rank-380 released init (not training dynamics);
nanochat's low start is its zero-init head; 6.9b differs because it's the one Pythia with
the σ=0.02 init (4× heavier bulk) plus a smaller LR, so learned directions never
transiently dominate. The "low→high→low→high→low" OLMo shape is three superimposed
artifacts (init recovery, spike growth, bulk reshaping) — raw curves are uninterpretable,
and "timings don't match" can't be assessed until the confounds are removed. Redo the
timing comparison on deflated/tail curves.

**What's refuted.** "Maybe there's a massive singular direction that's kind of orthogonal to
the actual tokens that pass through" — tested, and for 160m it's the *maximal opposite*:
cos(v₀, μ̂) = 0.996, capturing 95% of stream energy: opposite of orthogonal. For
410m/1b/6.9b/nanochat the top direction is NOT the stream mean (cos 0.05–0.30) — whether
it's orthogonal to the stream needs the alignment share metric (tr PΣ/trΣ, the existing
alignment script) — a concrete next measurement.

**The cone question.** Yes, real — anisotropy/"narrow cone" (Ethayarajh 2019; Gao et al.
2019 representation degeneration; Mu & Viswanath common-direction removal). The cone = a
large shared mean component + few dominant directions — precisely the frequency/mean
structure above. Consequence for the mechanism path: compare CENTERED stream geometry with
the DEFLATED head subspace; treat the mean/frequency directions as their own object (they're
already the project's sink/rogue-write family) rather than noise.

**One correction to the framing.** "Recommend running on padded / last token" belongs to
exp2, not exp1 — the head SVD is data-independent; only the alignment side has a
token-population choice, and there last-token/padded is the right call.

---

## Response 2

Understood — here's the actual reasoning pass over what you wrote, claim by claim.

### exp1

**"The classifier head's rows perfectly resemble the input space classes."** Checks out with
one caveat that turns out to matter a lot downstream: the CE fixed point aligns rows with
*centered* class means (each row is pulled toward its class's features and pushed away from
the softmax-weighted average of everything), so what the head mirrors is the between-class
geometry, not the raw feature geometry. In the balanced toy the distinction is invisible.
In the LLM it's the whole story — the "class average" component that the toy quietly
discards is the frequency/mean structure that dominates every real head spectrum.

**"The rankme of its squared singular values seems to imitate the rankme of f(x)."** This
follows *if* rows ≈ class means: then the head Gram equals the class-mean second moment, so
head-sv² RankMe = RankMe of the between-class covariance. That imitates RankMe(f(x)) only
when within-class scatter is a small share of total variance — true in the toy's
late-training collapse regime, false in an LLM, where contextual (within-token-class)
variance is most of the stream. So the imitation isn't a general law; it's a diagnostic of
the collapse regime. That's actually useful: *where* head RankMe and stream RankMe converge
is itself a signal.

**"We can theorize the final layers are handling the transition into a space suitable for
the output embedding."** The theorizing step is the weak joint: head↔stream resemblance is
symmetric evidence. Both objects are trained jointly, so co-adaptation guarantees
correlation without telling you which side conforms to which. To get direction you need an
asymmetry: timing (but see below — raw timing is currently unmeasurable), or an
intervention. One free quasi-experiment is already in your stable: nanochat trains its
unembedding with a 5× *smaller* LR than the matrices (0.004 vs 0.02) and the embedding 10×
larger — if the stream conforms to the head, a deliberately sluggish head should visibly
delay the final layers' entropy moves relative to the Pythia/OLMo schedules.

**"Naïve attempt failed… inconsistent initialisations."** Correct observation, and now fully
explained rather than mysterious — OLMo-7B's low start is a released-init artifact (rank
~380 masquerading as trunc-normal), nanochat's is a zero head, 6.9b's odd shape is the
σ=0.02 init bulk plus lower LR. Nothing about those curves was telling you about learned
geometry.

**"Timings don't match consistently in general."** This conclusion doesn't survive the
confounds. The raw curves' apparent timing is set by *when a spike overtakes the init
bulk*, which depends on init scale and LR — not on when the learned geometry actually
forms. Example: 6.9b's "missing" dip is purely a bulk-size effect. The claim needs
re-testing on drop-top1 / mean-deflated curves; on those, all seven models already look far
more congruent (early dip, monotone recovery), so my expectation is the timing mismatch
partly evaporates.

**"Maybe centering is also problematic… but both centered and uncentered show a curve on
their own, so it's unlikely to be this."** The instinct was right; the exonerating argument
is not. "Both variants show a curve" doesn't clear the confound — an artifact direction
produces structured curves in both. And empirically the head-side analog of centering
changes conclusions by an order of magnitude (OLMo-7B final: 187 raw vs 3708 with the top
direction removed). One conceptual untangling: for a weight matrix there are two different
"centerings" — subtracting the mean *row* (a token-space operation, removes the average
logit vector / the cone axis of the head) and deflating the stream-mean direction (a
stream-space operation, removes what responds to the constant offset). They coincide for
160m, not in general; both are worth plotting.

### exp2

**"Based on exp1, it seems this is unlikely."** Non sequitur — exp1 failed because of
confounds, which says nothing about whether stream↔head alignment holds. Alignment can be
strong while raw RankMe curves look unrelated (they measure different things: subspace
overlap vs spectral flatness). Don't discard exp2 on exp1's evidence; the alignment-share
metric (tr(P_kΣ)/tr Σ vs chance k/d — your own crashed job) is the direct test.

**Your explanation of why the toy aligns** (dot product maximized when directions align →
highest logit) — correct, with the same refinement as above: the pull is contrastive, so
alignment is with centered/contrast directions, and the clean symmetric geometry
additionally needs balanced classes. Both caveats predict exactly the LLM deviations you
then worried about.

**"Maybe class frequency is the thing interfering… would that distort the eigenvalue space
relative to the inputs?"** Yes, and via two separable mechanisms worth keeping distinct:
(a) row norms scale with token frequency, which reweights the head spectrum relative to the
stream's; (b) a global frequency/prior structure emerges as one or two colossal directions —
the measured spikes. Your proposed test (grab unigram counts from the training tokens) is
actionable and I'd sharpen it to a prediction: the top direction's token-space side should
correlate strongly with log unigram frequency. If it does, deflating it and residualizing
row norms by frequency should bring the toy correspondence back within reach.

**"Maybe there's a massive singular direction that's just kind of orthogonal to the actual
tokens that pass through."** Refuted where it's testable so far: in 160m the massive
direction is maximally *non*-orthogonal to the stream (cos 0.996 with the mean, 95% of
stream energy). For the others the top direction isn't the stream mean, so
orthogonality-to-the-data is still open — but it's exactly what the alignment share
measures, so this resolves itself when that job reruns. Your own aside — "this doesn't
match the toy model though" — is correct reasoning: the toy *cannot* produce such a
direction (balanced classes, no frequency skew), which is another vote for frequency as the
source.

**"Word embedding spaces are cone shaped — is this true?"** True and well-established
(anisotropy / narrow cone: Ethayarajh 2019; representation degeneration: Gao et al. 2019;
common-component removal: Mu & Viswanath 2017). The cone is the shared mean/frequency
component viewed from a different angle — the same object as your spikes. Implication for
the path: don't treat cone-correction as preprocessing noise removal; the cone/frequency
directions are a *finding* in this project (they're the sink/rogue-write family on the
stream side), and the clean comparison is centered stream geometry vs deflated head
subspace, with the removed directions studied separately.

**On the parent question.** One structural note: RQ1c asks about *layers* (fixed roles vs
conforming), but everything in RQ1c.3 measures the *head*. Head-side geometry alone can
never discriminate the two — it needs to be joined either to timing (deconfounded, per
above) or to your layer interventions (the LOO/ablation machinery in G/H), e.g. does
ablating the late quarter specifically destroy stream-head alignment while leaving centered
stream RankMe comparatively intact?
