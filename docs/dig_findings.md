# Dig findings — block_representations_samples (packed), first pass

Models: pythia-1b-deduped (58 ckpts), pythia-6.9b-deduped (58), OLMo-2-1124-7B (66),
OLMo-2-0425-1B (62, drift included) — all packed legs complete. All numbers are centered,
λ-entropy ledger terms summed over blocks unless stated.

## Headline: compression is concentration of the write ensemble — universal law, family-dependent localization

Universal across all models measured: the compression phase is never a per-write phase
turnover — Σχ stays comparatively stable, per-write spectra don't turn over in phase, and the
decline is always carried by the two *concentration* terms (quality + interference), i.e. write
energy piling into few shared directions. A rank-one write and several mutually-aligned writes
are the same geometry, split into different ledger bins; what differs per family is only which
bin — *where the concentration lives*:

| | pythia-1b | pythia-6.9b | OLMo-2-1B | OLMo-2-7B |
|---|---|---|---|---|
| RankMe (centered, before_final_norm) | 465 → peak 555@4k → 194 | 824 → peak ~546@13k → 260 | peak 663@5k → 314@320k → 447 | 74@150 → peak 939@4k → 646 |
| Σ quality over training | −0.5 → **−6.3** | −0.4 → **−5.0** | dips −3.9@6k → **recovers to −1.0** | dips −5.5@10k → **recovers to −1.3** |
| Σ interference over training | −3.5 → **+2.3** (crosses 0 ≈ RankMe peak) | −4.7 → −0.2 (rising, no cross) | −0.5@6k → **−4.1** (falling) | −0.1@6k → **−5.5** (falling) |
| compression driver | quality | quality | **interference** | **interference** |
| locus | blk3 (90% of Δquality) + blk15 | blk3 (~all of Δquality) + blk31 | blk14–15, aligned (+0.32) | late blocks 29–31, aligned (+0.39) |

**2+2 confirmed** (OLMo-1B, completed after the first pass): no rogue write (all write RankMe
1150–1420; late blocks ~350 at worst), interference-driven, late-locus with mutually aligned
final writes, and the same late re-entropy rise (S_final 5.75@320k → 6.08@1.8M). The
family split is now consistent at both scales of both families. OLMo-1B's drift metrics add:
mean CKA-drift starts at **0.12** (vs pythia-1b's mildest-point 0.82 — a far more violent early
reorganization), recovers to 0.93@10k, dips again ~0.79 at the RankMe peak (a second
reorganization), then climbs monotonically to 0.935.

**Pythia: compression = one early rogue write.** In BOTH pythia models, blk3.mlp's write
starts healthy (RankMe ≈ 470 / 830, trace ≈ 0.27× the stream) and collapses to rank ≈ 1–2
*exactly at the RankMe peak* (1b: 46→2.5 over steps 2k→5k, peak 4k; 6.9b: 12→2 over 10k→26k,
peak 13k), then grows in energy for the rest of training (trace ratio → 22.6× / 3.7× the
incoming stream). Same block index at both scales (16 vs 32 blocks) — absolute depth, not
relative. This looks like the massive-activations / rogue-dimension phenomenon, now with a
ledger price: it carries essentially the whole quality collapse.

**Pythia's late blocks cancel it.** The signed trace between blk3.mlp.out and the *last*
block's write trends 0 → **−0.65 in both models** — the final blocks increasingly write
against the rogue direction (the late positive interference, blk14/15: +1.9/+1.3; blk31:
+1.7, is rank *restoration*). The last write is also strongly head-targeted by the end
(eigendirection head-mass of blk31.mlp: 0.26 → 0.92 in 6.9b).

**OLMo-2-7B: no rogue write, milder compression, opposite attribution.** Every write keeps
RankMe 1000–2600 (only blks 29–31 lower, ~200–550); quality *recovers* over late training
while interference decays steadily to −5.5, concentrated in the last three blocks (I ≈ −0.2…
−0.5) whose adjacent writes are mutually *aligned* (blk30~blk31 signed trace +0.39, vs
Pythia's anti-aligned cancellation). Final compression is much shallower (646 vs 194/260).
Late rows also show S_final rising again (6.23 → 6.47 over the last ~15% of training) — the
re-emerging entropy phase Li et al note for OLMo-2-7B.

Architectural reading (speculative, testable): OLMo-2's norm reordering + QK-norm were
introduced to suppress outlier activations; the Pythia rogue-write mechanism may be exactly
what that architecture removes, leaving a slower, distributed, interference-driven compression
as the "intrinsic" optimization effect. OLMo-2-0425-1B (pending) is the immediate check.

## Consequences for the hypotheses (docs/research_questions.md)

- **H1.0:** emergent in both families, but differently: in Pythia the trajectory is dominated
  by one write's relational quality term + its cancellation; in OLMo by distributed
  cross-write interference. Not per-layer-independent anywhere.
- **H1.1:** the "compression is I-driven" original holds for **OLMo**, the quality-driven
  revision holds for **Pythia** — the hypothesis needs a family split, which is itself the
  finding: *the ledger term that carries compression is architecture-dependent.*
- **H1.2:** original (late-layer locus) holds for OLMo; refuted for Pythia (blk3 early +
  final-block cancellation). Head-of-spectrum targeting confirmed for Pythia's last write.
- **H1.3:** Pythia's last write behaves toy-like (head-mass → 0.92); OLMo's less so (0.63,
  decreasing). Toy correspondence may also be family-dependent.
- **Caution for the RankMe narrative:** in Pythia, a substantial part of measured "compression"
  is one direction's energy growth rather than distributed information consolidation — worth
  re-examining how much of Li et al's curve survives removing the top-1/2 eigendirections
  (cheap: the eigenspectra are stored). The padded (last-token) twin runs will show whether
  the rogue write even appears in last-token geometry.

## Follow-up: compression is a head phenomenon — the bulk never stops expanding (universal)

RankMe of the final-stream centered spectrum with the top-$k$ directions removed:

| model | k=0 (Li et al's curve) | k=32 |
|---|---|---|
| pythia-1b | 540@6k → 190@52k → 190 | 1327@6k → **1459@31k** → 1372 (rises through the "compression phase") |
| pythia-6.9b | 494@6k → 337@52k → 255 | 1576@6k → **2328@70k** → 2209 (still rising deep into it) |
| OLMo-2-1B | 663@5k → 314@320k → 447 | 1293@5k → **1516@120k** → 1417 (−6% while k=0 halves) |
| OLMo-2-7B | 931@8k → ~510 → 646 | 2299@8k → **2502@90k** → ~2390 (plateau; no bulk compression) |

The entire measured compression is confined to roughly the top 10–30 of 2048/4096 directions;
the bulk representation continues entropy-seeking (or plateaus) throughout. So the
"phase transition" is a **head-concentration event, not a contraction of the representation** —
RankMe conflates the two, and the k-removed curves separate them. This is the strongest
universal in the data (true in both families, both scales), it is precisely where the
family-specific concentration mechanisms deposit their energy (rogue write / aligned late
writes), and it dovetails with Li et al's own eigenvector-ablation table (top-10/50 removal
barely hurts SciQ; the bulk carries the capability — and the bulk, we now see, never
compressed). Candidate headline result. Caveats: packed all-token geometry (padded twins will
check last-token), k-sweep is coarse (0/1/2/8/32), and head-vs-bulk cut should be made
principled (e.g. k at the spectral knee) before it's a claim.

## Padded-vs-packed (last-token geometry; Pythia twins complete, 16,384 rows)

- **The rogue write survives the geometry change.** pythia-1b blk3 in padded/last-token:
  write RankMe 145 → ~2, trace ratio → 3.3×, quality → −3.9, same collapse timing at the
  RankMe peak — attenuated relative to packed (1.2 / 22.6× / −6.3) but unmistakably the same
  object. pythia-6.9b is much milder in this geometry (RankMe → ~13–16, ratio 0.68): the
  rogue direction lives more strongly on non-final token positions at 6.9b. The
  anti-alignment with the final write is also far weaker here (−0.11 vs packed's −0.65).
- **Head concentration is MORE extreme in last-token geometry.** Before the final norm, the
  full centered RankMe collapses to **single digits** (9 / 8) while removing the single top
  direction restores it to ~170–240 — one direction dominates last-token pre-norm variance
  almost completely. After the final norm (Li et al's exact measurement), pythia-1b k=0 falls
  430@15k → 101 while k=32 moves 1356 → 1304 (−4%); pythia-6.9b's k=32 bulk is *still rising*
  (2146@25k → 2277@69k) deep into the nominal compression phase. Across all three geometries
  (packed all-token, padded pre-norm, padded post-norm) the structure is the same; the final
  norm attenuates the head's dominance but doesn't change it.
- **OLMo padded twins (complete): geometry-robust replicas of their packed stories.** Both
  scales: quality recovers (1B −3.8 → −0.5; 7B −6.0 → −1.0) while interference falls (→ −5.3 /
  −6.5), locus in the last 2–3 blocks with mutually *aligned* writes (+0.33 / +0.37), no rogue
  write (all write RankMe ≳ 900 except late ~200–280), same late re-entropy rise, and
  OLMo-1B's drift double-dip (0.128 bottom → 0.947 → second dip ~0.84 at the RankMe peak).
  Net: the family split (rogue-write quality-compression vs distributed
  interference-compression) is robust to the token-selection geometry; only the Pythia rogue
  write's *strength* is geometry-sensitive (weaker on last tokens, esp. at 6.9b).

## The rogue direction identified: it's the newline direction

Raw-sample collection at the final checkpoint (`configs/block_rogue_id.yaml`; rows mapped back
to mix tokens) settles the token-identity question: projecting blk3.mlp.out's samples onto its
top centered eigendirection, the spiking tokens are **newlines** — `\n`/`\n\n` variants are 38
of the top-40 rows in BOTH pythias, the top eigenvalue carries 98.8% / 94.3% of the write's
variance, and the top 0.1% of tokens carry ~33% / 31% of the direction's variance. Combined
with mean_frac ≈ 0.003 (variance-like, not bias-like): the write is a sparse newline-activated direction —
enormous on line/paragraph boundaries, near-zero elsewhere.

So Pythia's compression phase, stated plainly: **from the RankMe peak onward, blk3's MLP
writes an ever-larger newline-activated direction**; the direction's energy (→22× the stream) crushes
the ledger's quality term and with it the measured RankMe of any all-token covariance; late
blocks partially cancel it before the output (signed trace −0.65); and last-token geometry
sees it attenuated but present — 13.9% of padded last tokens are newline-bearing (trailing
newlines + truncation boundaries). This connects to the massive-
activations / delimiter-token literature — the new pieces here are the training-time onset
(exactly at the RankMe peak), the exact rank-cost accounting, the late-layer cancellation, and
the OLMo-2 contrast (whose architecture shows no such direction and compresses via aligned
late writes instead).

## OLMo's late writes, token-attributed: the anti-Pythia

The same raw-sample attribution applied to OLMo-7B's late writes (blk29–31.mlp.out,
`block_write_id_olmo7b`) finds **no token-concentrated structure**: top eigenvalue share ~10% (vs Pythia's
94–99%), top 0.1% of tokens carry <1% of the direction's variance (vs ~33%), peak scores ±20
(vs ±2200). The content is diffuse and weakly thematic — blk31's top direction leans on
dates/timestamps/forum-metadata tokens ("Posted 2/", "2007-06-", "share|"), blk30's on
math/LaTeX and punctuation — while the writes' top directions are moderately *shared across
blocks* (cosines 0.47–0.58 for 29~31, 30~31). Exactly the ledger's prediction made concrete:
OLMo concentrates by many writes reinforcing common ordinary-token subspaces; Pythia by one
write screaming on newlines. OLMo-1B replicates it (top eigval 10–11%, concentration ~1%,
blk14 weakly paragraph-flavored, cross-write top-direction cosine 14~15 = **0.75**) — the
family contrast holds 2+2 at the token level too.

### Rogue-direction addenda (systematic attribution + preimage)

All-token attribution (mean |projection| per token type, n≥20, 1526 types): newline variants
rank top with dominant counts (`\n` 99.3 at n=11,583; `\n\t`; `\n\n`) — the newline claim
survives the systematic check; a small-sample tail of word types (n≈20–26) reaches 70–95.
Preimage (regress the score on blk3.attn.in samples, read against the embedding matrix): the
best linear input predictor is flat against all token embeddings (max cos 0.09; cos with the
raw `\n` embedding −0.013) and barely separates newline rows at the input (+23.0 vs +19.3)
while the output score separates hugely (+81 vs ~10) — so **the spike is constructed by the
MLP** (nonlinear gating / processed features), not linearly echoed from the newline embedding.
Literal layer inversion is ill-posed (down-projection nullspace + nonlinearity); this
statistical preimage is the sound version of "push the direction back and unembed it".

### Rogue-direction interventions: reconciliation with the massive-activations literature

Three inference-time interventions on the direction (pythia-1b, all post-blk3 streams, 65k
packed tokens), next-token CE deltas:

| intervention | overall | predicting `\n` | right after `\n` | elsewhere |
|---|---|---|---|---|
| zero the component | +1.04 | +0.53 | +0.98 | +1.06 |
| global mean substitution | +0.74 | +0.31 | +0.64 | +0.76 |
| **token-conditional** mean substitution (class pattern kept) | +0.43 | **+0.09** | **+1.15** | +0.41 |

Reconciliation with Sun et al (2402.17762: zeroing catastrophic, mean-substitution harmless):
our global mean-substitution "divergence" was a **protocol artifact** — a global mean over all
tokens ≈ zeroing for the rare spike tokens, i.e. closer to their catastrophic condition than
their benign one (their means are computed where the activations occur). Under the aligned
token-conditional protocol, their bias account holds exactly where it should: keeping the
class pattern makes newline-position prediction nearly free (+0.09).

**The new part their account misses:** "right after a newline" gets WORSE under
token-conditional substitution (+1.15, worse than zeroing's +0.98) — the within-class
variance of the newline spike is *read by the next position*. The direction is not a constant
input-agnostic flag: its per-occurrence modulation carries information the model consumes
when predicting what follows a boundary. Also consistent with "A Refined Analysis of Massive
Activations" (2503.22329: suppression is architecture-dependent, not universally
catastrophic — matching our family split), and convergent with "Attention Sinks and
Compression Valleys are Two Sides of the Same Coin" (2510.06477, ICLR'26), which proves
massive activation ⇒ dominant singular value ⇒ entropy compression on the DEPTH axis — our
training-time account (onset at the RankMe peak, ledger cost, norm-placement causality,
newline attribution in packed streams) is the temporal counterpart of their depth-wise
result and should be cited as convergent.

### Compression valleys: the depth-axis shadow of the rogue write (OLMo-2 has none)

Depth profile of stream RankMe (centered, attn.in per block + before_final_norm, final ckpt,
packed): **pythia crashes from ~850 to ≈2 at blk4** — immediately after blk3's write enters —
stays 2–13 across the mid-stack, and recovers late (102 → 194): an extreme compression valley
(2510.06477's phenomenon) whose entry is the rogue write and whose recovery is the late-block
cancellation (their "Refine"). Same canyon at 6.9b (crash at blk4 → recover to 260) and
nanochat-d12 (137 → 4 → 283) — H1.4 on the depth axis. **OLMo-2 has essentially no valley**
(1B: 1345 → ~812 mid → 975 → 447 at output; 7B similarly smooth) — no rogue, no dominant
direction, no canyon. Novel datum: 2510.06477 did not test OLMo-2; its valley-freeness is
predicted by the write-norm story. One object, three shadows: the massive activation
(feature axis), the compression valley (depth axis), the compression phase (training axis).
On the carrier token: 2510.06477's valleys are typically BOS-driven (BOS norm spiking ~10⁴×
in layers 0–5 — our blk3 onset sits in that depth range), but their theorem is
token-identity-agnostic, and Sun et al already list delimiters ("." and "\n") among massive-
activation carriers alongside first tokens (model-dependent; cf. secondary-sinks work,
2512.22213). Our packed streams contain NO BOS (GPT-NeoX prepends none; every position is
mid-text), so the sink/bias role necessarily lands on the delimiter class — the newline
direction is the BOS phenomenon's packed-stream form. Optional future check: prepend a
first-token/BOS and watch the spike partially migrate to position 0.

**Valley emergence timing:** valleys are not present early — they EMERGE at/just after the
spectrum-entropy peak, synchronized with the rogue write's rank collapse, then deepen
monotonically for the rest of training. pythia-1b: no valley at step 1000 (mid/early stream
ratio 1.26), forming by 5000 (0.11; final-RankMe peak 4000), 0.002 by the end. pythia-6.9b:
forms across 5k–14k around its ~13k peak. nanochat: peak ~256, valley crossing ratio 0.5
around step ~1300+, monotone after. OLMo-1B: no valley at ANY checkpoint (ratio 0.58–2.1
throughout; mid-stream RankMe even exceeds the early stream for much of training). Valley
emergence and the compression phase are the same temporal event. Animated:
experiments_anim.ipynb "Layerwise stream RankMe" (all 7 models incl. 160m/410m/nanochat).

## Refinement via windowed alphaReQ: a compression *gradient*, not a hard head/bulk split

Windowed alpha (Stringer-weighted log-log slope over eigenvalue ranks $[k_0, k_1)$; the stored
`alpha` uses 11–100, which overlaps the head — `alpha_window` virtual hook recomputes any
window from the stored spectra) sharpens the claim. pythia-1b packed, over the compression
phase: $\alpha_{11\text{–}100}$ +44%, $\alpha_{32\text{–}100}$ +36%, $\alpha_{32\text{–}300}$
+23%, while top-32-removed RankMe (dominated by ranks ≳300) moves −6%. So compression
**propagates down-spectrum with decaying amplitude**: strong in the head, decaying through the
upper bulk, negligible in the deep spectrum — "the bulk never stops expanding" is exact for
the deep spectrum and too strong for ranks ~32–300. Bulk alpha also resolves OLMo-7B's late
re-entropy turn ($\alpha_{32\text{–}300}$ peaks 0.92@510k, then declines) that head windows
exaggerate. Alpha is the better bulk instrument here: truncated-spectrum RankMe changes the
effective dimension (levels not commensurable; trends only), while alpha is scale-free with
the window an explicit part of the definition. Padded caveat: at $N/d \approx 4$–8,
Marchenko–Pastur broadening contaminates the deep tail — keep padded windows ≲ low hundreds.

## Most promising metrics (ranked)

1. **block_ledger** — the workhorse; every headline above is a ledger read-out.
2. **block_block_coupling.signed_trace** — found the rogue-cancellation structure
   (blk3↔last −0.65) and OLMo's late-block reinforcement; cheap to read, very interpretable.
3. **eigendirection_attribution** (head-mass summaries) — the H1.3 test; sharp signal.
4. **block_residual_coupling** (tr_P / R_over_r / cos_cr) — useful, partly redundant with 1–2.
5. **incremental_overlap (χ_k)** — stable and interpretable but low variance; mostly context.
6. **cka_drift / geneig_drift** — the early-reorganization V-shape (1b), otherwise quiet;
   keep for the 1Bs only (already the config).
7. **gen_block_vs_residual, mean_migration** — nothing notable yet; candidates to drop from
   future configs if space is needed.

Data caveats: packed all-token rows (correlated within sequence); N/d ≈ 25 for the 7Bs on some
per-write spectra; quality-carrier percentages are window-dependent (13k→143k shown). OLMo-1B
and the padded twins will settle the family story and the token-selection sensitivity.
