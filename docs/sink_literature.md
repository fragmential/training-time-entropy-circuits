# Sink literature — working summary for the rogue-write story

Three papers, read against our Pythia blk3.mlp rogue direction (docs/dig_findings.md,
docs/final_report.md §0.2–0.3). Sources under `reference/`:
- **MA** = Sun et al 2024, "Massive Activations in LLMs" (`massive-activations-arxiv-2024/main.tex`)
- **SSS** = "The Spike, the Sparse and the Sink" 2026 (`the-spike-the-sparse-and-the-sink-arxiv-2026/sections/*.tex`)
- **CV** = "Attention Sinks and Compression Valleys … Same Coin" (2510.06477, `attention-sinks-and-compression-valleys-arxiv/iclr2026_conference.tex`)

## (a) Triggers / carriers of massive activations

**MA taxonomy** (main.tex:230–244): carriers are model-dependent — (a) starting token only
(LLaMA2-13B, MPT, **GPT-2**); (b) starting token + the sequence's FIRST strong delimiter,
"." or "\n" (LLaMA2-7B, main.tex:220); (c) start + delimiters + weak-semantics words
(Mixtral, main.tex:225). Values are near-constant across inputs (main.tex:264, 279–287) and
appear abruptly after one early layer (main.tex:204–208). "Starting token" is positional, not
BOS-dependent: their main experiments run WITHOUT a BOS token, and prepending one leaves the
spike locations unchanged in LLaMA2 (main.tex:788–790). Rationalization: the start token is
in every forward pass; delimiters are semantically cheap storage (main.tex:326–331).

**SSS sharpens this to position-vs-identity** (3_1:194): >98% of the entire vocabulary spikes
when placed at position 0 and rarely elsewhere — "driven by architectural position rather
than token semantics". Mechanism: the first token's attention collapses to a static linear
map (it attends only to itself, 3_1:222–235), reliably steering it into the trigger direction
s\* of a few high-gain quadratic forms in an early MLP ("directional quadratic amplifier",
3_1:119–200; step-up/step-down block indices, 3_1:74–78). Delimiters get there by **self-sinking**
(3_1:237): elevated post-RMSNorm magnitude → heads attend to the token itself → it emulates
the first token's isolated environment and lands on the same s\*.

**Map to our finding.** Our position-split result (dig_findings "On the carrier token") is
exactly the SSS picture instantiated in a BOS-less packed stream: pos-0 fires unconditionally
regardless of identity (+1657, 510/512 windows — the >98%-of-vocab result), the window's
FIRST newline fires hardest (+2135; MA category-b's "first strong delimiter", singular by
construction), and 96% of newlines are bulk-ordinary — spikehood is a per-window ROLE, not a
token-type property. Two literature deviations worth keeping: (i) our spike is
**variance-carrying**, mean_frac ≈ 0.003, and the intervention triptych shows the
within-occurrence variance is read downstream (dig_findings "interventions") — MA's
constant-bias account (main.tex:264, 319) is only the mean; (ii) our onset is a
**training-time event** (at the RankMe peak), which none of the three papers time-resolve
beyond CV's checkpoint panel (iclr:194–207: all three phenomena lock in together by ~step 1k
in Pythia 410M/6.9b — consistent, coarser).

## (b) Norm placement

**SSS is the authority here** (4_anatomy:101–131, ablation table 4_anatomy:110–118; 7B
Llama-style models trained from scratch, 100B tokens):
- **Pre-norm baseline**: spike magnitude 3818, sink ratio 46%.
- **Sandwich norm** (extra RMSNorm on the block OUTPUT — i.e. write-norm): spike **520**,
  sinks intact (44.7%). "The extra RMSNorm bounds the block output, it prevents the residual
  stream from accumulating the unbounded values" (4_anatomy:127–128).
- **Sandwich + QK-norm**: spike **92** — near-total elimination, "confirming that these
  outliers are primarily generated to influence the query and key projections"
  (4_anatomy:129). Sinks still 42%.
- **DynamicTanh** (element-wise, no vector-wide norm): spike 153, sink ratio 61% — sinks
  survive spike removal entirely (4_anatomy:131).
Verdict (4_anatomy:265, 274): spikes are "an artifact of the Pre-Norm architecture's tendency
to accumulate unbounded values"; normalization placement is the causal lever, and no variant
costs perplexity.

**MA's complementary mechanism** (main.tex:442–444, 943–944): the pre-norm READ of a spiky
residual is what makes the trick work — the huge denominator crushes normal channels and
yields a sparse near-constant normalized vector. SSS formalizes it (3_2:10–38: bounded,
sparse, near-constant) and derives sink keys confined to 1–2 dimensions (3_2:50–66). MA also
shows the demand side: explicit learnable k'/v' attention biases make massive activations
disappear in GPT-2 (main.tex:473–486), while a prepended trainable sink token does NOT
(main.tex:484) — the model needs a bias *channel*, not a token.

**Our OLMo/Pythia split, restated in their terms.** OLMo-2's reordered norm IS sandwich-style
write-norm on the sublayer output, and OLMo-2 additionally has QK-norm — the exact pair SSS
finds eliminates spikes (their QKNorm variant cites OLMo, 4_anatomy:101). Pythia has
neither; **nanochat-d12 has QK-norm but NOT write-norm** (utils/nanochat_gpt.py:91) and shows
the full Pythia signature anyway — so QK-norm alone does not prevent the rogue mechanism in
this setting. Note SSS never tested QK-norm alone either (their "QKNorm" ablation is sandwich
PLUS QK-norm), so the two results are consistent: QK-norm alone insufficient (nanochat),
write-norm alone suppressive (SSS 3818→520; our toy 6/6), both together eliminate (SSS →92;
OLMo-2 clean). So H1.4 (final_report: write-norm forbids the outsized rogue
write by construction; toy write-norm 6/6 suppression; nanochat out-of-family confirmation)
is the same causal claim SSS establishes at 7B scale from scratch — independent, convergent,
and published. Their numbers even preserve our nuance: write-norm alone *attenuates* (3818→520)
rather than abolishes; QK-norm finishes the job. Note their framing slightly differs from ours:
we say write-norm forbids the *write*; they say QK-norm removes the *incentive* (spikes exist
to serve QK). Both predict OLMo-2 = no rogue. CV never tests OLMo-2 (our novel datum stands).

## (c) Propagation through attention — is sink content READ into other residuals?

The three papers give **conflicting-in-emphasis but reconcilable** answers:

- **MA: yes, deposit is real, constant, and load-bearing.** Attention concentrates on spike
  tokens (main.tex:353–357); decomposing attention output at every query position k into
  Σ_{i∈C} p_i v_i + rest (Eq. 7, main.tex:459–465) shows the value update contributed by the
  spike tokens is **nearly identical across query positions and across inputs** — "an
  additive bias term" deposited into *every* token's attention output, at layers 3/15/30
  alike (main.tex:949). Zeroing the four activations collapses the model (ppl → inf,
  main.tex:301, 315); mean-substitution is free (main.tex:302, 319). So sink-token value
  states are NOT near-zero in their models — they carry the shared bias every position
  receives.
- **CV: treats the deposit as ≈0.** Sinks are "approximate no-ops … attending to BOS tokens
  with near-zero value norms, heads effectively skip their contribution" (iclr:320, citing
  Bondarenko). This is the over-mixing-brake reading: the sink's job is where attention does
  NOT go. CV's own compression theorem doesn't need the deposit either way (it is about the
  slot rows' norms, iclr:218–246).
- **SSS: functional, but as routing not content.** Sinks "modulate attention outputs across
  heads and bias individual heads toward short-range dependencies" (0_abstract:6), act as a
  learned gate / "dumping ground" to switch heads off (4_anatomy:266, 276), induced by
  short-context training (4_anatomy:227–239, 267). No direct measurement of sink-token value
  states.

Reconciliation: a near-constant nonzero value deposit (MA) is *functionally* a no-op for
token discrimination (CV/SSS) — it shifts every residual by the same bias vector. What NONE
of the three papers measures is whether **per-occurrence variance** of the spike is
transmitted to other positions. Our H1.5 intervention says it is: token-conditional mean
substitution (which preserves the class pattern = MA's bias, kills the variance) costs +1.15
CE at the position AFTER a newline — worse than zeroing (+0.98) (dig_findings
"interventions"). Downstream reading of the residual exists; the open question was whether
the content also rides attention into non-slot residuals — i.e. whether the rogue variance
seen in padded last-token geometry is deposited there or merely reflects the slot rows that
sneak into the last-token sample.

## (d) Verdict + the deposit experiment

**Literature verdict: not settled.** MA documents an attention-mediated *constant* deposit
into all positions; CV asserts near-zero value states; neither addresses variance transport,
Pythia specifically, or what survives to the final stream. So we ran the experiment.

**Deposit test** (scratchpad `deposit_test.py`, staging node, Jul 13; block_rogue_id
pythia-1b step143000 — stored leaves are only blk3.attn.in / blk3.mlp.out /
before_final_norm, so a two-endpoint test, not a depth profile): project every row on v₁
(blk3.mlp.out top centered eigvec), split rows into slots (pos-0 + first newline, n=991) vs
bulk (n=260,641); compare bulk v₁-content entering blk3 vs at end of stack; per-window
correlation between slot spike size and mean bulk end-of-stack component; causal-order check.
NB projections are centered with the leaf global mean, so a *constant* deposit (MA's bias) is
invisible by construction — this measures **modulated (variance) deposit** only.

Results — **no modulated deposit survives to the final stream**:
- Bulk RMS v₁-component: 3.6 entering blk3 → 5.5 at before_final_norm (pos≥384 rows the
  same). Meanwhile bulk rows already receive a small direct write (mean −7.2, the −9.5 bulk
  value in dig_findings) which the late blocks cancel (final bulk mean −0.03), and the slots
  themselves are crushed from RMS ~1906 to 19.4. Nothing here needs an attention pathway.
- Bulk-centered variance **share** along v₁: 0.021 at blk3.attn.in → **0.0021** at
  before_final_norm — a 10× *decline* (bulk total variance grows 612 → 14,229 while v₁
  variance only 12.9 → 30.3). v₁ is also no longer a distinguished direction of the final
  bulk geometry (max |cos| with before_final_norm eigvecs 0.40, at rank #55).
- **Per-window coupling**: r(slot spike, mean bulk v₁@final) = **+0.10**, vs +0.05 for the
  blk3-input control — essentially no window-specific deposit.
- **Causal-order check**: bulk rows after vs before the window's first newline have
  *identical* final v₁ RMS (5.50 vs 5.50) — no downstream-of-slot signature.

**Verdict.** At the end of the stack, ordinary tokens carry no meaningful modulated rogue
component: whatever heads read from the sink slots (and H1.5 proves they read it — +1.15 CE
after newlines when the spike's variance is destroyed) is consumed and re-expressed in other
directions, not parked in bulk residuals along v₁. Two consequences: (i) CV's "no-op /
near-zero value" reading and MA's constant-bias reading are both compatible with our final
stream — but our padded last-token rogue variance is **not attention deposit along v₁**.
The tempting "direct slot inclusion" explanation (13.9% of padded last tokens are
newline-bearing) was CHECKED and is DEAD (Jul 13): only 5/2,331 of those trailing newlines
are their document's FIRST newline, so under the first-newline refinement they are
bulk-ordinary, not slots. The padded last-token persistence is therefore currently
unexplained — see dig_findings. (ii) The deposit measurement is post-cancellation: a
mid-stack deposit that the step-down blocks remove alongside the slots is not excluded
(the block_rogue_id_midstack run targets exactly this).

## What this means for us / open questions

1. **H1.4 has published convergent support at scale**: SSS's sandwich-norm and QK-norm
   ablations are our write-norm toy result run at 7B from scratch. Cite SSS next to the toy
   grid and nanochat prediction; note their added nuance (write-norm attenuates 3818→520,
   QK-norm eliminates →92; OLMo-2 has both).
2. **Functional story, three tiers**: constant bias (MA; our token-conditional +0.09 at
   newline positions confirms it), head gating/short-range routing (SSS), and per-occurrence
   variance that is *read* by the next position (our H1.5 +1.15) yet leaves **no v₁-aligned
   residue** in bulk final residuals (deposit test). The read is transient/rotated, not a
   parked component — a datum none of the three papers has.
3. **Last-token rogue variance is an open puzzle**: NOT v₁ deposit at the final stream
   (measured), and NOT slot inclusion (the 13.9% newline-bearing last tokens are almost
   never first newlines — 5/2,331). Remaining candidates: re-expressed next-position
   readout (H1.5's +1.15, rotated off v₁), a mid-stack deposit later removed, or the
   original "attenuated presence" claim over-reading other last-token structure.
4. Open: (i) mid-stack deposit profile — rerun block_rogue_id with blk{7,11,15}.attn.in to
   see whether attention deposits v₁ mid-stack and the step-down blocks remove it (SSS's
   step-up/step-down framing predicts the slots are cancelled; it says nothing about bulk);
   (ii) the `<|endoftext|>`-prepend twin experiment (does an explicit resting token absorb
   the first-newline slot — dig_findings); (iii) which heads do the H1.5 reading (SSS
   predicts short-range sink heads; our head-gradient compression finding may be its
   shadow); (iv) whether OLMo-2's QK-norm or its write-norm does the work — SSS says QK-norm
   is the stronger lever, our toy only tested write-norm.
