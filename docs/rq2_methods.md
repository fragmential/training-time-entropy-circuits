# RQ2 methods, explained from scratch

What was actually computed for the RQ2 experiments, step by step, with every formula on its
own line. Goal: readable in a few minutes, trivially translatable to code.

---

## 1. The tool: the generalized eigenvalue problem

Everything below compares two covariance matrices. The comparison tool is the generalized
eigenvalue problem:

$$\Sigma_A \, v = \lambda \, \Sigma_B \, v$$

Both matrices are `(d, d)` covariances of the same activation space. Think of it as asking,
direction by direction:

$$\lambda_i = \frac{v_i^\top \Sigma_A v_i}{v_i^\top \Sigma_B v_i}
= \frac{\text{variance of population A along } v_i}{\text{variance of population B along } v_i}$$

### How you compute it (this is literally the code)

You "whiten" by B. Whitening means changing coordinates so that population B has variance 1
in every direction — its covariance becomes the identity matrix. The whitening matrix comes
from B's eigendecomposition $\Sigma_B = V_B \Lambda_B V_B^\top$:

$$\Sigma_B^{-1/2} = V_B \, \Lambda_B^{-1/2} \, V_B^\top$$

(in practice a small damping $\varepsilon$ is added to $\Lambda_B$ so nothing divides by ~0).
Then build the whitened version of A:

$$W = \Sigma_B^{-1/2} \; \Sigma_A \; \Sigma_B^{-1/2}$$

and take the ORDINARY eigendecomposition of `W`. Its eigenvalues are the generalized
eigenvalues. In code:

```python
lam_B, V_B = eigh(Sigma_B)                       # B's eigendecomposition
Wh = V_B @ diag((lam_B + eps) ** -0.5) @ V_B.T   # whitener
W  = Wh @ Sigma_A @ Wh                           # A, in B-whitened coordinates
lam, U = eigh(W)                                 # lam = the generalized eigenvalues
```

(The repo's implementation is algebraically the same but avoids forming `Wh` explicitly.)

### How you read the spectrum

The whole point is the SHAPE of `lam`:

| what you see | what it means |
|---|---|
| all $\lambda_i \approx$ the same constant $c$ (flat spectrum) | $\Sigma_A \approx c \cdot \Sigma_B$: the two populations have the SAME geometry, just rescaled |
| some $\lambda_i \gg 1$ | directions where A has much more variance than B — A "over-expresses" them |
| some $\lambda_i \ll 1$ | directions A barely uses relative to B |
| one huge $\lambda$, rest tiny | A is dominated by a single direction B doesn't care about |

Sanity check of the flat case: if $\Sigma_A = c\,\Sigma_B$, then $W = c \cdot I$, so every
eigenvalue is exactly $c$.

To turn "how flat" into one number, we use the same entropy trick as RankMe: normalize the
spectrum to sum to 1, take its entropy, exponentiate:

$$\mathrm{RankMe}(\lambda) = \exp\Big(-\sum_i p_i \log p_i\Big), \qquad p_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

Maximally flat spectrum → RankMe = d. One dominant ratio → RankMe ≈ 1.

**So: high RankMe of a generalized spectrum = "these two covariances are near-proportional."**

One caveat: if the whitener $\Sigma_B$ is badly conditioned (near-zero directions), whitening
amplifies its noise directions and spreads the spectrum for reasons that have nothing to do
with A. So only compare this number between runs that share the same kind of whitener (same
model, same data geometry).

---

## 2. Experiment A — does a task have structure beyond the general population? (H2.2)

Populations: `T` = a task dataset (math text, quotes, memorized sequences, ...),
`G` = the general pretraining-like mix. Same model, final checkpoint, same hookpoints, same
token selection for both.

**Step 1 — collect.** Run the model over each population's text. At each hookpoint (33 of
them: every block's `attn.in`, every `mlp.up.in`, and `before_final_norm`), accumulate the
centered covariance of the activations `h`:

$$\Sigma = \mathbb{E}[h h^\top] - \mu \mu^\top$$

One $\Sigma_T$ and one $\Sigma_G$ per hookpoint.

**Step 2 — compare.** Per hookpoint, solve the generalized problem of T against G
(section 1, with $A = T$ and $B = G$):

$$W = \Sigma_G^{-1/2} \, \Sigma_T \, \Sigma_G^{-1/2} \;\;\longrightarrow\;\; \{\lambda_i, u_i\}$$

**Step 3 — interpret.** Each $\lambda_i$ is "task variance per unit of general variance"
along one direction:

- $\lambda_i \approx 1$: the task looks like general text in this direction.
- $\lambda_i \gg 1$: a direction the task over-expresses — candidate *shared task structure*.

**Step 4 — reduce to numbers.**

```python
excess_mass = sum(log(lam[lam > 1]))   # total over-expression
n_excess    = (lam > 2).sum()          # count of clearly over-expressed directions
```

averaged over the 33 hookpoints (× acts/grads).

**Step 5 — calibrate against a null.** Even two independent samples OF THE SAME population
give λ ≠ 1 everywhere, purely from sampling noise. So the entire pipeline (steps 1–4) is
also run with T replaced by a *second, independent sample of G* (different shuffle seed,
same size). That result is the noise floor. A task "has structure beyond G" only if its
excess is clearly above the G-vs-G number — not above 1.

Result example: math web text scored excess_mass ≈ 148 against a null of ≈ 61 → real
structure. The verbatim-memorized strings scored ≈ 1.3× the null → almost nothing beyond
lexical novelty.

**Step 6 — locate the excess in G's spectrum.** A precision note first, because it's easy to
get confused here. Generalized eigenvectors are genuinely NEW directions — they live in
neither T's nor G's eigenbasis. But every vector has coordinates in every basis, and the
implementation happens to build the whitened matrix already rotated into G's eigenbasis
($M = \Lambda_G^{-1/2} V_G^\top \Sigma_T V_G \Lambda_G^{-1/2}$, which equals $V_G^\top W V_G$
— same eigenvalues), so the eigenvectors $u_i$ it returns come out with their components
**indexed by G's eigenrank**: $u_{ij}$ = how much the i-th excess direction draws on G's
j-th eigendirection, in whitened coordinates.

So the profile

$$p_j = u_{ij}^2$$

says which of G's directions the excess direction is built from — on an equal-variance
footing, since whitening has set every G-direction's variance to 1. The rank-centroid

$$\text{tail\_centroid} = \sum_j j \cdot p_j$$

(averaged over the top 8 excess directions) then says WHERE in G's spectrum the task's extra
structure lives: small = G's head, large = G's tail. (Deliberate choice: the *raw-space*
generalized eigenvector would carry an extra $1/\lambda_{G,j}$ weighting that mechanically
skews every profile toward the tail — the whitened profile is the non-tautological version.)
H2.2 predicted the tail; the measured centroids matched the null's (mid-spectrum), so that
part was not confirmed.

**Not run:** the additive variant ("rotate $\Sigma_T$ onto $V_G$, subtract $\Lambda_G$,
resort"). Only this multiplicative/whitening version was implemented.

---

## 2b. Experiment A′ — split-half coherence (the sharp version of H2.2)

Excess-over-G (Experiment A) can't distinguish "one shared mechanism" from "many independent
per-item storage directions" — both create excess variance. This experiment can: split the
task into two DISJOINT halves and ask whether they over-express the SAME directions.

**Step 1 — collect** the task as two disjoint halves $T_a, T_b$ (different shuffle seeds, or
an explicit partition for the memorized set), plus FOUR independent samples of G
($G_1..G_4$ — why four: see step 4).

**Step 2 — excess directions per half, each against its own G.** Run Experiment A's geneig
for ($T_a$ vs $G_1$) and ($T_b$ vs $G_2$), keeping the top-k eigenvectors this time.

**Step 3 — compare in RAW space.** Two pitfalls this experiment exposed, both load-bearing:

- Whitened coordinates are whitener-specific: a direction's coordinates under $G_1$-whitening
  and $G_2$-whitening differ (eigenbases of two noisy estimates rotate arbitrarily inside
  near-degenerate eigenvalue regions). Comparing whitened coordinate vectors across whiteners
  collapses everything to chance spuriously. So: un-whiten each direction back to activation
  space, $v = V_G \Lambda_G^{-1/2} u$, orthonormalize the k of them (QR), and compare the two
  k-dimensional subspaces:

$$\text{overlap} = \frac{\lVert Q_a^\top Q_b \rVert_F^2}{k} \in [0, 1]$$

- With a SHARED whitener the null is hugely inflated (0.43–0.60 measured): both comparisons
  inherit the same reference-estimation error, so the top "excess" directions of ANY two
  samples contain the same ruler-dents. Independent whiteners per side remove this.

**Step 4 — null.** Same pipeline with the halves replaced by two more independent G samples:
($G_3$ vs $G_1$) × ($G_4$ vs $G_2$). Measured: 0.005–0.10 ≈ chance (k/d) — confirming the
shared-ruler diagnosis.

**Step 5 — read.** Shared mechanism → high overlap; independent per-item storage → ≈ null.
Measured (k=8/32): math 0.68–0.83 (both families), quotes 0.53–0.62, memorized 0.45–0.51
(pythia strings) and 0.52 (OLMo Merullo set, 325-seq halves), null ≤ 0.10. The strong "no shared memorization direction" prediction is refuted (memorized
halves share ~half their excess subspace — plausibly the lexical geometry of high-entropy
strings), but the graded ordering math > quotes > memorized holds in every measurable cell.

---

## 3. Experiment B — are gradients aligned with activations? (H2.1)

Populations as above, but now the collection also runs a backward pass (next-token
cross-entropy), so each hookpoint accumulates TWO covariances over the same tokens:

- $\Sigma_{\text{acts}}$ — covariance of the activations $h$
- $\Sigma_{\text{grads}}$ — covariance of the gradients $\partial L / \partial h$

**Step 1 — per hookpoint, solve** the generalized problem of grads against acts
($A = $ grads, $B = $ acts):

$$W = \Sigma_{\text{acts}}^{-1/2} \; \Sigma_{\text{grads}} \; \Sigma_{\text{acts}}^{-1/2}$$

**Step 2 — interpret.** $\lambda_i$ = gradient variance per unit of activation variance
along direction $i$. A FLAT spectrum means gradient variance tracks activation variance
direction-by-direction — the loss "pushes along" the geometry the representation already
uses. A spread spectrum means learning pressure concentrates in directions disproportionate
to what's represented.

**Step 3 — reduce.** RankMe of the spectrum (section 1), averaged over hookpoints. Higher =
more coupled.

**Step 4 — compare across populations,** within one model and one data geometry only (the
caveat from section 1).

Result: memorized 177 > math-answers 123 > general 104 > quotes 92 (pythia-1b, padded).
H2.1 predicted memorized would be the LEAST coupled; it is the most. A plausible reading:
already-memorized items produce small gradients that live along the directions representing
them.

**The G-whitened variant (the "prior step") — run.** Whiten the task's acts by G-acts and the
task's grads by G-grads, THEN take the gen spectrum between the two whitened matrices. A
population with no real excess (G′) sets the CEILING (~630 padded / ~1960 packed: both
whitened matrices ≈ identity, trivially proportional); every real task falls below it, math
furthest (answers 103 < memorized 183 < quotes 291 < ceiling 634; pythia padded). Honest
reading: the scalar conflates "amount of genuine excess" with "acts-excess vs grads-excess
mismatch", so it is best read as gradient-side corroboration of H2.2's ordering, not a clean
coupling verdict. H2.1 is refuted in both variants. **The separation-clean design — run**: raw-space overlap between the top-k excess-acts and
excess-grads subspaces per population (section 2b machinery across quantities). Result:
**≈ chance for every population, both models, both k** (max: OLMo math-web 0.049 vs null
0.022 at k=32 — a weak ~2× at best; everything else ≤ ~2× chance). The task's excess
activation geometry and excess gradient geometry are essentially unrelated subspaces. H2.1 is
therefore refuted in all three variants — and the scalar variants' signals are now clearly
attributable to their confounds (population conditioning; excess amount). Figures: analysis/rq2_results.ipynb (last section).

---

## 4. Side variant — the T-eigenbasis ratio (proposed during review)

$r_i = \lambda_{T,i} / (v_{T,i}^\top \Sigma_G v_{T,i})$: the T-vs-G variance ratio evaluated
at T's OWN principal directions instead of optimized over all directions (the geneig strictly
generalizes it — T's PCs can misalign with where the ratio is extreme). Implemented as a
comparison cell in analysis/rq2_results.ipynb; useful as a cross-check of whether the excess
is variance-dominant in T.

## 5. Population inventory (what was actually collected)

| population | source | geometry | N | notes |
|---|---|---|---|---|
| G, G′, G″, G‴ | native mix (pile / olmo-mix), seeds 42–45 | packed 2M tokens AND padded 16,384 rows | | four independent samples per geometry (whitener + nulls) |
| math web (+ half B) | open-web-math, seeds 42/43 | packed 2M | | |
| math answer-slot | GSM8K question+answer | padded/last | ~7.5k | small-N caveat: train split size |
| quotes (+ half B) | quotes-500k, seeds 42/43 | padded/last 16,384 | | |
| memorized (+ halves) | EleutherAI pythia-memorized-evals (deduped.1b) | padded/last 16,384 (halves 8,192) | | pythia-only; high-entropy strings |
| memorized packed twin | same | packed all-token | ~1M tokens | matched-geometry pair for the OLMo set |
| memorized OLMo | Merullo et al olmo2_1b_mem set (reference repo) | packed all-token | 650 seqs ≈ 73k tokens | noisiest population; padded/last impossible (N < d) |

Models: pythia-1b-deduped + OLMo-2-0425-1B, final checkpoint, hooks blk*.attn.in / blk*.mlp.up.in / before_final_norm with `:both` (grads via next-token CE), storage cov_svd.
