# The exact rank ledger — notes on decomposing the RankMe trajectory

## What's wrong with the "no cross-terms" formula

The quoted claim: when blocks don't interfere, with

$$\rho_Z = \frac{\Sigma_Z}{\operatorname{tr}\Sigma_Z}, \qquad \bar\rho = \sum_k w_k \rho_k,\quad w_k = \frac{\operatorname{tr}\Sigma_{c_k}}{\sum_j \operatorname{tr}\Sigma_{c_j}},$$

killing the cross-terms gives $\Sigma_Z = \sum_k \Sigma_{c_k}$, hence $\rho_Z = \bar\rho$ and

$$\mathrm{RankMe}(Z) = e^{\chi}\prod_k \mathrm{RankMe}(c_k)^{w_k},$$

with the measured-vs-predicted gap "being" the cross-term effect.

Three problems, in increasing order of importance:

1. **It silently drops the embedding stream.** It writes the residual as $Z = \sum_k c_k$, but
   the residual is $r_0 + \sum_k c_k$ — the embedding covariance and its couplings are part of
   the story, and at early checkpoints they're most of it.

2. **It's the aggregate again.** The $M$-component construction answers "how orthogonal are all
   blocks to each other, collectively," which is one number per model, and the
   "gap = cross-term effect" lumps every interaction at every layer into one scalar. It renames
   the problem instead of decomposing it.

3. **It's a counterfactual baseline where an exact identity is available.** No "no cross-terms"
   idealization is needed at all, because the residual is built *sequentially*, and sequential
   sums telescope exactly.

## The exact ledger

Write $S(\cdot)$ for the spectral entropy of the trace-normalized centered covariance (so
$\exp S = \mathrm{RankMe}$, the matrix-entropy variant — caveat below). For one write step
$r_{\text{next}} = r + c$, with $w = \operatorname{tr}\Sigma_r / (\operatorname{tr}\Sigma_r +
\operatorname{tr}\Sigma_c)$ and $\bar\rho$ the normalized covariance of $\Sigma_r + \Sigma_c$:

$$\Delta S \;=\; S(\rho_{\text{next}}) - S(\rho_r) \;=\; \chi \;+\; (1-w)\,\bigl(S(\rho_c) - S(\rho_r)\bigr) \;+\; I$$

where $\chi$ is exactly the per-block `incremental_overlap`, the middle term is the block's
*spectral quality* (does it write a flatter or sharper spectrum than the stream, scaled by its
energy share), and $I = S(\rho_{\text{next}}) - S(\bar\rho)$ is the *pure interference* term —
the only part that needs the cross-covariance, positive when correlations flatten the spectrum,
negative when reinforcement concentrates it. No assumptions; it holds at every layer, every
checkpoint. Summing over depth:

$$\log \mathrm{RankMe}(\text{final}) \;=\; \log \mathrm{RankMe}(\text{embeddings}) \;+\; \sum_k \bigl[\chi_k + \text{quality}_k + I_k\bigr]$$

That is the decomposition being circled: "immediate layer contribution" is $\text{quality}_k$
(+ its weight $w_k$), "overlap" is $\chi_k$, "interference/cancellation" is $I_k$ — and they
provably add up to the thing Li et al plot. Nothing is a baseline to compare against; the books
balance by construction.

Two honesty caveats. First, the identity is exact for the $\lambda$-entropy
(`matrix_entropy`/`rankme`), not the $\sigma$-weighted `true_rankme` — both are stored, so plot
both and say the decomposition is exact for one and tracks the other. Second, for Pythia's
parallel blocks the clean step is per *block*, not per sub-block: $r$ and both `attn.out`,
`mlp.out` form a 3-component step into the next block's `attn.in` (OLMo can use the same
block-level form, where it just composes its two sequential sub-steps).

## What this buys as a thesis

The implicit research question: **through what layerwise mechanism does the entropy-seeking →
compression-seeking rank trajectory arise in residual-stream transformers?** Li et al's toy
model can't answer it because a one-layer classifier has no stream to write into. This
instrument answers it by measurement: for every layer and checkpoint, the rank change is
attributed to one of three named mechanisms.

That gives falsifiable hypotheses instead of exploration:

- **H1 (mechanism of the transition):** entropy-seeking is $\chi$-driven — early in training,
  blocks write novel directions — while compression is $I$-driven: blocks increasingly write
  *into* occupied directions ($\operatorname{tr} P$ reinforcement rising, $I_k$ going
  negative). The transition point is where the dominant ledger term flips. Directly checkable
  in the ledger curves.
- **H2 (locus):** the compression phase is carried by the late layers' terms (consistent with
  late-layer trace dominance), and specifically by their interference with the top
  eigendirections — `eigendirection_attribution` tells whether block-$k$ energy lands on the
  head or the tail of the final spectrum.
- **H3 (toy-model correspondence):** the last-layer terms should qualitatively reproduce the
  classifier toy model (collapse toward class/top directions), while mid-stack terms should
  not — which is precisely the sense in which the toy model "explains" the transformer curve
  incompletely.

The stray observations stop being stray: the pythia-1b layer-3 anomaly becomes "what does its
ledger row look like — huge $w$ with low $\chi$ (amplifier) or high $\chi$ (novel subspace)?";
the mean/tail-alignment result is a separate short chapter (means don't move centered rank, but
*where* they sit in the spectrum is still a real finding).

## Cosine question

Yes, measure it, and it's cheap ($O(Nd)$, streaming, no $d^2$). But know what it is relative to
the existing metrics: `signed_trace` is $\mathbb{E}\langle a,b\rangle$ normalized by
*aggregate* energy — an energy-weighted cosine where big-norm tokens dominate; mean-cosine
$\mathbb{E}\bigl[\langle a,b\rangle / \lVert a\rVert\,\lVert b\rVert\bigr]$ is the democratic,
per-token version. They diverge exactly when token norms are heavy-tailed (rogue tokens), so
the *pair* is more informative than either alone. CKA is the third, orthogonal axis: unsigned,
basis-level ("same subspace?"), blind to cancellation. So: CKA = shared subspace,
`signed_trace` = net energy-weighted alignment, mean-cosine = typical-token alignment.

## What must land before the big run

Everything above that needs covariances or samples cannot be recomputed from the saved
eigenvalues — same logic as drift. Concretely: (1) `incremental_overlap` also returns the
entropies it already computes ($S_{\text{mix}}$, $S_{\text{in}}$, $S_{\text{out}}$ — making the
ledger assemblable downstream), (2) the block-level 3-component ledger variant for
Pythia/uniformly, (3) mean-cosine added to `block_residual_coupling` and the
`block_block_coupling` matrices. All small. (All three landed before the big runs.)
