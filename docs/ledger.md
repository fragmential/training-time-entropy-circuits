# The ledger — the exact decomposition, symbol by symbol

The instrument behind RQ1 (repo name: the rank ledger; thesis naming: the blockwise and
overlap/interference/block-intrinsic-entropy decompositions). Derivation, symbols, and the
metric taxonomy in one place.

## The block step

For one block, the residual update is

$$r_{\text{next}} = r + \sum_i c_i$$

(in the code: $r$ = `blk{k}.attn.in`, the $c_i$ = `attn.out` and `mlp.out`,
$r_{\text{next}}$ = `blk{k+1}.attn.in`, or `before_final_norm` for the last block).

Same leaves for BOTH families: the sum only says "next stream = incoming stream + what the
two sub-blocks wrote", which holds whether the writes were computed in parallel (Pythia:
both read $r$) or sequentially (OLMo-2: the MLP reads $r + c_{\text{attn}}$). The identity
never needs what a write was computed *from*, only what was added — it is
order-independent. OLMo's causal attn→mlp correlation just lands in $\chi$/$I$ like any
other correlation; the finer two-sub-step decomposition (via `mlp.in`) is deliberately not
used, so families stay comparable.

## Ingredients (all on centered covariances)

- $S(\rho)$ — spectral entropy of the trace-normalized covariance, so
  $\mathrm{RankMe} = e^{S}$.

- $w_i = \operatorname{tr}\Sigma_i \big/ \sum_j \operatorname{tr}\Sigma_j$ — energy weights
  over the components $\{r, c_1, c_2\}$; sum to 1; $w_0$ (the stream's share) is reported
  as `w_in`.

- $s_i = S(\rho_i)$ — each component's own entropy; $s_r$ the incoming stream's.

- $\bar\rho \propto \Sigma_r + \sum_i \Sigma_{c_i}$ — the **interference-free mix**
  (covariances added, cross-terms dropped); its entropy is $s_{\text{mix}}$.

- $s_{\text{next}} = S(\rho_{\text{next}})$ — entropy of the *actual* summed stream,
  cross-terms and all.

## The identity

$$\underbrace{s_{\text{next}} - s_r}_{\Delta S}
\;=\;
\underbrace{s_{\text{mix}} - \sum_i w_i s_i}_{\chi\ \text{(overlap)}}
\;+\;
\underbrace{\sum_i w_i\,(s_i - s_r)}_{\text{quality}}
\;+\;
\underbrace{s_{\text{next}} - s_{\text{mix}}}_{I\ \text{(interference)}}$$

Exact by pure telescoping: add the three terms and everything cancels except
$s_{\text{next}} - s_r$ (using $\sum_i w_i = 1$). Summed over blocks it telescopes again to

$$\log \mathrm{RankMe}(\text{final}) - \log \mathrm{RankMe}(\text{embeddings})
= \sum_k \Delta S_k .$$

## Term meanings

- **$\chi$ (overlap):** entropy the mix gains because components occupy *different*
  subspaces. Always $0 \le \chi \le H(w)$ — the only always-positive term. Needs only the
  marginal covariances.

- **quality (block-intrinsic entropy):** the writes' own entropies relative to the stream,
  energy-weighted — "is the block adding flatter or sharper energy than what's already
  flowing?" The term that collapses in Pythia (the blk3 sink write: rank-one energy at
  large $w$).

- **$I$ (interference):** the only term needing the cross-covariances. Positive =
  correlations flatten the spectrum (rank restoration, Pythia's late blocks); negative =
  reinforcement concentrates it (OLMo-2's compression driver).

## Why not the "no cross-terms" baseline

The multiplicative alternative ($\mathrm{RankMe}(Z) = e^{\chi}\prod_k
\mathrm{RankMe}(c_k)^{w_k}$ under $\Sigma_Z = \sum_k \Sigma_{c_k}$, gap = "cross-term
effect") fails three ways, in increasing order of importance:

1. It silently drops the embedding stream ($Z = \sum_k c_k$, but the residual is
   $r_0 + \sum_k c_k$ — at early checkpoints the embedding is most of the story).

2. It is one number per model — every interaction at every layer lumped into one scalar;
   it renames the problem instead of decomposing it.

3. It is a counterfactual baseline where an exact identity is available: the residual is
   built *sequentially*, and sequential sums telescope exactly.

## Exactness caveats

- The identity is exact for the $\lambda$-entropy RankMe (`matrix_entropy`/`rankme`), not
  the $\sigma$-weighted `true_rankme` — both are stored; the decomposition is exact for one
  and tracks the other.

- For Pythia's parallel blocks the clean step is per *block*, not per sub-block: $r$,
  `attn.out`, `mlp.out` form a 3-component step into the next block's `attn.in` (OLMo-2
  uses the same block-level form).

## Where the terms live in the results

`block_ledger` keys per `blk{k}` node: `delta_s`, `chi`, `quality`, `interference`, plus
the raw pieces `w`, `s`, `s_mix`, `s_next`.

The two-component variant (`incremental_overlap`, at each `blk*.{attn,mlp}.out` node) is
the same construction with just $\{r, c\}$; there $\chi$'s ceiling is the binary entropy
$H(w)$, and `chi_frac` $= \chi / H(w)$ is the normalized version.

## Alignment-metric taxonomy (which cosine is which)

`signed_trace` is $\mathbb{E}\langle a,b\rangle$ normalized by *aggregate* energy — an
energy-weighted cosine where big-norm tokens dominate; mean-cosine
$\mathbb{E}[\langle a,b\rangle / \lVert a\rVert \lVert b\rVert]$ is the democratic
per-token version. They diverge exactly when token norms are heavy-tailed (sink tokens),
so the *pair* is more informative than either alone. CKA is the third, orthogonal axis:
unsigned, basis-level ("same subspace?"), blind to cancellation. In short: CKA = shared
subspace, `signed_trace` = net energy-weighted alignment, mean-cosine = typical-token
alignment.
