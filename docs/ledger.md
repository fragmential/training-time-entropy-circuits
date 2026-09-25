# The ledger: the exact decomposition, symbol by symbol

The instrument behind RQ1. "Ledger" and "quality" are repo-internal names and do not
appear in thesis prose; the thesis names are the blockwise decomposition (the telescoping
over blocks) and the overlap / interference / block-intrinsic-entropy decomposition (the
within-block split). The thesis-facing derivation is drafted in
`thesis-writing/decomposition_method/main.tex`; this file records the symbols, the
identity, and the metric taxonomy in reference form.

## The block step

Indexing convention (thesis-facing): stream states are $r_0$ (embeddings) to $r_L$
(final); blocks are $1 \dots L$. Block $\ell$ is the transition from state $\ell{-}1$ to
state $\ell$, so $S_\ell$ is the stream entropy after block $\ell$, and block quantities
($\Delta S_\ell$, $\chi_\ell$, $\beta_\ell$, $I_\ell$) carry the index of the block that
produced them. The code is 0-indexed: thesis block $\ell$ corresponds to `blk{ℓ-1}`.

For block $\ell$, with $a_\ell$ and $m_\ell$ the attention and MLP writes, the residual
update is

$$r_\ell = r_{\ell-1} + a_\ell + m_\ell$$

(in the code, with $k = \ell{-}1$: $r_{\ell-1}$ = `blk{k}.attn.in`, $a_\ell$ =
`attn.out`, $m_\ell$ = `mlp.out` of that block, $r_\ell$ = `blk{k+1}.attn.in`, or
`before_final_norm` for the last block).

The same leaves serve both model families: the identity states that the next stream
equals the incoming stream plus what the two sub-blocks wrote, which holds whether the
writes were computed in parallel (Pythia: both read $r_{\ell-1}$) or sequentially
(OLMo-2: the MLP reads $r_{\ell-1} + a_\ell$). The identity depends on what was added,
not on what a write was computed from, so it is order-independent. OLMo's causal
attention-to-MLP correlation enters $\chi$ and $I$ like any other correlation. The finer
two-sub-step decomposition (via `mlp.in`) is deliberately not used, so that the families
remain comparable.

## Ingredients (all on centered covariances)

All quantities below are per block; the $\ell$ subscript on $w_c$, $s_c$,
$s_{\text{mix}}$ is suppressed for readability. The iterator $c$ runs over the components
of block $\ell$'s step, $c \in \{r_{\ell-1}, a_\ell, m_\ell\}$, not over layers; sums
over layers use $\ell$ and appear only in the telescoping identity.

- $\rho_x = \Sigma_x / \operatorname{tr}\Sigma_x$: the trace-normalized (centered)
  covariance of component $x$; it has unit trace, so its eigenvalues form a
  distribution.

- $S(\rho)$: the spectral entropy of that eigenvalue distribution, so
  $\mathrm{RankMe} = e^{S}$. $S_\ell := S(\rho_{r_\ell})$ is the stream entropy at state
  $\ell$ ($S_0$ embeddings, $S_L$ final).

- $w_c = \operatorname{tr}\Sigma_c \big/ \sum_{c'} \operatorname{tr}\Sigma_{c'}$: energy
  weights over the components; they sum to 1. The stream's share is reported as `w_in`.

- $s_c = S(\rho_c)$: each component's entropy; $s_{r_{\ell-1}} = S_{\ell-1}$.

- $\bar\rho \propto \Sigma_{r_{\ell-1}} + \Sigma_{a_\ell} + \Sigma_{m_\ell}$: the
  interference-free mix (covariances added, cross-terms dropped); its entropy is
  $s_{\text{mix}}$. Equivalently $\bar\rho = \sum_c w_c \rho_c$.

- $S_\ell = S(\rho_{r_\ell})$: entropy of the summed stream, cross-terms included.

## The identity

$$\underbrace{S_\ell - S_{\ell-1}}_{\Delta S_\ell}
\;=\;
\underbrace{s_{\text{mix}} - \sum_{c} w_c s_c}_{\chi_\ell\ \text{(overlap)}}
\;+\;
\underbrace{\sum_{c} w_c\,(s_c - S_{\ell-1})}_{\beta_\ell\ \text{(block-intrinsic entropy)}}
\;+\;
\underbrace{S_\ell - s_{\text{mix}}}_{I_\ell\ \text{(interference)}}$$

Exact by telescoping: the three terms sum to $S_\ell - S_{\ell-1}$, using
$\sum_c w_c = 1$. Summed over the $L$ blocks it telescopes again to

$$\log \mathrm{RankMe}(\text{final}) - \log \mathrm{RankMe}(\text{embeddings})
= S_L - S_0 = \sum_{\ell=1}^{L} \Delta S_\ell .$$

## Term meanings

- $\chi_\ell$ (overlap): entropy the mix gains because components occupy different
  subspaces; $0 \le \chi \le H(w)$, the one term with a guaranteed sign. It requires the
  marginal covariances alone. Formally it is the Holevo quantity of the ensemble
  $\{w_c, \rho_c\}$. The quantity measures non-overlap (it is large when the subspaces
  are disjoint); the name follows the same convention as calling a glass half full.

- $\beta_\ell$ (block-intrinsic entropy): the writes' entropies relative to the stream,
  energy-weighted; it records whether the block adds flatter or sharper energy than the
  stream already carries. It is layer-dependent through both the weights and the write
  entropies; the stream component contributes $w_{r}\,(S_{\ell-1} - S_{\ell-1}) = 0$, so
  only block $\ell$'s writes enter. This is the term that collapses in Pythia (the blk3
  sink write: rank-one energy at large $w$).

- $I_\ell$ (interference): the one term sensitive to the cross-covariances, although
  none is computed explicitly; $S_\ell$ is measured on the hooked next-block input,
  whose covariance contains the cross-terms. Explicit cross-covariances (`View.cross`)
  are needed to attribute $I$ to specific pairs, not to compute it. Positive values mean
  correlations flatten the spectrum (rank restoration, Pythia's late blocks); negative
  values mean reinforcement concentrates it (OLMo-2's compression driver).

## Why not the "no cross-terms" baseline

Unlike everything above, this baseline is model-wide: it sums the writes
$u_1, \dots, u_{2L}$ of all blocks at once. The alternative
($\mathrm{RankMe}(Z) = e^{\chi} \prod_j \mathrm{RankMe}(u_j)^{w_j}$ under
$\Sigma_Z = \sum_j \Sigma_{u_j}$, gap = "cross-term effect") fails three ways, in
increasing order of importance:

1. It drops the embedding stream ($Z = \sum_j u_j$, but the final residual is
   $r_0 + \sum_j u_j$; at early checkpoints the embedding is most of the story).

2. It is one number per model; every interaction at every layer is lumped into one
   scalar, which renames the problem rather than decomposing it.

3. It is a counterfactual baseline where an exact identity is available: the residual is
   built sequentially, and sequential sums telescope.

## Exactness caveats

- The identity is exact for the $\lambda$-entropy RankMe (`matrix_entropy`/`rankme`),
  not for the $\sigma$-weighted `true_rankme`; both are stored, and the decomposition is
  exact for the former while tracking the latter.

- For Pythia's parallel blocks the clean step is per block, not per sub-block:
  $r_{\ell-1}$, `attn.out`, `mlp.out` form a 3-component step into the next block's
  `attn.in` (OLMo-2 uses the same block-level form).

## Where the terms live in the results

`block_ledger` keys per `blk{k}` node (code-indexed, $k = \ell{-}1$): `delta_s`, `chi`,
`quality` (the stored key for $\beta$; the results files predate the rename),
`interference`, plus the raw pieces `w`, `s`, `s_mix`, `s_next`.

The two-component variant (`incremental_overlap`, at each `blk*.{attn,mlp}.out` node) is
the same construction with a single write, $c \in \{r_{\ell-1}, u\}$; there the ceiling
of $\chi$ is the binary entropy $H(w)$, and `chi_frac` $= \chi / H(w)$ is the normalized
version.

## Alignment-metric taxonomy (which cosine is which)

`signed_trace` is $\mathbb{E}\langle a,b\rangle$ normalized by aggregate energy, an
energy-weighted cosine in which large-norm tokens dominate; mean-cosine
$\mathbb{E}[\langle a,b\rangle / \lVert a\rVert \lVert b\rVert]$ is the per-token
version. They diverge when token norms are heavy-tailed (sink tokens), so the pair is
more informative than either alone. CKA is a third, orthogonal axis: unsigned,
basis-level (do the two occupy the same subspace?), and blind to cancellation. In
summary: CKA measures shared subspace, `signed_trace` measures net energy-weighted
alignment, and mean-cosine measures typical-token alignment.
