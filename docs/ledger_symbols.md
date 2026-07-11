# The ledger equation, symbol by symbol

For one block step, the residual update is

$$r_{\text{next}} = r + \sum_i c_i$$

(in the code: $r$ = `blk{k}.attn.in`, the $c_i$ = `attn.out` and `mlp.out`, and
$r_{\text{next}}$ = `blk{k+1}.attn.in`, or `before_final_norm` for the last block).

Same leaves for BOTH families: the sum only says "next stream = incoming stream + what the two
sub-blocks wrote", which holds whether the writes were computed in parallel (Pythia: both read
$r$) or sequentially (OLMo-2: the MLP reads $r + c_{\text{attn}}$). The ledger never needs what
a write was computed *from*, only what was added — the identity is order-independent. OLMo's
causal attn→mlp correlation just lands in $\chi$/$I$ like any other correlation; its finer
two-sub-step decomposition (via `mlp.in`) is deliberately not used, so families stay comparable.

## Ingredients (all on centered covariances)

- $S(\rho)$ — spectral entropy of the trace-normalized covariance, so
  $\mathrm{RankMe} = e^{S}$.
- $w_i = \operatorname{tr}\Sigma_i \big/ \sum_j \operatorname{tr}\Sigma_j$ — energy weights
  over the components $\{r, c_1, c_2\}$; they sum to 1, and $w_0$ (the stream's share) is
  reported as `w_in`.
- $s_i = S(\rho_i)$ — each component's own entropy; $s_r$ is the incoming stream's.
- $\bar\rho \propto \Sigma_r + \sum_i \Sigma_{c_i}$ — the **interference-free mix**
  (covariances just added, cross-terms dropped); its entropy is $s_{\text{mix}}$.
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
  subspaces. Always $0 \le \chi \le H(w)$ (entropy of the weight distribution) — the only
  always-positive term. Needs only the marginal covariances.
- **quality:** the writes' own entropies relative to the stream, energy-weighted — "is the
  block adding flatter or sharper energy than what's already flowing?" This is the term that
  collapses in Pythia (the blk3 rogue write: rank-one energy at large $w$).
- **$I$ (interference):** the only term needing the cross-covariances. Positive =
  correlations flatten the spectrum (rank restoration, Pythia's late blocks); negative =
  reinforcement concentrates it (OLMo's compression driver).

## Where they live in the results

`block_ledger` keys per `blk{k}` node: `delta_s`, `chi`, `quality`, `interference`, plus the
raw pieces `w`, `s`, `s_mix`, `s_next`.

The two-component variant (`incremental_overlap`, at each `blk*.{attn,mlp}.out` node) is the
same construction with just $\{r, c\}$; there $\chi$'s ceiling is the binary entropy $H(w)$,
and `chi_frac` $= \chi / H(w)$ is the normalized version.

## Mapping to your guessed form

"$S_{\text{after}} = S_{\text{before}} + S_{\text{added}} + S_{\text{overlap}} + S_{\text{interference}}$"
is structurally right:

| your guess | actual term |
|---|---|
| $S_{\text{after}}$ | $s_{\text{next}}$ |
| $S_{\text{before}}$ | $s_r$ |
| $S_{\text{added}}$ | quality $= \sum_i w_i (s_i - s_r)$ |
| $S_{\text{overlap}}$ | $\chi = s_{\text{mix}} - \sum_i w_i s_i$ |
| $S_{\text{interference}}$ | $I = s_{\text{next}} - s_{\text{mix}}$ |
