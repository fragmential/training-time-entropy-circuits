# Split-half overlap: what we did vs what we do now

Goal: do two disjoint halves $T_a, T_b$ of a task over-express the **same** directions
relative to general text?

## What we did (v1 — flawed)

One shared reference $G$. Per half, take the top-k eigenvectors of the whitened matrix:

$$U_a = \operatorname{topk\,eigvecs}\Big(\Sigma_G^{-1/2}\, \Sigma_{T_a}\, \Sigma_G^{-1/2}\Big),
\qquad
U_b = \operatorname{topk\,eigvecs}\Big(\Sigma_G^{-1/2}\, \Sigma_{T_b}\, \Sigma_G^{-1/2}\Big)$$

$$\text{overlap} = \frac{\lVert U_a^\top U_b \rVert_F^2}{k}$$

Problem: both sides divide by the **same noisy estimate** $\hat\Sigma_G$. Its estimation
error creates directions with $\lambda > 1$ for *any* sample — so even two independent
general-text samples "agree" (null overlap 0.43–0.60 instead of ~0.005). The ruler's dents
dominate the agreement.

## What we do now (v2)

Independent references per side, and the comparison moved to raw activation space:

$$U_a \;\text{from}\; \Sigma_{G_1}^{-1/2}\, \Sigma_{T_a}\, \Sigma_{G_1}^{-1/2},
\qquad
U_b \;\text{from}\; \Sigma_{G_2}^{-1/2}\, \Sigma_{T_b}\, \Sigma_{G_2}^{-1/2}$$

$$Q_a = \operatorname{orth}\big(V_{G_1} \Lambda_{G_1}^{-1/2} U_a\big),
\qquad
Q_b = \operatorname{orth}\big(V_{G_2} \Lambda_{G_2}^{-1/2} U_b\big)$$

$$\text{overlap} = \frac{\lVert Q_a^\top Q_b \rVert_F^2}{k}$$

Two changes, both necessary: (1) **independent whiteners** $G_1 \ne G_2$ remove the shared
ruler error (null drops to ~chance: 0.005–0.10); (2) the un-whitening
$v = V_G \Lambda_G^{-1/2} u$ is required because whitened coordinates are meaningless across
two different whiteners — without it even identical physical directions compare as random.
Null uses the same recipe with two further independent G samples:
$(G_3 \text{ vs } G_1) \times (G_4 \text{ vs } G_2)$.

Result (k=8/32): math 0.68–0.83, quotes 0.53–0.62, memorized 0.45–0.51, null ≤ 0.10.

## Does the v1 flaw contaminate the other experiments?

Mostly no, one caveat:

- **excess_mass / n(λ>2)** (Experiment A): unaffected in conclusion. These are scalars from a
  single whitening, and their null (G′ vs G) was measured with the *same shared ruler* — so
  the ruler's inflation is present equally in task and null, and "task clears the null"
  survives. (Absolute values are ruler-inflated; differences vs null are the meaningful part,
  and that's how they were read.)
- **tail_centroid**: conclusion (task ≈ null → no tail localization) stands for the same
  same-ruler-calibration reason, but with a power caveat: the top-8 directions of any single
  whitening contain some ruler-error directions, which dilutes the centroid toward the
  null's — a true weak tail preference could be partially masked. The v2 machinery (raw-space
  directions, independent whiteners) is the right way to re-measure this if it matters.
- **H2.1 (acts↔grads gen spectrum)**: no shared-whitener issue — it never compares two
  whitenings; both covariances come from the same population and the statistic was compared
  across populations, not across whitenings. (Its separate pending item — G-whitening as a
  preprocessing step — is about removing shared-G *content*, a different matter.)
