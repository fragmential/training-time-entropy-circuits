#!/usr/bin/env python
"""Compute spectral metrics (RankMe, alpha, eigenspectrum) from collected data.

Handles all storage formats uniformly via DataAccessor:
    - acts (.pt): PCA eigendecomposition per hook point → metrics
    - cov (.pt): eigendecompose covariance per hook point per factor → metrics
    - cov_svd (.pt): eigenvalues already stored → metrics
    - eigenvalues (.pt): eigenvalues already stored → metrics

Computes per hook point:
    - A / G / B spectral metrics (eigenspectrum, RankMe, alpha, R2)
    - K-FAC curvature: trace(G ⊗ A), damped log-determinant at multiple alpha values
    - K-FAC top-k eigenspectrum: sorted outer products λ_A^i * λ_G^j
    - K-FAC sampled spectrum: linearly spaced sample across full outer product distribution
    - Generalized eigendecomposition G vs B (when B derivable)

Results structure:
    {step: {hook_name: {
        "A": {eigenspectrum, rankme, alpha, r2, r2_100},
        "G": {...},
        "B": {...},             # when available
        "kfac": {trace, log_det, top_eigvals, sampled_eigvals, rankme, alpha, ...},
        "gen_GB": {eigvals, rankme, alpha, ...},  # generalized G vs B
    }}}

Usage:
    python scripts/compute_metrics.py inferences/full_limited pythia-14m-deduped
    python scripts/compute_metrics.py full_limited pythia-70m-deduped
    python scripts/compute_metrics.py fineweb pythia-14m-deduped --recompute
"""

import heapq
import os
import sys
import re
import numpy as np
import torch
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.accessor import DataAccessor


# ---------------------------------------------------------------------------
# Metric functions (moved from utils/powerlaw.py)
# ---------------------------------------------------------------------------

def fit_powerlaw(arr: torch.Tensor, start: int, end: int):
    """Fit power law in log-log space via least squares."""
    x_range = torch.arange(start, end + 1, dtype=torch.long)
    y_range = arr[x_range - 1]  # eigenval_{start} is at index (start-1)
    log_x = torch.log(x_range.to(y_range.dtype)).unsqueeze(1)  # (n, 1)
    log_y = torch.log(y_range).unsqueeze(1)                     # (n, 1)
    # Solve [log_x, 1] @ [slope; intercept] = log_y
    A = torch.cat([log_x, torch.ones_like(log_x)], dim=1)      # (n, 2)
    result = torch.linalg.lstsq(A, log_y)
    coeffs = result.solution.flatten()  # [slope, intercept]
    slope, intercept = coeffs[0], coeffs[1]
    y_pred = torch.exp(slope * log_x + intercept)
    return -slope.item(), x_range, y_pred


def robust_fit_powerlaw(arr: torch.Tensor, start: int, end: int, verbose: bool = False) -> float:
    window = int((end - start) / 10)
    slopes = torch.tensor([fit_powerlaw(arr, idx, idx + window)[0] for idx in range(start, end + 1)])
    robust_slope = float(slopes.median())
    if verbose:
        print(robust_slope)
        import matplotlib.pyplot as plt
        arr_np = arr.numpy()
        x_range_plt = np.arange(start, end + 1).astype(int)
        y_pred_full = np.exp(
            -robust_slope * np.log(x_range_plt) + np.log(arr_np[start - 1]) + robust_slope * np.log(start))
        plt.loglog(np.arange(1, 1 + len(arr_np)), arr_np)
        plt.loglog(x_range_plt, y_pred_full)
        plt.show()
    return robust_slope

def stringer_get_powerlaw(ss: torch.Tensor, trange: torch.Tensor, top_k: int = 2048):
    # COPIED FROM Stringer+Pachitariu 2018b github repo, ported to torch
    # (https://github.com/MouseLand/stringer-pachitariu-et-al-2018b/blob/master/python/utils.py)
    ''' fit exponent to variance curve'''
    if top_k is not None:
        ss = ss[:top_k]
    ss = ss[ss > 0]
    logss = torch.log(torch.abs(ss))
    y = logss[trange].unsqueeze(1)
    trange = trange + 1
    nt = trange.numel()
    x = torch.cat((-torch.log(trange.float()).unsqueeze(1), torch.ones(nt, 1)), dim=1)
    w = (1.0 / trange.float()).unsqueeze(1)
    b = torch.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = torch.arange(0, ss.numel(), dtype=torch.long) + 1
    x = torch.cat((-torch.log(allrange.float()).unsqueeze(1), torch.ones(ss.numel(), 1)), dim=1)
    ypred = torch.exp((x * b).sum(dim=1))
    alpha = b[0]
    max_range = 500 if len(ss) >= 512 else len(
        ss) - 10  # subtracting 10 here arbitrarily because we want to avoid the last tail!

    # Inline R2: 1 - SS_res / SS_tot
    def _r2(y_true, y_pred_slice):
        ss_res = ((y_true - y_pred_slice) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()
        if ss_tot == 0:
            return None
        return float(1.0 - ss_res / ss_tot)

    t0 = trange[0]
    fit_R2 = _r2(logss[t0:max_range], torch.log(torch.abs(ypred))[t0:max_range])
    try:
        fit_R2_100 = _r2(logss[t0:100], torch.log(torch.abs(ypred))[t0:100])
    except:
        fit_R2_100 = None
    return alpha, ypred, fit_R2, fit_R2_100

def rankme_metrics(eigen: torch.Tensor, top_k: int = 20000) -> dict:
    eigen = torch.clamp(eigen, min=0) # We've already done this a million times but it don't hurt to do it another
    if top_k is not None:
        eigen = eigen[:top_k]       # Truncate up to top_k
    eps = 1e-7

    def entropy(p):
        return -torch.sum(p * torch.log(torch.clamp(p, min=eps)))

    sv = torch.sqrt(eigen)
    p_sv = sv / sv.sum()          # true RankMe weights (∝ σᵢ)
    p_ev = eigen / eigen.sum()    # matrix entropy weights (∝ λᵢ)

    h_ev, h_sv = entropy(p_ev), entropy(p_sv)
    return {'matrix_entropy': float(h_ev), 'sv_entropy': float(h_sv),
            'rankme': float(torch.exp(h_ev)),       # RankMe per melody, arna and kumar
            'true_rankme': float(torch.exp(h_sv))}  # true RankMe per the RankMe paper

_STEP_RE = re.compile(r"step(\d+)\.pt$")


# ---------------------------------------------------------------------------
# Spectral metrics from an eigenvalue array
# ---------------------------------------------------------------------------

def spectral_metrics(eigen: torch.Tensor, damping: float = 1e-6, mean: torch.Tensor = None) -> dict:
    """Compute RankMe, alpha, R2, log-det from an eigenspectrum (descending, non-negative).

    eigen must be a torch.Tensor (descending, non-negative).
    If `mean` is given, also records `mean_norm` = ||μ||₂.
    """
    eigen = eigen.clamp(min=0)
    trace = float(eigen.sum())
    d = len(eigen)
    eps = damping * trace / max(d, 1)
    log_det = float(torch.sum(torch.log(eigen + eps)))

    eigen = eigen / eigen.sum()  # normalise into sum 1
    rm = rankme_metrics(eigen)
    alpha_fit, ypred, fit_r2, fit_r2_100 = stringer_get_powerlaw(eigen, torch.arange(11, 100))
    out = {
        "d": d,
        "eigenspectrum": eigen,
        "trace": trace,
        "avg_magnitude": trace / d,
        "log_det": log_det,
        **rm,
        "alpha": float(alpha_fit),
        "ypred": ypred,
        "r2": fit_r2,
        "r2_100": fit_r2_100,
    }
    if mean is not None:
        out["mean_norm"] = float(torch.linalg.norm(mean))
    return out

# ---------------------------------------------------------------------------
# Mean alignment metrics
# ---------------------------------------------------------------------------

def mean_metrics(eigvecs: torch.Tensor, eigvals: torch.Tensor, mean: torch.Tensor, damping: float = 1e-6) -> dict:
    """Metrics characterizing how the mean direction relates to a covariance eigenbasis.

    Args:
        eigvecs: (d, d) torch tensor, eigenvectors as columns, matching eigh convention.
                 Should correspond to the *centered* covariance for meaningful interpretation.
        eigvals: (d,) torch tensor, eigenvalues, assumed sorted descending. Non-negative.
        mean:    (d,) torch tensor, mean vector μ.
        damping: relative damping for Mahalanobis (added to eigvals as damping * mean(eigvals)).

    Returns dict with:
        mean_norm:           ||μ||₂
        max_overlap:         max_j |<μ̂, v_j>|         (which single eigvec μ aligns best with)
        max_overlap_idx:     argmax of the above
        pr:                  participation ratio of |<μ̂, v_j>|² over j ∈ [1, d]
        rayleigh:            μ̂ᵀ Σ μ̂                  (variance along the mean direction)
        rayleigh_normed:     μ̂ᵀ Σ μ̂ / λ_max          ∈ [0, 1]
        mahalanobis:         μ̂ᵀ Σ⁻¹ μ̂ (damped)       (low → mean lies in high-var subspace)
        pr_weighted:         participation ratio of energy-weighted profile λ_j |<μ̂, v_j>|²
        centroid_idx:        Σ_j j · p_j              (where in the spectrum μ lives, unweighted)
        centroid_idx_weighted: Σ_j j · p̃_j           (same, energy-weighted)
        profile:             (d,) torch tensor p_j = |<μ̂, v_j>|²       (sums to 1)
        profile_weighted:    (d,) torch tensor p̃_j ∝ λ_j p_j           (sums to 1)
    """
    eigvals = eigvals.clamp(min=0).to(torch.float64)
    mean = mean.to(torch.float64)
    d = len(eigvals)

    mean_norm = float(torch.linalg.norm(mean))
    if mean_norm == 0.0:
        # degenerate: μ = 0, every alignment metric is undefined
        return {
            "mean_norm": 0.0,
            "max_overlap": 0.0, "max_overlap_idx": -1,
            "pr": float("nan"), "rayleigh": 0.0, "rayleigh_normed": 0.0,
            "mahalanobis": 0.0, "pr_weighted": float("nan"),
            "centroid_idx": float("nan"), "centroid_idx_weighted": float("nan"),
            "profile": torch.zeros(d), "profile_weighted": torch.zeros(d),
        }
    mu_hat = mean / mean_norm

    # Projection profile: p_j = |<μ̂, v_j>|², sums to 1 since {v_j} is orthonormal
    coeffs = eigvecs.to(torch.float64).T @ mu_hat  # (d,) — coordinates of μ̂ in eigenbasis
    profile = coeffs ** 2
    profile = profile / profile.sum()        # numerical safety; should already sum to 1

    # Max overlap
    max_idx = int(torch.argmax(profile))
    max_overlap = float(torch.sqrt(profile[max_idx]))   # |<μ̂, v_j>|, not squared

    # Participation ratio (unweighted)
    pr = float(1.0 / torch.sum(profile ** 2))

    # Rayleigh quotient: variance of the data along the mean direction
    rayleigh = float(torch.sum(eigvals * profile))
    lam_max = float(eigvals[0]) if eigvals[0] > 0 else 1.0
    rayleigh_normed = rayleigh / lam_max

    # Mahalanobis: μ̂ᵀ Σ⁻¹ μ̂, damped
    eps = damping * float(eigvals.mean())
    mahalanobis = float(torch.sum(profile / (eigvals + eps)))

    # Energy-weighted profile
    weighted = eigvals * profile
    w_sum = float(weighted.sum())
    if w_sum > 0:
        profile_weighted = weighted / w_sum
        pr_weighted = float(1.0 / torch.sum(profile_weighted ** 2))
    else:
        profile_weighted = torch.zeros(d)
        pr_weighted = float("nan")

    idx = torch.arange(d, dtype=torch.float64)
    centroid_idx = float(torch.sum(idx * profile))
    centroid_idx_weighted = float(torch.sum(idx * profile_weighted)) if w_sum > 0 else float("nan")

    return {
        "mean_norm": mean_norm,
        "max_overlap": max_overlap,
        "max_overlap_idx": max_idx,
        "pr": pr,
        "rayleigh": rayleigh,
        "rayleigh_normed": rayleigh_normed,
        "mahalanobis": mahalanobis,
        "pr_weighted": pr_weighted,
        "centroid_idx": centroid_idx,
        "centroid_idx_weighted": centroid_idx_weighted,
        "profile": profile,
        "profile_weighted": profile_weighted,
    }

# ---------------------------------------------------------------------------
# K-FAC metrics
# ---------------------------------------------------------------------------

def _top_k_outer_products(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Top-k products from outer(a, b) where a, b are sorted descending.

    Uses flat outer product for small dims, priority queue for large.
    """
    m, n = len(a), len(b)
    if m * n <= 2_000_000:
        prods = np.outer(a, b).ravel()
        if len(prods) <= k:
            return np.sort(prods)[::-1].copy()
        idx = np.argpartition(prods, -k)[-k:]
        return np.sort(prods[idx])[::-1].copy()

    # Priority queue: O(k log(min(m,n)))
    heap = [(-(a[0] * b[0]), 0, 0)]
    seen = {(0, 0)}
    result = []
    while len(result) < k and heap:
        neg_prod, i, j = heapq.heappop(heap)
        result.append(-neg_prod)
        if i + 1 < m and (i + 1, j) not in seen:
            heapq.heappush(heap, (-(a[i + 1] * b[j]), i + 1, j))
            seen.add((i + 1, j))
        if j + 1 < n and (i, j + 1) not in seen:
            heapq.heappush(heap, (-(a[i] * b[j + 1]), i, j + 1))
            seen.add((i, j + 1))
    return np.array(result)


def _sample_outer_products_linspace(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Sample k products linearly spaced across the full outer product distribution.

    For small matrices, computes all products and samples exactly.
    For large matrices (> 2M products), approximates using sorted index mapping.
    Returns array of length min(k, m*n) in descending order.
    """
    m, n = len(a), len(b)
    total = m * n
    if total <= k:
        prods = np.outer(a, b).ravel()
        prods.sort()
        return prods[::-1].copy()
    if total <= 2_000_000:
        prods = np.outer(a, b).ravel()
        prods.sort()
        prods = prods[::-1].copy()
        idx = np.round(np.linspace(0, len(prods) - 1, k)).astype(int)
        return prods[idx]
    # Approximate: map rank -> (i, j) via sorted row-major order
    rank_idx = np.round(np.linspace(0, total - 1, k)).astype(int)
    i_idx = np.minimum(rank_idx // n, m - 1)
    j_idx = np.minimum(rank_idx % n, n - 1)
    sampled = a[i_idx] * b[j_idx]
    return np.sort(sampled)[::-1].copy()


def _histogram_outer_product(a: np.ndarray, b: np.ndarray, bins: int = 1024,
                             chunk_limit: int = 2_000_000) -> dict:
    """Histogram the full K-FAC outer product a ⊗ b in linear and log-spaced bins.

    Computes in chunks so the full outer product is never materialised.
    Returns density-normalised histograms (integral = 1) plus the edges and
    n_total = |a|*|b| so counts can be recovered as `density * n_total * diff(edges)`.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m, n = len(a), len(b)
    total = m * n

    max_val = float(a[0] * b[0])  # inputs are sorted descending
    lin_edges = np.linspace(0, max_val, bins + 1)

    a_pos = a[a > 0]
    b_pos = b[b > 0]
    has_log = len(a_pos) > 0 and len(b_pos) > 0 and max_val > 0
    log_edges = (np.logspace(np.log10(float(a_pos.min() * b_pos.min())),
                             np.log10(max_val), bins + 1)
                 if has_log else None)

    lin_counts = np.zeros(bins, dtype=np.int64)
    log_counts = np.zeros(bins, dtype=np.int64) if has_log else None
    chunk_rows = max(1, chunk_limit // max(n, 1))
    for i in range(0, m, chunk_rows):
        chunk = np.outer(a[i:i + chunk_rows], b).ravel()
        lin_counts += np.histogram(chunk, bins=lin_edges)[0]
        if has_log:
            log_counts += np.histogram(chunk, bins=log_edges)[0]

    out = {
        "histogram": lin_counts / (total * np.diff(lin_edges)),
        "histogram_edges": lin_edges,
        "n_total": total,
    }
    if has_log:
        out["loghistogram"] = log_counts / (total * np.diff(log_edges))
        out["loghistogram_edges"] = log_edges
    return out


def _check_negative_eigenvalues(eigvals: torch.Tensor, label: str):
    min_eig = float(eigvals[-1])
    if min_eig >= 0:
        return
    trace = float(eigvals.sum())
    d = len(eigvals)
    scale = max(float(eigvals[0]), abs(trace) / max(d, 1), 1e-12)
    rel = -min_eig / scale
    if rel > 1e-3:
        print(f"  WARNING {label}: large negative eigenvalue {min_eig:.3e} (relative {rel:.2e}) — suspect matrix")
    elif rel > 1e-5:
        print(f"  Note {label}: small negative eigenvalue {min_eig:.3e} (relative {rel:.2e})")


def kfac_metrics(
    eigvals_A: torch.Tensor,
    eigvals_G: torch.Tensor,
    top_k: int = 20000,
    sample_k: int = 1000,
    damping: float = 1e-6,
) -> dict:
    """K-FAC metrics from A and G eigenvalues (torch tensors).

    Returns:
        trace:          trace(G ⊗ A) = trace(G) * trace(A)
        log_det:        damped log-determinant: d_A * logdet(G + ε_G I) + d_G * logdet(A + ε_A I)
                        where ε_X = damping * trace(X) / d_X
        top_eigvals:    top-k exact products λ_A^i * λ_G^j
        sampled_eigvals: sample_k products linearly spaced across full distribution
        + spectral metrics on top_eigvals (rankme, alpha, r2, r2_100)
    """
    # Convert to numpy for numpy-only helper functions
    eA_np = eigvals_A.numpy()
    eG_np = eigvals_G.numpy()

    d_in, d_out = len(eigvals_A), len(eigvals_G)

    trace_A = float(eigvals_A.sum())
    trace_G = float(eigvals_G.sum())
    trace = trace_A * trace_G

    eps_A = damping * trace_A / max(d_in, 1)
    eps_G = damping * trace_G / max(d_out, 1)
    ld_A = float(torch.sum(torch.log(eigvals_A + eps_A)))
    ld_G = float(torch.sum(torch.log(eigvals_G + eps_G)))
    log_det = d_in * ld_G + d_out * ld_A

    k_top = min(top_k, d_in * d_out)
    top_eigvals = torch.from_numpy(_top_k_outer_products(eA_np, eG_np, k_top)).clamp(min=0)

    k_samp = min(sample_k, d_in * d_out)
    sampled_eigvals = torch.from_numpy(_sample_outer_products_linspace(eA_np, eG_np, k_samp)).clamp(min=0)

    sm = spectral_metrics(top_eigvals) if len(top_eigvals) >= 11 else {}

    hist = _histogram_outer_product(eA_np, eG_np, bins=1024)

    return {
        **sm,
        "trace": trace,
        "trace_A": trace_A,
        "trace_G": trace_G,
        "log_det": log_det,
        "log_det_A": ld_A,
        "log_det_G": ld_G,
        "top_eigvals": top_eigvals,
        "sampled_eigvals": sampled_eigvals,
        "d": d_in * d_out,
        "d_A": d_in,
        "d_G": d_out,
        **hist,
    }


# ---------------------------------------------------------------------------
# Generalized eigendecomposition G vs B
# ---------------------------------------------------------------------------

def generalized_eigenvalues_GB(
    eigvals_G: torch.Tensor,
    eigvecs_G: torch.Tensor,
    eigvals_B: torch.Tensor,
    eigvecs_B: torch.Tensor,
    eps_factor: float = 1e-6,
) -> torch.Tensor:
    """Generalized eigenvalues of G w.r.t. B: solve Gv = λBv.

    Both G and B must be in the same space (d_out × d_out).
    Returns the generalized eigenvalues (descending), which are
    ratios v^T G v / v^T B v — directions where gradient variance is
    most/least disproportionate to output activation variance.

    Formula: eigenvalues of B^{-1/2} G B^{-1/2}.
    Efficient via cross-basis:
        Q = V_B^T V_G
        M = Λ_B^{-1/2} Q Λ_G Q^T Λ_B^{-1/2}  (= C C^T where C = Λ_B^{-1/2} Q Λ_G^{1/2})
    eigenvalues of M = generalized eigenvalues of G w.r.t. B.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    eG = eigvals_G.to(dtype=torch.float32, device=device)
    vG = eigvecs_G.to(dtype=torch.float32, device=device)
    eB = eigvals_B.to(dtype=torch.float32, device=device)
    vB = eigvecs_B.to(dtype=torch.float32, device=device)

    eps = max(eB.max().item() * eps_factor, 1e-10)
    lb_inv_sqrt = (eB + eps).rsqrt()          # (d,)
    lg_sqrt = eG.clamp(min=0).sqrt()          # (d,)

    Q = vB.T @ vG                             # (d, d) cross-basis
    C = lb_inv_sqrt.unsqueeze(1) * Q * lg_sqrt.unsqueeze(0)  # (d, d)
    M = C @ C.T                               # symmetric PSD

    from utils.accessor import eigvalsh_descending
    gen_eigvals = eigvalsh_descending(M).cpu()
    return gen_eigvals

def compute_metrics_for_checkpoint(accessor: DataAccessor, verbose: bool = False, gpu_lock=None):
    import time
    from contextlib import nullcontext
    from utils.accessor import decomp_profiler
    if verbose:
        decomp_profiler.enable()
    avail = accessor.available()
    step_results = {}
    t0 = time.time()
    gpu_ctx = gpu_lock or nullcontext()
    gpu_wait = 0.0

    # Pre-warm ALL eigendecompositions under GPU lock so the rest is pure numpy
    with gpu_ctx as ctx:
        gpu_wait += getattr(ctx, "waited", 0.0)
        for hook_name, factors in avail.items():
            t = time.time()
            has_B = "B" in factors
            for factor in factors:
                accessor._ensure_eigh(hook_name, factor,
                                      need_vecs=True, need_centered_vecs=True)
            if has_B:
                accessor._ensure_B_eigh(hook_name, need_vecs=True, need_centered_vecs=True)
            if verbose:
                print(f"    prewarm {hook_name}: {time.time()-t:.1f}s")

    if verbose:
        print(f"    prewarm total: {time.time()-t0:.1f}s")

    for hook_name, factors in avail.items():
        t_hook = time.time()
        hook_results = {}
        hook = accessor[hook_name]
        entry = accessor._entry(hook_name)

        eA = eG = None  # cache for K-FAC / gen eigen (torch tensors)
        for factor in factors:
            fv = hook._factor(factor)
            eigvals = fv.eigvals
            if eigvals is None:
                continue
            _check_negative_eigenvalues(eigvals, f"{hook_name}.{factor}")
            eigvals = eigvals.clamp(min=0)
            hook_results[factor] = spectral_metrics(eigvals, mean=fv.mean)
            if factor == "A":
                eA = eigvals
            elif factor == "G":
                eG = eigvals

            # Centered eigenvalues
            centered = fv.eigvals_centered
            if centered is not None:
                centered = centered.clamp(min=0)
                hook_results[f"{factor}_centered"] = spectral_metrics(centered)

            # Mean overlaps
            if fv.mean is not None and fv.eigh_centered is not None:
                hook_results[f"{factor}_mean_metrics"] = mean_metrics(fv.eigvecs_centered, fv.eigvals_centered, fv.mean)

            # Cross-basis eigenvalues stored alongside
            for ek, ev in entry.items():
                if ek.startswith(f"{factor}_cross_eigvals_") and isinstance(ev, torch.Tensor):
                    cross_label = ek[len(f"{factor}_cross_eigvals_"):]
                    cross_eigvals = torch.clamp(ev, min=0)
                    hook_results[f"{factor}_cross_{cross_label}"] = spectral_metrics(cross_eigvals)

        # --- K-FAC metrics (needs both A and G eigenvalues) ---
        if eA is not None and eG is not None:
            hook_results["kfac"] = kfac_metrics(eA, eG)

        # --- Generalized eigendecomposition G vs B ---
        if "B" in factors:
            g_eigh = hook.G.eigh
            b_eigh = hook.B.eigh
            if g_eigh is not None and b_eigh is not None:
                eG_arr, vG = g_eigh
                eB_arr, vB = b_eigh
                with gpu_ctx as ctx:
                    gpu_wait += getattr(ctx, "waited", 0.0)
                    gen_eigvals = generalized_eigenvalues_GB(eG_arr, vG, eB_arr, vB)
                gen_sm = spectral_metrics(gen_eigvals) if len(gen_eigvals) >= 11 else {}
                hook_results["gen_GB"] = {"eigvals": gen_eigvals, **gen_sm}

        if hook_results:
            step_results[hook_name] = hook_results
            if verbose:
                print(f"    {hook_name} ({', '.join(factors)}): {time.time()-t_hook:.1f}s")

    if verbose:
        print(f"    metrics total: {time.time()-t0:.1f}s ({len(step_results)} hooks)")
        print(decomp_profiler.summary())
        decomp_profiler.disable()
    if gpu_wait > 0:
        step_results["__gpu_wait__"] = gpu_wait
    return step_results



# ---------------------------------------------------------------------------
# Per-file metric computation (called by workers)
# ---------------------------------------------------------------------------

def _compute_metrics_for_file(args):
    """Pool worker: compute metrics for a .pt file (no model loading)."""
    step, path = args
    data = torch.load(path, map_location="cpu", weights_only=False)
    return step, compute_metrics_for_checkpoint(DataAccessor(data))


def _needs_model_loading(data_path):
    """Check if computing full metrics for this file would require model weights."""
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    for hook_name, entry in data.items():
        if isinstance(entry, dict) and re.match(r"blk\d+\.(up|down|gate)", hook_name):
            has_A_cov = "A" in entry or "A_eigvecs" in entry
            has_B = "B_eigvals" in entry or "B" in entry or "B_eigvecs" in entry
            if has_A_cov and not has_B:
                return True
    if "before_final_norm" in data and "after_final_norm" not in data:
        entry = data.get("before_final_norm", {})
        t = entry.get("A")
        if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
            return True
    return False


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_step_files(data_dir: str) -> dict:
    """Return {step: path} for step files (.pt)."""
    step_files = {}
    for fname in os.listdir(data_dir):
        m = _STEP_RE.match(fname)
        if m:
            step = int(m.group(1))
            step_files[step] = os.path.join(data_dir, fname)
    return step_files

def _resolve_data_root(data_root: str, config_directory: str):
    config_directory = os.path.normpath(config_directory)
    parent = os.path.dirname(config_directory)
    if parent:
        return parent, os.path.basename(config_directory)
    return data_root, config_directory

# ---------------------------------------------------------------------------
# Sequential derive mode (prefetch/delete like collect.py)
# ---------------------------------------------------------------------------

def _compute_derive(to_compute, model_name, res_dict, results_path,
                     keep_cached=False, num_workers=2):
    """Process checkpoints with selective weight loading, threaded for parallelism.

    GPU operations (eigendecomposition, B derivation) are serialized via gpu_lock.
    CPU-heavy work (spectral metrics, K-FAC metrics) runs in parallel across threads.
    Disk usage bounded: at most num_workers + 2 checkpoints on disk at any time.
    """
    import gc
    import queue
    import threading
    import traceback
    from utils.model_registry import (
        get_model_config, get_checkpoint_schedule,
        prefetch_checkpoint, delete_cached_revision,
    )
    from utils.accessor import _resolve_hf_name

    hf_name = _resolve_hf_name(model_name)
    config = get_model_config(hf_name)
    schedule = {s: (r, m) for s, r, m in get_checkpoint_schedule(config, None)}

    # Resolve metadata for all items upfront (from schedule; avoid loading full files)
    items = []
    for step, path in sorted(to_compute, key=lambda x: x[0]):
        if step in schedule:
            rev, step_model = schedule[step]
        else:
            # Fall back to loading the file only if not in schedule
            meta = torch.load(path, map_location="cpu", weights_only=False)
            rev = meta.get("__revision__")
            step_model = meta.get("__hf_model__") or config.hf_repo
        if step_model is None:
            step_model = config.hf_repo
        items.append((step, path, rev, step_model))

    gpu_lock = threading.Lock()
    results_lock = threading.Lock()
    disk_sem = threading.Semaphore(num_workers + 2)
    work_queue = queue.Queue()

    # Per-thread timing stats
    stats_lock = threading.Lock()
    worker_stats = []  # list of (total_time, gpu_wait, download_wait, n_checkpoints)

    # Timed lock/queue helpers
    class _TimedLock:
        """Wraps a lock to track cumulative wait time."""
        def __init__(self, lock):
            self._lock = lock
            self._local = threading.local()
        def __enter__(self):
            import time
            t0 = time.monotonic()
            self._lock.acquire()
            self._local.waited = time.monotonic() - t0
            return self
        def __exit__(self, *args):
            self._lock.release()
        @property
        def waited(self):
            return getattr(self._local, "waited", 0.0)

    timed_gpu = _TimedLock(gpu_lock)

    # Producer: download checkpoints, respecting disk budget
    def producer():
        for item in items:
            disk_sem.acquire()
            prefetch_checkpoint(item[3], item[2])
            work_queue.put(item)
        for _ in range(num_workers):
            work_queue.put(None)

    # Consumer: process checkpoints from queue
    def worker():
        import time
        total_gpu_wait = 0.0
        total_dl_wait = 0.0
        n_done = 0
        t_start = time.monotonic()

        while True:
            t0 = time.monotonic()
            item = work_queue.get()
            dl_wait = time.monotonic() - t0
            if item is None:
                break
            total_dl_wait += dl_wait
            step, path, rev, step_model = item
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                acc = DataAccessor(data, model_name=hf_name, revision=rev)
                acc._hf_repo = step_model

                step_results = compute_metrics_for_checkpoint(
                    acc, gpu_lock=timed_gpu)
                total_gpu_wait += step_results.pop("__gpu_wait__", 0.0)

                with results_lock:
                    res_dict[step] = step_results
                    np.save(results_path, res_dict)

                n_done += 1
                hooks = list(step_results.keys())
                print(
                    f"  Step {step}: {len(hooks)} hook points "
                    f"({', '.join(hooks[:3])}{'...' if len(hooks) > 3 else ''})"
                )
            except Exception as e:
                print(f"Skipping step {step}: {e}")
                traceback.print_exc()
            finally:
                acc = data = None
                gc.collect()
                torch.cuda.empty_cache()
                if not keep_cached:
                    with gpu_lock:
                        delete_cached_revision(step_model, rev)
                disk_sem.release()

        total_time = time.monotonic() - t_start
        with stats_lock:
            worker_stats.append((total_time, total_gpu_wait, total_dl_wait, n_done))

    producer_thread = threading.Thread(target=producer)
    producer_thread.start()

    threads = [threading.Thread(target=worker) for _ in range(num_workers)]
    for w in threads:
        w.start()

    producer_thread.join()
    for w in threads:
        w.join()

    # Print timing summary
    total_wall = sum(s[0] for s in worker_stats)
    total_gpu = sum(s[1] for s in worker_stats)
    total_dl = sum(s[2] for s in worker_stats)
    total_ckpts = sum(s[3] for s in worker_stats)
    if total_wall > 0:
        print(f"\n  Timing ({num_workers} workers, {total_ckpts} checkpoints):")
        print(f"    GPU wait:      {total_gpu:7.1f}s  ({100*total_gpu/total_wall:.1f}% of worker time)")
        print(f"    Download wait: {total_dl:7.1f}s  ({100*total_dl/total_wall:.1f}% of worker time)")
        print(f"    Compute:       {total_wall-total_gpu-total_dl:7.1f}s  ({100*(total_wall-total_gpu-total_dl)/total_wall:.1f}% of worker time)")
        print(f"    Total worker:  {total_wall:7.1f}s  (wall: {max(s[0] for s in worker_stats):.1f}s)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    config_directory: str,
    model_name: str,
    num_workers: int = None,
    recompute: bool = False,
    derive: bool = True,
    keep_cached: bool = False,
    data_root: str = "inferences",
    output_root: str = "results",
):
    """Compute spectral metrics from collected data.

    Args:
        config_directory: Parent directory of model directory (e.g. inferences/rankme_alpha_packed).
        data_root: Parent directory of config_directory. (default: inferences; overwritten by
                   config_directory dirname if config_directory is more than a basename).
        output_root: Directory of all result files. (default: results).
        model_name: Short model name (e.g. pythia-14m-deduped).
        num_workers: Number of parallel workers (default: cpu count).
        recompute: If True, recompute all steps even if results exist.
        derive: If True (default), derive B and post-norm metrics when possible.
                Uses sequential processing with prefetch/delete.
    """

    data_root, config_directory = _resolve_data_root(data_root, config_directory)

    model_dir = os.path.join(data_root, config_directory, model_name)
    if not os.path.isdir(model_dir):
        print(f"No data directory found")
        return

    step_files = discover_step_files(model_dir)
    if not step_files:
        print(f"No step files found in {model_dir}")
        return

    print(f"Input: {model_dir} ({len(step_files)} steps)")

    results_dir = os.path.join(output_root, config_directory)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"results_{model_name}.npy")

    res_dict = _load_existing(results_path)

    to_compute = (
        list(step_files.items())
        if recompute
        else [(s, p) for s, p in step_files.items() if s not in res_dict]
    )

    if not to_compute:
        print(f"All {len(step_files)} steps already have metrics in {results_path}")
        return

    # Auto-detect: does this data need model loading for full metrics?
    sample_path = to_compute[0][1]
    use_derive = derive and _needs_model_loading(sample_path)

    if use_derive:
        if num_workers is None:
            cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
            num_workers = max(1, cpus // 2)
        derive_workers = min(num_workers, len(to_compute))
        print(
            f"Computing metrics for {len(to_compute)}/{len(step_files)} steps "
            f"({derive_workers} threads + selective weight loading + GPU lock)..."
        )
        _compute_derive(to_compute, model_name, res_dict, results_path,
                         keep_cached, num_workers=derive_workers)
    else:
        if num_workers is None:
            num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        print(
            f"Computing metrics for {len(to_compute)}/{len(step_files)} steps "
            f"using {num_workers} workers..."
        )
        with Pool(processes=num_workers) as pool:
            for step, metrics in pool.imap_unordered(_compute_metrics_for_file, to_compute):
                res_dict[step] = metrics
                hooks = list(metrics.keys())
                print(
                    f"  Step {step}: {len(hooks)} hook points "
                    f"({', '.join(hooks[:3])}{'...' if len(hooks) > 3 else ''})"
                )

    np.save(results_path, res_dict)
    print(f"Saved {len(res_dict)} results to {results_path}")


def _load_existing(results_path):
    if os.path.exists(results_path):
        try:
            return np.load(results_path, allow_pickle=True).item()
        except (OSError, ValueError, TypeError):
            pass
    return {}


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from jsonargparse import CLI
    CLI(main)
