#!/usr/bin/env python
"""Compute spectral / K-FAC metrics from collected data via DataAccessor.

Results: {step: {node_path: {metric_name: {...}}}}. Leaves get the spectral family
(acts/grads × uncentered/centered/mean/cross); nodes get gen / kfac / projections_kfac
where their operands resolve. The metric list (METRICS) is walked recursively.

Usage:
    python scripts/compute_metrics.py data/inferences/full_limited pythia-14m-deduped
    python scripts/compute_metrics.py fineweb pythia-14m-deduped --recompute
"""

import heapq
import os
import sys
import re
import numpy as np
import torch
from multiprocessing import Pool
from typing import Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.accessor import DataAccessor, decomp_profiler


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
    eigen = eigen.clamp(min=0).float()  # metric fit runs fp32; eigvals arrive fp64 from fp64 cov
    trace = float(eigen.sum())
    d = len(eigen)
    eps = damping * trace / max(d, 1)
    log_det = float(torch.sum(torch.log(eigen + eps)))

    eigen = eigen / eigen.sum()  # normalise into sum 1
    rm = rankme_metrics(eigen)
    alpha_fit, ypred, fit_r2, fit_r2_100 = stringer_get_powerlaw(eigen, torch.arange(11, min(100, int((eigen > 0).sum()))))  # window capped to positive-eigval count (stringer drops zeros; O is 128-d but rank d_head)
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
        top_overlap:         |<μ̂, v_0>|               (overlap with the top eigendirection)
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
            "max_overlap": 0.0, "max_overlap_idx": -1, "top_overlap": 0.0,
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
    top_overlap = float(torch.sqrt(profile[0]))         # |<μ̂, v_0>|, top eigendirection

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
        "top_overlap": top_overlap,
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

def _top_k_outer_products(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k products from outer(a, b) where a, b are sorted descending (numpy internally).

    Uses flat outer product for small dims, priority queue for large.
    """
    a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    m, n = len(a), len(b)
    if m * n <= 2_000_000:
        prods = np.outer(a, b).ravel()
        if len(prods) <= k:
            out = np.sort(prods)[::-1]
        else:
            idx = np.argpartition(prods, -k)[-k:]
            out = np.sort(prods[idx])[::-1]
    else:
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
        out = np.array(result)
    return torch.from_numpy(np.ascontiguousarray(out))


def _sample_outer_products_linspace(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """Sample k products linearly spaced across the full outer-product distribution
    (descending; numpy internally).

    Small matrices sample exactly; >2M products approximate via sorted-index mapping.
    """
    a, b = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    m, n = len(a), len(b)
    total = m * n
    if total <= k:
        prods = np.outer(a, b).ravel()
        prods.sort()
        out = prods[::-1]
    elif total <= 2_000_000:
        prods = np.outer(a, b).ravel()
        prods.sort()
        prods = prods[::-1]
        idx = np.round(np.linspace(0, len(prods) - 1, k)).astype(int)
        out = prods[idx]
    else:
        # Approximate: map rank -> (i, j) via sorted row-major order
        rank_idx = np.round(np.linspace(0, total - 1, k)).astype(int)
        i_idx = np.minimum(rank_idx // n, m - 1)
        j_idx = np.minimum(rank_idx % n, n - 1)
        out = np.sort(a[i_idx] * b[j_idx])[::-1]
    return torch.from_numpy(np.ascontiguousarray(out))


def _histogram_outer_product(a: torch.Tensor, b: torch.Tensor, bins: int = 1024,
                             chunk_limit: int = 2_000_000) -> dict:
    """Histogram the full K-FAC outer product a ⊗ b in linear and log-spaced bins
    (numpy internally; returns torch tensors).

    Computes in chunks so the full outer product is never materialised.
    Returns density-normalised histograms (integral = 1) plus the edges and
    n_total = |a|*|b| so counts can be recovered as `density * n_total * diff(edges)`.
    """
    a = a.detach().cpu().numpy().astype(np.float64)
    b = b.detach().cpu().numpy().astype(np.float64)
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
        "histogram": torch.from_numpy(lin_counts / (total * np.diff(lin_edges))),
        "histogram_edges": torch.from_numpy(lin_edges),
        "n_total": total,
    }
    if has_log:
        out["loghistogram"] = torch.from_numpy(log_counts / (total * np.diff(log_edges)))
        out["loghistogram_edges"] = torch.from_numpy(log_edges)
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
    eigvals_acts: torch.Tensor,
    eigvals_grads: torch.Tensor,
    top_k: int = 20000,
    sample_k: int = 1000,
    damping: float = 1e-6,
) -> dict:
    """K-FAC metrics from the acts and grads eigenvalues (torch tensors).

    Returns:
        trace:          trace(grads ⊗ acts) = trace(grads) * trace(acts)
        log_det:        damped log-determinant: d_acts * logdet(grads + ε I) + d_grads * logdet(acts + ε I)
                        where ε = damping * trace / d
        top_eigvals:    top-k exact products λ_acts^i * λ_grads^j
        sampled_eigvals: sample_k products linearly spaced across full distribution
        + spectral metrics on top_eigvals (rankme, alpha, r2, r2_100)
    """
    d_in, d_out = len(eigvals_acts), len(eigvals_grads)

    trace_acts = float(eigvals_acts.sum())
    trace_grads = float(eigvals_grads.sum())
    trace = trace_acts * trace_grads

    eps_acts = damping * trace_acts / max(d_in, 1)
    eps_grads = damping * trace_grads / max(d_out, 1)
    ld_acts = float(torch.sum(torch.log(eigvals_acts + eps_acts)))
    ld_grads = float(torch.sum(torch.log(eigvals_grads + eps_grads)))
    log_det = d_in * ld_grads + d_out * ld_acts

    k_top = min(top_k, d_in * d_out)
    top_eigvals = _top_k_outer_products(eigvals_acts, eigvals_grads, k_top).clamp(min=0)

    k_samp = min(sample_k, d_in * d_out)
    sampled_eigvals = _sample_outer_products_linspace(eigvals_acts, eigvals_grads, k_samp).clamp(min=0)

    sm = spectral_metrics(top_eigvals) if len(top_eigvals) >= 11 else {}

    hist = _histogram_outer_product(eigvals_acts, eigvals_grads, bins=1024)

    return {
        **sm,
        "trace": trace,
        "trace_acts": trace_acts,
        "trace_grads": trace_grads,
        "log_det": log_det,
        "log_det_acts": ld_acts,
        "log_det_grads": ld_grads,
        "top_eigvals": top_eigvals,
        "sampled_eigvals": sampled_eigvals,
        "d": d_in * d_out,
        "d_acts": d_in,
        "d_grads": d_out,
        **hist,
    }


# ---------------------------------------------------------------------------
# Generalized eigendecomposition: grads w.r.t. acts (same space)
# ---------------------------------------------------------------------------

def generalized_eigenvalues(
    eigvals_grads: torch.Tensor,
    eigvecs_grads: torch.Tensor,
    eigvals_acts: torch.Tensor,
    eigvecs_acts: torch.Tensor,
    eps_factor: float = 1e-6,
) -> torch.Tensor:
    """Generalized eigenvalues of grads w.r.t. acts: solve (grads) v = λ (acts) v.

    Both covariances must be in the same space (d × d). Returns the generalized
    eigenvalues (descending) = ratios v^T grads v / v^T acts v — directions where
    gradient variance is most/least disproportionate to activation variance.

    Formula: eigenvalues of acts^{-1/2} grads acts^{-1/2}.
    Efficient via cross-basis:
        Q = V_acts^T V_grads
        M = Λ_acts^{-1/2} Q Λ_grads Q^T Λ_acts^{-1/2}  (= C C^T, C = Λ_acts^{-1/2} Q Λ_grads^{1/2})
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e_grads = eigvals_grads.to(dtype=torch.float32, device=device)
    v_grads = eigvecs_grads.to(dtype=torch.float32, device=device)
    e_acts = eigvals_acts.to(dtype=torch.float32, device=device)
    v_acts = eigvecs_acts.to(dtype=torch.float32, device=device)

    eps = max(e_acts.max().item() * eps_factor, 1e-10)
    acts_inv_sqrt = (e_acts + eps).rsqrt()        # (d,)
    grads_sqrt = e_grads.clamp(min=0).sqrt()      # (d,)

    Q = v_acts.T @ v_grads                        # (d, d) cross-basis
    C = acts_inv_sqrt.unsqueeze(1) * Q * grads_sqrt.unsqueeze(0)  # (d, d)
    M = C @ C.T                                   # symmetric PSD

    from utils.accessor import eigvalsh_descending
    return eigvalsh_descending(M).cpu()


# A metric fires at any node where all its operands resolve. `kfac` thus fires at
# projection nodes AND at sub-block nodes (blk.mlp/blk.attn, from their boundary
# .in/.out leaves); `projections_kfac` only at blk.mlp (up.in x down.out).

from contextlib import nullcontext


class _Metric:
    def __init__(self, name, operands, fn, node_re=None):
        self.name, self.operands, self.fn = name, operands, fn
        self.match: Callable = re.compile(node_re).match if node_re else lambda _ : True


def _gen_metric(a, g):
    ae, ge = a.eigh, g.eigh
    if ae is None or ge is None:
        return None
    e_a, v_a = ae
    e_g, v_g = ge
    if v_a.shape != v_g.shape:
        return None  # acts and grads must live in the same space
    ctx = getattr(a._acc, "_gpu_ctx", None) or nullcontext()
    with ctx:
        gen = generalized_eigenvalues(e_g, v_g, e_a, v_a)
    sm = spectral_metrics(gen) if len(gen) >= 11 else {}
    return {"eigvals": gen, **sm}


def _kfac_metric(a, g):
    ea, eg = a.eigvals, g.eigvals
    if ea is None or eg is None:
        return None
    return kfac_metrics(ea.clamp(min=0), eg.clamp(min=0))

def _blk_mean_metrics(a_in, a_out):
    if a_out.mean is not None and a_in.eigh_centered is not None:
        return mean_metrics(a_in.eigvecs_centered, a_in.eigvals_centered, a_out.mean)


METRICS = [
    _Metric("gen",              {"a": "acts", "g": "grads"},                 _gen_metric),
    _Metric("kfac",             {"a": "in.acts", "g": "out.grads"},          _kfac_metric),
    _Metric("projections_kfac", {"a": "up.in.acts", "g": "down.out.grads"},  _kfac_metric),
    _Metric("mean_metrics_blk_vs_res", {"a_in": "in.acts", "a_out": "out.acts"},  _blk_mean_metrics,
            node_re=r"blk\d+\.(attn|mlp)$"),  # residual sub-block nodes only, not projections
]


def _quantity_metrics(q, fv, path):
    """Spectral family for one (leaf, quantity): uncentered/centered/mean/cross."""
    out = {}
    ev = fv.eigvals
    if ev is None:
        return out
    _check_negative_eigenvalues(ev, f"{path}.{q}")
    out[f"{q}_uncentered"] = spectral_metrics(ev.clamp(min=0), mean=fv.mean)

    if fv.mean is not None:
        out[f"{q}_mean_vec"] = fv.mean   # raw μ (d,) for cross-leaf cosine analysis

    cev = fv.eigvals_centered
    if cev is not None:
        out[f"{q}_centered"] = spectral_metrics(cev.clamp(min=0))

    if fv.mean is not None and fv.eigh_centered is not None:
        out[f"{q}_mean_metrics"] = mean_metrics(fv.eigvecs_centered, fv.eigvals_centered, fv.mean)

    # TODO(1b): this reaches into accessor internals to surface cross-basis projections;
    # expose them through a proper projection API once general projections are designed.
    entry = fv._acc._entry(path)
    prefix = f"{q}_cross_eigvals_"
    for ek, evv in entry.items():
        if ek.startswith(prefix) and isinstance(evv, torch.Tensor):
            out[f"{q}_cross_{ek[len(prefix):]}"] = spectral_metrics(evv.clamp(min=0))
    return out


def get_metrics(node, results):
    """Recursively walk the hook tree; each node writes its own flat result entry."""
    for child in node.children():
        get_metrics(child, results)
    out = {}
    for q in ("acts", "grads"):
        fv = node.get(q)
        if fv is not None:
            out.update(_quantity_metrics(q, fv, node._path))
    for m in METRICS:
        args = {k: node.get(rel) for k, rel in m.operands.items()}
        if all(v is not None for v in args.values()) and m.match(node._path):
            r = m.fn(**args)
            if r is not None:
                out[m.name] = r
    if out:
        results[node._path] = out


def compute_metrics_for_checkpoint(accessor: DataAccessor, verbose: bool = False, gpu_lock=None):
    import time
    if verbose:
        decomp_profiler.enable()
    gpu_ctx = gpu_lock or nullcontext()
    accessor._gpu_ctx = gpu_ctx
    t0 = time.time()
    gpu_wait = 0.0

    # Pre-warm every eigendecomposition under the GPU lock so the metric walk that
    # follows is pure-CPU cache hits — letting Pool workers overlap GPU and CPU work.
    with gpu_ctx as ctx:
        gpu_wait += getattr(ctx, "waited", 0.0)
        for leaf, q in accessor.leaf_quantities():
            accessor.resolve(leaf, q, "eigh")
            accessor.resolve(leaf, q, "eigh_centered")
    if verbose:
        print(f"    prewarm total: {time.time()-t0:.1f}s")

    results = {}
    get_metrics(accessor.v, results)

    if verbose:
        print(f"    metrics total: {time.time()-t0:.1f}s ({len(results)} nodes)")
        print(decomp_profiler.summary())
        decomp_profiler.disable()
    if gpu_wait > 0:
        results["__gpu_wait__"] = gpu_wait
    return _to_numpy(results)   # provider emits pure numpy; analysis/results never see torch


def _merge_step(old: dict, new: dict) -> dict:
    """Fold `new` into `old` at the (leaf, metric) key level — new wins on conflict,
    keys not recomputed this run are preserved. Lets a re-run add metrics without
    clobbering weight-derived leaves (e.g. head.contrib) absent under --derive false.
    """
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in old.items()}
    for k, v in new.items():
        out[k] = {**out[k], **v} if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def save_step_metrics(results_path: str, step: int, metrics: dict) -> dict:
    """Single owner of results-.npy writes: load existing, merge this step, save.
    Used by both `main` (batch) and collect.py's inline path so they can't diverge."""
    res_dict = _load_existing(results_path)
    res_dict[step] = _merge_step(res_dict.get(step, {}), metrics)
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    np.save(results_path, res_dict)
    return res_dict


# ---------------------------------------------------------------------------
# Per-file metric computation (called by workers)
# ---------------------------------------------------------------------------

def _to_numpy(o):
    """Recursively convert torch tensors to numpy so Pool results pickle by value —
    avoids torch's shared-memory IPC, which exhausts mmaps on big models (d~4096+)."""
    if torch.is_tensor(o):
        return o.detach().cpu().numpy()
    if isinstance(o, dict):
        return {k: _to_numpy(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return type(o)(_to_numpy(v) for v in o)
    return o


def _compute_metrics_for_file(args, derive=True):
    """Pool worker: compute metrics for a .pt file (derive=False skips weight-derived leaves)."""
    step, path = args
    data = torch.load(path, map_location="cpu", weights_only=False)
    return step, compute_metrics_for_checkpoint(DataAccessor(data, derive=derive))


def _needs_model_loading(data_path):
    """Whether full metrics for this file require model weights (derivation)."""
    return DataAccessor(data_path).needs_model_weights()


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
# Main
# ---------------------------------------------------------------------------

def main(
    config_directory: str,
    model_name: str,
    num_workers: int = None,
    recompute: bool = False,
    derive: bool = True,
    data_root: str = "data/inferences",
    output_root: str = "data/results",
):
    """Compute spectral metrics from collected data.

    Args:
        config_directory: Parent directory of model directory (e.g. data/inferences/rankme_alpha_packed).
        data_root: Parent directory of config_directory. (default: data/inferences; overwritten by
                   config_directory dirname if config_directory is more than a basename).
        output_root: Directory of all result files. (default: data/results).
        model_name: Short model name (e.g. pythia-14m-deduped).
        num_workers: Number of parallel workers (default: cpu count).
        recompute: If True, recompute all steps even if results exist.
        derive: If True (default), derive B and post-norm metrics when possible.
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

    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    print(
        f"Computing metrics for {len(to_compute)}/{len(step_files)} steps "
        f"using {num_workers} workers (derive={derive})..."
    )
    # spawn (not fork) when a GPU is present so each worker gets its own CUDA context
    # and the eigh runs on the GPU; fork is fine on CPU-only nodes. Execution model is
    # independent of `derive` (which only gates weight loading inside the worker).
    from functools import partial
    from multiprocessing import get_context
    mp_ctx = get_context("spawn") if os.environ.get("CUDA_VISIBLE_DEVICES") else get_context()
    worker = partial(_compute_metrics_for_file, derive=derive)
    final = res_dict
    with mp_ctx.Pool(processes=num_workers) as pool:
        for step, metrics in pool.imap_unordered(worker, to_compute):
            final = save_step_metrics(results_path, step, metrics)   # saves every step
            hooks = list(metrics.keys())
            print(
                f"  Step {step}: {len(hooks)} hook points "
                f"({', '.join(hooks[:3])}{'...' if len(hooks) > 3 else ''})",
                flush=True,
            )

    print(f"Saved {len(final)} results to {results_path}")


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
