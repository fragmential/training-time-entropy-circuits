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
from functools import reduce
from multiprocessing import Pool
from typing import Callable, overload

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.accessor import DataAccessor, View, Node, Quantity, eigvalsh_descending
from utils.gpu import decomp_profiler, prefer_gpu, run_pipeline
from utils.model_registry import load_inference

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

def spectral_metrics(eigen: torch.Tensor, damping: float = 1e-6, mean: torch.Tensor | None = None) -> dict:
    """Compute RankMe, alpha, R2, log-det from an eigenspectrum (descending, non-negative).

    eigen must be a torch.Tensor (descending, non-negative).
    If `mean` is given, also records `mean_norm` = ||μ||₂.
    """
    eigen = eigen.clamp(min=0).float()  # metric fit runs fp32; eigvals arrive fp64 from fp64 cov
    trace = float(eigen.sum())
    d = len(eigen)
    eps = damping * trace / max(d, 1)
    log_det = float(torch.sum(torch.log(eigen + eps)))

    # Degenerate spectra happen legitimately: nanochat zero-inits c_proj, so attn.out /
    # mlp.out are exactly zero at step 0. Normalization and the powerlaw window both
    # need positive mass — fall back to NaNs instead of crashing the whole checkpoint.
    n_pos = int((eigen > 0).sum())
    if trace > 0:
        eigen = eigen / eigen.sum()  # normalise into sum 1
        rm = rankme_metrics(eigen)
    else:
        rm = {k: float("nan") for k in ("matrix_entropy", "sv_entropy", "rankme", "true_rankme")}
    if n_pos > 12:  # the stringer fit window starts at index 11
        alpha_fit, ypred, fit_r2, fit_r2_100 = stringer_get_powerlaw(eigen, torch.arange(11, min(100, n_pos)))  # window capped to positive-eigval count (stringer drops zeros; O is 128-d but rank d_head)
    else:
        alpha_fit, ypred, fit_r2, fit_r2_100 = float("nan"), None, None, None
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

def _top_k_outer_products(a_t: torch.Tensor, b_t: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k products from outer(a, b) where a, b are sorted descending (numpy internally).

    Uses flat outer product for small dims, priority queue for large.
    """
    a, b = a_t.detach().cpu().numpy(), b_t.detach().cpu().numpy()
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


def _sample_outer_products_linspace(a_t: torch.Tensor, b_t: torch.Tensor, k: int) -> torch.Tensor:
    """Sample k products linearly spaced across the full outer-product distribution
    (descending; numpy internally).

    Small matrices sample exactly; >2M products approximate via sorted-index mapping.
    """
    a, b = a_t.detach().cpu().numpy(), b_t.detach().cpu().numpy()
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


def _histogram_outer_product(a_t: torch.Tensor, b_t: torch.Tensor, bins: int = 1024,
                             chunk_limit: int = 2_000_000) -> dict:
    """Histogram the full K-FAC outer product a ⊗ b in linear and log-spaced bins
    (numpy internally; returns torch tensors).

    Computes in chunks so the full outer product is never materialised.
    Returns density-normalised histograms (integral = 1) plus the edges and
    n_total = |a|*|b| so counts can be recovered as `density * n_total * diff(edges)`.
    """
    a = a_t.detach().cpu().numpy().astype(np.float64)
    b = b_t.detach().cpu().numpy().astype(np.float64)
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
            assert log_counts is not None and log_edges is not None  # has_log ⇒ both set
            log_counts += np.histogram(chunk, bins=log_edges)[0]

    out = {
        "histogram": torch.from_numpy(lin_counts / (total * np.diff(lin_edges))),
        "histogram_edges": torch.from_numpy(lin_edges),
        "n_total": total,
    }
    if has_log:
        assert log_counts is not None and log_edges is not None  # has_log ⇒ both set
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
    vecs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

    if not vecs:
        return eigvalsh_descending(M).cpu()   # M is synthetic; the funnel adds the degenerate-matrix CPU fallback
    lam, U = torch.linalg.eigh(M)                 # U columns are coordinates in the acts eigenbasis
    return lam.flip(0).clamp(min=0).cpu(), U.flip(1).cpu()


# A metric fires at any node where all its operands resolve. `kfac` thus fires at
# projection nodes AND at sub-block nodes (blk.mlp/blk.attn, from their boundary
# .in/.out leaves); `projections_kfac` only at blk.mlp (up.in x down.out).

class _Metric:
    def __init__(self, name: str, operands: dict[str, str], fn: Callable,
                 node_re: str | None = None, external: str | None = None):
        self.name, self.operands, self.fn = name, operands, fn
        self.external = external                 # ctx key merged into args (e.g. "prev"); skip if absent
        self.match: Callable = re.compile(node_re).match if node_re else lambda _ : True


def _gen_metric(a, g):
    e_a, v_a, e_g, v_g = a.eigvals, a.eigvecs, g.eigvals, g.eigvecs   # one shared eigh via the cache
    if any(x is None for x in (e_a, v_a, e_g, v_g)) or v_a.shape != v_g.shape:
        return None  # need both decompositions, in the same space
    gen = generalized_eigenvalues(e_g, v_g, e_a, v_a)
    sm = spectral_metrics(gen) if len(gen) >= 11 else {}
    return {"eigvals": gen, **sm}


def _kfac_metric(a, g):
    ea, eg = a.eigvals, g.eigvals
    if ea is None or eg is None:
        return None
    return kfac_metrics(ea.clamp(min=0), eg.clamp(min=0))

def _blk_mean_metrics(a_in, a_out):
    evecs, evals, mean = a_in.eigvecs_centered, a_in.eigvals_centered, a_out.mean
    if evecs is not None and evals is not None and mean is not None:
        return mean_metrics(evecs, evals, mean)


# --- Block-composition metrics (samples-mode runs; crosses via view.cross). Conventions:
# c_k = .out acts, r_<k = sibling .in acts, r = before_final_norm acts; all crosses centered.
# Every fn returns None when an operand doesn't resolve (e.g. cov-mode runs without samples).

_BLOCK_OUT_RE = re.compile(r"blk(\d+)\.(attn|mlp)\.out$")


def _views(*vs: "View | Node | None") -> tuple[View, ...] | None:
    """The arguments as a tuple, or None unless every one is a View."""
    return vs if all(isinstance(v, View) for v in vs) else None  # type: ignore[return-value]


def _comps(*xs: object) -> list[torch.Tensor] | None:
    """Narrowed tensor components, or None if any is missing — the one guard per metric."""
    return None if any(x is None for x in xs) else [_t(x) for x in xs]


def _c_r(node: Node) -> tuple[View, ...] | None:
    """(c_k, r_<k, r) views for a blk*.{attn,mlp}.out node."""
    return _views(node.get("acts"), node.root.get(node.path.removesuffix(".out") + ".in.acts"),
                  node.root.get("before_final_norm.acts"))


def _tr_fro(eigvals: torch.Tensor) -> tuple[float, float]:
    """(trace, Frobenius norm) of a symmetric matrix from its eigenvalues."""
    return float(eigvals.sum()), float(eigvals.square().sum().sqrt())


def _spectral_entropy(eigvals: torch.Tensor) -> float:
    eigvals = eigvals.clamp(min=0)
    if not float(eigvals.sum()) > 0:
        return 0.0   # zero spectrum: the w→0 limit of w·S is 0, keep it finite
    p = eigvals / eigvals.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def _mixture(lams: list[torch.Tensor], covs: list[torch.Tensor]) -> "tuple[torch.Tensor, torch.Tensor, float] | None":
    """Trace weights w, per-component entropies s, and the mixture entropy S(Σ w_i ρ_i) —
    the shared core of the 2- / 3- / M-component overlap metrics.

    Zero-trace components occur for real (nanochat zero-inits c_proj, so block outputs
    are 0 at step 0): they get weight 0 and entropy 0 (the w→0 limit) and are dropped
    from the mix sum, which would otherwise be 0/0 = NaN. All-zero → None (no mixture)."""
    tr = torch.tensor([float(l.sum()) for l in lams], dtype=torch.float64)
    if not float(tr.sum()) > 0:
        return None
    w = tr / tr.sum()
    s = torch.tensor([_spectral_entropy(l) if t > 0 else 0.0 for l, t in zip(lams, tr)],
                     dtype=torch.float64)
    mix = reduce(torch.add, (float(wk / tk) * C.double()
                             for wk, tk, C in zip(w, tr, covs) if tk > 0))
    return w, s, _spectral_entropy(eigvalsh_descending(mix))


_mean_cos = prefer_gpu(lambda A, B: ((A.float() * B.float()).sum(1)
                                     / (A.float().norm(dim=1) * B.float().norm(dim=1))).mean())


def _block_out_views(root: Node) -> list[tuple[str, View]]:
    """(path, acts view) per blk*.{attn,mlp}.out leaf, ordered by (block, attn|mlp)."""
    def walk(n: Node) -> list[Node]:
        return [n] + [m for ch in n.children() for m in walk(ch)]
    hits = sorted(((m, n) for n in walk(root) if (m := _BLOCK_OUT_RE.match(n.path))),
                  key=lambda t: (int(t[0].group(1)), t[0].group(2)))
    return [(n.path, v) for m, n in hits if isinstance(v := n.get("acts"), View)]


def _block_residual_coupling(node: Node) -> dict | None:
    """All signed/CKA scalars of c_k against r_<k and r.
    P_k = c.cross(r_<k).cov_centered, R_k = c.cross(r).cov_centered; traces + normalizations.

    Returns:
        tr_P:      float  # signed tr(P_k): reinforce(+)/cancel(-) at the write point
        tr_R:      float  # signed tr(R_k), raw absolute units
        tr_Q:      float  # tr(Q_k) = tr(R_k) - tr(P_k) - tr(Σ_ck): coupling to LATER writes
        R_over_ck: float  # tr(R_k)/tr(Σ_ck) — fraction of the block's own energy
        R_over_r:  float  # tr(R_k)/tr(Σ_r)  — block's signed share of the final
        cka_cr:    float  # CKA(c_k, r) ∈ [0,1], unsigned coupling strength
        cos_cr:    float  # mean per-token cos(c_i, r_i) — democratic, unlike the energy-weighted traces
    """
    if not (v := _c_r(node)) or not (g := _comps(v[0].cross(v[1]).cov_centered, v[0].cross(v[2]).cov_centered,
                                                 v[0].eigvals_centered, v[2].eigvals_centered)):
        return None
    P, R, lc, lr = g
    tr_P, tr_R, (tr_c, fro_c), (tr_r, fro_r) = float(P.trace()), float(R.trace()), _tr_fro(lc), _tr_fro(lr)
    if tr_c == 0 or tr_r == 0:
        return None    # zero block output (e.g. zero-init c_proj at step 0) — coupling undefined
    cos = {"cos_cr": float(_mean_cos(*s))} if (s := _comps(v[0].samples, v[2].samples)) else {}
    return {"tr_P": tr_P, "tr_R": tr_R, "tr_Q": tr_R - tr_P - tr_c,
            "R_over_ck": tr_R / tr_c, "R_over_r": tr_R / tr_r,
            "cka_cr": float(R.square().sum()) / (fro_c * fro_r), **cos}


def _ablation_contribution(node: Node) -> dict | None:
    """Leave-one-out effect of block k on the final RankMe: assemble
    Σ_{r\\k} = Σ_r - R_k - R_kᵀ + Σ_ck (all centered), eigendecompose, RankMe vs the final's.

    Returns:
        rankme_ablated: float  # RankMe(Σ_{r\\k})
        delta_rankme:   float  # RankMe(Σ_r) - RankMe(Σ_{r\\k})
    """
    if not (v := _c_r(node)) or not (g := _comps(v[0].cross(v[2]).cov_centered, v[0].cov_centered,
                                                 v[2].cov_centered, v[2].eigvals_centered)):
        return None
    R, Sc, Sr, lr = g
    ablated = rankme_metrics(eigvalsh_descending(Sr - R - R.T + Sc))["rankme"]
    return {"rankme_ablated": ablated, "delta_rankme": rankme_metrics(lr.clamp(min=0))["rankme"] - ablated}


def _eigendirection_attribution(node: Node) -> dict | None:
    """Per-direction share v_iᵀ R_k v_i of how block k feeds each final eigendirection —
    the diagonal of R_k rotated into Σ_r's centered eigenbasis. Sums over k to λ_i of Σ_r.

    Returns:
        contrib: torch.Tensor  # (d,) v_iᵀ R_k v_i, ordered by Σ_r eigenrank
    """
    if not (v := _c_r(node)) or not (g := _comps(v[0].cross(v[2]).cov_centered, v[2].eigvecs_centered)):
        return None
    return {"contrib": prefer_gpu(lambda R, V: ((R @ V) * V).sum(0))(*g)}


def _gen_block_vs_residual(node: Node) -> dict | None:
    """Generalized eigenvalues Σ_ck v = λ Σ_r v: directions the block emphasizes that the
    stream doesn't (large λ) and vice-versa. Reuses generalized_eigenvalues.

    Returns:
        eigvals: torch.Tensor  # (d,) generalized spectrum, descending
        **spectral_metrics(eigvals)
    """
    return _gen_pair(v[0], v[2]) if (v := _c_r(node)) else None


def _gen_pair(a: View, b: View, vecs: bool = False) -> dict | None:
    """Generalized eigenvalues Σ_a v = λ Σ_b v (centered) + spectral family. With vecs, also
    tail_centroid: mean spectral-rank centroid of the top-8 generalized eigenvectors' energy in
    b's eigenbasis — where a's excess structure lives in b's spectrum (RQ2 tail locality)."""
    if not (g := _comps(a.eigvals_centered, a.eigvecs_centered, b.eigvals_centered, b.eigvecs_centered)):
        return None
    ea, va, eb, vb = g
    gen = generalized_eigenvalues(ea, va, eb, vb, vecs=vecs)
    gen, U = gen if isinstance(gen, tuple) else (gen, None)
    out = {"eigvals": gen, **(spectral_metrics(gen) if len(gen) >= 11 else {})}
    if U is not None:
        idx = torch.arange(U.shape[0], dtype=torch.float64)
        out["tail_centroid"] = float((U[:, :8].double().square() * idx[:, None]).sum(0).mean())
    return out


def _incremental_overlap(node: Node) -> dict | None:
    """Two-component overlap χ_k for this block's addition into the stream: mixture entropy
    of {r_<k, c_k} minus their trace-weighted entropies — the overlap term in this block's
    own RankMe update. NOT the M-component aggregate χ (_overlap_chi); they don't reduce
    into each other.

    Returns:
        chi:      float  # χ_k ∈ [0, H(w)] for this 2-component mix
        h_w:      float  # H(w) = binary entropy of (w, 1-w); the ceiling
        chi_frac: float  # χ_k / H(w) ∈ [0,1], fraction orthogonal at this step
        w:        float  # trace weight of r_<k (scalar here)
        s_in:     float  # spectral entropy of the stream (r_<k)
        s_out:    float  # spectral entropy of the block output (c_k)
        s_mix:    float  # spectral entropy of the interference-free mix (the ledger's I needs it)
    """
    if not (v := _c_r(node)) or not (g := _comps(v[1].eigvals_centered, v[0].eigvals_centered,
                                                 v[1].cov_centered, v[0].cov_centered)):
        return None
    if (m := _mixture(g[:2], g[2:])) is None:
        return None
    w, s, s_mix = m
    chi, h_w = s_mix - float((w * s).sum()), _spectral_entropy(w)
    return {"chi": chi, "h_w": h_w, "chi_frac": chi / h_w if h_w > 0 else 0.0, "w": float(w[0]),
            "s_in": float(s[0]), "s_out": float(s[1]), "s_mix": s_mix}


def _block_ledger(node: Node) -> dict | None:
    """Exact per-block rank ledger for the step r_next = r_in + attn.out + mlp.out
    (order-independent, so it holds for parallel Pythia and sequential OLmo alike):
    ΔS = χ + Σ_i w_i (S_i − S_in) + I, with I = S(next) − S(mix) the pure interference.
    log RankMe(final) − log RankMe(embeddings) telescopes as Σ_blocks ΔS.

    Returns:
        delta_s:      float         # S(ρ_next) − S(ρ_in), this block's rank-entropy change
        chi:          float         # subspace-overlap term of the {in, attn, mlp} mix
        quality:      float         # Σ_i w_i (S_i − S_in): spectral quality of the writes
        interference: float         # S(ρ_next) − S(ρ_mix): the cross-covariance effect
        w:            torch.Tensor  # (2 or 3,) trace weights, [in, attn?, mlp?] order
        s:            torch.Tensor  # matching spectral entropies, + s_mix / s_next below
        s_mix:        float
        s_next:       float
    """
    k = int(node.path.removeprefix("blk"))
    parts = [node.get("attn.in.acts"), node.get("attn.out.acts"), node.get("mlp.out.acts")]
    r_next = node.root.get(f"blk{k + 1}.attn.in.acts") or node.root.get("before_final_norm.acts")
    views = [v for v in parts if isinstance(v, View)]
    if len(views) < 2 or not isinstance(parts[0], View) or not isinstance(r_next, View) \
            or not (g := _comps(*(v.eigvals_centered for v in views),
                                *(v.cov_centered for v in views), r_next.eigvals_centered)):
        return None
    if (m := _mixture(g[:len(views)], g[len(views):-1])) is None:
        return None
    w, s, s_mix = m
    s_next = _spectral_entropy(g[-1])
    chi = s_mix - float((w * s).sum())
    return {"delta_s": s_next - float(s[0]), "chi": chi, "quality": float((w * (s - s[0])).sum()),
            "interference": s_next - s_mix, "w": w, "s": s, "s_mix": s_mix, "s_next": s_next}


def _overlap_chi(node: Node) -> dict | None:
    """Model-wide AGGREGATE subspace-overlap χ of all block outputs (M-component mixture
    entropy minus variance-weighted block entropies), over the root's blk*.{attn,mlp}.out
    leaves. Distinct from the per-block 2-component χ_k (_incremental_overlap).

    Returns:
        chi:      float         # χ ∈ [0, H(w)]
        h_w:      float         # H(w), the ceiling
        chi_frac: float         # χ/H(w) ∈ [0,1], fraction orthogonal
        w:        torch.Tensor  # (M,) variance shares w_k
    """
    views = _block_out_views(node)
    if len(views) < 2 or not (g := _comps(*(v.eigvals_centered for _, v in views),
                                          *(v.cov_centered for _, v in views))):
        return None
    if (m := _mixture(g[:len(views)], g[len(views):])) is None:
        return None
    w, s, s_mix = m
    chi, h_w = s_mix - float((w * s).sum()), _spectral_entropy(w)
    return {"chi": chi, "h_w": h_w, "chi_frac": chi / h_w if h_w > 0 else 0.0, "w": w}


def _block_block_coupling(node: Node) -> dict | None:
    """Pairwise block↔block reductions over all blk*.{attn,mlp}.out leaves: the all-pairs
    CKA matrix (each Cov(c_j, c_k) materialized transiently, bypassing the resolver cache)
    + signed traces.

    Returns:
        leaves:       list[str]     # (M,) leaf paths, the row/column order
        cka:          torch.Tensor  # (M, M) CKA(c_j, c_k) ∈ [0,1], symmetric
        signed_trace: torch.Tensor  # (M, M) normalized signed tr Cov(c_j, c_k) ∈ [-1,1]
        mean_cos:     torch.Tensor  # (M, M) mean per-token cos(c_j, c_k) — democratic counterpart
    """
    views = _block_out_views(node)
    comps = [(v.samples, v.mean, v.eigvals_centered) for _, v in views]
    if len(views) < 2 or any(x is None for c3 in comps for x in c3):
        return None
    tr, fro = zip(*(_tr_fro(_t(l)) for *_, l in comps))
    if any(t == 0 for t in tr):
        return None    # a zero block output (e.g. zero-init c_proj at step 0) — coupling undefined
    stats = prefer_gpu(lambda Xj, mj, Xk, mk: (
        C := Xj.float().T @ Xk.float() / Xj.shape[0] - torch.outer(mj, mk),
        C.trace(), C.square().sum(),
        ((Xj.float() * Xk.float()).sum(1) / (Xj.float().norm(dim=1) * Xk.float().norm(dim=1))).mean())[1:])
    M = len(views)
    cka, st, mc = torch.eye(M), torch.eye(M), torch.eye(M)
    for j in range(M):
        for k in range(j + 1, M):
            trC, froC2, cos = stats(comps[j][0], comps[j][1], comps[k][0], comps[k][1])
            cka[j, k] = cka[k, j] = float(froC2) / (fro[j] * fro[k])
            st[j, k] = st[k, j] = float(trC) / (tr[j] * tr[k]) ** 0.5
            mc[j, k] = mc[k, j] = float(cos)
    return {"leaves": [p for p, _ in views], "cka": cka, "signed_trace": st, "mean_cos": mc}


def _mean_migration(node: Node, m: int = 32) -> dict | None:
    """Centered→uncentered migration: subspace overlap between the centered-tail and
    uncentered-top eigenvectors of one leaf (needs both eigvec sets, dropped from results).

    Returns:
        migration: float  # ‖V_bot_centeredᵀ V_top_uncentered‖_F² / m ∈ [0,1]
    """
    if not (v := _views(node.get("acts"))) or not (g := _comps(v[0].eigvecs_centered, v[0].eigvecs)):
        return None
    Vc, Vu = g
    m = min(m, Vu.shape[1])
    return {"migration": float((Vc[:, -m:].T @ Vu[:, :m]).square().sum() / m)}


def _drift_pair(node: Node, prev: Node) -> tuple[View, ...] | None:
    """(now, prev) acts views for the same leaf across two checkpoints."""
    return _views(node.get("acts"), prev.get(f"{node.path}.acts"))


def _cka_drift(node: Node, prev: Node) -> dict | None:
    """CKA of this leaf's acts now vs the previous checkpoint over SHARED tokens (every
    checkpoint sweeps identical batches; the crossed n guards alignment).

    Returns:
        cka_drift: float  # CKA(c^{(t)}, c^{(t-1)}) ∈ [0,1]; 1 = unchanged
    """
    if not (v := _drift_pair(node, prev)) or not (g := _comps(v[0].cross(v[1]).cov_centered,
                                                              v[0].eigvals_centered, v[1].eigvals_centered)):
        return None
    C, lv, lp = g
    return {"cka_drift": float(C.square().sum()) / (_tr_fro(lv)[1] * _tr_fro(lp)[1])}


def _geneig_drift(node: Node, prev: Node) -> dict | None:
    """Generalized eigenvalues Σ^{(t)} v = λ Σ^{(t-1)} v (centered): directions gained
    (large λ) / lost (small λ) between checkpoints. Autocovariance-only.

    Returns:
        eigvals: torch.Tensor  # (d,) drift spectrum, descending
        **spectral_metrics(eigvals)
    """
    return _gen_pair(v[0], v[1]) if (v := _drift_pair(node, prev)) else None


def _gen_vs_ref(node: Node, ref: Node) -> dict | None:
    """Generalized eigenvalues Σ_self v = λ Σ_ref v per quantity, against the same leaf of an
    external reference run (--ref: e.g. task population vs general population)."""
    out = {q: r for q in ("acts", "grads")
           if (v := _views(node.get(q), ref.get(f"{node.path}.{q}"))) and (r := _gen_pair(*v, vecs=True))}
    return out or None


METRICS = [
    _Metric("gen",              {"a": "acts", "g": "grads"},                 _gen_metric),
    _Metric("kfac",             {"a": "in.acts", "g": "out.grads"},          _kfac_metric),
    _Metric("projections_kfac", {"a": "up.in.acts", "g": "down.out.grads"},  _kfac_metric),
    _Metric("mean_metrics_blk_vs_res", {"a_in": "in.acts", "a_out": "out.acts"},  _blk_mean_metrics,
            node_re=r"blk\d+\.(attn|mlp)$"),  # residual sub-block nodes only, not projections
    # Block-composition metrics ({"node": ""} = the node itself); drift ones need ctx["prev"].
    _Metric("cka_drift",               {"node": ""}, _cka_drift,               external="prev"),
    _Metric("geneig_drift",            {"node": ""}, _geneig_drift,            external="prev"),
    _Metric("gen_vs_ref",              {"node": ""}, _gen_vs_ref,              external="ref"),
    _Metric("block_residual_coupling", {"node": ""}, _block_residual_coupling, node_re=r"blk\d+\.(attn|mlp)\.out$"),
    _Metric("ablation_contribution",   {"node": ""}, _ablation_contribution,   node_re=r"blk\d+\.(attn|mlp)\.out$"),
    _Metric("eigendirection_attrib",   {"node": ""}, _eigendirection_attribution, node_re=r"blk\d+\.(attn|mlp)\.out$"),
    _Metric("gen_block_vs_residual",   {"node": ""}, _gen_block_vs_residual,   node_re=r"blk\d+\.(attn|mlp)\.out$"),
    _Metric("incremental_overlap",     {"node": ""}, _incremental_overlap,     node_re=r"blk\d+\.(attn|mlp)\.out$"),
    _Metric("block_ledger",            {"node": ""}, _block_ledger,            node_re=r"blk\d+$"),
    _Metric("mean_migration",          {"node": ""}, _mean_migration,          node_re=r"blk\d+\.(attn|mlp)\.out$"),
    _Metric("overlap_chi",             {"node": ""}, _overlap_chi,             node_re=r"^$"),
    _Metric("block_block_coupling",    {"node": ""}, _block_block_coupling,    node_re=r"^$"),
]


def _t(x: object) -> torch.Tensor:
    """Narrow an atomic component to the Tensor it always is here (eigvals/vecs/mean are never scalars)."""
    assert isinstance(x, torch.Tensor), f"expected a Tensor component, got {type(x).__name__}"
    return x


def _quantity_metrics(q: Quantity, fv: View, path: str) -> dict:
    """Spectral family for one (leaf, quantity): uncentered/centered/mean/cross."""
    out = {}
    ev = fv.eigvals
    if ev is None:
        return out
    ev = _t(ev)
    mu = _t(fv.mean) if fv.mean is not None else None
    _check_negative_eigenvalues(ev, f"{path}.{q}")
    out[f"{q}_uncentered"] = spectral_metrics(ev, mean=mu)

    if mu is not None:
        out[f"{q}_mean_vec"] = mu   # raw μ (d,) for cross-leaf cosine analysis

    cvec, cval = fv.eigvecs_centered, fv.eigvals_centered
    if cval is not None:
        out[f"{q}_centered"] = spectral_metrics(_t(cval))

    if all(x is not None for x in (mu, cvec, cval)):
        out[f"{q}_mean_metrics"] = mean_metrics(_t(cvec), _t(cval), _t(mu))

    # persisted cross-checkpoint projections of this same (leaf, quantity): each is the
    # rotated (uncentered) covariance, so only its uncentered spectrum is meaningful.
    isource, ileaf, iq = fv.identity
    for proj in fv.projections():
        if proj.reference is None:
            continue
        psource, pleaf, pq = proj.reference.identity
        if psource == isource or pleaf != ileaf or pq != iq:
            continue                                  # only cross-checkpoint, same leaf+quantity
        if proj.eigvals is not None:
            out[f"{q}_cross_{'-'.join(str(x) for x in psource)}_uncentered"] = spectral_metrics(_t(proj.eigvals))
    return out


def get_metrics(node: Node, results: dict | None = None, ctx: dict[str, Node] | None = None) -> dict:
    """Recursively walk the hook tree; each node writes its own flat result entry.
    `ctx` carries external operands (e.g. "prev" = previous checkpoint's root Node)."""
    results = {} if results is None else results
    for child in node.children():
        get_metrics(child, results, ctx)
    out = {}
    for q in ("acts", "grads"):
        fv = node.get(q)
        if isinstance(fv, View):
            out.update(_quantity_metrics(q, fv, node.path))
    for m in METRICS:
        args = {k: node.get(rel) for k, rel in m.operands.items()}
        if m.external:
            args[m.external] = (ctx or {}).get(m.external)
        if all(v is not None for v in args.values()) and m.match(node.path):
            r = m.fn(**args)
            if r is not None:
                out[m.name] = r
    if out:
        results[node.path] = out
    return results


def compute_metrics_for_checkpoint(accessor: DataAccessor, verbose: bool = False,
                                   ctx: dict[str, Node] | None = None) -> dict:
    import time
    if verbose:
        decomp_profiler.enable()
    t0 = time.time()
    # Warm the (shared) eigendecompositions first; the metric walk is then cache hits. GPU
    # serialization across workers lives in utils.gpu's prefer/require_gpu, not here.
    accessor.prewarm("eigvecs", "eigvecs_centered")
    if verbose:
        print(f"    prewarm: {time.time()-t0:.1f}s")
    results = get_metrics(accessor.v, ctx=ctx)
    if verbose:
        print(f"    metrics: {time.time()-t0:.1f}s ({len(results)} nodes)")
        print(decomp_profiler.summary())
        decomp_profiler.disable()
    return _to_numpy(results)   # numpy boundary: analysis/results never see torch (dict→dict overload)


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
    np.save(results_path, res_dict)  # type: ignore[arg-type]  # dict saved as a 0-d object array (pickle)
    return res_dict


# ---------------------------------------------------------------------------
# Per-file metric computation (called by workers)
# ---------------------------------------------------------------------------

@overload
def _to_numpy(o: dict) -> dict: ...
@overload
def _to_numpy(o: object) -> object: ...
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
    num_workers: int | None = None,
    recompute: bool = False,
    derive: bool = True,
    data_root: str = "data/inferences",
    output_root: str = "data/results",
    ref: str | None = None,
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
        ref: External reference .pt file (or step-file directory, matched by step) for
             gen_vs_ref — generalized eigenvalues of every leaf against the same leaf there.
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
        # half the allocated CPUs: GPU work is serialized, so the extra threads only
        # overlap I/O / CPU walks; more than that just adds RAM pressure from loaded checkpoints.
        num_workers = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1) // 2)
    print(
        f"Computing metrics for {len(to_compute)}/{len(step_files)} steps "
        f"using {num_workers} workers (derive={derive})..."
    )
    # Parallel over checkpoints via the shared pipeline: a producer pre-loads each .pt while
    # `num_workers` threads compute (one CUDA context; GPU work serialized through the funnel's
    # RLock). Weights load lazily and hit the on-disk weight cache automatically.
    import threading
    save_lock = threading.Lock()

    def load(item):
        step, path = item
        return step, load_inference(path, derive)

    ref_files = discover_step_files(ref) if ref and os.path.isdir(ref) else {}

    def work(payload):
        step, (data, config, weights) = payload
        rpath = ref_files.get(step, None if ref_files else ref)
        ctx = {"ref": DataAccessor(rpath).v} if rpath else None
        metrics = compute_metrics_for_checkpoint(DataAccessor(data, config=config, weights=weights), ctx=ctx)
        with save_lock:
            save_step_metrics(results_path, step, metrics)   # saves every step
        print(f"  Step {step}: {len(metrics)} hook points "
              f"({', '.join(list(metrics)[:3])}{'...' if len(metrics) > 3 else ''})", flush=True)

    run_pipeline(to_compute, work, workers=num_workers, prefetch=load)
    print(f"Saved {len(_load_existing(results_path))} results to {results_path}")


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
