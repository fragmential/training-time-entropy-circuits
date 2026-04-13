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

from utils import powerlaw
from utils.accessor import DataAccessor

_STEP_RE = re.compile(r"step(\d+)\.pt$")


# ---------------------------------------------------------------------------
# Spectral metrics from an eigenvalue array
# ---------------------------------------------------------------------------

def spectral_metrics(eigen, damping=1e-6, mean=None) -> dict:
    """Compute RankMe, alpha, R2, log-det from an eigenspectrum (descending, non-negative).

    Accepts numpy array or torch.Tensor.
    If `mean` is given, also records `mean_norm` = ||μ||₂.
    """
    if hasattr(eigen, "numpy"):
        eigen = eigen.numpy()
    eigen = np.maximum(eigen, 0)
    trace = float(np.sum(eigen))
    d = len(eigen)
    eps = damping * trace / max(d, 1)
    log_det = float(np.sum(np.log(eigen + eps)))

    eigen = eigen / np.sum(eigen) # normalise into sum 1
    rm = powerlaw.rankme_metrics(eigen)
    alpha_fit, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11, 100))
    out = {
        "d": d,
        "eigenspectrum": eigen,
        "trace": trace,
        "avg_magnitude": trace / d,
        "log_det": log_det,
        **rm,
        "alpha": alpha_fit,
        "ypred": ypred,
        "r2": fit_r2,
        "r2_100": fit_r2_100,
    }
    if mean is not None:
        if hasattr(mean, "numpy"):
            mean = mean.numpy()
        out["mean_norm"] = float(np.linalg.norm(mean))
    return out


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
            return np.sort(prods)[::-1]
        idx = np.argpartition(prods, -k)[-k:]
        return np.sort(prods[idx])[::-1]

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
        return prods[::-1]
    if total <= 2_000_000:
        prods = np.outer(a, b).ravel()
        prods.sort()
        prods = prods[::-1]
        idx = np.round(np.linspace(0, len(prods) - 1, k)).astype(int)
        return prods[idx]
    # Approximate: map rank -> (i, j) via sorted row-major order
    rank_idx = np.round(np.linspace(0, total - 1, k)).astype(int)
    i_idx = np.minimum(rank_idx // n, m - 1)
    j_idx = np.minimum(rank_idx % n, n - 1)
    sampled = a[i_idx] * b[j_idx]
    return np.sort(sampled)[::-1]


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


def _check_negative_eigenvalues(eigvals: np.ndarray, label: str):
    """Warn if negative eigenvalues are large relative to the matrix scale."""
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
    eigvals_A: np.ndarray,
    eigvals_G: np.ndarray,
    top_k: int = 20000,
    sample_k: int = 1000,
    damping: float = 1e-6,
) -> dict:
    """K-FAC metrics from A and G eigenvalues.

    Returns:
        trace:          trace(G ⊗ A) = trace(G) * trace(A)
        log_det:        damped log-determinant: d_A * logdet(G + ε_G I) + d_G * logdet(A + ε_A I)
                        where ε_X = damping * trace(X) / d_X
        top_eigvals:    top-k exact products λ_A^i * λ_G^j
        sampled_eigvals: sample_k products linearly spaced across full distribution
        + spectral metrics on top_eigvals (rankme, alpha, r2, r2_100)
    """
    d_in, d_out = len(eigvals_A), len(eigvals_G)

    trace_A = float(eigvals_A.sum())
    trace_G = float(eigvals_G.sum())
    trace = trace_A * trace_G

    eps_A = damping * trace_A / max(d_in, 1)
    eps_G = damping * trace_G / max(d_out, 1)
    ld_A = float(np.sum(np.log(eigvals_A + eps_A)))
    ld_G = float(np.sum(np.log(eigvals_G + eps_G)))
    log_det = d_in * ld_G + d_out * ld_A

    k_top = min(top_k, d_in * d_out)
    top_eigvals = _top_k_outer_products(eigvals_A, eigvals_G, k_top)
    top_eigvals = np.maximum(top_eigvals, 0)

    k_samp = min(sample_k, d_in * d_out)
    sampled_eigvals = _sample_outer_products_linspace(eigvals_A, eigvals_G, k_samp)
    sampled_eigvals = np.maximum(sampled_eigvals, 0)

    sm = spectral_metrics(top_eigvals) if len(top_eigvals) >= 11 else {}

    hist = _histogram_outer_product(eigvals_A, eigvals_G, bins=1024)

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
    eigvals_G: np.ndarray,
    eigvecs_G,
    eigvals_B: np.ndarray,
    eigvecs_B,
    eps_factor: float = 1e-6,
) -> np.ndarray:
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
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eG = torch.as_tensor(eigvals_G, dtype=torch.float32, device=device)
    vG = torch.as_tensor(eigvecs_G, dtype=torch.float32, device=device)
    eB = torch.as_tensor(eigvals_B, dtype=torch.float32, device=device)
    vB = torch.as_tensor(eigvecs_B, dtype=torch.float32, device=device)

    eps = max(eB.max().item() * eps_factor, 1e-10)
    lb_inv_sqrt = (eB + eps).rsqrt()          # (d,)
    lg_sqrt = eG.clamp(min=0).sqrt()          # (d,)

    Q = vB.T @ vG                             # (d, d) cross-basis
    C = lb_inv_sqrt.unsqueeze(1) * Q * lg_sqrt.unsqueeze(0)  # (d, d)
    M = C @ C.T                               # symmetric PSD

    import time as _t; _t0 = _t.time()
    gen_eigvals = torch.linalg.eigvalsh(M).flip(0).clamp(min=0).cpu()
    from utils.powerlaw import decomp_profiler; decomp_profiler.log("torch.eigvalsh(gen_GB)", tuple(M.shape), _t.time() - _t0)
    return gen_eigvals.numpy()

def compute_metrics_for_checkpoint(accessor: DataAccessor, verbose: bool = False):
    import time
    from utils.powerlaw import decomp_profiler
    if verbose:
        decomp_profiler.enable()
    avail = accessor.available()
    step_results = {}
    t0 = time.time()

    # Pre-warm: request eigvecs upfront for factors that need generalized eigendecomp
    for hook_name, factors in avail.items():
        if "B" in factors:
            t = time.time()
            accessor._ensure_eigh(hook_name, "G", need_vecs=True)
            accessor._ensure_B_eigh(hook_name, need_vecs=True)
            if verbose:
                print(f"    prewarm {hook_name}: {time.time()-t:.1f}s")

    if verbose:
        print(f"    prewarm total: {time.time()-t0:.1f}s")

    for hook_name, factors in avail.items():
        t_hook = time.time()
        hook_results = {}
        hook = accessor[hook_name]
        entry = accessor._entry(hook_name)

        eA = eG = None  # cache for K-FAC / gen eigen (numpy arrays)
        for factor in factors:
            fv = hook._factor(factor)
            eigvals = fv.eigvals
            if eigvals is None:
                continue
            if hasattr(eigvals, "numpy"):
                eigvals = eigvals.numpy()
            _check_negative_eigenvalues(eigvals, f"{hook_name}.{factor}")
            eigvals = np.maximum(eigvals, 0)
            hook_results[factor] = spectral_metrics(eigvals, mean=fv.mean)
            if factor == "A":
                eA = eigvals
            elif factor == "G":
                eG = eigvals

            # Centered eigenvalues
            centered = fv.eigvals_centered
            if centered is not None:
                if hasattr(centered, "numpy"):
                    centered = centered.numpy()
                centered = np.maximum(centered, 0)
                hook_results[f"{factor}_centered"] = spectral_metrics(centered)

            # Cross-basis eigenvalues stored alongside
            for ek, ev in entry.items():
                if ek.startswith(f"{factor}_cross_eigvals_") and isinstance(ev, torch.Tensor):
                    cross_label = ek[len(f"{factor}_cross_eigvals_"):]
                    cross_eigvals = np.maximum(ev.numpy(), 0)
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

def _compute_sequential_derive(to_compute, model_name, res_dict, results_path, keep_cached=False):
    """Process checkpoints sequentially with selective weight loading + prefetch/delete."""
    from concurrent.futures import ThreadPoolExecutor
    from utils.model_registry import (
        get_model_config, get_checkpoint_schedule,
        prefetch_checkpoint, delete_cached_revision,
    )
    from utils.accessor import _resolve_hf_name

    hf_name = _resolve_hf_name(model_name)
    config = get_model_config(hf_name)
    schedule = {s: (r, m) for s, r, m in get_checkpoint_schedule(config, None)}

    sorted_items = sorted(to_compute, key=lambda x: x[0])
    executor = ThreadPoolExecutor(max_workers=1)

    for idx, (step, path) in enumerate(sorted_items):
        # Prefetch next checkpoint
        prefetch_future = None
        if idx + 1 < len(sorted_items):
            next_step = sorted_items[idx + 1][0]
            if next_step in schedule:
                nr, nm = schedule[next_step]
                prefetch_future = executor.submit(prefetch_checkpoint, nm, nr)

        data = torch.load(path, map_location="cpu", weights_only=False)
        rev = data.get("__revision__")
        hf_model = data.get("__hf_model__")
        if rev is None and step in schedule:
            rev, hf_model = schedule[step]

        acc = DataAccessor(data, model_name=hf_name, revision=rev)
        if hf_model:
            acc._hf_repo = hf_model

        step_results = compute_metrics_for_checkpoint(acc)
        res_dict[step] = step_results
        hooks = list(step_results.keys())
        print(
            f"  Step {step}: {len(hooks)} hook points "
            f"({', '.join(hooks[:3])}{'...' if len(hooks) > 3 else ''})"
        )

        # Save incrementally
        np.save(results_path, res_dict)

        if prefetch_future is not None:
            prefetch_future.result()
        if not keep_cached and rev is not None and hf_model is not None:
            delete_cached_revision(hf_model, rev)

    executor.shutdown(wait=True)

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
        print(
            f"Computing metrics for {len(to_compute)}/{len(step_files)} steps "
            f"(sequential + selective weight loading)..."
        )
        _compute_sequential_derive(to_compute, model_name, res_dict, results_path, keep_cached)
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
