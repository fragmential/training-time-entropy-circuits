"""Snapshot tests: store pre-refactor results, compare post-refactor.

Two modes:
    1. Generate snapshots (run once before refactor):
       uv run pytest tests/test_snapshots.py --snapshot-update -v

    2. Compare against snapshots (run after refactor):
       uv run pytest tests/test_snapshots.py -v

Snapshots are stored in tests/snapshots/ as .pt files.

These tests exercise every component that the refactor touches:
    - powerlaw: eigendecomposition, power-law fitting, RankMe
    - storage: save_factors in each format, convert chain
    - accessor: derivation chains (cov_svd → eigvals, cov → eigvals, centered eigvals)
    - compute_metrics: spectral_metrics, mean_metrics, kfac_metrics, generalized_eigenvalues_GB
    - hooks: covariance and acts accumulation
    - data_utils: token masks, labels

No model downloads needed — all synthetic data with fixed seeds.
"""
import os
import pytest
import torch
import numpy as np

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def _snapshot_path(name):
    return os.path.join(SNAPSHOT_DIR, f"{name}.pt")


def _save_snapshot(name, data):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    torch.save(data, _snapshot_path(name))


def _load_snapshot(name):
    path = _snapshot_path(name)
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _compare_or_update(name, current, request):
    """Compare current results against snapshot, or update if --snapshot-update."""
    if request.config.getoption("--snapshot-update", default=False):
        _save_snapshot(name, current)
        pytest.skip("snapshot updated")

    stored = _load_snapshot(name)
    if stored is None:
        _save_snapshot(name, current)
        pytest.skip("snapshot created (first run)")

    for key in current:
        cur, ref = current[key], stored[key]
        if isinstance(cur, (torch.Tensor, np.ndarray)):
            cur_t = torch.as_tensor(cur, dtype=torch.float64)
            ref_t = torch.as_tensor(ref, dtype=torch.float64)
            assert torch.allclose(cur_t, ref_t, atol=1e-6), f"Mismatch in {name}.{key}"
        elif isinstance(cur, float):
            assert abs(cur - ref) < 1e-6, f"Mismatch in {name}.{key}: {cur} vs {ref}"
        elif isinstance(cur, int):
            assert cur == ref, f"Mismatch in {name}.{key}: {cur} vs {ref}"


# ---- Deterministic fixtures (same seed as conftest autouse) ----

def _fixed_seed():
    torch.manual_seed(42)
    np.random.seed(42)


# ---- Snapshot: powerlaw ----

def test_snapshot_eigendecomp(request):
    _fixed_seed()
    from utils.accessor import _eigh, _eigh_full
    C = torch.randn(32, 10, dtype=torch.float64)
    C = C @ C.T + 0.01 * torch.eye(32, dtype=torch.float64)
    vals = _eigh(C, None)
    vals_full, vecs_full = _eigh_full(C)
    _compare_or_update("eigendecomp", {
        "eigvals": vals, "eigvals_full": vals_full, "eigvecs": vecs_full,
    }, request)


def test_snapshot_eigenspectrum(request):
    _fixed_seed()
    from utils.accessor import get_eigenspectrum
    X = torch.randn(200, 32, dtype=torch.float64)
    centered, uncentered = get_eigenspectrum(acts=X)
    _compare_or_update("eigenspectrum", {
        "centered": centered, "uncentered": uncentered,
    }, request)


def test_snapshot_rankme(request):
    _fixed_seed()
    from scripts.compute_metrics import rankme_metrics
    eigvals = torch.tensor([1.0 / (i + 1) ** 1.5 for i in range(200)])
    rm = rankme_metrics(eigvals)
    _compare_or_update("rankme", {
        "rankme": rm["rankme"], "true_rankme": rm["true_rankme"],
        "matrix_entropy": rm["matrix_entropy"], "sv_entropy": rm["sv_entropy"],
    }, request)


def test_snapshot_powerlaw_fit(request):
    _fixed_seed()
    from scripts.compute_metrics import stringer_get_powerlaw
    eigvals = torch.tensor([1.0 / (i + 1) ** 1.5 for i in range(500)])
    alpha, ypred, r2, r2_100 = stringer_get_powerlaw(eigvals, torch.arange(11, 100))
    _compare_or_update("powerlaw_fit", {
        "alpha": alpha, "r2": r2, "r2_100": r2_100,
    }, request)


# ---- Snapshot: storage formats ----

def test_snapshot_storage_cov_svd(request, tmp_path):
    _fixed_seed()
    from utils.accessor import DataAccessor
    d, N = 32, 200
    X = torch.randn(N, d, dtype=torch.float64)
    factors = {"hook0": {"A": X.T @ X, "n_A": N, "n": N, "A_mean": X.float().mean(0)}}
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd+m")
    data = torch.load(path, map_location="cpu", weights_only=False)
    entry = data["hook0"]
    _compare_or_update("storage_cov_svd", {
        "eigvals": entry["A_eigvals"],
        "eigvals_centered": entry.get("A_eigvals_centered", torch.tensor([])),
        "mean": entry["A_mean"],
    }, request)


def test_snapshot_storage_convert_chain(request, tmp_path):
    _fixed_seed()
    from utils.accessor import DataAccessor
    d, N = 32, 200
    X = torch.randn(N, d, dtype=torch.float64)
    factors = {"hook0": {"A": X.T @ X, "n_A": N, "n": N}}

    cov_path = str(tmp_path / "cov.pt")
    DataAccessor(factors).save(cov_path, format="cov")
    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(cov_path).save(svd_path, format="cov_svd")
    eig_path = str(tmp_path / "eig.pt")
    DataAccessor(svd_path).save(eig_path, format="eigenvalues")

    eig_data = torch.load(eig_path, map_location="cpu", weights_only=False)
    _compare_or_update("storage_convert_chain", {
        "eigvals": eig_data["hook0"]["A_eigvals"],
    }, request)


# ---- Snapshot: accessor derivation ----

def test_snapshot_accessor_from_cov(request):
    _fixed_seed()
    from utils.accessor import DataAccessor
    d, N = 32, 200
    X = torch.randn(N, d, dtype=torch.float64)
    cov = (X.T @ X / N).float()
    mu = X.float().mean(0)
    data = {
        "hook0": {"A": cov * N, "n_A": N, "n": N, "A_mean": mu},
        "__format__": "cov",
    }
    acc = DataAccessor(data)
    eigvals = acc["hook0"].A.eigvals
    _compare_or_update("accessor_from_cov", {"eigvals": eigvals}, request)


def test_snapshot_accessor_centered(request):
    _fixed_seed()
    from utils.accessor import DataAccessor
    d = 32
    C = torch.randn(d, d, dtype=torch.float64)
    C = C @ C.T + 0.01 * torch.eye(d, dtype=torch.float64)
    cov = C / 200
    vals, vecs = torch.linalg.eigh(cov)
    eigvals = vals.flip(0).clamp(min=0)
    eigvecs = vecs.flip(1)
    mu = torch.randn(d).float()
    data = {
        "hook0": {
            "A_eigvals": eigvals, "A_eigvecs": eigvecs.float(),
            "A_mean": mu, "n_A": 200, "n": 200,
        },
        "__format__": "cov_svd",
    }
    acc = DataAccessor(data)
    centered = acc["hook0"].A.eigvals_centered
    _compare_or_update("accessor_centered", {"centered": centered}, request)


# ---- Snapshot: compute_metrics ----

def test_snapshot_spectral_metrics(request):
    _fixed_seed()
    from compute_metrics import spectral_metrics
    eigvals = torch.tensor([1.0 / (i + 1) ** 1.5 for i in range(200)])
    out = spectral_metrics(eigvals)
    _compare_or_update("spectral_metrics", {
        "rankme": out["rankme"], "alpha": out["alpha"],
        "r2": out["r2"], "trace": out["trace"], "log_det": out["log_det"],
    }, request)


def test_snapshot_mean_metrics(request):
    _fixed_seed()
    from compute_metrics import mean_metrics
    d = 32
    eigvecs = torch.eye(d)
    eigvals = torch.tensor([100.0 / (i + 1) ** 1.5 for i in range(d)])
    mean = torch.from_numpy(np.random.randn(d))
    out = mean_metrics(eigvecs, eigvals, mean)
    _compare_or_update("mean_metrics", {
        "mean_norm": out["mean_norm"], "max_overlap": out["max_overlap"],
        "pr": out["pr"], "rayleigh": out["rayleigh"],
        "mahalanobis": out["mahalanobis"],
    }, request)


def test_snapshot_kfac_metrics(request):
    _fixed_seed()
    from compute_metrics import kfac_metrics
    eA = torch.tensor([10.0 / (i + 1) ** 1.5 for i in range(32)])
    eG = torch.tensor([5.0 / (i + 1) ** 1.2 for i in range(16)])
    out = kfac_metrics(eA, eG, top_k=100, sample_k=50)
    _compare_or_update("kfac_metrics", {
        "trace": out["trace"], "log_det": out["log_det"],
        "top_eigvals": out["top_eigvals"][:10],
    }, request)


def test_snapshot_generalized_eigvals(request):
    _fixed_seed()
    from compute_metrics import generalized_eigenvalues_GB
    d = 16
    eigvals = torch.tensor([10.0 / (i + 1) for i in range(d)])
    # Use random orthogonal bases
    Q1 = torch.from_numpy(np.linalg.qr(np.random.randn(d, d))[0])
    Q2 = torch.from_numpy(np.linalg.qr(np.random.randn(d, d))[0])
    gen = generalized_eigenvalues_GB(eigvals * 2, Q1, eigvals, Q2)
    _compare_or_update("generalized_eigvals", {"gen": gen}, request)


# ---- Snapshot: hooks ----

def test_snapshot_hook_cov(request):
    _fixed_seed()
    from utils.hooks import HookCollector
    X = torch.randn(100, 16, dtype=torch.float32)
    c = HookCollector(module=None, mode="cov", collect_means=True)
    c.accumulate(X)
    factors = c.factors()
    _compare_or_update("hook_cov", {
        "A": factors["A"], "n": factors["n"], "A_mean": factors["A_mean"],
    }, request)


# ---- Snapshot: data_utils ----

def test_snapshot_token_masks(request):
    _fixed_seed()
    from utils.data_utils import compute_token_mask, compute_labels
    ids = torch.tensor([[10, 20, 99, 30, 99, 40, 50, 60]])
    mask_all = compute_token_mask(ids, token_selection="all")
    mask_last = compute_token_mask(ids, token_selection="last", boundary_token_ids=[99])
    mask_skip = compute_token_mask(ids, token_selection="all", skip_positions=1, boundary_token_ids=[99])
    labels = compute_labels(ids)
    _compare_or_update("token_masks", {
        "mask_all": mask_all, "mask_last": mask_last, "mask_skip": mask_skip,
        "labels": labels,
    }, request)
