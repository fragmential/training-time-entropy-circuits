import numpy as np
import torch
from compute_metrics import (
    spectral_metrics, mean_metrics, kfac_metrics,
    generalized_eigenvalues_GB, _top_k_outer_products,
)


def test_spectral_metrics_keys():
    eigvals = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1] * 20)  # 120 values
    out = spectral_metrics(eigvals)
    for key in ("rankme", "alpha", "r2", "trace", "log_det", "d", "eigenspectrum"):
        assert key in out


def test_spectral_metrics_alpha(make_powerlaw_eigvals):
    eigvals = make_powerlaw_eigvals(d=500, alpha=1.5)
    out = spectral_metrics(eigvals)
    assert abs(out["alpha"] - 1.5) < 0.2


def test_spectral_metrics_trace():
    # Need >= 100 eigenvalues for stringer_get_powerlaw (uses trange 11-100)
    eigvals = torch.tensor([10.0 / (i + 1) for i in range(200)])
    out = spectral_metrics(eigvals)
    assert abs(out["trace"] - float(eigvals.sum())) < 1e-6


def test_spectral_metrics_rankme_range():
    eigvals = torch.ones(200)  # need >= 100 for stringer_get_powerlaw
    out = spectral_metrics(eigvals)
    assert 1 <= out["rankme"] <= 200 + 1  # small float overshoot OK


# --- mean_metrics ---

def test_mean_metrics_zero_mean():
    d = 16
    out = mean_metrics(torch.eye(d), torch.ones(d), torch.zeros(d))
    assert out["mean_norm"] == 0.0
    assert out["max_overlap"] == 0.0


def test_mean_metrics_aligned_top_eigvec():
    d = 16
    eigvecs = torch.eye(d)
    eigvals = torch.tensor([100.0] + [1.0] * (d - 1))
    mean = eigvecs[:, 0]  # aligned with top eigenvector
    out = mean_metrics(eigvecs, eigvals, mean)
    assert out["max_overlap"] > 0.99
    assert out["max_overlap_idx"] == 0


# --- kfac_metrics ---

def test_kfac_trace_product():
    eA = torch.tensor([10.0, 5.0, 2.0])
    eG = torch.tensor([4.0, 3.0])
    out = kfac_metrics(eA, eG, top_k=10, sample_k=6)
    expected_trace = sum(eA) * sum(eG)
    assert abs(out["trace"] - expected_trace) < 1e-6


def test_top_k_outer_products_largest():
    a = np.array([10.0, 5.0, 2.0])  # _top_k_outer_products takes numpy
    b = np.array([4.0, 3.0])
    top = _top_k_outer_products(a, b, 3)
    assert abs(top[0] - 40.0) < 1e-6  # 10 * 4


def test_top_k_outer_products_count():
    a = np.array([10.0, 5.0, 2.0])
    b = np.array([4.0, 3.0])
    top = _top_k_outer_products(a, b, 4)
    assert len(top) == 4


# --- generalized eigenvalues ---

def test_generalized_eigvals_identity():
    d = 16
    eigvals = torch.tensor([10.0 / (i + 1) for i in range(d)])
    eigvecs = torch.eye(d)
    # G = B → generalized eigenvalues should all be 1
    gen = generalized_eigenvalues_GB(eigvals, eigvecs, eigvals, eigvecs)
    assert torch.allclose(gen, torch.ones_like(gen), atol=0.05)


def test_generalized_eigvals_scaled():
    d = 16
    eigvals_B = torch.tensor([10.0 / (i + 1) for i in range(d)])
    eigvals_G = eigvals_B * 3.0
    eigvecs = torch.eye(d)
    gen = generalized_eigenvalues_GB(eigvals_G, eigvecs, eigvals_B, eigvecs)
    assert torch.allclose(gen, torch.full_like(gen, 3.0), atol=0.1)
