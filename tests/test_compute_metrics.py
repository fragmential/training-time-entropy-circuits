import numpy as np
import torch
from compute_metrics import (
    spectral_metrics, mean_metrics, kfac_metrics,
    generalized_eigenvalues, _top_k_outer_products,
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


# Real collection produces fp64 eigenvalues (fp64 cov -> eigvalsh). The metric
# functions must accept them; the fp32 fixtures above never exercised this.
def test_metrics_accept_fp64_eigvals():
    ev = torch.tensor([10.0 / (i + 1) for i in range(200)], dtype=torch.float64)
    o32 = spectral_metrics(ev.float())
    o64 = spectral_metrics(ev)  # crashed in stringer (float vs double) before the fix
    assert abs(o32["alpha"] - o64["alpha"]) < 1e-3
    assert kfac_metrics(ev, ev * 2.0)["trace"] > 0
    d = 200  # >100 so the alpha-fit window (trange 11..100) is valid
    vec = torch.eye(d, dtype=torch.float64)
    gen = generalized_eigenvalues(ev, vec, ev, vec)  # fp64 -> spectral_metrics
    assert spectral_metrics(gen)["d"] == d


# Per-OV-head slices are d_head-sized (32 < 100); the alpha-fit window must not
# index past the spectrum.
def test_spectral_metrics_small_dim():
    out = spectral_metrics(torch.tensor([10.0 / (i + 1) for i in range(32)], dtype=torch.float64))
    assert out["d"] == 32 and out["rankme"] > 0  # crashed (index OOB) before the fix


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
    e_acts = torch.tensor([10.0, 5.0, 2.0])
    e_grads = torch.tensor([4.0, 3.0])
    out = kfac_metrics(e_acts, e_grads, top_k=10, sample_k=6)
    expected_trace = sum(e_acts) * sum(e_grads)
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
    # grads == acts → generalized eigenvalues should all be 1
    gen = generalized_eigenvalues(eigvals, eigvecs, eigvals, eigvecs)
    assert torch.allclose(gen, torch.ones_like(gen), atol=0.05)


def test_generalized_eigvals_scaled():
    d = 16
    eigvals_acts = torch.tensor([10.0 / (i + 1) for i in range(d)])
    eigvals_grads = eigvals_acts * 3.0
    eigvecs = torch.eye(d)
    gen = generalized_eigenvalues(eigvals_grads, eigvecs, eigvals_acts, eigvecs)
    assert torch.allclose(gen, torch.full_like(gen, 3.0), atol=0.1)
