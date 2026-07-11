import numpy as np
import torch
from utils.accessor import _eigh, eigh_descending, get_eigenspectrum
from scripts.compute_metrics import fit_powerlaw, stringer_get_powerlaw, rankme_metrics


def test_eigh_descending_nonneg(make_cov):
    C = make_cov(d=32)
    vals = _eigh(C, None)
    assert (vals >= 0).all()
    assert (vals[:-1] >= vals[1:]).all(), "eigenvalues not descending"


def test_eigh_k_truncation(make_cov):
    C = make_cov(d=32)
    vals = _eigh(C, 10)
    assert len(vals) == 10


def test_eigh_full_reconstruction(make_cov):
    C = make_cov(d=32)
    vals, vecs = eigh_descending(C)
    reconstructed = vecs @ torch.diag(vals.to(vecs.dtype)) @ vecs.T
    assert torch.allclose(reconstructed, C.to(vecs.dtype), atol=1e-4)


def test_eigenspectrum_acts_vs_cov_match(make_acts):
    X = make_acts(N=200, d=32)
    n = X.shape[0]
    cov = X.T @ X / n
    mu = X.mean(0)

    centered_acts, uncentered_acts = get_eigenspectrum(acts=X)
    centered_cov, uncentered_cov = get_eigenspectrum(cov=cov, mu=mu)

    assert torch.allclose(uncentered_acts, uncentered_cov, atol=1e-4)
    assert torch.allclose(centered_acts, centered_cov, atol=1e-4)


def test_eigenspectrum_centered_vs_uncentered(make_acts):
    X = make_acts(N=200, d=32) + 5.0  # nonzero mean
    centered, uncentered = get_eigenspectrum(acts=X)
    # centered should generally have smaller top eigenvalue (mean removed)
    assert centered[0] < uncentered[0]


def test_fit_powerlaw_recovery(make_powerlaw_eigvals):
    eigvals = make_powerlaw_eigvals(d=500, alpha=1.5)
    alpha, x_range, y_pred = fit_powerlaw(eigvals, 11, 100)
    assert abs(alpha - 1.5) < 0.15


def test_stringer_powerlaw_r2(make_powerlaw_eigvals):
    eigvals = make_powerlaw_eigvals(d=500, alpha=1.5)
    alpha, ypred, r2, r2_100 = stringer_get_powerlaw(eigvals, torch.arange(11, 100))
    assert abs(alpha - 1.5) < 0.2
    assert r2 > 0.9


def test_rankme_uniform_high():
    eigvals = torch.ones(100)
    rm = rankme_metrics(eigvals)
    assert rm["rankme"] > 90  # should be close to 100


def test_rankme_peaked_low():
    eigvals = torch.zeros(100)
    eigvals[0] = 1.0
    rm = rankme_metrics(eigvals)
    assert rm["rankme"] < 2  # should be close to 1


def test_rankme_keys():
    rm = rankme_metrics(torch.ones(10))
    for key in ("matrix_entropy", "sv_entropy", "rankme", "true_rankme"):
        assert key in rm
