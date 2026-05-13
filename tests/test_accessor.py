import torch
from utils.accessor import DataAccessor


def _make_cov_svd_data(d=32):
    """Synthetic data dict in cov_svd format."""
    C = torch.randn(d, d, dtype=torch.float64)
    C = C @ C.T + 0.01 * torch.eye(d, dtype=torch.float64)
    cov = C / 200  # pretend n=200
    vals, vecs = torch.linalg.eigh(cov)
    eigvals = vals.flip(0).clamp(min=0)
    eigvecs = vecs.flip(1)
    mu = torch.randn(d).float()
    return {
        "hook0": {
            "A_eigvals": eigvals, "A_eigvecs": eigvecs.float(),
            "A_mean": mu, "n_A": 200, "n": 200,
            "G_eigvals": eigvals * 0.5, "G_eigvecs": eigvecs.float(),
            "n_G": 200,
        },
        "__format__": "cov_svd",
    }


def _make_cov_data(d=32, N=200):
    """Synthetic data dict in cov format (unnormalized covariance)."""
    X = torch.randn(N, d, dtype=torch.float64)
    return {
        "hook0": {"A": (X.T @ X).float(), "n_A": N, "n": N},
        "__format__": "cov",
    }


def _make_acts_data(d=32, N=200):
    """Synthetic data dict in acts format."""
    return {
        "hook0": {"A": torch.randn(N, d).float(), "n_A": N, "n": N},
        "__format__": "acts",
    }


def test_from_cov_svd_returns_eigvals():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    eigvals = acc["hook0"].A.eigvals
    assert torch.allclose(eigvals, data["hook0"]["A_eigvals"])


def test_from_cov_computes_eigvals():
    data = _make_cov_data(d=16)
    acc = DataAccessor(data)
    eigvals = acc["hook0"].A.eigvals
    assert eigvals is not None
    assert len(eigvals) == 16
    assert (eigvals[:-1] >= eigvals[1:]).all()


def test_from_acts_computes_eigvals():
    data = _make_acts_data(d=16)
    acc = DataAccessor(data)
    eigvals = acc["hook0"].A.eigvals
    assert eigvals is not None
    assert len(eigvals) == 16


def test_cov_reconstruction_from_eigdecomp():
    d = 16
    data = _make_cov_svd_data(d=d)
    acc = DataAccessor(data)
    cov = acc["hook0"].A.cov
    V = data["hook0"]["A_eigvecs"]
    S = data["hook0"]["A_eigvals"]
    expected = V @ torch.diag(S.to(V.dtype)) @ V.T
    assert torch.allclose(cov, expected, atol=1e-4)


def test_centered_eigenvalues():
    d = 16
    data = _make_cov_svd_data(d=d)
    acc = DataAccessor(data)
    centered = acc["hook0"].A.eigvals_centered
    uncentered = acc["hook0"].A.eigvals
    # Centered top eigenvalue should be <= uncentered (mean removed)
    assert centered is not None
    assert centered[0] <= uncentered[0] + 1e-4


def test_hook_names_excludes_metadata():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    names = acc.hook_names()
    assert "hook0" in names
    assert "__format__" not in names


def test_available_lists_factors():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    avail = acc.available()
    assert "hook0" in avail
    assert "A" in avail["hook0"]


def test_eigvals_descending():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    eigvals = acc["hook0"].A.eigvals
    assert (eigvals[:-1] >= eigvals[1:]).all()
