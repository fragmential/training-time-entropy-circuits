import copy
import torch
from utils.accessor import DataAccessor, reconstruct_cov

# A residual hook so captured signals are value.acts / value.grads and the hook
# shows up in available(). The signal string IS the factor identity now.
HOOK = "after_final_norm"


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
        HOOK: {
            "value.acts_eigvals": eigvals, "value.acts_eigvecs": eigvecs.float(),
            "value.acts_mean": mu, "n_value.acts": 200, "n": 200,
            "value.grads_eigvals": eigvals * 0.5, "value.grads_eigvecs": eigvecs.float(),
            "n_value.grads": 200,
        },
        "__format__": "cov_svd",
    }


def _make_cov_data(d=32, N=200):
    """Synthetic data dict in cov format (unnormalized covariance)."""
    X = torch.randn(N, d, dtype=torch.float64)
    return {
        HOOK: {"value.acts": (X.T @ X).float(), "n_value.acts": N, "n": N},
        "__format__": "cov",
    }


def _make_acts_data(d=32, N=200):
    """Synthetic data dict in acts format."""
    return {
        HOOK: {"value.acts": torch.randn(N, d).float(), "n_value.acts": N, "n": N},
        "__format__": "acts",
    }


def _acts(acc):
    return acc[HOOK]._factor("value.acts")


def test_from_cov_svd_returns_eigvals():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    eigvals = _acts(acc).eigvals
    assert torch.allclose(eigvals, data[HOOK]["value.acts_eigvals"])


def test_from_cov_computes_eigvals():
    data = _make_cov_data(d=16)
    acc = DataAccessor(data)
    eigvals = _acts(acc).eigvals
    assert eigvals is not None
    assert len(eigvals) == 16
    assert (eigvals[:-1] >= eigvals[1:]).all()


def test_from_acts_computes_eigvals():
    data = _make_acts_data(d=16)
    acc = DataAccessor(data)
    eigvals = _acts(acc).eigvals
    assert eigvals is not None
    assert len(eigvals) == 16


def test_cov_reconstruction_from_eigdecomp():
    d = 16
    data = _make_cov_svd_data(d=d)
    acc = DataAccessor(data)
    cov = _acts(acc).cov
    V = data[HOOK]["value.acts_eigvecs"]
    S = data[HOOK]["value.acts_eigvals"]
    expected = V @ torch.diag(S.to(V.dtype)) @ V.T
    assert torch.allclose(cov, expected, atol=1e-4)


def test_centered_eigenvalues():
    d = 16
    data = _make_cov_svd_data(d=d)
    acc = DataAccessor(data)
    centered = _acts(acc).eigvals_centered
    uncentered = _acts(acc).eigvals
    # Centered top eigenvalue should be <= uncentered (mean removed)
    assert centered is not None
    assert centered[0] <= uncentered[0] + 1e-4


def test_hook_names_excludes_metadata():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    names = acc.hook_names()
    assert HOOK in names
    assert "__format__" not in names


def test_available_lists_signals():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    avail = acc.available()
    assert HOOK in avail
    assert "value.acts" in avail[HOOK]


def test_eigvals_descending():
    data = _make_cov_svd_data()
    acc = DataAccessor(data)
    eigvals = _acts(acc).eigvals
    assert (eigvals[:-1] >= eigvals[1:]).all()


def test_eigvecs_access_order_independent():
    # Raw-cov source: eigvecs are computed, not stored. Reading eigvals first must
    # not leave eigvecs as None (the latent access-order bug).
    d = 16
    data = _make_cov_data(d=d)

    acc_vals_first = DataAccessor(copy.deepcopy(data))
    ev = _acts(acc_vals_first).eigvals
    vecs = _acts(acc_vals_first).eigvecs
    assert ev is not None and vecs is not None

    acc_vecs_first = DataAccessor(copy.deepcopy(data))
    vecs2 = _acts(acc_vecs_first).eigvecs
    ev2 = _acts(acc_vecs_first).eigvals
    assert ev2 is not None and vecs2 is not None
    assert torch.allclose(ev, ev2, atol=1e-6)

    assert torch.allclose(reconstruct_cov(ev, vecs), _acts(acc_vals_first).cov, atol=1e-4)


def test_eigvecs_centered_access_order_independent():
    # Meaned raw-cov source: same order-independence for the centered pair, which
    # also exercises the eigvecs-first ordering of FactorView.eigh_centered.
    d = 16
    X = torch.randn(200, d, dtype=torch.float64)
    factors = {HOOK: {"value.acts": (X.T @ X).float(), "n_value.acts": 200, "n": 200,
                      "value.acts_mean": X.float().mean(0)}}

    acc_vals_first = DataAccessor(copy.deepcopy(factors))
    ev = _acts(acc_vals_first).eigvals_centered
    vecs = _acts(acc_vals_first).eigvecs_centered
    assert ev is not None and vecs is not None

    acc_vecs_first = DataAccessor(copy.deepcopy(factors))
    vecs2 = _acts(acc_vecs_first).eigvecs_centered
    ev2 = _acts(acc_vecs_first).eigvals_centered
    assert ev2 is not None and vecs2 is not None
    assert torch.allclose(ev, ev2, atol=1e-6)
