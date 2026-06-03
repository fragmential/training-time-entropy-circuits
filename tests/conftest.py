import sys, os
import pytest
import torch
import numpy as np

# Ensure project root is importable (scripts/ is a package, imported as scripts.*)
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)


def pytest_addoption(parser):
    parser.addoption("--snapshot-update", action="store_true", default=False,
                     help="Regenerate snapshot files instead of comparing")


@pytest.fixture(autouse=True)
def deterministic_seed():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def make_cov():
    def _make(d=64, rank=10):
        A = torch.randn(d, rank, dtype=torch.float64)
        return A @ A.T + 0.01 * torch.eye(d, dtype=torch.float64)
    return _make


@pytest.fixture
def make_acts():
    def _make(N=200, d=64):
        return torch.randn(N, d, dtype=torch.float64)
    return _make


@pytest.fixture
def make_powerlaw_eigvals():
    def _make(d=200, alpha=1.5):
        return torch.tensor([1.0 / (i + 1) ** alpha for i in range(d)])
    return _make


# Synthetic residual leaf: quantities are acts / grads at one leaf.
HOOK = "after_final_norm"
ACTS = "acts"
GRADS = "grads"


@pytest.fixture
def make_factors_dict():
    """Factors dict as HookCollector.factors() would produce (uniform {q}_{fmt} keys,
    unnormalized Σxxᵀ at {q}_cov)."""
    def _make(d=64, N=200, with_means=True, with_grad=True):
        acts = torch.randn(N, d, dtype=torch.float64)
        entry = {f"{ACTS}_cov": acts.T @ acts, f"{ACTS}_n": N}  # unnormalized Σ
        if with_means:
            entry[f"{ACTS}_mean"] = acts.float().mean(0)
        if with_grad:
            grads = torch.randn(N, d, dtype=torch.float64)
            entry[f"{GRADS}_cov"] = grads.T @ grads
            entry[f"{GRADS}_n"] = N
            if with_means:
                entry[f"{GRADS}_mean"] = grads.float().mean(0)
        return {HOOK: entry}
    return _make
