import sys, os
import pytest
import torch
import numpy as np

# Ensure project root and scripts/ are importable
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [_root, os.path.join(_root, "scripts")]:
    if p not in sys.path:
        sys.path.insert(0, p)


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


# Synthetic residual-hook entry: signals are value.acts / value.grads.
HOOK = "after_final_norm"
ACTS = "value.acts"
GRADS = "value.grads"


@pytest.fixture
def make_factors_dict():
    """Factors dict as HookCollector.factors() would produce (unnormalized cov)."""
    def _make(d=64, N=200, with_means=True, with_grad=True):
        acts = torch.randn(N, d, dtype=torch.float64)
        entry = {ACTS: acts.T @ acts, f"n_{ACTS}": N, "n": N}  # unnormalized
        if with_means:
            entry[f"{ACTS}_mean"] = acts.float().mean(0)
        if with_grad:
            grads = torch.randn(N, d, dtype=torch.float64)
            entry[GRADS] = grads.T @ grads
            entry[f"n_{GRADS}"] = N
            if with_means:
                entry[f"{GRADS}_mean"] = grads.float().mean(0)
        return {HOOK: entry}
    return _make
