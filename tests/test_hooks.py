import torch
import torch.nn as nn
from utils.hooks import HookCollector


# Default collector signal keys are "in.acts" (forward) and "out.grads" (backward).

def test_cov_mode_correct():
    X = torch.randn(50, 16, dtype=torch.float32)
    c = HookCollector(module=None, mode="cov")
    c.accumulate(X)
    factors = c.factors()
    expected = (X.T @ X).double()  # accumulation is fp64
    assert torch.allclose(factors["in.acts"], expected, atol=1e-4)


def test_acts_mode_correct():
    X = torch.randn(50, 16)
    c = HookCollector(module=None, mode="acts")
    c.accumulate(X)
    factors = c.factors()
    assert torch.allclose(factors["in.acts"], X, atol=1e-6)


def test_mean_tracking():
    X = torch.randn(50, 16)
    c = HookCollector(module=None, mode="cov", collect_means=True)
    c.accumulate(X)
    factors = c.factors()
    expected_mean = X.float().mean(0)
    assert torch.allclose(factors["in.acts_mean"], expected_mean, atol=1e-5)


def test_grad_accumulation():
    G = torch.randn(50, 16, dtype=torch.float32)
    c = HookCollector(module=None, mode="cov", collect_grad=True)
    c.accumulate_grad(G)
    factors = c.factors()
    expected = (G.T @ G).double()
    assert torch.allclose(factors["out.grads"], expected, atol=1e-4)


def test_custom_signal_keys():
    X = torch.randn(20, 8)
    c = HookCollector(module=None, mode="cov", acts_key="slice.acts", grads_key="slice.grads")
    c.accumulate(X)
    factors = c.factors()
    assert "slice.acts" in factors and "n_slice.acts" in factors


def test_multiple_accumulations_sum():
    X1 = torch.randn(30, 16)
    X2 = torch.randn(20, 16)
    c = HookCollector(module=None, mode="cov")
    c.accumulate(X1)
    c.accumulate(X2)
    factors = c.factors()
    expected = (X1.T @ X1 + X2.T @ X2).double()
    assert torch.allclose(factors["in.acts"], expected, atol=1e-4)


def test_n_count_correct():
    c = HookCollector(module=None, mode="cov")
    c.accumulate(torch.randn(30, 8))
    c.accumulate(torch.randn(20, 8))
    factors = c.factors()
    assert factors["n_in.acts"] == 50
    assert factors["n"] == 50


def test_reset_zeros():
    c = HookCollector(module=None, mode="cov", collect_means=True)
    c.accumulate(torch.randn(30, 8))
    c.reset()
    assert c._n_acts == 0
    # After reset, the accumulator should be zeroed
    c.accumulate(torch.randn(10, 8))
    factors = c.factors()
    assert factors["n_in.acts"] == 10


# --- input-capture hook robustness ---

class _KwargChild(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2


class _KwargParent(nn.Module):
    """Calls its child by KEYWORD, as OLMo-2's decoder calls self_attn."""
    def __init__(self):
        super().__init__()
        self.child = _KwargChild()

    def forward(self, x):
        return self.child(hidden_states=x)


def test_fwd_pre_hook_handles_kwarg_call():
    # A forward-pre-hook on a module invoked with kwargs used to IndexError on
    # inp[0] (positional args empty). It must read the tensor from kwargs.
    m = _KwargParent()
    c = HookCollector(m.child, capture="input", mode="cov")
    c.set_token_mask(torch.ones(2, 4, dtype=torch.bool))
    m(torch.randn(2, 4, 8))
    factors = c.factors()
    c.close()
    assert factors["in.acts"].shape == (8, 8)
    assert factors["n"] == 8


def test_fwd_pre_hook_handles_positional_call():
    m = _KwargParent()
    c = HookCollector(m.child, capture="input", mode="cov")
    c.set_token_mask(torch.ones(2, 4, dtype=torch.bool))
    m.child(torch.randn(2, 4, 8))  # positional
    factors = c.factors()
    c.close()
    assert factors["n"] == 8
