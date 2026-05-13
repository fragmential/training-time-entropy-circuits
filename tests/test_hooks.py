import torch
from utils.hooks import HookCollector


def test_cov_mode_correct():
    X = torch.randn(50, 16, dtype=torch.float32)
    c = HookCollector(module=None, mode="cov")
    c.accumulate(X)
    factors = c.factors()
    expected = (X.T @ X).double()  # accumulation is fp64
    assert torch.allclose(factors["A"], expected, atol=1e-4)


def test_acts_mode_correct():
    X = torch.randn(50, 16)
    c = HookCollector(module=None, mode="acts")
    c.accumulate(X)
    factors = c.factors()
    assert torch.allclose(factors["A"], X, atol=1e-6)


def test_mean_tracking():
    X = torch.randn(50, 16)
    c = HookCollector(module=None, mode="cov", collect_means=True)
    c.accumulate(X)
    factors = c.factors()
    expected_mean = X.float().mean(0)
    assert torch.allclose(factors["A_mean"], expected_mean, atol=1e-5)


def test_grad_accumulation():
    G = torch.randn(50, 16, dtype=torch.float32)
    c = HookCollector(module=None, mode="cov", collect_grad=True)
    c.accumulate_grad(G)
    factors = c.factors()
    expected = (G.T @ G).double()
    assert torch.allclose(factors["G"], expected, atol=1e-4)


def test_multiple_accumulations_sum():
    X1 = torch.randn(30, 16)
    X2 = torch.randn(20, 16)
    c = HookCollector(module=None, mode="cov")
    c.accumulate(X1)
    c.accumulate(X2)
    factors = c.factors()
    expected = (X1.T @ X1 + X2.T @ X2).double()
    assert torch.allclose(factors["A"], expected, atol=1e-4)


def test_n_count_correct():
    c = HookCollector(module=None, mode="cov")
    c.accumulate(torch.randn(30, 8))
    c.accumulate(torch.randn(20, 8))
    factors = c.factors()
    assert factors["n_A"] == 50
    assert factors["n"] == 50


def test_reset_zeros():
    c = HookCollector(module=None, mode="cov", collect_means=True)
    c.accumulate(torch.randn(30, 8))
    c.reset()
    assert c.n_A == 0
    # After reset, A should be zeroed
    c.accumulate(torch.randn(10, 8))
    factors = c.factors()
    assert factors["n_A"] == 10
