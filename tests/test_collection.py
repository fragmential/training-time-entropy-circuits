"""HookCollector + MultiHeadOVDispatcher + hook-name grammar.

factors() returns {leaf: {q-keyed tensors}} with uniform `{q}_{fmt}` keys:
`acts_cov` (Σxxᵀ) or `acts_samples`, `acts_n`, `acts_mean`; same for grads."""
import torch
import torch.nn as nn

from utils.hooks import HookCollector, MultiHeadOVDispatcher
from utils import hook_names as hn


# --- HookCollector ---

def test_cov_mode_accumulates_sigma():
    X = torch.randn(50, 16, dtype=torch.float32)
    c = HookCollector(module=None, mode="cov")
    c.accumulate(X)
    assert torch.allclose(c.factors()["acts"]["acts_cov"], (X.T @ X).double(), atol=1e-4)


def test_acts_mode_stores_samples():
    X = torch.randn(50, 16)
    c = HookCollector(module=None, mode="acts")
    c.accumulate(X)
    assert torch.allclose(c.factors()["acts"]["acts_samples"], X, atol=1e-6)


def test_mean_tracking():
    X = torch.randn(50, 16)
    c = HookCollector(module=None, mode="cov", collect_means=True)
    c.accumulate(X)
    assert torch.allclose(c.factors()["acts"]["acts_mean"], X.float().mean(0), atol=1e-5)


def test_grad_accumulation():
    G = torch.randn(50, 16, dtype=torch.float32)
    c = HookCollector(module=None, mode="cov", quantities={"grads"}, leaf="grads")
    c.accumulate_grad(G)
    assert torch.allclose(c.factors()["grads"]["grads_cov"], (G.T @ G).double(), atol=1e-4)


def test_single_leaf():
    c = HookCollector(module=None, mode="cov", leaf="blk0.attn.head0.slice")
    c.accumulate(torch.randn(20, 8))
    entry = c.factors()["blk0.attn.head0.slice"]
    assert "acts_cov" in entry and "acts_n" in entry


def test_multiple_accumulations_sum():
    X1, X2 = torch.randn(30, 16), torch.randn(20, 16)
    c = HookCollector(module=None, mode="cov")
    c.accumulate(X1); c.accumulate(X2)
    assert torch.allclose(c.factors()["acts"]["acts_cov"], (X1.T @ X1 + X2.T @ X2).double(), atol=1e-4)


def test_n_count():
    c = HookCollector(module=None, mode="cov")
    c.accumulate(torch.randn(30, 8)); c.accumulate(torch.randn(20, 8))
    assert c.factors()["acts"]["acts_n"] == 50


def test_reset_zeros():
    c = HookCollector(module=None, mode="cov", collect_means=True)
    c.accumulate(torch.randn(30, 8))
    c.reset()
    assert c._n_acts == 0
    c.accumulate(torch.randn(10, 8))
    assert c.factors()["acts"]["acts_n"] == 10


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
    m = _KwargParent()
    c = HookCollector(m.child, capture="input", mode="cov")
    c.set_token_mask(torch.ones(2, 4, dtype=torch.bool))
    m(torch.randn(2, 4, 8))
    f = c.factors(); c.close()
    assert f["acts"]["acts_cov"].shape == (8, 8) and f["acts"]["acts_n"] == 8


def test_fwd_pre_hook_handles_positional_call():
    m = _KwargParent()
    c = HookCollector(m.child, capture="input", mode="cov")
    c.set_token_mask(torch.ones(2, 4, dtype=torch.bool))
    m.child(torch.randn(2, 4, 8))
    f = c.factors(); c.close()
    assert f["acts"]["acts_n"] == 8


class _AttnLike(nn.Module):
    """forward returns a (tensor, weights) tuple, like HF attention."""
    def forward(self, x):
        return x * 2.0, None


def test_fwd_post_unwraps_tuple_output():
    mod = _AttnLike()
    c = HookCollector(mod, capture="output", mode="cov")
    x = torch.randn(2, 4, 8)
    c.set_token_mask(torch.ones(2, 4, dtype=torch.bool))
    mod(x)
    entry = c.factors()["acts"]; c.close()
    expected = ((2 * x).reshape(-1, 8).T @ (2 * x).reshape(-1, 8)).double()
    assert torch.allclose(entry["acts_cov"], expected, atol=1e-3) and entry["acts_n"] == 8


# --- MultiHeadOVDispatcher ---

class _OProjOnly(nn.Module):
    def __init__(self, d_model, num_heads, head_dim):
        super().__init__()
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x):
        return self.o_proj(x)


def test_dispatcher_per_head_sum_matches_output():
    torch.manual_seed(0)
    d_model, num_heads, head_dim = 12, 3, 4
    model = _OProjOnly(d_model, num_heads, head_dim)
    disp = MultiHeadOVDispatcher(model.o_proj, num_heads, head_dim, block_idx=2, mode="acts")
    x = torch.randn(2, 5, num_heads * head_dim)
    disp.set_token_mask(torch.ones(2, 5, dtype=torch.bool))
    full = model(x)
    f = disp.factors(); disp.close()
    W = model.o_proj.weight.detach()
    contribs = []
    for h in range(num_heads):
        attended = f[f"blk2.attn.head{h}.slice"]["acts_samples"]
        assert attended.shape == (2 * 5, head_dim)
        contribs.append((attended @ W[:, h * head_dim:(h + 1) * head_dim].T).reshape(2, 5, d_model))
    assert torch.allclose(sum(contribs), full, atol=1e-4)


def test_dispatcher_slice_leaf_keys_and_shape():
    d_model, num_heads, head_dim = 8, 4, 2
    model = _OProjOnly(d_model, num_heads, head_dim)
    disp = MultiHeadOVDispatcher(model.o_proj, num_heads, head_dim, block_idx=7, mode="cov")
    disp.set_token_mask(torch.ones(1, 3, dtype=torch.bool))
    model(torch.randn(1, 3, num_heads * head_dim))
    f = disp.factors(); disp.close()
    assert set(f) == {f"blk7.attn.head{h}.slice" for h in range(num_heads)}
    for h in range(num_heads):
        assert f[f"blk7.attn.head{h}.slice"]["acts_cov"].shape == (head_dim, head_dim)


def test_dispatcher_cov_matches_manual_outer():
    torch.manual_seed(1)
    d_model, num_heads, head_dim = 6, 2, 3
    model = _OProjOnly(d_model, num_heads, head_dim)
    disp = MultiHeadOVDispatcher(model.o_proj, num_heads, head_dim, block_idx=0, mode="cov")
    x = torch.randn(1, 4, num_heads * head_dim)
    disp.set_token_mask(torch.ones(1, 4, dtype=torch.bool))
    model(x)
    f = disp.factors(); disp.close()
    for h in range(num_heads):
        a = x[..., h * head_dim:(h + 1) * head_dim].reshape(-1, head_dim)
        assert torch.allclose(f[f"blk0.attn.head{h}.slice"]["acts_cov"], (a.T @ a).double(), atol=1e-3)


# --- leaf-name grammar ---

def test_leaf_grammar():
    for name in ("blk3.attn.in", "blk3.attn.out", "blk3.attn.raw_out", "blk3.mlp.in", "blk3.mlp.out"):
        assert hn.boundary(name) is not None
    assert hn.ov_slice("blk3.attn.head5.slice") == (3, 5)
    assert hn.mlp_proj_leaf("blk3.mlp.up.in") == (3, "up", "in")
    assert hn.mlp_proj_leaf("blk3.mlp.down.out") == (3, "down", "out")
    # node forms and boundary/proj cross-matches are NOT leaves
    assert hn.mlp_proj_leaf("blk3.mlp.up") is None
    assert hn.boundary("blk3.mlp.up.in") is None
    assert hn.ov_slice("blk3.attn.head5") is None
