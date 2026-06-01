"""Unit tests for block-boundary hooks, per-OV-head dispatcher, and accessor aliasing."""
import torch
import torch.nn as nn

from utils.hooks import HookCollector, MultiHeadOVDispatcher
from utils.accessor import DataAccessor
from utils import hook_names as hn


def _tensor_key(entry):
    """The single primary tensor key in a collector entry (the captured signal)."""
    keys = [k for k in entry if isinstance(entry[k], torch.Tensor)
            and not k.startswith("n_") and not k.endswith("_mean")]
    assert len(keys) == 1, keys
    return keys[0]


# ---------------------------------------------------------------------------
# Tuple-output unwrap
# ---------------------------------------------------------------------------

class _AttnLike(nn.Module):
    """Module whose forward returns a (tensor, weights) tuple, like HF attention."""
    def forward(self, x):
        return x * 2.0, None


def test_fwd_post_unwraps_tuple_output():
    mod = _AttnLike()
    c = HookCollector(mod, capture="output", mode="cov")
    x = torch.randn(2, 4, 8)
    c.set_token_mask(torch.ones(2, 4, dtype=torch.bool))
    _ = mod(x)
    f = c.factors()
    c.close()
    # 2*x flattened to (8, 8), accumulated as cov
    expected = ((2 * x).reshape(-1, 8).T @ (2 * x).reshape(-1, 8)).double()
    key = _tensor_key(f)
    assert torch.allclose(f[key], expected, atol=1e-3)
    assert f[f"n_{key}"] == 8


# ---------------------------------------------------------------------------
# MultiHeadOVDispatcher
# ---------------------------------------------------------------------------

class _OProjOnly(nn.Module):
    """Tiny model whose forward is just an output projection on a (B,T,H*d) input."""
    def __init__(self, d_model, num_heads, head_dim):
        super().__init__()
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x):
        return self.o_proj(x)


def test_dispatcher_per_head_sum_matches_full_output():
    """Per-head raw acts hold the pre-W_o slice; sum_h (acts @ W_h.T) ≡ o_proj(x)."""
    torch.manual_seed(0)
    d_model, num_heads, head_dim = 12, 3, 4
    model = _OProjOnly(d_model, num_heads, head_dim)
    disp = MultiHeadOVDispatcher(
        model.o_proj, num_heads, head_dim, block_idx=2,
        mode="acts",  # raw slices for direct comparison
    )

    x = torch.randn(2, 5, num_heads * head_dim)
    disp.set_token_mask(torch.ones(2, 5, dtype=torch.bool))
    full_out = model(x)
    factors = disp.factors()
    disp.close()

    W = model.o_proj.weight.detach()
    contribs = []
    for h in range(num_heads):
        entry = factors[f"blk2.attn.head{h}"]
        attended = entry[_tensor_key(entry)]  # (B*T, d_head)
        assert attended.shape == (2 * 5, head_dim)
        Wh = W[:, h * head_dim:(h + 1) * head_dim]
        contribs.append((attended @ Wh.T).reshape(2, 5, d_model))
    assert torch.allclose(sum(contribs), full_out, atol=1e-4)


def test_dispatcher_factors_keys():
    """Per-head cov is now (d_head, d_head), not (d_model, d_model)."""
    d_model, num_heads, head_dim = 8, 4, 2
    model = _OProjOnly(d_model, num_heads, head_dim)
    disp = MultiHeadOVDispatcher(
        model.o_proj, num_heads, head_dim, block_idx=7, mode="cov",
    )
    disp.set_token_mask(torch.ones(1, 3, dtype=torch.bool))
    model(torch.randn(1, 3, num_heads * head_dim))
    f = disp.factors()
    disp.close()
    assert set(f) == {f"blk7.attn.head{h}" for h in range(num_heads)}
    for h in range(num_heads):
        entry = f[f"blk7.attn.head{h}"]
        assert entry[_tensor_key(entry)].shape == (head_dim, head_dim)


def test_dispatcher_cov_matches_manual_outer_product():
    """Per-head cov is the pre-W_o slice's cov: (attended_h^T @ attended_h)."""
    torch.manual_seed(1)
    d_model, num_heads, head_dim = 6, 2, 3
    model = _OProjOnly(d_model, num_heads, head_dim)
    disp = MultiHeadOVDispatcher(
        model.o_proj, num_heads, head_dim, block_idx=0, mode="cov",
    )
    x = torch.randn(1, 4, num_heads * head_dim)
    disp.set_token_mask(torch.ones(1, 4, dtype=torch.bool))
    model(x)
    f = disp.factors()
    disp.close()

    for h in range(num_heads):
        attended_h = x[..., h * head_dim:(h + 1) * head_dim].reshape(-1, head_dim)
        expected = (attended_h.T @ attended_h).double()
        entry = f[f"blk0.attn.head{h}"]
        assert torch.allclose(entry[_tensor_key(entry)], expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Accessor aliases for Pythia parallel-residual layout
# ---------------------------------------------------------------------------

def _block_boundary_data(stored_keys, d=8):
    """Synthetic data dict containing only the listed canonical keys.

    Boundary hooks carry the 'value.acts' captured signal.
    """
    base = {"__format__": "cov"}
    for k in stored_keys:
        X = torch.randn(20, d, dtype=torch.float64)
        base[k] = {"value.acts": (X.T @ X).float(), "n_value.acts": 20, "n": 20}
    return base


def _val(acc, hook):
    return acc[hook]._factor("value.acts")


def test_alias_mlp_in_to_attn_in():
    data = _block_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"])
    acc = DataAccessor(data)
    # mlp.in not stored -> resolves to attn.in via alias
    a = _val(acc, "blk3.mlp.in").cov
    b = _val(acc, "blk3.attn.in").cov
    assert a is not None and b is not None
    assert torch.equal(a, b)


def test_alias_raw_out_to_attn_out():
    data = _block_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"])
    acc = DataAccessor(data)
    a = _val(acc, "blk3.attn.raw_out").cov
    b = _val(acc, "blk3.attn.out").cov
    assert torch.equal(a, b)


def test_alias_not_used_when_canonical_stored():
    # OLMo-2-like layout where both keys are stored separately
    data = _block_boundary_data(["blk3.attn.in", "blk3.mlp.in"])
    acc = DataAccessor(data)
    assert torch.equal(_val(acc, "blk3.mlp.in").cov, _val(acc, "blk3.mlp.in").cov)
    # And they must not be the same object as attn.in
    assert not torch.equal(_val(acc, "blk3.mlp.in").cov, _val(acc, "blk3.attn.in").cov)


def test_alias_does_not_pollute_iteration():
    data = _block_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"])
    acc = DataAccessor(data)
    names = acc.hook_names()
    assert "blk3.mlp.in" not in names
    assert "blk3.attn.raw_out" not in names
    # And available() also only lists stored keys
    avail = acc.available()
    assert "blk3.mlp.in" not in avail
    assert "blk3.attn.raw_out" not in avail


def test_alias_format_convert_does_not_duplicate(tmp_path):
    data = _block_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"])
    out = tmp_path / "converted.pt"
    DataAccessor(data).save(str(out), format="cov_svd+m")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    stored_keys = [k for k in reloaded if not k.startswith("__")]
    assert "blk3.mlp.in" not in stored_keys
    assert "blk3.attn.raw_out" not in stored_keys
    assert sorted(stored_keys) == ["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"]


# ---------------------------------------------------------------------------
# Block-boundary hooks carry no derived (output) signal
# ---------------------------------------------------------------------------

def test_block_boundary_kind_and_signals():
    # Boundary hooks classify as 'boundary' with value.acts/value.grads, no derived.
    for name in ("blk3.attn.in", "blk3.attn.out", "blk3.attn.raw_out",
                 "blk3.mlp.in", "blk3.mlp.out"):
        assert hn.classify(name) == "boundary"
        assert hn.captured_signals(name) == ("value.acts", "value.grads")
        assert hn.derived_signals(name) == {}
    # OV-head has a derived contrib.acts (it owns a weight: an o_proj column block)
    assert hn.classify("blk3.attn.head5") == "ov_head"
    assert "contrib.acts" in hn.derived_signals("blk3.attn.head5")
    # MLP-projection names are mlp with a derived out.acts
    assert hn.classify("blk3.up") == "mlp"
    assert "out.acts" in hn.derived_signals("blk3.up")
    assert hn.derived_signals("blk3.down") == {"out.acts": ("in.acts", "mlp")}


def test_block_boundary_value_acts_only():
    data = _block_boundary_data(["blk3.attn.in"])
    acc = DataAccessor(data)
    a_cov = _val(acc, "blk3.attn.in").cov
    # No derived output signal for boundary hooks; only value.acts carries data.
    assert a_cov is not None
    assert acc["blk3.attn.in"]._factor("value.grads").cov is None


def test_blocks_indexer_resolves_subblocks():
    data = _block_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"])
    acc = DataAccessor(data)
    assert torch.equal(acc.blocks[3].attn.in_.value.acts.cov, _val(acc, "blk3.attn.in").cov)
    assert torch.equal(acc.blocks[3].attn.out.value.acts.cov, _val(acc, "blk3.attn.out").cov)
    assert torch.equal(acc.blocks[3].mlp.out.value.acts.cov, _val(acc, "blk3.mlp.out").cov)
