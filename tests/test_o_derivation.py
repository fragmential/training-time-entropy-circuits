"""Unit tests for per-OV-head .O factor derivation + the +o/-o format modifier.

Builds synthetic per-OV-head entries (A as a d_head cov) and injects a fake
o_proj weight as _selective_weights so derivation runs without a model load.
"""
from types import SimpleNamespace

import torch

from utils.accessor import (
    DataAccessor,
    _is_ov_head,
    parse_format_spec,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_per_head_acc(block_idx=3, num_heads=4, head_dim=5, d_model=None, seed=0,
                      include_mean=False, store_eigvals=False):
    """Return a DataAccessor with per-OV-head A cov and an injected o_proj weight."""
    torch.manual_seed(seed)
    if d_model is None:
        d_model = num_heads * head_dim
    W_o = torch.randn(d_model, num_heads * head_dim)

    data = {"__format__": "cov"}
    A_covs = {}
    A_means = {}
    for h in range(num_heads):
        X = torch.randn(64, head_dim, dtype=torch.float64)
        cov = X.T @ X  # accumulated (un-normalized) Σ xxT — collection convention
        A_covs[h] = (cov / 64).float()  # E[xxT]
        A_means[h] = X.float().mean(0)
        entry = {"A": cov.float(), "n_A": 64, "n": 64}
        if include_mean:
            entry["A_mean"] = A_means[h]
        if store_eigvals:
            v = torch.linalg.eigvalsh(A_covs[h]).flip(0).clamp(min=0).contiguous()
            entry = {"A_eigvals": v.double(), "n_A": 64, "n": 64}
            if include_mean:
                entry["A_mean"] = A_means[h]
        data[f"blk{block_idx}.attn.head{h}"] = entry

    acc = DataAccessor(data)
    acc._selective_weights = {
        f"blk{block_idx}.attn.head{h}": SimpleNamespace(weight=W_o, bias=None)
        for h in range(num_heads)
    }
    return acc, W_o, A_covs, A_means, block_idx, head_dim


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_is_ov_head_pattern():
    assert _is_ov_head("blk3.attn.head0")
    assert _is_ov_head("blk12.attn.head31")
    assert not _is_ov_head("blk3.attn.in")
    assert not _is_ov_head("blk3.attn.out")
    assert not _is_ov_head("blk3.up")


def test_B_alias_routes_to_O_for_ov_head():
    acc, W_o, A_covs, _, blk, d = _make_per_head_acc()
    hv = acc[f"blk{blk}.attn.head0"]
    # HookView.B and HookView.O return covs that match
    assert torch.allclose(hv.B.cov, hv.O.cov, atol=1e-5)


# ---------------------------------------------------------------------------
# Derivation correctness
# ---------------------------------------------------------------------------

def test_O_cov_equals_W_at_A_W_T():
    acc, W_o, A_covs, _, blk, d = _make_per_head_acc()
    for h in range(len(A_covs)):
        W_h = W_o[:, h * d:(h + 1) * d].float()
        expected = W_h @ A_covs[h] @ W_h.T
        got = acc[f"blk{blk}.attn.head{h}"].O.cov.cpu()
        assert torch.allclose(got, expected, atol=1e-4)


def test_O_eigvals_match_dhead_dual_problem():
    """Non-zero eigvals of W_h A W_h^T equal non-zero eigvals of W_h^T W_h A."""
    acc, W_o, A_covs, _, blk, d = _make_per_head_acc()
    for h in range(len(A_covs)):
        W_h = W_o[:, h * d:(h + 1) * d].float()
        O_eigvals = acc[f"blk{blk}.attn.head{h}"].O.eigvals
        # Dual problem in d_head space (cheaper)
        M = W_h.T @ W_h @ A_covs[h]
        dual_eigvals = torch.linalg.eigvals(M).real
        dual_eigvals, _ = dual_eigvals.sort(descending=True)
        dual_eigvals = dual_eigvals.clamp(min=0)

        top = O_eigvals[:d].cpu()
        assert torch.allclose(top, dual_eigvals, atol=1e-4)
        # Remaining eigvals should all be (essentially) zero
        assert O_eigvals[d:].abs().max().item() < 1e-4


def test_O_mean_equals_W_at_A_mean():
    acc, W_o, _, A_means, blk, d = _make_per_head_acc(include_mean=True)
    for h in range(len(A_means)):
        W_h = W_o[:, h * d:(h + 1) * d].float()
        expected = W_h @ A_means[h]
        got = acc._O_mean(f"blk{blk}.attn.head{h}")
        assert torch.allclose(got, expected, atol=1e-5)


def test_O_derivation_without_weights_returns_none():
    acc, _, _, _, blk, _ = _make_per_head_acc()
    acc._selective_weights = None  # strip access
    # No model and no weights and no model_name → derivation fails gracefully
    assert acc[f"blk{blk}.attn.head0"].O.cov is None


# ---------------------------------------------------------------------------
# +o / -o format modifier
# ---------------------------------------------------------------------------

def test_parse_format_spec_accepts_o():
    base, plus, minus = parse_format_spec("eigenvalues+o")
    assert base == "eigenvalues" and plus == {"o"} and minus == set()
    base, plus, minus = parse_format_spec("eigenvalues-o")
    assert base == "eigenvalues" and plus == set() and minus == {"o"}
    base, plus, minus = parse_format_spec("cov+o-b")
    assert base == "cov" and plus == {"o"} and minus == {"b"}


def test_should_store_o_defaults_per_format():
    acc, *_ = _make_per_head_acc()
    entry = acc._entry("blk3.attn.head0")
    # Defaults: eigenvalues → True, others → False
    assert acc._should_store_o(entry, "eigenvalues", set(), set())
    assert not acc._should_store_o(entry, "cov", set(), set())
    assert not acc._should_store_o(entry, "cov_svd", set(), set())
    assert not acc._should_store_o(entry, "acts", set(), set())
    assert not acc._should_store_o(entry, "acts_svd", set(), set())


def test_should_store_o_explicit_overrides():
    acc, *_ = _make_per_head_acc()
    entry = acc._entry("blk3.attn.head0")
    # +o forces True on cov and cov_svd
    assert acc._should_store_o(entry, "cov", {"o"}, set())
    assert acc._should_store_o(entry, "cov_svd", {"o"}, set())
    # -o forces False on eigenvalues (the only base that defaults True)
    assert not acc._should_store_o(entry, "eigenvalues", set(), {"o"})
    # acts/acts_svd: +o is silently a no-op (O is derived, not raw)
    assert not acc._should_store_o(entry, "acts", {"o"}, set())
    assert not acc._should_store_o(entry, "acts_svd", {"o"}, set())


# ---------------------------------------------------------------------------
# Save round-trip with +o / -o
# ---------------------------------------------------------------------------

def test_save_eigenvalues_default_stores_O_eigvals(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "step.pt"
    acc.save(str(out), format="eigenvalues")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "A_eigvals" in entry
        assert "O_eigvals" in entry


def test_save_eigenvalues_minus_o_omits_O(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "step_no_o.pt"
    acc.save(str(out), format="eigenvalues-o")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "A_eigvals" in entry
        assert "O_eigvals" not in entry
        assert "O" not in entry


def test_save_cov_plus_o_stores_O_cov(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "cov_plus_o.pt"
    acc.save(str(out), format="cov+o")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "A" in entry
        assert "O" in entry  # full d_model x d_model cov stored


def test_save_cov_default_no_O(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "cov_only.pt"
    acc.save(str(out), format="cov")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "A" in entry
        assert "O" not in entry
        assert "O_eigvals" not in entry


def test_saved_O_eigvals_match_in_memory(tmp_path):
    acc, W_o, A_covs, _, blk, d = _make_per_head_acc(include_mean=True)
    # In-memory eigvals first
    in_mem = {h: acc[f"blk{blk}.attn.head{h}"].O.eigvals.clone() for h in range(len(A_covs))}
    out = tmp_path / "rt.pt"
    acc.save(str(out), format="eigenvalues")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(len(A_covs)):
        stored = reloaded[f"blk{blk}.attn.head{h}"]["O_eigvals"].float()
        assert torch.allclose(stored, in_mem[h].float(), atol=1e-4)


# ---------------------------------------------------------------------------
# Metadata persistence for weight_cache_patterns
# ---------------------------------------------------------------------------

def test_weight_cache_patterns_persisted(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    acc._weight_cache_patterns = ["*.mlp.down_proj.weight", "!*.attention.dense.weight"]
    out = tmp_path / "p.pt"
    acc.save(str(out), format="cov")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    assert reloaded.get("__weight_cache_patterns__") == [
        "*.mlp.down_proj.weight",
        "!*.attention.dense.weight",
    ]
    # Re-opening surfaces them
    acc2 = DataAccessor(reloaded)
    assert acc2._weight_cache_patterns == [
        "*.mlp.down_proj.weight",
        "!*.attention.dense.weight",
    ]
