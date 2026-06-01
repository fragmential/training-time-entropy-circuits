"""Unit tests for per-OV-head contrib.acts derivation + the +o/-o format modifier.

Builds synthetic per-OV-head entries (slice.acts as a d_head cov) and injects a
fake o_proj weight as _selective_weights so derivation runs without a model load.
"""
from types import SimpleNamespace

import torch

from utils import hook_names as hn
from utils.accessor import (
    DataAccessor,
    parse_format_spec,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_per_head_acc(block_idx=3, num_heads=4, head_dim=5, d_model=None, seed=0,
                      include_mean=False, store_eigvals=False):
    """Return a DataAccessor with per-OV-head slice.acts cov and an injected o_proj weight."""
    torch.manual_seed(seed)
    if d_model is None:
        d_model = num_heads * head_dim
    W_o = torch.randn(d_model, num_heads * head_dim)

    data = {"__format__": "cov"}
    slice_covs = {}
    slice_means = {}
    for h in range(num_heads):
        X = torch.randn(64, head_dim, dtype=torch.float64)
        cov = X.T @ X  # accumulated (un-normalized) Σ xxT — collection convention
        slice_covs[h] = (cov / 64).float()  # E[xxT]
        slice_means[h] = X.float().mean(0)
        entry = {"slice.acts": cov.float(), "n_slice.acts": 64, "n": 64}
        if include_mean:
            entry["slice.acts_mean"] = slice_means[h]
        if store_eigvals:
            v = torch.linalg.eigvalsh(slice_covs[h]).flip(0).clamp(min=0).contiguous()
            entry = {"slice.acts_eigvals": v.double(), "n_slice.acts": 64, "n": 64}
            if include_mean:
                entry["slice.acts_mean"] = slice_means[h]
        data[f"blk{block_idx}.attn.head{h}"] = entry

    acc = DataAccessor(data)
    acc._selective_weights = {
        f"blk{block_idx}.attn.head{h}": SimpleNamespace(weight=W_o, bias=None)
        for h in range(num_heads)
    }
    return acc, W_o, slice_covs, slice_means, block_idx, head_dim


def _slice(acc, hook):
    return acc[hook]._factor("slice.acts")


def _contrib(acc, hook):
    return acc[hook]._factor("contrib.acts")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def test_ov_head_pattern():
    assert hn.ov_head("blk3.attn.head0")
    assert hn.ov_head("blk12.attn.head31")
    assert not hn.ov_head("blk3.attn.in")
    assert not hn.ov_head("blk3.attn.out")
    assert not hn.ov_head("blk3.up")


def test_contrib_signal_derived_for_ov_head():
    acc, W_o, slice_covs, _, blk, d = _make_per_head_acc()
    hv = acc[f"blk{blk}.attn.head0"]
    # The contrib.acts node alias and the explicit signal return matching covs.
    assert torch.allclose(hv.contrib.acts.cov, hv._factor("contrib.acts").cov, atol=1e-5)


# ---------------------------------------------------------------------------
# Derivation correctness
# ---------------------------------------------------------------------------

def test_contrib_cov_equals_W_at_slice_W_T():
    acc, W_o, slice_covs, _, blk, d = _make_per_head_acc()
    for h in range(len(slice_covs)):
        W_h = W_o[:, h * d:(h + 1) * d].float()
        expected = W_h @ slice_covs[h] @ W_h.T
        got = _contrib(acc, f"blk{blk}.attn.head{h}").cov.cpu()
        assert torch.allclose(got, expected, atol=1e-4)


def test_contrib_eigvals_match_dhead_dual_problem():
    """Non-zero eigvals of W_h A W_h^T equal non-zero eigvals of W_h^T W_h A."""
    acc, W_o, slice_covs, _, blk, d = _make_per_head_acc()
    for h in range(len(slice_covs)):
        W_h = W_o[:, h * d:(h + 1) * d].float()
        contrib_eigvals = _contrib(acc, f"blk{blk}.attn.head{h}").eigvals
        # Dual problem in d_head space (cheaper)
        M = W_h.T @ W_h @ slice_covs[h]
        dual_eigvals = torch.linalg.eigvals(M).real
        dual_eigvals, _ = dual_eigvals.sort(descending=True)
        dual_eigvals = dual_eigvals.clamp(min=0)

        top = contrib_eigvals[:d].cpu()
        assert torch.allclose(top, dual_eigvals, atol=1e-4)
        # Remaining eigvals should all be (essentially) zero
        assert contrib_eigvals[d:].abs().max().item() < 1e-4


def test_contrib_mean_equals_W_at_slice_mean():
    acc, W_o, _, slice_means, blk, d = _make_per_head_acc(include_mean=True)
    for h in range(len(slice_means)):
        W_h = W_o[:, h * d:(h + 1) * d].float()
        expected = W_h @ slice_means[h]
        got = acc._derived_mean(f"blk{blk}.attn.head{h}", "contrib.acts")
        assert torch.allclose(got, expected, atol=1e-5)


def test_contrib_derivation_without_weights_returns_none():
    acc, _, _, _, blk, _ = _make_per_head_acc()
    acc._selective_weights = None  # strip access
    # No model and no weights and no model_name → derivation fails gracefully
    assert _contrib(acc, f"blk{blk}.attn.head0").cov is None


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


def _store_contrib(acc, entry, fmt, plus, minus):
    return acc._should_store_derived(entry, "contrib.acts", fmt, plus, minus)


def test_should_store_o_defaults_per_format():
    acc, *_ = _make_per_head_acc()
    entry = acc._entry("blk3.attn.head0")
    # Defaults: eigenvalues → True, others → False
    assert _store_contrib(acc, entry, "eigenvalues", set(), set())
    assert not _store_contrib(acc, entry, "cov", set(), set())
    assert not _store_contrib(acc, entry, "cov_svd", set(), set())
    assert not _store_contrib(acc, entry, "acts", set(), set())
    assert not _store_contrib(acc, entry, "acts_svd", set(), set())


def test_should_store_o_explicit_overrides():
    acc, *_ = _make_per_head_acc()
    entry = acc._entry("blk3.attn.head0")
    # +o forces True on cov and cov_svd
    assert _store_contrib(acc, entry, "cov", {"o"}, set())
    assert _store_contrib(acc, entry, "cov_svd", {"o"}, set())
    # -o forces False on eigenvalues (the only base that defaults True)
    assert not _store_contrib(acc, entry, "eigenvalues", set(), {"o"})
    # acts/acts_svd: +o is silently a no-op (contrib.acts is derived, not raw)
    assert not _store_contrib(acc, entry, "acts", {"o"}, set())
    assert not _store_contrib(acc, entry, "acts_svd", {"o"}, set())


# ---------------------------------------------------------------------------
# Save round-trip with +o / -o
# ---------------------------------------------------------------------------

def test_save_eigenvalues_default_stores_contrib_eigvals(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "step.pt"
    acc.save(str(out), format="eigenvalues")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "slice.acts_eigvals" in entry
        assert "contrib.acts_eigvals" in entry


def test_save_eigenvalues_minus_o_omits_contrib(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "step_no_o.pt"
    acc.save(str(out), format="eigenvalues-o")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "slice.acts_eigvals" in entry
        assert "contrib.acts_eigvals" not in entry
        assert "contrib.acts" not in entry


def test_save_cov_plus_o_stores_contrib_cov(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "cov_plus_o.pt"
    acc.save(str(out), format="cov+o")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "slice.acts" in entry
        assert "contrib.acts" in entry  # full d_model x d_model cov stored


def test_save_cov_default_no_contrib(tmp_path):
    acc, *_ = _make_per_head_acc(include_mean=True)
    out = tmp_path / "cov_only.pt"
    acc.save(str(out), format="cov")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(4):
        entry = reloaded[f"blk3.attn.head{h}"]
        assert "slice.acts" in entry
        assert "contrib.acts" not in entry
        assert "contrib.acts_eigvals" not in entry


def test_saved_contrib_eigvals_match_in_memory(tmp_path):
    acc, W_o, slice_covs, _, blk, d = _make_per_head_acc(include_mean=True)
    # In-memory eigvals first
    in_mem = {h: _contrib(acc, f"blk{blk}.attn.head{h}").eigvals.clone()
              for h in range(len(slice_covs))}
    out = tmp_path / "rt.pt"
    acc.save(str(out), format="eigenvalues")
    reloaded = torch.load(str(out), map_location="cpu", weights_only=False)
    for h in range(len(slice_covs)):
        stored = reloaded[f"blk{blk}.attn.head{h}"]["contrib.acts_eigvals"].float()
        assert torch.allclose(stored, in_mem[h].float(), atol=1e-4)
