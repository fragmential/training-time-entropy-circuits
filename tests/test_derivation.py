"""Derivation of acts at non-captured leaves, through ONE resolver path:
  mlp `…out` <- `…in` via the projection weight (+bias for Pythia),
  ov  `…contrib` <- `…slice` via an o_proj head-column slice,
  `after_final_norm` <- `before_final_norm` via the final norm,
plus the +b/+o storage modifiers that materialize derived acts. Weights are
injected as _selective_weights so nothing downloads."""
from types import SimpleNamespace

import torch

from utils.accessor import DataAccessor, parse_format_spec

MODEL = "EleutherAI/pythia-14m"


# --- MLP .out acts = W @ in_cov @ Wᵀ (+ bias correction) ---

def _mlp_acc(d_in=8, d_out=6, bias=True, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(64, d_in, dtype=torch.float64)
    data = {
        "blk0.mlp.up.in": {"acts_cov": X.T @ X, "acts_n": 64, "acts_mean": X.float().mean(0)},
        "blk0.mlp.up.out": {"grads_cov": torch.eye(d_out) * d_out, "grads_n": 64},
        "__format__": "cov", "__hf_model__": MODEL,
    }
    acc = DataAccessor(data, model_name=MODEL)
    W = torch.randn(d_out, d_in)
    b = torch.randn(d_out) if bias else None
    acc._selective_weights = {"gpt_neox.layers.0.mlp.dense_h_to_4h": SimpleNamespace(weight=W, bias=b)}
    return acc, W, b, X


def test_mlp_out_acts_cov_with_bias_correction():
    acc, W, b, X = _mlp_acc(bias=True)
    C = (X.T @ X).float() / 64
    Wm = W @ X.float().mean(0)
    expect = W @ C @ W.T + torch.outer(Wm, b) + torch.outer(b, Wm) + torch.outer(b, b)
    assert torch.allclose(acc.factor("blk0.mlp.up.out", "acts").cov, expect, atol=1e-4)


def test_mlp_out_acts_cov_no_bias():
    acc, W, _, X = _mlp_acc(bias=False)
    C = (X.T @ X).float() / 64
    assert torch.allclose(acc.factor("blk0.mlp.up.out", "acts").cov, W @ C @ W.T, atol=1e-4)


def test_mlp_out_acts_eigvals_descending():
    acc, *_ = _mlp_acc()
    ev = acc.factor("blk0.mlp.up.out", "acts").eigvals
    assert ev is not None and (ev[:-1] >= ev[1:] - 1e-5).all()


def test_derive_false_skips_weight_derivation():
    # up.out acts needs the projection weight; derive=False must skip (no download), not crash
    X = torch.randn(64, 8, dtype=torch.float64)
    data = {
        "blk0.mlp.up.in": {"acts_cov": X.T @ X, "acts_n": 64, "acts_mean": X.float().mean(0)},
        "__format__": "cov", "__hf_model__": MODEL,
    }
    acc = DataAccessor(data, model_name=MODEL, derive=False)
    assert acc._ensure_weights() is False                    # no load attempted
    assert acc.factor("blk0.mlp.up.out", "acts") is None     # weight-derived leaf skipped
    assert acc.factor("blk0.mlp.up.in", "acts") is not None  # captured leaf still resolves


# --- per-OV-head contrib acts = W_h @ slice_cov @ W_hᵀ (head column slice) ---

def _ov_acc(block_idx=3, num_heads=4, head_dim=5, seed=0, include_mean=False):
    torch.manual_seed(seed)
    d_model = num_heads * head_dim
    W_o = torch.randn(d_model, num_heads * head_dim)
    data = {"__format__": "cov"}
    slice_covs, slice_means = {}, {}
    for h in range(num_heads):
        X = torch.randn(64, head_dim, dtype=torch.float64)
        slice_covs[h] = (X.T @ X / 64).float()
        slice_means[h] = X.float().mean(0)
        entry = {"acts_cov": (X.T @ X).float(), "acts_n": 64}
        if include_mean:
            entry["acts_mean"] = slice_means[h]
        data[f"blk{block_idx}.attn.head{h}.slice"] = entry
    acc = DataAccessor(data, model_name=MODEL)
    acc._selective_weights = {
        f"gpt_neox.layers.{block_idx}.attention.dense": SimpleNamespace(weight=W_o, bias=None)
    }
    return acc, W_o, slice_covs, slice_means, block_idx, head_dim


def _contrib(acc, blk, h):
    return acc.factor(f"blk{blk}.attn.head{h}.contrib", "acts")


def test_contrib_cov_equals_head_slice_map():
    acc, W_o, slice_covs, _, blk, d = _ov_acc()
    for h in slice_covs:
        W_h = W_o[:, h * d:(h + 1) * d].float()
        assert torch.allclose(_contrib(acc, blk, h).cov.cpu(), W_h @ slice_covs[h] @ W_h.T, atol=1e-4)


def test_contrib_eigvals_match_dual_problem():
    # Non-zero eigvals of W_h A W_hᵀ equal those of the cheaper d_head dual W_hᵀ W_h A.
    acc, W_o, slice_covs, _, blk, d = _ov_acc()
    for h in slice_covs:
        W_h = W_o[:, h * d:(h + 1) * d].float()
        contrib = _contrib(acc, blk, h).eigvals
        dual = torch.linalg.eigvals(W_h.T @ W_h @ slice_covs[h]).real.sort(descending=True)[0].clamp(min=0)
        assert torch.allclose(contrib[:d].cpu(), dual, atol=1e-4)
        assert contrib[d:].abs().max().item() < 1e-4


def test_contrib_mean_equals_head_slice_map():
    acc, W_o, _, slice_means, blk, d = _ov_acc(include_mean=True)
    for h in slice_means:
        W_h = W_o[:, h * d:(h + 1) * d].float()
        got = acc.resolve(f"blk{blk}.attn.head{h}.contrib", "acts", "mean")
        assert torch.allclose(got, W_h @ slice_means[h], atol=1e-5)


def test_contrib_navigation_matches_factor():
    acc, *_ , blk, _ = _ov_acc()
    assert torch.allclose(acc.v.blk3.attn.head0.contrib.acts.cov, _contrib(acc, blk, 0).cov, atol=1e-5)


def test_derivation_without_weights_returns_none():
    acc, *_, blk, _ = _ov_acc()
    acc._selective_weights = acc._model_name = acc.model_config = None
    assert _contrib(acc, blk, 0) is None


# --- after_final_norm acts derived from before_final_norm via the final norm ---

def test_after_final_norm_derived_from_before():
    torch.manual_seed(0)
    d = 16
    X = torch.randn(40, d)
    acc = DataAccessor({"before_final_norm": {"acts_samples": X, "acts_n": 40}, "__format__": "acts"},
                       model_name=MODEL)
    norm = torch.nn.LayerNorm(d)
    norm.weight.data.copy_(torch.randn(d)); norm.bias.data.copy_(torch.randn(d))
    acc._selective_weights = {"__norm__": norm.float()}
    with torch.no_grad():
        expect = norm(X.float())
    got = acc.resolve("after_final_norm", "acts", "samples")
    assert got is not None and torch.allclose(got, expect, atol=1e-5)
    assert acc.factor("after_final_norm", "acts").cov is not None  # cov via reformat(cov<-samples)


# --- +o / +b storage modifiers materialize derived acts at a new leaf ---

def test_parse_format_modifiers():
    assert parse_format_spec("eigenvalues+o") == ("eigenvalues", {"o"}, set())
    assert parse_format_spec("cov+o-b") == ("cov", {"o"}, {"b"})


def _should_write(acc, leaf, fmt, plus, minus):
    return acc._should_write(leaf, "acts", acc._entry(leaf), fmt, plus, minus)


def test_contrib_store_gated_by_o_flag():
    acc, *_ = _ov_acc(include_mean=True)
    leaf = "blk3.attn.head0.contrib"
    assert _should_write(acc, leaf, "eigenvalues", set(), set())        # default True
    assert not _should_write(acc, leaf, "cov", set(), set())            # default False
    assert _should_write(acc, leaf, "cov", {"o"}, set())               # +o forces
    assert not _should_write(acc, leaf, "eigenvalues", set(), {"o"})    # -o drops
    assert not _should_write(acc, leaf, "acts", {"o"}, set())           # contrib has no raw samples


def test_save_eigenvalues_materializes_contrib(tmp_path):
    acc, _, slice_covs, _, blk, _ = _ov_acc(include_mean=True)
    in_mem = {h: _contrib(acc, blk, h).eigvals.clone() for h in slice_covs}
    out = str(tmp_path / "ev.pt")
    acc.save(out, format="eigenvalues")
    re = torch.load(out, map_location="cpu", weights_only=False)
    for h in slice_covs:
        stored = re[f"blk{blk}.attn.head{h}.contrib"]["acts_eigvals"].float()
        assert torch.allclose(stored, in_mem[h].float(), atol=1e-4)


def test_save_minus_o_omits_contrib(tmp_path):
    acc, *_ = _ov_acc(include_mean=True)
    out = str(tmp_path / "no_o.pt")
    acc.save(out, format="eigenvalues-o")
    re = torch.load(out, map_location="cpu", weights_only=False)
    for h in range(4):
        assert "acts_eigvals" in re[f"blk3.attn.head{h}.slice"]
        assert f"blk3.attn.head{h}.contrib" not in re


def test_save_mlp_out_b_flag(tmp_path):
    # +b materializes the derived .out acts; -b drops it (leaf has no other acts quantity).
    acc, *_ = _mlp_acc()
    out = str(tmp_path / "b.pt")
    DataAccessor(acc.data, model_name=MODEL, revision=None)._selective_weights = acc._selective_weights
    acc.save(out, format="cov_svd+b")
    re = torch.load(out, map_location="cpu", weights_only=False)
    assert "acts_eigvals" in re["blk0.mlp.up.out"]  # derived acts stored
    assert "grads_eigvals" in re["blk0.mlp.up.out"]
