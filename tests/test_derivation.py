"""Derivation of acts at non-captured leaves, through ONE resolver path:
  mlp `…out` <- `…in` via the projection weight (+bias for Pythia),
  ov  `…contrib` <- `…slice` via an o_proj head-column slice,
  `after_final_norm` <- `before_final_norm` via the final norm,
plus the +b/+o save overrides that materialize derived acts. Weights are injected
via a dict-backed stub provider so nothing downloads."""
from types import SimpleNamespace
from typing import Callable

import torch
import pytest

from utils.accessor import DataAccessor
from utils.model_registry import WeightProvider, get_model_config

MODEL = "EleutherAI/pythia-14m"
OLMO = "allenai/OLMo-2-0425-1B"
CFG = get_model_config(MODEL)
ID = (MODEL, "step0", "test")          # save asserts a complete identity


class _StubWeights(WeightProvider):
    """Dict-backed WeightProvider: {sd_prefix: SimpleNamespace(weight, bias), '__norm__': norm}."""
    def __init__(self, weights: dict) -> None:
        self._w = weights

    def weight(self, sd_prefix: str) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        ns = self._w.get(sd_prefix)
        return (ns.weight, ns.bias) if ns is not None else None

    def norm(self) -> Callable[[torch.Tensor], torch.Tensor] | None:
        return self._w.get("__norm__")


# --- MLP .out acts = W @ in_cov @ Wᵀ (+ bias correction for Pythia; OLMo has none) ---

def _mlp_acc(d_in=8, d_out=6, bias=True, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(64, d_in, dtype=torch.float64)
    cfg = CFG if bias else get_model_config(OLMO)
    prefix = "gpt_neox.layers.0.mlp.dense_h_to_4h" if bias else "model.layers.0.mlp.up_proj"
    data = {
        "blk0.mlp.up.in": {"acts_gram": X.T @ X, "acts_n": 64, "acts_mean": X.float().mean(0)},
        "blk0.mlp.up.out": {"grads_gram": torch.eye(d_out) * d_out, "grads_n": 64},
        "__format__": "cov",
    }
    W = torch.randn(d_out, d_in)
    b = torch.randn(d_out) if bias else None
    wp = _StubWeights({prefix: SimpleNamespace(weight=W, bias=b)})
    return DataAccessor(data, identity=ID, config=cfg, weights=wp), W, b, X


def _out(acc):
    return acc["blk0.mlp.up.out"].acts


def test_mlp_out_acts_cov_with_bias_correction():
    acc, W, b, X = _mlp_acc(bias=True)
    C = (X.T @ X).float() / 64
    Wm = W @ X.float().mean(0)
    expect = W @ C @ W.T + torch.outer(Wm, b) + torch.outer(b, Wm) + torch.outer(b, b)
    assert torch.allclose(_out(acc).cov, expect, atol=1e-4)


def test_mlp_out_acts_cov_no_bias():
    acc, W, _, X = _mlp_acc(bias=False)             # OLMo MLP: no bias
    C = (X.T @ X).float() / 64
    assert torch.allclose(_out(acc).cov, W @ C @ W.T, atol=1e-4)


def test_mlp_out_acts_eigvals_descending():
    acc, *_ = _mlp_acc()
    ev = _out(acc).eigvals
    assert ev is not None and (ev[:-1] >= ev[1:] - 1e-5).all()


def test_no_weights_skips_weight_derivation():
    # up.out acts needs the projection weight; no provider must skip (no download), not crash.
    X = torch.randn(64, 8, dtype=torch.float64)
    data = {"blk0.mlp.up.in": {"acts_gram": X.T @ X, "acts_n": 64, "acts_mean": X.float().mean(0)},
            "__format__": "cov"}
    acc = DataAccessor(data, config=CFG)                  # no weights provider
    with pytest.raises(AttributeError):
        acc["blk0.mlp.up.out"].acts                       # weight-derived leaf skipped
    assert acc["blk0.mlp.up.in"].acts.eigvals is not None # captured leaf still resolves


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
        entry = {"acts_gram": (X.T @ X).float(), "acts_n": 64}
        if include_mean:
            entry["acts_mean"] = slice_means[h]
        data[f"blk{block_idx}.attn.head{h}.slice"] = entry
    wp = _StubWeights(
        {f"gpt_neox.layers.{block_idx}.attention.dense": SimpleNamespace(weight=W_o, bias=None)})
    acc = DataAccessor(data, identity=ID, config=CFG, weights=wp)
    return acc, W_o, slice_covs, slice_means, block_idx, head_dim


def _contrib(acc, blk, h):
    return acc[f"blk{blk}.attn.head{h}.contrib"].acts


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
        got = acc[f"blk{blk}.attn.head{h}.contrib"].acts.mean
        assert torch.allclose(got, W_h @ slice_means[h], atol=1e-5)


def test_contrib_navigation_matches_index():
    acc, *_, blk, _ = _ov_acc()
    assert torch.allclose(acc.v.blk3.attn.head0.contrib.acts.cov, _contrib(acc, blk, 0).cov, atol=1e-5)


def test_derivation_without_weights_raises():
    acc, *_, blk, _ = _ov_acc()
    bare = DataAccessor(acc.data)                         # no config, no weights
    with pytest.raises(AttributeError):
        bare[f"blk{blk}.attn.head0.contrib"].acts


# --- after_final_norm acts derived from before_final_norm via the final norm ---

def test_after_final_norm_derived_from_before():
    torch.manual_seed(0)
    d = 16
    X = torch.randn(40, d)
    norm = torch.nn.LayerNorm(d)
    norm.weight.data.copy_(torch.randn(d)); norm.bias.data.copy_(torch.randn(d))
    wp = _StubWeights({"__norm__": norm.float()})
    acc = DataAccessor({"before_final_norm": {"acts_samples": X, "acts_n": 40}, "__format__": "acts"},
                       identity=ID, config=CFG, weights=wp)
    with torch.no_grad():
        expect = norm(X.float())
    got = acc["after_final_norm"].acts.samples
    assert got is not None and torch.allclose(got, expect, atol=1e-5)
    assert acc["after_final_norm"].acts.cov is not None    # cov via conversion (cov <- samples)


# --- +b / +o save overrides materialize derived acts at a new leaf ---

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
    acc.save(out, format="eigenvalues", overrides=("-o",))
    re = torch.load(out, map_location="cpu", weights_only=False)
    for h in range(4):
        assert "acts_eigvals" in re[f"blk3.attn.head{h}.slice"]
        assert f"blk3.attn.head{h}.contrib" not in re


def test_save_mlp_out_b_flag(tmp_path):
    # +b materializes the derived .out acts alongside its stored grads.
    acc, *_ = _mlp_acc()
    out = str(tmp_path / "b.pt")
    acc.save(out, format="cov_svd", overrides=("+b",))
    re = torch.load(out, map_location="cpu", weights_only=False)
    assert "acts_eigvals" in re["blk0.mlp.up.out"]  # derived acts stored
    assert "grads_eigvals" in re["blk0.mlp.up.out"]
