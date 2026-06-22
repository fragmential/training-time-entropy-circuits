"""Resolver + Node/View navigation: produce any format from what's stored,
descend the tree, and resolve Pythia parallel-residual aliases."""
import copy
from types import SimpleNamespace

import torch
import pytest

from utils.accessor import DataAccessor, reconstruct_cov

LEAF = "after_final_norm"
MODEL = "EleutherAI/pythia-14m"


# --- format fixtures (one residual leaf, acts + grads) ---

def _cov_svd_data(d=32):
    C = torch.randn(d, d, dtype=torch.float64)
    C = C @ C.T + 0.01 * torch.eye(d, dtype=torch.float64)
    vals, vecs = torch.linalg.eigh(C / 200)
    eigvals, eigvecs = vals.flip(0).clamp(min=0), vecs.flip(1)
    return {
        LEAF: {
            "acts_eigvals": eigvals, "acts_eigvecs": eigvecs.float(),
            "acts_mean": torch.randn(d).float(), "acts_n": 200,
            "grads_eigvals": eigvals * 0.5, "grads_eigvecs": eigvecs.float(), "grads_n": 200,
        },
        "__format__": "cov_svd",
    }


def _cov_data(d=32, N=200):
    X = torch.randn(N, d, dtype=torch.float64)
    return {LEAF: {"acts_cov": (X.T @ X).float(), "acts_n": N}, "__format__": "cov"}


def _acts_data(d=32, N=200):
    return {LEAF: {"acts_samples": torch.randn(N, d).float(), "acts_n": N}, "__format__": "acts"}


def _acts(acc):
    return acc.view(LEAF, "acts")


def test_eigvals_from_cov_svd():
    data = _cov_svd_data()
    assert torch.allclose(_acts(DataAccessor(data)).eigvals, data[LEAF]["acts_eigvals"])


def test_eigvals_from_cov_computed_descending():
    ev = _acts(DataAccessor(_cov_data(d=16))).eigvals
    assert ev is not None and len(ev) == 16 and (ev[:-1] >= ev[1:]).all()


def test_eigvals_from_acts():
    ev = _acts(DataAccessor(_acts_data(d=16))).eigvals
    assert ev is not None and len(ev) == 16


def test_cov_reconstruction_from_eigdecomp():
    data = _cov_svd_data(d=16)
    cov = _acts(DataAccessor(data)).cov
    V, S = data[LEAF]["acts_eigvecs"], data[LEAF]["acts_eigvals"]
    assert torch.allclose(cov, V @ torch.diag(S.to(V.dtype)) @ V.T, atol=1e-4)


def test_centered_below_uncentered():
    data = _cov_svd_data(d=16)
    acc = DataAccessor(data)
    centered, uncentered = _acts(acc).eigvals_centered, _acts(acc).eigvals
    assert centered is not None and centered[0] <= uncentered[0] + 1e-4


def test_leaves_excludes_metadata():
    names = DataAccessor(_cov_svd_data()).leaves()
    assert LEAF in names and "__format__" not in names


def test_leaf_quantities_lists_both():
    lq = DataAccessor(_cov_svd_data()).leaf_quantities()
    assert (LEAF, "acts") in lq and (LEAF, "grads") in lq


def test_eigenvalues_only_no_cov_no_loop(tmp_path):
    # eigvals stored, no eigvecs: cov/eigvecs are unrecoverable and must not infinite-loop.
    p = str(tmp_path / "ev.pt")
    DataAccessor(_cov_data(d=16)).save(str(tmp_path / "cs.pt"), format="cov_svd")
    DataAccessor(str(tmp_path / "cs.pt")).save(p, format="eigenvalues")
    acc = DataAccessor(p)
    assert acc.resolve(LEAF, "acts", "eigvals") is not None
    assert acc.resolve(LEAF, "acts", "cov") is None
    assert acc.resolve(LEAF, "acts", "eigvecs") is None
    assert acc.can_resolve(LEAF, "acts", "cov") is False
    assert acc.view(LEAF, "acts") is not None  # eigvals reachable


def test_eigvecs_access_order_independent():
    # Reading eigvals first must not poison the cache so eigvecs reads None (the cycle-guard
    # caching bug). Both orders must agree and reconstruct the covariance.
    data = _cov_data(d=16)
    a = DataAccessor(copy.deepcopy(data))
    ev, vecs = _acts(a).eigvals, _acts(a).eigvecs
    b = DataAccessor(copy.deepcopy(data))
    vecs2, ev2 = _acts(b).eigvecs, _acts(b).eigvals
    assert ev is not None and vecs is not None and ev2 is not None and vecs2 is not None
    assert torch.allclose(ev, ev2, atol=1e-6)
    assert torch.allclose(reconstruct_cov(ev, vecs), _acts(a).cov, atol=1e-4)


def test_eigvecs_centered_access_order_independent():
    d = 16
    X = torch.randn(200, d, dtype=torch.float64)
    data = {LEAF: {"acts_cov": (X.T @ X).float(), "acts_n": 200, "acts_mean": X.float().mean(0)}}
    a = DataAccessor(copy.deepcopy(data))
    ev, vecs = _acts(a).eigvals_centered, _acts(a).eigvecs_centered
    b = DataAccessor(copy.deepcopy(data))
    vecs2, ev2 = _acts(b).eigvecs_centered, _acts(b).eigvals_centered
    assert all(t is not None for t in (ev, vecs, ev2, vecs2))
    assert torch.allclose(ev, ev2, atol=1e-6)


# --- Node / View navigation ---

def _nav_cov(d, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(64, d, generator=g)
    return X.T @ X / 64, X.mean(0)


def _mlp_acc():
    d_in, d_out = 8, 6
    in_cov, in_mean = _nav_cov(d_in, 1)
    grad_cov, _ = _nav_cov(d_out, 2)
    data = {
        "blk0.mlp.up.in": {"acts_cov": in_cov, "acts_n": 64, "acts_mean": in_mean},
        "blk0.mlp.up.out": {"grads_cov": grad_cov, "grads_n": 64},
        "__format__": "cov",
    }
    acc = DataAccessor(data, model_name=MODEL)
    W = torch.randn(d_out, d_in, generator=torch.Generator().manual_seed(3))
    acc._selective_weights = {"gpt_neox.layers.0.mlp.dense_h_to_4h": SimpleNamespace(weight=W, bias=None)}
    return acc


def _residual_acc():
    d = 6
    val_cov, val_mean = _nav_cov(d, 4)
    grad_cov, _ = _nav_cov(d, 5)
    return DataAccessor({
        "after_final_norm": {"acts_cov": val_cov, "acts_n": 64, "acts_mean": val_mean,
                             "grads_cov": grad_cov, "grads_n": 64},
        "blk0.attn.out": {"acts_cov": val_cov.clone(), "acts_n": 64},
        "__format__": "cov",
    })


def test_mlp_navigation_matches_factor():
    acc = _mlp_acc()
    assert torch.allclose(acc.v.blk0.mlp.up["in"].acts.eigvals,
                          acc.view("blk0.mlp.up.in", "acts").eigvals)
    assert torch.allclose(acc["blk0.mlp.up.in"].acts.eigvals,
                          acc.view("blk0.mlp.up.in", "acts").eigvals)
    assert torch.allclose(acc.v.blk0.mlp.up.out.acts.cov,
                          acc.view("blk0.mlp.up.out", "acts").cov)  # derived
    assert acc.view("blk0.mlp.up.in", "grads") is None


def test_residual_navigation_matches_factor():
    acc = _residual_acc()
    assert torch.allclose(acc["after_final_norm"].acts.eigvals,
                          acc.view("after_final_norm", "acts").eigvals)
    assert torch.allclose(acc.v.after_final_norm.grads.eigvals,
                          acc.view("after_final_norm", "grads").eigvals)
    assert acc.view("blk0.attn.out", "grads") is None


def test_navigation_matches_direct_decomposition():
    acc = _residual_acc()
    cov, _ = _nav_cov(6, 4)
    vals = torch.linalg.eigvalsh(cov / 64).flip(0).clamp(min=0)  # stored Σ ÷ n_acts
    assert torch.allclose(acc.v.after_final_norm.acts.eigvals, vals, atol=1e-5)


def test_absent_quantity_raises_attribute():
    acc = _residual_acc()
    with pytest.raises(AttributeError):
        acc.v.blk0.attn.out.grads
    with pytest.raises(AttributeError):
        _mlp_acc().v.blk0.mlp.up["in"].grads


# --- Pythia parallel-residual aliases (mlp.in -> attn.in, attn.raw_out -> attn.out) ---

def _boundary_data(keys, d=8):
    base = {"__format__": "cov"}
    for k in keys:
        X = torch.randn(20, d, dtype=torch.float64)
        base[k] = {"acts_cov": (X.T @ X).float(), "acts_n": 20}
    return base


def test_alias_mlp_in_resolves_to_attn_in():
    acc = DataAccessor(_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"]))
    assert torch.equal(acc.view("blk3.mlp.in", "acts").cov,
                       acc.view("blk3.attn.in", "acts").cov)
    assert torch.equal(acc.view("blk3.attn.raw_out", "acts").cov,
                       acc.view("blk3.attn.out", "acts").cov)


def test_alias_not_used_when_canonical_stored():
    acc = DataAccessor(_boundary_data(["blk3.attn.in", "blk3.mlp.in"]))
    assert not torch.equal(acc.view("blk3.mlp.in", "acts").cov,
                           acc.view("blk3.attn.in", "acts").cov)


def test_alias_absent_from_iteration():
    acc = DataAccessor(_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"]))
    assert "blk3.mlp.in" not in acc.leaves()
    lq = {leaf for leaf, _ in acc.leaf_quantities()}
    assert "blk3.mlp.in" not in lq and "blk3.attn.raw_out" not in lq


def test_alias_convert_does_not_duplicate(tmp_path):
    out = str(tmp_path / "c.pt")
    DataAccessor(_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"])).save(out, format="cov_svd+m")
    keys = [k for k in torch.load(out, map_location="cpu", weights_only=False) if not k.startswith("__")]
    assert sorted(keys) == ["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"]


def test_navigation_to_boundaries():
    acc = DataAccessor(_boundary_data(["blk3.attn.in", "blk3.attn.out", "blk3.mlp.out"]))
    assert torch.equal(acc.v.blk3.attn["in"].acts.cov, acc.view("blk3.attn.in", "acts").cov)
    assert torch.equal(acc.v.blk3.mlp.out.acts.cov, acc.view("blk3.mlp.out", "acts").cov)
    assert acc.view("blk3.attn.in", "grads") is None
