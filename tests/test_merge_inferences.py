"""Inference .pt merger: leaf/key union semantics, what is (and isn't) overwritten,
and the data-compatibility guard."""
import torch
import pytest

from scripts.merge_inferences import merge_inference, merge_files
from utils.accessor import DataAccessor

META = {"__format__": "cov", "__hf_model__": "EleutherAI/pythia-14m",
        "__revision__": "step1", "__n_chunks__": 100,
        "__token_filter__": {"token_selection": "all"}}


def _entry(d=4, n=20):
    X = torch.randn(n, d, dtype=torch.float64)
    return {"acts_cov": X.T @ X, "acts_n": n}


def _file(leaves, **meta_over):
    return {**META, **meta_over, **{leaf: _entry() for leaf in leaves}}


def _leaves(d):
    return {k for k in d if not k.startswith("__")}


# --- union of disjoint leaves (the mlp.raw_out fold-in case) ---

def test_disjoint_leaves_unioned():
    base = _file(["blk0.attn.in", "blk0.mlp.out"])
    add = _file(["blk0.mlp.raw_out"])
    m = merge_inference(base, add)
    assert _leaves(m) == {"blk0.attn.in", "blk0.mlp.out", "blk0.mlp.raw_out"}


def test_inputs_not_mutated():
    base = _file(["a"]); add = _file(["b"])
    merge_inference(base, add)
    assert _leaves(base) == {"a"} and _leaves(add) == {"b"}


def test_metadata_preserved_and_add_only_folded():
    base = _file(["a"])
    add = _file(["b"], __extra__="x")     # base lacks __extra__
    m = merge_inference(base, add)
    for k, v in META.items():
        assert m[k] == v                   # base metadata intact
    assert m["__extra__"] == "x"           # add-only metadata folded in


# --- overlap: same leaf in both ---

def test_overlapping_leaf_merges_disjoint_keys():
    # base has acts at the leaf, add has grads at the SAME leaf -> both kept (no loss)
    G = torch.randn(20, 4, dtype=torch.float64)
    base = _file([]); add = _file([])
    base["blk0.attn.in"] = {"acts_cov": _entry()["acts_cov"], "acts_n": 20}
    add["blk0.attn.in"] = {"grads_cov": G.T @ G, "grads_n": 20}
    m = merge_inference(base, add)
    assert set(m["blk0.attn.in"]) == {"acts_cov", "acts_n", "grads_cov", "grads_n"}


def test_overlapping_key_keeps_base_not_add(capsys):
    base = _file([]); add = _file([])
    e_base, e_add = _entry()["acts_cov"], _entry()["acts_cov"]
    base["blk0.attn.in"] = {"acts_cov": e_base, "acts_n": 20}
    add["blk0.attn.in"] = {"acts_cov": e_add, "acts_n": 99}   # conflicts
    m = merge_inference(base, add)
    assert m["blk0.attn.in"]["acts_cov"] is e_base            # base wins, not overwritten
    assert m["blk0.attn.in"]["acts_n"] == 20
    assert "keeping base" in capsys.readouterr().out          # conflict noted


# --- data-compatibility guard (refuse to mix covs over different tokens) ---

@pytest.mark.parametrize("key,bad", [
    ("__revision__", "step999"),
    ("__n_chunks__", 200),
    ("__token_filter__", {"token_selection": "last"}),
])
def test_incompatible_data_refused(key, bad):
    base = _file(["a"]); add = _file(["b"], **{key: bad})
    with pytest.raises(ValueError, match=key):
        merge_inference(base, add)


def test_missing_compat_key_is_not_a_mismatch():
    base = _file(["a"]); add = _file(["b"])
    del add["__n_chunks__"]                 # one side simply lacks it
    m = merge_inference(base, add)          # must not raise
    assert m["__n_chunks__"] == 100         # base value retained


# --- file round-trip + the merged file reads back through the accessor ---

def test_merge_files_roundtrip_and_readable(tmp_path):
    bp, ap, op = str(tmp_path / "b.pt"), str(tmp_path / "a.pt"), str(tmp_path / "o.pt")
    torch.save(_file(["blk0.attn.in"]), bp)
    torch.save(_file(["blk0.mlp.raw_out"]), ap)
    merge_files(bp, ap, op)
    merged = torch.load(op, map_location="cpu", weights_only=False)
    assert _leaves(merged) == {"blk0.attn.in", "blk0.mlp.raw_out"}
    acc = DataAccessor(merged)
    assert acc.view("blk0.mlp.raw_out", "acts").eigvals is not None
