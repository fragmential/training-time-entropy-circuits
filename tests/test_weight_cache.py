"""Unit tests for the on-disk single-tensor weight cache.

Covers glob resolution (gitignore semantics), cache read/write/round-trip,
corrupt-file fallback, and the family default scope.
"""
import os
import torch

from utils.model_registry import (
    DEFAULT_WEIGHT_CACHE_GLOBS,
    _cache_lookup,
    _cache_path,
    _cache_store,
    _should_cache,
    resolve_cache_globs,
)


# ---------------------------------------------------------------------------
# Glob resolution
# ---------------------------------------------------------------------------

def test_family_defaults_only():
    rules = resolve_cache_globs("pythia")
    assert _should_cache("gpt_neox.layers.0.attention.dense.weight", rules)
    assert not _should_cache("gpt_neox.layers.0.mlp.dense_h_to_4h.weight", rules)


def test_olmo_default_excludes_pythia_pattern():
    rules = resolve_cache_globs("olmo")
    assert _should_cache("model.layers.3.self_attn.o_proj.weight", rules)
    assert not _should_cache("gpt_neox.layers.3.attention.dense.weight", rules)


def test_extras_append_includes():
    rules = resolve_cache_globs("olmo", extras=["*.mlp.down_proj.weight"])
    assert _should_cache("model.layers.0.self_attn.o_proj.weight", rules)
    assert _should_cache("model.layers.0.mlp.down_proj.weight", rules)
    assert not _should_cache("model.layers.0.mlp.up_proj.weight", rules)


def test_bang_negates_default():
    rules = resolve_cache_globs("pythia", extras=["!*.attention.dense.weight"])
    assert not _should_cache("gpt_neox.layers.0.attention.dense.weight", rules)


def test_gitignore_last_match_wins():
    # Include attention, then explicitly exclude block 5, then re-include exact path
    rules = resolve_cache_globs(
        "pythia",
        extras=[
            "!gpt_neox.layers.5.attention.dense.weight",
            "gpt_neox.layers.5.attention.dense.weight",
        ],
    )
    assert _should_cache("gpt_neox.layers.5.attention.dense.weight", rules)


def test_extras_can_re_disable():
    rules = resolve_cache_globs(
        "pythia",
        extras=[
            "*.mlp.dense_h_to_4h.weight",
            "!*.mlp.*",
        ],
    )
    assert not _should_cache("gpt_neox.layers.0.mlp.dense_h_to_4h.weight", rules)
    # Default attention is still untouched
    assert _should_cache("gpt_neox.layers.0.attention.dense.weight", rules)


def test_unknown_family_no_defaults():
    rules = resolve_cache_globs("not_a_family")
    assert rules == []
    assert not _should_cache("any.key.weight", rules)


# ---------------------------------------------------------------------------
# Disk round-trip
# ---------------------------------------------------------------------------

def test_cache_store_then_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    tensor = torch.randn(8, 8)
    _cache_store("pythia", "EleutherAI/pythia-14m", "step1000",
                 "gpt_neox.layers.0.attention.dense.weight", tensor)
    got = _cache_lookup("pythia", "EleutherAI/pythia-14m", "step1000",
                       "gpt_neox.layers.0.attention.dense.weight")
    assert got is not None
    assert torch.equal(got, tensor)


def test_cache_path_uses_short_name(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    p = _cache_path("olmo", "allenai/OLMo-2-0425-1B", "stage1-step100",
                   "model.layers.0.self_attn.o_proj.weight")
    # repo slash is stripped; revision becomes a directory segment
    assert str(tmp_path) in p
    assert "olmo" in p
    assert "OLMo-2-0425-1B" in p
    assert "stage1-step100" in p
    assert p.endswith("model.layers.0.self_attn.o_proj.weight.pt")
    # No allenai/ leading segment
    assert "allenai" not in p


def test_cache_lookup_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    got = _cache_lookup("pythia", "EleutherAI/pythia-14m", "step1000",
                       "does.not.exist.weight")
    assert got is None


def test_cache_lookup_corrupt_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    sd_key = "gpt_neox.layers.0.attention.dense.weight"
    path = _cache_path("pythia", "EleutherAI/pythia-14m", "stepX", sd_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"this is not a torch tensor")
    got = _cache_lookup("pythia", "EleutherAI/pythia-14m", "stepX", sd_key)
    assert got is None  # corrupt → treated as miss


def test_cache_store_atomic_no_tmp_leftover(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    _cache_store("pythia", "EleutherAI/pythia-14m", "step1",
                 "gpt_neox.layers.0.attention.dense.weight", torch.randn(4, 4))
    # No .tmp leftover next to the final file
    p = _cache_path("pythia", "EleutherAI/pythia-14m", "step1",
                   "gpt_neox.layers.0.attention.dense.weight")
    assert os.path.isfile(p)
    assert not os.path.isfile(p + ".tmp")


# ---------------------------------------------------------------------------
# Family-default snapshot (so the surface stays explicit)
# ---------------------------------------------------------------------------

def test_default_globs_are_attention_output_only():
    assert DEFAULT_WEIGHT_CACHE_GLOBS["pythia"] == ["*.attention.dense.weight"]
    assert DEFAULT_WEIGHT_CACHE_GLOBS["olmo"] == ["*.self_attn.o_proj.weight"]
