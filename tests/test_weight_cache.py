"""Minimal cache round-trip tests for the on-disk weight cache."""
import torch

from utils.model_registry import _cache_lookup, _cache_path, _cache_store


def test_store_then_lookup(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    tensor = torch.randn(8, 8)
    _cache_store("pythia", "EleutherAI/pythia-14m", "step1000",
                 "gpt_neox.layers.0.attention.dense.weight", tensor)
    got = _cache_lookup("pythia", "EleutherAI/pythia-14m", "step1000",
                        "gpt_neox.layers.0.attention.dense.weight")
    assert got is not None
    assert torch.equal(got, tensor)


def test_cache_miss_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    got = _cache_lookup("pythia", "EleutherAI/pythia-14m", "step1000", "does.not.exist.weight")
    assert got is None


def test_cache_path_structure(monkeypatch, tmp_path):
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    p = _cache_path("olmo", "allenai/OLMo-2-0425-1B", "stage1-step100",
                    "model.layers.0.self_attn.o_proj.weight")
    assert str(tmp_path) in p
    assert "olmo" in p
    assert "OLMo-2-0425-1B" in p
    assert "stage1-step100" in p
    assert p.endswith("model.layers.0.self_attn.o_proj.weight.pt")
    assert "allenai" not in p


def test_corrupt_cache_returns_none(monkeypatch, tmp_path):
    import os
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    sd_key = "gpt_neox.layers.0.attention.dense.weight"
    path = _cache_path("pythia", "EleutherAI/pythia-14m", "stepX", sd_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not a tensor")
    assert _cache_lookup("pythia", "EleutherAI/pythia-14m", "stepX", sd_key) is None


def test_store_atomic_no_tmp_leftover(monkeypatch, tmp_path):
    import os
    monkeypatch.setenv("WEIGHT_CACHE_DIR", str(tmp_path))
    _cache_store("pythia", "EleutherAI/pythia-14m", "step1",
                 "gpt_neox.layers.0.attention.dense.weight", torch.randn(4, 4))
    p = _cache_path("pythia", "EleutherAI/pythia-14m", "step1",
                    "gpt_neox.layers.0.attention.dense.weight")
    assert os.path.isfile(p)
    assert not os.path.isfile(p + ".tmp")
