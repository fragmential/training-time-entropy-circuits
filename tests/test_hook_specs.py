"""Tests for the pattern-based hook resolver and the backward-compat shim."""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from utils import hook_specs as hs
from utils.hook_specs import (
    candidate_hook_names,
    resolve,
    synthesize_from_flags,
    SingleHookSpec,
    OVHeadSpec,
)


# ---------------------------------------------------------------------------
# Fake model: enough surface for the resolver to match Pythia/OLMo-2 paths
# ---------------------------------------------------------------------------

class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 4))


class _MLP(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family == "pythia":
            self.dense_h_to_4h = _Tiny()
            self.dense_4h_to_h = _Tiny()
        else:
            self.gate_proj = _Tiny()
            self.up_proj = _Tiny()
            self.down_proj = _Tiny()


class _Attn(nn.Module):
    def __init__(self, family):
        super().__init__()
        if family == "pythia":
            self.dense = _Tiny()
        else:
            self.o_proj = _Tiny()


class _PythiaBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(4)
        self.attention = _Attn("pythia")
        self.mlp = _MLP("pythia")


class _OlmoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn("olmo")
        self.mlp = _MLP("olmo")
        self.post_attention_layernorm = nn.LayerNorm(4)
        self.post_feedforward_layernorm = nn.LayerNorm(4)


class _FakeModel(nn.Module):
    """Mimics enough of HF GPTNeoX / OLMo-2 surface to drive the resolver."""

    def __init__(self, family, n_layers=4, n_heads=4):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=n_heads, hidden_size=4 * n_heads)
        if family == "pythia":
            blocks = nn.ModuleList([_PythiaBlock() for _ in range(n_layers)])
            self.gpt_neox = SimpleNamespace(layers=blocks, final_layer_norm=nn.LayerNorm(4))
        else:
            blocks = nn.ModuleList([_OlmoBlock() for _ in range(n_layers)])
            self.model = SimpleNamespace(layers=blocks, norm=nn.LayerNorm(4))


def _cfg(family):
    return SimpleNamespace(family=family)


# ---------------------------------------------------------------------------
# candidate_hook_names
# ---------------------------------------------------------------------------

def test_pythia_candidates_exclude_olmo_only_names():
    m = _FakeModel("pythia", n_layers=2, n_heads=3)
    cand = candidate_hook_names(m, _cfg("pythia"), [0, 1], num_heads_per_block=3)
    # Pythia: no gate, no attn.raw_out, no mlp.in
    assert "blk0.gate" not in cand
    assert "blk0.attn.raw_out" not in cand
    assert "blk0.mlp.in" not in cand
    # Should have the canonical Pythia set
    for k in ["blk0.up", "blk0.down", "blk0.attn.in", "blk0.attn.out", "blk0.mlp.out"]:
        assert k in cand
    for h in range(3):
        assert f"blk0.attn.head{h}" in cand


def test_olmo_candidates_include_all_post_norm_names():
    m = _FakeModel("olmo", n_layers=1, n_heads=2)
    cand = candidate_hook_names(m, _cfg("olmo"), [0], num_heads_per_block=2)
    for k in [
        "blk0.gate", "blk0.up", "blk0.down",
        "blk0.attn.in", "blk0.attn.raw_out", "blk0.attn.out",
        "blk0.mlp.in", "blk0.mlp.out",
        "blk0.attn.head0", "blk0.attn.head1",
    ]:
        assert k in cand


def test_candidates_include_global_residual_names():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    cand = candidate_hook_names(m, _cfg("pythia"), [0], num_heads_per_block=1)
    assert "after_final_norm" in cand
    assert "before_final_norm" in cand
    assert "identity_head" in cand


# ---------------------------------------------------------------------------
# resolve: pattern → SingleHookSpec / OVHeadSpec
# ---------------------------------------------------------------------------

def test_resolve_mlp_pattern_pythia_skips_gate():
    m = _FakeModel("pythia", n_layers=2, n_heads=2)
    singles, ovs, _ = resolve(m, _cfg("pythia"), [0, 1],
                              hooks=["blk*.up", "blk*.down", "blk*.gate"],
                              grad_all=False,
                              default_token_selection="last")
    names = sorted(s.name for s in singles)
    assert names == ["blk0.down", "blk0.up", "blk1.down", "blk1.up"]
    assert ovs == []
    # MLP hooks should force token_selection="all" regardless of default
    for s in singles:
        assert s.token_selection == "all"
        assert s.capture == "input"
        assert s.grad_capture == "output"
        assert not s.is_global


def test_resolve_olmo_includes_gate_and_raw_out():
    m = _FakeModel("olmo", n_layers=1, n_heads=2)
    singles, ovs, _ = resolve(m, _cfg("olmo"), [0],
                              hooks=["blk*.gate", "blk*.attn.raw_out", "blk*.mlp.in"],
                              grad_all=False,
                              default_token_selection="last")
    names = sorted(s.name for s in singles)
    assert names == ["blk0.attn.raw_out", "blk0.gate", "blk0.mlp.in"]


def test_resolve_groups_ov_heads_per_block():
    m = _FakeModel("pythia", n_layers=2, n_heads=4)
    singles, ovs, _ = resolve(m, _cfg("pythia"), [0, 1],
                              hooks=["blk*.attn.head*"],
                              grad_all=False,
                              default_token_selection="last")
    assert singles == []
    assert len(ovs) == 2
    by_block = {o.block_idx: o for o in ovs}
    assert sorted(by_block) == [0, 1]
    assert by_block[0].selected_heads == [0, 1, 2, 3]
    assert by_block[0].num_heads == 4
    assert not by_block[0].collect_grad


def test_resolve_single_head_pattern():
    m = _FakeModel("pythia", n_layers=2, n_heads=4)
    _, ovs, _ = resolve(m, _cfg("pythia"), [0, 1],
                        hooks=["blk0.attn.head2", "blk1.attn.head3"],
                        grad_all=False,
                        default_token_selection="last")
    by_block = {o.block_idx: o for o in ovs}
    assert by_block[0].selected_heads == [2]
    assert by_block[1].selected_heads == [3]


def test_resolve_inline_grad_modifier_marks_only_matching():
    m = _FakeModel("pythia", n_layers=2, n_heads=2)
    singles, ovs, _ = resolve(m, _cfg("pythia"), [0, 1],
                              hooks=["blk*.up", "blk*.down+G"],
                              grad_all=False,
                              default_token_selection="last")
    by_name = {s.name: s for s in singles}
    assert by_name["blk0.up"].collect_grad is False
    assert by_name["blk0.down"].collect_grad is True
    assert by_name["blk1.down"].collect_grad is True


def test_resolve_inline_grad_on_ov_head_pattern():
    m = _FakeModel("pythia", n_layers=1, n_heads=2)
    _, ovs, _ = resolve(m, _cfg("pythia"), [0],
                        hooks=["blk*.attn.head*+G"],
                        grad_all=False,
                        default_token_selection="all")
    assert all(o.collect_grad for o in ovs)


def test_resolve_grad_all_enables_grad_everywhere():
    m = _FakeModel("pythia", n_layers=1, n_heads=2)
    singles, ovs, _ = resolve(m, _cfg("pythia"), [0],
                              hooks=["blk*.up", "blk*.attn.head*"],
                              grad_all=True,
                              default_token_selection="all")
    assert all(s.collect_grad for s in singles)
    assert all(o.collect_grad for o in ovs)


def test_resolve_grad_all_overrides_missing_inline_marker():
    """grad_all=True should grad-mark hooks even when no entry had +G."""
    m = _FakeModel("pythia", n_layers=1, n_heads=2)
    singles, _, _ = resolve(m, _cfg("pythia"), [0],
                            hooks=["blk*.up", "blk*.down"],
                            grad_all=True,
                            default_token_selection="all")
    assert all(s.collect_grad for s in singles)


def test_resolve_residual_marked_global():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = resolve(m, _cfg("pythia"), [0],
                            hooks=["after_final_norm", "blk*.up"],
                            grad_all=False,
                            default_token_selection="last")
    by_name = {s.name: s for s in singles}
    assert by_name["after_final_norm"].is_global is True
    assert by_name["blk0.up"].is_global is False


def test_resolve_identity_head_has_no_module():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = resolve(m, _cfg("pythia"), [0],
                            hooks=["identity_head"],
                            grad_all=False,
                            default_token_selection="last")
    assert len(singles) == 1
    assert singles[0].name == "identity_head"
    assert singles[0].module is None
    assert singles[0].is_global is True


def test_resolve_block_boundary_uses_default_token_selection():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = resolve(m, _cfg("pythia"), [0],
                            hooks=["blk*.attn.in", "blk*.mlp.out"],
                            grad_all=False,
                            default_token_selection="last")
    for s in singles:
        assert s.token_selection == "last"


# ---------------------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------------------

def test_shim_collect_A_only_no_grad_marker():
    h = synthesize_from_flags(
        collect_A=True, collect_G=False,
        collect_final_acts=False, collect_final_grads=False,
        residual_hook_point="identity_head",
    )
    assert h == ["blk*.up", "blk*.down", "blk*.gate"]
    # No "+G" anywhere
    assert all(not entry.endswith("+G") for entry in h)


def test_shim_collect_AG_adds_inline_grad_on_mlp():
    h = synthesize_from_flags(
        collect_A=True, collect_G=True,
        collect_final_acts=False, collect_final_grads=False,
        residual_hook_point="identity_head",
    )
    assert h == ["blk*.up+G", "blk*.down+G", "blk*.gate+G"]


def test_shim_collect_final_acts_identity_head():
    h = synthesize_from_flags(
        collect_A=False, collect_G=False,
        collect_final_acts=True, collect_final_grads=False,
        residual_hook_point="identity_head",
    )
    assert h == ["identity_head"]


def test_shim_residual_both_with_final_grads():
    h = synthesize_from_flags(
        collect_A=False, collect_G=False,
        collect_final_acts=True, collect_final_grads=True,
        residual_hook_point="both",
    )
    # Both residual points present, both grad-marked
    assert h == ["before_final_norm+G", "after_final_norm+G"]


def test_shim_after_final_norm_no_grad():
    h = synthesize_from_flags(
        collect_A=False, collect_G=False,
        collect_final_acts=True, collect_final_grads=False,
        residual_hook_point="after_final_norm",
    )
    assert h == ["after_final_norm"]


def test_shim_mixed_mlp_grad_no_residual_grad():
    """collect_A=collect_G=True, collect_final_acts=True (no grad on residual)
    — MLP gets +G, residual hook does not."""
    h = synthesize_from_flags(
        collect_A=True, collect_G=True,
        collect_final_acts=True, collect_final_grads=False,
        residual_hook_point="after_final_norm",
    )
    assert "blk*.up+G" in h
    assert "blk*.down+G" in h
    assert "after_final_norm" in h
    assert "after_final_norm+G" not in h


def test_shim_pythia_gate_pattern_is_silent_no_op():
    # shim always emits blk*.gate, which won't match on Pythia (no gate proj).
    # The resolver should just produce no specs for it; not an error.
    m = _FakeModel("pythia", n_layers=2, n_heads=2)
    h = synthesize_from_flags(
        collect_A=True, collect_G=False,
        collect_final_acts=False, collect_final_grads=False,
        residual_hook_point="identity_head",
    )
    singles, _, _ = resolve(m, _cfg("pythia"), [0, 1], h, grad_all=False, default_token_selection="all")
    names = sorted(s.name for s in singles)
    assert "blk0.gate" not in names
    assert "blk0.up" in names
    assert "blk0.down" in names
