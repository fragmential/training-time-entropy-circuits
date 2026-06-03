"""Tests for the leaf-pattern hook resolver: candidates, :quantity, preset, node-error."""
import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

from utils.hook_specs import candidate_leaves, resolve, SingleHookSpec, OVHeadSpec


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


def _resolve(model, family, layers, hooks, sel="all"):
    return resolve(model, _cfg(family), layers, hooks, default_token_selection=sel)


# ---------------------------------------------------------------------------
# candidate_leaves — leaves only, no projection nodes
# ---------------------------------------------------------------------------

def test_pythia_candidates_are_leaves_only():
    m = _FakeModel("pythia", n_layers=2, n_heads=3)
    cand = candidate_leaves(m, _cfg("pythia"), [0, 1], num_heads_per_block=3)
    # projection NODES never appear — only .in/.out leaves
    assert "blk0.mlp.up" not in cand and "blk0.mlp.down" not in cand
    for k in ["blk0.mlp.up.in", "blk0.mlp.up.out", "blk0.mlp.down.in", "blk0.mlp.down.out",
              "blk0.attn.in", "blk0.attn.out", "blk0.mlp.out"]:
        assert k in cand
    # Pythia: no gate, no attn.raw_out, no mlp.in
    assert not any(".gate." in c for c in cand)
    assert "blk0.attn.raw_out" not in cand and "blk0.mlp.in" not in cand
    for h in range(3):
        assert f"blk0.attn.head{h}.slice" in cand


def test_olmo_candidates_include_all_post_norm_leaves():
    m = _FakeModel("olmo", n_layers=1, n_heads=2)
    cand = candidate_leaves(m, _cfg("olmo"), [0], num_heads_per_block=2)
    for k in ["blk0.mlp.gate.in", "blk0.mlp.gate.out", "blk0.mlp.up.in", "blk0.mlp.down.out",
              "blk0.attn.in", "blk0.attn.raw_out", "blk0.attn.out",
              "blk0.mlp.in", "blk0.mlp.out", "blk0.attn.head0.slice", "blk0.attn.head1.slice"]:
        assert k in cand


def test_candidates_include_residual_but_not_identity_head():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    cand = candidate_leaves(m, _cfg("pythia"), [0], num_heads_per_block=1)
    assert "after_final_norm" in cand and "before_final_norm" in cand
    assert "identity_head" not in cand


# ---------------------------------------------------------------------------
# resolve: leaf patterns + :quantity
# ---------------------------------------------------------------------------

def test_default_quantity_is_acts():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = _resolve(m, "pythia", [0], ["blk*.mlp.up.in", "blk*.attn.in"])
    for s in singles:
        assert s.quantities == {"acts"}


def test_mlp_leaf_capture_sides_and_token_selection():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = _resolve(m, "pythia", [0],
                             ["blk*.mlp.up.in:acts", "blk*.mlp.up.out:grads"], sel="last")
    by_leaf = {s.leaf: s for s in singles}
    assert by_leaf["blk0.mlp.up.in"].capture == "input"
    assert by_leaf["blk0.mlp.up.in"].quantities == {"acts"}
    assert by_leaf["blk0.mlp.up.out"].capture == "output"
    assert by_leaf["blk0.mlp.up.out"].quantities == {"grads"}
    # MLP leaves force token_selection="all" regardless of default
    for s in singles:
        assert s.token_selection == "all"
        assert not s.is_global


def test_both_unions_to_acts_and_grads():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = _resolve(m, "pythia", [0], ["blk*.attn.in:both"])
    assert singles[0].quantities == {"acts", "grads"}


def test_repeated_patterns_union_quantities():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = _resolve(m, "pythia", [0],
                             ["blk*.attn.in:acts", "blk*.attn.in:grads"])
    assert len(singles) == 1 and singles[0].quantities == {"acts", "grads"}


def test_bad_quantity_errors():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    with pytest.raises(ValueError, match="quantity"):
        _resolve(m, "pythia", [0], ["blk*.attn.in:foo"])


# ---------------------------------------------------------------------------
# node targets hard-error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node", ["blk*.up", "blk*.mlp.up", "blk3.down", "blk*.mlp.gate"])
def test_projection_node_target_errors(node):
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    with pytest.raises(ValueError, match="projection node"):
        _resolve(m, "pythia", [0], [node])


# ---------------------------------------------------------------------------
# preset:kfac
# ---------------------------------------------------------------------------

def test_preset_kfac_expands_to_in_acts_out_grads():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = _resolve(m, "pythia", [0], ["preset:kfac"])
    by_leaf = {s.leaf: s.quantities for s in singles}
    # pythia has up/down (gate is a silent no-op)
    assert by_leaf["blk0.mlp.up.in"] == {"acts"}
    assert by_leaf["blk0.mlp.up.out"] == {"grads"}
    assert by_leaf["blk0.mlp.down.in"] == {"acts"}
    assert by_leaf["blk0.mlp.down.out"] == {"grads"}


def test_preset_kfac_includes_gate_on_olmo():
    m = _FakeModel("olmo", n_layers=1, n_heads=1)
    singles, _, _ = _resolve(m, "olmo", [0], ["preset:kfac"])
    leaves = {s.leaf for s in singles}
    assert "blk0.mlp.gate.in" in leaves and "blk0.mlp.gate.out" in leaves


def test_unknown_preset_errors():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    with pytest.raises(ValueError, match="preset"):
        _resolve(m, "pythia", [0], ["preset:nope"])


# ---------------------------------------------------------------------------
# OV heads (.slice) + residual is_global
# ---------------------------------------------------------------------------

def test_ov_head_slice_groups_per_block():
    m = _FakeModel("pythia", n_layers=2, n_heads=4)
    singles, ovs, _ = _resolve(m, "pythia", [0, 1], ["blk*.attn.head*.slice"])
    assert singles == [] and len(ovs) == 2
    by_block = {o.block_idx: o for o in ovs}
    assert by_block[0].selected_heads == [0, 1, 2, 3]
    assert by_block[0].num_heads == 4
    assert by_block[0].quantities == {"acts"}


def test_ov_head_single_and_grads():
    m = _FakeModel("pythia", n_layers=2, n_heads=4)
    _, ovs, _ = _resolve(m, "pythia", [0, 1],
                         ["blk0.attn.head2.slice:both", "blk1.attn.head3.slice:grads"])
    by_block = {o.block_idx: o for o in ovs}
    assert by_block[0].selected_heads == [2] and by_block[0].quantities == {"acts", "grads"}
    assert by_block[1].selected_heads == [3] and by_block[1].quantities == {"grads"}


def test_residual_marked_global():
    m = _FakeModel("pythia", n_layers=1, n_heads=1)
    singles, _, _ = _resolve(m, "pythia", [0], ["after_final_norm", "blk*.mlp.up.in"])
    by_leaf = {s.leaf: s for s in singles}
    assert by_leaf["after_final_norm"].is_global is True
    assert by_leaf["after_final_norm"].capture == "output"
    assert by_leaf["blk0.mlp.up.in"].is_global is False


def test_olmo_only_leaves_silent_no_op_on_pythia():
    m = _FakeModel("pythia", n_layers=2, n_heads=2)
    singles, _, _ = _resolve(m, "pythia", [0, 1],
                             ["blk*.mlp.gate.in", "blk*.mlp.in", "blk*.attn.raw_out"])
    assert singles == []
