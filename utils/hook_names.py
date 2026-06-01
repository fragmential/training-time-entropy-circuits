"""Canonical hook-name grammar: the single source of truth for hook-name parsing.

Hook names:
    blk{i}.up / .down / .gate              MLP projection inputs (gate: OLMo-2 only)
    blk{i}.attn.head{h}                    per-OV-head pre-W_o slice
    blk{i}.{attn,mlp}.{in,out,raw_out}     sub-block residual boundaries
    after_final_norm / before_final_norm / identity_head   residual stream
    blk{i}.layer                           virtual whole-layer K-FAC (up x down)

Pure string grammar — no torch, no model knowledge. model_registry maps these
names to live modules / state-dict keys; this module only parses and classifies.
"""

import re

_MLP_PROJ = re.compile(r"blk(\d+)\.(up|down|gate)$")
_OV_HEAD = re.compile(r"blk(\d+)\.attn\.head(\d+)$")
_BOUNDARY = re.compile(r"blk(\d+)\.(attn|mlp)\.(in|out|raw_out)$")
_LAYER = re.compile(r"blk(\d+)\.layer$")
_BLOCK = re.compile(r"blk(\d+)\.")

RESIDUAL_NAMES = ("after_final_norm", "before_final_norm", "identity_head")

# Pythia parallel-residual aliases: requested name -> canonical stored name.
_ALIASES = [
    (re.compile(r"blk(\d+)\.mlp\.in$"), "blk{}.attn.in"),
    (re.compile(r"blk(\d+)\.attn\.raw_out$"), "blk{}.attn.out"),
]


def classify(name):
    """One of: 'mlp', 'ov_head', 'boundary', 'residual', 'layer', or None."""
    if _MLP_PROJ.match(name):
        return "mlp"
    if _OV_HEAD.match(name):
        return "ov_head"
    if _BOUNDARY.match(name):
        return "boundary"
    if name in RESIDUAL_NAMES:
        return "residual"
    if _LAYER.match(name):
        return "layer"
    return None


def mlp_proj(name):
    """(block_idx, proj) for an MLP hook, where proj in {up, down, gate}; else None."""
    m = _MLP_PROJ.match(name)
    return (int(m.group(1)), m.group(2)) if m else None


def ov_head(name):
    """(block_idx, head_idx) for an OV-head hook; else None."""
    m = _OV_HEAD.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def boundary(name):
    """(block_idx, sub, where) for a boundary hook, sub in {attn, mlp}; else None."""
    m = _BOUNDARY.match(name)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def block_idx(name):
    """Leading block index for any blk{i}.* name, else None."""
    m = _BLOCK.match(name)
    return int(m.group(1)) if m else None


def is_derivable(name):
    """Hooks whose output factor is derived by a linear map (MLP -> out, OV-head -> contrib)."""
    return classify(name) in ("mlp", "ov_head")


def is_residual(name):
    """Residual-stream points: final-norm and sub-block boundaries (not OV-heads)."""
    return classify(name) in ("residual", "boundary")


# ---------------------------------------------------------------------------
# node.signal vocabulary — the ONLY factor identity in the codebase.
#
# A signal is "<node>.<quantity>" where node is the role at this hook and
# quantity is acts|grads. Storage keys, FactorView ids, available(), and result
# keys all use these strings. There is no A/B/G/O anywhere.
#
#   captured: written by the collector / read from disk
#   derived:  computed on read by applying a linear map to a source signal
# ---------------------------------------------------------------------------

# kind -> (captured acts signal, captured grads signal)
_CAPTURED = {
    "mlp":      ("in.acts", "out.grads"),
    "ov_head":  ("slice.acts", "slice.grads"),
    "boundary": ("value.acts", "value.grads"),
    "residual": ("value.acts", "value.grads"),
}

# kind -> {derived signal: (source captured signal, weight kind)}
_DERIVED = {
    "mlp":     {"out.acts": ("in.acts", "mlp")},
    "ov_head": {"contrib.acts": ("slice.acts", "head")},
}


def captured_signals(name):
    """The acts/grads signal keys the collector writes for this hook's kind."""
    return _CAPTURED.get(classify(name), ())


def derived_signals(name):
    """{derived signal: (source signal, weight kind)} for this hook, possibly empty."""
    return _DERIVED.get(classify(name), {})


def signals(name):
    """All valid node.signal keys for this hook (captured + derived)."""
    return tuple(captured_signals(name)) + tuple(derived_signals(name))


def signal_node(signal):
    """The node part of a 'node.acts'/'node.grads' signal key, e.g. 'in.acts' -> 'in'."""
    return signal.rsplit(".", 1)[0]


def canonical(name, available):
    """Resolve a Pythia alias to its stored name when the requested one isn't present."""
    if name in available:
        return name
    for pat, tmpl in _ALIASES:
        m = pat.match(name)
        if m:
            resolved = tmpl.format(m.group(1))
            if resolved in available:
                return resolved
    return name
