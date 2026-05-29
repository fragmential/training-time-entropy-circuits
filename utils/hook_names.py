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
