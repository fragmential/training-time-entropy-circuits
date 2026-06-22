"""Canonical hook-name grammar: pure string parsing of dotted capture paths.

A hook is one physical capture-point. The collectable hooks are:
    blk{i}.mlp.{up,down,gate}.{in,out}     MLP projection input / output
    blk{i}.{attn,mlp}.{in,out,raw_out}     sub-block residual boundaries
    blk{i}.attn.head{h}.slice              per-OV-head pre-W_o slice
    after_final_norm / before_final_norm   residual stream around the final norm

Derived (read-time fill-in, never collected): blk{i}.attn.head{h}.contrib and
after_final_norm acts when only `before` was captured. model_registry maps these
names to live modules / derivation edges; this module only parses.
"""

import re

_MLP_PROJ_LEAF = re.compile(r"blk(\d+)\.mlp\.(up|down|gate)\.(in|out)$")
_OV_SLICE = re.compile(r"blk(\d+)\.attn\.head(\d+)\.slice$")
_BOUNDARY = re.compile(r"blk(\d+)\.(attn|mlp)\.(in|out|raw_out)$")
_BLOCK = re.compile(r"blk(\d+)\.")

RESIDUAL_NAMES = ("after_final_norm", "before_final_norm")

# Pythia parallel-residual aliases: requested leaf -> canonical stored leaf.
# Pythia has no input_layernorm split or post-sub-block norms, so its mlp.in /
# attn.raw_out / mlp.raw_out collapse onto attn.in / attn.out / mlp.out.
_ALIASES = [
    (re.compile(r"blk(\d+)\.mlp\.in$"), "blk{}.attn.in"),
    (re.compile(r"blk(\d+)\.attn\.raw_out$"), "blk{}.attn.out"),
    (re.compile(r"blk(\d+)\.mlp\.raw_out$"), "blk{}.mlp.out"),
]


def mlp_proj_leaf(name):
    """(block_idx, proj, side) for an MLP projection leaf, side in {in,out}; else None."""
    m = _MLP_PROJ_LEAF.match(name)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def ov_slice(name):
    """(block_idx, head_idx) for an OV-head slice leaf; else None."""
    m = _OV_SLICE.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def boundary(name):
    """(block_idx, sub, where) for a boundary leaf, sub in {attn,mlp}; else None."""
    m = _BOUNDARY.match(name)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else None


def block_idx(name):
    """Leading block index for any blk{i}.* name, else None."""
    m = _BLOCK.match(name)
    return int(m.group(1)) if m else None


def canonical(name, available):
    """Resolve a Pythia alias leaf to its stored name when the requested one is absent."""
    if name in available:
        return name
    for pat, tmpl in _ALIASES:
        m = pat.match(name)
        if m:
            resolved = tmpl.format(m.group(1))
            if resolved in available:
                return resolved
    return name
