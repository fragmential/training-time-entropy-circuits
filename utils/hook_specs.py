"""Hook resolution: turn `hooks` config entries into concrete collection specs.

Each entry is `<leaf-pattern>[:acts|:grads|:both]` (default `:acts`) or `preset:<name>`.
A leaf-pattern is an fnmatch glob over the leaves this model exposes (see
`candidate_leaves`); `:quantity` means the same thing at every leaf — capture acts
and/or grads at that one point. There are no node targets and no `+G`: K-FAC is just
two ordinary leaves (`…up.in:acts` + `…up.out:grads`), or `preset:kfac` for all of them.
"""

import fnmatch
import re
from dataclasses import dataclass
from typing import List, Set

import torch.nn as nn

from utils import hook_names
from utils.model_registry import (
    get_mlp_projections,
    get_block_boundary_hooks,
    get_attention_output_proj,
    get_final_layernorm,
)


_NODE_TARGET = re.compile(r".*\.(up|down|gate)$")

_PRESETS = {
    "kfac": [f"blk*.mlp.{p}.in:acts" for p in ("up", "down", "gate")]
          + [f"blk*.mlp.{p}.out:grads" for p in ("up", "down", "gate")],
}


@dataclass
class SingleHookSpec:
    """A concrete leaf capture-point and the quantities to collect there."""
    leaf: str
    module: nn.Module | None   # the module whose `capture` side is the leaf
    capture: str                  # "input" | "output"
    quantities: Set[str]          # subset of {"acts", "grads"}
    token_selection: str | None
    is_global: bool               # residual / final-norm (registered once across passes)


@dataclass
class OVHeadSpec:
    """Per-OV-head collection at one block's o_proj (covers `selected_heads` only)."""
    block_idx: int
    o_proj: nn.Module
    num_heads: int
    head_dim: int
    selected_heads: List[int]
    quantities: Set[str]
    token_selection: str | None


def _leaf_modules(model, model_config, target_layers, num_heads_per_block):
    """Derive the model's leaf universe straight from the architecture helpers, so
    family knowledge lives ONLY in model_registry. Returns:
        singles: {leaf: (module, capture)}   — every directly-hooked leaf
        ov:      {block_idx: (o_proj, num_heads, head_dim)}  — OV-head slice sources
    `.in`/`.out` of each MLP projection share that projection's module (input/output
    side); boundaries already carry their capture side; head slices route to the
    dispatcher, not a single module.
    """
    ln = get_final_layernorm(model, model_config)
    singles = {"after_final_norm": (ln, "output"), "before_final_norm": (ln, "input")}
    ov = {}
    for i in target_layers:
        for name, mod in get_mlp_projections(model, model_config, i):
            singles[f"{name}.in"] = (mod, "input")
            singles[f"{name}.out"] = (mod, "output")
        for name, mod, cap in get_block_boundary_hooks(model, model_config, i):
            singles[name] = (mod, cap)
        _, o_proj, total_heads, head_dim = get_attention_output_proj(model, model_config, i)
        ov[i] = (o_proj, total_heads, head_dim)
    return singles, ov


def _head_leaves(ov, num_heads_per_block):
    """{leaf: (block_idx, head_idx)} for every OV-head slice leaf."""
    return {f"blk{i}.attn.head{h}.slice": (i, h)
            for i in ov for h in range(num_heads_per_block)}


def candidate_leaves(model, model_config, target_layers, num_heads_per_block) -> List[str]:
    """Every collectable leaf this model exposes (for diagnostics / glob matching)."""
    singles, ov = _leaf_modules(model, model_config, target_layers, num_heads_per_block)
    return list(singles) + list(_head_leaves(ov, num_heads_per_block))


def _expand_presets(hooks) -> List[str]:
    out = []
    for entry in hooks or []:
        if entry.startswith("preset:"):
            name = entry[len("preset:"):]
            if name not in _PRESETS:
                raise ValueError(f"unknown preset {name!r}; known: {sorted(_PRESETS)}")
            out.extend(_PRESETS[name])
        else:
            out.append(entry)
    return out


def _parse_entry(entry: str) -> tuple:
    """(leaf_pattern, quantities) for `<leaf>[:acts|:grads|:both]`."""
    pat, _, q = entry.partition(":")
    q = q or "acts"
    if q not in ("acts", "grads", "both"):
        raise ValueError(f"bad quantity {q!r} in {entry!r}; use :acts, :grads, or :both")
    if _NODE_TARGET.match(pat):
        raise ValueError(
            f"{pat!r} is an MLP projection node, not a leaf. Capture its input and "
            f"output explicitly — '{pat}.in:acts' + '{pat}.out:grads' — or use 'preset:kfac'."
        )
    return pat, ({"acts", "grads"} if q == "both" else {q})


def resolve(model, model_config, target_layers, hooks, default_token_selection):
    """Expand `hooks` into (SingleHookSpec list, OVHeadSpec list, candidate leaves).

    Quantities requested for the same leaf by multiple patterns are unioned. MLP
    projection leaves force token_selection="all"; everything else uses the default.
    """
    n_heads = model.config.num_attention_heads
    singles_mod, ov_info = _leaf_modules(model, model_config, target_layers, n_heads)
    head_leaves = _head_leaves(ov_info, n_heads)

    leaf_quants: dict = {}
    for entry in _expand_presets(hooks):
        pat, quants = _parse_entry(entry)
        for leaf in (*singles_mod, *head_leaves):
            if fnmatch.fnmatchcase(leaf, pat):
                leaf_quants.setdefault(leaf, set()).update(quants)

    singles: List[SingleHookSpec] = []
    ov_groups: dict = {}  # block_idx -> list[(head_idx, quantities)]
    for leaf, quants in leaf_quants.items():
        if leaf in head_leaves:
            i, h = head_leaves[leaf]
            ov_groups.setdefault(i, []).append((h, quants))
            continue
        module, capture = singles_mod[leaf]
        singles.append(SingleHookSpec(
            leaf=leaf,
            module=module,
            capture=capture,
            quantities=quants,
            token_selection="all" if hook_names.mlp_proj_leaf(leaf) else default_token_selection,
            is_global=leaf in hook_names.RESIDUAL_NAMES,
        ))

    ovs: List[OVHeadSpec] = []
    for block_idx, hq in sorted(ov_groups.items()):
        o_proj, total_heads, head_dim = ov_info[block_idx]
        ovs.append(OVHeadSpec(
            block_idx=block_idx,
            o_proj=o_proj,
            num_heads=total_heads,
            head_dim=head_dim,
            selected_heads=sorted({h for h, _ in hq}),
            quantities=set().union(*(q for _, q in hq)),
            token_selection=default_token_selection,
        ))

    return singles, ovs, list(singles_mod) + list(head_leaves)
