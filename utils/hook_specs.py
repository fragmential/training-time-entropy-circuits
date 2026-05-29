"""Hook pattern resolution: turn glob patterns into concrete collection specs.

The collection config exposes:
    hooks    — list of fnmatch patterns naming forward-collection points.
               Each entry may have a trailing `+G` modifier marking that
               pattern's matches to also collect G (triggers backward pass).
               Example: "blk*.down+G".
    grad_all — bool. When True, every matched hook collects G regardless of
               inline `+G` markers. Use as a shortcut for "G everywhere".

Patterns match against the universe of hook names this model exposes, which is
derived from model architecture (MLP projections, block boundaries, per-head
OV decomposition, final layernorm). See `candidate_hook_names`.

Patterns like `"blk*.attn.head*"` or `"blk*.down"` are family-aware via the
candidate set: Pythia has no `mlp.in` / `attn.raw_out` / `gate`, so those
candidates never appear and patterns matching only them are harmless no-ops.
"""

import fnmatch
from dataclasses import dataclass
from typing import Optional, List

import torch.nn as nn

from utils import hook_names
from utils.model_registry import (
    get_mlp_projections,
    get_block_boundary_hooks,
    get_attention_output_proj,
    get_final_layernorm,
)


_GRAD_SUFFIX = "+G"


def _split_grad_suffix(pattern: str) -> tuple:
    """Return (bare_pattern, has_grad). A trailing '+G' marks the pattern for G collection."""
    if pattern.endswith(_GRAD_SUFFIX):
        return pattern[: -len(_GRAD_SUFFIX)], True
    return pattern, False


@dataclass
class SingleHookSpec:
    """A concrete forward-hook collection point."""
    name: str
    module: Optional[nn.Module]   # None for identity_head (manual feed)
    capture: str                  # "input" | "output"
    collect_grad: bool
    token_selection: Optional[str]
    grad_capture: str             # which side to grab on backward (matches HookCollector)
    is_global: bool               # True for residual / final-norm hooks (registered once across passes)


@dataclass
class OVHeadSpec:
    """Per-OV-head collection at one block's o_proj (covers `selected_heads` heads only)."""
    block_idx: int
    o_proj: nn.Module
    num_heads: int                # total query heads at this block (used for slicing W_o)
    head_dim: int
    selected_heads: List[int]
    collect_grad: bool
    token_selection: Optional[str]


def candidate_hook_names(model, model_config, target_layers, num_heads_per_block) -> List[str]:
    """All valid hook names this model exposes given `target_layers`.

    Family-aware: Pythia parallel-residual layers don't expose `mlp.in` or
    `attn.raw_out` (they're identical to `attn.in` / `attn.out` and resolved by
    DataAccessor alias), and Pythia MLPs have no `gate`.
    """
    out = list(hook_names.RESIDUAL_NAMES)
    olmo = model_config.family == "olmo"
    for i in target_layers:
        out.append(f"blk{i}.up")
        out.append(f"blk{i}.down")
        if olmo:
            out.append(f"blk{i}.gate")
        out.append(f"blk{i}.attn.in")
        out.append(f"blk{i}.attn.out")
        if olmo:
            out.append(f"blk{i}.attn.raw_out")
            out.append(f"blk{i}.mlp.in")
        out.append(f"blk{i}.mlp.out")
        for h in range(num_heads_per_block):
            out.append(f"blk{i}.attn.head{h}")
    return out


def _matches_any(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns or [])


def _parse_hook_patterns(hooks) -> tuple:
    """Split each entry by trailing '+G' suffix.

    Returns (bare_patterns, grad_patterns) where bare_patterns is the full list
    of forward-selection patterns and grad_patterns is the subset that had `+G`.
    """
    bare, grad = [], []
    for entry in hooks or []:
        b, g = _split_grad_suffix(entry)
        bare.append(b)
        if g:
            grad.append(b)
    return bare, grad


def _module_for(model, model_config, name: str):
    """Resolve a concrete hook name to (module, capture). (None, None) for identity_head."""
    if name == "identity_head":
        return None, None
    if name == "after_final_norm":
        return get_final_layernorm(model, model_config), "output"
    if name == "before_final_norm":
        return get_final_layernorm(model, model_config), "input"
    if hook_names.mlp_proj(name):
        block_idx = hook_names.block_idx(name)
        for n, layer in get_mlp_projections(model, model_config, block_idx):
            if n == name:
                return layer, "input"
    if hook_names.boundary(name):
        block_idx = hook_names.block_idx(name)
        for n, mod, cap in get_block_boundary_hooks(model, model_config, block_idx):
            if n == name:
                return mod, cap
    raise ValueError(f"Cannot resolve module for hook name: {name!r}")


def resolve(model, model_config, target_layers, hooks, grad_all, default_token_selection):
    """Expand patterns into concrete (SingleHookSpec, OVHeadSpec) lists.

    Args:
        model: instantiated HF model.
        model_config: ModelConfig from utils.model_registry.
        target_layers: list of block indices in scope (residual hooks ignore this).
        hooks: list of fnmatch patterns naming forward-collection points. Each
            entry may have a trailing '+G' modifier marking that pattern for G
            collection (e.g. "blk*.down+G").
        grad_all: if True, every matched hook collects G regardless of '+G'
            markers (shortcut for "G everywhere").
        default_token_selection: token_selection for non-MLP hooks (MLP forces "all").

    Returns:
        (singles, ovs, candidates) where `candidates` is the full universe (for diagnostics).
    """
    n_heads = model.config.num_attention_heads
    candidates = candidate_hook_names(model, model_config, target_layers, n_heads)
    bare_patterns, grad_patterns = _parse_hook_patterns(hooks)
    selected = [c for c in candidates if _matches_any(c, bare_patterns)]

    singles: List[SingleHookSpec] = []
    ov_groups: dict = {}  # block_idx -> list[(head_idx, collect_grad)]
    for name in selected:
        is_grad = grad_all or _matches_any(name, grad_patterns)
        oh = hook_names.ov_head(name)
        if oh:
            ov_groups.setdefault(oh[0], []).append((oh[1], is_grad))
            continue
        module, capture = _module_for(model, model_config, name)
        is_mlp = bool(hook_names.mlp_proj(name))
        is_residual = name in hook_names.RESIDUAL_NAMES
        singles.append(SingleHookSpec(
            name=name,
            module=module,
            capture=capture,
            collect_grad=is_grad,
            token_selection="all" if is_mlp else default_token_selection,
            grad_capture="output" if is_mlp else (capture or "output"),
            is_global=is_residual,
        ))

    ovs: List[OVHeadSpec] = []
    for block_idx, hg in sorted(ov_groups.items()):
        _, o_proj, total_heads, head_dim = get_attention_output_proj(model, model_config, block_idx)
        head_idxs = sorted({h for h, _ in hg})
        any_grad = any(g for _, g in hg)
        ovs.append(OVHeadSpec(
            block_idx=block_idx,
            o_proj=o_proj,
            num_heads=total_heads,
            head_dim=head_dim,
            selected_heads=head_idxs,
            collect_grad=any_grad,
            token_selection=default_token_selection,
        ))

    return singles, ovs, candidates


# ---------------------------------------------------------------------------
# Backward-compat shim: synthesize the `hooks` list (with inline +G) from old boolean flags
# ---------------------------------------------------------------------------

def synthesize_from_flags(
    collect_A: bool,
    collect_G: bool,
    collect_final_acts: bool,
    collect_final_grads: bool,
    residual_hook_point: str,
) -> List[str]:
    """Return a `hooks` list (with inline '+G' markers) equivalent to the legacy flags."""
    hooks: List[str] = []
    mlp_suffix = _GRAD_SUFFIX if collect_G else ""
    if collect_A or collect_G:
        hooks.extend(f"blk*.{p}{mlp_suffix}" for p in ("up", "down", "gate"))
    if collect_final_acts or collect_final_grads:
        if residual_hook_point == "both":
            r = ["before_final_norm", "after_final_norm"]
        else:
            r = [residual_hook_point]
        res_suffix = _GRAD_SUFFIX if collect_final_grads else ""
        hooks.extend(name + res_suffix for name in r)
    return hooks
