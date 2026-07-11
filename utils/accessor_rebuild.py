"""DataAccessor rebuild (WIP) — read/resolve core only. See docs/accessor_interface.md.

Axis: leaf (dotted hook path) . quantity (acts|grads) . component (cov/eigvals/eigvecs/...).
`_resolve(leaf, q, component)` produces any component: stored -> convert -> derive (cross-leaf).
weights, projections, save, prewarm, identity, CLI, profiler: not yet.
"""

import torch
from collections.abc import Iterator
from functools import partial
from typing import Callable

from utils.hook_names import canonical
from utils.model_registry import ModelConfig, derivation, derived_leaves

Component = torch.Tensor | int | float

COMPONENTS = (
    "gram", "n", "mask", "samples", "mean", "lvecs",
    "cov", "cov_centered",
    "eigvals", "eigvecs", "eigvals_centered", "eigvecs_centered",
)


def _eigh(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One decomposition shared by eigvals + eigvecs; descending, eigvals clamped >= 0."""
    vals, vecs = torch.linalg.eigh(cov)
    return vals.flip(0).clamp(min=0).contiguous(), vecs.flip(1).contiguous()

def _cov(samples: torch.Tensor) -> torch.Tensor:
    X = samples.float()
    return X.T @ X / X.shape[0]


# component -> [(source components, fn)], tried in priority order. `_eigh` is private
# (View blocks `_`), kept so eigvals + eigvecs share one decomposition.
CONVERSIONS: dict[str, list[tuple[tuple[str, ...], Callable]]] = {
    "cov":              [(("gram", "n"),          lambda g, n: g.float() / n),
                         (("samples", "mask"),    lambda X, m: _cov(X[m.bool()])),
                         (("samples",),           _cov),
                         (("eigvals", "eigvecs"), lambda l, V: V @ torch.diag(l.to(V.dtype)) @ V.T)],
    "cov_centered":     [(("cov", "mean"),        lambda c, m: c - torch.outer(m, m))],
    "n":                [(("samples", "mask"),    lambda X, m: int(m.bool().sum())),
                         (("samples",),           lambda X: X.shape[0])],
    "_eigh":            [(("cov",),               _eigh)],
    "_eigh_centered":   [(("cov_centered",),      _eigh)],
    "eigvals":          [(("_eigh",),             lambda e: e[0])],
    "eigvecs":          [(("_eigh",),             lambda e: e[1])],
    "eigvals_centered": [(("_eigh_centered",),    lambda e: e[0])],
    "eigvecs_centered": [(("_eigh_centered",),    lambda e: e[1])],
    "mean":             [(("samples", "mask"),    lambda X, m: X[m.bool()].float().mean(0)),
                         (("samples",),           lambda X: X.float().mean(0))],
    "lvecs":            [(("samples", "eigvals", "eigvecs", "n"),
                          lambda X, l, V, n: X.float() @ V / (l * n).sqrt())],
    "samples":          [(("lvecs", "eigvals", "eigvecs", "n"),
                          lambda U, l, V, n: (U * (l * n).sqrt()) @ V.T)],
}


def _build_tree(acc: "DataAccessor", leaves: list[str]) -> "Node":
    """The leaf tree as actual Nodes (structure only, no data); root has path ''."""
    root = Node(acc, "", {})
    for leaf in leaves:
        node = root
        for seg in leaf.split("."):
            if seg not in node._children:
                p = f"{node._path}.{seg}" if node._path else seg
                node._children[seg] = Node(acc, p, {})
            node = node._children[seg]
    return root


class DataAccessor:
    """Format-agnostic reader for collected data. Read via the tree (`.v` / `acc[path]`)."""

    def __init__(self, data: dict | str, *, weights=None, config: ModelConfig | None = None,
                 derive: bool = True) -> None:
        self.data: dict = torch.load(data, map_location="cpu", weights_only=False) if isinstance(data, str) else data
        self._weights = weights
        self._config = config
        self._derive = derive
        self._cache: dict[tuple[str, str, str], Component | tuple[torch.Tensor, torch.Tensor]] = {}
        self._resolving: set[tuple[str, str, str]] = set()   # cycle guard
        present = [k for k in self.data if not k.startswith("__")]
        self.v = _build_tree(self, sorted(set(present) | derived_leaves(present)))  # present + derivable

    # --- read entry: the only public read path ---
    def __getitem__(self, path: str) -> "Node":
        node = self.v
        for seg in filter(None, path.split(".")):
            node = node[seg]
        return node

    # --- resolver: _resolve memoizes + guards cycles; _produce runs the chain ---
    def _resolve(self, leaf: str, q: str, component: str) -> Component | tuple[torch.Tensor, torch.Tensor] | None:
        key = (leaf, q, component)
        if key in self._cache:
            return self._cache[key]
        if key in self._resolving:
            return None                                   # cycle cutoff (transient — never cached)
        self._resolving.add(key)
        try:
            result = self._produce(leaf, q, component)
        finally:
            self._resolving.discard(key)
        if result is not None:                            # cache positives only
            self._cache[key] = result
        return result

    def _produce(self, leaf: str, q: str, component: str) -> Component | tuple[torch.Tensor, torch.Tensor] | None:
        for inputs, fn in self._producers(leaf, q, component):
            args = [self._resolve(*i) for i in inputs]
            if all(a is not None for a in args) and (out := fn(*args)) is not None:
                return out
        return None

    def _producers(self, leaf: str, q: str, component: str) -> Iterator[tuple[tuple[tuple[str, str, str], ...], Callable]]:
        """(inputs, fn) in priority: stored, then convert, then derive (cross-leaf)."""
        leaf = canonical(leaf, self.data)                 # alias -> stored name (the one place)
        stored = self.data.get(leaf, {}).get(f"{q}_{component}")
        if stored is not None:
            yield (), lambda: stored
        for src, fn in CONVERSIONS.get(component, ()):
            yield tuple((leaf, q, s) for s in src), fn
        if (self._config and (d := derivation(self._config, leaf, q)) and self._weights
                and (ingr := d[2].ingredients(self._weights)) is not None):
            src_leaf, src_q, T = d
            for sources, fn in T.recipes().get(component, []):
                yield tuple((src_leaf, src_q, s) for s in sources), partial(fn, *ingr)

    # --- public surface: not this step ---
    def save(self, *a, **k):
        raise NotImplementedError
    def prewarm(self, components: list[str] | None = None) -> None:
        raise NotImplementedError
    def needs_model_weights(self) -> bool:
        raise NotImplementedError
    def info(self) -> str:
        raise NotImplementedError


class Node:
    """A node in the (data-less) leaf tree, holding its children. Descend by attribute
    / index; `.acts` / `.grads` yield a View; `.children()` lists sub-nodes."""

    def __init__(self, acc: DataAccessor, path: str, children: dict[str, "Node"]) -> None:
        self._acc = acc
        self._path = path
        self._children = children

    def __getattr__(self, seg: str) -> "Node | View":
        if seg.startswith("_"):
            raise AttributeError(seg)
        if seg in ("acts", "grads"):
            return View(self._acc, self._path, seg)
        child = self._children.get(seg)
        if child is None:
            raise AttributeError(f"{self._path!r} has no child {seg!r}")
        return child

    def __getitem__(self, seg: str) -> "Node":
        return self._children[str(seg)]

    def children(self) -> list["Node"]:
        return list(self._children.values())

    def __repr__(self) -> str:
        return f"Node({self._path!r})"


class View:
    """One (leaf, quantity). `view.<component>` -> resolved tensor/tuple, or None."""

    def __init__(self, acc: DataAccessor, leaf: str, q: str) -> None:
        self._acc = acc
        self._leaf = leaf
        self._q = q

    def __getattr__(self, component: str) -> Component | None:
        if component.startswith("_"):
            raise AttributeError(component)
        return self._acc._resolve(self._leaf, self._q, component)

    def __repr__(self) -> str:
        return f"View({self._leaf!r}, {self._q!r})"
