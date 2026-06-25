"""DataAccessor rebuild (WIP) — read/resolve/save core. See docs/accessor_interface.md.

Axis: leaf (dotted hook path) . quantity (acts|grads) . component (cov/eigvals/eigvecs/...),
optionally in a reference's basis (a projection). `_resolve(leaf, q, component, reference)`
produces any component: stored -> convert -> derive; projection: store -> rotate -> convert.
"""

import os
import re
import time
import torch
from collections import namedtuple
from collections.abc import Iterator
from functools import partial, wraps
from typing import Callable

from utils.hook_names import canonical
from utils.model_registry import ModelConfig, WeightProvider, derivation, derived_leaves, MLP_OUT, HEAD_CONTRIB

AtomicComponent = torch.Tensor | int | float
Component = AtomicComponent | tuple[AtomicComponent, ...]

Leaf = str
"""Dot-delimited path to a Leaf"""
Quantity = str
"""One of ("acts","grads") (essentially enum of QUANTITIES)"""
CompName = str
"""String name of a Component"""
Format = str
"""String name of a format (set of components to save)"""

SourceIdentity = namedtuple("SourceIdentity", "model revision run")

ViewId = tuple[SourceIdentity, Leaf, Quantity]            # a View's identity
Address = tuple[Leaf, Quantity, CompName, "View | None"]  # producer input / resolve call (+ reference)
Key = tuple[Leaf, Quantity, CompName, ViewId | None]      # resolve cache key: Address, reference reduced to its id
Conversion = tuple[tuple[CompName, ...], Callable[..., Component]]   # (source components, build fn)
Producer = tuple[tuple[Address, ...], Callable[..., Component]]  # (input addresses, build fn)

COMPONENTS = (
    "gram", "n", "mask", "samples", "mean", "lvecs",
    "cov", "cov_centered",
    "eigvals", "eigvecs", "eigvals_centered", "eigvecs_centered",
)
QUANTITIES = ("acts", "grads")

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_GPU_EXPECTED = bool(os.environ.get("CUDA_VISIBLE_DEVICES")) or torch.cuda.is_available()


class _DecompProfiler:
    """Times the decompositions it decorates, when enabled (off by default → no overhead)."""
    def __init__(self):
        self.enabled = False
        self.calls: list[tuple[str, tuple, float]] = []

    def enable(self): self.enabled, self.calls = True, []
    def disable(self): self.enabled = False

    def __call__(self, fn: Callable) -> Callable:
        @wraps(fn)
        def timed(M):
            if not self.enabled:
                return fn(M)
            t = time.time()
            out = fn(M)
            self.calls.append((fn.__name__, tuple(M.shape), time.time() - t))
            return out
        return timed

    def summary(self) -> str:
        total = sum(d for *_, d in self.calls)
        return "\n".join([f"{len(self.calls)} decompositions, {total:.2f}s total:",
                          *(f"  {n} {s}: {d:.3f}s" for n, s, d in self.calls)])

decomp_profiler = _DecompProfiler()


def _to_cpu(r: Component) -> Component:
    return r.cpu() if isinstance(r, torch.Tensor) else tuple(t.cpu() for t in r) if isinstance(r, tuple) else r


def prefer_gpu(fn: Callable) -> Callable:
    """Move tensor args to _DEV (a no-op when it's already CPU), run, bring the result back."""
    @wraps(fn)
    def wrapped(*args):
        return _to_cpu(fn(*(a.to(_DEV) if isinstance(a, torch.Tensor) else a for a in args)))
    return wrapped


def require_gpu(fn: Callable) -> Callable:
    """prefer_gpu, but refuse a large CPU run when a GPU is expected (the forked-worker regression)."""
    run = prefer_gpu(fn)
    @wraps(fn)
    def wrapped(M: torch.Tensor):
        if _GPU_EXPECTED and not torch.cuda.is_available() and M.shape[-1] >= 1024:
            raise RuntimeError(f"eigendecomp {tuple(M.shape)} on CPU while a GPU is present — refusing")
        return run(M)
    return wrapped


@require_gpu
@decomp_profiler
def _eigh(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One decomposition shared by eigvals + eigvecs; descending, eigvals clamped >= 0."""
    vals, vecs = torch.linalg.eigh(cov)
    return vals.flip(0).clamp(min=0).contiguous(), vecs.flip(1).contiguous()

def _cov(samples: torch.Tensor) -> torch.Tensor:
    X = samples.float()
    return X.T @ X / X.shape[0]

@require_gpu
@decomp_profiler
def _svd(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One SVD; lvecs comes off it (and eigvecs/eigvals could, secondarily). X = U·diag(S)·Vᵀ."""
    U, S, Vt = torch.linalg.svd(samples.float(), full_matrices=False)
    return U, S, Vt.T

def _fp32(t: AtomicComponent) -> AtomicComponent:
    return t.float() if isinstance(t, torch.Tensor) else t


class _AbortWeights:
    """A WeightProvider that refuses to load — the `abort_on_model_load` switch."""
    def __getattr__(self, _): raise RuntimeError("refusing to load model weights (abort_on_model_load)")


# component -> [(source components, fn)], tried in priority order. `_eigh` is private
# (View blocks `_`), kept so eigvals + eigvecs share one decomposition.
CONVERSIONS: dict[CompName, list[Conversion]] = {
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
    "_svd":             [(("samples",), _svd)],
    "lvecs":            [(("_svd",),                 lambda s: s[0]),    # primary: one shared svd
                         (("samples", "eigvals", "eigvecs", "n"),
                          lambda X, l, V, n: X.float() @ V / (l * n).sqrt())],
    "samples":          [(("lvecs", "eigvals", "eigvecs", "n"),
                          lambda U, l, V, n: (U * (l * n).sqrt()) @ V.T)],
}

FORMATS: dict[Format, tuple[CompName, ...]] = {
    "acts":        ("samples", "mask", "mean", "n"),
    "acts_svd":    ("lvecs", "eigvals", "eigvecs", "mask", "mean", "n"),
    "cov":         ("gram", "mean", "n"),
    "cov_svd":     ("eigvals", "eigvecs", "eigvals_centered", "mean", "n"),
    "eigenvalues": ("eigvals", "eigvals_centered", "mean", "n"),
}
MATERIALIZE_PRESETS = {"b": MLP_OUT, "o": HEAD_CONTRIB}   # derived-leaf families for save(materialize=…)

# Projection of a quantity into a reference basis: only `cov` (the rotation, basis injected) is
# special; eigvals/eigvecs/... derive from it through CONVERSIONS, threaded with the reference.
PROJECTIONS: dict[CompName, list[Conversion]] = {
    "cov": [(("cov",), lambda basis, cov: basis.T @ cov.to(basis.dtype) @ basis)],
}


def _build_tree(acc: "DataAccessor", leaves: list[Leaf]) -> "Node":
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

    def __init__(self, data: dict | str, *, identity: SourceIdentity | None = None,
                 weights: WeightProvider | None = None, config: ModelConfig | None = None,
                 abort_on_model_load: bool = False) -> None:
        self.data: dict = torch.load(data, map_location="cpu", weights_only=False) if isinstance(data, str) else data
        self.stamp(identity)
        self._weights = _AbortWeights() if abort_on_model_load else weights
        self._config = config
        self._cache: dict[Key, Component] = {}
        self._resolving: set[Key] = set()                     # cycle guard
        self._present = {k for k in self.data if not k.startswith("__")}
        self._leaves = sorted(self._present | derived_leaves(self._present))   # present + derivable
        self.v = _build_tree(self, self._leaves)

    @property
    def _identity(self) -> SourceIdentity:
        return SourceIdentity(self.data.get("__hf_model__"), self.data.get("__revision__"), self.data.get("__run__"))

    def stamp(self, identity: SourceIdentity | None = None, **metadata) -> "DataAccessor":
        """Write metadata — identity + any `__key__` entries — at init or any time after. Chainable."""
        if identity:
            self.data["__hf_model__"], self.data["__revision__"], self.data["__run__"] = identity
        self.data |= {f"__{k}__": v for k, v in metadata.items() if v is not None}
        return self

    # --- read entry: the only public read path ---
    def __getitem__(self, path: Leaf) -> "Node":
        node = self.v
        for seg in filter(None, path.split(".")):
            node = node[seg]
        return node

    # --- resolver: memoizes, guards cycles, runs the producer chain ---
    def _resolve(self, leaf: Leaf, q: Quantity, component: CompName,
                 reference: "View | None" = None) -> Component | None:
        key: Key = (leaf, q, component, reference.identity if reference is not None else None)
        if key in self._cache:
            return self._cache[key]
        if key in self._resolving:
            return None                                   # cycle cutoff (transient — never cached)
        self._resolving.add(key)
        result = None
        try:
            for inputs, fn in self._producers(leaf, q, component, reference):
                args = [self._resolve(*i) for i in inputs]
                if all(a is not None for a in args) and (out := fn(*args)) is not None:
                    result = out
                    break
        finally:
            self._resolving.discard(key)
        if result is not None:                            # cache positives only
            self._cache[key] = result
        return result

    def _producers(self, leaf: Leaf, q: Quantity, component: CompName,
                   reference: "View | None") -> Iterator[Producer]:
        """(inputs, fn) by priority. reference=None: stored → convert → derive (cross-leaf).
        reference set: persisted store → rotation (PROJECTIONS) → convert, threaded."""
        leaf = canonical(leaf, self.data)                 # alias -> stored name (the one place)
        if reference is not None:
            yield from self._projection_producers(leaf, q, component, reference)
            return
        stored = self.data.get(leaf, {}).get(f"{q}_{component}")
        if stored is not None:
            yield (), lambda: stored
        for src, fn in CONVERSIONS.get(component, ()):
            yield tuple((leaf, q, s, None) for s in src), prefer_gpu(fn)
        if (self._config and (d := derivation(self._config, leaf, q)) and self._weights
                and (ingr := d[2].ingredients(self._weights)) is not None):
            src_leaf, src_q, T = d
            for sources, fn in T.recipes().get(component, []):
                yield tuple((src_leaf, src_q, s, None) for s in sources), partial(fn, *ingr)

    def _projection_producers(self, leaf: Leaf, q: Quantity, component: CompName,
                              reference: View) -> Iterator[Producer]:
        """Projection of `leaf.q` into `reference`'s basis: persisted store → rotation → convert."""
        persisted = self.data.get(leaf, {}).get(f"__{q}_projections__", {}).get(reference.identity, {})
        if component in persisted:
            yield (), lambda: persisted[component]
        if (basis := reference.eigvecs) is not None:
            for src, fn in PROJECTIONS.get(component, ()):
                yield tuple((leaf, q, s, None) for s in src), partial(prefer_gpu(fn), basis)
        for src, fn in CONVERSIONS.get(component, ()):
            yield tuple((leaf, q, s, reference) for s in src), prefer_gpu(fn)

    # --- prewarm: fill the cache (notably the eigendecompositions) before metric compute ---
    def prewarm(self, components: list[str] | None = None) -> None:
        for leaf in self._leaves:
            for q in QUANTITIES:
                for c in (components or COMPONENTS):
                    self._resolve(leaf, q, c)

    # --- save: a format is just the set of components it writes (FORMATS) ---
    def save(self, path: str, format: Format | None = None, overrides: tuple[str, ...] = ()) -> str:
        assert all(self._identity), f"cannot save {path}: incomplete identity {self._identity} — stamp() it first"
        fmt = format or self.data.get("__format__")
        if fmt is None:
            raise ValueError(f"cannot save {path}: no format given and data has no __format__")
        out = self._materialize(format, overrides) if format else dict(self.data)
        out.update({k: v for k, v in self.data.items() if k.startswith("__")} | {"__format__": fmt})

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(out, path)
        return path

    def _materialize_keys(self, leaf: Leaf, format: Format, exclude: "DataAccessor | None" = None) -> dict:
        """`{q}_{component}` keys for `leaf` (fp32) that resolve here but NOT in `exclude`,
        plus the leaf's `__…__` keys (projection stores) carried verbatim."""
        keys = {f"{q}_{c}": _fp32(v)
                for q in QUANTITIES for c in FORMATS[format]
                if (exclude is None or exclude._resolve(leaf, q, c) is None)
                and (v := self._resolve(leaf, q, c)) is not None}
        return keys | {k: v for k, v in self.data.get(leaf, {}).items() if k.startswith("__")}

    def _materialize(self, format: Format, overrides: tuple[str, ...] = ()) -> dict:
        """Present leaves, plus any derived leaf a reload couldn't reproduce (a lossy
        format's orphans), then `±<regex|preset>` overrides add/remove leaves."""
        out = {leaf: e for leaf in self._present if (e := self._materialize_keys(leaf, format))}
        reload = DataAccessor(out, weights=self._weights, config=self._config)
        out |= {leaf: e for leaf in self._leaves if leaf not in self._present
                if (e := self._materialize_keys(leaf, format, reload))}

        for sign, *rest in overrides:
            name = "".join(rest)
            rx = MATERIALIZE_PRESETS.get(name) or re.compile(name)
            hits = {leaf for leaf in self._leaves if rx.search(leaf)}
            out = out | {leaf: self._materialize_keys(leaf, format) for leaf in hits} if sign == "+" \
                  else {leaf: e for leaf, e in out.items() if leaf not in hits}

        return out

    def needs_model_weights(self) -> bool:
        return len(self._leaves) > len(self._present)

    def info(self) -> str:
        def show(v): return f"{tuple(v.shape)} {v.dtype}" if isinstance(v, torch.Tensor) else v
        meta = [f"{k} = {v}" for k, v in self.data.items() if k.startswith("__")]
        body = [f"{leaf}\n" + "\n".join(f"  {k}: {show(v)}" for k, v in sorted(e.items()) if not k.startswith("__"))
                for leaf, e in sorted(self.data.items()) if not leaf.startswith("__")]
        return "\n".join(meta + body)


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
        if seg in QUANTITIES:
            return View(self._acc, self._acc._identity, self._path, seg)
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
    """One (leaf, quantity). `identity` = (source_identity, leaf, q) names it without the
    accessor (e.g. a cached projection reference). `view.<component>` -> tensor/tuple/None.
    `reference` set => this quantity projected into that reference's basis."""

    def __init__(self, acc: "DataAccessor | None", source: SourceIdentity, leaf: Leaf, q: Quantity,
                 reference: "View | None" = None) -> None:
        self._acc = acc
        self._source = source
        self._leaf = leaf
        self._q = q
        self.reference = reference                          # set => a projection

    @property
    def identity(self) -> ViewId:
        return (self._source, self._leaf, self._q)

    def __getattr__(self, component: CompName) -> AtomicComponent | None:
        if component.startswith("_"):
            raise AttributeError(component)
        return self._acc._resolve(self._leaf, self._q, component, self.reference) if self._acc is not None else None

    def in_basis(self, reference: "View") -> "View":
        return View(self._acc, self._source, self._leaf, self._q, reference)

    def persist(self, *components: CompName) -> "View":
        """Write the listed (computed) components of this projection into its on-disk store."""
        store = self._acc.data.setdefault(self._leaf, {}).setdefault(
            f"__{self._q}_projections__", {}).setdefault(self.reference.identity, {})
        store |= {c: v for c in components if (v := getattr(self, c)) is not None}
        return self

    def projections(self) -> list["View"]:
        """This quantity's persisted projections, each a readable projected View."""
        store = self._acc.data.get(self._leaf, {}).get(f"__{self._q}_projections__", {}) if self._acc else {}
        return [self.in_basis(View(None, *ref_id)) for ref_id in store]

    def __repr__(self) -> str:
        onto = f" onto {self.reference._leaf}@{self.reference._source.run}" if self.reference else ""
        return f"View({self._leaf!r}, {self._q!r}{onto})"
