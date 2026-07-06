"""Read, resolve, convert, and save collected data. See docs/accessor_interface.md.

Axis: leaf (dotted hook path) . quantity (acts|grads) . component (cov/eigvals/eigvecs/...),
optionally in a reference's basis (a projection) or paired with a partner view (a cross).
`_resolve(leaf, q, component, reference, partner)` produces any component: stored -> convert ->
derive; projection: store -> rotate -> convert; cross: pair-valid components only (CROSSES).
"""

import argparse
import os
import re
import torch
from glob import glob
from collections.abc import Iterator
from functools import partial, reduce
from typing import Callable

from utils.gpu import prefer_gpu, require_gpu, decomp_profiler, run_pipeline
from utils.hook_names import canonical
from utils.model_registry import (ModelConfig, WeightProvider, derivation, derivable_slots,
                                  is_derivable, load_inference, MLP_OUT, HEAD_CONTRIB)

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

SourceIdentity = tuple[str | None, str | None, str | None]   # (model, revision, run); a plain tuple
ViewId = tuple[SourceIdentity, Leaf, Quantity]               # so projection keys pickle portably
Address = tuple[Leaf, Quantity, CompName, "View | None", "View | None"]  # producer input / resolve call (+ reference, + cross partner)
Key = tuple[Leaf, Quantity, CompName, ViewId | None, ViewId | None]      # resolve cache key: Address, views reduced to their ids
Conversion = tuple[tuple[CompName, ...], Callable[..., Component]]   # (source components, build fn)
Cross = tuple[CompName, tuple[CompName, ...], Callable[..., Component | None]]  # (partner component, self sources, build fn; None = fail)
Producer = tuple[tuple[Address, ...], Callable[..., Component]]  # (input addresses, build fn)

COMPONENTS = (
    "gram", "n", "mask", "samples", "mean", "lvecs",
    "cov", "cov_centered",
    "eigvals", "eigvecs", "eigvals_centered", "eigvecs_centered",
)
QUANTITIES = ("acts", "grads")


@require_gpu
@decomp_profiler
def _eigh(cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One decomposition shared by eigvals + eigvecs; descending, eigvals clamped >= 0.
    cusolver fails to converge on (near-)zero matrices — real inputs, e.g. nanochat's
    zero-initialized c_proj outputs at step 0 — so fall back to CPU LAPACK."""
    try:
        vals, vecs = torch.linalg.eigh(cov)
    except torch.linalg.LinAlgError:  # type: ignore[reportPrivateImportUsage]  # public name, stub marks it private
        vals, vecs = torch.linalg.eigh(cov.cpu())
        vals, vecs = vals.to(cov.device), vecs.to(cov.device)
    return vals.flip(0).clamp(min=0).contiguous(), vecs.flip(1).contiguous()

@require_gpu
@decomp_profiler
def eigvalsh_descending(M: torch.Tensor) -> torch.Tensor:
    """Public GPU-funneled eigvalsh (descending, clamped >= 0) for one-off symmetric matrices
    (e.g. compute_metrics' generalized eigenproblem) — goes through the same GPU lock.
    Same cusolver-vs-degenerate-matrix fallback as _eigh."""
    try:
        vals = torch.linalg.eigvalsh(M)
    except torch.linalg.LinAlgError:  # type: ignore[reportPrivateImportUsage]  # public name, stub marks it private
        vals = torch.linalg.eigvalsh(M.cpu()).to(M.device)
    return vals.flip(0).clamp(min=0)

def _cov(samples: torch.Tensor) -> torch.Tensor:
    X = samples.float()
    return X.T @ X / X.shape[0]

@require_gpu
@decomp_profiler
def _svd(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One SVD; lvecs comes off it (and eigvecs/eigvals could, secondarily). X = U·diag(S)·Vᵀ."""
    U, S, Vt = torch.linalg.svd(samples.float(), full_matrices=False)
    return U, S, Vt.T

PRECISIONS: dict[CompName, torch.dtype] = {"eigvals": torch.float64, "eigvals_centered": torch.float64}

def _cast(c: CompName, t: AtomicComponent) -> AtomicComponent:
    """Storage cast: per-component dtype (PRECISIONS), default fp32. Eigvals stay fp64 (cheap, (d,))."""
    return t.to(PRECISIONS.get(c, torch.float32)) if isinstance(t, torch.Tensor) else t


class _AbortWeights(WeightProvider):
    """A WeightProvider that refuses to load — the `abort_on_model_load` switch."""
    def weight(self, sd_prefix: str): raise RuntimeError("refusing to load model weights (abort_on_model_load)")
    def norm(self): raise RuntimeError("refusing to load model weights (abort_on_model_load)")


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
MATERIALIZE_PRESETS = {"b": MLP_OUT, "o": HEAD_CONTRIB}   # leaf regexes whose derivable slots ± targets

# Projection of a quantity into a reference basis: only the rotations (basis injected) are
# special; eigvals/eigvecs/... derive from them through CONVERSIONS, threaded with the reference
# (rotation commutes with centering, so projected cov_centered comes from projected cov + mean).
PROJECTIONS: dict[CompName, list[Conversion]] = {
    "cov":  [(("cov",),  lambda basis, cov: basis.T @ cov.to(basis.dtype) @ basis)],
    "mean": [(("mean",), lambda basis, m: basis.T @ m.to(basis.dtype))],
}

# Cross of two views, A.cross(B): E[a bᵀ], rows = A, columns = B (partner, injected like a
# projection's basis). Non-symmetric ⇒ no eigh chain: only these components exist; crossed `n`
# doubles as the row-alignment guard.
CROSSES: dict[CompName, list[Cross]] = {
    "n":            [("n",       ("n",),           lambda pn, n: n if n == pn else None)],
    "cov":          [("samples", ("samples", "n"), lambda Y, X, n: X.float().T @ Y.float() / n)],
    "cov_centered": [("mean",    ("cov", "mean"),  lambda pm, c, m: c - torch.outer(m, pm))],
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
        self._stored = {(l, q) for l in self._present for q in QUANTITIES
                        if any(k.startswith(f"{q}_") for k in self.data[l] if not k.startswith("__"))}
        self._slots = self._stored | derivable_slots(self._stored)             # all reachable (leaf, q)
        self._leaves = sorted({l for l, _ in self._slots} | self._present)     # tree navigates leaves
        self.v = _build_tree(self, self._leaves)

    @property
    def _identity(self) -> SourceIdentity:
        return (self.data.get("__hf_model__"), self.data.get("__revision__"), self.data.get("__run__"))

    def stamp(self, identity: SourceIdentity | None = None, **metadata) -> "DataAccessor":
        """Write metadata — identity + any `__key__` entries — at init or any time after. Chainable."""
        if identity:
            self.data["__hf_model__"], self.data["__revision__"], self.data["__run__"] = identity
        self.data |= {f"__{k}__": v for k, v in metadata.items() if v is not None}
        return self

    # --- read entry: the only public read path (canonical = navigate Pythia aliases) ---
    def __getitem__(self, path: Leaf) -> "Node | View":
        node: "Node | View" = self.v
        for seg in filter(None, canonical(path, self.data).split(".")):
            assert isinstance(node, Node), f"{path!r}: only the final segment may be a quantity"
            node = node[seg]
        return node

    # --- resolver: public _resolve yields an atomic component; _resolve_aux is the recursive
    #     engine (memoizes, guards cycles) whose intermediates may be tuples (_eigh/_svd) ---
    def _resolve(self, leaf: Leaf, q: Quantity, component: CompName,
                 reference: "View | None" = None, partner: "View | None" = None) -> AtomicComponent | None:
        r = self._resolve_aux(leaf, q, component, reference, partner)
        assert not isinstance(r, tuple), f"{component!r} resolved to a non-atomic component"
        return r

    def _resolve_aux(self, leaf: Leaf, q: Quantity, component: CompName,
                     reference: "View | None" = None, partner: "View | None" = None) -> Component | None:
        key: Key = (leaf, q, component, reference.identity if reference is not None else None,
                    partner.identity if partner is not None else None)
        if key in self._cache:
            return self._cache[key]
        if key in self._resolving:
            return None                                   # cycle cutoff (transient — never cached)
        self._resolving.add(key)
        result = None
        try:
            for inputs, fn in self._producers(leaf, q, component, reference, partner):
                args = [self._resolve_aux(*i) for i in inputs]
                if all(a is not None for a in args) and (out := fn(*args)) is not None:
                    result = out
                    break
        finally:
            self._resolving.discard(key)
        if result is not None:                            # cache positives only
            self._cache[key] = result
        return result

    def _reachable(self, leaf: Leaf, q: Quantity) -> bool:
        """Is `leaf.q` resolvable? Probes `n` — present for every quantity, never a decomposition."""
        return self._resolve(leaf, q, "n") is not None

    def _producers(self, leaf: Leaf, q: Quantity, component: CompName, reference: "View | None",
                   partner: "View | None" = None) -> Iterator[Producer]:
        """(inputs, fn) by priority. Plain: stored → convert → derive (cross-leaf).
        reference set: persisted store → rotation (PROJECTIONS) → convert, threaded.
        partner set: pair-valid cross components only (CROSSES)."""
        leaf = canonical(leaf, self.data)                 # alias -> stored name (the one place)
        if partner is not None:
            yield from self._cross_producers(leaf, q, component, partner)
            return
        if reference is not None:
            yield from self._projection_producers(leaf, q, component, reference)
            return
        stored = self.data.get(leaf, {}).get(f"{q}_{component}")
        if stored is not None:
            yield (), lambda: stored
        for src, fn in CONVERSIONS.get(component, ()):
            yield tuple((leaf, q, s, None, None) for s in src), prefer_gpu(fn)
        if (self._config and (d := derivation(self._config, leaf, q)) and self._weights
                and (ingr := d[2].ingredients(self._weights)) is not None):
            src_leaf, src_q, T = d
            for sources, fn in T.recipes().get(component, []):
                yield tuple((src_leaf, src_q, s, None, None) for s in sources), partial(prefer_gpu(fn), *ingr)

    def _projection_producers(self, leaf: Leaf, q: Quantity, component: CompName,
                              reference: View) -> Iterator[Producer]:
        """Projection of `leaf.q` into `reference`'s basis: persisted store → rotation → convert."""
        persisted = self.data.get(leaf, {}).get(f"__{q}_projections__", {}).get(reference.identity, {})
        if component in persisted:
            yield (), lambda: persisted[component]
        if (basis := reference.eigvecs) is not None:
            for src, fn in PROJECTIONS.get(component, ()):
                yield tuple((leaf, q, s, None, None) for s in src), partial(prefer_gpu(fn), basis)
        for src, fn in CONVERSIONS.get(component, ()):
            yield tuple((leaf, q, s, reference, None) for s in src), prefer_gpu(fn)

    def _cross_producers(self, leaf: Leaf, q: Quantity, component: CompName,
                         partner: View) -> Iterator[Producer]:
        """Cross of `leaf.q` (rows) with `partner` (columns). A source component keeps the
        partner iff it is a different cross component — so crossed cov pulls crossed n (the
        alignment guard) but plain self samples/means; the partner side is injected."""
        for pcomp, src, fn in CROSSES.get(component, ()):
            if (pv := getattr(partner, pcomp)) is not None:
                yield (tuple((leaf, q, s, None, partner if s in CROSSES and s != component else None)
                             for s in src), partial(prefer_gpu(fn), pv))

    # --- prewarm: fill the cache (notably the eigendecompositions) before metric compute ---
    def prewarm(self, *components: CompName) -> None:
        for leaf, q in self._slots:
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

    def _slot_keys(self, leaf: Leaf, q: Quantity, fmt: Format, exclude: "DataAccessor | None" = None) -> dict:
        """`{q}_{component}` keys (fp32) for the slot that resolve here but not in `exclude`."""
        return {f"{q}_{c}": _cast(c, v) for c in FORMATS[fmt]
                if (exclude is None or exclude._resolve(leaf, q, c) is None)
                and (v := self._resolve(leaf, q, c)) is not None}

    def _materialize(self, fmt: Format, overrides: tuple[str, ...] = ()) -> dict:
        """Per (leaf, q): stored slots, then derived slots a reload of them can't reproduce
        (lossy-format orphans); ± overrides force / drop the derivable slots at matching
        leaves. Projection (`__…__`) stores ride along with their leaf."""
        out = {l: {k: v for k, v in self.data[l].items() if k.startswith("__")} for l in self._present}
        def write(slots, exclude=None):
            for l, q in slots: out.setdefault(l, {}).update(self._slot_keys(l, q, fmt, exclude))
        write(self._stored)
        write(self._slots - self._stored, DataAccessor(out, weights=self._weights, config=self._config))
        for sign, *rest in overrides:
            rx = MATERIALIZE_PRESETS.get("".join(rest)) or re.compile("".join(rest))
            slots = {(l, q) for l in self._leaves for q in QUANTITIES if rx.search(l) and is_derivable(l, q)}
            if sign == "+":
                write(slots)
            else:
                for l, q in slots:
                    for k in [k for k in out.get(l, {}) if k.startswith(f"{q}_")]: del out[l][k]
        return {l: e for l, e in out.items() if any(not k.startswith("__") for k in e)}

    def needs_model_weights(self) -> bool:
        return bool(self._slots - self._stored)

    def map(self, fn: Callable[["View"], object]) -> list:
        """Apply fn to every reachable (leaf, quantity) view — a shader over the data;
        returns fn's results. e.g. project onto a reference checkpoint `ref`:
        `acc.map(lambda v: (r := ref.v.get(f'{v.leaf}.{v.q}')) and v.in_basis(r).persist('eigvals'))`."""
        return [fn(View(self, self._identity, leaf, q)) for leaf, q in sorted(self._slots)]

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
        if seg in QUANTITIES:                              # strict: raise if unreachable (catches typos)
            if not self._acc._reachable(self._path, seg):
                raise AttributeError(f"{self._path!r} has no {seg}")
            return View(self._acc, self._acc._identity, self._path, seg)
        child = self._children.get(seg)
        if child is None:
            raise AttributeError(f"{self._path!r} has no child {seg!r}")
        return child

    def __getitem__(self, seg: str) -> "Node | View":
        return self.__getattr__(str(seg))    # index mirrors attribute: child Node, or View for a quantity

    def get(self, rel: str) -> "Node | View | None":
        """Soft dotted lookup — the metric engine's door. Quantity-terminated ('in.acts') ->
        View, resolved against the accessor (Pythia aliases included via _reachable) so the
        leaf needn't be a tree node; otherwise -> the Node at the relative path ('' = self);
        None when unreachable/absent."""
        *segs, last = rel.split(".")
        if last not in QUANTITIES:
            return reduce(lambda n, s: n and n._children.get(s), filter(None, (*segs, last)), self)
        leaf = ".".join(filter(None, (self._path, *segs)))
        return (View(self._acc, self._acc._identity, leaf, last)
                if self._acc._reachable(leaf, last) else None)

    def children(self) -> list["Node"]:
        return list(self._children.values())

    @property
    def root(self) -> "Node":
        """The tree root (path '') — lets node-fired metrics reach any other leaf."""
        return self._acc.v

    @property
    def path(self) -> str:
        """This node's dotted path (the metric engine keys results by it)."""
        return self._path

    def __repr__(self) -> str:
        return f"Node({self._path!r})"


class View:
    """One (leaf, quantity). `identity` = (source_identity, leaf, q) names it without the
    accessor (e.g. a cached projection reference). `view.<component>` -> tensor/tuple/None.
    `reference` set => this quantity projected into that reference's basis.
    `partner` set => the cross of this quantity (rows) with the partner (columns)."""

    def __init__(self, acc: "DataAccessor | None", source: SourceIdentity, leaf: Leaf, q: Quantity,
                 reference: "View | None" = None, partner: "View | None" = None) -> None:
        self._acc = acc
        self._source = source
        self._leaf = leaf
        self._q = q
        self.reference = reference                          # set => a projection
        self.partner = partner                              # set => a cross

    @property
    def identity(self) -> ViewId:
        return (self._source, self._leaf, self._q)

    @property
    def leaf(self) -> Leaf:
        return self._leaf

    @property
    def q(self) -> Quantity:
        return self._q

    def __getattr__(self, component: CompName) -> AtomicComponent | None:
        if component.startswith("_"):
            raise AttributeError(component)
        return (self._acc._resolve(self._leaf, self._q, component, self.reference, self.partner)
                if self._acc is not None else None)

    def in_basis(self, reference: "View") -> "View":
        return View(self._acc, self._source, self._leaf, self._q, reference)

    def cross(self, partner: "View") -> "View":
        """Cross with `partner`: cov = E[x yᵀ], rows = self. Needs row-aligned samples (same
        run + token selection); components per CROSSES; never persisted."""
        assert self.reference is None and partner.reference is None, "cross of projected views unsupported"
        return View(self._acc, self._source, self._leaf, self._q, partner=partner)

    def persist(self, *components: CompName) -> "View":
        """Write the listed (computed) components of this projection into its on-disk store."""
        if self.reference is None or self._acc is None:
            raise ValueError("persist() needs a projected View bound to an accessor (use in_basis)")
        store = self._acc.data.setdefault(self._leaf, {}).setdefault(
            f"__{self._q}_projections__", {}).setdefault(self.reference.identity, {})
        store |= {c: v for c in components if (v := getattr(self, c)) is not None}
        return self

    def projections(self) -> list["View"]:
        """This quantity's persisted projections, each a readable projected View."""
        store = self._acc.data.get(self._leaf, {}).get(f"__{self._q}_projections__", {}) if self._acc else {}
        return [self.in_basis(View(None, *ref_id)) for ref_id in store]

    def __repr__(self) -> str:
        onto = f" onto {self.reference._leaf}@{self.reference._source[2]}" if self.reference else ""
        by = f" x {self.partner._leaf}.{self.partner._q}" if self.partner else ""
        return f"View({self._leaf!r}, {self._q!r}{onto}{by})"


# --- CLI helpers (module-level; no per-program state) ---
def _pt_files(path: str) -> list[str]:
    return [path] if path.endswith(".pt") else sorted(glob(os.path.join(path, "**", "*.pt"), recursive=True))

def _out_path(f: str, input: str, output_dir: str | None, output: str | None) -> str:
    """Destination for input `f`: mirror under --output-dir, else --output, else in-place."""
    root = input if os.path.isdir(input) else os.path.dirname(input)
    return os.path.join(output_dir, os.path.relpath(f, root)) if output_dir else (output or f)

def _open(path: str, derive: bool = True, abort: bool = False) -> "DataAccessor":
    """Accessor for a stored file; derive=False -> no weights (skip derivable slots), abort=True -> raise on a weight load."""
    if abort:
        return DataAccessor(path, abort_on_model_load=True)
    data, config, weights = load_inference(path, derive)
    return DataAccessor(data, config=config, weights=weights)

def _run_pool(worker: Callable[[str], str], files: list[str], workers: int) -> None:
    run_pipeline(files, lambda f: print(f"  -> {worker(f)}", flush=True), workers=workers)


# --- CLI: each subcommand owns its parser args + a run() taking those args by name ---
class AccessorCLIProgram:
    """One CLI subcommand: set `name`, extend args() (super() = the defaults), define run()."""
    name = ""

    def args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--input", required=True, help=".pt file or directory (recursed)")
        p.add_argument("--output", help="output .pt (single-file input only)")
        p.add_argument("--output-dir", dest="output_dir", help="mirror the input tree here; default in-place")
        p.add_argument("--no-derive", action="store_true",
                       help="do everything possible without weights; skip derivable (B/O/final-norm) slots")
        p.add_argument("--abort-on-model-load", dest="abort", action="store_true",
                       help="raise if a derivation would load weights (guard: this shouldn't need a model)")

    def __call__(self, a: argparse.Namespace) -> None:
        self.run(**{k: v for k, v in vars(a).items() if k != "cmd"})

    def run(self, **_) -> None:
        raise NotImplementedError


class Info(AccessorCLIProgram):
    name = "info"

    def args(self, p: argparse.ArgumentParser) -> None:   # read-only: only --input
        p.add_argument("--input", required=True, help=".pt file or directory (recursed)")

    def run(self, *, input: str) -> None:
        for f in _pt_files(input):
            print(f"=== {f} ===\n{DataAccessor(f).info()}\n")


class Convert(AccessorCLIProgram):
    name = "convert"

    def args(self, p: argparse.ArgumentParser) -> None:
        super().args(p)
        p.add_argument("--to", required=True, help=f"target format, one of {sorted(FORMATS)}")
        p.add_argument("--materialize", default="", metavar='"+b -o"',
                       help="space-separated ±preset/regex overrides for derived families")
        p.add_argument("--workers", type=int, default=1, help="process this many files in parallel")

    def run(self, *, input: str, output: str | None, output_dir: str | None, no_derive: bool,
            abort: bool, to: str, materialize: str, workers: int) -> None:
        if to not in FORMATS:
            raise SystemExit(f"unknown format {to!r}; choose from {sorted(FORMATS)}")
        overrides = tuple(materialize.split())

        def convert(f: str) -> str:
            _open(f, not no_derive, abort).save(_out_path(f, input, output_dir, output), format=to, overrides=overrides)
            return f

        files = _pt_files(input)
        _run_pool(convert, files, workers)


class Project(AccessorCLIProgram):
    name = "project"

    def args(self, p: argparse.ArgumentParser) -> None:
        super().args(p)
        p.add_argument("--onto-file", dest="onto_file", required=True,
                       help="reference checkpoint .pt; persists each view's eigvals in its basis")
        p.add_argument("--no-ref-derive", dest="no_ref_derive", action="store_true",
                       help="don't load the reference's weights; project only onto its stored slots")
        p.add_argument("--workers", type=int, default=1, help="process this many files in parallel")

    def run(self, *, input: str, output: str | None, output_dir: str | None, no_derive: bool,
            abort: bool, onto_file: str, no_ref_derive: bool, workers: int) -> None:
        def project(f: str) -> str:
            ref = _open(onto_file, not no_ref_derive)
            acc = _open(f, not no_derive, abort)
            acc.map(lambda v: isinstance(r := ref.v.get(f"{v.leaf}.{v.q}"), View)
                    and v.in_basis(r).persist("eigvals"))
            acc.save(_out_path(f, input, output_dir, output))
            return f

        files = _pt_files(input)
        _run_pool(project, files, workers)


PROGRAMS = [Info(), Convert(), Project()]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m utils.accessor", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for p in PROGRAMS:
        p.args(sub.add_parser(p.name))
    args = ap.parse_args()
    {p.name: p for p in PROGRAMS}[args.cmd](args)
