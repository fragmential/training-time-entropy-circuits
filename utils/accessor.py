"""Read, write, convert, and inspect collected data.

Address axis: a `leaf` (dotted hook path), a `quantity` (acts|grads), a `format`
(cov/eigvals/eigvecs/eigh/eigvals_centered/mean/samples/n/svd). `resolve(leaf,q,fmt)`
greedily produces any format from what's stored: stored -> reformat -> derive;
`can_resolve` is its no-compute twin. Disk keys are uniform `{q}_{fmt}` per leaf.

    acc.v.blk0.mlp.up.in.acts.eigvals
    acc["blk0.attn.head0.slice"].acts.cov
    acc["after_final_norm"].acts.eigvals   # derived: final-norm @ before_final_norm

CLI: info | convert --to | project --onto | set-filter   (python -m utils.accessor).
"""

import os
import re
import time
import torch

from typing import Optional

from utils import hook_names as hn
from utils.model_registry import (
    derivation,
    derived_leaves,
    derivation_sd_prefixes,
    get_model_config,
    get_final_layernorm,
    load_selective_weights,
)


_DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# Eigendecomp helpers
# ===========================================================================

class _DecompProfiler:
    def __init__(self):
        self.calls = []
        self.enabled = False

    def enable(self):
        self.enabled = True
        self.calls = []

    def disable(self):
        self.enabled = False

    def log(self, label, shape, duration):
        if self.enabled:
            self.calls.append((label, shape, duration))

    def summary(self):
        if not self.calls:
            return "  no decompositions recorded"
        total = sum(d for _, _, d in self.calls)
        lines = [f"  {len(self.calls)} decompositions, {total:.1f}s total:"]
        for label, shape, dur in self.calls:
            lines.append(f"    {label} {shape}: {dur:.2f}s")
        return "\n".join(lines)

decomp_profiler = _DecompProfiler()


def eigh_descending(M):
    """Full eigendecomposition, sorted descending, eigenvalues clamped >= 0."""
    t = time.time()
    vals, vecs = torch.linalg.eigh(M)
    decomp_profiler.log(f"eigh[{M.device}]", tuple(M.shape), time.time() - t)
    return vals.flip(0).clamp(min=0).contiguous(), vecs.flip(1).contiguous()

def eigvalsh_descending(M):
    """Eigenvalues only, sorted descending, clamped >= 0."""
    t = time.time()
    v = torch.linalg.eigvalsh(M).flip(0).clamp(min=0)
    decomp_profiler.log(f"eigvalsh[{M.device}]", tuple(M.shape), time.time() - t)
    return v

def reconstruct_cov(eigvals, eigvecs):
    """V @ diag(lambda) @ V.T"""
    return eigvecs @ torch.diag(eigvals.to(eigvecs.dtype)) @ eigvecs.T

def cross_eigvals(cov, basis):
    """Eigenvalues of `cov` expressed in `basis` (columns): eigvalsh(basis.T @ cov @ basis)."""
    return eigvalsh_descending(basis.T @ cov.to(basis.dtype) @ basis)


def _eigh(C, k):
    v = eigvalsh_descending(C)
    return v[:k] if k else v

def _eigh_full(C):
    return eigh_descending(C)

def _from_acts(acts, k):
    mu = acts.mean(0)
    n = acts.shape[0]
    return _eigh((acts - mu).T @ (acts - mu) / n, k), _eigh(acts.T @ acts / n, k)

def _from_cov(cov, mu, k):
    return _eigh(cov - torch.outer(mu, mu), k) if mu is not None else None, _eigh(cov, k)

def get_eigenspectrum(acts=None, cov=None, mu=None, topk=None):
    """Returns (centered, uncentered) torch tensors. centered is None if mu unavailable."""
    return _from_acts(acts, topk) if acts is not None else _from_cov(cov, mu, topk)


# ===========================================================================
# Format algebra (REFORMAT) — pure (leaf,quantity)-internal conversions
# ===========================================================================

def _eigh_from_cov(C):
    vals, vecs = eigh_descending(C.to(_DEV))
    return vals.cpu(), vecs.cpu()

def _eigh_centered(C, mu):
    C = C.to(_DEV)
    mu = mu.to(device=_DEV, dtype=C.dtype)
    vals, vecs = eigh_descending(C - torch.outer(mu, mu))
    return vals.cpu(), vecs.cpu()

def _cov_from_samples(X):
    X = X.float()
    return (X.T @ X) / X.shape[0]

def _mean_from_samples(X):
    return X.float().mean(0)

def _svd_from_samples(X):
    U, S, Vt = torch.linalg.svd(X.float(), full_matrices=False)
    return U, S, Vt.T

def _svd_from_eigh(eigh, n):
    eigvals, eigvecs = eigh
    return (eigvals.clamp(min=0) * n).sqrt().float(), eigvecs

REFORMAT = {
    "cov":              [(("eigvals", "eigvecs"),
                          lambda ev, V: reconstruct_cov(ev.float(), V.float())),
                         (("samples",), _cov_from_samples)],
    "eigh":             [(("cov",), _eigh_from_cov)],
    "eigvals":          [(("eigh",), lambda eh: eh[0])],
    "eigvecs":          [(("eigh",), lambda eh: eh[1])],
    "eigh_centered":    [(("cov", "mean"), _eigh_centered)],
    "eigvals_centered": [(("eigh_centered",), lambda eh: eh[0])],
    "eigvecs_centered": [(("eigh_centered",), lambda eh: eh[1])],
    "mean":             [(("samples",), _mean_from_samples)],
    "svd":              [(("samples",), _svd_from_samples), (("eigh", "n"), _svd_from_eigh)],
}


# ===========================================================================
# Storage format spec parsing
# ===========================================================================

# Which storage modifier flag gates materializing a derived quantity, by leaf suffix.
_DERIVED_FLAG = {".out": "b", ".contrib": "o"}

def _derived_flag(leaf):
    for suf, fl in _DERIVED_FLAG.items():
        if leaf.endswith(suf):
            return fl
    return None

_FORMAT_ALIASES = {
    "activations": "acts",
    "covariance":  "cov",
    "eigvals":     "eigenvalues",
}

_BASE_FORMATS = {"acts", "acts_svd", "cov", "cov_svd", "eigenvalues"}
_VALID_MODIFIERS = {"b", "m", "o"}


def parse_format_spec(fmt):
    """Split a storage format into (base, plus_flags, minus_flags).

    Modifiers are individual chars after +/- signs, so +bm == +b+m == +mb.
    """
    m = re.match(r"([^+-]+)((?:[+-][^+-]+)*)$", fmt)
    if not m:
        raise ValueError(f"Unknown format: {fmt!r}")
    base, mods = m.groups()
    base = _FORMAT_ALIASES.get(base, base)
    if base not in _BASE_FORMATS:
        raise ValueError(f"Unknown format: {fmt!r}")
    plus, minus = set(), set()
    for sign, chars in re.findall(r"([+-])([^+-]+)", mods):
        unknown = set(chars) - _VALID_MODIFIERS
        if unknown:
            raise ValueError(
                f"Unknown format modifier(s) {sorted(unknown)} in {fmt!r}; "
                f"valid modifiers: {sorted(_VALID_MODIFIERS)}"
            )
        for ch in chars:
            if sign == "+":
                plus.add(ch); minus.discard(ch)
            else:
                minus.add(ch); plus.discard(ch)
    return base, plus, minus


def parse_format(fmt):
    """Split a storage format into (base, positive modifier flags)."""
    base, plus, _ = parse_format_spec(fmt)
    return base, plus

_DTYPE_MAP = {
    "float32":  torch.float32,  "fp32": torch.float32,
    "float64":  torch.float64,  "fp64": torch.float64,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16":  torch.float16,  "fp16": torch.float16,
}

_DEFAULT_STORAGE_DTYPES = {
    "eigvals": "fp64",
    "eigvecs": "fp32",
    "acts":    "fp32",
    "cov":     "fp32",
    "mean":    "fp32",
}

_STEP_FILE_RE = re.compile(r"step(\d+)\.pt$")


def _cast(tensor, dtype_str):
    return tensor.to(dtype=_DTYPE_MAP.get(dtype_str, torch.float32))


def _cast_for(tensor, item_type, storage_dtype):
    """Cast tensor using per-item storage dtype rules."""
    if isinstance(storage_dtype, dict):
        dtype_str = storage_dtype.get(item_type, "fp32")
    elif isinstance(storage_dtype, str):
        dtype_str = storage_dtype
    else:
        dtype_str = _DEFAULT_STORAGE_DTYPES.get(item_type, "fp32")
    return _cast(tensor, dtype_str)


def _convert_worker(args):
    in_path, out_path, fmt, abort_on_model_load = args
    return DataAccessor(in_path, abort_on_model_load=abort_on_model_load).save(out_path, format=fmt)

def _project_same_layer_worker(args):
    in_path, out_path, onto = args
    return DataAccessor(in_path).project_same_layer(onto=onto, output_path=out_path)

def _project_onto_file_worker(args):
    in_path, out_path, basis_path = args
    return DataAccessor(in_path).project_onto_basis(basis_path, output_path=out_path)

def _set_filter_worker(args):
    in_path, out_path, filter_dict = args
    return DataAccessor(in_path).set_token_filter(filter_dict, output_path=out_path)


def _resolve_hf_name(short_name):
    """Resolve short model name to full HuggingFace name."""
    if "/" in short_name:
        return short_name
    s = short_name.lower()
    if "pythia" in s:
        return f"EleutherAI/{short_name}"
    if "olmo" in s:
        return f"allenai/{short_name}"
    return short_name


# ===========================================================================
# DataAccessor
# ===========================================================================

class DataAccessor:
    """Format-agnostic reader / writer for collected activation / covariance data."""

    def __init__(self, data, model=None, model_config=None, model_name=None, revision=None,
                 abort_on_model_load: bool = False):
        if model_name is not None and "/" not in model_name:
            model_name = _resolve_hf_name(model_name)
        self.path = data if isinstance(data, str) else None
        if isinstance(data, str):
            data = torch.load(data, map_location="cpu", weights_only=False)
        self.data = data
        self.model = model
        self.model_config = model_config
        self._model_name = model_name or data.get("__hf_model__")
        self._revision = revision or data.get("__revision__")
        self._hf_repo = data.get("__hf_model__")
        self._selective_weights: Optional[dict] = None
        self.abort_on_model_load = abort_on_model_load
        self._cache: dict = {}        # (leaf, q, fmt) -> tensor / tuple / None
        self._resolving: set = set()  # cycle guard for resolve
        self._can_cache: dict = {}
        self._can_resolving: set = set()

    # ------------------------------------------------------------------
    # Configuration / weights
    # ------------------------------------------------------------------

    def _cfg(self):
        if self.model_config is None and self._model_name:
            self.model_config = get_model_config(self._model_name)
        return self.model_config

    def _weights_obtainable(self) -> bool:
        """Cheap flag: are derivation weights reachable? Never loads them."""
        return (self.model is not None or self._selective_weights is not None
                or self._model_name is not None)

    def _ensure_weights(self) -> bool:
        if self.model is not None or self._selective_weights is not None:
            return True
        if self._model_name is None:
            return False
        if self.abort_on_model_load:
            raise RuntimeError(
                f"Aborting before loading model weights for {self._model_name}"
                + (f"@{self._revision}" if self._revision else "")
            )
        cfg = self._cfg()
        leaves = self._all_leaves()
        prefixes = derivation_sd_prefixes(cfg, leaves)
        need_norm = any(
            (d := derivation(cfg, leaf, "acts")) and getattr(d[2], "kind", None) == "norm"
            for leaf in self._derived_leaves()
        )
        if not prefixes and not need_norm:
            return False
        hf_repo = self._hf_repo or cfg.hf_repo
        self._selective_weights = load_selective_weights(
            cfg, hf_repo, self._revision, prefixes, need_norm,
        )
        return True

    def _weight(self, sd_prefix):
        """SimpleNamespace(weight, bias)-like for a linear derivation, or None."""
        if not self._ensure_weights():
            return None
        if self._selective_weights is not None and sd_prefix in self._selective_weights:
            return self._selective_weights[sd_prefix]
        if self.model is not None:
            try:
                return self.model.get_submodule(sd_prefix)
            except AttributeError:
                return None
        return None

    def _norm_module(self, sd_prefix):
        if not self._ensure_weights():
            return None
        if self._selective_weights is not None and "__norm__" in self._selective_weights:
            return self._selective_weights["__norm__"].float()
        if self.model is not None:
            return get_final_layernorm(self.model, self._cfg()).float()
        return None

    # ------------------------------------------------------------------
    # Leaf universe / tree shape
    # ------------------------------------------------------------------

    def leaves(self) -> list:
        """Stored (present) leaf names — excludes metadata keys."""
        return [k for k in self.data if not k.startswith("__")]

    def _derived_leaves(self):
        cfg = self._cfg()
        if cfg is None:
            return set()
        return derived_leaves(self.leaves())

    def _all_leaves(self):
        return set(self.leaves()) | self._derived_leaves()

    def tree_children(self, path):
        """Immediate child segments of `path` among all (present + derivable) leaves."""
        prefix = path + "." if path else ""
        segs = set()
        for leaf in self._all_leaves():
            if leaf == path:
                continue
            if leaf.startswith(prefix) or path == "":
                rest = leaf[len(prefix):]
                if rest:
                    segs.add(rest.split(".", 1)[0])
        return sorted(segs)

    def _canon(self, leaf):
        return hn.canonical(leaf, self.data)

    # ------------------------------------------------------------------
    # The resolver
    # ------------------------------------------------------------------

    def resolve(self, leaf, q, fmt):
        """Produce (leaf, q, fmt) from stored / reformat / derive. Memoized."""
        key = (leaf, q, fmt)
        if key in self._cache:
            return self._cache[key]
        if key in self._resolving:
            return None  # cycle cutoff (transient — never cached)
        self._resolving.add(key)
        try:
            result = None
            for inputs, fn in self._producers(leaf, q, fmt):
                args = [self.resolve(*i) for i in inputs]
                if any(a is None for a in args):
                    continue
                result = fn(*args)
                if result is not None:
                    break
        finally:
            self._resolving.discard(key)
        # Cache positives only: a None may be a transient cycle-cutoff (a key that
        # IS reachable once an in-progress ancestor completes), so never memoize it.
        if result is not None:
            self._cache[key] = result
        return result

    def can_resolve(self, leaf, q, fmt) -> bool:
        """Structural twin of resolve: walk the same producer graph, inputs only."""
        key = (leaf, q, fmt)
        if key in self._can_cache:
            return self._can_cache[key]
        if key in self._can_resolving:
            return False  # cycle cutoff (transient — never cached)
        self._can_resolving.add(key)
        try:
            ok = any(all(self.can_resolve(*i) for i in inputs)
                     for inputs, _ in self._producers(leaf, q, fmt))
        finally:
            self._can_resolving.discard(key)
        if ok:  # cache positives only (a False may be a cycle cutoff)
            self._can_cache[key] = ok
        return ok

    def _producers(self, leaf, q, fmt):
        """(inputs, fn) in priority order: stored, reformat, derive. The single graph
        resolve() executes and can_resolve() walks structurally."""
        if self._has_stored(leaf, q, fmt):
            yield (), lambda: self._stored(leaf, q, fmt)
        for src_fmts, fn in REFORMAT.get(fmt, ()):
            yield tuple((leaf, q, sf) for sf in src_fmts), fn
        cfg = self._cfg()
        d = derivation(cfg, leaf, q) if cfg else None
        if d and self._weights_obtainable():
            src_leaf, src_q, T = d
            src = T.formats().get(fmt)
            if src is not None:
                yield tuple((src_leaf, src_q, sf) for sf in src), self._derive_fn(T, fmt)

    # ------------------------------------------------------------------
    # Stored access
    # ------------------------------------------------------------------

    def _entry(self, leaf):
        return self.data.get(self._canon(leaf), {})

    def _has_stored(self, leaf, q, fmt) -> bool:
        e = self._entry(leaf)
        if fmt in ("samples", "samples_raw"):
            return f"{q}_samples" in e or f"{q}_U" in e
        if fmt == "svd":
            return f"{q}_U" in e
        return f"{q}_{fmt}" in e

    def _stored(self, leaf, q, fmt):
        e = self._entry(leaf)
        if fmt in ("samples", "samples_raw"):
            x = e.get(f"{q}_samples")
            if x is None and f"{q}_U" in e:
                x = (e[f"{q}_U"].float() * e[f"{q}_S"].float()) @ e[f"{q}_V"].float().T
            if x is None:
                return None
            mask = e.get(f"{q}_mask") if fmt == "samples" else None
            return x[mask.bool()] if mask is not None else x
        if fmt == "svd":
            return (e[f"{q}_U"], e[f"{q}_S"], e[f"{q}_V"]) if f"{q}_U" in e else None
        if fmt == "cov":
            c = e.get(f"{q}_cov")
            return c.float() / (e.get(f"{q}_n") or 1) if c is not None else None
        if fmt == "n":
            v = e.get(f"{q}_n")
            return int(v) if v is not None else None
        return e.get(f"{q}_{fmt}")

    # ------------------------------------------------------------------
    # Derivation
    # ------------------------------------------------------------------

    def _derive_fn(self, T, fmt):
        if getattr(T, "kind", None) == "norm":
            def fn(X):
                norm = self._norm_module(T.sd_prefix)
                if norm is None:
                    return None
                with torch.no_grad():
                    return norm(X.float()).cpu()
            return fn

        def fn(*args):
            if fmt == "n":
                return args[0]
            w = self._weight(T.sd_prefix)
            if w is None:
                return None
            W = w.weight.detach().float()
            b = w.bias.detach().float() if (T.bias and getattr(w, "bias", None) is not None) else None
            if T.head is not None:
                d_head = args[0].shape[0] if fmt == "cov" else args[0].shape[-1]
                W = W[:, T.head * d_head:(T.head + 1) * d_head]
            W = W.to(_DEV)
            if b is not None:
                b = b.to(_DEV)
            if fmt == "mean":
                m = W @ args[0].to(_DEV).float()
                return (m + b if b is not None else m).cpu()
            if fmt == "samples":
                out = args[0].to(_DEV).float() @ W.T
                return (out + b if b is not None else out).cpu()
            if fmt == "cov":
                cov = W @ args[0].to(_DEV).float() @ W.T
                if b is not None:
                    Wmu = W @ args[1].to(_DEV).float()
                    cov = cov + torch.outer(Wmu, b) + torch.outer(b, Wmu) + torch.outer(b, b)
                return cov.cpu()
            return None
        return fn

    # ------------------------------------------------------------------
    # Node / FactorView access points
    # ------------------------------------------------------------------

    @property
    def v(self) -> "Node":
        return Node(self, "")

    def __getitem__(self, path: str) -> "Node":
        return Node(self, path)

    @property
    def after_final_norm(self) -> "Node":
        return Node(self, "after_final_norm")

    @property
    def before_final_norm(self) -> "Node":
        return Node(self, "before_final_norm")

    def factor(self, leaf, q) -> Optional["FactorView"]:
        """Handle-or-None: a FactorView iff (leaf, q) has a reachable representation."""
        return FactorView(self, leaf, q) if self.can_resolve(leaf, q, "eigvals") else None

    def leaf_quantities(self):
        """(leaf, q) pairs that resolve, across all tree leaves (for prewarm)."""
        out = []
        for leaf in sorted(self._all_leaves()):
            for q in ("acts", "grads"):
                if self.factor(leaf, q) is not None:
                    out.append((leaf, q))
        return out

    def needs_model_weights(self) -> bool:
        """True if any derived (leaf, acts) is required but not already stored."""
        cfg = self._cfg()
        if cfg is None:
            return False
        for leaf in self._all_leaves():
            if derivation(cfg, leaf, "acts") is None:
                continue
            if any(self._has_stored(leaf, "acts", f) for f in ("eigvals", "cov", "samples")):
                continue
            return True
        return False

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path, format="cov_svd", cross_basis_refs=None,
             storage_dtype=None, token_filter=None, n_chunks=None):
        result = self.to_dict(format=format, storage_dtype=storage_dtype)

        if cross_basis_refs:
            target = DataAccessor(result, model=self.model, model_config=self.model_config,
                                  model_name=self._model_name)
            for ref_path in cross_basis_refs:
                target._project_onto(ref_path)

        self._stamp_from_path(path)
        self._write_metadata(result, format, token_filter=token_filter, n_chunks=n_chunks)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(result, path)
        return path

    def to_dict(self, format="cov_svd", storage_dtype=None) -> dict:
        """Materialize this accessor's data as a leaf-keyed storage-format dict."""
        base, plus, minus = parse_format_spec(format)
        result = {}
        for leaf in sorted(self._all_leaves()):
            src = self._entry(leaf)
            out = {k: v for k, v in src.items() if "cross_eigvals" in k}
            for q in ("acts", "grads"):
                if self._should_write(leaf, q, src, base, plus, minus):
                    self._write_quantity(out, leaf, q, src, base, plus, minus, storage_dtype)
            if out:
                result[leaf] = out
        return result

    def _should_write(self, leaf, q, src, base, plus, minus) -> bool:
        if not self.can_resolve(leaf, q, "eigvals"):
            return False
        cfg = self._cfg()
        stored = any(self._has_stored(leaf, q, f) for f in ("cov", "samples", "eigvals", "eigvecs"))
        if cfg is None or derivation(cfg, leaf, q) is None:
            return True  # captured quantity
        # derived quantity — gate on its storage flag
        flag = _derived_flag(leaf)
        if flag is None:
            return stored  # no materialization flag (after_final_norm): keep iff captured
        if leaf.endswith(".contrib") and base in ("acts", "acts_svd"):
            return False  # contrib has no raw samples
        if flag in minus:
            return False
        if flag in plus or base == "eigenvalues":
            return True
        return stored  # materialized derived quantity already present -> keep

    def _store_mean(self, leaf, q, src, base, plus, minus) -> bool:
        if "m" in minus:
            return False
        if {"m", "b", "o"} & plus or base == "eigenvalues":
            return True
        if f"{q}_mean" in src:
            return True
        return base == "cov_svd" and self._has_stored(leaf, q, "samples")

    def _write_quantity(self, out, leaf, q, src, base, plus, minus, sd):
        n = self.resolve(leaf, q, "n")

        if base == "acts":
            s = self.resolve(leaf, q, "samples_raw")
            if s is not None:
                out[f"{q}_samples"] = _cast_for(s.cpu(), "acts", sd)
                out[f"{q}_n"] = s.shape[0]
                self._copy_mask(out, src, q)

        elif base == "acts_svd":
            svd = self.resolve(leaf, q, "svd")
            if svd is not None and len(svd) == 3:
                U, S, V = svd
                out[f"{q}_U"] = _cast_for(U.cpu(), "acts", sd)
                out[f"{q}_S"] = _cast_for(S.cpu(), "eigvals", sd)
                out[f"{q}_V"] = _cast_for(V.cpu(), "eigvecs", sd)
                if n is not None:
                    out[f"{q}_n"] = n
                self._copy_mask(out, src, q)

        elif base == "cov":
            cov = self.resolve(leaf, q, "cov")
            if cov is not None and n is not None:
                out[f"{q}_cov"] = _cast_for(cov.cpu() * n, "cov", sd)
                out[f"{q}_n"] = n

        elif base == "cov_svd":
            eh = self.resolve(leaf, q, "eigh")
            if eh is not None:
                ev, V = eh
                out[f"{q}_eigvals"] = _cast_for(ev.cpu(), "eigvals", sd)
                out[f"{q}_eigvecs"] = _cast_for(V.cpu(), "eigvecs", sd)
                if n is not None:
                    out[f"{q}_n"] = n
                c = self.resolve(leaf, q, "eigvals_centered")
                if c is not None:
                    out[f"{q}_eigvals_centered"] = _cast_for(c.cpu(), "eigvals", sd)

        elif base == "eigenvalues":
            ev = self.resolve(leaf, q, "eigvals")
            if ev is not None:
                out[f"{q}_eigvals"] = _cast_for(ev.cpu(), "eigvals", sd)
                if n is not None:
                    out[f"{q}_n"] = n
                c = self.resolve(leaf, q, "eigvals_centered")
                if c is not None:
                    out[f"{q}_eigvals_centered"] = _cast_for(c.cpu(), "eigvals", sd)

        if self._store_mean(leaf, q, src, base, plus, minus):
            mu = self.resolve(leaf, q, "mean")
            if mu is not None:
                out[f"{q}_mean"] = _cast_for(mu.cpu(), "mean", sd)

    @staticmethod
    def _copy_mask(out, src, q):
        mask = src.get(f"{q}_mask")
        if mask is not None:
            out[f"{q}_mask"] = mask.cpu().bool()

    def _write_metadata(self, result, format, token_filter=None, n_chunks=None):
        source_filter = self.data.get("__token_filter__")
        if token_filter is not None:
            result["__token_filter__"] = token_filter
        elif source_filter is not None:
            result["__token_filter__"] = source_filter

        source_chunks = self.data.get("__n_chunks__")
        if n_chunks is not None:
            result["__n_chunks__"] = n_chunks
        elif source_chunks is not None:
            result["__n_chunks__"] = source_chunks

        if self._model_name:
            result["__hf_model__"] = self._hf_repo or self._model_name
        if self._revision:
            result["__revision__"] = self._revision
        result["__format__"] = format

    def _stamp_from_path(self, output_path: str) -> None:
        if self._revision is not None and self._hf_repo is not None:
            return
        m = _STEP_FILE_RE.match(os.path.basename(output_path or ""))
        model_dir = os.path.basename(os.path.dirname(output_path or ""))
        if not m or not model_dir:
            return
        try:
            import contextlib, io
            from utils.model_registry import get_checkpoint_schedule
            config = get_model_config(_resolve_hf_name(model_dir))
            with contextlib.redirect_stdout(io.StringIO()):
                schedule = get_checkpoint_schedule(config, None)
        except Exception:
            return
        for step, rev, hf in schedule:
            if step == int(m.group(1)):
                self._model_name = self._model_name or hf
                self._hf_repo = self._hf_repo or hf
                self._revision = self._revision or rev
                return

    def info(self) -> str:
        fmt = self.data.get("__format__", "unknown")
        label = self.path or "<memory>"
        lines = [f"File: {label}", f"Format: {fmt}", ""]

        tf = self.data.get("__token_filter__")
        if tf:
            lines.append("Token filter:")
            for k, v in tf.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        if self._hf_repo or self._revision:
            lines.append("Model metadata:")
            if self._hf_repo:
                lines.append(f"  __hf_model__: {self._hf_repo}")
            if self._revision:
                lines.append(f"  __revision__: {self._revision}")
            lines.append("")

        for name, entry in sorted(self.data.items()):
            if name.startswith("__"):
                continue
            lines.append(f"  {name}:")
            for k, v in sorted(entry.items()):
                if isinstance(v, torch.Tensor):
                    size_mb = v.numel() * v.element_size() / (1024 * 1024)
                    lines.append(f"    {k}: {tuple(v.shape)} {v.dtype} ({size_mb:.2f} MB)")
                elif isinstance(v, (int, float)):
                    lines.append(f"    {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cross-basis projections / token filter
    # ------------------------------------------------------------------

    def set_token_filter(self, filter_dict: dict, output_path: str = None) -> str:
        self.data["__token_filter__"] = filter_dict
        return self._save_in_place(output_path)

    def project_same_layer(self, onto: str = "both", output_path: str = None) -> str:
        """Cross-project acts and grads onto each other's basis at every leaf that
        carries both in the same space (boundary / slice / a projection's .out)."""
        do_fwd = onto in ("fwd", "both")
        do_rev = onto in ("rev", "both")
        for leaf in self._all_leaves():
            a = self.factor(leaf, "acts")
            g = self.factor(leaf, "grads")
            if a is None or g is None:
                continue
            av, gv, ac, gc = a.eigvecs, g.eigvecs, a.cov, g.cov
            if any(x is None for x in (av, gv, ac, gc)) or av.shape != gv.shape:
                continue
            e = self.data.setdefault(self._canon(leaf), {})
            if do_fwd:
                e["grads_cross_eigvals_acts"] = cross_eigvals(gc, av)
            if do_rev:
                e["acts_cross_eigvals_grads"] = cross_eigvals(ac, gv)
        return self._save_in_place(output_path)

    def project_onto_basis(self, basis_path: str, output_path: str = None) -> str:
        self._project_onto(basis_path)
        return self._save_in_place(output_path)

    def _project_onto(self, basis_path: str) -> None:
        ref = DataAccessor(basis_path)
        label = os.path.splitext(os.path.basename(basis_path))[0]
        for leaf in self._all_leaves():
            for q in ("acts", "grads"):
                a = self.factor(leaf, q)
                b = ref.factor(leaf, q)
                if a is None or b is None:
                    continue
                basis, cov = b.eigvecs, a.cov
                if basis is None or cov is None or cov.shape[0] != basis.shape[0]:
                    continue
                self.data.setdefault(self._canon(leaf), {})[f"{q}_cross_eigvals_{label}"] = \
                    cross_eigvals(cov, basis)

    def _save_in_place(self, output_path: str = None) -> str:
        out = output_path or self.path
        if out is None:
            raise ValueError("output_path is required for in-memory data")
        self._stamp_from_path(out)
        self._write_metadata(self.data, self.data.get("__format__", "unknown"))
        torch.save(self.data, out)
        return out


# ---------------------------------------------------------------------------
# Tree navigation: Node (a path) -> FactorView (a leaf+quantity)
# ---------------------------------------------------------------------------

class Node:
    """An ephemeral tree node addressed by a dotted path. Descend with attribute
    access; `.acts` / `.grads` yield a FactorView when this path is a data leaf."""

    def __init__(self, acc: DataAccessor, path: str = ""):
        self._acc = acc
        self._path = path

    def children(self):
        return [Node(self._acc, f"{self._path}.{s}" if self._path else s)
                for s in self._acc.tree_children(self._path)]

    def __getitem__(self, seg):
        return Node(self._acc, f"{self._path}.{seg}" if self._path else str(seg))

    def __getattr__(self, seg):
        if seg.startswith("_"):
            raise AttributeError(seg)
        if seg in ("acts", "grads"):
            fv = self._acc.factor(self._path, seg)
            if fv is None:
                raise AttributeError(f"{self._path!r} has no {seg}")
            return fv
        return Node(self._acc, f"{self._path}.{seg}" if self._path else seg)

    def get(self, rel):
        """Resolve a relative path to a FactorView, or None (used by metrics)."""
        node = self
        for seg in rel.split("."):
            if not isinstance(node, Node):
                return None
            try:
                node = node.__getattr__(seg)
            except AttributeError:
                return None
        return node if isinstance(node, FactorView) else None

    def __repr__(self):
        return f"Node({self._path!r})"


class FactorView:
    """One quantity (acts/grads) of one leaf. `.<format>` -> resolve(leaf, q, fmt),
    reading None when that format is unavailable."""

    def __init__(self, acc: DataAccessor, leaf: str, q: str):
        self._acc = acc
        self._leaf = leaf
        self._q = q

    def __getattr__(self, fmt):
        if fmt.startswith("_"):
            raise AttributeError(fmt)
        return self._acc.resolve(self._leaf, self._q, fmt)

    def __repr__(self):
        return f"FactorView({self._leaf!r}, {self._q!r})"


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import glob as _glob
    from multiprocessing import Pool

    def _pt_files(path):
        if os.path.isdir(path):
            return sorted(_glob.glob(os.path.join(path, "*.pt")))
        return [path]

    def _run_pool(worker, task_args, workers):
        with Pool(processes=workers) as pool:
            for out in pool.imap_unordered(worker, task_args):
                print(f"  -> {out}")

    def _resolve_outputs(pts, output, output_dir):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            return [(p, os.path.join(output_dir, os.path.basename(p))) for p in pts]
        if output:
            if len(pts) == 1:
                return [(pts[0], output)]
            print("Warning: --output ignored for directory input (use --output-dir)")
        return [(p, p) for p in pts]

    parser = argparse.ArgumentParser(
        prog="python -m utils.accessor",
        description="Storage tool: convert, project, inspect .pt factor files",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("info", help="Show stored formats, sizes, projections")
    p.add_argument("input", help="Path to .pt file or model directory")

    p = sub.add_parser("convert", help="Convert storage format")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--to", required=True, metavar="FORMAT")
    p.add_argument("--output", default=None, metavar="PATH", help="Output path (single-file input only)")
    p.add_argument("--output-dir", default=None, dest="output_dir", metavar="DIR",
                   help="Write outputs here, mirroring input filenames (default: in-place)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--abort-on-model-load", action="store_true",
                   help="Abort instead of lazily loading model weights for derivations")

    p = sub.add_parser("project", help="Add cross-basis eigenvalue projections")
    p.add_argument("--input", required=True, metavar="PATH")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--onto", choices=["fwd", "rev", "both"],
                   help="Same-leaf cross-projection: grads->acts basis (fwd), acts->grads (rev), both")
    g.add_argument("--onto-file", dest="onto_file", metavar="FILE",
                   help="Cross-checkpoint: project each file onto eigenbasis from this reference file")
    p.add_argument("--output", default=None, metavar="PATH", help="Output path (single-file input only)")
    p.add_argument("--output-dir", default=None, dest="output_dir", metavar="DIR",
                   help="Write outputs here, mirroring input filenames (default: in-place)")
    p.add_argument("--workers", type=int, default=None)

    p = sub.add_parser("set-filter", help="Edit stored token filter metadata")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--token-selection", choices=["all", "last"], dest="token_selection")
    p.add_argument("--skip-positions", type=int, dest="skip_positions")
    p.add_argument("--boundary-token-ids", type=int, nargs="+", dest="boundary_token_ids")
    p.add_argument("--answer-only", action="store_true", dest="answer_only")
    p.add_argument("--output", default=None, metavar="PATH")
    p.add_argument("--output-dir", default=None, dest="output_dir", metavar="DIR",
                   help="Write outputs here, mirroring input filenames (default: in-place)")
    p.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()

    if args.command == "info":
        for pt in _pt_files(args.input):
            print(DataAccessor(pt).info())

    elif args.command == "convert":
        pairs = _resolve_outputs(_pt_files(args.input), args.output, args.output_dir)
        if len(pairs) == 1:
            ip, op = pairs[0]
            acc = DataAccessor(ip, abort_on_model_load=args.abort_on_model_load)
            print(f"Saved to {acc.save(op, format=args.to)}")
        else:
            print(f"Converting {len(pairs)} files to {args.to}...")
            _run_pool(_convert_worker,
                      [(ip, op, args.to, args.abort_on_model_load) for ip, op in pairs],
                      args.workers)

    elif args.command == "project":
        pairs = _resolve_outputs(_pt_files(args.input), args.output, args.output_dir)
        if args.onto:
            if len(pairs) == 1:
                ip, op = pairs[0]
                print(f"Saved to {DataAccessor(ip).project_same_layer(args.onto, output_path=op)}")
            else:
                print(f"Projecting {len(pairs)} files (onto={args.onto})...")
                _run_pool(_project_same_layer_worker, [(ip, op, args.onto) for ip, op in pairs], args.workers)
        else:
            if len(pairs) == 1:
                ip, op = pairs[0]
                print(f"Saved to {DataAccessor(ip).project_onto_basis(args.onto_file, op)}")
            else:
                print(f"Projecting {len(pairs)} files onto {args.onto_file}...")
                _run_pool(_project_onto_file_worker, [(ip, op, args.onto_file) for ip, op in pairs], args.workers)

    elif args.command == "set-filter":
        pts = _pt_files(args.input)
        sample = torch.load(pts[0], map_location="cpu", weights_only=False)
        filter_dict = dict(sample.get("__token_filter__", {}))
        if args.token_selection is not None:
            filter_dict["token_selection"] = args.token_selection
        if args.skip_positions is not None:
            filter_dict["skip_positions"] = args.skip_positions
        if args.boundary_token_ids is not None:
            filter_dict["boundary_token_ids"] = args.boundary_token_ids
        if args.answer_only:
            filter_dict["answer_only"] = True
        pairs = _resolve_outputs(pts, args.output, args.output_dir)
        if len(pairs) == 1:
            ip, op = pairs[0]
            print(f"Saved to {DataAccessor(ip).set_token_filter(filter_dict, op)}")
        else:
            print(f"Setting token filter on {len(pairs)} files...")
            _run_pool(_set_filter_worker, [(ip, op, filter_dict) for ip, op in pairs], args.workers)

    else:
        parser.print_help()
