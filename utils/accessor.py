"""Unified data interface: read, write, convert, and inspect collected data.

Combines the DataAccessor (format-agnostic read/write interface), storage
format handling, and eigendecomposition helpers into a single module.

The factor identity is the `<node>.<quantity>` signal string itself (see
utils.hook_names). DataAccessor provides a chainable property-based API:

    acc = DataAccessor(data)

    acc["blk3.up"].in_.acts.eigvals    # 1D float Tensor, descending
    acc["blk3.up"].out.acts.cov        # (d_out, d_out) Tensor, derived if model given
    acc["blk3.up"].out.grads.eigh      # (eigvals Tensor, eigvecs Tensor)
    acc["blk3.up"].in_.acts.mean       # stored mean (+m modifier)
    acc["blk3.up"].in_.acts.svd        # (S, V) from eigdecomp, or (U, S, V) from acts
    acc.blocks[3].up.in_.acts.eigvals  # same as above, block-indexed

    acc.after_final_norm.value.acts.eigvals  # final residual stream, post-norm
    acc.before_final_norm.value.acts.eigvals # final residual stream, pre-norm
    acc["blk3.attn.head0"].slice.acts.eigvals  # OV-head pre-W_o slice
    acc["blk3.attn.head0"].contrib.acts.cov    # OV-head post-W_o contribution

Storage formats (from most to least data):
    acts        — raw per-token activation vectors (N, d). Largest.
    cov         — accumulated (d,d) outer product matrix (Sigma xxT) + sample count n.
    acts_svd    — SVD of activations: U(N,d), S(d,), V(d,d).
    cov_svd     — eigendecomposition of covariance: eigvecs V(d,k) + eigvals lambda(k).
    eigenvalues — just eigenvalues lambda(k). Cheapest.

CLI usage:
    python -m utils.accessor info <path>
    python -m utils.accessor convert --input <path> --to cov_svd
    python -m utils.accessor project --input <path> --onto both
    python -m utils.accessor set-filter --input <path> --token-selection last

Convert/project/set-filter mutate in place by default; pass --output-dir <DIR>
(or --output <FILE> for single-file input) to redirect.
"""

import os
import re
import time
import torch

from typing import Optional

from utils import hook_names as hn


# ===========================================================================
# Section 2: Eigendecomp helpers (from powerlaw.py)
# ===========================================================================

class _DecompProfiler:
    def __init__(self):
        self.calls = []  # (label, shape, duration)
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
    """Returns (centered, uncentered) as torch tensors. centered is None if mu unavailable.
    Inputs must be torch tensors on the desired compute device."""
    return _from_acts(acts, topk) if acts is not None else _from_cov(cov, mu, topk)


# ===========================================================================
# Section 3: Storage constants and helpers
# ===========================================================================

# Format modifier flag controlling whether each derived signal is stored.
_DERIVED_FLAG = {"out.acts": "b", "contrib.acts": "o"}

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
    Later +/- occurrences win, allowing e.g. eigenvalues-b or cov_svd+m-b.
    Raises ValueError on an unknown base or modifier.
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


def _cast(tensor: torch.Tensor, dtype_str: str) -> torch.Tensor:
    dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
    return tensor.to(dtype=dtype)


def _cast_for(tensor: torch.Tensor, item_type: str, storage_dtype) -> torch.Tensor:
    """Cast tensor using per-item storage dtype rules.

    storage_dtype: None -> use _DEFAULT_STORAGE_DTYPES
                   str  -> apply to all item types
                   dict -> per-item lookup, fallback fp32
    """
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


# ===========================================================================
# Section 6: DataAccessor class
# ===========================================================================

def _resolve_hf_name(short_name: str) -> str:
    """Resolve short model name to full HuggingFace name."""
    if "/" in short_name:
        return short_name
    s = short_name.lower()
    if "pythia" in s:
        return f"EleutherAI/{short_name}"
    if "olmo" in s:
        return f"allenai/{short_name}"
    return short_name


class DataAccessor:
    """Format-agnostic reader for collected activation / covariance data.

    Args:
        data: Loaded .pt dict or path to a .pt file.
        model: Optional nn.Module for model-dependent derivations (B, norm).
        model_config: Optional ModelConfig for resolving hook names to layers.
        model_name: Optional HF model name (str). Model is lazily loaded on first
                    access that requires it. Ignored if model is already provided.
    """

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
        self._hf_repo = data.get("__hf_model__")  # actual repo (may differ from model_name for early-training)
        self._selective_weights: Optional[dict] = None
        self.abort_on_model_load = abort_on_model_load
        self._layer_cache: dict = {}
        self._eigh_cache: dict = {}  # (hook, signal) -> (centered_eigvals, uncentered_eigvals, eigvecs_or_None, centered_eigvecs_or_None)
        self._derived_cov_cache: dict = {}  # (hook, signal) -> derived covariance

    def _ensure_weights(self):
        """Ensure model weights are available (full model or selective loading)."""
        if self.model is not None or self._selective_weights is not None:
            return True
        if self._model_name is None:
            return False
        if self.abort_on_model_load:
            raise RuntimeError(
                f"Aborting before loading model weights for {self._model_name}"
                + (f"@{self._revision}" if self._revision else "")
            )
        # Selective loading: only the layers we actually need
        mlp_hooks = [k for k in self.data if hn.mlp_proj(k)]
        ov_hooks = [k for k in self.data if hn.ov_head(k)]
        need_norm = "before_final_norm" in self.data and "after_final_norm" not in self.data
        if not mlp_hooks and not ov_hooks and not need_norm:
            return False
        from utils.model_registry import get_model_config, load_selective_weights
        if self.model_config is None:
            self.model_config = get_model_config(self._model_name)
        hf_repo = self._hf_repo or self.model_config.hf_repo
        self._selective_weights = load_selective_weights(
            self.model_config, hf_repo, self._revision, mlp_hooks + ov_hooks, need_norm,
        )
        return True

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path, format="cov_svd", cross_basis_refs=None,
             storage_dtype=None, token_filter=None, n_chunks=None):
        """Save data in the specified storage format."""
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
        """Materialize this accessor's data as a storage-format dict."""
        base_format, plus, minus = parse_format_spec(format)
        result = {}

        for hook_name in self.hook_names():
            entry = self._entry(hook_name)
            out_entry = {"n": entry.get("n", 0)}
            for k, v in entry.items():
                if k.startswith("n_") or "cross_eigvals" in k:
                    out_entry[k] = v

            signals = list(hn.captured_signals(hook_name))
            for signal in hn.derived_signals(hook_name):
                if self._should_store_derived(entry, signal, base_format, plus, minus):
                    signals.append(signal)

            for signal in signals:
                store_mean = self._should_store_mean(hook_name, signal, base_format, plus, minus)
                self._write_signal(out_entry, hook_name, signal, base_format, store_mean, storage_dtype)

            result[hook_name] = out_entry

        return result

    def _should_store_derived(self, entry, signal, base_format, plus, minus) -> bool:
        flag = _DERIVED_FLAG[signal]
        # `contrib.acts` (the +o signal) is raw-less: never stored for acts formats.
        if signal == "contrib.acts" and base_format in ("acts", "acts_svd"):
            return False
        if flag in minus:
            return False
        if flag in plus or base_format == "eigenvalues":
            return True
        return any(k in entry for k in (signal, f"{signal}_eigvals", f"{signal}_eigvecs"))

    def _should_store_mean(self, hook_name, signal, base_format, plus, minus) -> bool:
        if "m" in minus:
            return False
        if "m" in plus or "b" in plus or "o" in plus or base_format == "eigenvalues":
            return True
        entry = self._entry(hook_name)
        if f"{signal}_mean" in entry:
            return True
        return base_format == "cov_svd" and self._has_raw_activations(entry, signal)

    @staticmethod
    def _has_raw_activations(entry, signal) -> bool:
        t = entry.get(signal)
        return isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]

    def _write_signal(self, out_entry, hook_name, signal, base_format, store_means, storage_dtype):
        view = self[hook_name]._factor(signal)
        n = view.n

        if base_format == "acts":
            acts = view.samples_raw
            if acts is not None:
                out_entry[signal] = _cast_for(acts.cpu(), "acts", storage_dtype)
                out_entry[f"n_{signal}"] = acts.shape[0]
                self._copy_mask(out_entry, hook_name, signal)

        elif base_format == "acts_svd":
            svd = view.svd
            if svd is not None and len(svd) == 3:
                U, S, V = svd
                out_entry[f"{signal}_U"] = _cast_for(U.cpu(), "acts", storage_dtype)
                out_entry[f"{signal}_S"] = _cast_for(S.cpu(), "eigvals", storage_dtype)
                out_entry[f"{signal}_V"] = _cast_for(V.cpu(), "eigvecs", storage_dtype)
                if n is not None:
                    out_entry[f"n_{signal}"] = n
                self._copy_mask(out_entry, hook_name, signal)

        elif base_format == "cov":
            cov = view.cov
            if cov is not None and n is not None:
                out_entry[signal] = _cast_for((cov.cpu() * n), "cov", storage_dtype)
                out_entry[f"n_{signal}"] = n

        elif base_format == "cov_svd":
            eigh = view.eigh
            if eigh is not None:
                eigvals, eigvecs = eigh
                out_entry[f"{signal}_eigvals"] = _cast_for(eigvals.cpu(), "eigvals", storage_dtype)
                out_entry[f"{signal}_eigvecs"] = _cast_for(eigvecs.cpu(), "eigvecs", storage_dtype)
                if n is not None:
                    out_entry[f"n_{signal}"] = n
                centered = view.eigvals_centered
                if centered is not None:
                    out_entry[f"{signal}_eigvals_centered"] = _cast_for(centered.cpu(), "eigvals", storage_dtype)

        elif base_format == "eigenvalues":
            eigvals = view.eigvals
            if eigvals is not None:
                out_entry[f"{signal}_eigvals"] = _cast_for(eigvals.cpu(), "eigvals", storage_dtype)
                if n is not None:
                    out_entry[f"n_{signal}"] = n
                centered = view.eigvals_centered
                if centered is not None:
                    out_entry[f"{signal}_eigvals_centered"] = _cast_for(centered.cpu(), "eigvals", storage_dtype)

        if store_means:
            mean = view.mean
            if mean is not None:
                out_entry[f"{signal}_mean"] = _cast_for(mean.cpu(), "mean", storage_dtype)

    def _copy_mask(self, out_entry, hook_name, signal):
        mask = self._entry(hook_name).get(f"{signal}_mask")
        if mask is not None:
            out_entry[f"{signal}_mask"] = mask.cpu().bool()

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
        """Backfill missing model/revision by inferring from a model-dir/stepN.pt path."""
        if self._revision is not None and self._hf_repo is not None:
            return
        m = _STEP_FILE_RE.match(os.path.basename(output_path or ""))
        model_dir = os.path.basename(os.path.dirname(output_path or ""))
        if not m or not model_dir:
            return
        try:
            import contextlib, io
            from utils.model_registry import get_model_config, get_checkpoint_schedule
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
        """Return a summary of stored data."""
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
            lines.append(f"    n = {entry.get('n', '?')}")
            for k, v in sorted(entry.items()):
                if k in ("n",):
                    continue
                if isinstance(v, torch.Tensor):
                    size_mb = v.numel() * v.element_size() / (1024 * 1024)
                    lines.append(f"    {k}: {tuple(v.shape)} {v.dtype} ({size_mb:.2f} MB)")
                elif isinstance(v, (int, float)):
                    lines.append(f"    {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    def set_token_filter(self, filter_dict: dict, output_path: str = None) -> str:
        """Edit token filter metadata and save."""
        self.data["__token_filter__"] = filter_dict
        return self._save_in_place(output_path)

    def project_same_layer(self, onto: str = "both", output_path: str = None) -> str:
        """Project the grads signal against the same hook's forward-acts signal and save.

        The forward-acts signal is the derived `out.acts` for MLP hooks (from
        weights if needed), the captured `value.acts` for residual/boundary hooks,
        and the derived `contrib.acts` for OV-heads. onto: "fwd" projects grads
        onto the forward basis, "rev" the reverse, "both" both directions."""
        do_fwd = onto in ("fwd", "both")
        do_G = onto in ("rev", "both")
        for hook in self.hook_names():
            grads_signal = self._grads_signal(hook)
            fwd_signal = self._forward_acts_signal(hook)
            if grads_signal is None or fwd_signal is None:
                continue
            G = self[hook]._factor(grads_signal)
            fwd = self[hook]._factor(fwd_signal)
            try:
                fwd_vecs, G_vecs = fwd.eigvecs, G.eigvecs
            except ValueError:
                continue  # MLP out.acts needs in.acts mean (bias term) that wasn't stored
            if fwd_vecs is None or G_vecs is None or fwd_vecs.shape != G_vecs.shape:
                continue
            if do_fwd:
                self._entry(hook)[f"{grads_signal}_cross_eigvals_{fwd_signal}"] = cross_eigvals(G.cov, fwd_vecs)
            if do_G:
                self._entry(hook)[f"{fwd_signal}_cross_eigvals_{grads_signal}"] = cross_eigvals(fwd.cov, G_vecs)
        return self._save_in_place(output_path)

    @staticmethod
    def _grads_signal(hook):
        """The captured grads signal for this hook, or None if its kind has none."""
        for s in hn.captured_signals(hook):
            if s.endswith(".grads"):
                return s
        return None

    @staticmethod
    def _forward_acts_signal(hook):
        """The output-acts signal at this hook: derived if any, else captured acts."""
        for s in hn.derived_signals(hook):
            if s.endswith(".acts"):
                return s
        for s in hn.captured_signals(hook):
            if s.endswith(".acts"):
                return s
        return None

    def project_onto_basis(self, basis_path: str, output_path: str = None) -> str:
        """Project this data onto another file's eigenbasis (cross-checkpoint) and save."""
        self._project_onto(basis_path)
        return self._save_in_place(output_path)

    def _project_onto(self, basis_path: str) -> None:
        """Add `{signal}_cross_eigvals_{ref}` from the reference file's eigenbasis. In place."""
        ref = DataAccessor(basis_path)
        label = os.path.splitext(os.path.basename(basis_path))[0]
        for hook in self.hook_names():
            if hook not in ref.data:
                continue
            for signal in hn.captured_signals(hook):
                basis = ref[hook]._factor(signal).eigvecs
                cov = self[hook]._factor(signal).cov
                if basis is None or cov is None or cov.shape[0] != basis.shape[0]:
                    continue
                self._entry(hook)[f"{signal}_cross_eigvals_{label}"] = cross_eigvals(cov, basis)

    def _save_in_place(self, output_path: str = None) -> str:
        """Persist self.data unchanged in format, refreshing metadata."""
        out = output_path or self.path
        if out is None:
            raise ValueError("output_path is required for in-memory data")
        self._stamp_from_path(out)
        self._write_metadata(self.data, self.data.get("__format__", "unknown"))
        torch.save(self.data, out)
        return out

    # ------------------------------------------------------------------
    # Public access points
    # ------------------------------------------------------------------

    def __getitem__(self, hook_name: str) -> "HookView":
        return HookView(self._canonical_key(hook_name), self)

    def _canonical_key(self, hook_name: str) -> str:
        """Resolve aliased names to their stored canonical key (read-only fallback).

        Only kicks in when the requested key isn't stored. Iteration paths
        (hook_names / available / CLI) ignore this, so disk converters never
        duplicate the aliased data.
        """
        if hook_name in self.data:
            return hook_name
        for pattern, template in _KEY_ALIASES:
            m = pattern.match(hook_name)
            if m:
                canonical = template.format(*m.groups())
                if canonical in self.data:
                    return canonical
        return hook_name

    @property
    def blocks(self) -> "BlocksView":
        return BlocksView(self)

    @property
    def after_final_norm(self) -> "HookView":
        return self["after_final_norm"]

    @property
    def before_final_norm(self) -> "HookView":
        return self["before_final_norm"]

    def hook_names(self) -> list:
        """All hook point names stored in this file (excludes metadata keys)."""
        return [k for k in self.data if not k.startswith("__")]

    # ------------------------------------------------------------------
    # Internal: entry access
    # ------------------------------------------------------------------

    def _entry(self, hook_name: str) -> dict:
        return self.data.get(hook_name, {})

    # ------------------------------------------------------------------
    # Internal: layer resolution for B derivation
    # ------------------------------------------------------------------

    def _get_layer(self, hook_name: str):
        """Return a Layer-like object (.weight, .bias) for a derivable-factor hook.

        Handles MLP keys (blk{i}.up/down/gate → that linear projection) and
        per-OV-head keys (blk{i}.attn.head{h} → the shared o_proj layer for that
        block).
        """
        if hook_name in self._layer_cache:
            return self._layer_cache[hook_name]
        mlp = hn.mlp_proj(hook_name)
        ov = hn.ov_head(hook_name)
        if not mlp and not ov:
            return None
        if not self._ensure_weights():
            return None
        if self.model is not None:
            from utils.model_registry import get_mlp_projections, get_attention_output_proj
            if mlp:
                block_idx = mlp[0]
                for name, layer in get_mlp_projections(self.model, self.model_config, block_idx):
                    self._layer_cache[name] = layer
            else:
                block_idx = ov[0]
                _, o_proj, _, _ = get_attention_output_proj(self.model, self.model_config, block_idx)
                for k in self.data:
                    kv = hn.ov_head(k)
                    if kv and kv[0] == block_idx:
                        self._layer_cache[k] = o_proj
        elif self._selective_weights is not None and hook_name in self._selective_weights:
            self._layer_cache[hook_name] = self._selective_weights[hook_name]
        return self._layer_cache.get(hook_name)

    # ------------------------------------------------------------------
    # Internal: computation methods (used by FactorView properties)
    # ------------------------------------------------------------------

    def _ensure_eigh(self, hook_name: str, signal: str, need_vecs: bool = False, need_centered_vecs: bool = False):
        """Ensure eigendecomposition is cached for (hook, signal).

        Stores (centered_eigvals, uncentered_eigvals, eigvecs_or_None, centered_eigvecs_or_None) in _eigh_cache.
        If need_vecs and we only have eigvals cached, recomputes with full eigh.
        """
        if need_centered_vecs:
            need_vecs = True
        # blkN.layer virtual hook: redirect in.acts->up.in.acts, out.grads->down.out.grads
        orig_key = (hook_name, signal)
        if hook_name.endswith(".layer"):
            hook_name = hook_name[:-6] + (".up" if signal == "in.acts" else ".down")
        cache_key = (hook_name, signal)
        if orig_key != cache_key:
            # Alias so callers can look up by either name
            if cache_key in self._eigh_cache:
                self._eigh_cache[orig_key] = self._eigh_cache[cache_key]
                return
        cached = self._eigh_cache.get(cache_key)
        if cached is not None:
            if not need_vecs or cached[2] is not None:
                if not need_centered_vecs or cached[3] is not None:
                    return

        entry = self._entry(hook_name)
        centered, uncentered, eigvecs, centered_eigvecs = None, None, None, None
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        # Carry forward any previously cached values
        if cached is not None:
            centered, uncentered, eigvecs, centered_eigvecs = cached

        # Stored eigvals (eigenvalues / cov_svd format)
        if f"{signal}_eigvals" in entry and (not need_vecs or f"{signal}_eigvecs" in entry):
            if uncentered is None:
                uncentered = entry[f"{signal}_eigvals"]
            if eigvecs is None:
                eigvecs = entry.get(f"{signal}_eigvecs")
            stored_centered = entry.get(f"{signal}_eigvals_centered")
            if stored_centered is not None and centered is None:
                centered = stored_centered
            # Recover centered eigvals/eigvecs from stored eigvecs + mean if needed
            if (centered is None or need_centered_vecs) and f"{signal}_eigvecs" in entry:
                mu_t = self._mean(hook_name, signal)
                if mu_t is not None:
                    V = entry[f"{signal}_eigvecs"].to(dev)
                    S = entry[f"{signal}_eigvals"].to(dev)
                    mu_d = mu_t.to(device=dev, dtype=V.dtype)
                    cov_centered = reconstruct_cov(S.to(dev), V.to(dev)) - torch.outer(mu_d, mu_d)
                    if need_centered_vecs and centered_eigvecs is None:
                        centered, centered_eigvecs = eigh_descending(cov_centered)
                        centered, centered_eigvecs = centered.cpu(), centered_eigvecs.cpu()
                    elif centered is None:
                        centered = eigvalsh_descending(cov_centered).cpu()
            self._eigh_cache[cache_key] = (centered, uncentered, eigvecs, centered_eigvecs)
            if orig_key != cache_key:
                self._eigh_cache[orig_key] = self._eigh_cache[cache_key]
            return

        # Compute from cov
        if (uncentered is None or (need_vecs and eigvecs is None)) and signal in entry:
            t = entry[signal]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get(f"n_{signal}", entry.get("n", 1))
                cov = (t.to(dev) / n)
                mu_t = self._mean(hook_name, signal)
                mu = mu_t.to(device=dev, dtype=cov.dtype) if mu_t is not None else None
                if need_vecs:
                    uncentered, eigvecs = _eigh_full(cov)
                    uncentered, eigvecs = uncentered.cpu(), eigvecs.cpu()
                    if mu is not None:
                        if need_centered_vecs:
                            c_vals, c_vecs = _eigh_full(cov - torch.outer(mu, mu))
                            centered = c_vals.cpu()
                            centered_eigvecs = c_vecs.cpu()
                        else:
                            centered = eigvalsh_descending(cov - torch.outer(mu, mu)).cpu()
                else:
                    c, u = get_eigenspectrum(cov=cov, mu=mu)
                    centered = c.cpu() if c is not None else None
                    uncentered = u.cpu()

        # Derived signal (out.acts / contrib.acts): build cov via the linear map, then same eigh logic
        if (uncentered is None or (need_vecs and eigvecs is None)) and signal in hn.derived_signals(hook_name):
            cov = self._derived_cov(hook_name, signal)
            if cov is not None:
                cov = cov.to(dev)
                mu_t = self._derived_mean(hook_name, signal)
                mu = mu_t.to(device=dev, dtype=cov.dtype) if mu_t is not None else None
                if need_vecs:
                    uncentered, eigvecs = _eigh_full(cov)
                    uncentered, eigvecs = uncentered.cpu(), eigvecs.cpu()
                    if mu is not None:
                        if need_centered_vecs:
                            c_vals, c_vecs = _eigh_full(cov - torch.outer(mu, mu))
                            centered = c_vals.cpu()
                            centered_eigvecs = c_vecs.cpu()
                        else:
                            centered = eigvalsh_descending(cov - torch.outer(mu, mu)).cpu()
                else:
                    c, u = get_eigenspectrum(cov=cov, mu=mu)
                    centered = c.cpu() if c is not None else None
                    uncentered = u.cpu()

        # Fallback: raw activations
        if uncentered is None or (need_vecs and eigvecs is None):
            acts = self._activations(hook_name, signal)
            if acts is not None:
                if need_vecs:
                    cov = self._covariance(hook_name, signal)
                    if cov is not None:
                        uncentered, eigvecs = _eigh_full(cov.to(dev))
                        uncentered, eigvecs = uncentered.cpu(), eigvecs.cpu()
                if uncentered is None:
                    c, u = get_eigenspectrum(acts=acts.to(dev))
                    centered = c.cpu() if c is not None else None
                    uncentered = u.cpu()

        self._eigh_cache[cache_key] = (centered, uncentered, eigvecs, centered_eigvecs)
        if orig_key != cache_key:
            self._eigh_cache[orig_key] = self._eigh_cache[cache_key]

    def _eigenvalues(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        self._ensure_eigh(hook_name, signal)
        return self._eigh_cache[(hook_name, signal)][1]

    def _eigenvalues_centered(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        self._ensure_eigh(hook_name, signal)
        return self._eigh_cache[(hook_name, signal)][0]

    def _eigenvectors(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        self._ensure_eigh(hook_name, signal, need_vecs=True)
        return self._eigh_cache[(hook_name, signal)][2]

    def _eigenvectors_centered(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        self._ensure_eigh(hook_name, signal, need_centered_vecs=True)
        return self._eigh_cache[(hook_name, signal)][3]

    def _covariance(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        if signal in hn.derived_signals(hook_name):
            c = self._derived_cov(hook_name, signal)
            return c.cpu() if c is not None else None

        entry = self._entry(hook_name)

        if signal in entry:
            t = entry[signal]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get(f"n_{signal}", entry.get("n", 1))
                return t.float() / n

        vk, sk = f"{signal}_eigvecs", f"{signal}_eigvals"
        if vk in entry and sk in entry:
            return reconstruct_cov(entry[sk].float(), entry[vk].float())

        if signal in entry:
            t = entry[signal]
            if isinstance(t, torch.Tensor) and t.dim() == 2:
                X = t.float()
                mask = entry.get(f"{signal}_mask")
                if mask is not None:
                    X = X[mask.bool()]
                return (X.T @ X) / X.shape[0]

        return None

    def _activations(self, hook_name: str, signal: str, apply_mask: bool = True) -> Optional[torch.Tensor]:
        # Virtual hook: after_final_norm from before_final_norm + norm layer
        if hook_name == "after_final_norm" and signal == "value.acts" and hook_name not in self.data:
            return self._post_norm_activations()

        entry = self._entry(hook_name)

        raw = None
        if signal in entry:
            t = entry[signal]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
                raw = t

        if raw is None:
            uk = f"{signal}_U"
            if uk in entry:
                U = entry[uk].float()
                S = entry[f"{signal}_S"].float()
                V = entry[f"{signal}_V"].float()
                raw = (U * S.unsqueeze(0)) @ V.T

        if raw is None and signal in hn.derived_signals(hook_name):
            deriv = self._derivation(hook_name, signal)
            src_signal = hn.derived_signals(hook_name)[signal][0]
            src_acts = self._activations(hook_name, src_signal, apply_mask=apply_mask)
            if deriv is not None and src_acts is not None:
                W, b = deriv
                out = src_acts.to(W.device).float() @ W.T
                return (out + b if b is not None else out).cpu()

        if raw is None:
            return None

        if apply_mask:
            mask = entry.get(f"{signal}_mask")
            if mask is not None:
                return raw[mask.bool()]
        return raw

    def _mean(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        if signal in hn.derived_signals(hook_name):
            return self._derived_mean(hook_name, signal)
        entry = self._entry(hook_name)
        stored = entry.get(f"{signal}_mean")
        if stored is not None:
            return stored
        acts = self._activations(hook_name, signal)
        return acts.float().mean(0) if acts is not None else None

    def _svd(self, hook_name: str, signal: str):
        """(U, S, V) from acts if available, else (S, V) from eigdecomp."""
        acts = self._activations(hook_name, signal)
        if acts is not None:
            U, S, Vt = torch.linalg.svd(acts.float(), full_matrices=False)
            return U, S, Vt.T

        eigvals = self._eigenvalues(hook_name, signal)
        eigvecs = self._eigenvectors(hook_name, signal)
        if eigvals is not None and eigvecs is not None:
            entry = self._entry(hook_name)
            n = entry.get(f"n_{signal}", entry.get("n", 1))
            S = (eigvals.clamp(min=0) * n).sqrt().float()
            return S, eigvecs

        return None

    # ------------------------------------------------------------------
    # Internal: generic linear derivation (B = MLP out, O = per-OV-head contrib)
    # ------------------------------------------------------------------

    def _derivation(self, hook_name: str, signal: str):
        """(W, bias) on the compute device for a derived `signal`, or None.

        weight_kind "mlp" -> full projection weight (+bias); "head" -> o_proj
        weight sliced to this head's columns (no bias)."""
        spec = hn.derived_signals(hook_name).get(signal)
        if spec is None:
            return None
        src_signal, weight_kind = spec
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        layer = self._get_layer(hook_name)
        if layer is None:
            return None
        if weight_kind == "mlp":
            W = layer.weight.detach().to(dev).float()
            b = layer.bias.detach().to(dev).float() if layer.bias is not None else None
            return W, b
        # "head": slice o_proj columns to this head using d_head from the source cov/mean
        src_cov = self._covariance(hook_name, src_signal)
        mu = self._mean(hook_name, src_signal)
        ref = src_cov if src_cov is not None else mu
        if ref is None:
            return None
        d_head = ref.shape[0]
        h = hn.ov_head(hook_name)[1]
        W = layer.weight.detach().to(dev).float()[:, h * d_head:(h + 1) * d_head]
        return W, None

    def _derived_cov(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        cache_key = (hook_name, signal)
        if cache_key in self._derived_cov_cache:
            return self._derived_cov_cache[cache_key]
        entry = self._entry(hook_name)
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        if signal in entry:
            t = entry[signal]
            if t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get(f"n_{signal}", entry.get("n", 1))
                return t.to(dev).float() / n
        if f"{signal}_eigvecs" in entry and f"{signal}_eigvals" in entry:
            return reconstruct_cov(entry[f"{signal}_eigvals"].to(dev).float(),
                                   entry[f"{signal}_eigvecs"].to(dev).float())

        deriv = self._derivation(hook_name, signal)
        src_signal = hn.derived_signals(hook_name)[signal][0]
        src_cov = self._covariance(hook_name, src_signal)
        if deriv is None or src_cov is None:
            return None
        W, b = deriv
        cov = W @ src_cov.to(dev) @ W.T
        if b is not None:
            mu = self._mean(hook_name, src_signal)
            if mu is None:
                raise ValueError(
                    f"Cannot derive '{signal}' for '{hook_name}': layer has a bias but no "
                    f"'{src_signal}_mean' is stored. Re-collect with a storage format that "
                    f"includes the '+m' modifier (e.g. cov_svd+m)."
                )
            Wmu = W @ mu.to(device=dev, dtype=W.dtype)
            cov = cov + torch.outer(Wmu, b) + torch.outer(b, Wmu) + torch.outer(b, b)
        self._derived_cov_cache[cache_key] = cov
        return cov

    def _derived_mean(self, hook_name: str, signal: str) -> Optional[torch.Tensor]:
        entry = self._entry(hook_name)
        if f"{signal}_mean" in entry:
            return entry[f"{signal}_mean"]
        spec = hn.derived_signals(hook_name).get(signal)
        if spec is None:
            return None
        mu = self._mean(hook_name, spec[0])
        deriv = self._derivation(hook_name, signal)
        if mu is None or deriv is None:
            return None
        W, b = deriv
        m = W @ mu.to(device=W.device, dtype=W.dtype)
        if b is not None:
            m = m + b
        return m.cpu()

    # ------------------------------------------------------------------
    # Internal: norm derivation (before_final_norm -> after_final_norm)
    # ------------------------------------------------------------------

    def _post_norm_activations(self) -> Optional[torch.Tensor]:
        """Derive after_final_norm acts from before_final_norm raw acts + norm layer."""
        acts = self._activations("before_final_norm", "value.acts")
        if acts is None:
            return None
        if not self._ensure_weights():
            return None
        if self.model is not None:
            from utils.model_registry import get_final_layernorm
            norm = get_final_layernorm(self.model, self.model_config).float()
        elif self._selective_weights is not None and "__norm__" in self._selective_weights:
            norm = self._selective_weights["__norm__"].float()
        else:
            return None
        with torch.no_grad():
            return norm(acts.float()).cpu()

    # ------------------------------------------------------------------
    # Public: available hooks and factors
    # ------------------------------------------------------------------

    def _has_eigenvalues(self, hook_name: str, signal: str) -> bool:
        """Check if eigenvalues are obtainable without computing them."""
        entry = self._entry(hook_name)
        if f"{signal}_eigvals" in entry:
            return True
        # Raw cov (square matrix) -> eigendecomposable
        if signal in entry:
            t = entry[signal]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == t.shape[1]:
                return True
        # Eigvecs + eigvals stored (cov_svd)
        if f"{signal}_eigvecs" in entry:
            return True
        # Raw activations (non-square matrix) -> PCA-able
        if signal in entry:
            t = entry[signal]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
                return True
        return False

    @staticmethod
    def _src_acts_signal(hook_name: str) -> Optional[str]:
        """The captured acts signal feeding this hook's derivations, or None."""
        for s in hn.captured_signals(hook_name):
            if s.endswith(".acts"):
                return s
        return None

    def available(self) -> dict:
        """Return {hook_name: [signal, ...]} for all obtainable data.

        Includes captured signals with data, derived signals obtainable when the
        source data + model weights are available, and the layer / after_final_norm
        virtual hooks.
        """
        result = {}
        for hook_name in self.hook_names():
            signals = [s for s in hn.captured_signals(hook_name)
                       if self._has_eigenvalues(hook_name, s)]
            # Derived signals: stored, or derivable from source acts + model weights.
            entry = self._entry(hook_name)
            src = self._src_acts_signal(hook_name)
            has_src_cov = bool(src) and (src in entry or f"{src}_eigvecs" in entry)
            has_model = self.model is not None or self._model_name is not None
            for ds in hn.derived_signals(hook_name):
                if any(k in entry for k in (ds, f"{ds}_eigvals", f"{ds}_eigvecs")):
                    signals.append(ds)
                elif has_src_cov and has_model:
                    signals.append(ds)
            if signals:
                result[hook_name] = signals

        # Virtual hook: blkN.layer -- whole-layer K-FAC (up.in.acts x down.out.grads)
        for blk in {h.split(".")[0] for h in result if hn.block_idx(h) is not None}:
            if "in.acts" in result.get(f"{blk}.up", []) and "out.grads" in result.get(f"{blk}.down", []):
                result[f"{blk}.layer"] = ["in.acts", "out.grads"]

        # Virtual hook: after_final_norm from before_final_norm acts
        if "after_final_norm" not in result and "before_final_norm" in result:
            entry = self._entry("before_final_norm")
            t = entry.get("value.acts")
            has_acts = isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]
            has_model = self.model is not None or self._model_name is not None
            if has_acts and has_model:
                result["after_final_norm"] = ["value.acts"]

        return result

    def needs_model_weights(self) -> bool:
        """True if any derivation (derived acts / post-norm) is required but not stored."""
        for hook_name, entry in self.data.items():
            if not isinstance(entry, dict):
                continue
            src = self._src_acts_signal(hook_name)
            has_src = bool(src) and (src in entry or f"{src}_eigvecs" in entry)
            for ds in hn.derived_signals(hook_name):
                if has_src and not any(k in entry for k in (ds, f"{ds}_eigvals", f"{ds}_eigvecs")):
                    return True
        if "before_final_norm" in self.data and "after_final_norm" not in self.data:
            t = self.data.get("before_final_norm", {}).get("value.acts")
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
                return True
        return False


# ---------------------------------------------------------------------------
# View classes
# ---------------------------------------------------------------------------

class BlocksView:
    """acc.blocks[N] -> BlockView."""

    def __init__(self, acc: DataAccessor):
        self._acc = acc

    def __getitem__(self, block_idx: int) -> "BlockView":
        return BlockView(block_idx, self._acc)


class BlockView:
    """acc.blocks[N].up / .down / .gate / .attn / .mlp -> HookView or SubBlockView."""

    def __init__(self, block_idx: int, acc: DataAccessor):
        self._idx = block_idx
        self._acc = acc

    @property
    def up(self) -> "HookView":
        return self._acc[f"blk{self._idx}.up"]

    @property
    def down(self) -> "HookView":
        return self._acc[f"blk{self._idx}.down"]

    @property
    def gate(self) -> "HookView":
        return self._acc[f"blk{self._idx}.gate"]

    @property
    def attn(self) -> "SubBlockView":
        return SubBlockView(self._idx, "attn", self._acc)

    @property
    def mlp(self) -> "SubBlockView":
        return SubBlockView(self._idx, "mlp", self._acc)


class SubBlockView:
    """acc.blocks[i].attn.in / .out / .raw_out / .head[h], or .mlp.in / .out -> HookView."""

    def __init__(self, block_idx: int, sub: str, acc: DataAccessor):
        self._idx = block_idx
        self._sub = sub  # "attn" or "mlp"
        self._acc = acc

    def _hv(self, suffix: str) -> "HookView":
        return self._acc[f"blk{self._idx}.{self._sub}.{suffix}"]

    @property
    def in_(self) -> "HookView":  # 'in' is a keyword; alias as in_
        return self._hv("in")

    @property
    def out(self) -> "HookView":
        return self._hv("out")

    @property
    def raw_out(self) -> "HookView":
        return self._hv("raw_out")

    @property
    def head(self) -> "HeadsView":
        if self._sub != "attn":
            raise AttributeError("Per-head decomposition is only defined for attn sub-block")
        return HeadsView(self._idx, self._acc)


class HeadsView:
    """acc.blocks[i].attn.head[h] -> HookView for blk{i}.attn.head{h}."""

    def __init__(self, block_idx: int, acc: DataAccessor):
        self._idx = block_idx
        self._acc = acc

    def __getitem__(self, head_idx: int) -> "HookView":
        return self._acc[f"blk{self._idx}.attn.head{head_idx}"]


# Read-only aliases for Pythia parallel-residual block boundaries.
_KEY_ALIASES = [
    (re.compile(r"blk(\d+)\.mlp\.in$"),      "blk{0}.attn.in"),
    (re.compile(r"blk(\d+)\.attn\.raw_out$"), "blk{0}.attn.out"),
]


class NodeView:
    """A representation node. Always exposes .acts and .grads; either reads as
    absent (None quantities) when this hook has no such signal."""

    def __init__(self, hook_name, node, acc):
        self._hook = hook_name
        self._node = node  # in | out | value | slice | contrib
        self._acc = acc

    @property
    def acts(self):
        return FactorView(self._hook, f"{self._node}.acts", self._acc)

    @property
    def grads(self):
        return FactorView(self._hook, f"{self._node}.grads", self._acc)


class HookView:
    """acc[hook_name] -> HookView. Nodes via .in_/.out/.value/.slice/.contrib."""

    def __init__(self, hook_name: str, acc: DataAccessor):
        self._hook = hook_name
        self._acc = acc

    def _factor(self, signal: str) -> "FactorView":
        return FactorView(self._hook, signal, self._acc)

    def _node(self, kinds, node) -> "NodeView":
        if hn.classify(self._hook) not in kinds:
            raise AttributeError(f"{node!r} node does not apply to {self._hook!r}")
        return NodeView(self._hook, node, self._acc)

    @property
    def in_(self) -> "NodeView":
        return self._node(("mlp",), "in")

    @property
    def out(self) -> "NodeView":
        return self._node(("mlp",), "out")

    @property
    def value(self) -> "NodeView":
        return self._node(("residual", "boundary"), "value")

    @property
    def slice(self) -> "NodeView":
        return self._node(("ov_head",), "slice")

    @property
    def contrib(self) -> "NodeView":
        return self._node(("ov_head",), "contrib")

    def __repr__(self):
        return f"HookView({self._hook!r})"


class FactorView:
    """acc[hook]._factor(signal) -> FactorView. Identity is the signal string
    (e.g. 'in.acts', 'out.grads'). All properties are lazily computed and read as
    None when the signal has no data."""

    def __init__(self, hook_name: str, signal: str, acc: DataAccessor):
        self._hook = hook_name
        self._signal = signal
        self._acc = acc

    @property
    def eigvals(self) -> Optional[torch.Tensor]:
        """Eigenvalues (uncentered) as 1D float tensor, descending order."""
        return self._acc._eigenvalues(self._hook, self._signal)

    @property
    def eigvals_centered(self) -> Optional[torch.Tensor]:
        """Centered eigenvalues (of Cov[x] = E[xxT] - E[x]E[x]T), descending."""
        return self._acc._eigenvalues_centered(self._hook, self._signal)

    @property
    def eigvecs(self) -> Optional[torch.Tensor]:
        """Eigenvectors as columns (d, k), descending order."""
        return self._acc._eigenvectors(self._hook, self._signal)

    @property
    def eigvecs_centered(self) -> Optional[torch.Tensor]:
        """Centered eigenvectors (of Cov[x] = E[xxT] - E[x]E[x]T), columns (d, k), descending."""
        return self._acc._eigenvectors_centered(self._hook, self._signal)

    @property
    def cov(self) -> Optional[torch.Tensor]:
        """Normalized covariance E[xx^T] as (d, d) float tensor."""
        return self._acc._covariance(self._hook, self._signal)

    @property
    def samples(self) -> Optional[torch.Tensor]:
        """Selected raw sample matrix (N, d). Applies stored mask if present."""
        return self._acc._activations(self._hook, self._signal)

    @property
    def samples_raw(self) -> Optional[torch.Tensor]:
        """All raw samples (N_total, d) without applying stored mask."""
        return self._acc._activations(self._hook, self._signal, apply_mask=False)

    @property
    def mean(self) -> Optional[torch.Tensor]:
        """Mean vector (d,) if stored via +m modifier or derivable."""
        return self._acc._mean(self._hook, self._signal)

    @property
    def eigh(self) -> Optional[tuple]:
        """(eigvals: Tensor, eigvecs: Tensor) -- sorted descending."""
        V = self.eigvecs  # eigvecs first: one decomposition, eigvals from cache
        ev = self.eigvals
        if ev is None or V is None:
            return None
        return ev, V

    @property
    def eigh_centered(self) -> Optional[tuple]:
        """(eigvals_centered: Tensor, eigvecs_centered: Tensor) -- sorted descending."""
        V = self.eigvecs_centered
        ev = self.eigvals_centered
        if ev is None or V is None:
            return None
        return ev, V

    @property
    def svd(self):
        """(U, S, V) from samples if available, else (S, V) from eigdecomp."""
        return self._acc._svd(self._hook, self._signal)

    @property
    def n(self) -> Optional[int]:
        """Number of tokens/samples this signal was accumulated over.

        For covariance formats: reads stored n_{signal} or n count.
        For raw samples (N, d): reads from tensor shape.
        """
        entry = self._acc._entry(self._hook)
        v = entry.get(f"n_{self._signal}") or entry.get("n")
        if v is not None:
            return int(v)
        # Fallback: raw samples tensor shape
        t = entry.get(self._signal)
        if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
            return int(t.shape[0])
        return None

    def __repr__(self):
        return f"FactorView({self._hook!r}, {self._signal!r})"


# ===========================================================================
# Section 8: CLI
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
        """Map each input .pt to its output path. In-place by default."""
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

    # --- info ---
    p = sub.add_parser("info", help="Show stored formats, sizes, projections")
    p.add_argument("input", help="Path to .pt file or model directory")

    # --- convert ---
    p = sub.add_parser("convert", help="Convert format (acts->cov->cov_svd->eigenvalues, or reverse where possible)")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--to", required=True, metavar="FORMAT")
    p.add_argument("--output", default=None, metavar="PATH", help="Output path (only valid for single-file input)")
    p.add_argument("--output-dir", default=None, dest="output_dir", metavar="DIR",
                   help="Write outputs here, mirroring input filenames (default: in-place)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--abort-on-model-load", action="store_true",
                   help="Abort instead of lazily loading model weights for derivations such as B")

    # --- project ---
    p = sub.add_parser("project", help="Add cross-basis eigenvalue projections")
    p.add_argument("--input", required=True, metavar="PATH")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--onto", choices=["fwd", "rev", "both"],
                   help="Same-layer cross-projection: grads->forward basis (fwd), "
                        "forward->grads basis (rev), or both")
    g.add_argument("--onto-file", dest="onto_file", metavar="FILE",
                   help="Cross-checkpoint: project each file onto eigenbasis from this reference file")
    p.add_argument("--output", default=None, metavar="PATH", help="Output path (only valid for single-file input)")
    p.add_argument("--output-dir", default=None, dest="output_dir", metavar="DIR",
                   help="Write outputs here, mirroring input filenames (default: in-place)")
    p.add_argument("--workers", type=int, default=None)

    # --- set-filter ---
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
            _run_pool(
                _convert_worker,
                [(ip, op, args.to, args.abort_on_model_load) for ip, op in pairs],
                args.workers,
            )

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
        # Load first file to get existing filter as base
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
