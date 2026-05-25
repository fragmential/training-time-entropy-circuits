"""Unified data interface: read, write, convert, and inspect collected data.

Combines the DataAccessor (format-agnostic read/write interface), storage
format handling, and eigendecomposition helpers into a single module.

DataAccessor provides a chainable property-based API:

    acc = DataAccessor(data)

    acc["blk3.up"].A.eigvals           # 1D float Tensor, descending
    acc["blk3.up"].B.cov               # (d_out, d_out) Tensor, derived if model given
    acc["blk3.up"].G.eigh              # (eigvals Tensor, eigvecs Tensor)
    acc["blk3.up"].A.mean              # stored mean (+m modifier)
    acc["blk3.up"].A.svd               # (S, V) from eigdecomp, or (U, S, V) from acts
    acc.blocks[3].up.A.eigvals         # same as above, block-indexed

    acc.after_final_norm.A.eigvals     # final residual stream, post-norm
    acc.before_final_norm.A.eigvals    # final residual stream, pre-norm
    acc["blk3.up"].input.eigvals       # alias: input = A
    acc["blk3.up"].output.cov          # alias: output = B
    acc["blk3.up"].grad.eigvals        # alias: grad = G

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
"""

import os
import re
import time
import torch

from typing import Optional


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

_FACTOR_KEYS = ("A", "G")

_FORMAT_ALIASES = {
    "activations": "acts",
    "covariance":  "cov",
    "eigvals":     "eigenvalues",
}

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
    path, fmt = args
    acc = DataAccessor(path)
    return acc.save(path, format=fmt)

def _project_same_layer_worker(args):
    path, onto = args
    acc = DataAccessor(path)
    return acc.project_same_layer(onto=onto, output_path=path)

def _project_onto_file_worker(args):
    path, basis_path = args
    acc = DataAccessor(path)
    return acc.project_onto_basis(basis_path, output_path=path)

def _set_filter_worker(args):
    path, filter_dict = args
    acc = DataAccessor(path)
    return acc.set_token_filter(filter_dict, output_path=path)


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

    def __init__(self, data, model=None, model_config=None, model_name=None, revision=None):
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
        self._layer_cache: dict = {}
        self._eigh_cache: dict = {}  # (hook, factor) -> (centered_eigvals, uncentered_eigvals, eigvecs_or_None, centered_eigvecs_or_None)
        self._B_cov_cache: dict = {}  # hook -> derived B covariance (shared by .cov and .eigvecs)

    def _ensure_weights(self):
        """Ensure model weights are available (full model or selective loading)."""
        if self.model is not None or self._selective_weights is not None:
            return True
        if self._model_name is None:
            return False
        # Selective loading: only the layers we actually need
        hook_names = [k for k in self.data if re.match(r"blk\d+\.(up|down|gate)", k)]
        need_norm = "before_final_norm" in self.data and "after_final_norm" not in self.data
        if not hook_names and not need_norm:
            return False
        from utils.model_registry import get_model_config, load_selective_weights
        if self.model_config is None:
            self.model_config = get_model_config(self._model_name)
        hf_repo = self._hf_repo or self.model_config.hf_repo
        self._selective_weights = load_selective_weights(
            self.model_config, hf_repo, self._revision, hook_names, need_norm,
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
        base_format = _FORMAT_ALIASES.get(format.split("+")[0], format.split("+")[0])
        store_means = "+m" in format
        store_b = "+b" in format
        result = {}

        if base_format not in {"acts", "acts_svd", "cov", "cov_svd", "eigenvalues"}:
            raise ValueError(f"Unknown format: {format!r}")

        for hook_name in self.hook_names():
            entry = self._entry(hook_name)
            out_entry = {"n": entry.get("n", 0)}
            for k, v in entry.items():
                if k.startswith("n_") or "cross_eigvals" in k:
                    out_entry[k] = v

            factors = list(_FACTOR_KEYS)
            if store_b:
                factors.append("B")

            for factor in factors:
                self._write_factor(out_entry, hook_name, factor, base_format, store_means, storage_dtype)

            result[hook_name] = out_entry

        return result

    def _write_factor(self, out_entry, hook_name, factor, base_format, store_means, storage_dtype):
        view = self[hook_name]._factor(factor)
        n = view.n

        if base_format == "acts":
            acts = view.acts_raw
            if acts is not None:
                out_entry[factor] = _cast_for(acts.cpu(), "acts", storage_dtype)
                out_entry[f"n_{factor}"] = acts.shape[0]
                self._copy_mask(out_entry, hook_name, factor)

        elif base_format == "acts_svd":
            svd = view.svd
            if svd is not None and len(svd) == 3:
                U, S, V = svd
                out_entry[f"{factor}_U"] = _cast_for(U.cpu(), "acts", storage_dtype)
                out_entry[f"{factor}_S"] = _cast_for(S.cpu(), "eigvals", storage_dtype)
                out_entry[f"{factor}_V"] = _cast_for(V.cpu(), "eigvecs", storage_dtype)
                if n is not None:
                    out_entry[f"n_{factor}"] = n
                self._copy_mask(out_entry, hook_name, factor)

        elif base_format == "cov":
            cov = view.cov
            if cov is not None and n is not None:
                out_entry[factor] = _cast_for((cov.cpu() * n), "cov", storage_dtype)
                out_entry[f"n_{factor}"] = n

        elif base_format == "cov_svd":
            eigh = view.eigh
            if eigh is not None:
                eigvals, eigvecs = eigh
                out_entry[f"{factor}_eigvals"] = _cast_for(eigvals.cpu(), "eigvals", storage_dtype)
                out_entry[f"{factor}_eigvecs"] = _cast_for(eigvecs.cpu(), "eigvecs", storage_dtype)
                if n is not None:
                    out_entry[f"n_{factor}"] = n
                centered = view.eigvals_centered
                if centered is not None:
                    out_entry[f"{factor}_eigvals_centered"] = _cast_for(centered.cpu(), "eigvals", storage_dtype)

        elif base_format == "eigenvalues":
            eigvals = view.eigvals
            if eigvals is not None:
                out_entry[f"{factor}_eigvals"] = _cast_for(eigvals.cpu(), "eigvals", storage_dtype)
                if n is not None:
                    out_entry[f"n_{factor}"] = n
                centered = view.eigvals_centered
                if centered is not None:
                    out_entry[f"{factor}_eigvals_centered"] = _cast_for(centered.cpu(), "eigvals", storage_dtype)

        if store_means:
            mean = view.mean
            if mean is not None:
                out_entry[f"{factor}_mean"] = _cast_for(mean.cpu(), "mean", storage_dtype)

    def _copy_mask(self, out_entry, hook_name, factor):
        mask = self._entry(hook_name).get(f"{factor}_mask")
        if mask is not None:
            out_entry[f"{factor}_mask"] = mask.cpu().bool()

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
        """Project gradient G against the same hook's forward factor and save.

        The forward factor is B for MLP hooks (derived from weights if needed) and
        A for residual hooks — exactly what `hook.B` resolves to. onto: "B" projects
        G onto the forward basis, "G" the reverse, "both" both directions."""
        do_fwd = onto in ("B", "both", "BG", "GB")
        do_G = onto in ("G", "both", "BG", "GB")
        for hook in self.hook_names():
            G, fwd = self[hook].G, self[hook].B
            label = "A" if _is_residual_hook(hook) else "B"
            try:
                fwd_vecs, G_vecs = fwd.eigvecs, G.eigvecs
            except ValueError:
                continue  # MLP B needs A_mean (bias term) that wasn't stored
            if fwd_vecs is None or G_vecs is None or fwd_vecs.shape != G_vecs.shape:
                continue
            if do_fwd:
                self._entry(hook)[f"G_cross_eigvals_{label}"] = cross_eigvals(G.cov, fwd_vecs)
            if do_G:
                self._entry(hook)[f"{label}_cross_eigvals_G"] = cross_eigvals(fwd.cov, G_vecs)
        return self._save_in_place(output_path)

    def project_onto_basis(self, basis_path: str, output_path: str = None) -> str:
        """Project this data onto another file's eigenbasis (cross-checkpoint) and save."""
        self._project_onto(basis_path)
        return self._save_in_place(output_path)

    def _project_onto(self, basis_path: str) -> None:
        """Add `{factor}_cross_eigvals_{ref}` from the reference file's eigenbasis. In place."""
        ref = DataAccessor(basis_path)
        label = os.path.splitext(os.path.basename(basis_path))[0]
        for hook in self.hook_names():
            if hook not in ref.data:
                continue
            for factor in _FACTOR_KEYS:
                basis = getattr(ref[hook], factor).eigvecs
                cov = getattr(self[hook], factor).cov
                if basis is None or cov is None or cov.shape[0] != basis.shape[0]:
                    continue
                self._entry(hook)[f"{factor}_cross_eigvals_{label}"] = cross_eigvals(cov, basis)

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
        return HookView(hook_name, self)

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
        if hook_name in self._layer_cache:
            return self._layer_cache[hook_name]
        m = re.match(r"blk(\d+)\.(up|down|gate)", hook_name)
        if not m:
            return None
        if not self._ensure_weights():
            return None
        if self.model is not None:
            from utils.model_registry import get_mlp_projections
            block_idx = int(m.group(1))
            for name, layer in get_mlp_projections(self.model, self.model_config, block_idx):
                self._layer_cache[name] = layer
        elif self._selective_weights is not None and hook_name in self._selective_weights:
            self._layer_cache[hook_name] = self._selective_weights[hook_name]
        return self._layer_cache.get(hook_name)

    # ------------------------------------------------------------------
    # Internal: computation methods (used by FactorView properties)
    # ------------------------------------------------------------------

    def _ensure_eigh(self, hook_name: str, factor: str, need_vecs: bool = False, need_centered_vecs: bool = False):
        """Ensure eigendecomposition is cached for (hook, factor).

        Stores (centered_eigvals, uncentered_eigvals, eigvecs_or_None, centered_eigvecs_or_None) in _eigh_cache.
        If need_vecs and we only have eigvals cached, recomputes with full eigh.
        """
        if need_centered_vecs:
            need_vecs = True
        # blkN.layer virtual hook: redirect A->up.A, G->down.G
        orig_key = (hook_name, factor)
        if hook_name.endswith(".layer"):
            hook_name = hook_name[:-6] + (".up" if factor == "A" else ".down")
        cache_key = (hook_name, factor)
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
        if f"{factor}_eigvals" in entry and (not need_vecs or f"{factor}_eigvecs" in entry):
            if uncentered is None:
                uncentered = entry[f"{factor}_eigvals"]
            if eigvecs is None:
                eigvecs = entry.get(f"{factor}_eigvecs")
            stored_centered = entry.get(f"{factor}_eigvals_centered")
            if stored_centered is not None and centered is None:
                centered = stored_centered
            # Recover centered eigvals/eigvecs from stored eigvecs + mean if needed
            if (centered is None or need_centered_vecs) and f"{factor}_eigvecs" in entry:
                mu_t = self._mean(hook_name, factor)
                if mu_t is not None:
                    V = entry[f"{factor}_eigvecs"].to(dev)
                    S = entry[f"{factor}_eigvals"].to(dev)
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
        if (uncentered is None or (need_vecs and eigvecs is None)) and factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get(f"n_{factor}", entry.get("n", 1))
                cov = (t.to(dev) / n)
                mu_t = self._mean(hook_name, factor)
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
            acts = self._activations(hook_name, factor)
            if acts is not None:
                if need_vecs:
                    cov = self._covariance(hook_name, factor)
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

    def _eigenvalues(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            return self._B_eigenvalues(hook_name)
        self._ensure_eigh(hook_name, factor)
        return self._eigh_cache[(hook_name, factor)][1]

    def _eigenvalues_centered(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            self._ensure_B_eigh(hook_name)
            return self._eigh_cache[(hook_name, "B")][0]
        self._ensure_eigh(hook_name, factor)
        return self._eigh_cache[(hook_name, factor)][0]

    def _eigenvectors(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            return self._B_eigenvectors(hook_name)
        self._ensure_eigh(hook_name, factor, need_vecs=True)
        return self._eigh_cache[(hook_name, factor)][2]

    def _eigenvectors_centered(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            self._ensure_B_eigh(hook_name, need_centered_vecs=True)
            return self._eigh_cache[(hook_name, "B")][3]
        self._ensure_eigh(hook_name, factor, need_centered_vecs=True)
        return self._eigh_cache[(hook_name, factor)][3]

    def _covariance(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            c = self._B_covariance(hook_name)
            return c.cpu() if c is not None else None

        entry = self._entry(hook_name)

        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get(f"n_{factor}", entry.get("n", 1))
                return t.float() / n

        vk, sk = f"{factor}_eigvecs", f"{factor}_eigvals"
        if vk in entry and sk in entry:
            return reconstruct_cov(entry[sk].float(), entry[vk].float())

        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2:
                X = t.float()
                mask = entry.get(f"{factor}_mask")
                if mask is not None:
                    X = X[mask.bool()]
                return (X.T @ X) / X.shape[0]

        return None

    def _activations(self, hook_name: str, factor: str, apply_mask: bool = True) -> Optional[torch.Tensor]:
        # Virtual hook: after_final_norm from before_final_norm + norm layer
        if hook_name == "after_final_norm" and factor == "A" and hook_name not in self.data:
            return self._post_norm_activations()

        entry = self._entry(hook_name)

        raw = None
        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
                raw = t

        if raw is None:
            uk = f"{factor}_U"
            if uk in entry:
                U = entry[uk].float()
                S = entry[f"{factor}_S"].float()
                V = entry[f"{factor}_V"].float()
                raw = (U * S.unsqueeze(0)) @ V.T

        if raw is None:
            return None

        if apply_mask:
            mask = entry.get(f"{factor}_mask")
            if mask is not None:
                return raw[mask.bool()]
        return raw

    def _mean(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            return self._B_mean(hook_name)
        entry = self._entry(hook_name)
        return entry.get(f"{factor}_mean")

    def _svd(self, hook_name: str, factor: str):
        """(U, S, V) from acts if available, else (S, V) from eigdecomp."""
        acts = self._activations(hook_name, factor)
        if acts is not None:
            U, S, Vt = torch.linalg.svd(acts.float(), full_matrices=False)
            return U, S, Vt.T

        eigvals = self._eigenvalues(hook_name, factor)
        eigvecs = self._eigenvectors(hook_name, factor)
        if eigvals is not None and eigvecs is not None:
            entry = self._entry(hook_name)
            n = entry.get(f"n_{factor}", entry.get("n", 1))
            S = (eigvals.clamp(min=0) * n).sqrt().float()
            return S, eigvecs

        return None

    # ------------------------------------------------------------------
    # Internal: B derivation
    # ------------------------------------------------------------------

    def _B_covariance(self, hook_name: str) -> Optional[torch.Tensor]:
        if hook_name in self._B_cov_cache:
            return self._B_cov_cache[hook_name]
        entry = self._entry(hook_name)
        dev = "cuda" if torch.cuda.is_available() else "cpu"

        if "B" in entry:
            t = entry["B"]
            if t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get("n_B", entry.get("n", 1))
                return (t.to(dev).float() / n)

        if "B_eigvecs" in entry and "B_eigvals" in entry:
            return reconstruct_cov(entry["B_eigvals"].to(dev).float(), entry["B_eigvecs"].to(dev).float())

        A_cov = self._covariance(hook_name, "A")
        if A_cov is None:
            return None
        layer = self._get_layer(hook_name)
        if layer is None:
            return None

        W = layer.weight.detach().to(dev).float()
        b = layer.bias.detach().to(dev).float() if layer.bias is not None else None
        A_cov = A_cov.to(dev)

        B_cov = W @ A_cov @ W.T
        if b is not None:
            mu = self._mean(hook_name, "A")
            if mu is None:
                raise ValueError(
                    f"Cannot derive B for '{hook_name}': layer has a bias but no A_mean is stored. "
                    f"Re-collect with a storage format that includes the '+m' modifier (e.g. cov_svd+m)."
                )
            Wmu = W @ mu.to(device=dev, dtype=W.dtype)
            B_cov = B_cov + torch.outer(Wmu, b) + torch.outer(b, Wmu) + torch.outer(b, b)

        self._B_cov_cache[hook_name] = B_cov
        return B_cov

    def _ensure_B_eigh(self, hook_name: str, need_vecs: bool = False, need_centered_vecs: bool = False):
        """Ensure B eigendecomposition is cached."""
        if need_centered_vecs:
            need_vecs = True
        cache_key = (hook_name, "B")
        cached = self._eigh_cache.get(cache_key)
        if cached is not None:
            if not need_vecs or cached[2] is not None:
                if not need_centered_vecs or cached[3] is not None:
                    return

        entry = self._entry(hook_name)
        # Stored (sufficient if we have eigvecs or don't need them)
        if "B_eigvals" in entry and (not need_vecs or "B_eigvecs" in entry):
            self._eigh_cache[cache_key] = (None, entry["B_eigvals"], entry.get("B_eigvecs"), None)
            return

        # Compute from B covariance
        B_cov = self._B_covariance(hook_name)
        if B_cov is None:
            self._eigh_cache[cache_key] = (None, None, None, None)
            return

        centered, centered_eigvecs = None, None
        B_mean = self._B_mean(hook_name)
        if B_mean is not None:
            B_centered_cov = B_cov - torch.outer(B_mean.to(B_cov.device).float(), B_mean.to(B_cov.device).float())
            if need_centered_vecs:
                centered, centered_eigvecs = eigh_descending(B_centered_cov)
                centered, centered_eigvecs = centered.cpu(), centered_eigvecs.cpu()
            else:
                centered = eigvalsh_descending(B_centered_cov).cpu()

        if need_vecs:
            vals, vecs = eigh_descending(B_cov)
            self._eigh_cache[cache_key] = (
                centered, vals.cpu(), vecs.cpu(), centered_eigvecs)
        else:
            eigvals = eigvalsh_descending(B_cov)
            self._eigh_cache[cache_key] = (
                centered, eigvals.cpu(), None, centered_eigvecs)

    def _B_eigenvalues(self, hook_name: str) -> Optional[torch.Tensor]:
        self._ensure_B_eigh(hook_name)
        return self._eigh_cache[(hook_name, "B")][1]

    def _B_eigenvectors(self, hook_name: str) -> Optional[torch.Tensor]:
        self._ensure_B_eigh(hook_name, need_vecs=True)
        return self._eigh_cache[(hook_name, "B")][2]

    def _B_mean(self, hook_name: str) -> Optional[torch.Tensor]:
        entry = self._entry(hook_name)
        if "B_mean" in entry:
            return entry["B_mean"]
        mu = self._mean(hook_name, "A")
        if mu is None:
            return None
        layer = self._get_layer(hook_name)
        if layer is None:
            return None
        W = layer.weight.detach().float()
        b = layer.bias.detach().float() if layer.bias is not None else None
        B_mean = W @ mu.to(W.device).float()
        if b is not None:
            B_mean = B_mean + b
        return B_mean.cpu()

    # ------------------------------------------------------------------
    # Internal: norm derivation (before_final_norm -> after_final_norm)
    # ------------------------------------------------------------------

    def _post_norm_activations(self) -> Optional[torch.Tensor]:
        """Derive after_final_norm acts from before_final_norm raw acts + norm layer."""
        acts = self._activations("before_final_norm", "A")
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

    def _has_eigenvalues(self, hook_name: str, factor: str) -> bool:
        """Check if eigenvalues are obtainable without computing them."""
        entry = self._entry(hook_name)
        if f"{factor}_eigvals" in entry:
            return True
        # Raw cov (square matrix) -> eigendecomposable
        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == t.shape[1]:
                return True
        # Eigvecs + eigvals stored (cov_svd)
        if f"{factor}_eigvecs" in entry:
            return True
        # Raw activations (non-square matrix) -> PCA-able
        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
                return True
        return False

    def available(self) -> dict:
        """Return {hook_name: [factor_names]} for all obtainable data.

        Includes stored hooks and derivable virtual hooks (e.g. after_final_norm
        from before_final_norm, B from A + model weights).
        """
        result = {}
        for hook_name in self.hook_names():
            factors = []
            for f in ("A", "G"):
                if self._has_eigenvalues(hook_name, f):
                    factors.append(f)
            # B: check stored first, then derivability without triggering model load
            entry = self._entry(hook_name)
            if "B_eigvals" in entry or "B" in entry or "B_eigvecs" in entry:
                factors.append("B")
            elif re.match(r"blk\d+\.(up|down|gate)", hook_name):
                # B derivable if we have A cov (or can compute it) and model
                has_A_cov = ("A" in entry or "A_eigvecs" in entry)
                has_model = self.model is not None or self._model_name is not None
                if has_A_cov and has_model:
                    factors.append("B")
            if factors:
                result[hook_name] = factors

        # Virtual hook: blkN.layer -- whole-layer K-FAC (up.A x down.G)
        for blk in {h.split(".")[0] for h in result if re.match(r"blk\d+\.", h)}:
            if "A" in result.get(f"{blk}.up", []) and "G" in result.get(f"{blk}.down", []):
                result[f"{blk}.layer"] = ["A", "G"]

        # Virtual hook: after_final_norm from before_final_norm acts
        if "after_final_norm" not in result and "before_final_norm" in result:
            entry = self._entry("before_final_norm")
            has_acts = "A" in entry and isinstance(entry["A"], torch.Tensor) and entry["A"].dim() == 2 and entry["A"].shape[0] != entry["A"].shape[1]
            has_model = self.model is not None or self._model_name is not None
            if has_acts and has_model:
                result["after_final_norm"] = ["A"]

        return result


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
    """acc.blocks[N].up / .down / .gate -> HookView."""

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


# Residual-stream hook names: no weight matrix, so A = B (forward acts = output acts).
_RESIDUAL_HOOK_PREFIXES = ("after_final_norm", "before_final_norm", "post_attn_", "pre_block_")


def _is_residual_hook(hook_name: str) -> bool:
    return any(hook_name.startswith(p) for p in _RESIDUAL_HOOK_PREFIXES)


class HookView:
    """acc[hook_name] -> HookView. Access factors via .A / .B / .G."""

    def __init__(self, hook_name: str, acc: DataAccessor):
        self._hook = hook_name
        self._acc = acc

    def _factor(self, key: str) -> "FactorView":
        return FactorView(self._hook, key, self._acc)

    @property
    def A(self) -> "FactorView":
        return self._factor("A")

    @property
    def B(self) -> "FactorView":
        # Residual hooks have no weight matrix -- A and B are the same activations.
        if _is_residual_hook(self._hook):
            return self._factor("A")
        return self._factor("B")

    @property
    def G(self) -> "FactorView":
        return self._factor("G")

    # Aliases
    @property
    def input(self) -> "FactorView":
        return self.A

    @property
    def output(self) -> "FactorView":
        return self.B

    @property
    def grad(self) -> "FactorView":
        return self.G

    def __repr__(self):
        return f"HookView({self._hook!r})"


class FactorView:
    """acc[hook].A -> FactorView. All properties are lazily computed."""

    def __init__(self, hook_name: str, factor: str, acc: DataAccessor):
        self._hook = hook_name
        self._factor = factor
        self._acc = acc

    @property
    def eigvals(self) -> Optional[torch.Tensor]:
        """Eigenvalues (uncentered) as 1D float tensor, descending order."""
        return self._acc._eigenvalues(self._hook, self._factor)

    @property
    def eigvals_centered(self) -> Optional[torch.Tensor]:
        """Centered eigenvalues (of Cov[x] = E[xxT] - E[x]E[x]T), descending."""
        return self._acc._eigenvalues_centered(self._hook, self._factor)

    @property
    def eigvecs(self) -> Optional[torch.Tensor]:
        """Eigenvectors as columns (d, k), descending order."""
        return self._acc._eigenvectors(self._hook, self._factor)

    @property
    def eigvecs_centered(self) -> Optional[torch.Tensor]:
        """Centered eigenvectors (of Cov[x] = E[xxT] - E[x]E[x]T), columns (d, k), descending."""
        return self._acc._eigenvectors_centered(self._hook, self._factor)

    @property
    def cov(self) -> Optional[torch.Tensor]:
        """Normalized covariance E[xx^T] as (d, d) float tensor."""
        return self._acc._covariance(self._hook, self._factor)

    @property
    def acts(self) -> Optional[torch.Tensor]:
        """Selected activations (N, d). Applies stored mask if present."""
        return self._acc._activations(self._hook, self._factor)

    @property
    def acts_raw(self) -> Optional[torch.Tensor]:
        """All activations (N_total, d) without applying stored mask."""
        return self._acc._activations(self._hook, self._factor, apply_mask=False)

    @property
    def mean(self) -> Optional[torch.Tensor]:
        """Mean activation vector (d,) if stored via +m modifier."""
        return self._acc._mean(self._hook, self._factor)

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
        """(U, S, V) from acts if available, else (S, V) from eigdecomp."""
        return self._acc._svd(self._hook, self._factor)

    @property
    def n(self) -> Optional[int]:
        """Number of tokens/samples this factor was accumulated over.

        For covariance formats: reads stored n_{factor} or n count.
        For raw acts (N, d): reads from tensor shape.
        """
        entry = self._acc._entry(self._hook)
        v = entry.get(f"n_{self._factor}") or entry.get("n")
        if v is not None:
            return int(v)
        # Fallback: raw acts tensor shape
        t = entry.get(self._factor)
        if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
            return int(t.shape[0])
        return None

    def __repr__(self):
        return f"FactorView({self._hook!r}, {self._factor!r})"


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
    p.add_argument("--workers", type=int, default=None)

    # --- project ---
    p = sub.add_parser("project", help="Add cross-basis eigenvalue projections")
    p.add_argument("--input", required=True, metavar="PATH")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--onto", choices=["B", "G", "both", "BG", "GB"],
                   help="Same-layer cross-factor: G->B/A basis, B/A->G basis, or both")
    g.add_argument("--onto-file", dest="onto_file", metavar="FILE",
                   help="Cross-checkpoint: project each file onto eigenbasis from this reference file")
    p.add_argument("--output", default=None, metavar="PATH", help="Output path (only valid for single-file input)")
    p.add_argument("--workers", type=int, default=None)

    # --- set-filter ---
    p = sub.add_parser("set-filter", help="Edit stored token filter metadata")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--token-selection", choices=["all", "last"], dest="token_selection")
    p.add_argument("--skip-positions", type=int, dest="skip_positions")
    p.add_argument("--boundary-token-ids", type=int, nargs="+", dest="boundary_token_ids")
    p.add_argument("--answer-only", action="store_true", dest="answer_only")
    p.add_argument("--output", default=None, metavar="PATH")
    p.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()

    if args.command == "info":
        for pt in _pt_files(args.input):
            print(DataAccessor(pt).info())

    elif args.command == "convert":
        pts = _pt_files(args.input)
        if args.output and len(pts) > 1:
            print("Warning: --output ignored for directory input (files converted in-place)")
            args.output = None
        if len(pts) == 1:
            out = DataAccessor(pts[0]).save(args.output or pts[0], format=args.to)
            print(f"Saved to {out}")
        else:
            print(f"Converting {len(pts)} files to {args.to}...")
            _run_pool(_convert_worker, [(p, args.to) for p in pts], args.workers)

    elif args.command == "project":
        pts = _pt_files(args.input)
        if args.output and len(pts) > 1:
            print("Warning: --output ignored for directory input")
            args.output = None
        if args.onto:
            if len(pts) == 1:
                out = args.output or pts[0]
                DataAccessor(pts[0]).project_same_layer(args.onto, output_path=out)
                print(f"Saved to {out}")
            else:
                print(f"Projecting {len(pts)} files (onto={args.onto})...")
                _run_pool(_project_same_layer_worker, [(p, args.onto) for p in pts], args.workers)
        else:
            if len(pts) == 1:
                out = DataAccessor(pts[0]).project_onto_basis(args.onto_file, args.output or pts[0])
                print(f"Saved to {out}")
            else:
                print(f"Projecting {len(pts)} files onto {args.onto_file}...")
                _run_pool(_project_onto_file_worker, [(p, args.onto_file) for p in pts], args.workers)

    elif args.command == "set-filter":
        # Build filter dict
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
        if len(pts) == 1:
            out = DataAccessor(pts[0]).set_token_filter(filter_dict, args.output or pts[0])
            print(f"Saved to {out}")
        else:
            print(f"Setting token filter on {len(pts)} files...")
            _run_pool(_set_filter_worker, [(p, filter_dict) for p in pts], args.workers)

    else:
        parser.print_help()
