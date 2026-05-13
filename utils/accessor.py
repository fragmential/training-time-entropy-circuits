"""Unified data interface: read, write, convert, and inspect collected data.

Combines the DataAccessor (format-agnostic reader), storage format handling
(save/convert/project), and eigendecomposition helpers into a single module.

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
# Section 3: Storage constants and helpers (from storage.py)
# ===========================================================================

_FACTOR_KEYS = ("A", "G")
_STORABLE_FACTOR_KEYS = ("A", "G", "B")  # includes derived factors

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


# ===========================================================================
# Section 4: Storage write functions (from storage.py)
# ===========================================================================

def save_factors(
    factors_dict: dict,
    save_path: str,
    storage_format: str = "cov",
    cross_basis_refs: Optional[list] = None,
    storage_dtype: "str | dict | None" = None,
    token_filter: Optional[dict] = None,
    n_chunks: Optional[int] = None,
):
    """Save collected factors in the specified storage format.

    Args:
        factors_dict: {hook_name: {"A": Tensor, "G": Tensor, "n": int, ...}}
                      A can be (d,d) covariance or (N,d) raw activations.
        save_path: output .pt path
        storage_format: "acts" | "cov" | "cov_svd" | "eigenvalues" (with optional +m)
        cross_basis_refs: list of .pt paths with cov_svd data to project onto
        storage_dtype: None   -> per-item defaults (eigvals:fp64, eigvecs/acts/cov:fp32)
                       str    -> apply to all item types (e.g. "bf16" to halve all sizes)
                       dict   -> per-item overrides, e.g. {"eigvals": "fp64", "eigvecs": "fp32"}
    """
    base_format = _FORMAT_ALIASES.get(storage_format.split("+")[0], storage_format.split("+")[0])
    store_means = "+m" in storage_format

    if base_format == "acts":
        result = _prepare_acts(factors_dict, storage_dtype)
    elif base_format == "acts_svd":
        result = _prepare_acts_svd(factors_dict, storage_dtype)
    elif base_format == "cov":
        result = _prepare_cov(factors_dict, store_means=store_means, storage_dtype=storage_dtype)
    elif base_format == "cov_svd":
        result = _prepare_cov_svd(factors_dict, store_means=store_means, storage_dtype=storage_dtype)
    elif base_format == "eigenvalues":
        result = _prepare_eigenvalues(factors_dict, store_means=store_means, storage_dtype=storage_dtype)
    else:
        raise ValueError(f"Unknown storage_format: {storage_format!r}")

    # Add cross-basis projections if requested
    if cross_basis_refs:
        for ref_path in cross_basis_refs:
            _add_cross_basis(result, ref_path)

    if token_filter:
        result["__token_filter__"] = token_filter
    if n_chunks is not None:
        result["__n_chunks__"] = n_chunks
    result["__format__"] = storage_format
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(result, save_path)


# ---------------------------------------------------------------------------
# Format preparation
# ---------------------------------------------------------------------------

def _prepare_acts(factors_dict: dict, storage_dtype=None) -> dict:
    """Store raw activation tensors + counts."""
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in _FACTOR_KEYS:
            if key in data and data[key] is not None:
                t = data[key]
                entry[key] = _cast_for(t.cpu(), "acts", storage_dtype)
            n_key = f"n_{key}"
            if n_key in data:
                entry[n_key] = data[n_key]
            mask_key = f"{key}_mask"
            if mask_key in data and data[mask_key] is not None:
                entry[mask_key] = data[mask_key].cpu().bool()
        result[name] = entry
    return result


def _prepare_acts_svd(factors_dict: dict, storage_dtype=None) -> dict:
    """Compute full SVD of raw activations. Stores U(N,d), S(d,), V(d,d) + counts.

    Reconstructable: (U * S.unsqueeze(0)) @ V.T  ->  original (N, d) activations.
    Convert to cov_svd by dropping U (see convert()).
    """
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in _FACTOR_KEYS:
            if key not in data or data[key] is None:
                continue
            t = data[key]
            if not (isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]):
                continue  # skip covariance matrices
            X = t.float().cpu()
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            entry[f"{key}_U"] = _cast_for(U, "acts", storage_dtype)
            entry[f"{key}_S"] = _cast_for(S, "eigvals", storage_dtype)
            entry[f"{key}_V"] = _cast_for(Vt.T, "eigvecs", storage_dtype)
            entry[f"n_{key}"] = X.shape[0]
            mask_key = f"{key}_mask"
            if mask_key in data and data[mask_key] is not None:
                entry[mask_key] = data[mask_key].cpu().bool()
        result[name] = entry
    return result


def _prepare_cov(factors_dict: dict, store_means: bool = False, storage_dtype=None) -> dict:
    """Store raw covariance matrices + counts, optionally with means."""
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in _FACTOR_KEYS:
            if key in data and data[key] is not None:
                entry[key] = _cast_for(data[key].cpu(), "cov", storage_dtype)
            n_key = f"n_{key}"
            if n_key in data:
                entry[n_key] = data[n_key]
            if store_means:
                mean_key = f"{key}_mean"
                if mean_key in data and data[mean_key] is not None:
                    entry[mean_key] = _cast_for(data[mean_key].cpu(), "mean", storage_dtype)
        result[name] = entry
    return result


def _prepare_cov_svd(factors_dict: dict, store_means: bool = False, storage_dtype=None) -> dict:
    """Eigendecompose each covariance matrix. Store eigvecs + eigvals, optionally means.

    Uses torch.linalg.eigh on GPU when available, falls back to CPU.
    Also stores centered eigenvalues when means are available.
    """
    import time as _t
    decomp_profiler.enable()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_total = _t.time()
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in _FACTOR_KEYS:
            if key in data and data[key] is not None:
                M = data[key]
                n = data.get(f"n_{key}", data.get("n", 1))
                entry[f"n_{key}"] = n
                cov = (M.float() / n).to(device)
                eigvals, eigvecs = eigh_descending(cov)
                eigvals = eigvals.cpu()
                eigvecs = eigvecs.cpu()
                entry[f"{key}_eigvals"] = _cast_for(eigvals, "eigvals", storage_dtype)
                entry[f"{key}_eigvecs"] = _cast_for(eigvecs, "eigvecs", storage_dtype)
                # Centered eigenvalues if mean available
                mean_key = f"{key}_mean"
                if mean_key in data and data[mean_key] is not None:
                    mu = data[mean_key].float().to(device)
                    centered_cov = cov - torch.outer(mu, mu)
                    c = eigvalsh_descending(centered_cov).cpu()
                    entry[f"{key}_eigvals_centered"] = _cast_for(c, "eigvals", storage_dtype)
            if store_means:
                mean_key = f"{key}_mean"
                if mean_key in data and data[mean_key] is not None:
                    entry[mean_key] = _cast_for(data[mean_key].cpu(), "mean", storage_dtype)
        result[name] = entry
    print(f"  cov_svd save ({device}): {_t.time()-t_total:.1f}s")
    print(decomp_profiler.summary())
    decomp_profiler.disable()
    return result


def _prepare_eigenvalues(factors_dict: dict, store_means: bool = False, storage_dtype=None) -> dict:
    """Compute eigenvalues only. No basis stored."""
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in _STORABLE_FACTOR_KEYS:
            if key in data and data[key] is not None:
                M = data[key].cpu()  # preserve accumulator dtype
                n = data.get(f"n_{key}", data.get("n", 1))
                entry[f"n_{key}"] = n
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                cov = (M.to(dev) / n)
                mean_key = f"{key}_mean"
                mu = data[mean_key].to(device=dev, dtype=cov.dtype) if mean_key in data and data[mean_key] is not None else None
                centered, uncentered = get_eigenspectrum(cov=cov, mu=mu)
                entry[f"{key}_eigvals"] = _cast_for(uncentered.cpu(), "eigvals", storage_dtype)
                if centered is not None:
                    entry[f"{key}_eigvals_centered"] = _cast_for(centered.cpu(), "eigvals", storage_dtype)
            if store_means:
                mean_key = f"{key}_mean"
                if mean_key in data and data[mean_key] is not None:
                    entry[mean_key] = _cast_for(data[mean_key].cpu(), "mean", storage_dtype)
        result[name] = entry
    return result


# ===========================================================================
# Section 5: Cross-basis and format operations (from storage.py)
# ===========================================================================

def _add_cross_basis(result: dict, ref_path: str):
    """Project data in result onto eigenbasis from ref_path. Adds cross-eigenvalues."""
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_label = os.path.splitext(os.path.basename(ref_path))[0]

    for name in result:
        if name.startswith("__"):
            continue
        if name not in ref:
            continue
        entry = result[name]
        ref_entry = ref[name]

        for key in _FACTOR_KEYS:
            eigvecs_key = f"{key}_eigvecs"
            if eigvecs_key not in ref_entry:
                continue

            U_ref = ref_entry[eigvecs_key]  # (d, k)

            # Try to project from covariance matrix or from our own svd
            M = None
            if key in entry:
                t = entry[key]
                # Only project from (d,d) covariance, not (N,d) raw acts
                if t.dim() == 2 and t.shape[0] == t.shape[1]:
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    M = t / n
            if M is None and f"{key}_eigvals" in entry and eigvecs_key in entry:
                # Reconstruct from our svd
                V = entry[eigvecs_key]
                S = entry[f"{key}_eigvals"]
                M = reconstruct_cov(S, V)

            if M is None:
                continue

            # Project: M_proj = U_ref.T @ M @ U_ref, then eigenvalues
            M_proj = U_ref.T @ M @ U_ref
            cross_eigvals = eigvalsh_descending(M_proj)
            entry[f"{key}_cross_eigvals_{ref_label}"] = cross_eigvals


def project_onto_basis(data_path: str, basis_path: str, output_path: str = None):
    """CLI-friendly: add cross-basis eigenvalues from basis_path into data_path."""
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    _add_cross_basis(data, basis_path)
    out = output_path or data_path
    torch.save(data, out)
    return out


def project_spectra_onto_this(data_path: str, spectra_path: str, output_path: str = None):
    """CLI-friendly: project spectra from spectra_path onto eigenbasis in data_path."""
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    spectra = torch.load(spectra_path, map_location="cpu", weights_only=False)
    spectra_label = os.path.splitext(os.path.basename(spectra_path))[0]

    for name in data:
        if name.startswith("__"):
            continue
        if name not in spectra:
            continue
        entry = data[name]
        spec_entry = spectra[name]

        for key in _FACTOR_KEYS:
            eigvecs_key = f"{key}_eigvecs"
            if eigvecs_key not in entry:
                continue
            U_this = entry[eigvecs_key]

            # Get their covariance
            M_other = None
            if key in spec_entry:
                t = spec_entry[key]
                if t.dim() == 2 and t.shape[0] == t.shape[1]:
                    n = spec_entry.get(f"n_{key}", spec_entry.get("n", 1))
                    M_other = t.float() / n
            if M_other is None and f"{key}_eigvals" in spec_entry and eigvecs_key in spec_entry:
                V = spec_entry[eigvecs_key]
                S = spec_entry[f"{key}_eigvals"]
                M_other = reconstruct_cov(S, V)

            if M_other is None:
                continue

            M_proj = U_this.T @ M_other @ U_this
            cross_eigvals = eigvalsh_descending(M_proj)
            entry[f"{key}_cross_eigvals_{spectra_label}"] = cross_eigvals

    out = output_path or data_path
    torch.save(data, out)
    return out


def add_same_layer_cross_basis(data: dict, onto: str = "both") -> None:
    """Add same-layer cross-factor eigenvalue projections in-place.

    For each hook point, detects compatible same-dimension factor pairs:
      G <-> B  for MLP hooks (both d_out)
      G <-> A  for residual hooks (both d_model, B not stored)

    Stores results as:
      G_cross_eigvals_B  (or G_cross_eigvals_A for residual)
      B_cross_eigvals_G  (or A_cross_eigvals_G for residual)

    Args:
        data: loaded .pt dict, modified in-place
        onto: "B" -- project G onto fwd-factor basis
              "G" -- project fwd-factor onto G basis
              "both" / "BG" / "GB" -- both directions
    """
    do_onto_fwd = onto in ("B", "both", "BG", "GB")
    do_onto_G   = onto in ("G", "both", "BG", "GB")

    for name, entry in data.items():
        if name.startswith("__"):
            continue

        has_G = "G_eigvecs" in entry and "G_eigvals" in entry
        if not has_G:
            continue

        # Prefer B (MLP), fall back to A (residual)
        if "B_eigvecs" in entry and "B_eigvals" in entry:
            fwd_fac = "B"
        elif "A_eigvecs" in entry and "A_eigvals" in entry:
            fwd_fac = "A"
        else:
            continue

        fwd_V = entry[f"{fwd_fac}_eigvecs"].float()
        fwd_S = entry[f"{fwd_fac}_eigvals"].float()
        G_V   = entry["G_eigvecs"].float()
        G_S   = entry["G_eigvals"].float()

        if fwd_V.shape != G_V.shape:
            continue  # dimensions differ -- skip

        if do_onto_fwd:
            M_G    = reconstruct_cov(G_S, G_V)
            M_proj = fwd_V.T @ M_G @ fwd_V
            entry[f"G_cross_eigvals_{fwd_fac}"] = eigvalsh_descending(M_proj)

        if do_onto_G:
            M_fwd  = reconstruct_cov(fwd_S, fwd_V)
            M_proj = G_V.T @ M_fwd @ G_V
            entry[f"{fwd_fac}_cross_eigvals_G"] = eigvalsh_descending(M_proj)


def set_token_filter(path: str, filter_dict: dict, output_path: str = None) -> str:
    """Edit the stored token filter metadata in a .pt file."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    data["__token_filter__"] = filter_dict
    out = output_path or path
    torch.save(data, out)
    return out


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def convert(input_path: str, to_format: str, output_path: str = None):
    """Convert a stored .pt file to a different format.

    Supported conversions:
        acts     -> cov -> cov_svd -> eigenvalues   (progressive reduction, lossless until eigenvalues)
        cov_svd  -> cov                            (reconstruct from eigdecomp x n)
        acts_svd -> acts                           (reconstruct U S V^T)
        acts_svd -> cov_svd                        (drop U, derive eigenvalues from S^2/n)
    """
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    src_format = _FORMAT_ALIASES.get(data.get("__format__", "cov").split("+")[0],
                                     data.get("__format__", "cov").split("+")[0])
    to_format = _FORMAT_ALIASES.get(to_format, to_format)

    if to_format == src_format:
        return input_path

    result = {}
    for name, entry in data.items():
        if name.startswith("__"):
            continue

        new_entry = {"n": entry.get("n", 0)}
        # Copy over cross-basis, counts, and means
        for k, v in entry.items():
            if k.startswith("n_") or "cross_eigvals" in k or k.endswith("_mean"):
                new_entry[k] = v

        for key in _FACTOR_KEYS:
            # Handle acts -> cov conversion: compute xxT from raw activations
            is_raw_acts = (
                key in entry
                and isinstance(entry[key], torch.Tensor)
                and entry[key].dim() == 2
                and entry[key].shape[0] != entry[key].shape[1]
            )

            if to_format == "cov":
                if is_raw_acts:
                    # acts -> cov
                    X = entry[key].float()
                    new_entry[key] = X.T @ X
                    new_entry[f"n_{key}"] = X.shape[0]
                elif f"{key}_eigvecs" in entry and f"{key}_eigvals" in entry:
                    # Reconstruct from svd
                    V = entry[f"{key}_eigvecs"]
                    S = entry[f"{key}_eigvals"]
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    new_entry[key] = reconstruct_cov(S, V) * n
                elif key in entry and not is_raw_acts:
                    new_entry[key] = entry[key]

            elif to_format == "cov_svd":
                if is_raw_acts:
                    # acts -> cov_svd: compute covariance then eigendecompose
                    X = entry[key].float()
                    n = X.shape[0]
                    M = (X.T @ X) / n
                    eigvals, eigvecs = eigh_descending(M)
                    new_entry[f"{key}_eigvals"] = eigvals
                    new_entry[f"{key}_eigvecs"] = eigvecs
                    new_entry[f"n_{key}"] = n
                elif key in entry and not is_raw_acts:
                    # cov -> cov_svd
                    M = entry[key].float()
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    eigvals, eigvecs = eigh_descending(M / n)
                    new_entry[f"{key}_eigvals"] = eigvals
                    new_entry[f"{key}_eigvecs"] = eigvecs
                elif f"{key}_U" in entry:
                    # acts_svd -> cov_svd (drop U, keep V and S)
                    new_entry[f"{key}_eigvecs"] = entry[f"{key}_V"]
                    S = entry[f"{key}_S"]
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    new_entry[f"{key}_eigvals"] = (S ** 2) / n
                elif f"{key}_eigvals" in entry:
                    # Already have svd data, copy
                    new_entry[f"{key}_eigvals"] = entry[f"{key}_eigvals"]
                    if f"{key}_eigvecs" in entry:
                        new_entry[f"{key}_eigvecs"] = entry[f"{key}_eigvecs"]

            elif to_format == "eigenvalues":
                if is_raw_acts:
                    # acts -> eigenvalues
                    X = entry[key].float()
                    n = X.shape[0]
                    eigvals = eigvalsh_descending((X.T @ X) / n)
                    new_entry[f"{key}_eigvals"] = eigvals
                    new_entry[f"n_{key}"] = n
                elif f"{key}_eigvals" in entry:
                    new_entry[f"{key}_eigvals"] = entry[f"{key}_eigvals"]
                elif key in entry and not is_raw_acts:
                    M = entry[key].float()
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    eigvals = eigvalsh_descending(M / n)
                    new_entry[f"{key}_eigvals"] = eigvals

            elif to_format == "acts":
                if is_raw_acts:
                    new_entry[key] = entry[key]
                elif f"{key}_U" in entry:
                    # acts_svd -> acts: reconstruct from U S V^T
                    U = entry[f"{key}_U"].float()
                    S = entry[f"{key}_S"].float()
                    V = entry[f"{key}_V"].float()
                    new_entry[key] = (U * S.unsqueeze(0)) @ V.T

            elif to_format == "acts_svd":
                if f"{key}_U" in entry:
                    new_entry[f"{key}_U"] = entry[f"{key}_U"]
                    new_entry[f"{key}_S"] = entry[f"{key}_S"]
                    new_entry[f"{key}_V"] = entry[f"{key}_V"]
                    if f"n_{key}" in entry:
                        new_entry[f"n_{key}"] = entry[f"n_{key}"]
                elif is_raw_acts:
                    X = entry[key].float()
                    U, S, Vt = torch.linalg.svd(X, full_matrices=False)
                    new_entry[f"{key}_U"] = U
                    new_entry[f"{key}_S"] = S
                    new_entry[f"{key}_V"] = Vt.T
                    new_entry[f"n_{key}"] = X.shape[0]

        result[name] = new_entry

    result["__format__"] = to_format
    out = output_path or input_path
    torch.save(result, out)
    return out


def trim_svd(input_path: str, output_path: str = None):
    """Drop U matrices from acts_svd format, converting to cov_svd."""
    return convert(input_path, "cov_svd", output_path)


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

def info(path: str) -> str:
    """Print summary of stored data."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    fmt = data.get("__format__", "unknown")
    lines = [f"File: {path}", f"Format: {fmt}", ""]

    tf = data.get("__token_filter__")
    if tf:
        lines.append("Token filter:")
        for k, v in tf.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    for name, entry in sorted(data.items()):
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


def _convert_worker(args):
    path, fmt = args
    return convert(path, fmt)

def _project_same_layer_worker(args):
    path, onto = args
    data = torch.load(path, map_location="cpu", weights_only=False)
    add_same_layer_cross_basis(data, onto)
    torch.save(data, path)
    return path

def _project_onto_file_worker(args):
    path, basis_path = args
    return project_onto_basis(path, basis_path)

def _set_filter_worker(args):
    path, filter_dict = args
    return set_token_filter(path, filter_dict)


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
        if uncentered is None and factor in entry:
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
                            centered = get_eigenspectrum(cov=cov, mu=mu)[0].cpu()
                else:
                    c, u = get_eigenspectrum(cov=cov, mu=mu)
                    centered = c.cpu() if c is not None else None
                    uncentered = u.cpu()

        # Fallback: raw activations
        if uncentered is None:
            acts = self._activations(hook_name, factor)
            if acts is not None:
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
        ev = self.eigvals
        V = self.eigvecs
        if ev is None or V is None:
            return None
        return ev, V

    @property
    def eigh_centered(self) -> Optional[tuple]:
        """(eigvals_centered: Tensor, eigvecs_centered: Tensor) -- sorted descending."""
        ev = self.eigvals_centered
        V = self.eigvecs_centered
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
# Section 8: CLI (from storage.py)
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
    p.add_argument("--to", required=True, choices=["acts", "acts_svd", "cov", "cov_svd", "eigenvalues"], metavar="FORMAT")
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
            print(info(pt))

    elif args.command == "convert":
        pts = _pt_files(args.input)
        if args.output and len(pts) > 1:
            print("Warning: --output ignored for directory input (files converted in-place)")
            args.output = None
        if len(pts) == 1:
            out = convert(pts[0], args.to, args.output)
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
                data = torch.load(pts[0], map_location="cpu", weights_only=False)
                add_same_layer_cross_basis(data, args.onto)
                out = args.output or pts[0]
                torch.save(data, out)
                print(f"Saved to {out}")
            else:
                print(f"Projecting {len(pts)} files (onto={args.onto})...")
                _run_pool(_project_same_layer_worker, [(p, args.onto) for p in pts], args.workers)
        else:
            if len(pts) == 1:
                out = project_onto_basis(pts[0], args.onto_file, args.output)
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
            out = set_token_filter(pts[0], filter_dict, args.output)
            print(f"Saved to {out}")
        else:
            print(f"Setting token filter on {len(pts)} files...")
            _run_pool(_set_filter_worker, [(p, filter_dict) for p in pts], args.workers)

    else:
        parser.print_help()
