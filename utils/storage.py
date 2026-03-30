#!/usr/bin/env python
"""Storage format handling for collected activation / covariance data.

Storage formats (from most to least data):
    acts        — raw per-token activation vectors (N, d). Largest.
    cov         — accumulated (d,d) outer product matrix (Σ xxT) + sample count n.
    acts_svd    — SVD of activations: U(N,d), S(d,), V(d,d). Larger than acts (stores U+S+V).
                  Drops U to get cov_svd. Recoverable: (U * S) @ V.T
    cov_svd     — eigendecomposition of covariance: eigvecs V(d,k) + eigvals λ(k).
                  Default for multi-checkpoint. Has basis for cross-projection.
                  Recoverable: V @ diag(λ) @ V.T
    eigenvalues — just eigenvalues λ(k). Cheapest, no basis stored.

Full names are accepted as aliases: "activations" = "acts", "covariance" = "cov".

Modifiers (usage example: cov_svd+m):
    +m          - for cov and cov_svd: store the means of the activations.
                  Used to recover B from A.

Cross-basis projections are stored *additionally* alongside the primary format.

All files are torch .pt dicts keyed by hook point name (e.g., "blk0.up", "residual").
Each entry contains factor data with keys like "A", "G", "A_eigvals", etc.
"""

import os
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Factor keys — the signals we store per hook point
# ---------------------------------------------------------------------------
_FACTOR_KEYS = ("A", "G")

# Accepted aliases for storage format names
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

# Per-item storage dtype defaults.
# eigvals in fp64 to preserve precision from fp64 accumulation.
# eigvecs in fp32 — large (d×d), fp32 sufficient for projection.
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

    storage_dtype: None → use _DEFAULT_STORAGE_DTYPES
                   str  → apply to all item types
                   dict → per-item lookup, fallback fp32
    """
    if isinstance(storage_dtype, dict):
        dtype_str = storage_dtype.get(item_type, "fp32")
    elif isinstance(storage_dtype, str):
        dtype_str = storage_dtype
    else:
        dtype_str = _DEFAULT_STORAGE_DTYPES.get(item_type, "fp32")
    return _cast(tensor, dtype_str)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_factors(
    factors_dict: dict,
    save_path: str,
    storage_format: str = "cov",
    cross_basis_refs: Optional[list] = None,
    storage_dtype: "str | dict | None" = None,
    token_filter: Optional[dict] = None,
):
    """Save collected factors in the specified storage format.

    Args:
        factors_dict: {hook_name: {"A": Tensor, "G": Tensor, "n": int, ...}}
                      A can be (d,d) covariance or (N,d) raw activations.
        save_path: output .pt path
        storage_format: "acts" | "cov" | "cov_svd" | "eigenvalues" (with optional +m)
        cross_basis_refs: list of .pt paths with cov_svd data to project onto
        storage_dtype: None   → per-item defaults (eigvals:fp64, eigvecs/acts/cov:fp32)
                       str    → apply to all item types (e.g. "bf16" to halve all sizes)
                       dict   → per-item overrides, e.g. {"eigvals": "fp64", "eigvecs": "fp32"}
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
        result = _prepare_eigenvalues(factors_dict, storage_dtype)
    else:
        raise ValueError(f"Unknown storage_format: {storage_format!r}")

    # Add cross-basis projections if requested
    if cross_basis_refs:
        for ref_path in cross_basis_refs:
            _add_cross_basis(result, ref_path)

    if token_filter:
        result["__token_filter__"] = token_filter
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

    Reconstructable: (U * S.unsqueeze(0)) @ V.T  →  original (N, d) activations.
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

    Eigendecomposition runs in the accumulator's native dtype (fp64 if accumulated in fp64).
    """
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in _FACTOR_KEYS:
            if key in data and data[key] is not None:
                M = data[key].cpu()  # preserve accumulator dtype (fp64 or fp32)
                n = data.get(f"n_{key}", data.get("n", 1))
                entry[f"n_{key}"] = n
                eigvals, eigvecs = torch.linalg.eigh(M / n)
                idx = eigvals.argsort(descending=True)
                entry[f"{key}_eigvals"] = _cast_for(eigvals[idx], "eigvals", storage_dtype)
                entry[f"{key}_eigvecs"] = _cast_for(eigvecs[:, idx], "eigvecs", storage_dtype)
            if store_means:
                mean_key = f"{key}_mean"
                if mean_key in data and data[mean_key] is not None:
                    entry[mean_key] = _cast_for(data[mean_key].cpu(), "mean", storage_dtype)
        result[name] = entry
    return result


def _prepare_eigenvalues(factors_dict: dict, storage_dtype: str = "bf16") -> dict:
    """Compute eigenvalues only. No basis stored."""
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in _FACTOR_KEYS:
            if key in data and data[key] is not None:
                M = data[key].cpu()  # preserve accumulator dtype
                n = data.get(f"n_{key}", data.get("n", 1))
                entry[f"n_{key}"] = n
                eigvals = torch.linalg.eigvalsh(M / n).flip(0)
                entry[f"{key}_eigvals"] = _cast_for(eigvals, "eigvals", storage_dtype)
        result[name] = entry
    return result


# ---------------------------------------------------------------------------
# Cross-basis projection
# ---------------------------------------------------------------------------

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
                M = V @ torch.diag(S) @ V.T

            if M is None:
                continue

            # Project: M_proj = U_ref.T @ M @ U_ref, then eigenvalues
            M_proj = U_ref.T @ M @ U_ref
            cross_eigvals = torch.linalg.eigvalsh(M_proj).flip(0)
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
                M_other = V @ torch.diag(S) @ V.T

            if M_other is None:
                continue

            M_proj = U_this.T @ M_other @ U_this
            cross_eigvals = torch.linalg.eigvalsh(M_proj).flip(0)
            entry[f"{key}_cross_eigvals_{spectra_label}"] = cross_eigvals

    out = output_path or data_path
    torch.save(data, out)
    return out


def add_same_layer_cross_basis(data: dict, onto: str = "both") -> None:
    """Add same-layer cross-factor eigenvalue projections in-place.

    For each hook point, detects compatible same-dimension factor pairs:
      G ↔ B  for MLP hooks (both d_out)
      G ↔ A  for residual hooks (both d_model, B not stored)

    Stores results as:
      G_cross_eigvals_B  (or G_cross_eigvals_A for residual)
      B_cross_eigvals_G  (or A_cross_eigvals_G for residual)

    Args:
        data: loaded .pt dict, modified in-place
        onto: "B" — project G onto fwd-factor basis
              "G" — project fwd-factor onto G basis
              "both" / "BG" / "GB" — both directions
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
            continue  # dimensions differ — skip

        if do_onto_fwd:
            M_G    = G_V @ torch.diag(G_S) @ G_V.T
            M_proj = fwd_V.T @ M_G @ fwd_V
            entry[f"G_cross_eigvals_{fwd_fac}"] = torch.linalg.eigvalsh(M_proj).flip(0)

        if do_onto_G:
            M_fwd  = fwd_V @ torch.diag(fwd_S) @ fwd_V.T
            M_proj = G_V.T @ M_fwd @ G_V
            entry[f"{fwd_fac}_cross_eigvals_G"] = torch.linalg.eigvalsh(M_proj).flip(0)


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
        acts     → cov → cov_svd → eigenvalues   (progressive reduction, lossless until eigenvalues)
        cov_svd  → cov                            (reconstruct from eigdecomp × n)
        acts_svd → acts                           (reconstruct U S V^T)
        acts_svd → cov_svd                        (drop U, derive eigenvalues from S²/n)
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
        # Copy over cross-basis and counts
        for k, v in entry.items():
            if k.startswith("n_") or "cross_eigvals" in k:
                new_entry[k] = v

        for key in _FACTOR_KEYS:
            # Handle acts → cov conversion: compute xxT from raw activations
            is_raw_acts = (
                key in entry
                and isinstance(entry[key], torch.Tensor)
                and entry[key].dim() == 2
                and entry[key].shape[0] != entry[key].shape[1]
            )

            if to_format == "cov":
                if is_raw_acts:
                    # acts → cov
                    X = entry[key].float()
                    new_entry[key] = X.T @ X
                    new_entry[f"n_{key}"] = X.shape[0]
                elif f"{key}_eigvecs" in entry and f"{key}_eigvals" in entry:
                    # Reconstruct from svd
                    V = entry[f"{key}_eigvecs"]
                    S = entry[f"{key}_eigvals"]
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    new_entry[key] = (V @ torch.diag(S) @ V.T) * n
                elif key in entry and not is_raw_acts:
                    new_entry[key] = entry[key]

            elif to_format == "cov_svd":
                if is_raw_acts:
                    # acts → cov_svd: compute covariance then eigendecompose
                    X = entry[key].float()
                    n = X.shape[0]
                    M = (X.T @ X) / n
                    eigvals, eigvecs = torch.linalg.eigh(M)
                    idx = eigvals.argsort(descending=True)
                    new_entry[f"{key}_eigvals"] = eigvals[idx]
                    new_entry[f"{key}_eigvecs"] = eigvecs[:, idx]
                    new_entry[f"n_{key}"] = n
                elif key in entry and not is_raw_acts:
                    # cov → cov_svd
                    M = entry[key].float()
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    eigvals, eigvecs = torch.linalg.eigh(M / n)
                    idx = eigvals.argsort(descending=True)
                    new_entry[f"{key}_eigvals"] = eigvals[idx]
                    new_entry[f"{key}_eigvecs"] = eigvecs[:, idx]
                elif f"{key}_U" in entry:
                    # acts_svd → cov_svd (drop U, keep V and S)
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
                    # acts → eigenvalues
                    X = entry[key].float()
                    n = X.shape[0]
                    eigvals = torch.linalg.eigvalsh((X.T @ X) / n).flip(0)
                    new_entry[f"{key}_eigvals"] = eigvals
                    new_entry[f"n_{key}"] = n
                elif f"{key}_eigvals" in entry:
                    new_entry[f"{key}_eigvals"] = entry[f"{key}_eigvals"]
                elif key in entry and not is_raw_acts:
                    M = entry[key].float()
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    eigvals = torch.linalg.eigvalsh(M / n).flip(0)
                    new_entry[f"{key}_eigvals"] = eigvals

            elif to_format == "acts":
                if is_raw_acts:
                    new_entry[key] = entry[key]
                elif f"{key}_U" in entry:
                    # acts_svd → acts: reconstruct from U S V^T
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
        prog="python -m utils.storage",
        description="Storage tool: convert, project, inspect .pt factor files",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- info ---
    p = sub.add_parser("info", help="Show stored formats, sizes, projections")
    p.add_argument("input", help="Path to .pt file or model directory")

    # --- convert ---
    p = sub.add_parser("convert", help="Convert format (acts→cov→cov_svd→eigenvalues, or reverse where possible)")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--to", required=True, choices=["acts", "acts_svd", "cov", "cov_svd", "eigenvalues"], metavar="FORMAT")
    p.add_argument("--output", default=None, metavar="PATH", help="Output path (only valid for single-file input)")
    p.add_argument("--workers", type=int, default=None)

    # --- project ---
    p = sub.add_parser("project", help="Add cross-basis eigenvalue projections")
    p.add_argument("--input", required=True, metavar="PATH")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--onto", choices=["B", "G", "both", "BG", "GB"],
                   help="Same-layer cross-factor: G→B/A basis, B/A→G basis, or both")
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
