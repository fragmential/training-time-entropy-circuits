"""Storage format handling for collected activation / covariance data.

Storage formats (from most to least data):
    acts        — raw per-token activation vectors (N, d). Largest.
    cov         — accumulated (d,d) outer product matrix (Σ xxT) + sample count n.
    acts_svd    — SVD of activations: U(N,k), S(k), V(d,k). Numerically stable.
    cov_svd     — eigendecomposition of covariance: eigvecs V(d,k) + eigvals λ(k).
                  Default for multi-checkpoint. Has basis for cross-projection.
                  Recoverable: V @ diag(λ) @ V.T
    eigenvalues — just eigenvalues λ(k). Cheapest, no basis stored.

Modifiers (usage example: cov_svd+m):
    +m          - for cov and cov_svd: store the means of the activations.
                  Used to recover B from A.

Cross-basis projections are stored *additionally* alongside the primary format.

All files are torch .pt dicts keyed by projection name (e.g., "blk0.up").
"""

import os
import torch
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_factors(
    factors_dict: dict,
    save_path: str,
    storage_format: str = "cov",
    cross_basis_refs: Optional[list] = None,
):
    """Save collected factors in the specified storage format.

    Args:
        factors_dict: {proj_name: {"A": Tensor, "G": Tensor, "B": Tensor, "n": int, ...}}
        save_path: output .pt path
        storage_format: "cov" | "cov_svd" | "eigenvalues"
        cross_basis_refs: list of .pt paths with cov_svd data to project onto
    """
    base_format = storage_format.split("+")[0]
    store_means = "+m" in storage_format

    if base_format == "cov":
        result = _prepare_cov(factors_dict, store_means=store_means)
    elif base_format == "cov_svd":
        result = _prepare_cov_svd(factors_dict, store_means=store_means)
    elif base_format == "eigenvalues":
        result = _prepare_eigenvalues(factors_dict)
    else:
        raise ValueError(f"Unknown storage_format: {storage_format}")

    # Add cross-basis projections if requested
    if cross_basis_refs:
        for ref_path in cross_basis_refs:
            _add_cross_basis(result, ref_path)

    result["__format__"] = storage_format
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(result, save_path)


def save_activations(activations: np.ndarray, save_path: str):
    """Save raw activations as numpy array (.npy)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, activations)


# ---------------------------------------------------------------------------
# Format preparation
# ---------------------------------------------------------------------------

def _prepare_cov(factors_dict: dict, store_means: bool = False) -> dict:
    """Store raw covariance matrices + counts, optionally with means."""
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in ("A", "B", "G"):
            if key in data and data[key] is not None:
                entry[key] = data[key].cpu().float()
            n_key = f"n_{key}"
            if n_key in data:
                entry[n_key] = data[n_key]
            if store_means:
                mean_key = f"{key}_mean"
                if mean_key in data and data[mean_key] is not None:
                    entry[mean_key] = data[mean_key].cpu().float()
        result[name] = entry
    return result


def _prepare_cov_svd(factors_dict: dict, store_means: bool = False) -> dict:
    """Eigendecompose each covariance matrix. Store eigvecs + eigvals, optionally means."""
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in ("A", "B", "G"):
            if key in data and data[key] is not None:
                M = data[key].cpu().float()
                n = data.get(f"n_{key}", data.get("n", 1))
                entry[f"n_{key}"] = n
                # Eigendecompose the normalized covariance
                eigvals, eigvecs = torch.linalg.eigh(M / n)
                # Sort descending
                idx = eigvals.argsort(descending=True)
                entry[f"{key}_eigvals"] = eigvals[idx]
                entry[f"{key}_eigvecs"] = eigvecs[:, idx]
            if store_means:
                mean_key = f"{key}_mean"
                if mean_key in data and data[mean_key] is not None:
                    entry[mean_key] = data[mean_key].cpu().float()
        result[name] = entry
    return result


def _prepare_eigenvalues(factors_dict: dict) -> dict:
    """Compute eigenvalues only. No basis stored."""
    result = {}
    for name, data in factors_dict.items():
        entry = {"n": data.get("n", 0)}
        for key in ("A", "B", "G"):
            if key in data and data[key] is not None:
                M = data[key].cpu().float()
                n = data.get(f"n_{key}", data.get("n", 1))
                entry[f"n_{key}"] = n
                eigvals = torch.linalg.eigvalsh(M / n).flip(0)
                entry[f"{key}_eigvals"] = eigvals
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

        for key in ("A", "B", "G"):
            eigvecs_key = f"{key}_eigvecs"
            if eigvecs_key not in ref_entry:
                continue

            U_ref = ref_entry[eigvecs_key]  # (d, k)

            # Try to project from covariance matrix or from our own svd
            M = None
            if key in entry:
                # We have the raw covariance
                n = entry.get(f"n_{key}", entry.get("n", 1))
                M = entry[key] / n
            elif f"{key}_eigvals" in entry and eigvecs_key in entry:
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

        for key in ("A", "B", "G"):
            eigvecs_key = f"{key}_eigvecs"
            if eigvecs_key not in entry:
                continue
            U_this = entry[eigvecs_key]

            # Get their covariance
            M_other = None
            if key in spec_entry:
                n = spec_entry.get(f"n_{key}", spec_entry.get("n", 1))
                M_other = spec_entry[key].float() / n
            elif f"{key}_eigvals" in spec_entry and eigvecs_key in spec_entry:
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


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def convert(input_path: str, to_format: str, output_path: str = None):
    """Convert a stored .pt file to a different format.

    Supported conversions:
        cov → cov_svd → eigenvalues  (progressive reduction)
        acts_svd → cov_svd           (drop U)
    """
    data = torch.load(input_path, map_location="cpu", weights_only=False)
    src_format = data.get("__format__", "cov")

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

        for key in ("A", "B", "G"):
            if to_format == "cov_svd":
                if key in entry:
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
                if f"{key}_eigvals" in entry:
                    new_entry[f"{key}_eigvals"] = entry[f"{key}_eigvals"]
                elif key in entry:
                    M = entry[key].float()
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    eigvals = torch.linalg.eigvalsh(M / n).flip(0)
                    new_entry[f"{key}_eigvals"] = eigvals

            elif to_format == "cov":
                # Reconstruct from svd if possible
                if f"{key}_eigvecs" in entry and f"{key}_eigvals" in entry:
                    V = entry[f"{key}_eigvecs"]
                    S = entry[f"{key}_eigvals"]
                    n = entry.get(f"n_{key}", entry.get("n", 1))
                    new_entry[key] = (V @ torch.diag(S) @ V.T) * n
                elif key in entry:
                    new_entry[key] = entry[key]

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
