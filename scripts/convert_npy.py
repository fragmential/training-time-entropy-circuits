#!/usr/bin/env python
"""Convert old .npy activation files to the unified .pt format.

Old format: numpy array of shape (N, d) saved as step{N}.npy
New format: .pt dict keyed by hook name, e.g. {"after_final_norm": {"A": tensor(N,d), "n_A": N}, "__format__": "acts"}

Usage:
    python scripts/convert_npy.py activations/fineweb/pythia-14m-deduped
    python scripts/convert_npy.py activations/fineweb/pythia-14m-deduped --to_format cov_svd
    python scripts/convert_npy.py activations/fineweb --recursive True
    python scripts/convert_npy.py activations/fineweb --output_dir inferences/fineweb --recursive True
"""

import os, re, sys
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_STEP_RE = re.compile(r'step(\d+)\.npy$')


def convert_npy_file(npy_path: str, to_format: str = "acts", hook_name: str = "after_final_norm",
                     storage_dtype: str = "fp32", output_dir: str = None):
    """Convert a single .npy activation file to .pt format.

    Args:
        storage_dtype: dtype for the output tensors. Defaults to fp32 to preserve
                       the original float32 precision of .npy files.
        output_dir: If set, write .pt files here (preserving subdirectory structure
                    relative to the input_dir). If None, write next to the .npy file.

    Returns the output .pt path.
    """
    from utils.accessor import save_factors

    acts = np.load(npy_path)
    acts_tensor = torch.from_numpy(acts)

    factors = {
        hook_name: {
            "A": acts_tensor,
            "n_A": acts_tensor.shape[0],
            "n": acts_tensor.shape[0],
        }
    }

    if output_dir is not None:
        out_path = os.path.join(output_dir, os.path.basename(npy_path).replace('.npy', '.pt'))
    else:
        out_path = npy_path.replace('.npy', '.pt') # let's just assume checkpoint name format is stepn.npy
    save_factors(factors, out_path, storage_format=to_format, storage_dtype=storage_dtype)
    return out_path


def main(input_dir: str, to_format: str = "acts", hook_name: str = "after_final_norm",
         storage_dtype: str = "fp32", recursive: bool = False, delete_npy: bool = False,
         output_dir: str = None):
    """Convert .npy files in input_dir to .pt format.

    Args:
        input_dir: Directory containing .npy files, or parent directory if recursive.
        to_format: Target storage format (acts, cov, cov_svd, eigenvalues).
        hook_name: Hook point name for the converted data.
        recursive: Search subdirectories too.
        delete_npy: Delete .npy files after successful conversion.
        output_dir: Write .pt files here instead of next to the .npy files.
                    Subdirectory structure relative to input_dir is preserved.
    """
    npy_files = []
    if recursive:
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                if _STEP_RE.match(f):
                    npy_files.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            if _STEP_RE.match(f):
                npy_files.append(os.path.join(input_dir, f))

    if not npy_files:
        print(f"No .npy step files found in {input_dir}")
        return

    print(f"Converting {len(npy_files)} .npy files to {to_format} format...")

    for path in sorted(npy_files):
        # Determine output path
        if output_dir is not None:
            rel = os.path.relpath(path, input_dir)
            out_path = os.path.join(output_dir, rel.replace('.npy', '.pt'))
        else:
            out_path = path.replace('.npy', '.pt')

        if os.path.exists(out_path):
            print(f"  SKIP {path} (.pt already exists at {out_path})")
            continue
        out = convert_npy_file(path, to_format=to_format, hook_name=hook_name,
                               storage_dtype=storage_dtype, output_dir=os.path.dirname(out_path))
        size_npy = os.path.getsize(path) / (1024 * 1024)
        size_pt = os.path.getsize(out) / (1024 * 1024)
        print(f"  {path} ({size_npy:.1f} MB) -> {out} ({size_pt:.1f} MB)")
        if delete_npy:
            os.remove(path)

    print("Done.")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
