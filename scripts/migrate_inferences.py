#!/usr/bin/env python
"""Migrate inference .pt from the OLDEST letter-key era straight to the current schema.

    {flat_hook: {A | A_eigvals | A_eigvecs | A_mean | n_A | B* | G* | O* | ...}}  (+ __meta__)
 -> {leaf_path: {acts_cov|acts_samples | acts_eigvals | acts_mean | acts_n | grads_* | ...}}

Bare `A` is read by shape: square -> {q}_cov (Σxxᵀ), else -> {q}_samples. __format__
and other __meta__ keys are preserved. Writes to --output-dir / --output; never in place.

    python scripts/migrate_inferences.py old_step0.pt --output new_step0.pt
"""
import os
import re
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_MLP = re.compile(r"blk(\d+)\.(up|down|gate)$")
_OV = re.compile(r"blk(\d+)\.attn\.head(\d+)$")
_BOUNDARY = re.compile(r"blk(\d+)\.(attn|mlp)\.(in|out|raw_out)$")
_RESID = ("before_final_norm", "after_final_norm", "identity_head")


def _target(hook, letter):
    """(new leaf, quantity) for an old (hook, letter), or None to drop."""
    m = _MLP.match(hook)
    if m:
        i, p = m.group(1), m.group(2)
        return {"A": (f"blk{i}.mlp.{p}.in", "acts"),
                "B": (f"blk{i}.mlp.{p}.out", "acts"),
                "G": (f"blk{i}.mlp.{p}.out", "grads")}.get(letter)
    m = _OV.match(hook)
    if m:
        i, h = m.group(1), m.group(2)
        return {"A": (f"blk{i}.attn.head{h}.slice", "acts"),
                "G": (f"blk{i}.attn.head{h}.slice", "grads"),
                "O": (f"blk{i}.attn.head{h}.contrib", "acts")}.get(letter)
    if _BOUNDARY.match(hook) or hook in _RESID:
        return {"A": (hook, "acts"), "G": (hook, "grads")}.get(letter)
    return None  # blk{i}.layer is metric-only; never stored as inference data


def _is_square(t):
    return hasattr(t, "dim") and t.dim() == 2 and t.shape[0] == t.shape[1]


def _route(hook, key, val):
    """(leaf, new_key) for one old storage key, or None to drop."""
    if key == "n":
        return None  # bare count dropped; each quantity now has its own {q}_n
    if key.startswith("n_"):
        t = _target(hook, key[2:])
        return (t[0], f"{t[1]}_n") if t else None
    for L in ("A", "B", "G", "O"):
        if key == L or key.startswith(L + "_"):
            t = _target(hook, L)
            if t is None:
                return None
            leaf, q = t
            rest = key[len(L):]
            if rest == "":  # bare letter: the primary tensor (cov or raw samples)
                return leaf, f"{q}_cov" if _is_square(val) else f"{q}_samples"
            return leaf, f"{q}{rest}"  # _eigvals/_eigvecs/_eigvals_centered/_mean/_U/_S/_V/_mask/_cross_eigvals_*
    return None


def migrate_inference(data):
    out = {k: v for k, v in data.items() if k.startswith("__")}
    for hook, entry in data.items():
        if hook.startswith("__") or not isinstance(entry, dict):
            continue
        for key, val in entry.items():
            r = _route(hook, key, val)
            if r:
                out.setdefault(r[0], {})[r[1]] = val
    return out


def migrate_file(in_path, out_path, dry_run=False):
    import torch
    migrated = migrate_inference(torch.load(in_path, map_location="cpu", weights_only=False))
    if not dry_run:
        torch.save(migrated, out_path)
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="python scripts/migrate_inferences.py")
    p.add_argument("input", help=".pt file or directory")
    p.add_argument("--output-dir", dest="output_dir", help="mirror input filenames here")
    p.add_argument("--output", help="output path (single-file input only)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = ([args.input] if args.input.endswith(".pt")
             else sorted(glob.glob(os.path.join(args.input, "**", "*.pt"), recursive=True)))
    root = args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
    for f in files:
        if args.output_dir:
            out = os.path.join(args.output_dir, os.path.relpath(f, root))
            os.makedirs(os.path.dirname(out), exist_ok=True)
        elif args.output and len(files) == 1:
            out = args.output
        else:
            raise SystemExit("refusing in-place migration; pass --output-dir (or --output for one file)")
        migrate_file(f, out, dry_run=args.dry_run)
        print(f"{'would migrate' if args.dry_run else 'migrated'} {f} -> {out}")
