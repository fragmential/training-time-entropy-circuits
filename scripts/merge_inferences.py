#!/usr/bin/env python
"""Merge two inference .pt collections (disjoint leaves, same model/data) into one.

Use to fold an add-on run into a base run without recollecting — e.g. a
`mlp.raw_out`-only run into a `block_representations` run. Per matching checkpoint
file the per-leaf entries are unioned. Refuses to merge files that cover DIFFERENT
data (mismatched __revision__ / __n_chunks__ / __token_filter__), since covariances
accumulated over different tokens are not comparable.

    python scripts/merge_inferences.py data/inferences/block_representations \
        data/inferences/block_mlp_raw_out --output-dir data/inferences/block_repr_merged
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_COMPAT = ("__revision__", "__n_chunks__", "__token_filter__")


def merge_inference(base, add):
    """Union add into base. Disjoint leaves are folded in whole; a leaf present in
    both has its entry merged key-by-key (add's new keys added, base's kept on
    conflict). base metadata wins; add-only metadata is folded in. Raises on
    incompatible data so covariances over different tokens are never mixed."""
    for k in _COMPAT:
        if k in base and k in add and base[k] != add[k]:
            raise ValueError(f"incompatible {k}: {base[k]!r} vs {add[k]!r} — different data, refusing to merge")
    out = {k: (dict(v) if isinstance(v, dict) and not k.startswith("__") else v)
           for k, v in base.items()}
    for leaf, entry in add.items():
        if leaf.startswith("__") or not isinstance(entry, dict):
            out.setdefault(leaf, entry)            # base metadata wins; add-only folded in
        elif leaf not in out:
            out[leaf] = dict(entry)                # new leaf
        else:
            for kk, vv in entry.items():           # same leaf: merge keys, base wins conflicts
                if kk in out[leaf]:
                    print(f"  note: {leaf!r} key {kk!r} in both; keeping base copy")
                else:
                    out[leaf][kk] = vv
    return out


def merge_files(base_path, add_path, out_path):
    import torch
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    add = torch.load(add_path, map_location="cpu", weights_only=False)
    torch.save(merge_inference(base, add), out_path)
    return out_path


def _rel_pts(root):
    if root.endswith(".pt"):
        return {os.path.basename(root): root}
    return {os.path.relpath(p, root): p
            for p in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)}


if __name__ == "__main__":
    import shutil
    p = argparse.ArgumentParser(prog="python scripts/merge_inferences.py")
    p.add_argument("base", help="base .pt file or directory")
    p.add_argument("add", help="add-on .pt file or directory (its extra leaves are folded in)")
    p.add_argument("--output-dir", dest="output_dir", help="write merged tree here (mirrors names)")
    p.add_argument("--output", help="output path (single-file inputs only)")
    args = p.parse_args()

    base_pts, add_pts = _rel_pts(args.base), _rel_pts(args.add)
    single = args.base.endswith(".pt") and args.add.endswith(".pt")

    for rel in sorted(set(base_pts) | set(add_pts)):
        if args.output_dir:
            out = os.path.join(args.output_dir, rel)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        elif args.output and single:
            out = args.output
        else:
            raise SystemExit("pass --output-dir (or --output for single-file inputs)")

        if rel in base_pts and rel in add_pts:
            merge_files(base_pts[rel], add_pts[rel], out)
            print(f"merged {rel}")
        else:  # present in only one input — copy through unchanged
            src = base_pts.get(rel) or add_pts[rel]
            if os.path.abspath(src) != os.path.abspath(out):
                shutil.copyfile(src, out)
            print(f"copied {rel} (only in {'base' if rel in base_pts else 'add'})")
