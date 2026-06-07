#!/usr/bin/env python
"""Merge two results .npy collections (disjoint leaves, same model/data) into one.

The results-space twin of merge_inferences.py. Use to fold an add-on metrics run
into a base run without recomputing — e.g. a `mlp.raw_out`-only run into a
`block_representations` run. A results file is {step: {leaf: {metric: value}}};
per matching step the per-leaf entries are unioned (add's new leaves added,
base's kept on conflict). Steps present in only one input are folded in whole.

Verified equivalent to recomputing metrics on a merged-inference checkpoint:
each leaf's metrics are computed identically regardless of which run produced it,
so unioning leaves per step reproduces the same leaf/step coverage. Results .npy
files carry no data-provenance metadata, so the caller is responsible for only
merging runs over the same model/data (e.g. ones whose inferences were mergeable).

    python scripts/merge_results.py data/results/block_representations \
        data/results/block_mlp_raw_out --output-dir data/results/block_repr_merged
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def merge_results(base, add):
    """Union add into base. {step: {leaf: metrics}}. A step in only one input is
    folded in whole; a step in both has its leaf dicts merged (add's new leaves
    added, base's kept on conflict)."""
    out = {step: dict(leaves) for step, leaves in base.items()}
    for step, leaves in add.items():
        if step not in out:
            out[step] = dict(leaves)               # new step
        else:
            for leaf, metrics in leaves.items():   # same step: merge leaves, base wins conflicts
                if leaf in out[step]:
                    print(f"  note: step {step} leaf {leaf!r} in both; keeping base copy")
                else:
                    out[step][leaf] = metrics
    return out


def merge_files(base_path, add_path, out_path):
    import numpy as np
    base = np.load(base_path, allow_pickle=True).item()
    add = np.load(add_path, allow_pickle=True).item()
    np.save(out_path, merge_results(base, add))
    return out_path


def _rel_npys(root):
    if root.endswith(".npy"):
        return {os.path.basename(root): root}
    return {os.path.relpath(p, root): p
            for p in glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)}


if __name__ == "__main__":
    import shutil
    p = argparse.ArgumentParser(prog="python scripts/merge_results.py")
    p.add_argument("base", help="base .npy file or directory")
    p.add_argument("add", help="add-on .npy file or directory (its extra leaves are folded in)")
    p.add_argument("--output-dir", dest="output_dir", help="write merged tree here (mirrors names)")
    p.add_argument("--output", help="output path (single-file inputs only)")
    args = p.parse_args()

    base_npys, add_npys = _rel_npys(args.base), _rel_npys(args.add)
    single = args.base.endswith(".npy") and args.add.endswith(".npy")

    for rel in sorted(set(base_npys) | set(add_npys)):
        if args.output_dir:
            out = os.path.join(args.output_dir, rel)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        elif args.output and single:
            out = args.output
        else:
            raise SystemExit("pass --output-dir (or --output for single-file inputs)")

        if rel in base_npys and rel in add_npys:
            merge_files(base_npys[rel], add_npys[rel], out)
            print(f"merged {rel}")
        else:  # present in only one input — copy through unchanged
            src = base_npys.get(rel) or add_npys[rel]
            if os.path.abspath(src) != os.path.abspath(out):
                shutil.copyfile(src, out)
            print(f"copied {rel} (only in {'base' if rel in base_npys else 'add'})")
