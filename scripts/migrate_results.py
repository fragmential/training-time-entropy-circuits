#!/usr/bin/env python
"""Migrate result .npy from the OLDEST letter-key era straight to the current schema.

    {step: {flat_hook: {A/B/G/O[/_centered/_mean_metrics/_cross_*], gen_GB, kfac}}}
 -> {step: {node_path:  {acts_uncentered/.../gen/kfac/projections_kfac}}}

Reads the letter era only (e.g. data/results_backup_premigration); writes to
--output-dir (mirrors names) or --output (single file); never edits in place.

    python scripts/migrate_results.py data/results_backup_premigration --output-dir data/results
"""
import os
import re
import sys
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_MLP = re.compile(r"blk(\d+)\.(up|down|gate)$")
_LAYER = re.compile(r"blk(\d+)\.layer$")
_OV = re.compile(r"blk(\d+)\.attn\.head(\d+)$")
_BOUNDARY = re.compile(r"blk(\d+)\.(attn|mlp)\.(in|out|raw_out)$")
_RESID = ("before_final_norm", "after_final_norm", "identity_head")

_KFAC_SUB = {"trace_A": "trace_acts", "trace_G": "trace_grads",
             "log_det_A": "log_det_acts", "log_det_G": "log_det_grads",
             "d_A": "d_acts", "d_G": "d_grads"}


def _target(hook, letter):
    """(new leaf/node, quantity) for an old (hook, letter), or None to drop."""
    m = _MLP.match(hook)
    if m:
        i, p = m.group(1), m.group(2)
        return {"A": (f"blk{i}.mlp.{p}.in", "acts"),
                "B": (f"blk{i}.mlp.{p}.out", "acts"),
                "G": (f"blk{i}.mlp.{p}.out", "grads")}.get(letter)
    m = _LAYER.match(hook)
    if m:
        i = m.group(1)
        return {"A": (f"blk{i}.mlp.up.in", "acts"),
                "B": (f"blk{i}.mlp.down.out", "acts"),
                "G": (f"blk{i}.mlp.down.out", "grads")}.get(letter)
    m = _OV.match(hook)
    if m:
        i, h = m.group(1), m.group(2)
        return {"A": (f"blk{i}.attn.head{h}.slice", "acts"),
                "G": (f"blk{i}.attn.head{h}.slice", "grads"),
                "O": (f"blk{i}.attn.head{h}.contrib", "acts")}.get(letter)
    if _BOUNDARY.match(hook) or hook in _RESID:
        return {"A": (hook, "acts"), "G": (hook, "grads")}.get(letter)
    return None


def _remap_kfac(val):
    return {_KFAC_SUB.get(k, k): v for k, v in val.items()} if isinstance(val, dict) else val


def _result_targets(hook, key, val):
    """[(node, metric, value), ...] for one old result key (usually one)."""
    if key == "kfac":
        m = _MLP.match(hook)
        if m:
            return [(f"blk{m.group(1)}.mlp.{m.group(2)}", "kfac", _remap_kfac(val))]
        m = _LAYER.match(hook)
        if m:
            return [(f"blk{m.group(1)}.mlp", "projections_kfac", _remap_kfac(val))]
        return []  # ov/boundary/residual same-point kfac was dropped in the redesign
    if key in ("gen_GB", "gen_GO", "gen"):
        m = _MLP.match(hook)
        if m:
            return [(f"blk{m.group(1)}.mlp.{m.group(2)}.out", "gen", val)]
        m = _OV.match(hook)
        if m:
            return [(f"blk{m.group(1)}.attn.head{m.group(2)}.slice", "gen", val)]
        if _BOUNDARY.match(hook) or hook in _RESID:
            return [(hook, "gen", val)]
        return []
    for L in ("A", "B", "G", "O"):
        if key == L or key.startswith(L + "_"):
            t = _target(hook, L)
            if t is None:
                return []
            node, q = t
            suffix = key[len(L):]  # "", "_centered", "_mean_metrics", "_cross_<label>"
            return [(node, f"{q}_uncentered" if suffix == "" else f"{q}{suffix}", val)]
    return []


def migrate_results(results):
    out = {}
    for step, step_metrics in results.items():
        ns = {}
        for hook, hm in step_metrics.items():
            if not isinstance(hm, dict):
                ns[hook] = hm  # e.g. __gpu_wait__
                continue
            for key, val in hm.items():
                for node, metric, v in _result_targets(hook, key, val):
                    ns.setdefault(node, {})[metric] = v
        out[step] = ns
    return out


def migrate_file(in_path, out_path, dry_run=False):
    import numpy as np
    migrated = migrate_results(np.load(in_path, allow_pickle=True).item())
    if not dry_run:
        np.save(out_path, migrated)
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="python scripts/migrate_results.py")
    p.add_argument("input", help=".npy file or directory")
    p.add_argument("--output-dir", dest="output_dir", help="mirror input filenames here")
    p.add_argument("--output", help="output path (single-file input only)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = ([args.input] if args.input.endswith(".npy")
             else sorted(glob.glob(os.path.join(args.input, "**", "*.npy"), recursive=True)))
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
