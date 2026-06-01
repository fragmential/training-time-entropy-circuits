#!/usr/bin/env python
"""Migrate result .npy files from letter keys (A/B/G/O/gen_GB/kfac _A/_G) to the
role.quantity schema. Pure key rename, metric values untouched.

    python scripts/migrate_results.py data/results            # all .npy under dir
    python scripts/migrate_results.py path/to/results.npy --dry-run
"""
import os
import sys
import glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import hook_names as hn

# Legacy letter -> node.signal, by hook kind. This script reads OLD letter-keyed
# result files; the live pipeline no longer uses letters anywhere, so the table
# lives here (its only remaining home).
_ROLE_KEY = {
    "mlp":      {"A": "in.acts",    "B": "out.acts",   "G": "out.grads"},
    "layer":    {"A": "in.acts",    "G": "out.grads"},
    "residual": {"A": "value.acts", "G": "value.grads"},
    "boundary": {"A": "value.acts", "G": "value.grads"},
    "ov_head":  {"A": "slice.acts", "G": "slice.grads", "O": "contrib.acts"},
}
_LETTERS = ("A", "B", "G", "O")
_SPECIAL = {"gen_GB": "out.gen", "gen_GO": "slice.gen"}
_KFAC_SUB = {"trace_A": "trace_acts", "trace_G": "trace_grads",
             "log_det_A": "log_det_acts", "log_det_G": "log_det_grads",
             "d_A": "d_acts", "d_G": "d_grads"}


def _remap_key(kind, key):
    if key in _SPECIAL:
        return _SPECIAL[key]
    roles = _ROLE_KEY.get(kind, {})
    for letter in _LETTERS:
        if key == letter:
            return roles.get(letter, key)
        if key.startswith(letter + "_"):
            return roles.get(letter, letter) + key[len(letter):]
    return key


def _migrate_hook(kind, hook_metrics):
    out = {}
    for key, val in hook_metrics.items():
        if key == "kfac" and isinstance(val, dict):
            val = {_KFAC_SUB.get(k, k): v for k, v in val.items()}
        out[_remap_key(kind, key)] = val
    return out


def migrate_results(results):
    out = {}
    for step, step_metrics in results.items():
        new_step = {}
        for hook, hook_metrics in step_metrics.items():
            if isinstance(hook_metrics, dict):
                new_step[hook] = _migrate_hook(hn.classify(hook), hook_metrics)
            else:
                new_step[hook] = hook_metrics  # e.g. __gpu_wait__
        out[step] = new_step
    return out


def migrate_file(path, dry_run=False):
    results = np.load(path, allow_pickle=True).item()
    migrated = migrate_results(results)
    if not dry_run:
        np.save(path, migrated)
    return path


def _npy_files(path):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "**", "*.npy"), recursive=True))
    return [path]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path", help=".npy file or directory")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for f in _npy_files(args.path):
        migrate_file(f, dry_run=args.dry_run)
        print(f"{'would migrate' if args.dry_run else 'migrated'} {f}")
