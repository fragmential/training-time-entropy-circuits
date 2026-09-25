"""Migrate current-schema inference .pt metadata to the latest conventions (metadata-only):
normalise `__format__` (strip `+m`/`+b`/`+o` modifiers; `eigvals`->`eigenvalues`) and inject
`__run__` from the path. Renames any `{q}_cov` -> `{q}_gram` (none on disk now; kept for safety).
Leaf names are NOT touched — pre-one-axis dirs (letter-era leaf naming, e.g. `kfac_small_shuffled_`)
are excluded; use scripts/migrate_inferences.py for those, or regenerate.

    python scripts/migrate_metadata.py --input data/inferences --dry-run
    python scripts/migrate_metadata.py --input data/inferences                  # in place
    python scripts/migrate_metadata.py --input data/inferences --output-dir <DIR>
"""
import argparse, glob, os
import torch

_FMT_ALIAS = {"eigvals": "eigenvalues", "activations": "acts", "covariance": "cov"}
_VALID = {"acts", "acts_svd", "cov", "cov_svd", "eigenvalues"}


def _norm_format(fmt: str | None) -> str | None:
    base = (fmt or "").split("+")[0]
    return _FMT_ALIAS.get(base, base) or None


def _migrate(data: dict, run: str) -> dict:
    out = {k: (data[k] if k.startswith("__") else
               {(kk[:-4] + "_gram" if kk.endswith("_cov") else kk): vv for kk, vv in data[k].items()})
           for k in data}
    out["__format__"] = _norm_format(data.get("__format__"))
    out["__run__"] = run
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="metadata-only inference migrator")
    p.add_argument("--input", required=True, help="inferences root (scans <run>/<model>/*.pt)")
    p.add_argument("--output-dir", default=None, help="mirror <run>/<model>/ here; default in-place")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--exclude", nargs="*", default=["kfac_small_shuffled_", "backups"])
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.input, "*", "*", "*.pt")))
    written = scanned = skipped = 0
    for f in files:
        rel = os.path.relpath(f, a.input)
        run = rel.split(os.sep)[0]
        if run in a.exclude:
            skipped += 1
            continue
        scanned += 1
        data = torch.load(f, map_location="cpu", weights_only=False)
        old_fmt = data.get("__format__")
        new = _migrate(data, run)
        new_fmt = new["__format__"]
        if new_fmt not in _VALID:
            print(f"!! {rel}: normalised format {new_fmt!r} not in {_VALID} — SKIPPED")
            continue
        changed = new_fmt != old_fmt or "__run__" not in data
        dst = f if a.output_dir is None else os.path.join(a.output_dir, rel)
        print(f"{'WOULD ' if a.dry_run else 'MIGRATE '}{'(noop) ' if not changed else ''}"
              f"{rel}: format {old_fmt!r}->{new_fmt!r}  run={run!r}")
        if not a.dry_run:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            torch.save(new, dst)
            written += 1
    print(f"\n{'dry-run: ' if a.dry_run else ''}{scanned} scanned, {skipped} excluded, {written} written"
          f"{f' to {a.output_dir}' if a.output_dir else ' in place'}")


if __name__ == "__main__":
    main()
