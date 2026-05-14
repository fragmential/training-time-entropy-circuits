#!/usr/bin/env python
"""Map changed source files to relevant test files.

Usage:
    python tests/testmap.py utils/accessor.py scripts/compute_metrics.py
    # prints: tests/test_powerlaw.py tests/test_storage.py tests/test_accessor.py tests/test_compute_metrics.py tests/test_snapshots.py

    # Pipe to pytest:
    uv run pytest $(python tests/testmap.py $(git diff --name-only)) -v
"""
import sys

# source file → test files that exercise it
_MAP = {
    "utils/accessor.py": [
        "tests/test_powerlaw.py",      # eigendecomp functions moved here
        "tests/test_storage.py",       # save/convert/info moved here
        "tests/test_accessor.py",
        "tests/test_compute_metrics.py",
        "tests/test_snapshots.py",
    ],
    "utils/data_utils.py": [
        "tests/test_data_utils.py",
        "tests/test_snapshots.py",
    ],
    "utils/hooks.py": [
        "tests/test_hooks.py",
        "tests/test_snapshots.py",
    ],
    "scripts/compute_metrics.py": [
        "tests/test_powerlaw.py",      # metric functions moved here
        "tests/test_compute_metrics.py",
        "tests/test_snapshots.py",
    ],
    "utils/model_registry.py": [
        "tests/test_e2e.py",
    ],
    "utils/checkpoint_info.py": [
        "tests/test_e2e.py",
    ],
    "scripts/collect.py": [
        "tests/test_e2e.py",
    ],
}

def get_tests(changed_files):
    tests = set()
    for f in changed_files:
        # Normalize path
        f = f.strip().lstrip("./")
        if f in _MAP:
            tests.update(_MAP[f])
        # Also match by prefix for data/ loaders
        for key in _MAP:
            if f.startswith(key.rsplit("/", 1)[0] + "/") and key in _MAP:
                tests.update(_MAP[key])
                break
    return sorted(tests)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/testmap.py <file1> [file2] ...")
        sys.exit(1)
    tests = get_tests(sys.argv[1:])
    if tests:
        print(" ".join(tests))
    else:
        print("tests/test_snapshots.py")  # fallback: always run snapshots
