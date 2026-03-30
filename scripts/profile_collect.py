#!/usr/bin/env python
"""Profile collect.py memory and timing for a given config.

Wraps the collection pipeline with CUDA memory tracking and wall-clock timing.
Prints peak VRAM, per-phase breakdown, and total time.

Usage:
    python scripts/profile_collect.py --config configs/debug.yaml
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from scripts.collect import CollectConfig, main as collect_main

def fmt_gb(b):
    return f"{b / 1e9:.2f} GB"

def fmt_mb(b):
    return f"{b / 1e6:.1f} MB"

if __name__ == "__main__":
    from jsonargparse import CLI

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available. Run on a GPU node.")
        sys.exit(1)

    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {fmt_gb(torch.cuda.get_device_properties(0).total_memory)}")
    print()

    baseline = torch.cuda.memory_allocated()
    print(f"Baseline allocated: {fmt_mb(baseline)}")

    t0 = time.time()
    cfg = CLI(CollectConfig, args=sys.argv[1:])
    t_parse = time.time()
    print(f"Config parsed in {t_parse - t0:.1f}s")

    # Run collection
    collect_main(cfg)
    t_end = time.time()

    # Report
    peak = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    current = torch.cuda.memory_allocated()

    print()
    print("=" * 60)
    print("MEMORY PROFILE")
    print("=" * 60)
    print(f"  Peak allocated:  {fmt_gb(peak)}")
    print(f"  Peak reserved:   {fmt_gb(peak_reserved)}")
    print(f"  Current alloc:   {fmt_mb(current)}")
    print(f"  Baseline alloc:  {fmt_mb(baseline)}")
    print()
    print("TIMING")
    print("=" * 60)
    print(f"  Total wall time: {t_end - t0:.1f}s")
    print(f"  Collection time: {t_end - t_parse:.1f}s")
    print()

    # Also dump CUDA memory summary
    print("CUDA MEMORY SUMMARY")
    print("=" * 60)
    print(torch.cuda.memory_summary(abbreviated=True))
