#!/usr/bin/env python
"""Stamp .pt files with __revision__ and __hf_model__ from checkpoint schedule.

Usage: python scripts/stamp_revisions.py inferences/kfac_small          # whole config
       python scripts/stamp_revisions.py inferences/kfac_small/OLMo-2-0425-1B  # one model
       python scripts/stamp_revisions.py inferences/kfac_small/OLMo-2-0425-1B/step1000.pt  # one file
"""
import os, sys, re, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_registry import get_model_config, get_checkpoint_schedule
from utils.accessor import _resolve_hf_name

_STEP = re.compile(r"step(\d+)\.pt$")

def process(path):
    if path.endswith(".pt"):
        return _stamp_dir(os.path.dirname(path), only={path})
    entries = os.listdir(path)
    if any(_STEP.match(e) for e in entries):
        _stamp_dir(path)
    else:
        for e in sorted(entries):
            if os.path.isdir(p := os.path.join(path, e)):
                process(p)

def _stamp_dir(d, only=None):
    cfg = get_model_config(_resolve_hf_name(os.path.basename(d)))
    sched = {s: (r, m) for s, r, m in get_checkpoint_schedule(cfg, None)}
    print(f"{os.path.basename(d)}: {len(sched)} revisions in schedule")
    for f in only or sorted(os.path.join(d, e) for e in os.listdir(d)):
        m = _STEP.match(os.path.basename(f))
        if not m or (step := int(m.group(1))) not in sched:
            continue
        data = torch.load(f, map_location="cpu", weights_only=False)
        rev, hf = sched[step]
        if data.get("__revision__") == rev and data.get("__hf_model__") == hf:
            continue
        data["__revision__"], data["__hf_model__"] = rev, hf
        torch.save(data, f)
        print(f"  step{step}.pt: rev={rev}")

if __name__ == "__main__":
    process(sys.argv[1] if len(sys.argv) > 1 else ".")
