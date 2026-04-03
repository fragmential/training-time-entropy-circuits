#!/usr/bin/env python
"""Add trace and avg_magnitude to existing results .npy from source .pt covariances.

No eigendecomposition — uses diagonal sums for A/G, matmul trace for B.
"""
import os, sys, re, torch, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concurrent.futures import ThreadPoolExecutor
from utils.model_registry import (
    get_model_config, load_selective_weights, get_checkpoint_schedule,
    prefetch_checkpoint, delete_cached_revision,
)

results_path = sys.argv[1] if len(sys.argv) > 1 else "results/kfac_small/results_OLMo-2-0425-1B.npy"
source_dir = sys.argv[2] if len(sys.argv) > 2 else "inferences/kfac_small/OLMo-2-0425-1B"
keep_cached = "--keep-cached" in sys.argv

res = np.load(results_path, allow_pickle=True).item()

# Check if any hook has B — need weights only if so
sample_step = next(iter(res.values()))
need_weights = any("B" in factors for factors in sample_step.values())

config, schedule = None, {}
if need_weights:
    # Detect model from source data
    sample_pt = torch.load(os.path.join(source_dir, f"step{min(res.keys())}.pt"),
                           map_location="cpu", weights_only=False)
    model_name = sample_pt.get("__hf_model__", "allenai/OLMo-2-0425-1B")
    config = get_model_config(model_name)
    schedule = {s: (r, m) for s, r, m in get_checkpoint_schedule(config, None)}

sorted_steps = sorted(res.keys())
executor = ThreadPoolExecutor(max_workers=1) if need_weights else None

for idx, step in enumerate(sorted_steps):
    pt_path = os.path.join(source_dir, f"step{step}.pt")
    if not os.path.exists(pt_path):
        print(f"  Skip step {step}: no source .pt")
        continue

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    rev = data.get("__revision__")
    hf_model = data.get("__hf_model__")

    # Prefetch next checkpoint
    pf = None
    if executor and idx + 1 < len(sorted_steps):
        ns = sorted_steps[idx + 1]
        if ns in schedule:
            nr, nm = schedule[ns]
            pf = executor.submit(prefetch_checkpoint, nm, nr)

    # Load weights for B
    weights = None
    blk_hooks = [h for h in res[step] if re.match(r"blk\d+\.(up|down|gate)", h)]
    if need_weights and blk_hooks and rev:
        weights = load_selective_weights(
            config, hf_model or config.hf_repo, rev, blk_hooks, False)

    for hook_name, factors in res[step].items():
        entry = data.get(hook_name, {})

        for fk in list(factors.keys()):
            if fk in ("kfac", "gen_GB"):
                continue

            trace = None
            if fk == "A" and "A" in entry:
                n = entry.get("n_A", entry.get("n", 1))
                trace = float(entry["A"].diagonal().sum() / n)
                d = entry["A"].shape[0]
            elif fk == "A_centered" and "A" in entry and "A_mean" in entry:
                n = entry.get("n_A", entry.get("n", 1))
                mu = entry["A_mean"].float()
                trace = float(entry["A"].diagonal().sum() / n - mu.dot(mu))
                d = entry["A"].shape[0]
            elif fk == "G" and "G" in entry:
                n = entry.get("n_G", entry.get("n", 1))
                trace = float(entry["G"].diagonal().sum() / n)
                d = entry["G"].shape[0]
            elif fk == "B" and weights is not None and hook_name in weights:
                W = weights[hook_name].weight.float()
                n = entry.get("n_A", entry.get("n", 1))
                cov_A = entry["A"].float() / n
                trace = float((W @ cov_A).mul_(W).sum())
                d = W.shape[0]

            if trace is not None:
                factors[fk]["trace"] = trace
                factors[fk]["avg_magnitude"] = trace / d

    print(f"  Step {step}: done")

    if pf is not None:
        pf.result()
    if not keep_cached and rev and hf_model:
        delete_cached_revision(hf_model, rev)

if executor:
    executor.shutdown()
np.save(results_path, res)
print(f"Saved {len(res)} steps to {results_path}")
