#!/usr/bin/env python
"""Estimate per-layer Hessian trace via Hutchinson and append to results .npy.

Usage:
    python scripts/hessian_trace.py \
        --results_path results/kfac_small/results_OLMo-2-0425-1B.npy \
        --source_dir inferences/kfac_small/OLMo-2-0425-1B \
        --model_name allenai/OLMo-2-0425-1B \
        --dataset_name olmo_mix \
        --hooks all \
        --n_batches 24 --batch_size 24 --n_draws 6 --seq_len 512
"""
import os, sys, re, argparse, torch, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from concurrent.futures import ThreadPoolExecutor
from utils.hessian_trace import hutchinson_trace
from utils.model_registry import (
    get_model_config, load_model, load_tokenizer, get_mlp_projections,
    get_num_layers, get_checkpoint_schedule,
    prefetch_checkpoint, delete_cached_revision,
)
from utils.data_utils import get_loader, load_and_cache_texts, pack_sequences


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_path", required=True)
    p.add_argument("--source_dir", required=True,
                   help="Dir with per-step .pt files (to read revisions)")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset_name", default="fineweb")
    p.add_argument("--hooks", default="all",
                   help="Comma-separated hook names or 'all'")
    p.add_argument("--n_batches", type=int, default=24)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--n_draws", type=int, default=6)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--max_bytes", type=int, default=None)
    p.add_argument("--keep-cached", action="store_true")
    return p.parse_args()


def resolve_hooks(res, hooks_arg):
    """Return list of hook names from results keys."""
    sample = next(iter(res.values()))
    all_hooks = [h for h in sample if re.match(r"blk\d+\.(up|down|gate)", h)]
    if hooks_arg == "all":
        return sorted(all_hooks)
    requested = [h.strip() for h in hooks_arg.split(",")]
    return [h for h in requested if h in all_hooks]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load results
    res = np.load(args.results_path, allow_pickle=True).item()
    hook_names = resolve_hooks(res, args.hooks)
    print(f"Target hooks: {hook_names}")

    # Prepare data
    config = get_model_config(args.model_name)
    tokenizer = load_tokenizer(config)
    loader_fn = get_loader(args.dataset_name)
    texts = load_and_cache_texts(
        loader_fn, args.num_samples, min_length=64, tokenizer=tokenizer,
        dataset_name=args.dataset_name, max_bytes=args.max_bytes,
    )
    packed_ids = pack_sequences(texts, tokenizer, args.seq_len)
    print(f"Packed data: {packed_ids.shape[0]} chunks of {args.seq_len} tokens")

    n_total = args.n_batches * args.batch_size
    if packed_ids.shape[0] < n_total:
        print(f"Warning: only {packed_ids.shape[0]} chunks available, "
              f"need {n_total} for {args.n_batches} batches of {args.batch_size}")

    # Checkpoint schedule for prefetch/cleanup
    schedule = {s: (r, m) for s, r, m in get_checkpoint_schedule(config, None)}
    sorted_steps = sorted(res.keys())
    executor = ThreadPoolExecutor(max_workers=1)

    for idx, step in enumerate(sorted_steps):
        # Check if already computed
        sample_hook = hook_names[0] if hook_names else None
        if sample_hook and "hutchinson_trace" in res[step].get(sample_hook, {}):
            print(f"  Step {step}: already has hutchinson_trace, skipping")
            continue

        # Get revision from source .pt
        pt_path = os.path.join(args.source_dir, f"step{step}.pt")
        if not os.path.exists(pt_path):
            print(f"  Step {step}: no source .pt, skipping")
            continue
        src = torch.load(pt_path, map_location="cpu", weights_only=False)
        rev = src.get("__revision__")
        hf_model = src.get("__hf_model__", args.model_name)
        del src

        # Prefetch next checkpoint
        pf = None
        if idx + 1 < len(sorted_steps):
            ns = sorted_steps[idx + 1]
            if ns in schedule:
                nr, nm = schedule[ns]
                pf = executor.submit(prefetch_checkpoint, nm, nr)

        # Load model
        print(f"  Step {step}: loading model (rev={rev})...")
        model = load_model(config, hf_model, rev)
        model.to(device).train()
        for p in model.parameters():
            p.requires_grad_(False)

        # Resolve hooks to layers
        layers = {}
        for hook_name in hook_names:
            m = re.match(r"blk(\d+)\.(up|down|gate)", hook_name)
            if not m:
                continue
            block_idx = int(m.group(1))
            for name, layer in get_mlp_projections(model, config, block_idx):
                if name == hook_name:
                    layers[hook_name] = layer
                    break

        # Sample batches (seeded for reproducibility)
        rng = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(packed_ids.shape[0], generator=rng)[:n_total]
        batches = packed_ids[indices].view(args.n_batches, args.batch_size, -1)

        # Hutchinson estimation
        gpu_gen = torch.Generator(device=device).manual_seed(args.seed)
        for hook_name, layer in layers.items():
            params = [layer.weight]
            if layer.bias is not None:
                params.append(layer.bias)

            for p in params:
                p.requires_grad_(True)

            accum = 0.0
            n_est = 0
            for bi in range(args.n_batches):
                batch = batches[bi].to(device)
                tr = hutchinson_trace(model, batch, params, args.n_draws, gpu_gen)
                accum += tr * args.n_draws  # weighted by n_draws for correct averaging
                n_est += args.n_draws

            for p in params:
                p.requires_grad_(False)

            mean_trace = accum / n_est
            res[step].setdefault(hook_name, {})
            res[step][hook_name]["hutchinson_trace"] = mean_trace
            res[step][hook_name]["hutchinson_n_estimates"] = n_est
            print(f"    {hook_name}: tr(H)={mean_trace:.4e} ({n_est} estimates)")

        # Save after each step (crash-safe)
        np.save(args.results_path, res)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        if pf is not None:
            pf.result()
        if not args.keep_cached and rev:
            delete_cached_revision(hf_model, rev)

        print(f"  Step {step}: done")

    executor.shutdown()
    print(f"Saved {len(res)} steps to {args.results_path}")


if __name__ == "__main__":
    main()
