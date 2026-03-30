#!/usr/bin/env python
"""Sweep batch_size and max_layers_per_pass to find VRAM limits.

Loads each model once, then runs minimal forward+backward passes at different
batch sizes to measure peak GPU memory. ~1-2 minutes per model.

Usage:
    python scripts/profile_sweep.py                              # all models
    python scripts/profile_sweep.py --models pythia-1b olmo-1b   # subset
"""
import os, sys, gc, time, traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

# ── Model configs to sweep ──────────────────────────────────────────────
SWEEP = {
    "pythia-14m": {
        "hf_name": "EleutherAI/pythia-14m",
        "batch_sizes": [32, 64, 96, 128, 192, 256],
        "mlpp_values": [3, 6],
    },
    "pythia-1b": {
        "hf_name": "EleutherAI/pythia-1b-deduped",
        "batch_sizes": [4, 8, 16, 32, 48, 64],
        "mlpp_values": [2, 4, 8],
    },
    "pythia-7b": {
        "hf_name": "EleutherAI/pythia-6.9b-deduped",
        "batch_sizes": [1, 2, 4, 8, 12, 16],
        "mlpp_values": [1, 2, 4],
    },
    "olmo-1b": {
        "hf_name": "allenai/OLMo-2-0425-1B",
        "batch_sizes": [4, 8, 16, 32, 48, 64],
        "mlpp_values": [2, 4, 8],
    },
    "olmo-7b": {
        "hf_name": "allenai/OLMo-2-1124-7B",
        "batch_sizes": [1, 2, 4, 8, 12, 16],
        "mlpp_values": [1, 2, 4],
    },
}

SEQ_LEN = 512


def fmt_gib(b):
    return f"{b / 2**30:.2f} GiB"


def _run_forward_backward(model, model_config, batch_size, max_layers_per_pass, device):
    """Core forward+backward logic with hooks, matching collect.py's actual path."""
    from utils.hooks import HookCollector
    from utils.model_registry import get_num_layers, get_mlp_projections

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, SEQ_LEN), device=device)
    attention_mask = torch.ones_like(input_ids)

    num_layers = get_num_layers(model, model_config)
    if max_layers_per_pass == 0 or max_layers_per_pass >= num_layers:
        block_groups = [list(range(num_layers))]
    else:
        block_groups = [list(range(i, min(i + max_layers_per_pass, num_layers)))
                        for i in range(0, num_layers, max_layers_per_pass)]

    for blk_group in block_groups:
        collectors = {}

        # Freeze everything, unfreeze target MLP projections
        for p in model.parameters():
            p.requires_grad_(False)

        for b in blk_group:
            for name, layer in get_mlp_projections(model, model_config, b):
                layer.weight.requires_grad_(True)
                collectors[name] = HookCollector(
                    layer, capture="input", mode="cov",
                    collect_grad=True,
                    accumulation_dtype=torch.float64,
                    activation_dtype=torch.float32,
                )

        # Forward + sampled-label backward (matches collect.py)
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1))
        flat_logits = logits.reshape(-1, logits.size(-1))
        with torch.no_grad():
            y = torch.multinomial(probs, 1).squeeze(1)
        loss = torch.nn.functional.cross_entropy(flat_logits, y)
        loss.backward()

        # Cleanup
        for c in collectors.values():
            c.close()
        del collectors, logits, probs, flat_logits, y, loss

        for p in model.parameters():
            p.requires_grad_(False)
            if p.grad is not None:
                p.grad = None

    gc.collect()
    torch.cuda.empty_cache()


def profile_one_batch(model, model_config, batch_size, max_layers_per_pass, device):
    """Run one forward+backward with hooks and return peak VRAM bytes."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    _run_forward_backward(model, model_config, batch_size, max_layers_per_pass, device)

    return torch.cuda.max_memory_allocated()


def profile_detailed(model, model_config, batch_size, max_layers_per_pass, device):
    """Run with torch.profiler and print top ops by CUDA time and memory."""
    from torch.profiler import profile, ProfilerActivity

    # Warmup run (profiler needs it)
    _run_forward_backward(model, model_config, batch_size, max_layers_per_pass, device)
    gc.collect()
    torch.cuda.empty_cache()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        _run_forward_backward(model, model_config, batch_size, max_layers_per_pass, device)

    print("\n  --- Top 20 CUDA ops by GPU time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print("\n  --- Top 20 ops by GPU memory ---")
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--profile", action="store_true",
                        help="Run torch.profiler on first config of each model")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {fmt_gib(torch.cuda.get_device_properties(0).total_memory)}")
    print()

    models_to_test = args.models if args.models else list(SWEEP.keys())
    results = []

    for model_key in models_to_test:
        if model_key not in SWEEP:
            print(f"Unknown model key: {model_key}, skipping")
            continue
        spec = SWEEP[model_key]
        hf_name = spec["hf_name"]

        print(f"\n{'='*70}")
        print(f"MODEL: {model_key} ({hf_name})")
        print(f"{'='*70}")

        # Load model once
        from utils.model_registry import get_model_config, load_model
        mc = get_model_config(hf_name)
        t0 = time.time()
        model = load_model(mc, hf_name, revision=None)
        model.to(device)
        model.eval()
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s")

        model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"  Model VRAM: {fmt_gib(model_bytes)}")

        # Detailed profiler run (first mlpp, first batch size)
        if args.profile:
            first_mlpp = spec["mlpp_values"][0]
            first_bs = spec["batch_sizes"][0]
            print(f"\n  PROFILER: bs={first_bs}, mlpp={first_mlpp}")
            try:
                profile_detailed(model, mc, first_bs, first_mlpp, device)
            except Exception as e:
                print(f"  Profiler failed: {e}")
                traceback.print_exc()

        for mlpp in spec["mlpp_values"]:
            print(f"\n  max_layers_per_pass={mlpp}")
            print(f"  {'batch':>6}  {'peak VRAM':>12}  {'time':>8}  status")
            print(f"  {'-'*6}  {'-'*12}  {'-'*8}  {'-'*10}")

            for bs in spec["batch_sizes"]:
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    t0 = time.time()
                    peak = profile_one_batch(model, mc, bs, mlpp, device)
                    wall = time.time() - t0
                    peak_gib = peak / 2**30
                    print(f"  {bs:>6}  {peak_gib:>10.2f} GiB  {wall:>6.1f}s  OK")
                    results.append((model_key, bs, mlpp, peak_gib, wall, "OK"))
                except torch.cuda.OutOfMemoryError:
                    print(f"  {bs:>6}  {'—':>12}  {'—':>8}  OOM")
                    results.append((model_key, bs, mlpp, None, None, "OOM"))
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    msg = str(e)[:60]
                    print(f"  {bs:>6}  {'—':>12}  {'—':>8}  ERR: {msg}")
                    traceback.print_exc()
                    results.append((model_key, bs, mlpp, None, None, f"ERR: {msg}"))
                    gc.collect()
                    torch.cuda.empty_cache()
                    break

        # Free model before loading next
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'model':>12}  {'bs':>4}  {'mlpp':>4}  {'peak':>10}  {'time':>8}  {'status'}")
    print(f"{'-'*12}  {'-'*4}  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}")
    for model, bs, mlpp, peak, wall, status in results:
        peak_s = f"{peak:.2f} GiB" if peak is not None else "—"
        wall_s = f"{wall:.1f}s" if wall is not None else "—"
        print(f"{model:>12}  {bs:>4}  {mlpp:>4}  {peak_s:>10}  {wall_s:>8}  {status}")


if __name__ == "__main__":
    main()
