#!/usr/bin/env python
"""Estimate peak VRAM (GB) for each model in a collection config.

Usage:
    python scripts/estimate_vram.py --config configs/reproduce_rankme_alpha.yaml
    python scripts/estimate_vram.py --config configs/reproduce_kfac.yaml
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.collect import CollectConfig

# (n_params, hidden, n_layers, intermediate, vocab, bytes_per_param)
_SPECS = {
    "EleutherAI/pythia-14m":          (14e6,    128,  6,   512,   50304, 2),
    "EleutherAI/pythia-14m-deduped":  (14e6,    128,  6,   512,   50304, 2),
    "EleutherAI/pythia-31m":          (31e6,    256,  6,   1024,  50304, 2),
    "EleutherAI/pythia-31m-deduped":  (31e6,    256,  6,   1024,  50304, 2),
    "EleutherAI/pythia-70m":          (70e6,    512,  6,   2048,  50304, 2),
    "EleutherAI/pythia-70m-deduped":  (70e6,    512,  6,   2048,  50304, 2),
    "EleutherAI/pythia-160m":         (162e6,   768,  12,  3072,  50304, 2),
    "EleutherAI/pythia-160m-deduped": (162e6,   768,  12,  3072,  50304, 2),
    "EleutherAI/pythia-410m":         (405e6,   1024, 24,  4096,  50304, 2),
    "EleutherAI/pythia-410m-deduped": (405e6,   1024, 24,  4096,  50304, 2),
    "EleutherAI/pythia-1b":           (1.01e9,  2048, 16,  8192,  50304, 2),
    "EleutherAI/pythia-1b-deduped":   (1.01e9,  2048, 16,  8192,  50304, 2),
    "EleutherAI/pythia-1.4b":         (1.41e9,  2048, 24,  8192,  50304, 2),
    "EleutherAI/pythia-1.4b-deduped": (1.41e9,  2048, 24,  8192,  50304, 2),
    "EleutherAI/pythia-2.8b":         (2.77e9,  2560, 32,  10240, 50304, 2),
    "EleutherAI/pythia-2.8b-deduped": (2.77e9,  2560, 32,  10240, 50304, 2),
    "EleutherAI/pythia-6.9b":         (6.86e9,  4096, 32,  16384, 50304, 2),
    "EleutherAI/pythia-6.9b-deduped": (6.86e9,  4096, 32,  16384, 50304, 2),
    "EleutherAI/pythia-12b":          (11.85e9, 5120, 36,  20480, 50304, 2),
    "EleutherAI/pythia-12b-deduped":  (11.85e9, 5120, 36,  20480, 50304, 2),
    "allenai/OLMo-2-0425-1B":        (1.24e9,  2048, 16,  8192,  100352, 2),
    "allenai/OLMo-2-1124-7B":        (6.89e9,  4096, 32,  11008, 100352, 2),
}


def _resolve(val, model, idx):
    if isinstance(val, dict): return val.get(model, val)
    if isinstance(val, list): return val[idx]
    return val


def estimate_vram_gb(model_name, batch_size, seq_len, collect_A, collect_G, collect_B, max_layers_per_pass=4):
    if model_name not in _SPECS:
        return float("nan")
    n_params, hidden, n_layers, intermediate, vocab, bpp = _SPECS[model_name]
    is_olmo = "olmo" in model_name.lower()

    # Always present: model weights + forward pass peak (one layer at a time)
    model_bytes = n_params * bpp
    forward_peak = batch_size * seq_len * max(4 * hidden, intermediate) * bpp
    total = model_bytes + forward_peak

    # Backward pass (only if collecting G)
    if collect_G:
        total += batch_size * (seq_len - 1) * vocab * 4  # logits (fp32)
        total += batch_size * (seq_len - 1) * vocab * 4  # probs (fp32)
        total += model_bytes * 1.5  # gradient checkpointing + retain_graph overhead

    # Covariance accumulators for A/G (fp32, d×d per projection per layer in pass group)
    n_target = min(max_layers_per_pass, n_layers)
    projs_per_layer = 3 if is_olmo else 2
    cov_per_proj = (hidden ** 2 + intermediate ** 2) * 4  # A: d_in², G: d_out² (or vice versa)
    if collect_A: total += cov_per_proj * projs_per_layer * n_target
    if collect_G: total += cov_per_proj * projs_per_layer * n_target
    # B accumulators are similar scale to A
    if collect_B: total += cov_per_proj * projs_per_layer * n_target

    return total * 1.08 / 1e9  # +8% CUDA overhead


def estimate_all(cfg):
    models = cfg.model_name if isinstance(cfg.model_name, list) else [cfg.model_name]
    return {
        m: estimate_vram_gb(
            m, _resolve(cfg.batch_size, m, i), cfg.seq_len,
            cfg.collect_A, cfg.collect_G, cfg.collect_B,
            _resolve(cfg.max_layers_per_pass, m, i),
        )
        for i, m in enumerate(models)
    }


if __name__ == "__main__":
    from jsonargparse import CLI
    for model, gb in estimate_all(CLI(CollectConfig)).items():
        print(f"{model.split('/')[-1]:30s} {gb:6.1f} GB")
