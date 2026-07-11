#!/usr/bin/env python
"""STALE / UNMIGRATED (unused) — predates the accessor rebuild and still calls removed
`.leaves()` / `.view()`; kept only for the *idea* (per-band projection analysis on stored
eigenvectors + activations). Do not run as-is; rework onto the tree (`acc.v` / `acc[leaf].q`)
if revived.

Activation ratio analysis: per-band projection comparison across datasets.

For each target layer, partitions the A eigenvectors into percentile bands
(e.g. top 10%, 10-25%, 25-50%, bottom 50%) and measures the mean absolute
projection |U_band^T x| per band per dataset.

Comparing memorized vs clean data reveals which eigenspace bands are
differentially activated for memorized content.

Simplified from Merullo et al.: the FIM product banding λ_G^i × λ_A^j
collapses to a marginal on λ_A (G drops out of |Zx||), so we partition
A eigenvectors by λ_A rank directly. See activation_ratio_info.txt.

Results structure:
    {hook_name: {dataset_name: {band_name: mean_abs_projection}}}

Usage:
    python scripts/activation_ratio.py \\
        --basis_path covariance_factors/olmo_mix/OLMo-2-1B/step1000.pt \\
        --hook_names blk14.up blk14.down blk15.up blk15.down \\
        --model_name allenai/OLMo-2-1B \\
        --revision final \\
        --datasets clean:olmo_mix \\
        --datasets mem:memorization_kfac\\ (reference\\ repo)/data/olmo2_1b_mem_extra_dedup_j70.jsonl \\
        --output_path data/results/activation_ratios.npy
"""

import json
import os
import sys
import torch
import numpy as np
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_registry import get_model_config, load_model, load_tokenizer, get_mlp_projections
from utils.hooks import HookCollector
from utils.accessor import DataAccessor
from utils.data_utils import get_loader, load_and_cache_texts, pack_sequences, compute_token_mask


# ---------------------------------------------------------------------------
# Default bands (percentile ranges into sorted A eigenvectors)
# ---------------------------------------------------------------------------

DEFAULT_BANDS = [
    ("top_10",   0.00, 0.10),
    ("10_25",    0.10, 0.25),
    ("25_50",    0.25, 0.50),
    ("bottom_50",0.50, 1.00),
]


def make_band_indices(d: int, bands=None):
    """Return {band_name: index_array} for eigenvector partitioning.

    Indices are into sorted-descending A eigenvectors (index 0 = largest λ_A).
    """
    if bands is None:
        bands = DEFAULT_BANDS
    result = {}
    for name, lo, hi in bands:
        start = int(lo * d)
        end = int(hi * d)
        result[name] = np.arange(start, end)
    return result


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl_sequences(path: str, tokenizer=None, max_len: int = 512):
    """Yield input_ids tensors from a jsonl file.

    Handles two formats:
        - {"prefix_ids": [...], "suffix_ids": [...]} — pre-tokenized
        - {"prefix_text": "...", ...}                — text, needs tokenizer
    """
    with open(path) as f:
        for line in f:
            obj = json.loads(line.strip())

            if "prefix_ids" in obj:
                ids = obj["prefix_ids"]
                if "suffix_ids" in obj:
                    ids = ids + obj["suffix_ids"]
                yield torch.tensor(ids[:max_len], dtype=torch.long)

            elif "prefix_text" in obj:
                if tokenizer is None:
                    raise ValueError(
                        f"tokenizer required for text-format jsonl: {path}"
                    )
                text = obj["prefix_text"]
                if "suffix_text" in obj:
                    text = text + obj.get("suffix_text", "")
                enc = tokenizer(text, return_tensors="pt", max_length=max_len,
                                truncation=True)
                yield enc.input_ids[0]


def load_text_dataset(dataset_spec: str, tokenizer, num_samples: int,
                      min_length: int, max_len: int, seq_len: int,
                      packing: str = "padded"):
    """Load a dataset from name (registry) or jsonl path.

    Returns:
        If packing="padded": list of text strings
        If packing="packed": (N, seq_len) token tensor
    """
    if dataset_spec.endswith(".jsonl"):
        seqs = list(load_jsonl_sequences(dataset_spec, tokenizer, max_len))
        if packing == "packed":
            # Concatenate and chunk
            all_ids = torch.cat(seqs)
            n_chunks = len(all_ids) // seq_len
            return all_ids[:n_chunks * seq_len].reshape(n_chunks, seq_len)
        return seqs  # list of variable-length tensors
    else:
        loader_fn = get_loader(dataset_spec)
        texts = load_and_cache_texts(
            loader_fn, num_samples, min_length, tokenizer, dataset_spec,
        )
        if packing == "packed":
            return pack_sequences(texts, tokenizer, seq_len)
        return texts


# ---------------------------------------------------------------------------
# Band measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_band_projections(
    model,
    collectors: dict,               # hook_name -> HookCollector (acts mode)
    basis_dict: dict,               # hook_name -> (U, band_indices)
    dataset_spec: str,
    tokenizer,
    num_samples: int = 2000,
    min_length: int = 32,
    max_len: int = 512,
    seq_len: int = 512,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict:
    """Stream dataset and accumulate per-band mean |U_band^T x|.

    Returns:
        {hook_name: {band_name: mean_projection}}
    """
    # Running accumulators: {hook_name: {band_name: (sum, count)}}
    band_sums = {
        hn: {bn: 0.0 for bn in basis_dict[hn][1].keys()}
        for hn in collectors
    }
    band_counts = {
        hn: {bn: 0 for bn in basis_dict[hn][1].keys()}
        for hn in collectors
    }

    is_jsonl = dataset_spec.endswith(".jsonl")

    if is_jsonl:
        # Variable-length sequences — use padded batching
        seqs = list(load_jsonl_sequences(dataset_spec, tokenizer, max_len))
        if num_samples and len(seqs) > num_samples:
            seqs = seqs[:num_samples]

        def _jsonl_batches():
            for i in range(0, len(seqs), batch_size):
                batch = seqs[i:i + batch_size]
                max_len_b = max(s.size(0) for s in batch)
                input_ids = torch.zeros(len(batch), max_len_b, dtype=torch.long)
                attention_mask = torch.zeros(len(batch), max_len_b, dtype=torch.long)
                for j, s in enumerate(batch):
                    input_ids[j, :s.size(0)] = s
                    attention_mask[j, :s.size(0)] = 1
                yield input_ids.to(device), attention_mask.to(device)

        batches = _jsonl_batches()
    else:
        packed_ids = load_text_dataset(
            dataset_spec, tokenizer, num_samples, min_length, max_len, seq_len,
            packing="packed",
        )
        loader = DataLoader(
            TensorDataset(packed_ids), batch_size=batch_size, shuffle=False,
        )
        batches = ((ids.to(device), None) for (ids,) in loader)

    for input_ids, attention_mask in tqdm(batches, desc=f"  {dataset_spec}", leave=False):
        # Clear collector act lists before each batch so we don't accumulate across batches
        for c in collectors.values():
            if c._acts_list is not None:
                c._acts_list.clear()
            c._n_acts = 0
            c._token_mask = None  # use all tokens

        fwd_kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            fwd_kwargs["attention_mask"] = attention_mask

        model(**fwd_kwargs)

        # Process each hook
        for hook_name, collector in collectors.items():
            if not collector._acts_list:
                continue
            x = torch.cat(collector._acts_list, dim=0).float()  # (N_tokens, d_in)

            U, band_indices = basis_dict[hook_name]  # (d_in, d_in), {band: indices}
            U = U.to(device)

            # All projections at once: (N_tokens, d_in)
            projections = (x @ U).abs()

            for band_name, indices in band_indices.items():
                band_sums[hook_name][band_name] += projections[:, indices].mean(dim=1).sum().item()
                band_counts[hook_name][band_name] += projections.size(0)

        # Clear collector lists
        for c in collectors.values():
            if c._acts_list is not None:
                c._acts_list.clear()
            c._n_acts = 0

    # Compute means
    results = {}
    for hook_name in collectors:
        results[hook_name] = {
            bn: (band_sums[hook_name][bn] / max(band_counts[hook_name][bn], 1))
            for bn in band_sums[hook_name]
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    basis_path: str,
    model_name: str,
    hook_names: "list[str]" = None,
    revision: str = "main",
    datasets: "list[str]" = None,
    bands: "list[tuple]" = None,
    num_samples: int = 2000,
    min_length: int = 32,
    max_len: int = 512,
    seq_len: int = 512,
    batch_size: int = 16,
    output_path: str = None,
):
    """Measure activation band projections across datasets.

    Args:
        basis_path: .pt file with cov_svd data containing acts eigenvectors.
        model_name: Full HuggingFace model name (e.g. allenai/OLMo-2-1B).
        hook_names: Which hook points to analyse. Defaults to all in basis_path.
        revision: Model revision / checkpoint to load.
        datasets: List of "name:spec" strings, e.g. "clean:olmo_mix" or
                  "mem:path/to/mem.jsonl". Defaults to ["clean:olmo_mix"].
        bands: Custom band definitions as list of (name, lo, hi) tuples.
        num_samples: Max text samples per dataset (for registry loaders).
        min_length: Min sample length in tokens.
        max_len: Max sequence length.
        seq_len: Chunk length for packed sequences.
        batch_size: Inference batch size.
        output_path: Where to save results (.npy). Defaults to
                     data/results/activation_ratios_{model_short}_{step}.npy.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load basis
    import torch as _torch
    basis_data = _torch.load(basis_path, map_location="cpu", weights_only=False)
    acc = DataAccessor(basis_data)

    if hook_names is None:
        hook_names = acc.leaves()

    # Build basis dict: {leaf: (U_tensor, band_indices)} from each leaf's acts eigvecs
    basis_dict = {}
    for hook in hook_names:
        fv = acc.view(hook, "acts")
        eigvecs = fv.eigvecs if fv is not None else None
        if eigvecs is None:
            print(f"  WARNING: no acts eigenvectors for {hook}, skipping")
            continue
        d = eigvecs.size(0)
        band_idx = make_band_indices(d, bands)
        basis_dict[hook] = (eigvecs, band_idx)

    if not basis_dict:
        print("No valid hook points found in basis file.")
        return

    print(f"Loaded basis: {len(basis_dict)} hook points from {basis_path}")

    # Load model
    model_config = get_model_config(model_name)
    tokenizer = load_tokenizer(model_config, revision=revision)
    model = load_model(model_config, model_name, revision)
    model.to(device)
    model.eval()

    # Set up hook collectors on target layers
    collectors = {}
    for hn in basis_dict:
        # Resolve hook_name to layer
        import re
        m = re.match(r"blk(\d+)\.(up|down|gate)", hn)
        if not m:
            print(f"  WARNING: can't resolve layer for {hn}, skipping")
            continue
        block_idx = int(m.group(1))
        proj_name_suffix = m.group(2)
        for name, layer in get_mlp_projections(model, model_config, block_idx):
            if name == hn:
                collectors[hn] = HookCollector(layer, capture="input", mode="acts")
                break

    if not collectors:
        print("No layers hooked. Check hook_names match model architecture.")
        return

    print(f"Hooked {len(collectors)} layers: {list(collectors.keys())}")

    # Parse datasets
    if datasets is None:
        datasets = ["clean:olmo_mix"]
    dataset_specs = {}
    for ds in datasets:
        if ":" in ds:
            name, spec = ds.split(":", 1)
        else:
            name = spec = ds
        dataset_specs[name] = spec

    # Measure per dataset
    all_results = {}  # {hook_name: {dataset_name: {band_name: mean}}}
    for ds_name, ds_spec in dataset_specs.items():
        print(f"Dataset: {ds_name} ({ds_spec})")
        ds_results = measure_band_projections(
            model, collectors, basis_dict, ds_spec, tokenizer,
            num_samples=num_samples, min_length=min_length,
            max_len=max_len, seq_len=seq_len,
            batch_size=batch_size, device=device,
        )
        for hn, band_means in ds_results.items():
            if hn not in all_results:
                all_results[hn] = {}
            all_results[hn][ds_name] = band_means

    # Compute ratios (first dataset over second, if two datasets given)
    ds_names = list(dataset_specs.keys())
    if len(ds_names) == 2:
        d1, d2 = ds_names
        for hn in all_results:
            ratios = {}
            for band in all_results[hn].get(d1, {}):
                v1 = all_results[hn][d1][band]
                v2 = all_results[hn][d2].get(band, 1.0)
                ratios[band] = v1 / v2 if v2 > 0 else float("nan")
            all_results[hn]["ratio"] = ratios

    # Close hooks
    for c in collectors.values():
        c.close()

    # Print summary
    print("\nResults:")
    for hn, ds_data in all_results.items():
        print(f"  {hn}:")
        for ds_name, band_means in ds_data.items():
            vals = "  ".join(f"{bn}={v:.4f}" for bn, v in band_means.items())
            print(f"    {ds_name}: {vals}")

    # Save
    if output_path is None:
        step_str = os.path.splitext(os.path.basename(basis_path))[0]
        short = model_name.split("/")[-1]
        output_path = os.path.join("data", "results", f"activation_ratios_{short}_{step_str}.npy")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, all_results)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
