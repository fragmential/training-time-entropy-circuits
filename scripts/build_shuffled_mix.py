#!/usr/bin/env python
"""Build shuffled, tokenized, pre-chunked dataset mixes for olmo-mix and dolmino-mix.

Samples from each source config proportionally, tokenizes, chunks into seq_len
windows, shuffles chunks, and saves as a .pt file of shape (n_chunks, seq_len).

Usage:
    python scripts/build_shuffled_mix.py --dataset olmo_mix --total_tokens 6_000_000
    python scripts/build_shuffled_mix.py --dataset dolmino_mix --total_tokens 6_000_000
    python scripts/build_shuffled_mix.py --dataset both --total_tokens 6_000_000
"""

import argparse
import os
import sys
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Source proportions (by token count in actual training)
# ---------------------------------------------------------------------------

# olmo-mix-1124 stage 1 proportions (from HF dataset card)
OLMO_MIX_PROPORTIONS = {
    "dclm":            0.949,
    "starcoder":       0.021,
    "pes2o":           0.015,
    "arxiv":           0.005,
    "open-web-math":   0.003,
    "algebraic-stack": 0.003,
    "wiki":            0.001,
}
# Renormalize to sum to 1
_olmo_total = sum(OLMO_MIX_PROPORTIONS.values())
OLMO_MIX_PROPORTIONS = {k: v / _olmo_total for k, v in OLMO_MIX_PROPORTIONS.items()}

# dolmino-mix-1124 stage 2 proportions (50B training mix for 1B, from tech report)
DOLMINO_MIX_PROPORTIONS = {
    "dclm":          0.47,
    "math":          0.21,
    "flan":          0.17,
    "wiki":          0.07,
    "pes2o":         0.06,
    "stackexchange": 0.02,
}
_dolmino_total = sum(DOLMINO_MIX_PROPORTIONS.values())
DOLMINO_MIX_PROPORTIONS = {k: v / _dolmino_total for k, v in DOLMINO_MIX_PROPORTIONS.items()}

HF_DATASETS = {
    "olmo_mix":    "allenai/olmo-mix-1124",
    "dolmino_mix": "allenai/dolmino-mix-1124",
    "pile":        "EleutherAI/the_pile_deduplicated",
}

# Pile is pre-shuffled single source — no proportional mixing needed
PILE_PROPORTIONS = {"__single__": 1.0}

TOKENIZERS = {
    "olmo_mix":    "allenai/OLMo-2-0425-1B",
    "dolmino_mix": "allenai/OLMo-2-0425-1B",
    "pile":        "EleutherAI/pythia-14m",
}


def build_mix(dataset_key: str, proportions: dict, total_tokens: int,
              seq_len: int, tokenizer, seed: int, output_path: str):
    """Build a shuffled chunked mix from the given dataset and proportions."""
    hf_name = HF_DATASETS[dataset_key]
    rng = random.Random(seed)
    all_chunks = []

    for config_name, proportion in proportions.items():
        target_tokens = int(total_tokens * proportion)
        target_chunks = max(1, target_tokens // seq_len)

        print(f"\n  {config_name}: {proportion*100:.1f}% -> {target_chunks} chunks ({target_tokens:,} tokens)")

        if config_name == "__single__":
            ds = load_dataset(hf_name, split="train", streaming=True)
        else:
            ds = load_dataset(hf_name, name=config_name, split="train", streaming=True)
        buf = []
        chunks = []
        docs_read = 0

        for sample in tqdm(ds, desc=f"    {config_name}", leave=False):
            text = sample.get("text", "")
            if not text or not text.strip():
                continue
            ids = tokenizer(text.strip(), add_special_tokens=False).input_ids
            buf.extend(ids)
            docs_read += 1

            # Emit complete chunks from buffer
            while len(buf) >= seq_len:
                chunks.append(buf[:seq_len])
                buf = buf[seq_len:]
                if len(chunks) >= target_chunks:
                    break

            if len(chunks) >= target_chunks:
                break

        if len(chunks) < target_chunks:
            print(f"    WARNING: only got {len(chunks)}/{target_chunks} chunks "
                  f"(source exhausted after {docs_read} docs)")

        print(f"    Got {len(chunks)} chunks from {docs_read} docs")
        all_chunks.extend(chunks)

    # Shuffle all chunks together
    print(f"\n  Shuffling {len(all_chunks)} total chunks...")
    rng.shuffle(all_chunks)

    tensor = torch.tensor(all_chunks, dtype=torch.long)
    print(f"  Final shape: {tensor.shape} ({tensor.shape[0] * seq_len / 1e6:.1f}M tokens)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(tensor, output_path)
    print(f"  Saved to {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["olmo_mix", "dolmino_mix", "pile", "all"],
                   default="all")
    p.add_argument("--total_tokens", type=int, default=6_000_000,
                   help="Target token count per dataset")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="data/mixes",
                   help="Output directory for .pt files")
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/projects/prjs1815/hf_cache")

    n_chunks = args.total_tokens // args.seq_len
    print(f"Target: {args.total_tokens:,} tokens = {n_chunks} chunks of {args.seq_len}")

    ALL_DATASETS = {
        "olmo_mix": OLMO_MIX_PROPORTIONS,
        "dolmino_mix": DOLMINO_MIX_PROPORTIONS,
        "pile": PILE_PROPORTIONS,
    }
    datasets_to_build = list(ALL_DATASETS.items()) if args.dataset == "all" else \
                         [(args.dataset, ALL_DATASETS[args.dataset])]

    for dataset_key, proportions in datasets_to_build:
        tok_name = TOKENIZERS[dataset_key]
        print(f"\nLoading tokenizer: {tok_name}")
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        output_path = os.path.join(
            args.output_dir,
            f"{dataset_key}_{args.total_tokens // 1_000_000}M_{args.seq_len}.pt"
        )
        print(f"\n{'='*60}")
        print(f"Building {dataset_key} ({output_path})")
        print(f"{'='*60}")
        build_mix(dataset_key, proportions, args.total_tokens,
                  args.seq_len, tokenizer, args.seed, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
