#!/usr/bin/env python
"""
Multi-checkpoint K-FAC factor collection for Pythia and OLMo model families.

Collects activation covariance (A) and gradient covariance (G) matrices for MLP
projections across training checkpoints.  Follows the same checkpoint-iteration
pattern as rankme_alpha_scripts/extract_activations.py, reusing model_registry
for checkpoint discovery and model loading.

The KFAC collector class is adapted from
memorization_kfac/data/collect_kfac_multilayer.py.
"""
import gc
import json
import os
import sys
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader, wikitext_loader
from utils.model_registry import (
    get_model_config,
    get_checkpoint_schedule,
    load_model,
    load_tokenizer,
    get_mlp_projections,
    get_num_layers,
    prefetch_checkpoint,
    delete_cached_revision,
)

DATASET_LOADERS = {
    "fineweb": fineweb_loader.get_dataset,
    "wikitext": wikitext_loader.get_dataset,
}


# ---------------------------------------------------------------------------
# KFAC collector (from memorization_kfac/data/collect_kfac_multilayer.py)
# ---------------------------------------------------------------------------
class KFAC:
    """Collect A = E[xxT] and G = E[ggT] for a nn.Linear layer."""

    def __init__(self, layer: nn.Linear):
        d_out, d_in = layer.weight.shape
        dev = layer.weight.device
        self.A = torch.zeros(d_in, d_in, dtype=torch.float32, device=dev)
        self.G = torch.zeros(d_out, d_out, dtype=torch.float32, device=dev)
        self.n = 0
        self._buf = None
        self._h_fwd = layer.register_forward_pre_hook(self._fwd, prepend=False)
        self._h_bwd = layer.register_full_backward_hook(self._bwd, prepend=False)

    def _fwd(self, _, inp):
        if torch.is_grad_enabled():
            x = inp[0][:, :-1].detach()  # exclude last position (no target)
            self._buf = x.reshape(-1, x.size(-1)).float()

    def _bwd(self, _, __, go):
        if go[0] is None or self._buf is None:
            return
        g = go[0][:, :-1].detach().reshape(-1, go[0].size(-1)).float()
        self.A.add_(self._buf.T @ self._buf)
        self.G.add_(g.T @ g)
        self.n += g.size(0)
        self._buf = None

    def factors(self):
        return self.A / self.n, self.G / self.n

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()
        self._buf = None


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def load_and_cache_texts(dataset_name, num_samples, min_length, tokenizer):
    """Load dataset texts, filtering by minimum token length.  Caches to disk."""
    os.makedirs("data/cache", exist_ok=True)
    cache_path = os.path.join("data", "cache", f"filtered_texts_{dataset_name}_{num_samples}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            texts = json.load(f)
        print(f"Loaded {len(texts)} cached sequences from {cache_path}")
        return texts

    dataset = DATASET_LOADERS[dataset_name]()
    texts = []
    for seq in tqdm(dataset, desc="Filtering"):
        text = seq["text"]
        if tokenizer(text, return_tensors="pt").input_ids.shape[-1] > min_length:
            texts.append(text)
            if len(texts) >= num_samples:
                break
    with open(cache_path, "w") as f:
        json.dump(texts, f)
    print(f"Filtered and cached {len(texts)} sequences to {cache_path}")
    return texts


def pack_sequences(texts, tokenizer, seq_len):
    """Tokenize texts and pack into fixed-length chunks (no padding)."""
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
    n_chunks = len(all_ids) // seq_len
    if n_chunks == 0:
        raise ValueError(f"Not enough tokens ({len(all_ids)}) for even one chunk of {seq_len}")
    all_ids = all_ids[: n_chunks * seq_len]
    return torch.tensor(all_ids, dtype=torch.long).view(n_chunks, seq_len)


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------------------------------------------------------------------------
# K-FAC collection for a single checkpoint
# ---------------------------------------------------------------------------
def collect_kfac_for_checkpoint(
    model, config, packed_ids, target_blocks, layers_per_pass, batch_size,
    sample_labels, device,
):
    """Run K-FAC collection on packed_ids for the given model.

    Returns dict: {name: {"A": Tensor(cpu), "G": Tensor(cpu), "n": int}, ...}
    """
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    loader = DataLoader(TensorDataset(packed_ids), batch_size=batch_size, shuffle=False)
    all_factors = {}

    for blk_group in chunked(target_blocks, layers_per_pass):
        # Freeze everything, then unfreeze only target layers
        for p in model.parameters():
            p.requires_grad_(False)

        targets = []
        for b in blk_group:
            targets.extend(get_mlp_projections(model, config, b))
        for _, layer in targets:
            layer.weight.requires_grad_(True)

        collectors = {name: KFAC(layer) for name, layer in targets}

        for (ids_batch,) in tqdm(loader, desc=f"KFAC blk {blk_group}", leave=False):
            x = ids_batch.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)
            logits = model(x).logits[:, :-1].float()

            if sample_labels:
                with torch.no_grad():
                    y = torch.multinomial(
                        torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1)), 1
                    ).squeeze(1)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y
                )
            else:
                labels = x.clone()
                labels[:, :-1] = x[:, 1:]
                labels[:, -1] = -100
                loss = ce(logits.reshape(-1, logits.size(-1)), labels[:, :-1].reshape(-1))

            loss.backward()

        # Save factors and clean up
        for name, collector in collectors.items():
            A, G = collector.factors()
            all_factors[name] = {"A": A.cpu(), "G": G.cpu(), "n": collector.n}
            collector.close()
        del collectors
        torch.cuda.empty_cache()

    return all_factors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    model_name: str = "EleutherAI/pythia-14m",
    dataset_name: str = "fineweb",
    num_samples: int = 5000,
    seq_len: int = 512,
    min_length: int = 32,
    batch_size: int = 32,
    max_checkpoints: int = 30,
    target_blocks: list = None,
    layers_per_pass: int = 4,
    sample_labels: bool = True,
    revisions_file: str = None,
    early_training_model: str = None,
):
    config = get_model_config(model_name)
    print(f"Model: {model_name} (family={config.family})")

    # Checkpoint schedule
    schedule = get_checkpoint_schedule(
        config, max_checkpoints,
        revisions_file=revisions_file, early_training_model=early_training_model,
    )
    print(f"Total checkpoints: {len(schedule)}")

    # Output directory
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    out_dir = os.path.join("kfac_factors", dataset_name, short_name)
    os.makedirs(out_dir, exist_ok=True)

    # Filter to unprocessed checkpoints
    to_process = []
    for step_num, revision, step_model in schedule:
        if os.path.exists(os.path.join(out_dir, f"step{step_num}.pt")):
            continue
        to_process.append((step_num, revision, step_model))

    if not to_process:
        print("All checkpoints already collected.")
        return

    # Tokenizer
    first_revision = to_process[0][1]
    tokenizer = load_tokenizer(config, revision=first_revision)

    # Load and pack data
    texts = load_and_cache_texts(dataset_name, num_samples, min_length, tokenizer)
    packed_ids = pack_sequences(texts, tokenizer, seq_len)
    print(f"Packed data: {packed_ids.shape[0]} chunks of {seq_len} tokens")

    # Main loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    executor = ThreadPoolExecutor(max_workers=1)

    print(f"Processing {len(to_process)} remaining checkpoints (sample_labels={sample_labels})...")
    for idx, (step_num, revision, step_model) in enumerate(tqdm(to_process, desc="Checkpoints")):
        # Prefetch next
        if idx + 1 < len(to_process):
            _, next_rev, next_model = to_process[idx + 1]
            prefetch_future = executor.submit(prefetch_checkpoint, next_model, next_rev)
        else:
            prefetch_future = None

        try:
            model = load_model(config, step_model, revision)
            model.to(device)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = False
            model.train()
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = 0.0

            # Resolve target blocks
            n_layers = get_num_layers(model, config)
            blocks = target_blocks if target_blocks is not None else list(range(n_layers))

            factors = collect_kfac_for_checkpoint(
                model, config, packed_ids, blocks, layers_per_pass,
                batch_size, sample_labels, device,
            )

            save_path = os.path.join(out_dir, f"step{step_num}.pt")
            torch.save(factors, save_path)
            tqdm.write(f"Step {step_num}: saved {len(factors)} projections to {save_path}")

        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        if prefetch_future is not None:
            prefetch_future.result()
        delete_cached_revision(step_model, revision)

    executor.shutdown(wait=True)
    print(f"Completed! K-FAC factors saved to {out_dir}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
