#!/usr/bin/env python
"""Unified collection pipeline for residual activations and covariance factors.

Replaces both rankme_alpha_scripts/extract_activations.py and kfac_scripts/collect_kfac.py
with a single configurable script driven by YAML config files.

Supports:
    - Residual stream capture: identity_head, after_final_norm, before_final_norm, per-block hooks
    - Covariance collection: A (input), G (gradient), B (post-weight output)
    - Data modes: packed (no padding) or padded
    - Token selection: all tokens or last token (per sequence or per document)
    - Storage formats: cov, cov_svd, eigenvalues
    - Answer-only gradient collection for downstream tasks
    - Cross-basis projections at save time

Usage:
    python scripts/collect.py --config configs/residual_pythia.yaml --model_name EleutherAI/pythia-14m
    python scripts/collect.py --config configs/covariance_pythia.yaml --array_id 0
"""

import gc
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import get_loader
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
from utils.hooks import CovarianceCollector, ResidualCapture
from utils.storage import save_factors, save_activations
from utils.data_utils import (
    load_and_cache_texts,
    pack_sequences,
    compute_token_mask,
    compute_labels,
)


# Fields that may vary per model in a sweep (accept scalar, list, or dict).
VECTORIZABLE_FIELDS = {"batch_size", "max_checkpoints", "max_layers_per_pass"}


@dataclass
class CollectConfig:
    # --- Model ---
    model_name: "str | list[str]" = "EleutherAI/pythia-14m"
    max_checkpoints: "int | dict[str, int] | list[int]" = 50
    target_blocks: "list | None" = None
    max_layers_per_pass: "int | dict[str, int] | list[int]" = 4

    # --- Data ---
    dataset_name: str = "fineweb"
    dataset_content_key: str = "text"
    num_samples: int = 5000
    seq_len: int = 512
    max_length: int = 512
    min_length: int = 32
    batch_size: "int | dict[str, int] | list[int]" = 128

    # --- Data format ---
    packing: str = "padded"
    token_selection: str = "last"
    skip_positions: int = 0
    boundary_token_ids: "list | None" = None

    # --- What to collect ---
    collect_residual: bool = True
    residual_hook_point: str = "identity_head"
    collect_A: bool = False
    collect_G: bool = False
    collect_B: bool = False
    sample_labels: bool = True
    label_samples: int = 1
    seed: int = 42

    # --- Answer-only mode ---
    answer_only: bool = False
    answer_start_key: "str | None" = None

    # --- Storage ---
    storage_format: str = "cov"
    cross_basis_refs: "list | None" = None
    output_dir: "str | None" = None

    # --- Sweep ---
    array_id: "int | None" = None

    def __post_init__(self):
        if not isinstance(self.model_name, list):
            # Single model — validate no vectorized fields are lists
            for name in VECTORIZABLE_FIELDS:
                val = getattr(self, name)
                if isinstance(val, (list, dict)):
                    raise ValueError(
                        f"{name} is vectorized but model_name is a single string. "
                        f"Use a model_name list or pass a scalar {name}."
                    )
            return

        # Non-vectorizable fields must not be lists
        for f in fields(self):
            if f.name not in VECTORIZABLE_FIELDS and f.name != "model_name":
                val = getattr(self, f.name)
                if isinstance(val, list) and f.name not in (
                    "target_blocks", "boundary_token_ids", "cross_basis_refs",
                ):
                    raise ValueError(
                        f"{f.name} must be the same for all models (got a list). "
                        f"Only {VECTORIZABLE_FIELDS} can vary per model."
                    )

        model_list = self.model_name

        # Convert list-format vectorized args to dict by zipping with model_name
        for name in VECTORIZABLE_FIELDS:
            val = getattr(self, name)
            if isinstance(val, list):
                if len(val) != len(model_list):
                    raise ValueError(
                        f"{name} list length ({len(val)}) != model_name list length ({len(model_list)})"
                    )
                setattr(self, name, dict(zip(model_list, val)))

        # If array_id is set, resolve to a single model
        if self.array_id is not None:
            model = model_list[self.array_id]
            for name in VECTORIZABLE_FIELDS:
                val = getattr(self, name)
                if isinstance(val, dict):
                    setattr(self, name, val.get(model, None))
            self.model_name = model


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _collect_residual_for_checkpoint(
    model,
    config,
    texts,
    tokenizer,
    residual_hook_point,
    token_selection,
    max_length,
    batch_size,
    packing,
    packed_ids,
    boundary_token_ids,
):
    """Collect residual stream activations for one checkpoint. Returns numpy array."""
    capture = ResidualCapture(model, config, hook_point=residual_hook_point)
    all_acts = []

    with torch.no_grad():
        if packing == "packed":
            loader = DataLoader(TensorDataset(packed_ids), batch_size=batch_size, shuffle=False)
            for (ids_batch,) in tqdm(loader, desc="Residual", leave=False):
                x = ids_batch.to(next(model.parameters()).device, non_blocking=True)

                if residual_hook_point == "identity_head":
                    outputs = model(x)
                    acts = outputs.logits.detach()
                else:
                    model(x)
                    acts = capture.activations

                if acts is None:
                    continue

                if token_selection == "last":
                    mask = compute_token_mask(
                        x, token_selection="last",
                        boundary_token_ids=boundary_token_ids,
                    )
                    selected = acts[mask]
                else:
                    # All except last position
                    selected = acts[:, :-1].reshape(-1, acts.size(-1))

                all_acts.append(selected.float().cpu().numpy())
        else:
            # Padded mode
            for bidx in tqdm(range(0, len(texts), batch_size), desc="Residual", leave=False):
                batch = texts[bidx : bidx + batch_size]
                tokenized = tokenizer(
                    batch, padding="longest", return_tensors="pt",
                    max_length=max_length, truncation=True,
                )
                device = next(model.parameters()).device
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)

                if residual_hook_point == "identity_head":
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    acts = outputs.logits.detach()
                else:
                    model(input_ids=input_ids, attention_mask=attention_mask)
                    acts = capture.activations

                if acts is None:
                    continue

                if token_selection == "last":
                    last_indices = attention_mask.sum(dim=1) - 1
                    batch_indices = torch.arange(acts.size(0), device=device)
                    selected = acts[batch_indices, last_indices]
                else:
                    selected = acts[attention_mask.bool()]

                all_acts.append(selected.float().cpu().numpy())
                del input_ids, attention_mask

    capture.restore(model)
    if not all_acts:
        return None
    return np.vstack(all_acts)


def _collect_covariance_for_checkpoint(
    model,
    config,
    packed_ids,
    target_blocks,
    max_layers_per_pass,
    batch_size,
    sample_labels,
    label_samples,
    device,
    collect_A,
    collect_G,
    collect_B,
    token_mask_fn,
):
    """Collect covariance factors for one checkpoint (packed data). Returns factors dict."""
    loader = DataLoader(TensorDataset(packed_ids), batch_size=batch_size, shuffle=False)
    all_factors = {}

    for blk_group in chunked(target_blocks, max_layers_per_pass):
        # Freeze everything, then unfreeze only target layers if we need gradients
        needs_grad = collect_G
        for p in model.parameters():
            p.requires_grad_(False)

        targets = []
        for b in blk_group:
            targets.extend(get_mlp_projections(model, config, b))

        if needs_grad:
            for _, layer in targets:
                layer.weight.requires_grad_(True)

        collectors = {
            name: CovarianceCollector(layer, collect_A=collect_A, collect_G=collect_G, collect_B=collect_B)
            for name, layer in targets
        }

        for (ids_batch,) in tqdm(loader, desc=f"Cov blk {blk_group}", leave=False):
            x = ids_batch.to(device, non_blocking=True)

            # Compute and set token mask
            mask = token_mask_fn(x)
            for collector in collectors.values():
                collector.set_token_mask(mask)

            if needs_grad:
                model.zero_grad(set_to_none=True)
                logits = model(x).logits[:, :-1].float()

                if sample_labels:
                    probs = torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1))
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    # Accumulate gradients from multiple samples (Monte Carlo Fisher)
                    for _ in range(label_samples):
                        with torch.no_grad():
                            y = torch.multinomial(probs, 1).squeeze(1)
                        loss = nn.functional.cross_entropy(flat_logits, y)
                        (loss / label_samples).backward(retain_graph=True)
                else:
                    labels = compute_labels(x)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels[:, :-1].reshape(-1),
                        ignore_index=-100,
                    )
                    loss.backward()
            else:
                with torch.no_grad():
                    model(x)

        # Collect factors and clean up
        for name, collector in collectors.items():
            all_factors[name] = collector.factors()
            collector.close()
        del collectors
        torch.cuda.empty_cache()

    return all_factors


def _collect_covariance_padded_for_checkpoint(
    model,
    config,
    texts,
    tokenizer,
    target_blocks,
    max_layers_per_pass,
    batch_size,
    max_length,
    sample_labels,
    label_samples,
    device,
    collect_A,
    collect_G,
    collect_B,
    token_selection,
    answer_start_positions_all=None,
):
    """Collect covariance factors with padded data for one checkpoint."""
    all_factors = {}
    needs_grad = collect_G

    for blk_group in chunked(target_blocks, max_layers_per_pass):
        for p in model.parameters():
            p.requires_grad_(False)

        targets = []
        for b in blk_group:
            targets.extend(get_mlp_projections(model, config, b))

        if needs_grad:
            for _, layer in targets:
                layer.weight.requires_grad_(True)

        collectors = {
            name: CovarianceCollector(layer, collect_A=collect_A, collect_G=collect_G, collect_B=collect_B)
            for name, layer in targets
        }

        for bidx in tqdm(range(0, len(texts), batch_size), desc=f"Cov blk {blk_group}", leave=False):
            batch = texts[bidx : bidx + batch_size]
            tokenized = tokenizer(
                batch, padding="longest", return_tensors="pt",
                max_length=max_length, truncation=True,
            )
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            ans_starts = None
            if answer_start_positions_all is not None:
                ans_starts = answer_start_positions_all[bidx : bidx + batch_size].to(device)

            mask = compute_token_mask(
                input_ids,
                attention_mask=attention_mask,
                token_selection=token_selection,
                answer_start_positions=ans_starts,
            )
            for collector in collectors.values():
                collector.set_token_mask(mask)

            if needs_grad:
                model.zero_grad(set_to_none=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()

                if sample_labels:
                    probs = torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1))
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    for _ in range(label_samples):
                        with torch.no_grad():
                            y = torch.multinomial(probs, 1).squeeze(1)
                        loss = nn.functional.cross_entropy(flat_logits, y)
                        (loss / label_samples).backward(retain_graph=True)
                else:
                    labels = compute_labels(input_ids, attention_mask, ans_starts)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels[:, :-1].reshape(-1),
                        ignore_index=-100,
                    )
                    loss.backward()
            else:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

            del input_ids, attention_mask

        for name, collector in collectors.items():
            all_factors[name] = collector.factors()
            collector.close()
        del collectors
        torch.cuda.empty_cache()

    return all_factors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: CollectConfig):
    model_name = cfg.model_name
    if isinstance(model_name, list):
        raise ValueError(
            "model_name is still a list — pass --array_id to select a model, "
            "or pass a single --model_name."
        )

    model_config = get_model_config(model_name)
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    print(f"Model: {model_name} (family={model_config.family})")

    needs_covariance = cfg.collect_A or cfg.collect_G or cfg.collect_B

    # Output directories
    output_dir = cfg.output_dir
    if output_dir is None:
        if cfg.collect_residual and not needs_covariance:
            output_dir = os.path.join("activations", cfg.dataset_name, short_name)
        elif needs_covariance and not cfg.collect_residual:
            output_dir = os.path.join("covariance_factors", cfg.dataset_name, short_name)
        else:
            output_dir = os.path.join("collected", cfg.dataset_name, short_name)
    os.makedirs(output_dir, exist_ok=True)

    # Checkpoint schedule
    schedule = get_checkpoint_schedule(model_config, cfg.max_checkpoints)
    print(f"Total checkpoints: {len(schedule)}")

    # Filter to unprocessed checkpoints
    to_process = []
    for step_num, revision, step_model in schedule:
        residual_done = not cfg.collect_residual or os.path.exists(
            os.path.join(output_dir, f"step{step_num}.npy")
        )
        cov_done = not needs_covariance or os.path.exists(
            os.path.join(output_dir, f"step{step_num}.pt")
        )
        if residual_done and cov_done:
            continue
        to_process.append((step_num, revision, step_model))

    if not to_process:
        print("All checkpoints already collected.")
        return

    # Tokenizer
    first_revision = to_process[0][1]
    tokenizer = load_tokenizer(model_config, revision=first_revision)

    # Load data
    loader_fn = get_loader(cfg.dataset_name)
    texts = load_and_cache_texts(
        loader_fn, cfg.num_samples, cfg.min_length, tokenizer,
        cfg.dataset_name, cfg.dataset_content_key,
    )

    # Pack sequences if needed
    packed_ids = None
    if cfg.packing == "packed":
        packed_ids = pack_sequences(texts, tokenizer, cfg.seq_len)
        print(f"Packed data: {packed_ids.shape[0]} chunks of {cfg.seq_len} tokens")

    # Boundary tokens for packed + last-token mode
    boundary_token_ids = cfg.boundary_token_ids
    if boundary_token_ids is None and cfg.packing == "packed" and cfg.token_selection == "last":
        if tokenizer.eos_token_id is not None:
            boundary_token_ids = [tokenizer.eos_token_id]

    # Token mask function for covariance collection
    def token_mask_fn(input_ids):
        return compute_token_mask(
            input_ids,
            token_selection="all" if needs_covariance else cfg.token_selection,
            skip_positions=cfg.skip_positions,
            boundary_token_ids=boundary_token_ids,
        )

    # Main loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    executor = ThreadPoolExecutor(max_workers=1)

    print(f"Processing {len(to_process)} remaining checkpoints...")
    print(f"  collect_residual={cfg.collect_residual} ({cfg.residual_hook_point})")
    print(f"  collect_A={cfg.collect_A}, collect_G={cfg.collect_G}, collect_B={cfg.collect_B}")
    print(f"  packing={cfg.packing}, token_selection={cfg.token_selection}")
    print(f"  sample_labels={cfg.sample_labels}, label_samples={cfg.label_samples}, seed={cfg.seed}")
    print(f"  storage_format={cfg.storage_format}")

    for idx, (step_num, revision, step_model) in enumerate(tqdm(to_process, desc="Checkpoints")):
        # Prefetch next
        if idx + 1 < len(to_process):
            _, next_rev, next_model = to_process[idx + 1]
            prefetch_future = executor.submit(prefetch_checkpoint, next_model, next_rev)
        else:
            prefetch_future = None

        try:
            model = load_model(model_config, step_model, revision)
            model.to(device)

            needs_backward = cfg.collect_G
            if needs_backward:
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                model.config.use_cache = False
                model.train()
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = 0.0

            n_layers = get_num_layers(model, model_config)
            blocks = cfg.target_blocks if cfg.target_blocks is not None else list(range(n_layers))

            # --- Collect residual activations ---
            if cfg.collect_residual:
                residual_path = os.path.join(output_dir, f"step{step_num}.npy")
                if not os.path.exists(residual_path):
                    acts = _collect_residual_for_checkpoint(
                        model, model_config, texts, tokenizer,
                        cfg.residual_hook_point, cfg.token_selection,
                        cfg.max_length, cfg.batch_size,
                        cfg.packing, packed_ids, boundary_token_ids,
                    )
                    if acts is not None:
                        save_activations(acts, residual_path)
                        tqdm.write(f"Step {step_num}: residual {acts.shape} -> {residual_path}")

            # --- Collect covariance factors ---
            if needs_covariance:
                cov_path = os.path.join(output_dir, f"step{step_num}.pt")
                if not os.path.exists(cov_path):
                    if cfg.sample_labels and cfg.seed is not None:
                        torch.manual_seed(cfg.seed + step_num)

                    if cfg.packing == "packed":
                        factors = _collect_covariance_for_checkpoint(
                            model, model_config, packed_ids, blocks,
                            cfg.max_layers_per_pass,
                            cfg.batch_size, cfg.sample_labels, cfg.label_samples,
                            device, cfg.collect_A, cfg.collect_G, cfg.collect_B,
                            token_mask_fn,
                        )
                    else:
                        factors = _collect_covariance_padded_for_checkpoint(
                            model, model_config, texts, tokenizer, blocks,
                            cfg.max_layers_per_pass,
                            cfg.batch_size, cfg.max_length,
                            cfg.sample_labels, cfg.label_samples,
                            device, cfg.collect_A, cfg.collect_G, cfg.collect_B,
                            cfg.token_selection,
                        )
                    save_factors(factors, cov_path, cfg.storage_format, cfg.cross_basis_refs)
                    tqdm.write(f"Step {step_num}: {len(factors)} projections -> {cov_path}")

        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        if prefetch_future is not None:
            prefetch_future.result()
        delete_cached_revision(step_model, revision)

    executor.shutdown(wait=True)
    print(f"Completed! Output saved to {output_dir}")


if __name__ == "__main__":
    from jsonargparse import CLI
    cfg = CLI(CollectConfig)
    main(cfg)
