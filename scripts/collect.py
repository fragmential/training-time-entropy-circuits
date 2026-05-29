#!/usr/bin/env python
"""Unified collection pipeline for activations and covariance factors.

Collects data at arbitrary hook points using a single configurable script
driven by YAML config files. All hook points go through the same HookCollector
and produce the same output format.

Hook selection is pattern-based via the `hooks` config field (and the
orthogonal `grad_all` bool). Patterns use fnmatch wildcards. Append `+G` to
any entry to also collect G at those matches (triggers backward pass).

Examples of valid patterns:
    blk*.up                   # all MLP up-projections, A only
    blk*.down+G               # MLP down-proj, A + G (K-FAC)
    blk*.attn.in              # residual into each block's attention sub-block
    blk*.attn.out             # attention's contribution to residual
    blk*.mlp.out              # MLP's contribution
    blk*.attn.raw_out         # (OLMo-2 only) pre-post-norm attention output
    blk*.mlp.in               # (OLMo-2 only) residual into MLP sub-block
    blk*.attn.head*           # per-OV-head contributions, all heads
    blk3.attn.head0           # just one head
    after_final_norm          # post-final-norm residual
    before_final_norm         # pre-final-norm residual
    identity_head             # fast post-norm via head replacement

Set `grad_all: true` as a shortcut to grad on every matched hook without
writing "+G" on each entry. Pattern matches are family-aware: patterns
matching only Pythia-absent names (e.g. blk*.gate, blk*.mlp.in) are silent
no-ops on Pythia.

Legacy boolean flags (collect_A / collect_G / collect_final_acts /
collect_final_grads / residual_hook_point) are still accepted; when `hooks`
is unset they desugar to the equivalent pattern list with inline `+G`.

Other features:
    - Data modes: packed (no padding) or padded
    - Token selection: all tokens or last token (per sequence or per document)
    - Storage formats: acts, cov, cov_svd, eigenvalues (with +m modifier)
    - Answer-only gradient collection for downstream tasks
    - Cross-basis projections at save time

Usage:
    python scripts/collect.py --config configs/reproduce_rankme_alpha.yaml --model_name EleutherAI/pythia-14m
    python scripts/collect.py --config configs/full.yaml --array_id 0
"""

import gc
import os
import sys
import time
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Disable tqdm nested-bar cursor movement when stderr is not a terminal (e.g. SLURM logs)
# This prevents [A escape codes from cluttering log files.
_IS_TTY = sys.stderr.isatty()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_registry import (
    get_model_config,
    get_checkpoint_schedule,
    load_model,
    load_tokenizer,
    get_num_layers,
    prefetch_checkpoint,
    delete_cached_revision,
)
from utils.hooks import HookCollector, MultiHeadOVDispatcher, setup_identity_head, restore_head
from utils.hook_specs import resolve as resolve_hook_specs, synthesize_from_flags
from utils.accessor import DataAccessor, parse_format, parse_format_spec
from utils.data_utils import (
    get_loader,
    load_and_cache_texts,
    pack_sequences,
    compute_token_mask,
    compute_labels,
)


# Fields that may vary per model in a sweep (accept scalar, list, or dict).
VECTORIZABLE_FIELDS = {"batch_size", "max_checkpoints", "max_layers_per_pass", "dataset_name",
                       "accumulation_dtype", "activation_dtype", "packed_data_path"}


def _resolve_wildcard_dict(d: dict, model_name: str, default=None):
    """Look up model_name in a dict that may contain wildcard keys (ending with *).

    Resolution order:
        1. Exact match
        2. Longest matching wildcard prefix (e.g. "EleutherAI/pythia-14m*" beats "EleutherAI/*")
        3. default
    """
    if model_name in d:
        return d[model_name]
    best_val, best_len = default, -1
    for key, val in d.items():
        if key.endswith("*"):
            prefix = key[:-1]
            if model_name.startswith(prefix) and len(prefix) > best_len:
                best_val, best_len = val, len(prefix)
    return best_val


@dataclass
class CollectConfig:
    # --- Model ---
    model_name: "str | list[str]" = "EleutherAI/pythia-14m"
    max_checkpoints: "int | dict[str, int] | list[int]" = 50
    checkpoint_spacing: str = "linear"  # "linear" or "log" or "sqrt" for non-early checkpoint subsampling
    checkpoints: "list[int] | list[list[int]] | dict | str | None" = None
    target_layers: "list | dict[str, list] | None" = None
    max_layers_per_pass: "int | dict[str, int] | list[int]" = 4

    # --- Data ---
    dataset_name: "str | dict[str, str] | list[str]" = "fineweb"
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
    max_bytes: "int | None" = None  # if set, replaces num_samples as data budget (e.g. 80_000_000)
    max_tokens: "int | None" = None  # if set, limit packed data to this many tokens (rounds up to full chunks)
    packed_data_path: "str | dict[str, str] | list[str] | None" = None  # path to pre-built .pt of shape (n_chunks, seq_len)

    # --- What to collect ---
    # Preferred: glob patterns matching hook names directly. Append "+G" to any
    # entry to also collect G at the matching hooks (triggers backward pass).
    # Examples:
    #   - "blk*.up"            (MLP up-proj A, forward only)
    #   - "blk*.down+G"        (MLP down-proj A AND G — K-FAC)
    #   - "blk*.attn.in"       (residual into each block's attn sub-block)
    #   - "blk*.attn.head*"    (per-OV-head contributions, all heads, all blocks)
    #   - "blk3.attn.head0"    (just one head)
    #   - "after_final_norm"   (residual after final norm)
    #   - "identity_head"      (fast post-norm via head replacement)
    # Set `grad_all: true` as a shortcut to enable G on every matched hook
    # without writing "+G" on each entry.
    hooks: "list | None" = None
    grad_all: bool = False

    # Legacy convenience flags (kept for backward compat with existing configs).
    # If `hooks` is None they are desugared into a `hooks` list; otherwise ignored.
    collect_final_acts: bool = True
    collect_final_grads: bool = False
    residual_hook_point: str = "identity_head"  # or "both" for before + after final norm
    collect_A: bool = False
    collect_G: bool = False
    sample_labels: bool = True
    label_samples: int = 1
    seed: int = 42

    # --- Answer-only mode ---
    answer_only: bool = False
    answer_start_key: "str | None" = None

    # --- Precision ---
    accumulation_dtype: "str | dict[str, str] | list[str]" = "fp64"
    activation_dtype: "str | dict[str, str] | list[str]" = "fp32"

    # --- Storage ---
    storage_format: str = "cov"
    storage_dtype: "str | None" = None  # None → smart defaults (eigvals:fp64, eigvecs/acts/cov:fp32)
    cross_basis_refs: "list | None" = None
    output_dir: "str | None" = None

    # --- Metrics ---
    compute_metrics: bool = False  # compute spectral metrics inline while model is loaded

    # --- Cache ---
    keep_cached: bool = False  # don't delete HF checkpoints after processing

    # Extra gitignore-style globs (state-dict-key patterns) added on top of the
    # per-family DEFAULT_WEIGHT_CACHE_GLOBS in utils.model_registry. A leading '!'
    # negates a default include. Persisted to .pt metadata so downstream
    # compute_metrics / convert see the same set.
    weight_cache_patterns: "list | None" = None

    # --- Continue ---
    continue_from: "str | None" = None  # path to a previous run's output_dir to continue from

    # --- Profiling ---
    profile_vram: bool = False  # print peak CUDA memory after collection

    # --- Sweep ---
    array_id: "int | None" = None

    def _resolved_hooks(self) -> tuple:
        """Return (hooks_list, grad_all) — explicit values or desugared from legacy flags.

        `hooks_list` may contain entries with trailing '+G' (per-pattern grad markers);
        `grad_all` (bool) is an orthogonal shortcut meaning "G on every matched hook".
        """
        if self.hooks is not None:
            return list(self.hooks), bool(self.grad_all)
        return synthesize_from_flags(
            collect_A=self.collect_A,
            collect_G=self.collect_G,
            collect_final_acts=self.collect_final_acts,
            collect_final_grads=self.collect_final_grads,
            residual_hook_point=self.residual_hook_point,
        ), False

    def __post_init__(self):
        _LIST_FIELDS = {"target_layers", "boundary_token_ids", "cross_basis_refs", "checkpoints", "hooks"}

        if not isinstance(self.model_name, list):
            # Single model — validate no vectorized fields are lists
            for name in VECTORIZABLE_FIELDS:
                val = getattr(self, name)
                if isinstance(val, (list, dict)):
                    raise ValueError(
                        f"{name} is vectorized but model_name is a single string. "
                        f"Use a model_name list or pass a scalar {name}."
                    )
            # Resolve dict fields for single model (with wildcard support)
            if isinstance(self.target_layers, dict):
                self.target_layers = _resolve_wildcard_dict(
                    self.target_layers, self.model_name)
            if isinstance(self.checkpoints, dict):
                self.checkpoints = _resolve_wildcard_dict(
                    self.checkpoints, self.model_name)
            return

        # Non-vectorizable fields must not be lists (exempting known list fields)
        for f in fields(self):
            if f.name not in VECTORIZABLE_FIELDS and f.name != "model_name":
                val = getattr(self, f.name)
                if isinstance(val, list) and f.name not in _LIST_FIELDS:
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

        # checkpoints: list[list|str] → vectorized dict, list[int] → same for all (leave as-is)
        if isinstance(self.checkpoints, list) and self.checkpoints and isinstance(self.checkpoints[0], (list, str)):
            if len(self.checkpoints) != len(model_list):
                raise ValueError(
                    f"checkpoints list-of-lists length ({len(self.checkpoints)}) "
                    f"!= model_name list length ({len(model_list)})"
                )
            self.checkpoints = dict(zip(model_list, self.checkpoints))

        # If array_id is set, resolve to a single model
        if self.array_id is not None:
            model = model_list[self.array_id]
            for name in VECTORIZABLE_FIELDS:
                val = getattr(self, name)
                if isinstance(val, dict):
                    setattr(self, name, _resolve_wildcard_dict(val, model))
            if isinstance(self.target_layers, dict):
                self.target_layers = _resolve_wildcard_dict(self.target_layers, model)
            if isinstance(self.checkpoints, dict):
                self.checkpoints = _resolve_wildcard_dict(self.checkpoints, model)
            self.model_name = model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _batch_iterator(texts, tokenizer, packed_ids, packing, batch_size, max_length, device):
    """Yield (input_ids, attention_mask_or_None) batches.

    Abstracts the packed vs padded data preparation so the collection loop
    doesn't need to branch.
    """
    if packing == "packed":
        loader = DataLoader(TensorDataset(packed_ids), batch_size=batch_size, shuffle=False)
        for (ids_batch,) in loader:
            yield ids_batch.to(device, non_blocking=True), None
    else:
        for bidx in range(0, len(texts), batch_size):
            batch = texts[bidx : bidx + batch_size]
            tokenized = tokenizer(
                batch, padding="longest", return_tensors="pt",
                max_length=max_length, truncation=True,
            )
            yield (
                tokenized.input_ids.to(device),
                tokenized.attention_mask.to(device),
            )


# ---------------------------------------------------------------------------
# Unified collection
# ---------------------------------------------------------------------------

def _build_single(spec, storage_mode, collect_means, cfg):
    """Build a HookCollector from a SingleHookSpec."""
    return HookCollector(
        module=spec.module,
        capture=spec.capture or "output",
        mode=storage_mode,
        collect_grad=spec.collect_grad,
        collect_means=collect_means,
        accumulation_dtype=cfg.accumulation_dtype,
        activation_dtype=cfg.activation_dtype,
        grad_capture=spec.grad_capture,
        token_selection=spec.token_selection,
    )


def _build_ov(spec, storage_mode, collect_means, cfg):
    return MultiHeadOVDispatcher(
        o_proj_module=spec.o_proj,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        block_idx=spec.block_idx,
        selected_heads=spec.selected_heads,
        mode=storage_mode,
        collect_means=collect_means,
        collect_grad=spec.collect_grad,
        accumulation_dtype=cfg.accumulation_dtype,
        activation_dtype=cfg.activation_dtype,
        token_selection=spec.token_selection,
    )


def _collect_for_checkpoint(
    model, model_config, cfg, texts, tokenizer, packed_ids,
    target_layers, device, boundary_token_ids,
):
    """Collect all requested data for one checkpoint.

    Resolves cfg.hooks (or the legacy-flag-derived equivalent) into concrete
    SingleHookSpec / OVHeadSpec lists, registers HookCollectors per block-group
    pass, runs forward (+ optional backward), returns the merged factors dict.
    """
    hook_patterns, grad_all = cfg._resolved_hooks()
    storage_base, storage_flags, storage_drop_flags = parse_format_spec(cfg.storage_format)
    storage_mode = "acts" if storage_base == "acts" else "cov"
    if "b" in storage_flags and "m" not in storage_flags and "m" not in storage_drop_flags:
        print("  NOTE: +b implies +m — storing means for B derivation")
    collect_means = (
        "m" not in storage_drop_flags
        and ("m" in storage_flags or "b" in storage_flags or storage_base in ("cov_svd", "eigenvalues"))
    )

    # Resolve patterns once against the model's hook universe
    single_specs, ov_specs, _candidates = resolve_hook_specs(
        model, model_config, target_layers,
        hook_patterns, grad_all,
        default_token_selection=cfg.token_selection,
    )

    needs_grad = any(s.collect_grad for s in single_specs) or any(o.collect_grad for o in ov_specs)
    has_per_block_work = any(not s.is_global for s in single_specs) or bool(ov_specs)
    has_identity_head = any(s.module is None and s.name == "identity_head" for s in single_specs)
    if has_identity_head and any(s.name == "identity_head" and s.collect_grad for s in single_specs):
        raise ValueError(
            "identity_head cannot collect gradients (it replaces the head with Identity); "
            "use after_final_norm / before_final_norm patterns for grad-aware residual collection"
        )

    global_specs = [s for s in single_specs if s.is_global]
    perblk_specs = [s for s in single_specs if not s.is_global]
    perblk_by_block: dict = {}
    for s in perblk_specs:
        b = int(s.name.split(".")[0][3:])
        perblk_by_block.setdefault(b, []).append(s)
    ov_by_block = {o.block_idx: o for o in ov_specs}

    all_factors: dict = {}

    # Block-group iteration only matters when there is per-block work
    if has_per_block_work:
        if cfg.max_layers_per_pass == 0:
            block_groups = [target_layers]
        else:
            block_groups = list(chunked(target_layers, cfg.max_layers_per_pass))
    else:
        block_groups = [None]

    # --- Global collectors (residual / final-norm) — registered once ---
    identity_head_state = None
    residual_collector_for_id = None
    global_collectors: dict = {}
    for spec in global_specs:
        if spec.module is None:
            # identity_head: swap output head for nn.Identity() and manually feed
            identity_head_state = setup_identity_head(model)
            rc = HookCollector(
                module=None, mode=storage_mode,
                collect_means=collect_means,
                accumulation_dtype=cfg.accumulation_dtype,
                activation_dtype=cfg.activation_dtype,
                token_selection=spec.token_selection,
            )
            global_collectors["after_final_norm"] = rc  # identity_head stores under this canonical key
            if residual_collector_for_id is None:
                residual_collector_for_id = rc
        else:
            global_collectors[spec.name] = _build_single(spec, storage_mode, collect_means, cfg)

    for blk_group in block_groups:
        first_pass = blk_group is block_groups[0] if block_groups else True
        # Globals are registered once (first pass); per-block specs each pass
        collectors = dict(global_collectors) if first_pass else {}
        ov_dispatchers = []

        if blk_group is not None:
            for p in model.parameters():
                p.requires_grad_(False)
            for b in blk_group:
                for spec in perblk_by_block.get(b, []):
                    if spec.collect_grad and spec.module is not None and hasattr(spec.module, "weight"):
                        spec.module.weight.requires_grad_(True)
                    collectors[spec.name] = _build_single(spec, storage_mode, collect_means, cfg)
                ov = ov_by_block.get(b)
                if ov is not None:
                    if ov.collect_grad and hasattr(ov.o_proj, "weight"):
                        ov.o_proj.weight.requires_grad_(True)
                    ov_dispatchers.append(_build_ov(ov, storage_mode, collect_means, cfg))

        # --- Run batches ---
        parts = []
        if first_pass and global_collectors:
            parts.append("+".join(sorted(global_collectors)))
        if blk_group is not None:
            parts.append(f"blk {blk_group}")
        desc = " + ".join(parts) if parts else "Collecting"
        batches = _batch_iterator(
            texts, tokenizer, packed_ids, cfg.packing,
            cfg.batch_size, cfg.max_length, device,
        )
        # Compute total for progress bar
        if cfg.packing == "packed" and packed_ids is not None:
            n_batches = (len(packed_ids) + cfg.batch_size - 1) // cfg.batch_size
        else:
            n_batches = (len(texts) + cfg.batch_size - 1) // cfg.batch_size

        bar_kwargs = dict(desc=desc, total=n_batches)
        if _IS_TTY:
            bar_kwargs.update(leave=False)
        else:
            # Non-TTY: write one clean line per update, no cursor codes
            bar_kwargs.update(leave=True, bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]")

        for input_ids, attention_mask in tqdm(batches, **bar_kwargs):
            # Compute one mask per unique token_selection, cache by selection value
            _mask_cache = {}

            def _mask_for(sel):
                if sel not in _mask_cache:
                    _mask_cache[sel] = compute_token_mask(
                        input_ids, attention_mask=attention_mask,
                        token_selection=sel,
                        skip_positions=cfg.skip_positions,
                        boundary_token_ids=boundary_token_ids,
                    )
                return _mask_cache[sel]

            for collector in collectors.values():
                collector.set_token_mask(_mask_for(collector.token_selection or cfg.token_selection))
            for disp in ov_dispatchers:
                disp.set_token_mask(_mask_for(cfg.token_selection))

            # Forward (+ backward) pass
            fwd_kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                fwd_kwargs["attention_mask"] = attention_mask

            if needs_grad:
                model.zero_grad(set_to_none=True)
                logits = model(**fwd_kwargs).logits[:, :-1].float()

                if cfg.sample_labels:
                    with torch.no_grad():
                        probs = torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1))
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    for i in range(cfg.label_samples):
                        with torch.no_grad():
                            y = torch.multinomial(probs, 1).squeeze(1)
                        loss = nn.functional.cross_entropy(flat_logits, y)
                        last = (i == cfg.label_samples - 1)
                        (loss / cfg.label_samples).backward(retain_graph=not last)
                else:
                    labels = compute_labels(input_ids, attention_mask)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels[:, :-1].reshape(-1),
                        ignore_index=-100,
                    )
                    loss.backward()
            else:
                with torch.no_grad():
                    outputs = model(**fwd_kwargs)

                # For identity_head: manually feed model output to the collector
                if identity_head_state is not None and residual_collector_for_id is not None:
                    acts_masked = residual_collector_for_id._apply_mask(outputs.logits.detach())
                    residual_collector_for_id.accumulate(acts_masked)

        # --- Collect factors and clean up ---
        for cname, collector in collectors.items():
            if cname not in all_factors:
                all_factors[cname] = collector.factors()
            collector.close()
        for disp in ov_dispatchers:
            for cname, fac in disp.factors().items():
                if cname not in all_factors:
                    all_factors[cname] = fac
            disp.close()

        # Restore identity head after first pass
        if first_pass and identity_head_state is not None:
            restore_head(model, *identity_head_state)
            identity_head_state = None

        del collectors, ov_dispatchers
        torch.cuda.empty_cache()

    return all_factors


# ---------------------------------------------------------------------------
# Continue-run helpers
# ---------------------------------------------------------------------------


def _reconstruct_raw_factors(existing_data: dict) -> dict:
    """Reconstruct raw accumulator dicts from a saved .pt file so they can be merged.

    Reads __format__ from the file itself to determine how to decode it.
    """
    fmt_str = existing_data.get("__format__", "cov")
    base_format, _ = parse_format(fmt_str)
    raw = {}
    for hook_name, entry in existing_data.items():
        if hook_name.startswith("__"):
            continue
        raw_entry = {}
        for fk in ("A", "G"):
            n_key = f"n_{fk}"
            if n_key not in entry:
                continue
            n = entry[n_key]
            if base_format == "acts":
                if fk in entry:
                    raw_entry[fk] = entry[fk]  # (N, d) — will concat
            elif base_format == "cov":
                if fk in entry:
                    raw_entry[fk] = entry[fk].double()  # raw unnormalized cov
            elif base_format in ("cov_svd", "acts_svd"):
                vec_key, val_key = f"{fk}_eigvecs", f"{fk}_eigvals"
                if val_key not in entry:
                    continue
                if vec_key not in entry:
                    raise ValueError(
                        f"Cannot continue {hook_name}.{fk}: eigvecs missing "
                        f"(eigenvalues-only format cannot be merged)"
                    )
                V = entry[vec_key].double()    # (d, k)
                lam = entry[val_key].double()  # (k,)
                raw_entry[fk] = (V * lam) @ V.T * n  # unnormalized cov (d, d)
            else:
                raise ValueError(
                    f"Cannot continue with storage format '{base_format}': "
                    f"eigenvalues-only format has no eigvecs to reconstruct from"
                )
            raw_entry[n_key] = n
            mean_key = f"{fk}_mean"
            if mean_key in entry:
                raw_entry[mean_key] = entry[mean_key].float()  # normalized mean (d,)
        raw[hook_name] = raw_entry
    return raw


def _merge_with_existing(existing_raw: dict, new_factors: dict) -> dict:
    """Add existing raw accumulators into new_factors (in-place). Returns new_factors."""
    for hook_name, old in existing_raw.items():
        if hook_name not in new_factors:
            continue
        new = new_factors[hook_name]
        for fk in ("A", "G"):
            n_key = f"n_{fk}"
            if fk not in old or n_key not in old:
                continue
            old_n = old[n_key]
            new_n = new.get(n_key, 0)
            merged_n = old_n + new_n
            if fk in new:
                if old[fk].dim() == 2 and old[fk].shape[0] != old[fk].shape[1]:
                    # Acts mode: concat along token dim
                    new[fk] = torch.cat([old[fk], new[fk]], dim=0)
                else:
                    # Cov mode: add unnormalized accumulators
                    new[fk] = old[fk].to(dtype=new[fk].dtype) + new[fk]
            else:
                new[fk] = old[fk]
            # Merge means (weighted average of normalized means)
            mean_key = f"{fk}_mean"
            if mean_key in old:
                if mean_key in new and new_n > 0:
                    new[mean_key] = (old[mean_key] * old_n + new[mean_key] * new_n) / merged_n
                else:
                    new[mean_key] = old[mean_key]
            new[n_key] = merged_n
        new["n"] = new.get("n_A", 0)
    return new_factors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: CollectConfig):
    if cfg.profile_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model_name = cfg.model_name
    if isinstance(model_name, list):
        raise ValueError(
            "model_name is still a list — pass --array_id to select a model, "
            "or pass a single --model_name."
        )

    model_config = get_model_config(model_name)
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name

    # Resolve "native" dataset to model's training data
    if cfg.dataset_name == "native":
        if model_config.training_dataset is None:
            raise ValueError(
                f"dataset_name='native' but {model_name} has no training_dataset configured "
                f"in model_registry. Set training_dataset or use an explicit dataset_name."
            )
        cfg.dataset_name = model_config.training_dataset

    print(f"Model: {model_name} (family={model_config.family}, dataset={cfg.dataset_name})")

    # Output directory — always include model subfolder
    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = os.path.join("collected", cfg.dataset_name, short_name)
    else:
        output_dir = os.path.join(output_dir, short_name)
    os.makedirs(output_dir, exist_ok=True)

    # Checkpoint schedule — explicit checkpoints list takes priority over max_checkpoints
    if cfg.checkpoints is not None:
        cfg.max_checkpoints = None
    schedule = get_checkpoint_schedule(model_config, cfg.max_checkpoints, cfg.checkpoint_spacing)
    if cfg.checkpoints == "final":
        cfg.checkpoints = [schedule[-1][0]] if schedule else []
    if cfg.checkpoints is not None:
        wanted = set(cfg.checkpoints)
        missing = wanted - {s for s, _, _ in schedule}
        if missing:
            print(f"  WARNING: {len(missing)} requested steps not in schedule: {sorted(missing)}")
        schedule = [t for t in schedule if t[0] in wanted]
    print(f"Total checkpoints: {len(schedule)}")

    # Filter to unprocessed checkpoints
    to_process = []
    for step_num, revision, step_model in schedule:
        out_path = os.path.join(output_dir, f"step{step_num}.pt")
        if os.path.exists(out_path):
            continue
        # Also check for old .npy format (backward compat)
        npy_path = os.path.join(output_dir, f"step{step_num}.npy")
        if os.path.exists(npy_path):
            continue
        to_process.append((step_num, revision, step_model))

    if not to_process:
        print("All checkpoints already collected.")
        return

    # Tokenizer
    first_revision = to_process[0][1]
    tokenizer = load_tokenizer(model_config, revision=first_revision)

    # Load data
    packed_ids = None
    texts = None
    if cfg.packed_data_path and cfg.packing == "packed":
        # Pre-built shuffled mix: load directly
        packed_ids = torch.load(cfg.packed_data_path, map_location="cpu", weights_only=True)
        if cfg.max_tokens:
            n_chunks = (cfg.max_tokens + cfg.seq_len - 1) // cfg.seq_len  # round up
            packed_ids = packed_ids[:n_chunks]
        print(f"Loaded pre-built packed data: {packed_ids.shape[0]} chunks of {packed_ids.shape[1]} tokens"
              f" from {cfg.packed_data_path}")
    else:
        loader_fn = get_loader(cfg.dataset_name)
        texts = load_and_cache_texts(
            loader_fn, cfg.num_samples, cfg.min_length, tokenizer,
            cfg.dataset_name, cfg.dataset_content_key,
            max_bytes=cfg.max_bytes,
        )
        if cfg.packing == "packed":
            packed_ids = pack_sequences(texts, tokenizer, cfg.seq_len)
            print(f"Packed data: {packed_ids.shape[0]} chunks of {cfg.seq_len} tokens")

    # Boundary tokens for packed + last-token mode
    boundary_token_ids = cfg.boundary_token_ids
    if boundary_token_ids is None and cfg.packing == "packed" and cfg.token_selection == "last":
        if tokenizer.eos_token_id is not None:
            boundary_token_ids = [tokenizer.eos_token_id]

    # Continue-from: determine how much data the previous run already processed and skip it
    n_chunks_done = 0  # for packed: chunks already accumulated in existing files
    if cfg.continue_from:
        continue_model_dir = os.path.join(cfg.continue_from, short_name)
        # Find any existing checkpoint to read metadata from
        ref_data = None
        for step_num, _, _ in to_process:
            candidate = os.path.join(continue_model_dir, f"step{step_num}.pt")
            if os.path.exists(candidate):
                ref_data = torch.load(candidate, map_location="cpu", weights_only=False)
                break
        if ref_data is None:
            raise ValueError(
                f"continue_from={cfg.continue_from!r} but no existing checkpoint files found "
                f"in {continue_model_dir} for the steps we need to process."
            )
        old_fmt = ref_data.get("__format__", "cov")
        if old_fmt != cfg.storage_format:
            raise ValueError(f"storage_format must match continued data (existing={old_fmt!r}, new={cfg.storage_format!r})")

        if cfg.packing == "padded":
            # n_A == n_sequences for last-token padded collection
            n_done = next(
                v["n_A"] for k, v in ref_data.items()
                if not k.startswith("__") and "n_A" in v
            )
            print(f"  continue_from: existing run has {n_done} sequences; skipping to texts[{n_done}:]")
            texts = texts[n_done:]
            if len(texts) == 0:
                raise ValueError(f"No new texts to process — existing run already covers all {n_done} sequences.")
        else:
            # Packed: need __n_chunks__ stored by a previous continue-aware run
            n_chunks_done = ref_data.get("__n_chunks__")
            if n_chunks_done is None:
                raise ValueError(
                    "continue_from: existing .pt has no __n_chunks__ metadata. "
                    "The original run must have been collected with this version of collect.py."
                )
            if packed_ids is not None:
                total_avail = len(packed_ids)
                if n_chunks_done >= total_avail:
                    raise ValueError(
                        f"Not enough chunks to continue: existing run used {n_chunks_done} chunks "
                        f"but total data only has {total_avail} chunks."
                    )
                packed_ids = packed_ids[n_chunks_done:]
                print(f"  continue_from: skipping {n_chunks_done} chunks → {len(packed_ids)} new chunks remaining")

    # Main loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    executor = ThreadPoolExecutor(max_workers=1)

    print(f"Processing {len(to_process)} remaining checkpoints...")
    _hooks, _grad_all = cfg._resolved_hooks()
    print(f"  hooks    = {_hooks}")
    print(f"  grad_all = {_grad_all}")
    print(f"  packing={cfg.packing}, token_selection={cfg.token_selection}")
    print(f"  sample_labels={cfg.sample_labels}, label_samples={cfg.label_samples}, seed={cfg.seed}")
    print(f"  storage_format={cfg.storage_format}")

    ckpt_bar_kwargs = dict(desc="Checkpoints")
    if not _IS_TTY:
        ckpt_bar_kwargs.update(bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}]")

    t_load, t_collect, t_save, t_metrics = 0.0, 0.0, 0.0, 0.0

    for idx, (step_num, revision, step_model) in enumerate(tqdm(to_process, **ckpt_bar_kwargs)):
        # Prefetch next
        if idx + 1 < len(to_process):
            _, next_rev, next_model = to_process[idx + 1]
            prefetch_future = executor.submit(prefetch_checkpoint, next_model, next_rev)
        else:
            prefetch_future = None

        model = None
        try:
            _t0 = time.time()
            model = load_model(model_config, step_model, revision)
            model.to(device)

            _hooks_list, _grad_all = cfg._resolved_hooks()
            if _grad_all or any(h.endswith("+G") for h in _hooks_list):
                model.gradient_checkpointing_enable()
                model.enable_input_require_grads()
                model.config.use_cache = False
                model.train()
                for m in model.modules():
                    if isinstance(m, nn.Dropout):
                        m.p = 0.0

            n_layers = get_num_layers(model, model_config)
            blocks = cfg.target_layers if cfg.target_layers is not None else list(range(n_layers))

            if cfg.sample_labels and cfg.seed is not None:
                torch.manual_seed(cfg.seed + step_num)

            t_load += time.time() - _t0; _t0 = time.time()
            factors = _collect_for_checkpoint(
                model, model_config, cfg, texts, tokenizer, packed_ids,
                blocks, device, boundary_token_ids,
            )

            # Merge with existing data if continuing a previous run
            if cfg.continue_from:
                exist_path = os.path.join(cfg.continue_from, short_name, f"step{step_num}.pt")
                if os.path.exists(exist_path):
                    existing_data = torch.load(exist_path, map_location="cpu", weights_only=False)
                    existing_raw = _reconstruct_raw_factors(existing_data)
                    factors = _merge_with_existing(existing_raw, factors)
                else:
                    tqdm.write(f"  WARNING: no existing file to merge at {exist_path}")

            t_collect += time.time() - _t0; _t0 = time.time()
            out_path = os.path.join(output_dir, f"step{step_num}.pt")
            token_filter = {
                "token_selection": cfg.token_selection,
                "skip_positions": cfg.skip_positions,
                "boundary_token_ids": cfg.boundary_token_ids,
            }
            if cfg.answer_only:
                token_filter["answer_only"] = True
                token_filter["answer_start_key"] = cfg.answer_start_key
            # For packed runs: record total chunks so future continuations know where to start
            save_n_chunks = None
            if cfg.packing == "packed" and packed_ids is not None:
                save_n_chunks = n_chunks_done + len(packed_ids)
            acc = DataAccessor(factors, model=model, model_config=model_config,
                               model_name=step_model, revision=revision,
                               weight_cache_patterns=cfg.weight_cache_patterns)
            acc.save(out_path, format=cfg.storage_format,
                     cross_basis_refs=cfg.cross_basis_refs,
                     storage_dtype=cfg.storage_dtype, token_filter=token_filter,
                     n_chunks=save_n_chunks)
            t_save += time.time() - _t0
            tqdm.write(f"Step {step_num}: {len(factors)} hook points -> {out_path}")

            if cfg.compute_metrics:
                _t0 = time.time()
                import numpy as np
                from scripts.compute_metrics import compute_metrics_for_checkpoint
                step_metrics = compute_metrics_for_checkpoint(acc, verbose=True)
                metrics_dir = os.path.join(cfg.output_dir.replace("inferences", "results", 1)
                                           if cfg.output_dir else "data/results")
                os.makedirs(metrics_dir, exist_ok=True)
                metrics_path = os.path.join(metrics_dir, f"results_{short_name}.npy")
                res_dict = {}
                if os.path.exists(metrics_path):
                    try:
                        res_dict = np.load(metrics_path, allow_pickle=True).item()
                    except (OSError, ValueError, TypeError):
                        pass
                res_dict[step_num] = step_metrics
                np.save(metrics_path, res_dict)
                t_metrics += time.time() - _t0
                tqdm.write(f"  Metrics: {len(step_metrics)} hooks -> {metrics_path}")

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
        if not cfg.keep_cached:
            delete_cached_revision(step_model, revision)

    executor.shutdown(wait=True)
    print(f"Completed! Output saved to {output_dir}")

    total = t_load + t_collect + t_save + t_metrics
    if total > 0:
        print(f"\n  Timing: load {t_load:.1f}s, collect {t_collect:.1f}s, save {t_save:.1f}s, metrics {t_metrics:.1f}s (total {total:.1f}s)")

    if cfg.profile_vram and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        print(f"  Peak VRAM: {peak / 2**30:.2f} GiB")
        print(torch.cuda.memory_summary(abbreviated=True))


if __name__ == "__main__":
    from jsonargparse import CLI
    cfg = CLI(CollectConfig)
    main(cfg)
