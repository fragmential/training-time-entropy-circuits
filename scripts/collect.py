#!/usr/bin/env python
"""Unified collection pipeline for activations and gradients at leaf hook points.

Every hook is a leaf — one physical capture-point — selected via the `hooks`
config field. Each entry is `<leaf-pattern>[:acts|:grads|:both]` (default `:acts`)
or `preset:<name>`. `:quantity` means the same thing at every leaf: capture acts
and/or grads there (`:both`/`:grads` trigger a backward pass).

Examples of valid entries:
    preset:kfac               # blk*.mlp.{up,down,gate}.in:acts + .out:grads
    blk*.mlp.up.in:acts       # MLP up-proj input
    blk*.mlp.down.out:grads   # MLP down-proj output gradient
    blk*.attn.in              # residual into each block's attention sub-block
    blk*.attn.out:both        # attention's contribution, acts + grads
    blk*.mlp.out              # MLP's contribution
    blk*.attn.raw_out         # (OLMo-2 only) pre-post-norm attention output
    blk*.mlp.in               # (OLMo-2 only) residual into MLP sub-block
    blk*.attn.head*.slice     # per-OV-head pre-W_o slice, all heads
    after_final_norm          # post-final-norm residual
    before_final_norm:both    # pre-final-norm residual, acts + grads

Pattern matches are family-aware: patterns matching only Pythia-absent leaves
(e.g. blk*.mlp.gate.in, blk*.mlp.in) are silent no-ops on Pythia. An MLP
projection node (`blk*.up`, `blk*.mlp.up`) is NOT a leaf and hard-errors.

`fast_final_norm: true` captures after_final_norm acts by swapping the output
head for nn.Identity() (cheaper than a norm hook). It breaks the backward pass,
so it errors if any hook needs grads.

Other features:
    - Data modes: packed (no padding) or padded
    - Token selection: all tokens or last token (per sequence or per document)
    - Storage formats: acts, cov, cov_svd, eigenvalues (with +m modifier)
    - Cross-basis projections at save time

Usage:
    python scripts/collect.py --config configs/reproduce_rankme_alpha.yaml --model_name EleutherAI/pythia-14m
    python scripts/collect.py --config configs/full.yaml --array_id 0
"""

import gc
import os
import re
import sys
import tempfile
import time
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from typing import cast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Disable tqdm nested-bar cursor movement when stderr is not a terminal (e.g. SLURM logs)
# This prevents [A escape codes from cluttering log files.
_IS_TTY = sys.stderr.isatty()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_registry import (
    get_model_config,
    get_checkpoint_schedule,
    get_block_boundary_hooks,
    load_model,
    load_tokenizer,
    get_num_layers,
    prefetch_checkpoint,
    delete_cached_revision,
    ModelWeights,
)
from utils.hooks import HookCollector, MultiHeadOVDispatcher, setup_identity_head, restore_head
from utils.hook_specs import resolve as resolve_hook_specs
from utils.accessor import DataAccessor
from utils.data_utils import (
    get_loader,
    load_and_cache_texts,
    pack_sequences,
    compute_token_mask,
    compute_labels,
)


# Fields that may vary per model in a sweep (accept scalar, list, or dict).
VECTORIZABLE_FIELDS = {"batch_size", "max_checkpoints", "max_layers_per_pass", "dataset_name",
                       "accumulation_dtype", "activation_dtype", "packed_data_path", "max_tokens",
                       "num_samples", "drift_metrics"}


def _blk_tag(g):
    contiguous = list(g) == list(range(g[0], g[-1] + 1))
    return (f"blk{g[0]}" if len(g) == 1 else
            f"blk{g[0]}-{g[-1]}" if contiguous else "blk" + "+".join(map(str, g)))


def _zero_output(_mod, _inp, out):
    """Forward hook ablating a residual write: replaces the module output with zeros
    (tuple-returning modules, e.g. Pythia attention, keep their non-tensor extras)."""
    return ((torch.zeros_like(out[0]), *out[1:]) if isinstance(out, tuple)
            else torch.zeros_like(out))


def _mean_output(_mod, _inp, out):
    """Forward hook mean-ablating a residual write: replaces the module output with its
    mean over all token positions in the batch (broadcast back), keeping the write's mean
    contribution but removing its fluctuation. Batch token counts here (~1e4–1e5) make the
    per-batch mean numerically indistinguishable from the write's global mean."""
    def _m(t):
        return t.mean(dim=tuple(range(t.dim() - 1)), keepdim=True).expand_as(t)
    return (_m(out[0]), *out[1:]) if isinstance(out, tuple) else _m(out)


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
    num_samples: "int | dict[str, int] | list[int]" = 5000
    text_shuffle_seed: "int | None" = None  # shuffle the text stream (padded runs otherwise sample the dataset HEAD)
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
    max_tokens: "int | None | dict[str, int] | list[int]" = None  # if set, limit packed data to this many tokens (rounds up to full chunks)
    packed_data_path: "str | dict[str, str] | list[str] | None" = None  # path to pre-built .pt of shape (n_chunks, seq_len)

    # --- What to collect ---
    # Glob leaf-patterns, each `<leaf>[:acts|:grads|:both]` (default :acts) or
    # `preset:<name>`. See module docstring. Examples:
    #   - "preset:kfac"            (MLP K-FAC: in:acts + out:grads for every projection)
    #   - "blk*.mlp.up.in:acts"    (MLP up-proj input)
    #   - "blk*.attn.in"           (residual into each block's attn sub-block, acts)
    #   - "blk*.attn.out:both"     (attention's contribution, acts + grads)
    #   - "blk*.attn.head*.slice"  (per-OV-head pre-W_o slice, all heads, all blocks)
    #   - "after_final_norm"       (residual after final norm)
    hooks: "list | None" = None
    # Capture after_final_norm acts via output-head replacement (forward-only;
    # errors if any hook needs grads).
    fast_final_norm: bool = False
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
    # Accumulator mode ("acts"|"cov"); None → from storage_format. "acts" + a small
    # storage_format enables in-memory cross metrics (such runs can't be continue_from'd).
    collect_format: "str | None" = None
    storage_dtype: "str | None" = None  # None → smart defaults (eigvals:fp64, eigvecs/acts/cov:fp32)
    cross_basis_refs: "list | None" = None
    output_dir: "str | None" = None

    # --- Metrics ---
    compute_metrics: bool = False  # compute spectral metrics inline while model is loaded
    # Inline drift metrics vs the previous checkpoint: spills each checkpoint's raw data to
    # $TMPDIR and mmap-reloads it next iteration. NOTE: $TMPDIR is RAM-backed tmpfs on the
    # GPU nodes, so the spill counts against the job's memory — 7B-scale runs leave this off.
    drift_metrics: "bool | dict[str, bool] | list[bool]" = False
    # Per-layer vocabulary-entropy lens (utils/entropy_lens.py): mean next-token-distribution
    # entropy AND its alphaReQ (rank-11–100 slope of the sorted probabilities) at each residual
    # depth, computed teacher-forced on the collection batches.
    # Saved into the results file under node "vocab_entropy" (requires compute_metrics).
    vocab_entropy: bool = False
    # Leave-one-out groups for the inline `loo` metric: list of block-index lists, or
    # "auto" = every block + the four L/4 chunks + the middle half.
    loo_groups: "list[list[int]] | str | None" = None
    metrics_skip: "list[str] | None" = None  # metric names get_metrics should skip

    # --- Intervention ---
    # Interventions: each entry is a block-index list whose residual writes are zeroed
    # during the forward pass. All interventions run per checkpoint (model loaded once),
    # each writing to <output_dir>_<blk-tag>/<model>.
    ablate: "list[list[int]] | None" = None
    ablate_mode: str = "zero"  # "zero" = write→0; "mean" = write→its batch-token mean

    # --- Cache ---
    keep_cached: bool = False  # don't delete HF checkpoints after processing

    # --- Continue ---
    continue_from: "str | None" = None  # path to a previous run's output_dir to continue from

    # --- Profiling ---
    profile_vram: bool = False  # print peak CUDA memory after collection
    profile_metrics: bool = False  # print per-hook eigendecomp prewarm timings
    hf_progress_bars: bool = True  # show transformers/hub weight-loading + download bars; set false to silence

    # --- Sweep ---
    array_id: "int | None" = None

    def __post_init__(self):
        _LIST_FIELDS = {"target_layers", "boundary_token_ids", "cross_basis_refs", "checkpoints", "hooks",
                        "metrics_skip", "loo_groups", "ablate"}

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


class ScalarConfig(CollectConfig):
    """A `CollectConfig` after `__post_init__`/`array_id` resolution: every vectorizable field
    now holds a single scalar, so we re-type them. Obtain via `scalarized(cfg)`."""
    model_name: str
    dataset_name: str
    accumulation_dtype: str
    activation_dtype: str
    packed_data_path: "str | None"
    batch_size: int
    max_layers_per_pass: int
    max_tokens: "int | None"
    num_samples: int
    drift_metrics: bool
    max_checkpoints: "int | None"   # main() clears it to None when an explicit checkpoints list wins


def scalarized(cfg: CollectConfig) -> ScalarConfig:
    """Re-type an already-resolved config so call sites see scalar (not union) field types.
    Resolution happened in __post_init__; this is just a checked cast of the same object."""
    return cast(ScalarConfig, cfg)


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
        capture=spec.capture,
        quantities=spec.quantities,
        mode=storage_mode,
        collect_means=collect_means,
        accumulation_dtype=cfg.accumulation_dtype,
        activation_dtype=cfg.activation_dtype,
        token_selection=spec.token_selection,
        leaf=spec.leaf,
    )


def _build_ov(spec, storage_mode, collect_means, cfg):
    return MultiHeadOVDispatcher(
        o_proj_module=spec.o_proj,
        num_heads=spec.num_heads,
        head_dim=spec.head_dim,
        block_idx=spec.block_idx,
        selected_heads=spec.selected_heads,
        quantities=spec.quantities,
        mode=storage_mode,
        collect_means=collect_means,
        accumulation_dtype=cfg.accumulation_dtype,
        activation_dtype=cfg.activation_dtype,
        token_selection=spec.token_selection,
    )


def _prepare_model_for_grad(model):
    """Enable the gradient path for collection. enable_input_require_grads is what
    lets input-side hooks (attn.in, mlp.in, residual) see gradients at all — without
    it those points silently collect no grads."""
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0


def _parse_storage_format(spec: str) -> tuple[str, tuple[str, ...]]:
    """Split a storage-format spec into (base, overrides). +b/+o add the derived-leaf
    families (MLP_OUT / HEAD_CONTRIB), -b/-o drop them; legacy +m/-m are obsolete
    (means are in every format)."""
    m = re.match(r"([^+-]+)((?:[+-][^+-]+)*)$", spec)
    if not m:
        raise ValueError(f"Unknown storage format: {spec!r}")
    base, mods = m.groups()
    overrides = tuple(f"{sign}{ch}" for sign, chars in re.findall(r"([+-])([^+-]+)", mods)
                      for ch in chars if ch in ("b", "o"))
    return base, overrides


def _collect_for_checkpoint(
    model, model_config, cfg, texts, tokenizer, packed_ids,
    target_layers, device, boundary_token_ids,
):
    """Collect all requested data for one checkpoint.

    Resolves cfg.hooks into concrete SingleHookSpec / OVHeadSpec lists, registers
    HookCollectors per block-group pass, runs forward (+ optional backward),
    returns the merged captured dict.
    """
    storage_base, _ = _parse_storage_format(cfg.storage_format)
    storage_mode = cfg.collect_format or ("acts" if storage_base == "acts" else "cov")
    assert storage_mode in ("acts", "cov"), f"collect_format must be acts|cov, got {storage_mode!r}"
    collect_means = True   # every storage format includes means

    # Resolve patterns once against the model's leaf universe
    single_specs, ov_specs, _candidates = resolve_hook_specs(
        model, model_config, target_layers,
        cfg.hooks, default_token_selection=cfg.token_selection,
    )

    needs_grad = (any("grads" in s.quantities for s in single_specs)
                  or any("grads" in o.quantities for o in ov_specs))
    use_fast_norm = cfg.fast_final_norm and any(s.leaf == "after_final_norm" for s in single_specs)
    if cfg.fast_final_norm and needs_grad:
        raise ValueError(
            "fast_final_norm swaps the output head for nn.Identity(), which breaks the "
            "backward pass; remove fast_final_norm or drop all :grads/:both hooks."
        )
    if needs_grad:
        _prepare_model_for_grad(model)
    has_per_block_work = any(not s.is_global for s in single_specs) or bool(ov_specs)

    global_specs = [s for s in single_specs if s.is_global]
    perblk_specs = [s for s in single_specs if not s.is_global]
    perblk_by_block: dict = {}
    for s in perblk_specs:
        perblk_by_block.setdefault(int(s.leaf.split(".")[0][3:]), []).append(s)
    ov_by_block = {o.block_idx: o for o in ov_specs}

    captured: dict = {}

    # Block-group iteration only matters when there is per-block work
    if has_per_block_work:
        if cfg.max_layers_per_pass == 0:
            block_groups = [target_layers]
        else:
            block_groups = list(chunked(target_layers, cfg.max_layers_per_pass))
    else:
        block_groups = [None]

    # --- Global collectors (residual / final-norm) — registered once ---
    fast_norm_head = None          # saved (head, attr) when fast_final_norm swaps in Identity()
    fast_norm_collector = None     # the after_final_norm collector fed manually
    global_collectors: dict = {}
    for spec in global_specs:
        if use_fast_norm and spec.leaf == "after_final_norm":
            # swap output head for nn.Identity() so the model output IS the
            # post-final-norm residual; fed manually below (forward-only)
            fast_norm_head = setup_identity_head(model)
            rc = HookCollector(
                module=None, capture="output", quantities={"acts"}, mode=storage_mode,
                collect_means=collect_means,
                accumulation_dtype=cfg.accumulation_dtype,
                activation_dtype=cfg.activation_dtype,
                token_selection=spec.token_selection, leaf="after_final_norm",
            )
            global_collectors["after_final_norm"] = rc
            fast_norm_collector = rc
        else:
            global_collectors[spec.leaf] = _build_single(spec, storage_mode, collect_means, cfg)

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
                    if "grads" in spec.quantities and spec.module is not None and hasattr(spec.module, "weight"):
                        spec.module.weight.requires_grad_(True)
                    collectors[spec.leaf] = _build_single(spec, storage_mode, collect_means, cfg)
                ov = ov_by_block.get(b)
                if ov is not None:
                    if "grads" in ov.quantities and hasattr(ov.o_proj, "weight"):
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

        bar_kwargs: dict[str, object] = dict(desc=desc, total=n_batches)
        if _IS_TTY:
            bar_kwargs.update(leave=False)
        else:
            # Non-TTY: write one clean line per update, no cursor codes
            bar_kwargs.update(leave=True, bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]")

        for input_ids, attention_mask in tqdm(batches, **bar_kwargs):  # type: ignore[arg-type]  # dynamic **kwargs vs tqdm's overloaded signature
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

                # fast_final_norm: model output IS the post-norm residual — feed it manually
                if fast_norm_head is not None and fast_norm_collector is not None:
                    fast_norm_collector.feed(outputs.logits.detach())

        # --- Collect captured data and clean up (collectors yield {leaf: {...}}) ---
        for collector in collectors.values():
            for leaf, data in collector.captured().items():
                captured.setdefault(leaf, data)
            collector.close()
        for disp in ov_dispatchers:
            for leaf, data in disp.captured().items():
                captured.setdefault(leaf, data)
            disp.close()

        # Restore the original output head after the first (only) global pass
        if first_pass and fast_norm_head is not None:
            restore_head(model, *fast_norm_head)
            fast_norm_head = None

        del collectors, ov_dispatchers
        torch.cuda.empty_cache()

    return captured


# ---------------------------------------------------------------------------
# Continue-run helpers
# ---------------------------------------------------------------------------


def _entry_quantities(entry: dict):
    """Quantities present in an entry (each has an `{q}_n` count): e.g. acts, grads."""
    return [k[:-2] for k in entry if k.endswith("_n")]


def _reconstruct_captured(existing_data: dict) -> dict:
    """Rebuild collector-style raw accumulators ({q}_cov=Σxxᵀ or {q}_samples, {q}_n,
    {q}_mean) from a saved file so a continued run can merge into them."""
    base_format, _ = _parse_storage_format(existing_data.get("__format__", "cov"))
    raw = {}
    for leaf, entry in existing_data.items():
        if leaf.startswith("__"):
            continue
        re_entry = {}
        for q in _entry_quantities(entry):
            n = entry[f"{q}_n"]
            re_entry[f"{q}_n"] = n
            if base_format == "acts":
                if f"{q}_samples" in entry:
                    re_entry[f"{q}_samples"] = entry[f"{q}_samples"]
            elif base_format == "cov":
                if f"{q}_gram" in entry:
                    re_entry[f"{q}_gram"] = entry[f"{q}_gram"].double()
            elif base_format in ("cov_svd", "acts_svd"):
                if f"{q}_eigvals" not in entry:
                    continue
                if f"{q}_eigvecs" not in entry:
                    raise ValueError(f"Cannot continue {leaf}.{q}: eigvecs missing")
                V, lam = entry[f"{q}_eigvecs"].double(), entry[f"{q}_eigvals"].double()
                re_entry[f"{q}_gram"] = (V * lam) @ V.T * n
            else:
                raise ValueError(f"Cannot continue with storage format '{base_format}'")
            if f"{q}_mean" in entry:
                re_entry[f"{q}_mean"] = entry[f"{q}_mean"].float()
        raw[leaf] = re_entry
    return raw


def _merge_with_existing(existing_raw: dict, new_captured: dict) -> dict:
    """Add existing raw accumulators into new_captured (in-place)."""
    for leaf, old in existing_raw.items():
        if leaf not in new_captured:
            continue
        new = new_captured[leaf]
        for q in _entry_quantities(old):
            old_n, new_n = old[f"{q}_n"], new.get(f"{q}_n", 0)
            gram_k, samp_k = f"{q}_gram", f"{q}_samples"
            if gram_k in old:
                new[gram_k] = old[gram_k].to(new[gram_k].dtype) + new[gram_k] if gram_k in new else old[gram_k]
            elif samp_k in old:
                new[samp_k] = torch.cat([old[samp_k], new[samp_k]], dim=0) if samp_k in new else old[samp_k]
            mean_k = f"{q}_mean"
            if mean_k in old:
                if mean_k in new and new_n > 0:
                    new[mean_k] = (old[mean_k] * old_n + new[mean_k] * new_n) / (old_n + new_n)
                else:
                    new[mean_k] = old[mean_k]
            new[f"{q}_n"] = old_n + new_n
    return new_captured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: CollectConfig):
    if not cfg.hf_progress_bars:
        from huggingface_hub.utils import disable_progress_bars  # type: ignore[attr-defined]  # runtime-exported, missing from stubs
        from transformers.utils import logging as hf_logging
        disable_progress_bars()
        hf_logging.disable_progress_bar()

    if cfg.profile_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    model_name = cfg.model_name
    if isinstance(model_name, list):
        raise ValueError(
            "model_name is still a list — pass --array_id to select a model, "
            "or pass a single --model_name."
        )
    cfg = scalarized(cfg)   # resolution is done; re-type so vectorizable fields read as scalars

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

    # Filter to unprocessed checkpoints (ablation runs: done = every intervention's file exists)
    run_dirs = ([os.path.join(f"{cfg.output_dir}_{_blk_tag(g)}", short_name) for g in cfg.ablate]
                if cfg.ablate else [output_dir])
    to_process = []
    for step_num, revision, step_model in schedule:
        if all(os.path.exists(os.path.join(d, f"step{step_num}.pt"))
               or os.path.exists(os.path.join(d, f"step{step_num}.npy"))  # old format, backward compat
               for d in run_dirs):
            continue
        to_process.append((step_num, revision, step_model))

    if not to_process:
        print("All checkpoints already collected.")
        return

    # Tokenizer
    first_revision = to_process[0][1]
    try:
        tokenizer = load_tokenizer(model_config, revision=first_revision)
    except Exception:   # e.g. OLMo early-training revisions live in a different repo
        tokenizer = load_tokenizer(model_config)

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
            max_bytes=cfg.max_bytes, shuffle_seed=cfg.text_shuffle_seed,
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
            assert texts is not None  # padded ⇒ the else-branch above loaded texts (never the packed_ids path)
            # sample count == n_sequences for last-token padded collection
            n_done = next(
                v[f"{q}_n"]
                for k, v in ref_data.items() if not k.startswith("__")
                for q in _entry_quantities(v)
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
    if cfg.vocab_entropy and not cfg.compute_metrics:
        raise ValueError("vocab_entropy results are saved via the metrics path; set compute_metrics: true")
    drift_spill = (os.path.join(os.environ.get("TMPDIR") or tempfile.gettempdir(),
                                f"drift_prev_{short_name}.pt")
                   if cfg.drift_metrics and cfg.compute_metrics else None)
    if drift_spill and os.path.exists(drift_spill):
        os.remove(drift_spill)      # a stale spill from a dead run is NOT this run's prev

    print(f"Processing {len(to_process)} remaining checkpoints...")
    print(f"  hooks    = {cfg.hooks}")
    print(f"  fast_final_norm = {cfg.fast_final_norm}")
    print(f"  packing={cfg.packing}, token_selection={cfg.token_selection}")
    print(f"  sample_labels={cfg.sample_labels}, label_samples={cfg.label_samples}, seed={cfg.seed}")
    print(f"  storage_format={cfg.storage_format}")

    ckpt_bar_kwargs = dict(desc="Checkpoints")
    if not _IS_TTY:
        ckpt_bar_kwargs.update(bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}]")

    t_load, t_collect, t_save, t_metrics = 0.0, 0.0, 0.0, 0.0

    for idx, (step_num, revision, step_model) in enumerate(tqdm(to_process, **ckpt_bar_kwargs)):  # type: ignore[arg-type]  # dynamic **kwargs vs tqdm's overloaded signature
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
            model.to(device)  # type: ignore[arg-type]  # model is an HF model-class union; .to(str) is valid

            # Grad-run model setup (enable_input_require_grads etc.) now happens inside
            # _collect_for_checkpoint when any hook needs gradients.
            n_layers = get_num_layers(model, model_config)
            blocks = cfg.target_layers if cfg.target_layers is not None else list(range(n_layers))

            t_load += time.time() - _t0
            # One pass per intervention (cfg.ablate), model + checkpoint loaded once;
            # a plain run is the single intervention None.
            for grp in (cfg.ablate or [None]):
                if grp is None:
                    run_dir, g_out = cfg.output_dir, output_dir
                else:
                    run_dir = f"{cfg.output_dir}_{_blk_tag(grp)}"
                    g_out = os.path.join(run_dir, short_name)
                out_path = os.path.join(g_out, f"step{step_num}.pt")
                if grp is not None and os.path.exists(out_path):
                    continue
                os.makedirs(g_out, exist_ok=True)
                ablate_hook = _mean_output if cfg.ablate_mode == "mean" else _zero_output
                handles = [mod.register_forward_hook(ablate_hook)
                           for b in grp or ()
                           for leaf, mod, _cap in get_block_boundary_hooks(model, model_config, b)
                           if leaf.endswith(".out")]

                if cfg.sample_labels and cfg.seed is not None:
                    torch.manual_seed(cfg.seed + step_num)

                lens = None
                if cfg.vocab_entropy:
                    from utils.entropy_lens import EntropyLens
                    lens = EntropyLens(model, model_config)

                _t0 = time.time()
                captured = _collect_for_checkpoint(
                    model, model_config, cfg, texts, tokenizer, packed_ids,
                    blocks, device, boundary_token_ids,
                )
                entropy_result = lens.close() if lens is not None else None
                for h in handles:
                    h.remove()

                # Merge with existing data if continuing a previous run
                if cfg.continue_from:
                    exist_path = os.path.join(cfg.continue_from, short_name, f"step{step_num}.pt")
                    if os.path.exists(exist_path):
                        existing_data = torch.load(exist_path, map_location="cpu", weights_only=False)
                        existing_raw = _reconstruct_captured(existing_data)
                        captured = _merge_with_existing(existing_raw, captured)
                    else:
                        tqdm.write(f"  WARNING: no existing file to merge at {exist_path}")

                t_collect += time.time() - _t0; _t0 = time.time()
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
                run = os.path.basename((run_dir or "default").rstrip("/"))
                base_fmt, overrides = _parse_storage_format(cfg.storage_format)
                acc = DataAccessor(captured, config=model_config,
                                   weights=ModelWeights(model, model_config),
                                   identity=(step_model, revision, run))
                acc.stamp(token_filter=token_filter, n_chunks=save_n_chunks)
                acc.save(out_path, format=base_fmt, overrides=overrides)
                t_save += time.time() - _t0
                tqdm.write(f"Step {step_num}: {len(captured)} hook points -> {out_path}")

                if cfg.compute_metrics:
                    _t0 = time.time()
                    from scripts.compute_metrics import compute_metrics_for_checkpoint, save_step_metrics
                    ctx: dict = ({"prev": DataAccessor(torch.load(drift_spill, mmap=True, weights_only=False)).v}
                                 if drift_spill and os.path.exists(drift_spill) else {})
                    if cfg.loo_groups:
                        ctx["loo_groups"] = cfg.loo_groups
                    if cfg.metrics_skip:
                        ctx["skip"] = cfg.metrics_skip
                    step_metrics = compute_metrics_for_checkpoint(acc, verbose=cfg.profile_metrics, ctx=ctx)
                    if entropy_result is not None:
                        step_metrics["vocab_entropy"] = {"entropy_lens": entropy_result}
                    metrics_dir = (run_dir.replace("inferences", "results", 1)
                                   if run_dir else "data/results")
                    metrics_path = os.path.join(metrics_dir, f"results_{short_name}.npy")
                    save_step_metrics(metrics_path, step_num, step_metrics)
                    if drift_spill:
                        torch.save(acc.data, drift_spill)
                    t_metrics += time.time() - _t0
                    tqdm.write(f"  Metrics: {len(step_metrics)} hooks -> {metrics_path}")
                del captured, acc   # free this sweep's samples BEFORE the next intervention collects
                gc.collect()        # accessor graph is cyclic — without this the buffers linger

        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Release THIS checkpoint's model, samples and resolver caches now — otherwise
            # they stay bound through the next checkpoint's collection (double residency).
            model = captured = acc = None
            gc.collect()
            torch.cuda.empty_cache()

        if prefetch_future is not None:
            prefetch_future.result()
        if not cfg.keep_cached:
            delete_cached_revision(step_model, revision)

    executor.shutdown(wait=True)
    if drift_spill and os.path.exists(drift_spill):
        os.remove(drift_spill)
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
