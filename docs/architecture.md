# Codebase Architecture

## Overview

This project tracks how LLM representations evolve during pretraining through spectral methods (RankMe, alpha/power-law exponent) and K-FAC covariance curvature analysis. The codebase has three main processes: **collection**, **metric computation**, and **storage operations**. Each is described below with its call chain, responsibilities, and boundaries.

---

## The Three Main Processes

### 1. Collection

The central pipeline. Given a config YAML, it orchestrates:
- Loading a dataset (via `data/` loaders + `utils/data_utils.py` for caching, packing, padding)
- Iterating over checkpoints of a model (discovered via `utils/model_registry.py`)
- Running forward (and optionally backward) passes with hooks attached
- Accumulating statistics in `HookCollector` instances (`utils/hooks.py`)
- Converting accumulated data to a storage format and saving to disk (via `DataAccessor.save()` in `utils/accessor.py`)

Key design: HookCollector does the accumulation (covariance or raw acts), but knows nothing about storage formats. `collect.py` wraps raw accumulators in a `DataAccessor` and calls `.save()`, which transforms them into the chosen format (acts/cov/cov_svd/eigenvalues) and writes to disk.

### 2. Metric Computation

A post-processing step, completely decoupled from collection. It:
- Reads stored `.pt` files via `DataAccessor` (`utils/accessor.py`)
- The accessor abstracts away which storage format was used -- it can derive eigenvalues from cov, cov from acts, B from A+weights, etc.
- Computes spectral metrics (RankMe, alpha, K-FAC log-det, generalized eigenvalues) using `scripts/compute_metrics.py`
- Saves results as `.npy` dicts keyed by step and hook name

### 3. Post-hoc Storage Operations

Format conversion, cross-basis projection, and metadata editing on already-collected `.pt` files, via `DataAccessor` and the `python -m utils.accessor` CLI:
- `convert`: implemented as `DataAccessor(input).save(..., format=...)`
- `project`: cross-basis eigenvalue projections (same-layer G<->B or cross-checkpoint)
- `set-filter`: edit token selection metadata
- `info`: inspect file contents

---

## Responsibility Map

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| `scripts/collect.py` | Orchestration: data loading, checkpoint loop, hook setup, batch loop, save | Metric computation (except optional inline), format conversion after the fact |
| `utils/hooks.py` | Real-time accumulation during forward/backward pass (cov or raw acts); `MultiHeadOVDispatcher` for per-OV-head decomposition | Storage format decisions, token mask computation, pattern resolution |
| `utils/hook_specs.py` | Resolves the `hooks` glob patterns (with optional `+G` per-entry grad markers and a `grad_all` shortcut) against the model's family-aware hook universe; produces concrete `SingleHookSpec` / `OVHeadSpec` lists. Also holds the backward-compat shim that turns legacy boolean flags into patterns. | Registering hooks on the model (collect.py does that), accumulation |
| `utils/accessor.py` | Format transformation, disk I/O, CLI post-processing, format-agnostic read with lazy derivation (B from A + MLP weights; O from A + o_proj weight slice for per-OV-head entries), eigendecomposition | Running models for collection |
| `utils/model_registry.py` | Model config, checkpoint discovery, model/tokenizer loading, architecture introspection (per-family block-boundary + per-OV-head module resolution), selective weight loading, single-tensor on-disk weight cache (`data/weight_cache/`) read/write-through inside `_load_selective_tensors`, step-to-token-count | Data loading, metric computation |
| `utils/data_utils.py` | Text loading/caching, packing/padding, token mask computation, label computation | Model loading, storage |
| `data/` loaders | HuggingFace streaming dataset access | Tokenization, caching (that's data_utils) |
| `scripts/compute_metrics.py` | Orchestrates metric computation over checkpoints; parallelization | Collection, storage format conversion |

---

## Key Architectural Observations

1. **HookCollector has two modes** (cov vs acts) and the choice is made by `collect.py` based on `storage_format`. The collector doesn't know about storage formats -- it just accumulates what it's told to.

2. **`utils/accessor.py` handles both read and write.** `DataAccessor` is the unified interface: `.save()` writes in a specific format, property access reads any format and derives what's missing. The accessor also does B-derivation (needing model weights via model_registry's selective loading) and post-norm derivation.

3. **Token masking has a split**: MLP linear hooks (`blk*.up/down/gate`) always use "all" tokens (per-spec, set by the resolver); residual + block-boundary + per-OV-head hooks use the config's `token_selection` (last or all). Per-hook `token_selection` lives on the `SingleHookSpec` / `OVHeadSpec` produced by `hook_specs.resolve`, not on the collector itself.

4. **Hook selection is pattern-based** (introduced 2026-05). `CollectConfig` exposes `hooks: list[str]` of fnmatch patterns, with optional trailing `+G` per entry to mark that pattern for gradient collection. A `grad_all: bool` flag shortcuts "G on every matched hook". `utils/hook_specs.resolve` matches against the model's family-aware hook universe and returns concrete specs. Legacy boolean flags (`collect_A` / `collect_G` / `collect_final_acts` / `collect_final_grads` / `residual_hook_point`) are still accepted and desugared into a `hooks` list (with inline `+G`) by `synthesize_from_flags` — existing configs work unchanged.

5. **The data loaders in `data/` are thin** -- they just return HF streaming iterators. All the intelligence (caching, filtering, packing) is in `data_utils.py`.

6. **`compute_metrics.py` has two parallelization modes**: simple multiprocessing (when no model weights needed) and threaded derive mode (when B-derivation or post-norm derivation needs selective weight loading).

---

## Process 1: Collection

```
slurm/collect.sh
  └─▶ scripts/collect.py
        ├─▶ data/__init__.py
        │     └─▶ data/<dataset>_loader.py    (fineweb, pile, olmomix, etc.)
        ├─▶ utils/data_utils.py
        ├─▶ utils/model_registry.py
        ├─▶ utils/hook_specs.py               (resolve cfg.hooks / grad_all against model)
        │     └─▶ utils/model_registry.py
        ├─▶ utils/hooks.py                    (HookCollector, MultiHeadOVDispatcher)
        ├─▶ utils/accessor.py                 (DataAccessor.save() + optional derive/inline metrics)
        │     └─▶ utils/model_registry.py
        └─▶ scripts/compute_metrics.py        (optional: inline metrics)
              └─▶ (see Process 2)
```

## Process 2: Metric Computation

```
slurm/compute_metrics.sh
  └─▶ scripts/compute_metrics.py
        └─▶ utils/accessor.py
              └─▶ utils/model_registry.py     (selective weight loading for B derivation)
```

## Process 3: Storage Operations

```
slurm/storage.sh
  └─▶ utils/accessor.py  (python -m utils.accessor)
```

---

## Shared Dependencies

```
                    scripts/                          utils/
              ┌─────────────────┐            ┌──────────────────────┐
              │  collect.py     │───────────▶│  model_registry.py   │
              │                 │            │  hooks.py            │
              │                 │───────────▶│  accessor.py         │
              │                 │            │  data_utils.py       │
              └────────┬────────┘            └──────────────────────┘
                       │
              ┌────────┴────────┐
              │compute_metrics.py│──────────▶│  accessor.py         │
              └─────────────────┘            └──────────────────────┘

              utils/ internal dependencies:
              ┌─────────────────────────────────────────────────────┐
              │  accessor.py ──────▶ model_registry.py             │
              │                      (selective weight loading)    │
              │                                                     │
              │  hook_specs.py ────▶ model_registry.py             │
              │                      (block boundaries, o_proj)    │
              │                                                     │
              │  hooks.py ─────────▶ (no utils/ dependencies)      │
              │  data_utils.py ────▶ (no utils/ dependencies)      │
              │  model_registry.py ▶ (no utils/ dependencies)      │
              └─────────────────────────────────────────────────────┘

              utils/revisions/ contains 1b_revisions.txt, 7b_revisions.txt
```

---

## Storage Format Lifecycle

```
Collection                    Post-hoc conversion              Metric computation
─────────                     ────────────────────             ──────────────────

HookCollector                 accessor.py CLI                  DataAccessor
accumulates:                  convert command:                 reads any format,
  - raw acts (N,d)              acts -> cov                   derives what's missing:
  - cov matrix (d,d)           cov -> cov_svd                  cov_svd -> eigvals
  - sample count n             cov_svd -> eigenvalues          cov -> eigh -> eigvals
        │                      acts -> acts_svd                acts -> PCA -> eigvals
        ▼                      (reverse paths too)             A + W -> B (derive)
DataAccessor.save()                   │                               │
transforms to                         ▼                               ▼
storage format:               .pt files on disk               spectral_metrics()
  acts, cov, cov_svd,        (same format, different          rankme, alpha, kfac
  eigenvalues, acts_svd        compression level)
  (+m for means,
   +b for B derivation)
```
