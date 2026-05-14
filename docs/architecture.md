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

Format conversion, cross-basis projection, and metadata editing on already-collected `.pt` files, via the `python -m utils.accessor` CLI:
- `convert`: acts -> cov -> cov_svd -> eigenvalues (progressive lossy chain)
- `project`: cross-basis eigenvalue projections (same-layer G<->B or cross-checkpoint)
- `set-filter`: edit token selection metadata
- `info`: inspect file contents

---

## Responsibility Map

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| `scripts/collect.py` | Orchestration: data loading, checkpoint loop, hook setup, batch loop, save | Metric computation (except optional inline), format conversion after the fact |
| `utils/hooks.py` | Real-time accumulation during forward/backward pass (cov or raw acts) | Storage format decisions, token mask computation |
| `utils/accessor.py` | Format transformation, disk I/O, CLI post-processing, format-agnostic read with lazy derivation, eigendecomposition | Running models for collection |
| `utils/model_registry.py` | Model config, checkpoint discovery, model/tokenizer loading, architecture introspection, selective weight loading, step-to-token-count | Data loading, metric computation |
| `utils/data_utils.py` | Text loading/caching, packing/padding, token mask computation, label computation | Model loading, storage |
| `data/` loaders | HuggingFace streaming dataset access | Tokenization, caching (that's data_utils) |
| `scripts/compute_metrics.py` | Orchestrates metric computation over checkpoints; parallelization | Collection, storage format conversion |

---

## Key Architectural Observations

1. **HookCollector has two modes** (cov vs acts) and the choice is made by `collect.py` based on `storage_format`. The collector doesn't know about storage formats -- it just accumulates what it's told to.

2. **`utils/accessor.py` handles both read and write.** `DataAccessor` is the unified interface: `.save()` writes in a specific format, property access reads any format and derives what's missing. The accessor also does B-derivation (needing model weights via model_registry's selective loading) and post-norm derivation.

3. **Token masking has a split**: residual hooks use the config's `token_selection` (last or all), while MLP covariance hooks always use "all" tokens. This is enforced in `collect.py`, not in hooks.py.

4. **The data loaders in `data/` are thin** -- they just return HF streaming iterators. All the intelligence (caching, filtering, packing) is in `data_utils.py`.

5. **`compute_metrics.py` has two parallelization modes**: simple multiprocessing (when no model weights needed) and threaded derive mode (when B-derivation or post-norm derivation needs selective weight loading).

---

## Process 1: Collection

```
slurm/collect.sh
  └─▶ scripts/collect.py
        ├─▶ data/__init__.py
        │     └─▶ data/<dataset>_loader.py    (fineweb, pile, olmomix, etc.)
        ├─▶ utils/data_utils.py
        ├─▶ utils/model_registry.py
        ├─▶ utils/hooks.py
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
