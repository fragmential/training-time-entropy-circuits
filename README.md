# Tracing Representation Geometry — Reproduction & Extension

Reproduction and extension of spectral analysis of LLM representations during pretraining. Tracks RankMe, α-ReQ, and K-FAC curvature across training checkpoints. Supports Pythia (14m–12b) and OLMo-2 (1B, 7B).

## Installation

This project is managed with uv. if you have it installed, all you need to get setup is:
```bash
uv sync
```

To activate the venv, use
```bash
source .venv/bin/activate
```

## Quick Start

### Collection

```bash
# Residual activations across all checkpoints
python scripts/collect.py --config configs/reproduce_rankme_alpha.yaml \
    --model_name EleutherAI/pythia-6.9b

# K-FAC covariance factors
python scripts/collect.py --config configs/reproduce_kfac.yaml \
    --model_name EleutherAI/pythia-1.4b-deduped

# OLMo models work the same way
python scripts/collect.py --config configs/full_limited.yaml \
    --model_name allenai/OLMo-2-1124-7B
```

Config files set defaults; CLI flags override them. Each config has `output_dir: data/inferences/<config_name>` at the top.

| Config | What it does |
|--------|-------------|
| `reproduce_rankme_alpha.yaml` | Padded fineweb, identity head, last token |
| `reproduce_kfac.yaml` | Packed fineweb, A+G covariance, all tokens |
| `full.yaml` | Full sweep: residual + A + G, all checkpoints |
| `full_limited.yaml` | Same, max 10 checkpoints |
| `rankme_alpha_packed.yaml` | Residual only, packed, all checkpoints |

### Compute metrics

```bash
python scripts/compute_metrics.py --model_name pythia-6.9b \
    --input_dir data/inferences/reproduce_rankme_alpha/pythia-6.9b
```

### Verify the pipeline

```bash
python scripts/verify.py --mode self_consistency
python scripts/verify.py --mode all
```

## Repository Structure

```
scripts/
├── collect.py            # Unified collection (residual + K-FAC)
├── compute_metrics.py    # RankMe, alpha, K-FAC log-det metrics
├── activation_ratio.py   # Band analysis on eigenvectors
├── verify.py             # Verification tests
├── estimate_vram.py      # GPU VRAM estimator
└── convert_npy.py        # Legacy .npy → .pt converter

configs/                  # YAML configs (one per experiment)

utils/
├── model_registry.py     # Model loading, checkpoint discovery
├── hooks.py              # HookCollector (unified hook class)
├── accessor.py           # DataAccessor (read/write/convert/project/info interface)
├── data_utils.py         # Packing, padding, token masks
└── powerlaw.py           # Eigenspectrum and metric utilities

data/                     # Dataset loaders (HuggingFace streaming)
├── fineweb_loader.py, pile_loader.py, olmomix_loader.py
├── dolmino_loader.py, tulu_sft_loader.py
├── wikitext_loader.py, lam_loader.py, sciq_loader.py

data/
├── inferences/           # Symlink → /projects/prjs1815/inferences
├── results/              # Computed metrics (.npy)
├── filtered_texts/       # Cached text JSONs (auto-generated)
└── mixes/                # Pre-shuffled token tensors
rankme_alpha_scripts/     # Legacy pipeline (reference)
kfac_scripts/             # Legacy pipeline (reference)
memorization_kfac/        # Merullo et al. code (unmodified reference)
analysis/                 # Visualization scripts
slurm/                    # SLURM job scripts and wrappers
```

## Collection Features

- **Residual capture**: `identity_head`, `after_final_norm`, `before_final_norm`, per-block hooks
- **Covariance collection**: A (input) and G (gradient) for MLP projections; fp64 accumulation by default
- **Data modes**: packed or padded; `max_bytes` budget for consistent data volume across model families
- **Storage formats**: `acts`, `acts_svd`, `cov`, `cov_svd`, `eigenvalues` (with `+m` modifier for means)
- **Cross-basis projections**: same-layer G↔B/A and cross-checkpoint via storage CLI
- **DataAccessor**: chainable property API plus `.save()` conversion — `acc["blk3.up"].A.eigvals`, `DataAccessor(path).save(out, format="cov_svd")`

## Key Metrics

**RankMe** — effective dimensionality of representations

**α-ReQ** — power-law exponent of eigenspectrum decay

**K-FAC log-determinant** — damped: `L(α) = d_A·logdet(G + ε_G·I) + d_G·logdet(A + ε_A·I)`, computed at α ∈ {1e-4, 1e-5, 1e-6}

## License

MIT License
