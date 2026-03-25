# Tracing the Representation Geometry of Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2509.23024-b31b1b.svg)](https://arxiv.org/abs/2509.23024)
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/119054)

Official code for our NeurIPS 2025 paper on analyzing LLM training through geometric phases.

## Overview

We use spectral methods (RankMe and α-ReQ) to track how LLM representations evolve during training. We find three consistent phases:

1. **Warmup**: Rapid representational collapse
2. **Entropy-seeking**: Expansion of dimensionality, peak memorization
3. **Compression-seeking**: Anisotropic consolidation, improved generalization

## Installation

```bash
pip install torch transformers datasets numpy matplotlib seaborn scikit-learn tqdm jsonargparse
```

## Quick Start

### Unified pipeline (recommended)

The unified collection script replaces the separate activation and K-FAC pipelines. It is driven by YAML config files.

```bash
# Collect residual activations (equivalent to old extract_activations.py)
python scripts/collect.py --config configs/reproduce_rankme_alpha.yaml \
    --model_name EleutherAI/pythia-6.9b

# Collect K-FAC covariance factors (equivalent to old collect_kfac.py)
python scripts/collect.py --config configs/reproduce_kfac.yaml \
    --model_name EleutherAI/pythia-1.4b-deduped --batch_size 64

# OLMo models (revisions_file and early_training_model are in model_registry)
python scripts/collect.py --config configs/reproduce_rankme_alpha.yaml \
    --model_name allenai/OLMo-2-1124-7B
```

Config files set defaults; CLI flags override them. Available configs:

| Config | What it does |
|--------|-------------|
| `reproduce_rankme_alpha.yaml` | Padded fineweb, identity head, last token |
| `reproduce_rankme_alpha_hook.yaml` | Same but hook-based post-norm |
| `reproduce_kfac.yaml` | Packed fineweb, A+G covariance, sampled labels |

### Legacy scripts (still work)

```bash
python rankme_alpha_scripts/extract_activations.py --model_name EleutherAI/pythia-6.9b
python kfac_scripts/collect_kfac.py --model_name EleutherAI/pythia-1.4b-deduped
```

### Compute metrics

```bash
python rankme_alpha_scripts/compute_metrics.py pythia-6.9b
python kfac_scripts/compute_kfac_metrics.py --model_name EleutherAI/pythia-1.4b-deduped
```

### Visualize results

```bash
python analysis/plot_alpha_traj.py
```

### Verify the pipeline

```bash
# Self-consistency tests (CPU, no existing data needed)
python scripts/verify_collect.py --mode self_consistency

# Compare against existing computed data (on server)
python scripts/verify_collect.py --mode all
```

###  Memorization analysis

```bash
python memorization_scripts/compute_infgrams.py
python memorization_scripts/compute_llm_likelihood.py
```

## Repository Structure

```
scripts/                            # Unified pipeline
├── collect.py                      # Main collection (residual + K-FAC)
├── verify_collect.py               # Verification tests

configs/                            # YAML configs for collect.py

rankme_alpha_scripts/               # Legacy activation pipeline
├── extract_activations.py
└── compute_metrics.py

kfac_scripts/                       # Legacy K-FAC pipeline
├── collect_kfac.py
└── compute_kfac_metrics.py

utils/
├── model_registry.py               # Model loading, checkpoint discovery
├── hooks.py                        # CovarianceCollector, ResidualCapture
├── storage.py                      # Storage format handling
├── data_utils.py                   # Packing, padding, token masks
├── powerlaw.py                     # Eigenspectrum and metric utilities
└── checkpoint_info.py              # Step-to-token mapping

data/                               # Dataset loaders
├── fineweb_loader.py
├── wikitext_loader.py
├── lam_loader.py
└── sciq_loader.py

memorization_kfac/                  # Merullo et al. released code (unmodified)
analysis/                           # Visualization scripts
slurm/                              # SLURM job scripts
```

## Collection Features

The unified `scripts/collect.py` supports:

- **Residual capture**: `identity_head` (fast, after final norm), `after_final_norm` (hook), `before_final_norm` (raw residual), per-block hooks
- **Covariance collection**: A (input), G (gradient), B (post-weight output) for MLP projections
- **Data modes**: packed (no padding) or padded sequences
- **Token selection**: all tokens or last token (per sequence or per document)
- **Storage formats**: `cov` (raw matrix), `cov_svd` (eigendecomposition), `eigenvalues` (cheapest)
- **Cross-basis projections**: project onto previously stored eigenbases
- **Skip positions**: skip first k tokens after document boundaries in packed data
- **Answer-only gradients**: mask prompt tokens for task-specific G collection (planned)

## Key Metrics

**RankMe (Effective Rank)**
- Measures effective dimensionality of representations
- Higher = more isotropic (entropy-seeking phase)
- Lower = more collapsed (compression-seeking phase)

**α-ReQ (Power-law Exponent)**
- Measures eigenspectrum decay rate
- Lower α = more uniform (entropy-seeking)
- Higher α = more concentrated (compression-seeking)

## Results Format

Output saved as `.npy` files containing:

```python
{
    step_num: {
        'rankme': float,
        'sq_rankme': float,
        'lin_matent': float,
        'matent': float,
        'alpha': float,
        'eigenspectrum': array,
        'r2': float,
        'r2_100': float
    }
}
```

## Citation

```bibtex
@inproceedings{li2025tracing,
  title={Tracing the Representation Geometry of Language Models from Pretraining to Post-training},
  author={Li, Melody Zixuan and Agrawal, Kumar Krishna and Ghosh, Arna and Teru, Komal Kumar and Santoro, Adam and Lajoie, Guillaume and Richards, Blake A.},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) 2025},
  year={2025},
  url={https://arxiv.org/abs/2509.23024}
}
```

## License

MIT License
