# Codebase Guide

## What this project does
Tracks how LLM representations evolve during pretraining through spectral methods (RankMe, alpha/power-law exponent) and K-FAC curvature analysis. Supports Pythia (14m–12b) and OLMo-2 (1B, 7B) model families.

## Project structure

```
scripts/              # Unified pipeline (new)
  collect.py          # Main collection: residual activations + K-FAC covariance
  verify_collect.py   # CPU-friendly verification tests
  storage_tool.py     # CLI for storage format manipulation (planned)

configs/              # YAML configs for collect.py (--config flag)
  reproduce_rankme_pythia.yaml      # padded fineweb, identity head, last token
  reproduce_rankme_pythia_hook.yaml # same but hook-based post-norm
  reproduce_kfac_pythia.yaml        # packed fineweb, A+G cov, all tokens

rankme_alpha_scripts/ # Legacy activation pipeline (kept for reference)
  extract_activations.py
  compute_metrics.py

kfac_scripts/         # Legacy K-FAC pipeline (kept for reference)
  collect_kfac.py
  compute_kfac_metrics.py

utils/
  model_registry.py   # Model loading, checkpoint discovery, architecture helpers
  hooks.py            # CovarianceCollector, ResidualCapture hook classes
  storage.py          # Storage format handling (cov, cov_svd, eigenvalues)
  data_utils.py       # Packing, padding, token mask computation
  powerlaw.py         # Eigenspectrum, RankMe, alpha fitting
  checkpoint_info.py  # Token count lookups

data/                 # Dataset loaders (HuggingFace)
  fineweb_loader.py, wikitext_loader.py, lam_loader.py, sciq_loader.py

memorization_kfac/    # Merullo et al. released code (unmodified)

slurm/                # SLURM job scripts for cluster
analysis/             # Plotting scripts
```

## Running the unified pipeline

```bash
# Collect residual activations (like old extract_activations.py)
python scripts/collect.py --config configs/reproduce_rankme_pythia.yaml \
    --model_name EleutherAI/pythia-70m-deduped

# Collect K-FAC factors (like old collect_kfac.py)
python scripts/collect.py --config configs/reproduce_kfac_pythia.yaml \
    --model_name EleutherAI/pythia-1.4b-deduped --batch_size 64

# CLI flags override config values
python scripts/collect.py --config configs/reproduce_kfac_pythia.yaml \
    --model_name EleutherAI/pythia-14m --max_checkpoints 3 --num_samples 50
```

## Key concepts

### Collection modes
- **collect_residual**: Capture residual stream. Hook points: `identity_head` (fast, post-norm), `post_norm` (hook), `pre_norm` (raw residual), `post_attn_N`, `pre_block_N`
- **collect_A/G/B**: Covariance matrices for MLP projections. A=input, G=gradient, B=output. G requires backward pass.

### Data modes
- **packing=padded**: Individual sequences, padding to longest. Good for last-token extraction.
- **packing=packed**: Concatenated tokens, no padding. Standard for K-FAC and training-like data.
- **token_selection=last**: Last token per sequence (padded) or per document (packed, needs boundary_token_ids).
- **token_selection=all**: All positions except last (which has no next-token target).

### Storage formats
- **cov**: Raw d×d covariance matrix + count n. `M = Σ xxT`, divide by n for E[xxT].
- **cov_svd**: Eigenvectors + eigenvalues of E[xxT]. Default for multi-checkpoint. Reconstruct: `V @ diag(λ) @ V.T`.
- **eigenvalues**: Just eigenvalues. Cheapest.
- Cross-basis projections stored additionally alongside primary format.

### Model families
- **Pythia** (GPTNeoX): 2 MLP projections (dense_h_to_4h, dense_4h_to_h). float16.
- **OLMo-2** (LLaMA-style): 3 MLP projections (gate_proj, up_proj, down_proj). bfloat16. RMSNorm.
- Checkpoint discovery via `utils/model_registry.get_checkpoint_schedule()`.

## Verification

```bash
# Self-consistency tests (CPU, no existing data needed)
python scripts/verify_collect.py --mode self_consistency

# Compare against existing computed data (on server)
python scripts/verify_collect.py --mode server

# Both
python scripts/verify_collect.py --mode all
```

## On the SLURM cluster

```bash
export HF_HOME="/projects/prjs1815/hf_cache"
cd ~/Tracing-representation-geometry-reproduction

# Array jobs use model list + batch size logic
sbatch slurm/run_kfac_pythia.job
sbatch slurm/run_olmo.job
```

## Dependencies
torch, transformers, numpy, sklearn, datasets, tqdm, jsonargparse, matplotlib, seaborn
