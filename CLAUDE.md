# Codebase Guide

## What this project does
Tracks how LLM representations evolve during pretraining through spectral methods (RankMe, alpha/power-law exponent) and covariance curvature analysis. Supports Pythia (14m–12b) and OLMo-2 (1B, 7B) model families.

## Project structure

```
scripts/              # Unified pipeline
  collect.py          # Main collection: residual activations + covariance factors
  verify.py   # Verification tests (self-consistency + server comparison)

configs/              # YAML configs for collect.py (--config flag)
  reproduce_rankme_alpha.yaml      # padded fineweb, identity head, last token
  reproduce_rankme_alpha_hook.yaml # same but hook-based post-norm (verification)
  reproduce_kfac.yaml              # packed fineweb, A+G covariance, all tokens

rankme_alpha_scripts/ # Legacy activation pipeline (kept for reference)
  extract_activations.py
  compute_metrics.py

kfac_scripts/         # Legacy covariance pipeline (kept for reference)
  collect_kfac.py
  compute_kfac_metrics.py

utils/
  model_registry.py   # ModelConfig, checkpoint discovery, model loading, architecture helpers
  hooks.py            # CovarianceCollector, ResidualCapture hook classes
  storage.py          # Storage format handling (cov, cov_svd, eigenvalues)
  data_utils.py       # Packing, padding, token mask computation
  powerlaw.py         # Eigenspectrum, RankMe, alpha fitting
  checkpoint_info.py  # Token count lookups

data/                 # Dataset loaders (HuggingFace)
  fineweb_loader.py, wikitext_loader.py, lam_loader.py, sciq_loader.py

memorization_kfac (reference repo)/  # Merullo et al. released code (unmodified)

slurm/                # SLURM job scripts and wrapper
  run_collect.sh      # Array job wrapper: reads config, resolves models, submits sbatch
  run_pythia.job      # Legacy: Pythia activation extraction
  run_olmo.job        # Legacy: OLMo activation extraction
  run_kfac_pythia.job # Legacy: Pythia covariance collection
  run_kfac_olmo.job   # Legacy: OLMo covariance collection

analysis/             # Plotting scripts
```

## Running the unified pipeline

```bash
# Single model (scalar config)
python scripts/collect.py --model_name EleutherAI/pythia-70m-deduped \
    --config configs/reproduce_rankme_alpha.yaml

# Model sweep via SLURM wrapper (comment out models in the script to skip)
./slurm/run_collect.sh configs/reproduce_rankme_alpha.yaml
./slurm/run_collect.sh configs/reproduce_kfac.yaml --time 12:00:00

# CLI flags override config values
python scripts/collect.py --config configs/reproduce_kfac.yaml \
    --model_name EleutherAI/pythia-14m --max_checkpoints 3 --num_samples 50
```

## Config system

Configs use a `CollectConfig` dataclass (in collect.py). YAML files can specify:
- **`model_name`**: single string or list of models for sweep
- **Vectorizable fields** (`batch_size`, `max_checkpoints`, `max_layers_per_pass`): scalar (same for all), list (zipped with model_name), or dict keyed by model name
- **`array_id`**: passed by SLURM wrapper to select one model from the list

Model-specific loading details (`revisions_file`, `early_training_model`) are in `utils/model_registry.py`, not in configs.

## Key concepts

### Collection modes
- **collect_residual**: Capture residual stream. Hook points: `identity_head` (fast, after final norm), `after_final_norm` (hook), `before_final_norm` (raw residual), `post_attn_N`, `pre_block_N`
- **collect_A/G/B**: Covariance matrices for MLP projections. A=input, G=gradient, B=output. G requires backward pass.

### Data modes
- **packing=padded**: Individual sequences, padding to longest. Good for last-token extraction.
- **packing=packed**: Concatenated tokens, no padding. Standard for covariance collection and training-like data.
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
- Per-model config (revisions, early training repos) in `utils/model_registry.py`.
- Checkpoint discovery via `utils/model_registry.get_checkpoint_schedule()`.

## Verification

```bash
# Self-consistency tests (CPU, no existing data needed)
python scripts/verify.py --mode self_consistency

# Compare against existing computed data (on server)
python scripts/verify.py --mode server

# Both
python scripts/verify.py --mode all
```

## On the SLURM cluster

```bash
export HF_HOME="/projects/prjs1815/hf_cache"
cd ~/Tracing-representation-geometry-reproduction

# New unified wrapper (comment out models in run_collect.sh to run a subset)
./slurm/run_collect.sh configs/reproduce_rankme_alpha.yaml
./slurm/run_collect.sh configs/reproduce_kfac.yaml --time 12:00:00
```

## Dependencies
torch, transformers, numpy, sklearn, datasets, tqdm, jsonargparse, matplotlib, seaborn
