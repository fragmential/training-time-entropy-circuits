# Reproduction & Extension Plan

## Part 1: Reproduce paper results (RankMe, alpha-ReQ)

### Paper settings (from supplementary)

| Parameter | Value |
|-----------|-------|
| Dataset | [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) sample-10BT |
| Max sequence length | 512 |
| Number of sequences | 15,000 |
| Batch size | 16 |
| Min token length | 32 |
| Compute | Single 80GB A100 or 40GB L40S |

### Models

**Pythia** (18 models): pythia-{14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b} each with and without deduplication. Trained on The Pile, checkpoints every 1000 steps up to 143k.

**OLMo-2** (2 models): OLMo-2-0425-1B and OLMo-2-1124-7B. Trained on dolmino-mix-1124, checkpoints listed in `{1b,7b}_revisions.txt`.

### Scripts

```bash
# Pythia (SLURM array job, 18 tasks)
sbatch slurm/run_pythia.job

# OLMo (SLURM array job, 2 tasks)
sbatch slurm/run_olmo.job

# Test runs (smallest models, 100 samples, 30min)
sbatch slurm/test_pythia.job   # pythia-14m
sbatch slurm/test_olmo.job     # OLMo-2-0425-1B
```

### Output

Results saved to `results/results_{model_short_name}.npy`, each a dict keyed by step number:

```python
{
    step_num: {
        'eigenspectrum': np.array,
        'rankme': float,
        'alpha': float,
        'r2': float,
        'r2_100': float,
    }
}
```

### Key changes from original repo

- Switched Pythia script from WikiText to FineWeb (matching paper)
- Fixed `fineweb-edu` to `fineweb` in data loader (matching paper)
- num_samples: 2000 -> 15,000
- max_length: 128 -> 512
- batch_size: 32 -> 16
- Added dataset filtering cache (`data/cache/`)
- Added checkpoint prefetching
- Removed `device_map="auto"` from OLMo script (no accelerate dependency)
- All SLURM jobs use `uv run` (no manual venv)

---

## Part 2: Memorization metrics (Figure 3)

### What it measures

**Distributional memorization** = Spearman correlation between:
- LLM next-token probabilities on TriviaQA answers
- Infini-gram (corpus n-gram) probabilities on the same tokens

High correlation = model output aligns with short-context n-gram statistics (memorization).
Low correlation = model uses long-context reasoning beyond n-gram patterns.

### Scripts

```bash
# Infini-gram probabilities (uses external API, ~slow)
python memorization_scripts/compute_infgrams.py

# LLM token probabilities (per model/checkpoint)
python memorization_scripts/compute_llm_likelihood.py --model_name EleutherAI/pythia-1b --step 10000
```

### Output

- `results/infgram/triviaqa_{N}samples.npy` — dict[idx] -> {Question, Answer, infgram_prob, logprob_result, prob_result}
- `results/llm_likelihood/{model_name}/triviaqa_step_{step}.npy` — dict[idx] -> {Question, Answer, llm_prob, llm_logprob, logprob_result}

### Status

Scripts are functional but results are **not yet plotted** anywhere. Need to add analysis script to compute Spearman correlation and overlay with alpha-ReQ trajectories.

---

## Part 3: K-FAC curvature metrics (extension)

### Motivation

The [K-FAC curvature paper](https://arxiv.org/abs/2510.24256) shows that loss curvature eigenspectra distinguish memorized from generalized knowledge. We extract these spectra as **read-only metrics** (no weight editing) to track how curvature geometry evolves alongside representation geometry during training.

### Background

K-FAC approximates the Fisher Information Matrix per layer as a Kronecker product:

```
F_layer ≈ G ⊗ A
```

- **A = E[xx^T]** — covariance of pre-activation inputs (shape `[d_in, d_in]`)
- **G = E[gg^T]** — covariance of pre-activation output gradients (shape `[d_out, d_out]`)

Collected via forward pre-hooks (capture x) and backward hooks (capture gradient g) during a forward+backward pass with cross-entropy loss.

### MLP layer names

| Model family | Up projection | Down projection | Gate projection |
|-------------|--------------|-----------------|-----------------|
| Pythia (GPTNeoX) | `gpt_neox.layers[b].mlp.dense_h_to_4h` | `gpt_neox.layers[b].mlp.dense_4h_to_h` | N/A |
| OLMo-2 (LLaMA-style) | `model.layers[b].mlp.up_proj` | `model.layers[b].mlp.down_proj` | `model.layers[b].mlp.gate_proj` |

### Metrics to extract (per layer, per projection)

| Metric | Formula | Meaning |
|--------|---------|---------|
| `eigenspectrum_A` | eigenvalues of A | Input covariance spectrum |
| `eigenspectrum_G` | eigenvalues of G | Gradient covariance spectrum |
| `rankme_A` | exp(H(p)) where p_i = lambda_i / sum(lambda) | Effective rank of input space |
| `rankme_G` | same for G | Effective rank of gradient space |
| `alpha_A` | Power-law exponent fit | Input covariance decay rate |
| `alpha_G` | Power-law exponent fit | Gradient covariance decay rate |
| `trace_A` | sum(lambda_i(A)) | Total input variance |
| `trace_G` | sum(lambda_i(G)) | Total gradient variance |
| `condition_A` | lambda_max / lambda_min | Input conditioning |
| `condition_G` | lambda_max / lambda_min | Gradient conditioning |
| `eff_rank_95_A` | min k: cumsum(lambda)/sum >= 0.95 | 95% variance dimensionality |
| `eff_rank_95_G` | same for G | 95% variance dimensionality |
| `kron_total_mass` | trace(G) * trace(A) | Total approximate Fisher mass |

### Data sources

| Dataset | Purpose | Models |
|---------|---------|--------|
| FineWeb sample-10BT | Held-out (both families) | All |
| The Pile | Training data for Pythia | Pythia only |
| dolmino-mix-1124 | Training data for OLMo-2 | OLMo only |

The K-FAC paper uses ~20M tokens for factor collection (~40k sequences at 512 tokens).

### Layer selection

Last 4 layers + layer 0 as control:

| Model | Layers |
|-------|--------|
| Pythia-1B (16 layers) | 0, 12, 13, 14, 15 |
| Pythia-2.8B (32 layers) | 0, 28, 29, 30, 31 |
| OLMo-2-1B (16 layers) | 0, 12, 13, 14, 15 |
| OLMo-2-7B (32 layers) | 0, 28, 29, 30, 31 |

### Checkpoint selection

~20 checkpoints per model, spread across training phases:
- Early: steps 0, 64, 128, 256, 512
- Warmup/entropy boundary: 1000, 2000, 3000, 5000
- Entropy phase: 8000, 12000, 16000, 20000
- Compression phase: 30000, 50000, 70000, 100000, 120000, 143000

### Implementation steps

1. **Data loaders**: `data/pile_loader.py`, `data/dolmino_loader.py`
2. **Factor collection**: `curvature_scripts/collect_kfac_factors.py` — hook-based A,G collection for both GPTNeoX and OLMo architectures
3. **Metric computation**: `curvature_scripts/compute_kfac_metrics.py` — eigendecompose + extract metrics using existing `utils/powerlaw.py`
4. **SLURM jobs**: `slurm/collect_kfac_pythia.job`, `slurm/collect_kfac_olmo.job`
5. **Test**: Smallest model (pythia-14m), 1 checkpoint, FineWeb only
6. **Full runs**: All models, all checkpoints, both datasets
7. **Analysis**: `analysis/plot_kfac_trajectories.py`, `analysis/plot_kfac_train_vs_heldout.py`

### Estimated compute

- Factor collection requires forward + backward (~3x slower than forward-only)
- ~2-4h per checkpoint per model on H100
- ~20 checkpoints x 2 datasets x 2-4h = 80-160 GPU-hours per model

### File structure

```
curvature_scripts/
├── collect_kfac_factors.py
├── compute_kfac_metrics.py
data/
├── pile_loader.py
├── dolmino_loader.py
├── fineweb_loader.py          (existing)
slurm/
├── collect_kfac_pythia.job
├── collect_kfac_olmo.job
analysis/
├── plot_kfac_trajectories.py
├── plot_kfac_train_vs_heldout.py
results/
├── kfac_factors/{model}/{dataset}/factors_step{N}.pt
└── kfac_metrics/{model}/{dataset}/metrics_step{N}.npy
```

### Hypotheses

1. **Entropy phase = curvature flattening**: During manifold expansion, K-FAC eigenspectra flatten (lower alpha, higher effective rank in A and G).
2. **Compression phase = curvature sharpening**: During anisotropic consolidation, curvature concentrates into fewer eigen-directions.
3. **Training vs held-out gap**: Curvature on training data has sharper directions than on held-out data, and this gap is largest during the entropy phase (peak memorization).
4. **Correlation with memorization**: The distributional memorization metric correlates with K-FAC effective rank or spectral properties.

---

## Cluster setup (Snellius)

```bash
# One-time setup
bash slurm/setup_cluster.sh

# Submit jobs
sbatch slurm/test_pythia.job    # smoke test first
sbatch slurm/run_pythia.job     # full run
sbatch slurm/run_olmo.job       # full run
```

- Partition: `gpu_h100`
- HF cache: `/projects/prjs1815/hf_cache`
- Project dir: `$HOME/Tracing-representation-geometry-reproduction`
- Package manager: `uv` (no manual venvs)
