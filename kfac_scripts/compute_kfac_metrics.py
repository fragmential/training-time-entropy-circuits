#!/usr/bin/env python
"""
Compute spectral metrics from stored K-FAC covariance matrices.

For each checkpoint × layer × projection, eigendecomposes A and G and computes:
  - Effective rank (RankMe), spectral entropy, participation ratio
  - Trace, top-k energy fractions
  - Combined FIM trace

Output format mirrors rankme_alpha_scripts/compute_metrics.py:
  results/kfac/{dataset}/kfac_metrics_{model_name}.npy
"""
import os
import re
import sys
import numpy as np
import torch
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_STEP_RE = re.compile(r"step(\d+)\.pt$")


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------
def spectral_metrics(eigenvalues):
    """Compute spectral metrics from eigenvalues (descending order, non-negative).

    Returns dict with: rankme, spectral_entropy, participation_ratio, trace,
    top10_energy, top50_energy, top100_energy, eigenspectrum.
    """
    evals = np.array(eigenvalues, dtype=np.float64)
    evals = np.maximum(evals, 0.0)  # clamp numerical negatives
    total = evals.sum()

    if total <= 0:
        return {
            "rankme": 0.0,
            "spectral_entropy": 0.0,
            "participation_ratio": 0.0,
            "trace": 0.0,
            "top10_energy": 0.0,
            "top50_energy": 0.0,
            "top100_energy": 0.0,
            "eigenspectrum": evals,
        }

    p = evals / total
    eps = 1e-30
    entropy = -np.sum(p * np.log(p + eps))
    rankme = np.exp(entropy)
    pr = total ** 2 / np.sum(evals ** 2) if np.sum(evals ** 2) > 0 else 0.0

    cumsum = np.cumsum(evals)
    top10 = cumsum[min(9, len(evals) - 1)] / total
    top50 = cumsum[min(49, len(evals) - 1)] / total
    top100 = cumsum[min(99, len(evals) - 1)] / total

    return {
        "rankme": float(rankme),
        "spectral_entropy": float(entropy),
        "participation_ratio": float(pr),
        "trace": float(total),
        "top10_energy": float(top10),
        "top50_energy": float(top50),
        "top100_energy": float(top100),
        "eigenspectrum": evals,
    }


def compute_metrics_for_step(args):
    """Process a single checkpoint file.  Returns (step, metrics_dict)."""
    step, path = args
    data = torch.load(path, map_location="cpu")

    step_metrics = {}
    for proj_name, proj_data in data.items():
        A = proj_data["A"].float()
        G = proj_data["G"].float()

        eva_A = torch.linalg.eigvalsh(A).flip(0).numpy()
        eva_G = torch.linalg.eigvalsh(G).flip(0).numpy()

        a_metrics = spectral_metrics(eva_A)
        g_metrics = spectral_metrics(eva_G)

        step_metrics[proj_name] = {
            **{f"A_{k}": v for k, v in a_metrics.items()},
            **{f"G_{k}": v for k, v in g_metrics.items()},
            "fim_trace": float(a_metrics["trace"] * g_metrics["trace"]),
            "n": int(proj_data["n"]),
        }

    return step, step_metrics


# ---------------------------------------------------------------------------
# Discovery + main
# ---------------------------------------------------------------------------
def discover_kfac_files(kfac_dir):
    """Return {step: path} for K-FAC factor files."""
    step_files = {}
    if not os.path.isdir(kfac_dir):
        return step_files
    for fname in os.listdir(kfac_dir):
        m = _STEP_RE.match(fname)
        if m:
            step_files[int(m.group(1))] = os.path.join(kfac_dir, fname)
    return step_files


def main(
    model_name: str = "EleutherAI/pythia-14m",
    dataset_name: str = "fineweb",
    num_workers: int = None,
    recompute: bool = False,
):
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    kfac_dir = os.path.join("kfac_factors", dataset_name, short_name)
    step_files = discover_kfac_files(kfac_dir)

    if not step_files:
        print(f"No K-FAC factor files found in {kfac_dir}")
        return

    results_dir = os.path.join("results", "kfac", dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"kfac_metrics_{short_name}.npy")

    # Load existing results
    res_dict = {}
    if os.path.exists(results_path) and not recompute:
        try:
            res_dict = np.load(results_path, allow_pickle=True).item()
        except (OSError, ValueError, TypeError):
            res_dict = {}

    to_compute = [(s, p) for s, p in step_files.items() if recompute or s not in res_dict]
    if not to_compute:
        print(f"All {len(step_files)} steps already have metrics in {results_path}")
        return

    n_workers = num_workers or os.cpu_count()
    print(f"Computing K-FAC metrics for {len(to_compute)}/{len(step_files)} steps "
          f"using {n_workers} workers...")

    with Pool(processes=n_workers) as pool:
        for step, metrics in pool.imap_unordered(compute_metrics_for_step, to_compute):
            res_dict[step] = metrics
            # Print summary for first projection
            first_proj = next(iter(metrics))
            m = metrics[first_proj]
            print(f"  Step {step}: {first_proj} "
                  f"A_rankme={m['A_rankme']:.1f} G_rankme={m['G_rankme']:.1f} "
                  f"fim_trace={m['fim_trace']:.2e}")

    np.save(results_path, res_dict)
    print(f"Saved {len(res_dict)} results to {results_path}")


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from jsonargparse import CLI
    CLI(main)
