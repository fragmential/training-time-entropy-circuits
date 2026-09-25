"""Final-stream ↔ unembedding alignment (plan.md item 1b): per model, leaf, checkpoint,
the share of the stream's variance lying in the head's top-k right-singular subspace,
    share_k = tr(P_k Σ) / tr(Σ),   P_k = Q_k Q_kᵀ,  Q_k = top-k rows of Vh, transposed.
Σ is reconstructed from the stored cov_svd files (matrix products only, no new
eigendecompositions): centered Σc = V diag(λ) Vᵀ − μμᵀ, so
    tr(P Σc) = ‖Q_kᵀ V diag(√λ)‖_F² − ‖Q_kᵀ μ‖²,   tr(Σc) = Σλ − ‖μ‖².
The uncentered share drops the μ terms. Chance level for a random k-subspace is k/d.
Writes {model: {leaf: {"steps", "ks", "share_centered", "share_uncentered"}}} to
data/results/unembedding_alignment.pt."""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

SVD_DIR = "data/inferences/final_stream_svd"
VEC_DIR = "data/results/unembedding_svd"
LEAVES = ("before_final_norm", "after_final_norm")
KS = (1, 8, 32, 128, 512)


def main(out_path: str = "data/results/unembedding_alignment.pt") -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict = {}
    for model in sorted(os.listdir(SVD_DIR)):
        steps = sorted(int(re.search(r"step(\d+)", f).group(1))
                       for f in glob.glob(f"{SVD_DIR}/{model}/step*.pt")
                       if os.path.exists(f"{VEC_DIR}/{model}/{os.path.basename(f)}"))
        if not steps:
            print(f"{model}: no overlapping steps, skipped", flush=True)
            continue
        res = {leaf: {"steps": steps, "ks": list(KS),
                      "share_centered": [], "share_uncentered": []} for leaf in LEAVES}
        for step in steps:
            d = torch.load(f"{SVD_DIR}/{model}/step{step}.pt", map_location=dev, weights_only=False)
            vh = torch.load(f"{VEC_DIR}/{model}/step{step}.pt", map_location=dev, weights_only=True)
            for leaf in LEAVES:
                V = d[leaf]["acts_eigvals"].to(dev), d[leaf]["acts_eigvecs"].to(dev)
                lam, vecs = V[0].float().clamp(min=0), V[1].float()
                mu = d[leaf]["acts_mean"].to(dev).float()
                A = vh.float() @ (vecs * lam.sqrt())        # (d, d): Q_kᵀ V √λ for every k prefix
                b = vh.float() @ mu                          # (d,): Q_kᵀ μ
                num_u = (A ** 2).sum(1).cumsum(0)            # tr(P_k Σ_uncentered) per k prefix
                num_c = num_u - (b ** 2).cumsum(0)
                tr_u = lam.sum()
                tr_c = tr_u - mu.square().sum()
                res[leaf]["share_uncentered"].append([float(num_u[k - 1] / tr_u) for k in KS])
                res[leaf]["share_centered"].append([float(num_c[k - 1] / tr_c) for k in KS])
        out[model] = res
        torch.save(out, out_path)
        print(f"{model}: {len(steps)} steps done", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
