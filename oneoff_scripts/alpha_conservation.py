"""Does the stream compress because the unembedding flattens to compensate?

The logits are z = W_U h. In the unembedding's own basis the logit variance along direction j
is sigma_j^2 * e_j, where sigma_j is the head's singular value and e_j the stream's energy in
that direction. If the model holds its OUTPUT spectrum roughly fixed and splits the work
between the two factors, then 2 log sigma_j + log e_j is roughly constant in j -- which on a
log-rank axis is exactly the observed mirroring of the two alphaReQ slopes, and predicts that
the LOGIT covariance has a far flatter alphaReQ trajectory than either factor.

Three spectra are stored per checkpoint so the notebook can fit alphaReQ over any rank window
with the project's own _alpha:

    stream   centered eigenvalues of the leaf's covariance
    head     sigma(W_U)^2
    logit    eigenvalues of W_U Sigma_c W_U^T, from the d x d surrogate B^T G B, where
             Sigma_c = B B^T and G = W_U^T W_U = V^T diag(sigma^2) V

Everything on the head side comes from what is already stored -- unembedding_svd holds V and
unembedding_spectra.pt holds sigma -- so G needs no weights and nothing here downloads
anything. Recomputing it from a snapshot per checkpoint is what filled the cache before.

Writes data/results/alpha_conservation.pt. GPU node.
"""

import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from oneoff_scripts.unembedding_spectra import MODELS
from utils.model_registry import get_token_count

SVD_DIR = "data/inferences/final_stream_svd"
VEC_DIR = "data/results/unembedding_svd"
SPECTRA = "data/results/unembedding_spectra*.pt"   # glob: array jobs write one file per model,
                                                   # a single shared file loses races (see below)
LEAVES = ("after_final_norm", "before_final_norm")
SKIP = ("pythia-160m-deduped",)    # quirky at the end of its run; excluded from every readout


def _steps(d: str) -> list:
    return sorted(int(os.path.basename(f)[4:-3]) for f in glob.glob(f"{d}/step*.pt"))


def _centered_cov(leaf: dict, dev: str) -> torch.Tensor:
    lam = leaf["acts_eigvals"].to(dev).float().clamp_min(0)
    vecs = leaf["acts_eigvecs"].to(dev).float()
    mu = leaf["acts_mean"].to(dev).float()
    return (vecs * lam) @ vecs.T - torch.outer(mu, mu)


def _sqrt_factor(sigma: torch.Tensor) -> torch.Tensor:
    """B with Sigma_c = B B^T, from Sigma_c's own eigendecomposition."""
    lam, vecs = torch.linalg.eigh(sigma.double())
    return (vecs * lam.clamp_min(0).sqrt()).float()


def main(out_path: str = "data/results/alpha_conservation.pt", svd_dir: str = SVD_DIR,
         vec_dir: str = VEC_DIR, spectra_path: str = SPECTRA, models: tuple = MODELS,
         leaves: tuple = LEAVES) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    spectra: dict = {}                      # merge every per-model file over the shared one
    for f in sorted(glob.glob(spectra_path)):
        for m, v in torch.load(f, weights_only=False).items():
            spectra.setdefault(m, {}).update(v)
    for name in models:
        model = name.split("/")[-1]
        if model in SKIP:
            continue
        steps = _steps(f"{svd_dir}/{model}")
        done = out.setdefault(model, {})
        sv_all = spectra.get(model, {})
        for step in steps:
            vec_path = f"{vec_dir}/{model}/step{step}.pt"
            if step in done:
                continue
            if not os.path.exists(vec_path) or step not in sv_all:
                why = "no stored head SVD" if not os.path.exists(vec_path) else "no stored sigma"
                print(f"{model} step{step}: SKIPPED ({why})", flush=True)
                continue
            d = torch.load(f"{svd_dir}/{model}/step{step}.pt", map_location="cpu",
                           weights_only=False)
            vh = torch.load(vec_path, map_location=dev, weights_only=True).float()
            sv = sv_all[step].to(dev).float()
            gram = vh.T @ (sv[:, None] ** 2 * vh)              # W_U^T W_U, no weights needed
            entry: dict = {"head": (sv ** 2).cpu(), "tokens": get_token_count(model, step)}
            for leaf in leaves:
                if leaf not in d:
                    continue
                b = _sqrt_factor(_centered_cov(d[leaf], dev))
                logit = torch.linalg.eigvalsh((b.T @ gram @ b).double()).flip(0).clamp_min(0)
                entry[leaf] = {"stream": d[leaf]["acts_eigvals_centered"].float().cpu(),
                               "logit": logit.float().cpu(),
                               "n": int(d[leaf]["acts_n"])}
            done[step] = entry
            torch.save(out, out_path)
            print(f"{model} step{step}: head sv1={float(sv[0]):.1f} "
                  f"logit lam1={float(entry[leaves[0]]['logit'][0]):.3e}", flush=True)
        miss = len(steps) - len([s for s in steps if s in done])
        print(f"COVERAGE {model}: {len(steps) - miss}/{len(steps)} checkpoints"
              + (f"  <-- {miss} MISSING, results are PARTIAL" if miss else ""), flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
