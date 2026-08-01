"""Which directions of the unembedding does the compression phase concentrate variance into?

RankMe of the final stream falls late in training. RankMe is direction-blind: it says the
variance concentrated, not WHERE. This removes named subspaces of the unembedding from the
stream's covariance and re-measures RankMe. Whichever removal flattens the compression phase
names the carrier.

    dark    the bottom-k right singular directions of W_U -- the effective null space of
            Stolfo et al., Confidence Regulation Neurons in Language Models (arXiv:2406.16254),
            and the subspace Cancedda, Spectral Filters, Dark Signals, and Attention Sinks
            (arXiv:2402.09221) shows carries attention sinks
    top1    the single direction the unembedding reads hardest (Cancedda reports its token-side
            image tracks token frequency)
    topk    the top k, same k as dark, so the two ends are comparable
    random  a random k-subspace: the control

The same pass also answers the PREMISE question those subspace names rest on -- is the
stream's dominant direction actually dark? -- for three directions per leaf: the stream MEAN,
the top UNCENTERED eigenvector, and the top CENTERED eigenvector. Centering is the trap: a
massive activation is largely a constant offset, which a centered covariance removes by
construction, so a centered eigenvector can miss it entirely.

Reads the head SVD from data/results/unembedding_svd, which stores the RAW W_U's right
singular vectors -- Cancedda's convention, and the right one here since the after_final_norm
leaf already carries the final norm's gain. Recomputing it from weights would cost a full
snapshot download per checkpoint and move the bottom-k span by <0.01 (see
utils/nullspace.head_subspace). Nothing here downloads anything.

Writes data/results/stream_deflation.pt. GPU node (one d x d eigendecomposition per variant).
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
LEAVES = ("after_final_norm", "before_final_norm")
K_FRAC = 0.01          # Stolfo et al's k ~ 0.01 * d_model. Cancedda uses 0.05; pass --k_frac
                       # 0.05 and a separate --out_path to get his band instead.
SEED = 0
SKIP = ("pythia-160m-deduped",)    # quirky at the end of its run; excluded from every readout


def _steps(d: str) -> list:
    return sorted(int(os.path.basename(f)[4:-3]) for f in glob.glob(f"{d}/step*.pt"))


def _rankme(ev: torch.Tensor) -> float:
    p = ev.clamp_min(0)
    p = p / p.sum().clamp_min(1e-30)
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


def _centered_cov(leaf: dict, dev: str) -> torch.Tensor:
    """Sigma_c = V diag(lam) V^T - mu mu^T, the reconstruction unembedding_alignment uses."""
    lam = leaf["acts_eigvals"].to(dev).float().clamp_min(0)
    vecs = leaf["acts_eigvecs"].to(dev).float()
    mu = leaf["acts_mean"].to(dev).float()
    return (vecs * lam) @ vecs.T - torch.outer(mu, mu)


def _subspaces(vh: torch.Tensor, k: int, dev: str) -> dict:
    """name -> (m, d) orthonormal rows to project OUT of the stream."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    q = torch.linalg.qr(torch.randn(vh.shape[1], k, generator=g))[0].T.to(dev)
    return {"dark": vh[-k:], "top1": vh[:1], "topk": vh[:k], "random": q}


def _deflate(sigma: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Spectrum of sigma with the row space of q removed from both sides."""
    proj = sigma - q.T @ (q @ sigma) - ((sigma @ q.T) @ q) + q.T @ ((q @ sigma @ q.T) @ q)
    return torch.linalg.eigvalsh(proj.double()).flip(0).clamp_min(0).float()


def _where(leaf: dict, vh: torch.Tensor, k: int, dev: str) -> dict:
    """Share of three stream directions inside the bottom-k. Chance = sqrt(k/d)."""
    lam = leaf["acts_eigvals"].to(dev).float().clamp_min(0)
    vecs = leaf["acts_eigvecs"].to(dev).float()
    mu = leaf["acts_mean"].to(dev).float()
    cent = (vecs * lam) @ vecs.T - torch.outer(mu, mu)
    dirs = {"mean": mu, "top_uncentered": vecs[:, 0],
            "top_centered": torch.linalg.eigh(cent.double())[1][:, -1].float()}
    v0 = vh[-k:]
    out = {n: float((v0 @ (v / v.norm().clamp_min(1e-30))).norm()) for n, v in dirs.items()}
    out["chance"] = float((k / vh.shape[1]) ** 0.5)
    return out


def main(out_path: str = "data/results/stream_deflation.pt", svd_dir: str = SVD_DIR,
         vec_dir: str = VEC_DIR, models: tuple = MODELS, leaves: tuple = LEAVES,
         k_frac: float = K_FRAC) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for name in models:
        model = name.split("/")[-1]
        if model in SKIP:
            continue
        steps = _steps(f"{svd_dir}/{model}")
        done = out.setdefault(model, {})
        for step in steps:
            vec_path = f"{vec_dir}/{model}/step{step}.pt"
            if step in done:
                continue
            if not os.path.exists(vec_path):
                print(f"{model} step{step}: SKIPPED (no stored head SVD)", flush=True)
                continue
            d = torch.load(f"{svd_dir}/{model}/step{step}.pt", map_location="cpu",
                           weights_only=False)
            vh = torch.load(vec_path, map_location=dev, weights_only=True).float()
            k = max(1, round(k_frac * vh.shape[1]))
            subs = _subspaces(vh, k, dev)
            entry: dict = {"k": k, "k_frac": k_frac, "d": vh.shape[1],
                           "tokens": get_token_count(model, step)}
            for leaf in leaves:
                if leaf not in d:
                    continue
                sigma = _centered_cov(d[leaf], dev)
                base = torch.linalg.eigvalsh(sigma.double()).flip(0).clamp_min(0).float()
                res = {"base": base.cpu(), "rankme_base": _rankme(base),
                       "where": _where(d[leaf], vh, k, dev)}
                for nm, q in subs.items():
                    ev = _deflate(sigma, q)
                    res[nm] = ev.cpu()
                    res[f"rankme_{nm}"] = _rankme(ev)
                entry[leaf] = res
            done[step] = entry
            torch.save(out, out_path)
            a = entry.get(leaves[0], {})
            w = a.get("where", {})
            print(f"{model} step{step}: k={k} base={a.get('rankme_base', 0):.0f} "
                  f"dark={a.get('rankme_dark', 0):.0f} top1={a.get('rankme_top1', 0):.0f} "
                  f"topk={a.get('rankme_topk', 0):.0f} rand={a.get('rankme_random', 0):.0f}"
                  f" | dark share chance={w.get('chance', 0):.3f} mean={w.get('mean', 0):.3f} "
                  f"uncent={w.get('top_uncentered', 0):.3f} cent={w.get('top_centered', 0):.3f}",
                  flush=True)
        miss = len(steps) - len([s for s in steps if s in done])
        print(f"COVERAGE {model}: {len(steps) - miss}/{len(steps)} checkpoints"
              + (f"  <-- {miss} MISSING, results are PARTIAL" if miss else ""), flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
