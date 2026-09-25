"""Stream ↔ unembedding CKA (docs/cka_alignment.md) per model, checkpoint, leaf, and head
correction: s = tr(Σc·G) / (‖Σc‖·‖G‖), with Σc the centered stream covariance rebuilt from
the final_stream_svd files (V diag(λ) Vᵀ − μμᵀ) and G = W̃ᵀW̃ for W̃ ∈ {standard,
freq_centered, mean_deflated, freq_mean} (the unembedding_variant_spectra.py corrections;
the deflation μ̂ is the same stream file's mean). Controls: rotated-W (W → WQ, Q seeded
orthogonal, seeds 0–2) for standard and freq_centered. Also stores the freq-centered
head's right singular basis under data/results/unembedding_svd_freq/ — together with the
standard basis in unembedding_svd/, every variant's Gram is offline-derivable.
Writes {model: {step: {leaf: {key: float}}}} to data/results/unembedding_cka.pt.
Steps without a valid stream-SVD file are skipped. Safe next to running collection jobs:
only hub revisions this script itself fetched are deleted."""

import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_cache_path, _fam, delete_cached_revision, get_checkpoint_schedule,
                                  get_model_config, load_selective_weights)

MODELS = ("EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-410m-deduped",
          "EleutherAI/pythia-1b-deduped", "EleutherAI/pythia-6.9b-deduped",
          "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-1124-7B", "nanochat-d12")
MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}
STREAM_DIR = "data/inferences/final_stream_svd"
VH_FREQ_DIR = "data/results/unembedding_svd_freq"
LEAVES = ("before_final_norm", "after_final_norm")
ROT_SEEDS = (0, 1, 2)
_freq_cache: dict = {}


def _freq(family: str, vocab: int, dev: str) -> torch.Tensor:
    if family not in _freq_cache:
        _freq_cache[family] = torch.load(MIXES[family], weights_only=False).flatten()
    f = torch.bincount(_freq_cache[family], minlength=vocab).to(dev, torch.float32)
    return f / f.sum()


def _hub_ref_exists(repo: str, revision: str) -> bool:
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.exists(os.path.join(hf_home, "hub", f"models--{repo.replace('/', '--')}",
                                       "refs", str(revision)))


def _cka(Sc: torch.Tensor, G: torch.Tensor) -> float:
    return float((Sc * G).sum() / (Sc.norm() * G.norm()).clamp(min=1e-30))


def main(out_path: str = "data/results/unembedding_cka.pt",
         stream_dir: str = STREAM_DIR, vh_dir: str = VH_FREQ_DIR,
         max_checkpoints: int = 20, spacing: str = "log") -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for name in MODELS:
        config = get_model_config(name)
        head = _fam(config).head
        short = name.split("/")[-1]
        if not os.path.isdir(os.path.join(stream_dir, short)):
            continue                                     # e.g. nanochat in the padded set
        done = out.setdefault(short, {})
        os.makedirs(os.path.join(vh_dir, short), exist_ok=True)
        for step, revision, repo in get_checkpoint_schedule(config, max_checkpoints, spacing):
            vh_path = os.path.join(vh_dir, short, f"step{step}.pt")
            if step in done and os.path.exists(vh_path):
                continue
            sp = f"{stream_dir}/{short}/step{step}.pt"
            try:                                        # fsvd files may be mid-rewrite; retry later
                if not zipfile.is_zipfile(sp):
                    continue
                stream = torch.load(sp, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"{short} step{step}: stream file unreadable ({type(e).__name__}), skipped", flush=True)
                continue
            prefetched = config.family == "nanochat" or _hub_ref_exists(repo, revision)
            w = None
            for attempt in range(4):                    # hub rate limits + transient timeouts
                try:
                    w = load_selective_weights(config, repo, revision, [head])[head].weight.to(dev, torch.float32)
                    break
                except Exception as e:
                    print(f"{short} step{step}: fetch failed ({type(e).__name__}), attempt {attempt + 1}/4", flush=True)
                    time.sleep(120)
            if w is None:
                continue                                # picked up by a later rerun
            m = w.T @ _freq(config.family, w.shape[0], dev)
            wf = w - m[None, :]
            if not os.path.exists(vh_path):              # head is stream-independent; save once
                torch.save(torch.linalg.svd(wf, full_matrices=False)[2].cpu(), vh_path)
            d = w.shape[1]
            Qs = []
            for s in ROT_SEEDS:
                g = torch.Generator().manual_seed(s)
                Qs.append(torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(dev))
            res: dict = {}
            for leaf in LEAVES:
                n = stream[leaf]
                V = n["acts_eigvecs"].to(dev, torch.float32)
                lam = n["acts_eigvals"].to(dev, torch.float32)
                mu = n["acts_mean"].to(dev, torch.float32)
                Sc = (V * lam) @ V.T - torch.outer(mu, mu)
                muh = mu / mu.norm().clamp(min=1e-30)
                grams = {"standard": w.T @ w, "freq_centered": wf.T @ wf}
                P = torch.eye(d, device=dev) - torch.outer(muh, muh)
                grams["mean_deflated"] = P @ grams["standard"] @ P
                grams["freq_mean"] = P @ grams["freq_centered"] @ P
                r = {k: _cka(Sc, G) for k, G in grams.items()}
                for v in ("standard", "freq_centered"):
                    for s, Q in zip(ROT_SEEDS, Qs):
                        r[f"{v}_rot{s}"] = _cka(Sc, Q.T @ grams[v] @ Q)
                res[leaf] = r
            done[step] = res
            for k in (f"{head}.weight", f"{head}.bias"):    # don't hoard the cached tensor
                p = _cache_path(config.family, repo, revision, k)
                if os.path.exists(p):
                    os.remove(p)
            if config.family != "nanochat" and not prefetched:   # delete only what we fetched
                delete_cached_revision(repo, revision)
            torch.save(out, out_path)
            print(f"{short} step{step}: " + "  ".join(
                f"{leaf.split('_')[0]} std={res[leaf]['standard']:.3f} freq={res[leaf]['freq_centered']:.3f} "
                f"rot={res[leaf]['standard_rot0']:.3f}" for leaf in LEAVES), flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
