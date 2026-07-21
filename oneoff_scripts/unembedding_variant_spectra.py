"""Corrected unembedding singular spectra across checkpoints: for each model and checkpoint
of the 20-log schedule, fetch the head weight (selective loader) and store the singular
values of four spectra:
    standard        W
    freq_centered   W − 1·mᵀ, m = Wᵀf (empirical unigram f from the collection mix);
                    shifts every logit equally, so softmax-invariant
    mean_deflated   W(I − μ̂μ̂ᵀ), μ = the after_final_norm stream mean at the same step
                    (acts_mean_vec from the samples-run results; falls back to the
                    final_stream_svd files; skipped where neither is available)
    freq_mean       (W − 1·mᵀ)(I − μ̂μ̂ᵀ), both corrections
Writes {model_short: {step: {variant: Tensor(min(d, V))}}} to
data/results/unembedding_variant_spectra.pt. Incremental: existing (model, step) entries
are skipped."""

import os
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch

from utils.model_registry import (_cache_path, _fam, delete_cached_revision, get_checkpoint_schedule,
                                  get_model_config, load_selective_weights)

MODELS = ("EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-410m-deduped",
          "EleutherAI/pythia-1b-deduped", "EleutherAI/pythia-6.9b-deduped",
          "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-1124-7B", "nanochat-d12")
MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}
STREAM_DIR = "data/inferences/final_stream_svd"
_freq_cache: dict = {}
_mean_cache: dict = {}


def _freq(family: str, vocab: int, dev: str) -> torch.Tensor:
    if family not in _freq_cache:
        _freq_cache[family] = torch.load(MIXES[family], weights_only=False).flatten()
    f = torch.bincount(_freq_cache[family], minlength=vocab).to(dev, torch.float32)
    return f / f.sum()


def _stream_mean(short: str, step: int) -> "torch.Tensor | None":
    if short not in _mean_cache:
        cfg = "nanochat_samples" if short == "nanochat-d12" else "block_representations_samples"
        p = f"data/results/{cfg}/results_{short}.npy"
        r = np.load(p, allow_pickle=True).item() if os.path.exists(p) else {}
        _mean_cache[short] = {int(s): d["after_final_norm"]["acts_mean_vec"] for s, d in r.items()}
    if step in _mean_cache[short]:
        return torch.as_tensor(np.asarray(_mean_cache[short][step]))
    p = f"{STREAM_DIR}/{short}/step{step}.pt"   # steps outside the samples schedule
    if not (os.path.exists(p) and zipfile.is_zipfile(p)):
        return None
    return torch.load(p, map_location="cpu", weights_only=False)["after_final_norm"]["acts_mean"]


def main(out_path: str = "data/results/unembedding_variant_spectra.pt",
         max_checkpoints: int = 20, spacing: str = "log") -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for name in MODELS:
        config = get_model_config(name)
        head = _fam(config).head
        short = name.split("/")[-1]
        done = out.setdefault(short, {})
        for step, revision, repo in get_checkpoint_schedule(config, max_checkpoints, spacing):
            if step in done and ("mean_deflated" in done[step] or _stream_mean(short, step) is None):
                continue   # complete, or still lacking a valid stream file to add the mean variants
            w = load_selective_weights(config, repo, revision, [head])[head].weight.to(dev, torch.float32)
            m = w.T @ _freq(config.family, w.shape[0], dev)
            res = {"standard": torch.linalg.svdvals(w).cpu(),
                   "freq_centered": torch.linalg.svdvals(w - m[None, :]).cpu()}
            if (mu := _stream_mean(short, step)) is not None:
                muh = (mu / mu.norm()).to(dev, torch.float32)
                for tag, a in (("mean_deflated", w), ("freq_mean", w - m[None, :])):
                    res[tag] = torch.linalg.svdvals(a - torch.outer(a @ muh, muh)).cpu()
            done[step] = res
            for k in (f"{head}.weight", f"{head}.bias"):    # don't hoard the cached tensor
                p = _cache_path(config.family, repo, revision, k)
                if os.path.exists(p):
                    os.remove(p)
            if config.family != "nanochat":                 # nor the full snapshot the miss pulled
                delete_cached_revision(repo, revision)
            torch.save(out, out_path)
            print(f"{short} step{step}: " + "  ".join(f"{k} σ0={float(v[0]):.1f}" for k, v in res.items()),
                  flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
