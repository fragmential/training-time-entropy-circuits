"""Unembedding singular spectra across checkpoints (plan.md item 1a): for each model and
checkpoint of the 20-log schedule, fetch only the head weight (selective loader; cache
under $WEIGHT_CACHE_DIR) and store its singular values.
Writes {model_short: {step: Tensor(min(d, V))}} to data/results/unembedding_spectra.pt.

With vectors_dir set (plan.md item 1b), additionally saves the RIGHT singular vectors
(the head's stream-space basis, Vh (d, d), fp32) to {vectors_dir}/{short}/step{t}.pt —
needed for the stream ↔ unembedding alignment; values-only entries are re-fetched when
their vector file is missing."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_cache_path, _fam, delete_cached_revision, get_checkpoint_schedule,
                                  get_model_config, load_selective_weights)

MODELS = ("EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-410m-deduped",
          "EleutherAI/pythia-1b-deduped", "EleutherAI/pythia-6.9b-deduped",
          "allenai/OLMo-2-0425-1B", "allenai/OLMo-2-1124-7B", "nanochat-d12")


def main(out_path: str = "data/results/unembedding_spectra.pt",
         max_checkpoints: int = 20, spacing: str = "log",
         vectors_dir: "str | None" = None, models: tuple = MODELS,
         only_steps: tuple = ()) -> None:
    """only_steps restricts to an explicit checkpoint list, for filling a specific gap (e.g.
    the dense late-training grid) without walking the whole schedule. Every revision this
    fetches is deleted again before the next one, so at most one model sits on disk."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    wanted = set(only_steps)
    for name in models:
        config = get_model_config(name)
        head = _fam(config).head
        short = name.split("/")[-1]
        done = out.setdefault(short, {})
        sched = get_checkpoint_schedule(config, None if wanted else max_checkpoints, spacing)
        for step, revision, repo in sched:
            if wanted and step not in wanted:
                continue
            vec_path = os.path.join(vectors_dir, short, f"step{step}.pt") if vectors_dir else None
            if step in done and (vec_path is None or os.path.exists(vec_path)):
                continue
            w = load_selective_weights(config, repo, revision, [head])[head].weight
            if vec_path is not None:
                os.makedirs(os.path.dirname(vec_path), exist_ok=True)
                _, s, vh = torch.linalg.svd(w.to(dev, torch.float32), full_matrices=False)
                torch.save(vh.cpu(), vec_path)
                done[step] = s.cpu()
            else:
                done[step] = torch.linalg.svdvals(w.to(dev, torch.float32)).cpu()
            for k in (f"{head}.weight", f"{head}.bias"):    # don't hoard the cached tensor
                p = _cache_path(config.family, repo, revision, k)
                if os.path.exists(p):
                    os.remove(p)
            if config.family != "nanochat":                 # nor the full snapshot the miss pulled
                delete_cached_revision(repo, revision)
            torch.save(out, out_path)
            print(f"{short} step{step}: {tuple(w.shape)} top σ={float(done[step][0]):.3f}", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
