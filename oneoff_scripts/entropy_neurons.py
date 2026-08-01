"""Entropy-neuron composition with the unembedding null space, across checkpoints.

Stolfo et al., Confidence Regulation Neurons in Language Models (arXiv:2406.16254) call a final-layer MLP neuron an *entropy neuron* when its
write direction w_out sits mostly in the effective null space of the unembedding — the span
V0 of the bottom-k right singular vectors of W_U — so it moves the residual norm (and hence
the final-norm gain, and hence logit temperature) without moving the logits directly. Their
statistic is

    rho_i = ||V0^T w_out^(i)|| / ||w_out^(i)||,        k = 12 (GPT-2) or ~0.01 d_model,

together with a high weight norm and a low LogitVar (variance over the vocabulary of the
row-normalised logit attribution). The paper never asks WHEN these neurons form; this script
does, by recomputing the three statistics at every checkpoint of the log schedule.

Weights only, no activations. Conventions:
  * The final norm's gain is folded into the head, as in the paper: W_eff = W_U @ diag(gamma).
  * --center (default on, NOT in the paper) additionally centres W_U over the vocabulary:
    softmax is shift-invariant, so the all-ones direction carries no information.
  * w_out^(i) is column i of the block's down-projection. --fold_post_norm (default on, not
    in the paper, OLMo-2 only) folds the post-feedforward norm's gain into it; that norm's
    normalisation is a scalar, so the direction stays well defined up to scale.
  * Blocks L-1 (the paper's layer) and L-2 (control) are measured; --mid_fracs adds blocks
    at given depth fractions, e.g. 0.5 for a mid-stack control with no direct path out.

Writes {model_short: {block_idx: {step: {"rho": (n_k, d_ff), "norm", "logitvar",
"tokens", "ks", "d_model"}}}} to data/results/entropy_neurons.pt.
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_cache_path, _fam, _norm_sd_prefix, delete_cached_revision,
                                  get_checkpoint_schedule, get_model_config, get_token_count,
                                  load_selective_weights)

MODELS = ("EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-410m-deduped",
          "EleutherAI/pythia-1b-deduped", "allenai/OLMo-2-0425-1B", "nanochat-d12")
POST_MLP_NORM = {"olmo": "post_feedforward_layernorm"}   # gain applied after the down-proj


def _n_layers(config) -> int:
    if config.family == "nanochat":
        m = re.search(r"d(\d+)", config.hf_repo)   # the tag's depth, e.g. nanochat-d12 -> 12
        if m is None:
            raise ValueError(f"Cannot read depth from nanochat tag '{config.hf_repo}'")
        return int(m.group(1))
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(config.hf_repo, trust_remote_code=config.trust_remote_code
                                      ).num_hidden_layers


def _prefixes(config, blocks):
    """(head, final-norm, {block: down-proj}, {block: post-mlp-norm}) state-dict prefixes."""
    fam = _fam(config)
    down = {b: f"{fam.blocks}.{b}.mlp.{fam.mlp_projs['down']}" for b in blocks}
    sub = POST_MLP_NORM.get(config.family)
    post = {b: f"{fam.blocks}.{b}.{sub}" for b in blocks} if sub else {}
    return fam.head, _norm_sd_prefix(config), down, post


def _stats(w_eff, vh, w_out, ks):
    """rho for each k, weight norm, and LogitVar, for every column of w_out (d, d_ff)."""
    norms = w_out.norm(dim=0)
    unit = w_out / norms.clamp_min(1e-12)
    rho = torch.stack([(vh[-k:] @ unit).norm(dim=0) for k in ks])
    logits = (w_eff @ unit) / w_eff.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return rho.cpu(), norms.cpu(), logits.var(dim=0).cpu()


def main(out_path: str = "data/results/entropy_neurons.pt", models: tuple = MODELS,
         max_checkpoints: int = 20, spacing: str = "log", n_blocks: int = 2,
         center: bool = True, fold_post_norm: bool = True, keep_weights: bool = True,
         only_steps: tuple = (), mid_fracs: tuple = (), keep_cached: bool = True) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for name in models:
        config = get_model_config(name)
        short = name.split("/")[-1]
        last = _n_layers(config) - 1
        blocks = sorted({last - i for i in range(n_blocks)} |
                        {round(f * last) for f in mid_fracs}, reverse=True)
        head, norm_p, down_p, post_p = _prefixes(config, blocks)
        done = out.setdefault(short, {})
        for step, revision, repo in get_checkpoint_schedule(config, max_checkpoints, spacing):
            if only_steps and step not in only_steps:
                continue
            if all(step in done.get(b, {}) for b in blocks):
                continue
            want = [head, norm_p, *down_p.values(), *post_p.values()]
            w = load_selective_weights(config, repo, revision, want)
            wu = w[head].weight.to(dev, torch.float32)
            gamma = w[norm_p].weight.to(dev, torch.float32) if norm_p in w else None
            w_eff = wu - wu.mean(0, keepdim=True) if center else wu
            if gamma is not None:
                w_eff = w_eff * gamma
            d = w_eff.shape[1]
            ks = sorted({1, 8, max(1, round(0.01 * d)), max(1, round(0.05 * d))})
            vh = torch.linalg.svd(w_eff, full_matrices=False).Vh    # rows = right singular vectors
            for b in blocks:
                w_out = w[down_p[b]].weight.to(dev, torch.float32)      # (d, d_ff)
                if fold_post_norm and b in post_p and post_p[b] in w:
                    w_out = w_out * w[post_p[b]].weight.to(dev, torch.float32)[:, None]
                rho, norms, lv = _stats(w_eff, vh, w_out, ks)
                done.setdefault(b, {})[step] = {
                    "rho": rho, "norm": norms, "logitvar": lv, "ks": ks, "d_model": d,
                    "tokens": get_token_count(name, step)}
            if not keep_weights:                    # else the few small tensors stay cached,
                for p in want:                      # so a re-run downloads nothing at all
                    for k in (f"{p}.weight", f"{p}.bias"):
                        path = _cache_path(config.family, repo, revision, k)
                        if os.path.exists(path):
                            os.remove(path)
            if not keep_cached and config.family != "nanochat":   # 7B snapshots are too big to hoard
                delete_cached_revision(repo, revision)
            torch.save(out, out_path)
            top = done[blocks[0]][step]["rho"]
            print(f"{short} step{step}: d={d} ks={ks} max rho={[round(float(r.max()), 3) for r in top]}",
                  flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
