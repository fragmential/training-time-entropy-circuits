"""Where do the last block's compressing units write, and where do its decompressing ones?

rho asks one yes/no question -- is a unit's write in the unembedding's bottom 1%? -- and in
Pythia it does not separate the units that compress the final stream from the units that
decompress it. This describes both groups across the WHOLE unembedding spectrum instead of
one end of it, following the banded projections of Cancedda, Spectral Filters, Dark Signals,
and Attention Sinks (arXiv:2402.09221), and adds the two scalars that name the two candidate
second routes:

    dark_share    fraction of a unit's write direction in the bottom-k right singular
                  directions of W_U -- rho, as in Stolfo et al., Confidence Regulation
                  Neurons in Language Models (arXiv:2406.16254)
    bright_share  the same for the TOP direction v1, whose token-side image tracks token
                  frequency; the frequency route writes here, not in the null space
    bright_mass   RMS(a_i) * |v1^T w_out^(i)| -- the magnitude version. Stolfo et al identify
                  token frequency neurons by an ablation effect, NOT by a high weight norm,
                  so a frequency neuron can be an ordinary-weight, high-activation unit and
                  a direction-only statistic would miss it. Same for dark_mass.
    sink_cos      SIGNED cosine with the sink direction s. Units that CANCEL the sink write
                  score negative; the cancellation hypothesis predicts the decompressors sit
                  there. Unsigned magnitude would hide exactly the distinction being tested.

s is picked without a per-model table: over the first half of the blocks, the mlp write whose
top centered eigenvector holds the largest share of that write's own variance. In Pythia that
selects the known rank-collapsed early write; in OLMo-2 no write is concentrated and the
reported share says so.

Compression itself is NOT recomputed: `solo` (dRankMe/d(ablation strength), taken through the
network) is read from the ablation-curve files, so the groups here are the same groups those
curves rank. Writes data/results/compression_routes.pt. GPU node.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from oneoff_scripts.neuron_ablation_curve import _col_rms, _unit_module
from oneoff_scripts.unembedding_spectra import MODELS
from utils.model_registry import (delete_cached_revision, get_checkpoint_schedule,
                                  get_final_layernorm, get_head_module, get_model_config,
                                  get_num_layers, get_post_mlp_norm, get_token_count, load_model)
from utils.nullspace import head_subspace

FULL = {n.split("/")[-1]: n for n in MODELS}    # result files key on the short name; the HF
ABL = "data/results/ablation_curves/mlp_mean.pt"                     # repo does not resolve from it
MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}
N_BANDS = 20          # Cancedda's 20 equal bands of the unembedding spectrum, brightest first
MAX_ROWS = 262_144


@torch.no_grad()
def _run(model, config, blocks, last, n_ff, tokens, batch_size, dev, max_rows):
    """Last-block MLP activations (host) and a streaming covariance of each early mlp write."""
    cur: dict = {}
    cov: dict = {}
    hs = [_unit_module(model, config, last, "mlp").register_forward_pre_hook(
        lambda _m, args: cur.__setitem__("a", args[0].detach()))]
    hs += [_unit_module(model, config, b, "mlp").register_forward_hook(
        lambda _m, _a, o, k=b: cur.__setitem__(f"w{k}", o.detach())) for b in blocks]
    acts = torch.empty(max_rows, n_ff, dtype=torch.float32)     # preallocated: torch.cat at
    n = 0                                                       # this size doubles peak host RAM
    for i in range(0, tokens.shape[0], batch_size):
        if n >= max_rows:
            break
        model.forward(tokens[i:i + batch_size].to(dev))
        a = cur["a"].reshape(-1, cur["a"].shape[-1])
        rows = min(a.shape[0], max_rows - n)
        acts[n:n + rows] = a[:rows].float().cpu()
        for b in blocks:
            w = cur[f"w{b}"].reshape(-1, cur[f"w{b}"].shape[-1])[:rows].double()
            w = w - w.mean(0, keepdim=True)
            cov[b] = w.T @ w if b not in cov else cov[b] + w.T @ w
        n += rows
    for h in hs:
        h.remove()
    return acts[:n], cov, n


def _sink_direction(cov: dict) -> "tuple[torch.Tensor, int, float]":
    """(direction, block, top-eigenvalue share) for the most concentrated early mlp write."""
    best: "tuple[torch.Tensor, int, float]" = (torch.zeros(1), -1, 0.0)
    for b, c in cov.items():
        lam, vecs = torch.linalg.eigh(c)
        share = float(lam[-1] / lam.clamp_min(0).sum().clamp_min(1e-30))
        if share > best[2]:
            best = (vecs[:, -1].float(), b, share)
    return best


def _bands(vh: torch.Tensor, unit_w: torch.Tensor) -> torch.Tensor:
    """(N_BANDS, d_ff): share of each unit write direction's norm per spectral band."""
    d = vh.shape[0]
    edges = [round(i * d / N_BANDS) for i in range(N_BANDS + 1)]
    return torch.stack([(vh[a:b] @ unit_w).norm(dim=0) for a, b in zip(edges, edges[1:])])


@torch.no_grad()
def analyse(model_name: str, steps: list, batch_size: int, dev: str, max_rows: int,
            keep_cached: bool) -> dict:
    config = get_model_config(FULL.get(model_name, model_name))
    tokens = torch.load(MIXES[config.family], weights_only=True)
    res: dict = {}
    for step, revision, repo in get_checkpoint_schedule(config, None, "linear"):
        if step not in steps:
            continue
        loaded = load_model(config, repo, revision)
        last = get_num_layers(loaded, config) - 1
        model: Any = torch.nn.Module.to(loaded, dev).eval()
        early = list(range(max(1, (last + 1) // 2)))
        w = _unit_module(model, config, last, "mlp").weight.detach().float()
        acts, cov, n = _run(model, config, early, last, w.shape[1], tokens, batch_size, dev,
                            max_rows)
        s, s_block, s_share = _sink_direction(cov)

        gamma = getattr(get_final_layernorm(model, config), "weight", None)
        head = get_head_module(model, config).weight.detach()
        k = max(1, round(0.01 * head.shape[1]))
        vh = head_subspace(head, gamma, head.shape[1])          # all right singular vectors
        post = get_post_mlp_norm(model, config, last)
        if post is not None:                    # OLMo-2 post-feedforward norm: fold the gain,
            w = w * post.weight.detach().float()[:, None]       # or the direction is wrong
        rms = _col_rms(acts).to(dev)
        unit_w = w / w.norm(dim=0).clamp_min(1e-12)

        res[step] = {"dark_share": (vh[-k:] @ unit_w).norm(dim=0).cpu(),
                     "bright_share": (vh[:1] @ unit_w).norm(dim=0).cpu(),
                     "dark_mass": ((vh[-k:] @ w) * rms).norm(dim=0).cpu(),
                     "bright_mass": ((vh[:1] @ w) * rms).norm(dim=0).cpu(),
                     "sink_cos": (s.to(dev) @ unit_w).cpu(),
                     "bands": _bands(vh, unit_w).cpu(),
                     "act_rms": rms.cpu(), "w_norm": w.norm(dim=0).cpu(),
                     "sink_block": s_block, "sink_share": s_share, "k": k, "block": last,
                     "n": n, "tokens": get_token_count(model_name, step)}
        print(f"{model_name} step{step}: sink=blk{s_block} (top-eig share {s_share:.2f}), "
              f"{n} rows, k={k}", flush=True)
        del model, acts, cov
        torch.cuda.empty_cache()
        if not keep_cached and config.family != "nanochat":
            delete_cached_revision(repo, revision)
    return res


def main(out_path: str = "data/results/compression_routes.pt", abl_path: str = ABL,
         models: tuple = (), batch_size: int = 32, max_rows: int = MAX_ROWS,
         keep_cached: bool = False) -> None:
    """Steps are taken from the ablation-curve file, so the groups match its `solo` exactly."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    abl = torch.load(abl_path, weights_only=False)
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for model in (models or tuple(abl)):
        blocks = abl[model]
        steps = sorted(blocks[max(blocks)])
        todo = [s for s in steps if s not in out.get(model, {})]
        if not todo:
            continue
        got = analyse(model, todo, batch_size, dev, max_rows, keep_cached)
        out.setdefault(model, {}).update(got)
        torch.save(out, out_path)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
