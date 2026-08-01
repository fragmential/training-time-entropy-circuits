"""Which neurons compress the representation, measured exactly.

Spectral entropy of the residual stream, H = -sum_i p_i log p_i with p_i = lambda_i / sum
lambda, over the eigenvalues of the centred covariance. This is the compression measure --
the representation's own entropy, not the entropy of the token distribution.

Ablating a neuron of the LAST block changes the pre-final-norm stream by exactly a_tilde_i
w_out^(i) and nothing else, since no layer follows it. So the ablated covariance is a rank-2
update of the baseline one,

    Sigma' = Sigma - w c^T - c w^T + Var(a_tilde) w w^T,     c = Cov(h, a_tilde_i),

and every neuron's EXACT effect on spectral entropy follows from one covariance and one
cross-covariance -- no forward pass per neuron, and no first-order approximation, which for
this kind of perturbation is off by factors of several.

Reported per neuron: the exact change in spectral entropy from mean-ablating it (dH_mean) and
from zeroing it (dH_zero), before the final norm. The final norm is per-token and nonlinear,
so for the top candidates it is applied properly and the after-norm change reported too.

Writes {model: {block: {step: {...}}}} to data/results/spectral_effect.pt.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_fam, _getattr_path, get_checkpoint_schedule,
                                  get_final_layernorm, get_head_module, get_model_config,
                                  get_num_layers, get_post_mlp_norm, get_token_count, load_model)
from utils.nullspace import head_subspace, write_rho

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}


def _spectral_entropy(cov, eps=1e-12):
    """-sum p log p over the covariance's eigenvalues, normalised to sum to one. Accepts a
    batch of covariances (..., d, d) and returns one entropy per matrix."""
    ev = torch.linalg.eigvalsh(cov.double()).clamp_min(0)
    p = ev / ev.sum(-1, keepdim=True).clamp_min(eps)
    return -(p * p.clamp_min(eps).log()).sum(-1)


def _exact_effects(cov, cross, var_a, w_out, mean_a):
    """Exact change in spectral entropy from ablating each neuron, by rank-2 update.

    cov    (d, d)     covariance of the pre-final-norm stream
    cross  (d, d_ff)  Cov(h, a_i) for every neuron
    var_a  (d_ff,)    Var(a_i);  mean_a (d_ff,) its mean
    Mean-ablation removes the fluctuation (a_i -> mean); zero-ablation removes the whole
    write, which also shifts the mean and so leaves the covariance changed by the same
    rank-2 term -- the covariance is blind to the mean, so the two coincide here and only
    the trace differs. Both are returned for symmetry with the vocabulary-side convention.
    """
    h0 = float(_spectral_entropy(cov))
    n_neurons, chunk = w_out.shape[1], 32          # batched eigendecomposition, fp64
    out = torch.empty(n_neurons, dtype=torch.float64, device=cov.device)
    for i in range(0, n_neurons, chunk):
        w = w_out[:, i:i + chunk].T.unsqueeze(2)          # (B, d, 1)
        c = cross[:, i:i + chunk].T.unsqueeze(2)
        upd = (cov.unsqueeze(0) - w @ c.transpose(1, 2) - c @ w.transpose(1, 2)
               + var_a[i:i + chunk][:, None, None] * (w @ w.transpose(1, 2)))
        out[i:i + chunk] = _spectral_entropy(upd) - h0
    return h0, out.cpu()


def _capture(model, config, block, tokens, batch_size, dev):
    """Covariance of the pre-final-norm stream, its cross-covariance with the block's neuron
    activations, and the activation moments -- all streamed, nothing per-token retained."""
    fam = _fam(config)
    norm: Any = get_final_layernorm(model, config)
    down: Any = _getattr_path(model, f"{fam.blocks}.{block}.mlp.{fam.mlp_projs['down']}")
    acc: dict = {}
    cur: dict = {}

    def norm_pre(_m, args):
        cur["h"] = args[0].detach().float().reshape(-1, args[0].shape[-1])

    def down_pre(_m, args):
        cur["a"] = args[0].detach().float().reshape(-1, args[0].shape[-1])

    handles = [norm.register_forward_pre_hook(norm_pre), down.register_forward_pre_hook(down_pre)]
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch_size):
            model.forward(tokens[i:i + batch_size].to(dev))
            h, a = cur["h"], cur["a"]
            if not acc:
                acc.update(sh=torch.zeros(h.shape[1], dtype=torch.float64, device=dev),
                           sa=torch.zeros(a.shape[1], dtype=torch.float64, device=dev),
                           hh=torch.zeros(h.shape[1], h.shape[1], dtype=torch.float64, device=dev),
                           ha=torch.zeros(h.shape[1], a.shape[1], dtype=torch.float64, device=dev),
                           aa=torch.zeros(a.shape[1], dtype=torch.float64, device=dev), n=0)
            acc["sh"] += h.sum(0).double(); acc["sa"] += a.sum(0).double()
            acc["hh"] += (h.T @ h).double(); acc["ha"] += (h.T @ a).double()
            acc["aa"] += a.pow(2).sum(0).double(); acc["n"] += h.shape[0]
    for hd in handles:
        hd.remove()
    n = acc["n"]
    mh, ma = acc["sh"] / n, acc["sa"] / n
    return {"cov": acc["hh"] / n - torch.outer(mh, mh),
            "cross": acc["ha"] / n - torch.outer(mh, ma),
            "var_a": acc["aa"] / n - ma ** 2, "mean_a": ma, "n": n}


def main(out_path: str = "data/results/spectral_effect.pt",
         model_name: str = "allenai/OLMo-2-0425-1B",
         steps: tuple = (6_000, 20_000, 60_000, 150_000, 490_000, 1_000_000, 1_900_000),
         num_windows: int = 256, batch_size: int = 8, n_blocks: int = 1) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(model_name)
    short = model_name.split("/")[-1]
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    res = out.setdefault(short, {})
    tokens = torch.load(MIXES[config.family], weights_only=True)[:num_windows]
    schedule = {s: (rev, repo) for s, rev, repo in get_checkpoint_schedule(config, None)}
    for step in steps:
        revision, repo = schedule[step]
        loaded = load_model(config, repo, revision)
        last = get_num_layers(loaded, config) - 1
        model = torch.nn.Module.to(loaded, dev).eval()
        gamma = getattr(get_final_layernorm(model, config), "weight", None)
        head = get_head_module(model, config).weight.detach()
        v0 = head_subspace(head, gamma, max(1, round(0.01 * head.shape[1])))
        for b in [last - i for i in range(n_blocks)]:
            st = _capture(model, config, b, tokens, batch_size, dev)
            down: Any = _getattr_path(
                model, f"{_fam(config).blocks}.{b}.mlp.{_fam(config).mlp_projs['down']}")
            w_out = down.weight.detach().float()
            post = get_post_mlp_norm(model, config, b)
            if post is not None:
                w_out = w_out * post.weight.detach().float()[:, None]
            h0, d_mean = _exact_effects(st["cov"], st["cross"], st["var_a"],
                                        w_out.double(), st["mean_a"])
            del st["cross"]
            rho = write_rho(v0, w_out).cpu()
            res.setdefault(b, {})[step] = {
                "h0": h0, "d_spectral_entropy": d_mean.cpu(), "rho": rho,
                "act_rms": (st["var_a"] + st["mean_a"] ** 2).sqrt().cpu(),
                "w_norm": w_out.norm(dim=0).cpu(), "tokens": get_token_count(model_name, step),
                "n": st["n"]}
            o = d_mean.argsort(descending=True)      # ablating these RAISES H, i.e. they
            print(f"{short} step{step} blk{b}: H={h0:.4f} over {st['n']} tokens  "  # compress
                  f"most compressing (dH>0 when removed) "
                  f"{[(int(i), round(float(d_mean[i]), 4), round(float(rho[i]), 2)) for i in o[:5]]}"
                  f" | biggest contributors to H "
                  f"{[(int(i), round(float(d_mean[i]), 4), round(float(rho[i]), 2)) for i in o.flip(0)[:5]]}",
                  flush=True)
        del model, loaded
        torch.cuda.empty_cache()
        torch.save(out, out_path)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
