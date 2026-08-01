"""What are the biggest directions at after_final_norm, seen through the unembedding?

rho and LogitVar have only ever been asked of NEURON write directions. This asks them of the
stream's own top eigendirections: take the centred after-final-norm covariance, and for each
of its leading eigenvectors u_j report

  rho(k)        ||V0(k)^T u_j||, the share of u_j in the BOTTOM k right singular directions of
                W_U -- the null space, same definition the neuron work uses
  top_share(k)  ||Vtop(k)^T u_j||, the share in the TOP k, i.e. what actually reaches logits
  logitvar      variance over the vocabulary of the row-normalised logit attribution of u_j
  var_share     lambda_j / sum(lambda), how much of the stream this direction is
  d_rankme      rankme(spectrum[n:]) - rankme(spectrum[n-1:]): what removing direction n does
                to the effective rank once every LARGER direction is already gone. Positive
                means the stream is higher-rank without it. At n = 1 it reduces to
                rankme(without the top direction) - rankme(everything)
  d_rankme_solo the same for direction n alone, every other direction left in place

Writes {model: {step: {...}}} to data/results/direction_identity.pt.
"""

import copy
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from oneoff_scripts.neuron_ablation_curve import MIXES, _capture
from utils.model_registry import (delete_cached_revision, get_checkpoint_schedule,
                                  get_final_layernorm, get_head_module, get_model_config,
                                  get_token_count, load_model)

K_FRACS = (0.001, 0.01, 0.05, 0.1)     # null-space / top-space sizes, as fractions of d_model


def _rankme(ev, eps=1e-12):
    """exp of the matrix entropy of an eigenvalue spectrum -- the project's rankme."""
    p = ev.clamp_min(0) / ev.clamp_min(0).sum().clamp_min(eps)
    return float(torch.exp(-(p * p.clamp_min(eps).log()).sum()))


def _spectrum(x):
    """Eigenvalues (descending) and eigenvectors of the centred covariance of x (n, d)."""
    xc = (x.detach() - x.detach().mean(0)).double()   # the final norm's gain is a Parameter
    ev, u = torch.linalg.eigh(xc.T @ xc / xc.shape[0])
    return ev.flip(0), u.flip(1)


def _shares(vh, u, ks):
    """Per direction (columns of u): share in the bottom k and in the top k right singular
    directions of the unembedding. vh rows are ordered by descending singular value."""
    return {k: ((vh[-k:] @ u).norm(dim=0).cpu(), (vh[:k] @ u).norm(dim=0).cpu()) for k in ks}


def _logitvar(w_eff, u):
    """Row-normalised logit attribution of each direction, variance over the vocabulary."""
    return ((w_eff @ u) / w_eff.norm(dim=1, keepdim=True).clamp_min(1e-12)).var(dim=0).cpu()


def main(out_path: str = "data/results/direction_identity.pt",
         model_name: str = "allenai/OLMo-2-0425-1B",
         steps: tuple = (480_000, 1_900_000), n_dirs: int = 32,
         num_windows: int = 1024, batch_size: int = 8, max_rows: int = 262_144,
         keep_cached: bool = True, max_checkpoints: int = 0, spacing: str = "log") -> None:
    """n_dirs leading directions per checkpoint. Same capture and token budget as the ablation
    curves, so the spectra are comparable with them. max_checkpoints > 0 sweeps the schedule
    at that many points instead of using the explicit steps."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(model_name)
    short = model_name.split("/")[-1]
    tokens = torch.load(MIXES[config.family], weights_only=True)[:num_windows]
    schedule = {s: (rev, repo) for s, rev, repo in get_checkpoint_schedule(config, None)}
    if max_checkpoints:      # the scheduler treats its argument as a hint, so thin it here
        sched = [s for s, _, _ in get_checkpoint_schedule(config, max_checkpoints, spacing) if s]
        idx = torch.linspace(0, len(sched) - 1, min(max_checkpoints, len(sched)))
        steps = tuple(sorted({sched[int(i)] for i in idx.round()}))
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    res = out.setdefault(short, {})

    for step in steps:
        revision, repo = schedule[step]
        model = torch.nn.Module.to(load_model(config, repo, revision), dev).eval()
        nrm: Any = copy.deepcopy(get_final_layernorm(model, config)).float()
        gamma = getattr(nrm, "weight", None)
        gamma = None if gamma is None else gamma.detach()
        head = get_head_module(model, config).weight.detach().float()
        w_eff = head - head.mean(0, keepdim=True)
        w_eff = w_eff * gamma.float() if gamma is not None else w_eff
        vh = torch.linalg.svd(w_eff, full_matrices=False).Vh

        h = _capture(model, config, {}, tokens, batch_size, dev, max_rows)["h"]
        ev, u = _spectrum(nrm(h).float())
        base = _rankme(ev)
        top = u[:, :n_dirs].float()
        ks = [max(1, round(f * head.shape[1])) for f in K_FRACS]
        shares = _shares(vh, top, ks)
        # marginal: direction j removed once every larger one already is
        d_rankme = torch.tensor([_rankme(ev[j + 1:]) - _rankme(ev[j:]) for j in range(n_dirs)])
        d_solo = torch.tensor([_rankme(torch.cat([ev[:j], ev[j + 1:]])) - base
                               for j in range(n_dirs)])
        res[step] = {"tokens": get_token_count(model_name, step), "n": h.shape[0],
                     "rankme": base, "eigenvalues": ev[:n_dirs].float().cpu(),
                     # the whole spectrum, so every rank quantity stays recomputable offline
                     "spectrum": ev.float().cpu(),
                     "var_share": (ev[:n_dirs] / ev.sum()).float().cpu(),
                     "d_rankme": d_rankme, "d_rankme_solo": d_solo,
                     "logitvar": _logitvar(w_eff, top),
                     "rho": {k: v[0] for k, v in shares.items()},
                     "top_share": {k: v[1] for k, v in shares.items()},
                     "d_model": head.shape[1], "ks": ks}
        print(f"{short} step{step}: rankme {base:.1f}; top direction var_share "
              f"{res[step]['var_share'][0]:.3f}, d_rankme {d_rankme[0]:+.1f}, "
              f"rho(k={ks[1]}) {res[step]['rho'][ks[1]][0]:.3f}, "
              f"logitvar {res[step]['logitvar'][0]:.3e}", flush=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(out, out_path)
        del model, h, u
        torch.cuda.empty_cache()
        if not keep_cached and config.family != "nanochat":
            delete_cached_revision(repo, revision)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
