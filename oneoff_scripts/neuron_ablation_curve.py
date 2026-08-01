"""Where in the unit population does the last layer's compression live?

The MLP's write is exactly sum_i a_i w_out^(i), so the block's effect IS its neurons'. Ablate
the top-k units for k = 1..all, ranked by after-norm entropy gradient, by rho, by void mass and
at random, and watch RankMe move from unchanged to the whole-branch effect. Attention heads are
the same experiment one unit up.

One model load per checkpoint, one file per model: {out_dir}/results_{model}.pt,
keyed by {unit}_{mode} -- so per-model jobs never share a file and can run in parallel.
"""

import copy
import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_fam, _getattr_path, delete_cached_revision,
                                  get_checkpoint_schedule, get_final_layernorm, get_head_module,
                                  get_model_config, get_num_layers, get_post_mlp_norm,
                                  get_token_count, load_model)
from utils.nullspace import head_subspace

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}
KS = (1, 4, 16, 64, 256, 1024, 4096, 100_000)      # the last one means "all of them"
KS_HEAD = (1, 2, 4, 8, 16, 32, 64, 1_000)          # attention heads, same idea
ROW_CHUNK = 16_384        # rows streamed from host memory at a time


@torch.no_grad()
def _metrics(x, eps=1e-12):
    """Named as scripts/compute_metrics.rankme_metrics names them: rankme is exp of the
    MATRIX entropy (the project's), true_rankme exp of the SINGULAR-value entropy. 27x apart."""
    xc = (x - x.mean(0)).double()
    ev = torch.linalg.eigvalsh(xc.T @ xc / xc.shape[0]).clamp_min(0)
    p = ev / ev.sum().clamp_min(eps)
    sv = ev.sqrt()
    q = sv / sv.sum().clamp_min(eps)
    h_ev = -(p * p.clamp_min(eps).log()).sum()
    h_sv = -(q * q.clamp_min(eps).log()).sum()
    return {"matrix_entropy": float(h_ev), "rankme": float(torch.exp(h_ev)),
            "true_rankme": float(torch.exp(h_sv)), "trace": float(ev.sum())}


def _col_rms(a):
    """Per-column RMS activation, streamed: no n x d_ff temporary."""
    acc = torch.zeros(a.shape[1], dtype=torch.float64)
    for i in range(0, a.shape[0], ROW_CHUNK):
        acc += a[i:i + ROW_CHUNK].double().pow(2).sum(0)
    return (acc / a.shape[0]).sqrt().float()


def _dh(h, nrm):
    """dH/dh at the final norm's input, H the after-norm spectral entropy."""
    hr = h.detach().clone().requires_grad_(True)
    gc = nrm(hr).float()
    gc = gc - gc.mean(0)
    ev = torch.linalg.eigvalsh(gc.T @ gc / gc.shape[0]).clamp_min(1e-12)
    p = ev / ev.sum()
    (-(p * p.log()).sum()).backward()
    return hr.grad


def _lerp_hook(mu, eps):
    """x -> x - (x - mu) * eps: eps = 0 leaves the unit alone, eps = 1 ablates it fully."""
    return lambda _m, args: ((args[0] - (args[0] - mu) * eps).to(args[0].dtype),)


def _model_gradient(model, config, block, unit, mu, gh, tokens, batch_size, dev, max_rows):
    """dH/d(ablation strength) for every unit, taken THROUGH the network.

    H couples all rows, so it cannot be differentiated batch by batch -- but dH/dh can be had
    from one cheap pass (_dh), and after that sum_n <dH/dh_n, h_n(eps)> has the same eps
    gradient and does split by batch. Autograd then carries the post-FF norm and everything
    downstream, which the closed form got wrong on OLMo-2."""
    eps = torch.zeros(mu.shape[0], device=dev, dtype=torch.float32, requires_grad=True)
    cur: dict = {}
    hs = [_unit_module(model, config, block, unit).register_forward_pre_hook(_lerp_hook(mu, eps)),
          get_final_layernorm(model, config).register_forward_pre_hook(
              lambda _m, args: cur.__setitem__("h", args[0]))]
    n = 0
    for i in range(0, tokens.shape[0], batch_size):
        if n >= max_rows:
            break
        model.forward(tokens[i:i + batch_size].to(dev))
        hb = cur["h"].reshape(-1, cur["h"].shape[-1])
        rows = min(hb.shape[0], max_rows - n)
        (hb[:rows].float() * gh[n:n + rows]).sum().backward()
        n += rows
    for hd in hs:
        hd.remove()
    return eps.grad.detach().cpu() if eps.grad is not None else torch.zeros(mu.shape[0])


def _ablated_stream(model, config, block, unit, idx, mu, tokens, batch_size, dev, max_rows):
    """The pre-final-norm stream with the selected units clamped to mu, measured by actually
    running the model. No closed form, so norms, residual adds and downstream blocks are the
    network's own."""
    def pre(_m, args):
        x = args[0].clone()
        x[..., idx] = mu[idx].to(x.dtype)
        return (x,)
    hd = _unit_module(model, config, block, unit).register_forward_pre_hook(pre)
    h = _capture(model, config, {}, tokens, batch_size, dev, max_rows)["h"]
    hd.remove()
    return h


def _capture(model, config, blocks, tokens, batch_size, dev, max_rows):
    """Pre-final-norm stream (GPU) and each unit's activations (host), kept as samples so the
    per-token norm can be reapplied after every ablation. units maps name -> the module whose
    INPUT is the activation vector. Buffers preallocated; torch.cat would double peak memory."""
    norm: Any = get_final_layernorm(model, config)
    cur: dict = {}
    handles = [norm.register_forward_pre_hook(
        lambda _m, args: cur.__setitem__("h", args[0].detach()))]
    handles += [mod.register_forward_pre_hook(
        lambda _m, args, k=name: cur.__setitem__(k, args[0].detach()))
        for name, mod in blocks.items()]
    bufs: dict = {}
    n = 0
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch_size):
            if n >= max_rows:
                break
            model.forward(tokens[i:i + batch_size].to(dev))
            rows = min(cur["h"].shape[0] * cur["h"].shape[1], max_rows - n)
            for k, v in cur.items():
                t = v.float().reshape(-1, v.shape[-1])[:rows]
                if k not in bufs:      # h stays resident, unit activations go to host memory
                    bufs[k] = torch.empty(max_rows, t.shape[1], dtype=torch.float32,
                                          device=dev if k == "h" else "cpu")
                bufs[k][n:n + rows] = t.to(bufs[k].device)
            n += rows
    for hd in handles:
        hd.remove()
    return {k: v[:n] for k, v in bufs.items()}


def _greedy_order(score, feat):
    """Redundancy-aware order: repeatedly take the highest-scoring unit after discounting the
    part of its write direction the chosen units already cover. Ranking by score alone picks
    units that duplicate each other, which is why the top-k by individual effect is not the
    best set of size k."""
    f = feat / feat.norm(dim=0).clamp_min(1e-12)
    d, n = f.shape
    q_basis = torch.zeros(d, min(d, n), device=f.device)
    resid = torch.ones(n, device=f.device)        # share of each write still uncovered
    taken = torch.zeros(n, dtype=torch.bool, device=f.device)
    order, t = [], 0
    while len(order) < n:
        # the span holds at most d directions; residuals stay faintly positive past that from
        # rounding, so bound on the basis too or the buffer overruns
        if t >= q_basis.shape[1] or resid.max() < 1e-6:
            return torch.cat([torch.tensor(order, device=f.device),
                              torch.argsort(score.masked_fill(taken, -torch.inf),
                                            descending=True)[:n - len(order)]])
        i = int((score * resid.sqrt()).masked_fill(taken, -torch.inf).argmax())
        order.append(i)
        taken[i] = True
        q = f[:, i] - q_basis[:, :t] @ (q_basis[:, :t].T @ f[:, i])
        if q.norm() > 1e-6:
            q_basis[:, t] = q / q.norm()
            resid -= (q_basis[:, t] @ f).pow(2)   # incremental: recomputing it is O(rank * n)
            t += 1
    return torch.tensor(order, device=f.device)


def _logitvar(head, gamma, w):
    """Variance over the vocabulary of the row-normalised logit attribution (Stolfo et al.).
    Low means the unit's write barely moves relative logits -- half their entropy-neuron
    filter, the other half being a high weight norm."""
    we = head.float() - head.float().mean(0, keepdim=True)
    we = we * gamma.float() if gamma is not None else we
    u = w / w.norm(dim=0).clamp_min(1e-12)
    return ((we @ u) / we.norm(dim=1, keepdim=True).clamp_min(1e-12)).var(dim=0)


def _clique_order(score, feat):
    """Correlated set: grow the clique whose writes REINFORCE each other. Seed with the top
    unit, then repeatedly take the highest score * signed cosine with the set's running sum.

    This is the direction the population's super-additivity points: ablating everything moves
    entropy ~117x more than the summed individual gradients, so what matters is units acting
    together, not a spread-out basis."""
    f = feat / feat.norm(dim=0).clamp_min(1e-12)
    taken = torch.zeros(f.shape[1], dtype=torch.bool, device=f.device)
    i = int(score.argmax())
    order, acc, taken[i] = [i], f[:, i].clone(), True
    for _ in range(f.shape[1] - 1):
        eff = score * ((acc / acc.norm().clamp_min(1e-12)) @ f)
        j = int(eff.masked_fill(taken, -torch.inf).argmax())
        order.append(j)
        taken[j] = True
        acc += f[:, j]
    return torch.tensor(order, device=f.device)


def _n_units(model, unit, n_cols) -> int:
    """A neuron owns one down-projection column; a head owns head_dim contiguous o_proj
    columns. Ablating columns instead ablates fractions of heads and never reaches all of them."""
    cfg = model.config
    return n_cols if unit == "mlp" else int(getattr(cfg, "num_attention_heads", 0) or cfg.n_head)


def _unit_module(model, config, block, unit):
    fam = _fam(config)
    sub = f"mlp.{fam.mlp_projs['down']}" if unit == "mlp" else fam.oproj
    return _getattr_path(model, f"{fam.blocks}.{block}.{sub}")


def main(out_dir: str = "data/results/ablation_curves",
         model_name: str = "allenai/OLMo-2-0425-1B",
         steps: tuple = (60_000, 490_000, 1_900_000),
         modes: tuple = ("mlp:mean", "mlp:zero", "attn:mean"),
         num_windows: int = 1024, batch_size: int = 8, max_rows: int = 262_144,
         n_blocks: int = 1, seed: int = 0, keep_cached: bool = True,
         rankings: tuple = ()) -> None:
    """modes are "<unit>:<ablate_mode>": mlp neurons or attn heads, a_i -> its dataset mean or
    -> 0. Mean and zero are identical before the norm (centred covariance ignores constants)
    and differ after it by exactly the mean write's contribution.

    max_rows 262144 is where a direct RankMe reproduces the main sweep; 65536 is biased low.
    keep_cached=False deletes each revision after use -- needed for the 7B models.
    rankings restricts which orders are measured (all of them when empty); a run merges into
    whatever is already in the file, so one ranking can be added without redoing the rest."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(model_name)
    short = model_name.split("/")[-1]
    units = sorted({m.split(":")[0] for m in modes})
    tokens = torch.load(MIXES[config.family], weights_only=True)[:num_windows]
    schedule = {s: (rev, repo) for s, rev, repo in get_checkpoint_schedule(config, None)}
    for step in steps:
        revision, repo = schedule[step]
        loaded = load_model(config, repo, revision)
        last = get_num_layers(loaded, config) - 1
        model = torch.nn.Module.to(loaded, dev).eval()
        # a float32 copy: the model runs in bf16, and feeding fp32 activations to a bf16
        # norm errors. Copying rather than casting in place leaves model.forward() intact.
        nrm: Any = copy.deepcopy(get_final_layernorm(model, config)).float()
        gamma = getattr(nrm, "weight", None)
        gamma = None if gamma is None else gamma.detach()   # a Parameter otherwise, and
        # every tensor derived from v0 would carry a graph into the saved scores
        head = get_head_module(model, config).weight.detach()
        v0 = head_subspace(head, gamma, max(1, round(0.01 * head.shape[1])))
        for b in [last - i for i in range(n_blocks)]:
            caught = _capture(model, config,
                              {u: _unit_module(model, config, b, u) for u in units},
                              tokens, batch_size, dev, max_rows)
            h = caught["h"]
            base_bfn, base_afn = _metrics(h), _metrics(nrm(h).float())
            gh = _dh(h, nrm)
            for mode in modes:
                unit, ablate_mode = mode.split(":")
                a = caught[unit]
                w = _unit_module(model, config, b, unit).weight.detach().float()
                post = get_post_mlp_norm(model, config, b) if unit == "mlp" else None
                if post is not None:
                    w = w * post.weight.detach().float()[:, None]
                n_g = _n_units(model, unit, w.shape[1])
                gs = w.shape[1] // n_g
                # rho is a fraction of a UNIT write direction, so it ignores both how big the
                # write is and how hard the unit fires. void_mass is the unnormalised
                # quantity: how much magnitude actually lands in the null space per token.
                # Both reduce to the neuron definitions when gs == 1.
                proj = (v0 @ w).view(-1, n_g, gs)
                rho = proj.norm(dim=(0, 2)) / w.view(-1, n_g, gs).norm(dim=(0, 2)).clamp_min(1e-12)
                void_mass = (v0 @ (w * _col_rms(a).to(dev))).view(-1, n_g, gs).norm(dim=(0, 2)).cpu()
                mu = (a.mean(0) if ablate_mode == "mean" else torch.zeros(a.shape[1])).to(dev)
                solo = _model_gradient(model, config, b, unit, mu, gh, tokens, batch_size,
                                       dev, max_rows).view(n_g, gs).sum(1)
                g = torch.Generator().manual_seed(seed)
                scores = {"effect": solo.to(dev), "rho": rho.to(dev),
                          "void_mass": void_mass.to(dev)}
                orders = {k: torch.argsort(v, descending=True) for k, v in scores.items()}
                orders["random"] = torch.randperm(n_g, generator=g).to(dev)
                # the paper detects entropy neurons by LOW LogitVar, so this one sorts ASCENDING
                lv = _logitvar(head, gamma, w).view(n_g, gs).mean(1)
                scores["logitvar"] = lv
                orders["logitvar"] = torch.argsort(lv)
                # the set-aware twin of each ranking; heads own several columns, so their write
                # is a subspace rather than a direction and the same construction does not apply
                if gs == 1:
                    desc = {k: v for k, v in scores.items() if k != "logitvar"}
                    orders.update({f"{k}_corr": _clique_order(v.float(), w)
                                   for k, v in desc.items()})
                    orders.update({f"{k}_set": _greedy_order(v.float(), w)
                                   for k, v in desc.items()})
                curve: dict = {"base_bfn": base_bfn, "base_afn": base_afn,
                               # under their own key: the loop below writes curve[<ranking>]
                               "scores": {k: v.cpu() for k, v in scores.items()},
                               "ablate_mode": ablate_mode,
                               "unit": unit, "n_units": n_g, "max_rows": max_rows,
                               "tokens": get_token_count(model_name, step), "n": h.shape[0]}
                cols = torch.arange(gs, device=dev)
                orders = {k: v for k, v in orders.items() if not rankings or k in rankings}
                for name, order in orders.items():
                    curve[name] = {}
                    ks = sorted({min(k, n_g) for k in (KS_HEAD if unit == "attn" else KS)})
                    for k in ks:
                        # a head owns gs contiguous columns of the output projection; a
                        # neuron owns one. Either way the whole unit goes at once.
                        idx = (order[:k, None] * gs + cols).flatten()
                        hk = _ablated_stream(model, config, b, unit, idx, mu, tokens,
                                             batch_size, dev, max_rows)
                        curve[name][k] = {"bfn": _metrics(hk), "afn": _metrics(nrm(hk).float())}
                        del hk
                    m = curve[name][ks[-1]]
                    print(f"{short} step{step} blk{b} {mode} rank-by-{name}: all {n_g} "
                          f"-> afn rankme {base_afn['rankme']:.1f} -> {m['afn']['rankme']:.1f}, "
                          f"bfn {base_bfn['rankme']:.1f} -> {m['bfn']['rankme']:.1f}", flush=True)
                # one file per model, as data/results/<experiment>/results_<model> everywhere
                # else: models never share a file, so per-model jobs run in parallel safely
                path = os.path.join(out_dir, f"results_{short}.pt")
                os.makedirs(out_dir, exist_ok=True)
                out = torch.load(path, weights_only=False) if os.path.exists(path) else {}
                at = out.setdefault(f"{unit}_{ablate_mode}", {}).setdefault(b, {})
                at[step] = {**at.get(step, {}), **curve}   # merge: keep rankings not rerun
                torch.save(out, path)
                del mu
            del caught, h, gh
            torch.cuda.empty_cache()
        del model, loaded
        torch.cuda.empty_cache()
        if not keep_cached and config.family != "nanochat":
            delete_cached_revision(repo, revision)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
