"""Which direction grows across the last blocks, is it in the unembedding null space, and
which neurons write it?

The ablation says the high-rho neurons carry most of the last MLP's output variance but
barely move the final stream's rank. So ask the question the other way round: find the
direction the stream actually gains across a block, then ask what wrote it.

Three directions per block, all from the centered covariances:
  ratio  top generalized eigenvector of (Sigma_out, Sigma_in) -- largest variance RATIO
  gain   top eigenvector of (Sigma_out - Sigma_in)            -- largest absolute GAIN
  afn    top eigenvector of the after-final-norm covariance   -- largest direction overall

For each direction v (unit): its share in the unembedding's bottom-k right-singular subspace
(the null space rho applies to) and in the top-k, against the sqrt(k/d) chance level. Then
per neuron of the block's MLP:
  cos_i    cos(w_out^(i), v)                    -- does it point that way at all
  contrib_i  RMS(a_i) * ||w_out^(i)|| * cos_i   -- how much it actually puts there
so a neuron that points at v but never fires is ranked below one that fires hard.

Self-contained: one forward pass per checkpoint accumulates the covariances and the neuron
activation statistics together. Writes data/results/grown_direction_neurons.pt.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_fam, _getattr_path, get_checkpoint_schedule,
                                  get_final_layernorm, get_head_module, get_model_config,
                                  get_num_layers, get_output_head, get_post_mlp_norm, load_model,
                                  load_tokenizer)
from utils.nullspace import head_subspace

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}


class _Cov:
    """Streaming mean and second moment, kept per half of the data as well as in total.

    The halves exist for a split-half check: a generalized eigendecomposition inverts
    Sigma_in, so its top eigenvector is the place where sample noise in the SMALL directions
    lands. If the direction found on one half does not point the same way as the one found
    on the other, the direction is noise, whatever it looks like."""

    def __init__(self, d, dev):
        z = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=dev)
        self.s, self.m2, self.n = [z(d), z(d)], [z(d, d), z(d, d)], [0, 0]

    def add(self, x, half):
        x = x.reshape(-1, x.shape[-1]).double()
        self.s[half] += x.sum(0)
        self.m2[half] += x.T @ x
        self.n[half] += x.shape[0]

    def cov(self, half=None):
        hs = (0, 1) if half is None else (half,)
        n = sum(self.n[h] for h in hs)
        mu = torch.stack([self.s[h] for h in hs]).sum(0) / n
        m2 = torch.stack([self.m2[h] for h in hs]).sum(0) / n
        return m2 - torch.outer(mu, mu)


def _capture(model, config, blocks, tokens, batch_size, dev):
    """Streams needed for the analysis, in one pass: the residual leaving each block in
    `blocks` and the one before it, the after-final-norm residual, and per-neuron activation
    RMS at each block's down-projection.

    Block-level hooks only: a block's output IS the residual after it, so Sigma_in for block
    b is block b-1's output. Hooking sub-modules would mean reading positional inputs, which
    these models call with kwargs."""
    fam = _fam(config)
    d = _getattr_path(model, _fam(config).final_norm).weight.shape[0] if hasattr(
        _getattr_path(model, _fam(config).final_norm), "weight") else model.config.hidden_size
    covs, acts, handles = {}, {}, []
    half = [0]                                  # alternating batches -> two disjoint halves
    block_list = _getattr_path(model, fam.blocks)

    def add_cov(name, capture_out=True):
        covs[name] = _Cov(d, dev)

        def hook(_m, inp, out):
            t = (out[0] if isinstance(out, tuple) else out) if capture_out else inp[0]
            covs[name].add(t.detach().float(), half[0])
        return hook

    for i in sorted({b for b in blocks} | {b - 1 for b in blocks}):
        if i >= 0:
            handles.append(block_list[i].register_forward_hook(add_cov(f"blk{i}.stream")))
    handles.append(_getattr_path(model, fam.final_norm).register_forward_hook(
        add_cov("after_final_norm")))

    for b in blocks:
        acts[b] = None

        def pre(_m, args, b=b):
            a = args[0].detach().float().reshape(-1, args[0].shape[-1])
            if acts[b] is None:
                acts[b] = {"s2": torch.zeros(a.shape[1], dtype=torch.float64, device=dev), "n": 0}
            acts[b]["s2"] += a.pow(2).sum(0).double()
            acts[b]["n"] += a.shape[0]
        handles.append(_getattr_path(
            model, f"{fam.blocks}.{b}.mlp.{fam.mlp_projs['down']}").register_forward_pre_hook(pre))

    with torch.no_grad():
        for j, i in enumerate(range(0, tokens.shape[0], batch_size)):
            half[0] = j % 2
            model.forward(tokens[i:i + batch_size].to(dev))
    for h in handles:
        h.remove()
    return covs, {b: (v["s2"] / v["n"]).sqrt() for b, v in acts.items()}


def _directions(sigma_in, sigma_out, sigma_afn, damp=1e-6):
    """The three candidate directions, each a unit vector in stream space."""
    d = sigma_in.shape[0]
    reg = sigma_in + damp * torch.eye(d, dtype=sigma_in.dtype, device=sigma_in.device) * sigma_in.diag().mean()
    ev, evec = torch.linalg.eigh(reg)
    w = evec @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ evec.T      # Sigma_in^{-1/2}
    ratio = w @ torch.linalg.eigh(w @ sigma_out @ w).eigenvectors[:, -1]
    return {"ratio": ratio / ratio.norm(),
            "gain": torch.linalg.eigh(sigma_out - sigma_in).eigenvectors[:, -1],
            "afn": torch.linalg.eigh(sigma_afn).eigenvectors[:, -1]}


def _magnitude(v, s_in, s_out, s_afn):
    """How big the direction actually is: variance along it entering and leaving the block,
    and in the final stream, each also as a share of that covariance's trace. A direction
    can top the gain ranking while carrying a negligible share of the variance."""
    q = lambda m: float(v @ m @ v)
    return {"var_in": q(s_in), "var_out": q(s_out), "var_afn": q(s_afn),
            "share_in": q(s_in) / float(s_in.trace()), "share_out": q(s_out) / float(s_out.trace()),
            "share_afn": q(s_afn) / float(s_afn.trace()),
            "gain_share": (q(s_out) - q(s_in)) / float(s_out.trace() - s_in.trace())
            if float(s_out.trace() - s_in.trace()) != 0 else float("nan")}


def _analyse(v, v0, vtop, w_out, rms, top=20):
    """Where the direction sits relative to the unembedding, and who wrote it."""
    v = v / v.norm()
    cos = (w_out.T @ v.float()) / w_out.norm(dim=0).clamp_min(1e-12)
    contrib = rms.float() * w_out.norm(dim=0) * cos
    order = contrib.abs().argsort(descending=True)[:top]
    return {"null_share": float((v0 @ v.float()).norm()),
            "top_share": float((vtop @ v.float()).norm()),
            "cos": cos.cpu(), "contrib": contrib.cpu(),
            "top_neurons": [(int(i), round(float(cos[i]), 4), round(float(contrib[i]), 4))
                            for i in order]}


def _projections(model, config, dirs, tokens, batch_size, dev):
    """Per-token projection onto each direction, from a second pass now that the directions
    are known, together with what the model PREDICTS at each position -- we have only ever
    looked at which tokens produce a direction, never at what is being predicted there.
    Returns ({(block, name): Tensor(n_tokens)}, pred_top1, pred_entropy)."""
    fam = _fam(config)
    blocks = _getattr_path(model, fam.blocks)
    store: dict = {key: [] for key in dirs}
    pred, ent = [], []

    def hook(b):
        def h(_m, _inp, out):
            x = (out[0] if isinstance(out, tuple) else out).detach().float()
            x = x.reshape(-1, x.shape[-1])
            for (bb, name), v in dirs.items():
                if bb == b:
                    store[(bb, name)].append((x @ v.to(x.device)).cpu())
        return h

    # Hook the norm's OUTPUT and apply the head alone. Hooking its input and calling the
    # full head_fn (which re-enters the norm) recurses forever.
    head_mod = get_head_module(model, config)
    cap = fam.logit_softcap

    def norm_hook(_m, _inp, out):
        x = (out[0] if isinstance(out, tuple) else out).detach()
        logits = head_mod(x.reshape(-1, x.shape[-1])).float()
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)
        lp = torch.log_softmax(logits, dim=-1)
        pred.append(lp.argmax(-1).cpu())
        ent.append(-(lp.exp() * lp).sum(-1).cpu())

    handles = [blocks[b].register_forward_hook(hook(b)) for b in {b for b, _ in dirs}]
    handles.append(_getattr_path(model, fam.final_norm).register_forward_hook(norm_hook))
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch_size):
            model.forward(tokens[i:i + batch_size].to(dev))
    for h in handles:
        h.remove()
    return {k: torch.cat(v) for k, v in store.items()}, torch.cat(pred), torch.cat(ent)


def _predicted(p, pred, ent, tok, frac=0.01, top=12):
    """At the positions that carry this direction hardest, what is the model predicting, and
    how uncertain is it? Contrast against the corpus-wide baseline."""
    e = p.double() ** 2
    hi = e.argsort(descending=True)[:max(1, int(frac * e.numel()))]
    counts: dict = {}
    for t in pred[hi].tolist():
        counts[t] = counts.get(t, 0) + 1
    common = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return {"pred_entropy_hi": float(ent[hi].mean()), "pred_entropy_all": float(ent.mean()),
            "top_predictions": [(tok.convert_ids_to_tokens([t])[0], n / len(hi)) for t, n in common]}


def _concentration(p, ids, tok, top_types=15):
    """Is the direction carried by a few token positions or spread out? Shares of the total
    squared projection held by the largest positions, the participation ratio (an effective
    number of contributing tokens), and the token types that carry the most."""
    e = (p.double() ** 2)
    tot = float(e.sum())
    srt = e.sort(descending=True).values
    n = e.numel()
    out = {"n_tokens": n, "participation_ratio": float(tot ** 2 / (e ** 2).sum()),
           **{f"share_top_{q}": float(srt[:max(1, int(q * n))].sum() / tot)
              for q in (0.001, 0.01, 0.1)}}
    if ids is not None:
        flat = ids.reshape(-1)[:n]
        types = []
        for t in torch.unique(flat).tolist():
            m = flat == t
            if int(m.sum()) >= 20:
                types.append((tok.convert_ids_to_tokens([t])[0],
                              float(e[m].mean()), float(e[m].sum() / tot), int(m.sum())))
        out["types"] = sorted(types, key=lambda r: -r[1])[:top_types]
    return out


def main(out_path: str = "data/results/grown_direction_neurons.pt",
         model_name: str = "allenai/OLMo-2-0425-1B", steps: tuple = (490_000, 1_900_000),
         num_windows: int = 512, batch_size: int = 8, n_blocks: int = 2) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(model_name)
    short = model_name.split("/")[-1]
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    res = out.setdefault(short, {})
    tokens = torch.load(MIXES[config.family], weights_only=True)[:num_windows]
    tokenizer = load_tokenizer(config)
    if isinstance(steps, int):
        steps = (steps,)
    schedule = {s: (rev, repo) for s, rev, repo in get_checkpoint_schedule(config, None)}
    for step in steps:
        revision, repo = schedule[step]
        loaded = load_model(config, repo, revision)
        last = get_num_layers(loaded, config) - 1
        blocks = [last - i for i in range(n_blocks)]
        model = torch.nn.Module.to(loaded, dev).eval()
        covs, rms = _capture(model, config, sorted(set(blocks)), tokens, batch_size, dev)

        gamma = getattr(get_final_layernorm(model, config), "weight", None)
        head = get_head_module(model, config).weight.detach()
        d = head.shape[1]
        k = max(1, round(0.01 * d))
        v0 = head_subspace(head, gamma, k)              # bottom-k: the effective null space
        vtop = head_subspace(head, gamma, d)[:k]        # top-k: what the head reads hardest

        step_res: dict = {"k": k, "d": d, "chance": (k / d) ** 0.5}
        all_dirs: dict = {}
        for b in blocks:
            cin, cout = covs[f"blk{b - 1}.stream"], covs[f"blk{b}.stream"]
            s_in, s_out = cin.cov().float(), cout.cov().float()
            halves = [_directions(cin.cov(h).float(), cout.cov(h).float(),
                                  covs["after_final_norm"].cov(h).float()) for h in (0, 1)]
            stability = {k: abs(float(halves[0][k] @ halves[1][k])) for k in halves[0]}
            down: Any = _getattr_path(
                model, f"{_fam(config).blocks}.{b}.mlp.{_fam(config).mlp_projs['down']}")
            w_out = down.weight.detach().float()
            post = get_post_mlp_norm(model, config, b)
            if post is not None:
                w_out = w_out * post.weight.detach().float()[:, None]
            dirs = _directions(s_in, s_out, covs["after_final_norm"].cov().float())
            all_dirs.update({(b, name): v for name, v in dirs.items()})
            s_afn = covs["after_final_norm"].cov().float()
            step_res[b] = {name: {**_analyse(v, v0, vtop, w_out, rms[b]),
                                  **_magnitude(v / v.norm(), s_in, s_out, s_afn),
                                  "split_half_cos": stability[name]}
                           for name, v in dirs.items()}
            step_res[b]["traces"] = {"in": float(s_in.trace()), "out": float(s_out.trace()),
                                     "afn": float(s_afn.trace())}
            for name, v in dirs.items():                      # keep the vectors themselves
                step_res[b][name]["vector"] = (v / v.norm()).cpu()
            step_res[b]["cos_between"] = {                    # is the gained direction the top one?
                f"{a}-{c}": abs(float(dirs[a] @ dirs[c] / (dirs[a].norm() * dirs[c].norm())))
                for a, c in (("gain", "afn"), ("ratio", "afn"), ("gain", "ratio"))}
            print(f"{short} step{step} blk{b} cos between directions: "
                  f"{step_res[b]['cos_between']}", flush=True)
            for name, r in dict(step_res[b]).items():
                if name in ("traces", "cos_between"):
                    continue
                print(f"{short} step{step} blk{b} {name}: var_out={r['var_out']:.4g} "
                      f"({100 * r['share_out']:.1f}% of trace), gain={r['var_out'] - r['var_in']:.4g} "
                      f"({100 * r['gain_share']:.1f}% of the block's), null={r['null_share']:.3f} "
                      f"top={r['top_share']:.3f} (chance {step_res['chance']:.3f}) "
                      f"split-half |cos|={r['split_half_cos']:.3f}  "
                      f"top neurons {r['top_neurons'][:5]}", flush=True)
        # second pass: how concentrated is each direction over token positions and types
        proj, pred, ent = _projections(model, config, all_dirs, tokens, batch_size, dev)
        ids = tokens[:, :proj[next(iter(proj))].numel() // tokens.shape[0]]
        for (b, name), p in ((k, v) for k, v in proj.items()):
            conc = _concentration(p, ids, tokenizer)
            pr = _predicted(p, pred, ent, tokenizer)
            step_res[b][name]["concentration"] = conc          # type: ignore[index]
            step_res[b][name]["predicted"] = pr                # type: ignore[index]
            print(f"    predicted at its top 1%: H={pr['pred_entropy_hi']:.3f} vs "
                  f"{pr['pred_entropy_all']:.3f} overall; {[t for t, _ in pr['top_predictions'][:6]]}",
                  flush=True)
            print(f"{short} step{step} blk{b} {name}: PR={conc['participation_ratio']:.1f}"
                  f"/{conc['n_tokens']}  top0.1%={conc['share_top_0.001']:.3f} "
                  f"top1%={conc['share_top_0.01']:.3f}  "
                  f"types {[t[0] for t in conc.get('types', [])[:6]]}", flush=True)
        res[step] = step_res
        del model, loaded
        torch.cuda.empty_cache()
        torch.save(out, out_path)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
