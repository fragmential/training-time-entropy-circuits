"""Which neurons actually move the output entropy, and when they start to.

Ablation answers this one neuron at a time. Entropy is a differentiable function of the
residual stream, so the whole layer can be read off in one backward pass instead:

    E_i = mean_t (g_t . w_out^(i)) (a_t,i - mean_i a),    g_t = dH_t / dh_t

with H_t the Shannon entropy of the next-token distribution and h_t the pre-final-norm
stream. g_t comes from autograd on h -> final norm -> unembedding only, never the whole
model. E_i is the first-order version of mean-ablating neuron i: positive means the neuron's
fluctuation RAISES entropy (a hedging neuron), negative means it sharpens the distribution.

Every neuron of the chosen blocks, at every requested checkpoint, so "whose entropy effect
GROWS between 10^10 and 10^12 tokens" is a difference of two columns. rho travels alongside,
to see whether the entropy movers are the null-space writers.

Writes {model: {block: {step: {"effect", "abs_effect", "rho", "act_rms", "w_norm"}}}} to
data/results/entropy_effect.pt.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_fam, _getattr_path, get_checkpoint_schedule,
                                  get_final_layernorm, get_head_module, get_model_config,
                                  get_num_layers, get_output_head, get_post_mlp_norm, load_model)
from utils.nullspace import head_subspace, write_rho

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}


def _entropy_grad(h, head_fn, dtype, chunk=4096):
    """dH/dh for each token, H = entropy of softmax(head(h)). Chunked: the (tokens x vocab)
    logit matrix is the only large intermediate and never needs to exist all at once."""
    out = torch.empty_like(h)
    for i in range(0, h.shape[0], chunk):
        x = h[i:i + chunk].detach().float().requires_grad_(True)
        logp = torch.log_softmax(head_fn(x.to(dtype)).float(), dim=-1)   # head weights are bf16
        (-(logp.exp() * logp).sum(-1)).sum().backward()
        out[i:i + chunk] = x.grad if x.grad is not None else 0
    return out


def _layer_entropy(streams, head_fn, dtype, chunk=4096):
    """Mean next-token entropy for each supplied stream, exactly (no linearisation).

    head_fn must be the model's FULL output head -- final norm, then unembedding, then any
    softcap. Applying the unembedding alone to a pre-norm stream removes the very mechanism
    this measures, since the neurons act by changing the norm's scale.
    """
    out = {}
    for name, h in streams.items():
        tot, n = 0.0, 0
        for i in range(0, h.shape[0], chunk):
            x = h[i:i + chunk]
            lp = torch.log_softmax(head_fn(x.to(dtype)).float(), dim=-1)
            tot += float(-(lp.exp() * lp).sum(-1).sum())
            n += x.shape[0]
        out[name] = tot / n
    return out


def _effects(model, config, blocks, tokens, batch_size, dev):
    """Per-neuron entropy effect, plus the activation RMS, for each block."""
    fam = _fam(config)
    head_fn = get_output_head(model, config)
    head_dtype = get_head_module(model, config).weight.dtype
    acts: dict = {}
    stream: list = []
    handles = []

    def pre(b):
        def hook(_m, args):
            acts[b] = args[0].detach().float().reshape(-1, args[0].shape[-1])
        return hook

    for b in blocks:
        handles.append(_getattr_path(
            model, f"{fam.blocks}.{b}.mlp.{fam.mlp_projs['down']}").register_forward_pre_hook(pre(b)))
    norm: Any = get_final_layernorm(model, config)

    def norm_pre(_m, args):                       # returns None: never replace the output
        stream.append(args[0].detach().float().reshape(-1, args[0].shape[-1]))
    handles.append(norm.register_forward_pre_hook(norm_pre))

    # the last block's actual write (post-feedforward norm where the family has one) and the
    # stream entering the block, so the net layer effect can be split from the per-neuron one
    last = max(blocks)
    write: list = []
    into: list = []
    writer: Any = get_post_mlp_norm(model, config, last) or _getattr_path(
        model, f"{fam.blocks}.{last}.mlp")

    def grab(store):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store.append(t.detach().float().reshape(-1, t.shape[-1]))
        return hook
    handles.append(writer.register_forward_hook(grab(write)))
    handles.append(_getattr_path(model, fam.blocks)[last - 1].register_forward_hook(grab(into)))

    w_outs = {}
    for b in blocks:
        down: Any = _getattr_path(model, f"{fam.blocks}.{b}.mlp.{fam.mlp_projs['down']}")
        w = down.weight.detach().float()
        post = get_post_mlp_norm(model, config, b)
        w_outs[b] = w * post.weight.detach().float()[:, None] if post is not None else w

    tot = {b: torch.zeros(w_outs[b].shape[1], dtype=torch.float64, device=dev) for b in blocks}
    tot_full = {b: torch.zeros(w_outs[b].shape[1], dtype=torch.float64, device=dev) for b in blocks}
    s2 = {b: torch.zeros(w_outs[b].shape[1], dtype=torch.float64, device=dev) for b in blocks}
    n = 0
    ent_tot: dict = {}
    for i in range(0, tokens.shape[0], batch_size):
        stream.clear(); write.clear(); into.clear()
        with torch.no_grad():
            model.forward(tokens[i:i + batch_size].to(dev))
        g = _entropy_grad(stream[0], head_fn, head_dtype)                    # (tokens, d)
        for b in blocks:
            a = acts[b]
            gw = g @ w_outs[b]
            tot[b] += (gw * (a - a.mean(0))).sum(0).double()      # fluctuation only
            tot_full[b] += (gw * a).sum(0).double()               # the whole write, mean included
            s2[b] += a.pow(2).sum(0).double()
        n += stream[0].shape[0]
        h_out, w, h_in = stream[0], write[0], into[0]
        ents = _layer_entropy({"out": h_out, "in": h_in,
                               "mlp_mean_ablated": h_out - w + w.mean(0),
                               "mlp_zero_ablated": h_out - w},
                              head_fn, head_dtype)
        for k2, v in ents.items():
            ent_tot[k2] = ent_tot.get(k2, 0.0) + v * h_out.shape[0]
    for h in handles:
        h.remove()
    return ({b: (v / n).cpu() for b, v in tot.items()},
            {b: (v / n).sqrt().cpu() for b, v in s2.items()}, w_outs, n,
            {k: v / n for k, v in ent_tot.items()},
            {b: (v / n).cpu() for b, v in tot_full.items()})


def main(out_path: str = "data/results/entropy_effect.pt",
         model_name: str = "allenai/OLMo-2-0425-1B",
         steps: tuple = (6_000, 60_000, 490_000, 1_900_000),
         num_windows: int = 128, batch_size: int = 8, n_blocks: int = 2) -> None:
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
        blocks = [last - i for i in range(n_blocks)]
        model = torch.nn.Module.to(loaded, dev).eval()
        eff, rms, w_outs, n, ents, eff_full = _effects(model, config, blocks, tokens, batch_size, dev)
        res.setdefault("layer_entropy", {})[step] = ents
        print(f"{short} step{step}: H(in)={ents['in']:.4f} H(out)={ents['out']:.4f} "
              f"net={ents['out'] - ents['in']:+.4f} | MLP fluctuation "
              f"{ents['out'] - ents['mlp_mean_ablated']:+.4f}, whole MLP write "
              f"{ents['out'] - ents['mlp_zero_ablated']:+.4f}", flush=True)

        gamma = getattr(get_final_layernorm(model, config), "weight", None)
        head = get_head_module(model, config).weight.detach()
        v0 = head_subspace(head, gamma, max(1, round(0.01 * head.shape[1])))
        for b in blocks:
            rho = write_rho(v0, w_outs[b]).cpu()
            res.setdefault(b, {})[step] = {"effect": eff[b], "effect_full": eff_full[b],
                                           "rho": rho, "act_rms": rms[b],
                                           "w_norm": w_outs[b].norm(dim=0).cpu(), "n": n}
            e = eff[b]
            up = e.argsort(descending=True)[:5]
            print(f"    blk{b} sum E_i: fluctuation {float(e.sum()):+.4f}, "
                  f"whole write {float(eff_full[b].sum()):+.4f}\n"
                  f"{short} step{step} blk{b}: entropy-raising {[(int(i), round(float(e[i]), 4), round(float(rho[i]), 3)) for i in up]}\n"
                  f"    sharpening {[(int(i), round(float(e[i]), 4), round(float(rho[i]), 3)) for i in e.argsort()[:5]]}\n"
                  f"    |E| sum={float(e.abs().sum()):.3f}  corr(|E|, rho)="
                  f"{float(torch.corrcoef(torch.stack([e.abs(), rho]))[0, 1]):.3f}", flush=True)
        del model, loaded
        torch.cuda.empty_cache()
        torch.save(out, out_path)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
