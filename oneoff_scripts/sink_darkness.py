"""Is the massive direction in the stream actually inside the unembedding's dark subspace?

Cancedda, Spectral Filters, Dark Signals, and Attention Sinks (arXiv:2402.09221) shows the
sink's residual stream is almost entirely U-dark. This checks that claim directly on our
models, on the DIRECTION itself -- no neuron selection anywhere -- because everything built on
top of it depends on the premise being true here.

Conventions follow the paper, not this repo's rho:
  * RAW unembedding. No vocabulary centering, no final-norm gain folded in. Both of those are
    defensible elsewhere; neither is what Cancedda measures.
  * dark = the span of the bottom 5% of W_U's right singular vectors (1% also reported).
  * U-dark RATIO for a vector h is ||P h|| / ||(I-P) h||, his eq. 5 -- a ratio, not a share.

Centering is the trap. A massive activation is largely a constant offset, which a centered
covariance removes by construction, so a centered top eigenvector can miss it entirely. Three
directions are therefore reported at every probe: the stream MEAN, the top UNCENTERED
eigenvector, and the top CENTERED eigenvector.

Two probe points per model, both read as the residual stream ENTERING a block:
  * `post_sink` -- the block just after the model's known sink write
  * `late`      -- the third-to-last block, far enough from the end to be safe
OLMo-2 has no sink write, so it gets `late` only, plus `post_sink` at the same nominal depth
for comparison.

Writes data/results/sink_darkness.pt. GPU node. Reads nothing but model weights.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_fam, _getattr_path, delete_cached_revision,
                                  get_checkpoint_schedule, get_head_module, get_model_config,
                                  get_num_layers, get_token_count, load_model)

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}

# The block the sink write lands in, per model (final_report.md 0.2 / dig_findings). The probe
# sits one block LATER, so the write is already in the stream being measured.
SINK_WRITE = {"pythia-410m-deduped": 8, "pythia-1b-deduped": 3, "pythia-6.9b-deduped": 5,
              "nanochat-d12": 4}
MODELS = ("EleutherAI/pythia-410m-deduped", "EleutherAI/pythia-1b-deduped",
          "EleutherAI/pythia-6.9b-deduped", "allenai/OLMo-2-0425-1B",
          "allenai/OLMo-2-1124-7B", "nanochat-d12")
FRACS = (0.01, 0.05)
MAX_ROWS = 131_072


def _probes(short: str, n_blocks: int) -> dict:
    """name -> index of the block whose INPUT is measured."""
    late = max(1, n_blocks - 3)
    post = SINK_WRITE.get(short)
    out = {"late": late}
    out["post_sink"] = min(post + 1, late) if post is not None else max(1, n_blocks // 4)
    return out


@torch.no_grad()
def _capture(model, config, probes, tokens, batch_size, dev, max_rows, v0s):
    """Streaming mean, second moment, and per-token dark ratios at each probe."""
    cur: dict = {}
    hs = [_getattr_path(model, f"{_fam(config).blocks}.{b}").register_forward_pre_hook(
        lambda _m, args, k=name: cur.__setitem__(k, args[0].detach()))
        for name, b in probes.items()]
    acc: dict = {name: None for name in probes}
    n = 0
    for i in range(0, tokens.shape[0], batch_size):
        if n >= max_rows:
            break
        model.forward(tokens[i:i + batch_size].to(dev))
        first = cur[list(probes)[0]]
        rows = min(first.shape[0] * first.shape[1], max_rows - n)   # one row count for every
        for name in probes:                                         # probe, so n stays shared
            h = cur[name]
            seq = h.shape[1]
            x = h.reshape(-1, h.shape[-1])[:rows].double()
            a = acc[name]
            if a is None:
                a = acc[name] = {"s": torch.zeros(x.shape[1], dtype=torch.float64, device=dev),
                                 "ss": torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64,
                                                   device=dev),
                                 "ratio": {f: [] for f in FRACS}, "norm": [], "pos0": []}
            a["s"] += x.sum(0)
            a["ss"] += x.T @ x
            a["norm"].append(x.norm(dim=1).float().cpu())
            pos = torch.arange(rows, device=dev) % seq
            a["pos0"].append((pos == 0).cpu())
            for f, v0 in v0s.items():                       # Cancedda eq. 5, per token
                d = (x.float() @ v0.T).norm(dim=1)
                a["ratio"][f].append((d / (x.float().norm(dim=1) ** 2 - d ** 2)
                                      .clamp_min(1e-12).sqrt()).cpu())
        n += rows
    for h in hs:
        h.remove()
    return acc, n


def _summarise(a: dict, vh: torch.Tensor, ks: dict, n: int) -> dict:
    """Share of each of the three directions inside the bottom-k subspace, plus dark ratios."""
    mu = a["s"] / n
    second = a["ss"] / n
    uncent = second
    cent = second - torch.outer(mu, mu)
    dirs = {"mean": mu.float(),
            "top_uncentered": torch.linalg.eigh(uncent)[1][:, -1].float(),
            "top_centered": torch.linalg.eigh(cent)[1][:, -1].float()}
    out: dict = {"n": n}
    for f, k in ks.items():
        v0 = vh[-k:]
        out[f"share_{f}"] = {nm: float((v0 @ (v / v.norm())).norm()) for nm, v in dirs.items()}
        out[f"chance_{f}"] = float((k / vh.shape[1]) ** 0.5)
        r = torch.cat(a["ratio"][f])
        norms = torch.cat(a["norm"])
        pos0 = torch.cat(a["pos0"])
        hi = norms >= torch.quantile(norms.float(), 0.99)
        out[f"ratio_{f}"] = {"all": float(r.mean()), "pos0": float(r[pos0].mean()),
                             "top1pct_norm": float(r[hi].mean())}
    return out


@torch.no_grad()
def main(out_path: str = "data/results/sink_darkness.pt", models: tuple = MODELS,
         batch_size: int = 16, max_rows: int = MAX_ROWS, n_checkpoints: int = 1,
         keep_cached: bool = False) -> None:
    """Only run on models NO other job is touching. load_model pulls a full snapshot and
    keep_cached=False deletes it afterwards, so a concurrent reader of the same revision
    breaks -- and a concurrent DELETER breaks this one."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for name in models:
        config = get_model_config(name)
        short = name.split("/")[-1]
        tokens = torch.load(MIXES[config.family], weights_only=True)
        sched = get_checkpoint_schedule(config, n_checkpoints, "log")   # a trajectory, not just
        done = out.setdefault(short, {})                                # the final checkpoint
        for step, revision, repo in sched:
            if step in done:
                continue
            loaded = load_model(config, repo, revision)
            n_blocks = get_num_layers(loaded, config)
            model: Any = torch.nn.Module.to(loaded, dev).eval()
            wu = get_head_module(model, config).weight.detach().to(dev, torch.float32)
            vh = torch.linalg.svd(wu, full_matrices=False).Vh          # RAW: no centering
            ks = {f: max(1, round(f * vh.shape[1])) for f in FRACS}
            v0s = {f: vh[-k:] for f, k in ks.items()}
            probes = _probes(short, n_blocks)
            acc, n = _capture(model, config, probes, tokens, batch_size, dev, max_rows, v0s)
            done[step] = {"probes": probes, "n_blocks": n_blocks,
                          "tokens": get_token_count(short, step),
                          "at": {nm: _summarise(a, vh, ks, n) for nm, a in acc.items()}}
            torch.save(out, out_path)
            for nm, r in done[step]["at"].items():
                print(f"{short} step{step} {nm} (blk{probes[nm]}): chance={r['chance_0.05']:.3f} "
                      f"share5% mean={r['share_0.05']['mean']:.3f} "
                      f"uncent={r['share_0.05']['top_uncentered']:.3f} "
                      f"cent={r['share_0.05']['top_centered']:.3f} | "
                      f"dark ratio all={r['ratio_0.05']['all']:.3f} "
                      f"pos0={r['ratio_0.05']['pos0']:.3f} "
                      f"top1%norm={r['ratio_0.05']['top1pct_norm']:.3f}", flush=True)
            del model, loaded, acc
            torch.cuda.empty_cache()
            if not keep_cached and config.family != "nanochat":
                delete_cached_revision(repo, revision)
        print(f"COVERAGE {short}: {len(done)}/{len(sched)} checkpoints", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
