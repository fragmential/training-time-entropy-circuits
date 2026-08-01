"""Is the mid-stack compression the growth of SHARED features, or of token-identity features?

Li et al., Tracing the Representation Geometry of Language Models from Pretraining to
Post-training (arXiv:2509.23024) explain the compression phase by the output bottleneck:
skewed token frequencies plus d << |V| force the model to reuse its dominant directions. That
account predicts compression should be strongest nearest the output, and needs the bottleneck
to bite at all. Neither matches what the mid-stack does here.

The alternative tested here: mid-depth representations compress because the features living
there become more SHARED. A useful contextual feature (register, topic, syntactic role,
position in the document) is by construction one that many different tokens load on, so it is
a common-mode direction; token identity is idiosyncratic and therefore high-rank. Learning
better context should then concentrate variance with no bottleneck involved.

The split is an analysis of variance on the residual stream, grouping rows by the CURRENT
token id:

    total     Sigma    -- every row against the global mean
    identity  B        -- the group means against the global mean, weighted by group size:
                          the part of the representation predictable from the token alone
    context   W = Sigma - B -- what is left: the same token in different contexts

Both accounts predict compression; they disagree about which term carries it. The sharing
account says context becomes dominant AND low-rank while identity stays high-rank. The
bottleneck account says identity concentrates onto the frequent tokens' directions.

Writes data/results/context_vs_identity.pt. GPU node.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from oneoff_scripts.unembedding_spectra import MODELS
from utils.model_registry import (_fam, _getattr_path, delete_cached_revision,
                                  get_checkpoint_schedule, get_final_layernorm,
                                  get_model_config, get_num_layers, get_token_count, load_model)

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}
DEPTHS = (0.25, 0.5, 0.75)      # fractions of the stack, plus before_final_norm always
MAX_ROWS = 262_144


def _rankme(ev: torch.Tensor) -> float:
    p = ev.clamp_min(0)
    p = p / p.sum().clamp_min(1e-30)
    return float(torch.exp(-(p * p.clamp_min(1e-30).log()).sum()))


class _Acc:
    """Running total covariance and per-token-id means, in one pass and fp64."""

    def __init__(self, d: int, vocab: int, dev: str):
        self.s = torch.zeros(d, dtype=torch.float64, device=dev)
        self.ss = torch.zeros(d, d, dtype=torch.float64, device=dev)
        self.gs = torch.zeros(vocab, d, dtype=torch.float64, device=dev)
        self.gn = torch.zeros(vocab, dtype=torch.float64, device=dev)
        self.n = 0

    def add(self, x: torch.Tensor, ids: torch.Tensor) -> None:
        x = x.double()
        self.s += x.sum(0)
        self.ss += x.T @ x
        self.gs.index_add_(0, ids, x)
        self.gn.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float64))
        self.n += x.shape[0]

    def split(self) -> dict:
        mu = self.s / self.n
        total = self.ss / self.n - torch.outer(mu, mu)
        seen = self.gn > 0
        gm = self.gs[seen] / self.gn[seen, None]                  # per-token mean
        w = (self.gn[seen] / self.n)[:, None]
        c = gm - mu
        identity = (c * w).T @ c                                  # between-group covariance
        context = total - identity
        out = {}
        for name, m in (("total", total), ("identity", identity), ("context", context)):
            ev = torch.linalg.eigvalsh(m).flip(0).clamp_min(0)
            out[name] = {"eigvals": ev.float().cpu(), "rankme": _rankme(ev),
                         "trace": float(ev.sum())}
        out["n_types"] = int(seen.sum())
        out["n"] = self.n
        return out


@torch.no_grad()
def _measure(model, config, blocks, tokens, batch_size, dev, max_rows, vocab):
    """One pass: residual stream entering each chosen block, plus before_final_norm."""
    cur: dict = {}
    names = [f"blk{b}" for b in blocks] + ["before_final_norm"]
    mods = [_getattr_path(model, f"{_fam(config).blocks}.{b}") for b in blocks]
    mods.append(get_final_layernorm(model, config))
    hs = [m.register_forward_pre_hook(
        lambda _m, args, k=n: cur.__setitem__(k, args[0].detach())) for m, n in zip(mods, names)]
    acc: dict = {}
    n = 0
    for i in range(0, tokens.shape[0], batch_size):
        if n >= max_rows:
            break
        batch = tokens[i:i + batch_size].to(dev)
        model.forward(batch)
        ids = batch[:, :cur[names[0]].shape[1]].reshape(-1)
        rows = min(ids.shape[0], max_rows - n)
        for name in names:
            x = cur[name].reshape(-1, cur[name].shape[-1])[:rows]
            if name not in acc:
                acc[name] = _Acc(x.shape[1], vocab, dev)
            acc[name].add(x, ids[:rows])
        n += rows
    for h in hs:
        h.remove()
    return {name: a.split() for name, a in acc.items()}


def main(out_path: str = "data/results/context_vs_identity.pt", models: tuple = MODELS,
         max_checkpoints: int = 14, spacing: str = "log", batch_size: int = 32,
         max_rows: int = MAX_ROWS, keep_cached: bool = False) -> None:
    """keep_cached defaults FALSE: load_model pulls a full snapshot per revision, so keeping
    them puts hundreds of GB in the cache over a sweep. Only pass True when another job is
    reading the same model's revisions."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for name in models:
        config = get_model_config(name)
        short = name.split("/")[-1]
        tokens = torch.load(MIXES[config.family], weights_only=True)
        done = out.setdefault(short, {})
        sched = get_checkpoint_schedule(config, max_checkpoints, spacing)
        for step, revision, repo in sched:
            if step in done:
                continue
            try:
                loaded = load_model(config, repo, revision)
            except Exception as exc:
                print(f"{short} step{step}: SKIPPED ({type(exc).__name__})", flush=True)
                continue
            n_blocks = get_num_layers(loaded, config)
            model: Any = torch.nn.Module.to(loaded, dev).eval()
            vocab = int(model.config.vocab_size)
            blocks = sorted({max(1, round(f * (n_blocks - 1))) for f in DEPTHS})
            res = _measure(model, config, blocks, tokens, batch_size, dev, max_rows, vocab)
            done[step] = {"depths": res, "blocks": blocks, "n_blocks": n_blocks,
                          "tokens": get_token_count(short, step)}
            torch.save(out, out_path)
            mid = res[f"blk{blocks[len(blocks) // 2]}"]
            print(f"{short} step{step}: mid-stack RankMe total={mid['total']['rankme']:.0f} "
                  f"identity={mid['identity']['rankme']:.0f} "
                  f"context={mid['context']['rankme']:.0f}  context share="
                  f"{mid['context']['trace'] / max(mid['total']['trace'], 1e-30):.2f}", flush=True)
            del model, loaded
            torch.cuda.empty_cache()
            if not keep_cached and config.family != "nanochat":
                delete_cached_revision(repo, revision)
        miss = len(sched) - len([s for s, *_ in sched if s in done])
        print(f"COVERAGE {short}: {len(sched) - miss}/{len(sched)} checkpoints"
              + (f"  <-- {miss} MISSING, results are PARTIAL" if miss else ""), flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
