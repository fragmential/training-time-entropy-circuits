"""How hard do the null-space neurons actually fire?

rho and ||w_out|| are weight-only statistics: they say where a neuron would write, not how
much it does write. Stolfo et al. characterise entropy neurons by weight norm and never
quantify activations on data (their only activation claim is that four GPT-2 neurons change
average activation during induction). The write a neuron actually contributes to the residual
stream is |a_i| * ||w_out^(i)||, so this measures a_i.

One forward pass over a few packed batches at the final checkpoint, hooking the input of the
final block's down-projection -- which IS the post-nonlinearity neuron activation vector --
and accumulating per-neuron sum, sum of squares, and how often the neuron is off. Nothing is
stored per token: only the four accumulators.

Writes {model: {block: {step: {"mean", "rms", "mean_abs", "frac_zero", "n"}}}} to
data/results/neuron_write_stats.pt; pair with entropy_neurons.pt for rho. Var(a_i)*||w_out||^2
from this is what lets an ablation control be matched on write size rather than on count.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_fam, _getattr_path, delete_cached_revision,
                                  get_checkpoint_schedule, get_model_config, get_num_layers,
                                  load_model)

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}
MODELS = ("EleutherAI/pythia-160m-deduped", "EleutherAI/pythia-410m-deduped",
          "EleutherAI/pythia-1b-deduped", "allenai/OLMo-2-0425-1B", "nanochat-d12")


def _accumulate(model, config, block, tokens, batch_size, dev):
    """Per-neuron sum, sum of squares and zero count at the final block's down-proj input."""
    fam = _fam(config)
    down = _getattr_path(model, f"{fam.blocks}.{block}.mlp.{fam.mlp_projs['down']}")
    acc = {}

    def hook(_mod, args):
        a = args[0].detach().float().reshape(-1, args[0].shape[-1])
        if not acc:
            acc.update(s=torch.zeros(a.shape[1], device=dev, dtype=torch.float64),
                       s2=torch.zeros(a.shape[1], device=dev, dtype=torch.float64),
                       sa=torch.zeros(a.shape[1], device=dev, dtype=torch.float64),
                       z=torch.zeros(a.shape[1], device=dev, dtype=torch.float64), n=0)
        acc["s"] += a.sum(0).double()
        acc["s2"] += a.pow(2).sum(0).double()
        acc["sa"] += a.abs().sum(0).double()
        acc["z"] += (a.abs() < 1e-6).sum(0).double()
        acc["n"] += a.shape[0]

    handle = down.register_forward_pre_hook(hook)
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch_size):
            model.forward(tokens[i:i + batch_size].to(dev))
    handle.remove()
    n = acc["n"]
    return {"mean": (acc["s"] / n).cpu(), "rms": (acc["s2"] / n).sqrt().cpu(),
            "mean_abs": (acc["sa"] / n).cpu(), "frac_zero": (acc["z"] / n).cpu(), "n": n}


def main(out_path: str = "data/results/neuron_write_stats.pt", models: tuple = MODELS,
         num_windows: int = 64, batch_size: int = 8, max_checkpoints: "int | None" = None,
         spacing: str = "log", n_blocks: int = 2, keep_cached: bool = False) -> None:
    """max_checkpoints=None sweeps the whole schedule; give an int to match a collect run's
    (20 for the pythias/OLMo, 40 for nanochat). n_blocks counts back from the last.

    keep_cached defaults to FALSE here because this sweeps every checkpoint: retaining the
    snapshots costs 2-13 GB each and has twice filled the project allocation, which corrupts
    in-flight downloads and takes unrelated jobs down with it."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    for name in models:
        config = get_model_config(name)
        short = name.split("/")[-1]
        done = out.setdefault(short, {})
        tokens = torch.load(MIXES[config.family], weights_only=True)[:num_windows]
        for step, revision, repo in get_checkpoint_schedule(config, max_checkpoints, spacing):
            loaded = load_model(config, repo, revision)
            blocks = [get_num_layers(loaded, config) - 1 - i for i in range(n_blocks)]
            if all(step in done.get(b, {}) for b in blocks):
                del loaded
                continue
            model = torch.nn.Module.to(loaded, dev).eval()
            for block in blocks:
                res = _accumulate(model, config, block, tokens, batch_size, dev)
                done.setdefault(block, {})[step] = res
                print(f"{short} step{step} blk{block}: {res['n']} tokens, mean|a| in "
                      f"[{res['mean_abs'].min():.3f}, {res['mean_abs'].max():.3f}]", flush=True)
            del model, loaded
            torch.cuda.empty_cache()
            torch.save(out, out_path)
            if not keep_cached and config.family != "nanochat":
                delete_cached_revision(repo, revision)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
