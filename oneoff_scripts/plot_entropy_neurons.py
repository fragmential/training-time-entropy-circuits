"""Plots for entropy_neurons.py: does null-space composition in the last MLP have a phase shift?

Row per model. Column 1: quantiles of rho (share of a neuron's write direction lying in the
bottom-k right-singular subspace of the centred, norm-folded unembedding) across training,
with the random-direction level sqrt(k/d) marked. Column 2: how many neurons clear rho > 0.5.
Column 3: the paper's identification plane (LogitVar vs rho, sized by weight norm) at the
first and last checkpoint. The penultimate block is drawn dashed throughout as a control.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT_DIR = "analysis/figures/entropy_neurons"
QUANTILES = (0.5, 0.99, 1.0)


def _series(per_step, ki, fn):
    steps = sorted(per_step)
    xs = [per_step[s]["tokens"] for s in steps]
    return np.array(xs), np.array([fn(per_step[s]["rho"][ki].numpy(), per_step[s]) for s in steps])


def _k_index(entry, frac=0.01):
    ks = entry["ks"]
    return ks.index(max(1, round(frac * entry["d_model"]))), ks


def main(in_path: str = "data/results/entropy_neurons.pt", out_dir: str = OUT_DIR,
         threshold: float = 0.5) -> None:
    data = torch.load(in_path, weights_only=False)
    os.makedirs(out_dir, exist_ok=True)
    models = [m for m in data if data[m]]
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 3.2 * len(models)), squeeze=False)

    for row, model in enumerate(models):
        blocks = sorted(data[model], reverse=True)          # last block first
        ax_q, ax_n, ax_s = axes[row]
        for bi, b in enumerate(blocks):
            per_step = data[model][b]
            probe = per_step[next(iter(per_step))]
            ki, ks = _k_index(probe)
            k, d = ks[ki], probe["d_model"]
            style = "-" if bi == 0 else "--"
            for q, colour in zip(QUANTILES, ("C0", "C1", "C3")):
                xs, ys = _series(per_step, ki, lambda r, _e, q=q: np.quantile(r, q))
                ax_q.plot(xs, ys, style, color=colour, lw=1.4, label=f"blk{b} q={q:g}")
            xs, ns = _series(per_step, ki, lambda r, _e: float((r > threshold).sum()))
            ax_n.plot(xs, ns, style, color="C2", lw=1.4, label=f"blk{b}")
            if bi == 0:
                ax_q.axhline(np.sqrt(k / d), color="k", ls=":", lw=1,
                             label=f"chance $\\sqrt{{k/d}}$, k={k}")
                for step, marker, alpha in ((sorted(per_step)[0], "o", 0.3),
                                            (sorted(per_step)[-1], "o", 0.8)):
                    e = per_step[step]
                    ax_s.scatter(e["rho"][ki].numpy(), e["logitvar"].numpy(), s=2 + 40 *
                                 (e["norm"] / e["norm"].max()).numpy() ** 2, alpha=alpha,
                                 edgecolors="none", label=f"step {step}")

        ax_q.set(xscale="log", ylabel=f"{model}\nrho (k = 0.01 d)", xlabel="tokens")
        ax_q.legend(fontsize=6, ncol=2)
        ax_n.set(xscale="log", ylabel=f"# neurons rho > {threshold}", xlabel="tokens")
        ax_n.legend(fontsize=6)
        ax_s.set(yscale="log", xlabel="rho", ylabel="LogitVar", title="size = weight norm")
        ax_s.legend(fontsize=6)

    fig.tight_layout()
    path = os.path.join(out_dir, "entropy_neurons.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
