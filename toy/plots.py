"""Paper-figure plots for the toy models: Li et al. Fig 4 panels B-D, the controls figure,
and per-multi-variant stream / ledger / trace-weight figures. Reads compute_metrics results
(data/results/toy_<variant>/results_toy-<variant>.npy) + raw trajectories."""

import os

import matplotlib
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy.train import VARIANTS

RESULTS = "data/results"
FIGURES = "toy/figures"
PHASE_STYLES = (("warmup", ":"), ("entropy-seeking", "-"), ("compression-seeking", "--"))
CLASS_COLORS = ("magenta", "orange", "royalblue", "seagreen", "orchid", "goldenrod", "steelblue", "olive")
LEDGER_TERMS = ("delta_s", "chi", "quality", "interference")


def _results(variant: str) -> dict[int, dict]:
    return np.load(f"{RESULTS}/toy_{variant}/results_toy-{variant}.npy", allow_pickle=True).item()


def _traj(variant: str) -> dict[str, torch.Tensor]:
    return torch.load(f"{RESULTS}/toy/trajectories_{variant}.pt", weights_only=True)


def _rankme(res: dict[int, dict], node: str = "before_final_norm") -> tuple[np.ndarray, np.ndarray]:
    steps = np.array(sorted(res))
    return steps, np.array([res[s][node]["acts_uncentered"]["rankme"] for s in steps])


def _phase_bounds(rm: np.ndarray) -> tuple[int, int]:
    """(dip, peak) indices splitting warmup / entropy-seeking / compression-seeking."""
    peak = int(np.argmax(rm))
    return int(np.argmin(rm[:peak + 1])), peak


def _plot_phased(ax: Axes, path: np.ndarray, bounds: tuple[int, int], color: str) -> None:
    """One 2D trajectory split into the three phase segments (paper line styles)."""
    for lo, hi, (_, style) in zip((0, *bounds), (*bounds, len(path) - 1), PHASE_STYLES):
        ax.plot(path[lo:hi + 1, 0], path[lo:hi + 1, 1], style, color=color, lw=1.5)
    ax.scatter(*path[-1], color=color, s=60, zorder=3)


def _labels(variant: str) -> torch.Tensor:
    s = VARIANTS[variant]
    return torch.repeat_interleave(torch.arange(len(s.counts)), s.dup * torch.tensor(s.counts))


def fig_single(variant: str = "single") -> None:
    """Fig 4 B (classifier weights), C (features), D (RankMe + top eigenvalues)."""
    res, t = _results(variant), _traj(variant)
    steps, rm = _rankme(res)
    bounds = _phase_bounds(np.array([rm[np.searchsorted(steps, s)] for s in t["steps"].tolist()]))
    fig, (bx, cx, dx) = plt.subplots(1, 3, figsize=(15, 4.2))
    for i in range(len(VARIANTS[variant].counts)):
        _plot_phased(bx, t["W"][:, :2, i].numpy(), bounds, CLASS_COLORS[i])
    y = _labels(variant)
    for j in range(0, len(y), VARIANTS[variant].dup):
        _plot_phased(cx, t["F"][:, j, :2].numpy(), bounds, CLASS_COLORS[int(y[j])])
    bx.set(title="$W_i$", xlabel="dim 1", ylabel="dim 2")
    cx.set(title=r"$f_\theta(x)$", xlabel="dim 1", ylabel="dim 2")
    dx.plot(steps, rm, color="navy", lw=2, label="RankMe")
    dx.set(title="RankMe & eigenvalues", xlabel="training steps", ylabel="RankMe")
    ex = dx.twinx()
    lam = t["sigma"][:, -1, :2].numpy() ** 2
    for i, (color, label) in enumerate((("violet", r"$\sigma_1$"), ("yellowgreen", r"$\sigma_2$"))):
        ex.plot(t["steps"], lam[:, i], color=color, lw=2, label=label)
    ex.set_ylabel(r"eigenvalues ($\lambda$)")
    for idx in bounds:
        dx.axvline(float(t["steps"][idx]), color="gray", lw=0.8, alpha=0.6)
    fig.legend(loc="lower center", ncol=3,
               handles=[Line2D([], [], color="gray", ls=s, label=n) for n, s in PHASE_STYLES])
    fig.suptitle(f"toy {variant}: cross-entropy GD phases")
    _save(fig, f"fig4_{variant}")


def fig_controls(variants: tuple[str, ...] = ("single", "uniform", "nobottleneck",
                                              "mse_uniform", "mse_skew")) -> None:
    """RankMe curves: only `single` keeps the compression decline; every control is monotone."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in variants:
        ax.plot(*_rankme(_results(v)), lw=2, label=v)
    ax.set(xscale="log", xlabel="training steps (log)", ylabel="RankMe(features)",
           title="Controls remove the compression phase")
    ax.legend()
    _save(fig, "fig_controls")


def fig_multi(variant: str) -> None:
    """Stream RankMe by depth, ledger term sums, and ledger trace weights (H3.3 guard)."""
    res = _results(variant)
    steps = np.array(sorted(res))
    depth = VARIANTS[variant].depth
    fig, (sx, lx, wx) = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, node in enumerate([f"blk{k}.attn.in" for k in range(depth)] + ["before_final_norm"]):
        sx.plot(*_rankme(res, node), lw=1.5, color=plt.get_cmap("viridis")(k / depth), label=f"depth {k}")
    sx.set(xscale="log", xlabel="training steps (log)", ylabel="RankMe(stream)", title="stream by depth")
    sx.legend(fontsize=7)
    ledgers = {s: [res[s][f"blk{k}"]["block_ledger"] for k in range(depth)] for s in steps}
    for term in LEDGER_TERMS:
        lx.plot(steps, [sum(float(l[term]) for l in ledgers[s]) for s in steps], lw=2, label=term)
    lx.axhline(0, color="gray", lw=0.8)
    lx.set(xscale="log", xlabel="training steps (log)", ylabel="sum over blocks", title="rank ledger")
    lx.legend()
    w = np.stack([res[s][""]["overlap_chi"]["w"] for s in steps])
    for k in range(w.shape[1]):
        wx.plot(steps, w[:, k], lw=1, color=plt.get_cmap("viridis")(k / max(w.shape[1] - 1, 1)))
    wx.plot(steps, w.max(1), color="crimson", lw=2, label=r"$\max_k w_k$")
    wx.set(xscale="log", xlabel="training steps (log)", ylabel="trace weight $w_k$",
           title="write energy shares (representativeness guard)", ylim=(0, 1.05))
    wx.legend()
    fig.suptitle(f"toy {variant}")
    _save(fig, f"fig_multi_{variant}")


def _save(fig: Figure, name: str) -> None:
    os.makedirs(FIGURES, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/{name}.png", dpi=180)
    plt.close(fig)
    print(f"saved {FIGURES}/{name}.png")


def main() -> None:
    fig_single()
    fig_controls()
    for v in ("multi_residual", "multi_residual_nonlinear", "multi_plain"):
        fig_multi(v)


if __name__ == "__main__":
    main()
