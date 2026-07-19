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
PHASE_STYLES = (("warmup", ":"), ("entropy-seeking", "-"), ("compression-seeking", "--"),
                ("recovery", "-"))
CLASS_COLORS = ("magenta", "orange", "royalblue", "seagreen", "orchid", "goldenrod", "steelblue", "olive")
CLASS_MARKERS = ("^", "o", "s", "D", "v", "P", "X", "*")   # Li et al: blue=square, green=diamond
LEDGER_TERMS = ("delta_s", "chi", "quality", "interference")
NORM_COLORS = {"none": "#D55E00", "prenorm": "#0072B2", "writenorm": "#009E73", "bothnorm": "#CC79A7"}
ARCH_CELLS = {"arch": "none", "arch_pre": "prenorm", "arch_wn": "writenorm", "arch_bn": "bothnorm"}


def _results(variant: str) -> dict[int, dict]:
    return np.load(f"{RESULTS}/toy_{variant}/results_toy-{variant}.npy", allow_pickle=True).item()


def _traj(variant: str) -> dict[str, torch.Tensor]:
    return torch.load(f"{RESULTS}/toy/trajectories_{variant}.pt", weights_only=True)


def _rankme(res: dict[int, dict], node: str = "before_final_norm") -> tuple[np.ndarray, np.ndarray]:
    steps = np.array(sorted(res))
    return steps, np.array([res[s][node]["acts_uncentered"]["rankme"] for s in steps])


def _phase_bounds(rm: np.ndarray, eps: float = 5e-4) -> tuple[int, int, int]:
    """(dip, peak, trough): the compression segment is the largest post-warmup DRAWDOWN
    (running max minus curve, computed after the warmup dip) — horizon-independent, unlike
    a global argmax, which lands on the window edge whenever the transient recovers.
    Drawdown < eps -> no compression segment (peak = trough = end)."""
    g = int(np.argmax(rm))
    dip = int(np.argmin(rm[:g + 1]))
    seg = rm[dip:]
    dd = np.maximum.accumulate(seg) - seg
    if dd.max() < eps:
        return dip, len(rm) - 1, len(rm) - 1
    trough = dip + int(np.argmax(dd))
    peak = dip + int(np.argmax(rm[dip:trough + 1]))
    return dip, peak, trough


def _plot_phased(ax: Axes, path: np.ndarray, bounds: tuple[int, int, int], color: str,
                 marker: str = "o") -> None:
    """One 2D trajectory split into the three phase segments (paper line styles)."""
    for lo, hi, (_, style) in zip((0, *bounds), (*bounds, len(path) - 1), PHASE_STYLES):
        ax.plot(path[lo:hi + 1, 0], path[lo:hi + 1, 1], style, color=color, lw=1.5)
    ax.scatter(*path[-1], color=color, s=60, marker=marker, zorder=3)


def _labels(variant: str) -> torch.Tensor:
    s = VARIANTS[variant]
    return torch.repeat_interleave(torch.arange(len(s.counts)), s.dup * torch.tensor(s.counts))


def fig_single(variant: str = "single") -> None:
    """Fig 4 B (classifier weights), C (features), D (RankMe + top eigenvalues)."""
    res, t = _results(variant), _traj(variant)
    steps, rm = _rankme(res)
    ts = t["steps"].numpy()
    rm_t = np.array([rm[np.searchsorted(steps, s)] for s in ts.tolist()])
    bounds = _phase_bounds(rm_t)
    counts = VARIANTS[variant].counts
    fig, (bx, cx, dx) = plt.subplots(1, 3, figsize=(15, 4.2))
    for i in range(len(counts)):
        _plot_phased(bx, t["W"][:, :2, i].numpy(), bounds, CLASS_COLORS[i], CLASS_MARKERS[i])
    y = _labels(variant)
    for j in range(0, len(y), VARIANTS[variant].dup):
        c = int(y[j])
        _plot_phased(cx, t["F"][:, j, :2].numpy(), bounds, CLASS_COLORS[c], CLASS_MARKERS[c])
    bx.set(title="$W_i$", xlabel="dim 1", ylabel="dim 2", aspect="equal")
    cx.set(title=r"$f_\theta(x)$", xlabel="dim 1", ylabel="dim 2", aspect="equal")
    bx.legend(fontsize=7, handles=[
        Line2D([], [], color=CLASS_COLORS[i], marker=CLASS_MARKERS[i], ls="-",
               label=f"class {i} (n={n})") for i, n in enumerate(counts)])
    dx.set(title="RankMe & eigenvalues", xlabel="training steps", ylabel="RankMe")
    ex = dx.twinx()
    lam = t["sigma"][:, -1, :2].numpy() ** 2 / VARIANTS[variant].dup   # per unique sample set:
    # sigma is computed on the dup-replicated rows, which scales every eigenvalue by dup;
    # dividing restores the paper's N=6 scale (the dynamics are dup-exact already)
    for lo, hi, (_, style) in zip((0, *bounds), (*bounds, len(ts) - 1), PHASE_STYLES):
        dx.plot(ts[lo:hi + 1], rm_t[lo:hi + 1], style, color="navy", lw=2)
        for i, color in enumerate(("violet", "yellowgreen")):
            ex.plot(ts[lo:hi + 1], lam[lo:hi + 1, i], style, color=color, lw=2)
    ex.set_ylabel(r"eigenvalues ($\lambda_i = \sigma_i^2$)")
    for idx in bounds:
        dx.axvline(float(ts[idx]), color="gray", lw=0.8, alpha=0.6)
    dx.legend(fontsize=8, handles=[
        Line2D([], [], color="navy", lw=2, label="RankMe (phase-styled)"),
        Line2D([], [], color="violet", lw=2, label=r"$\lambda_1$"),
        Line2D([], [], color="yellowgreen", lw=2, label=r"$\lambda_2$")])
    fig.legend(loc="lower center", ncol=3,
               handles=[Line2D([], [], color="gray", ls=s, label=n) for n, s in PHASE_STYLES])
    fig.suptitle(f"toy {variant}: cross-entropy GD phases")
    _save(fig, f"fig4_{variant}")


def fig_controls(variants: tuple[str, ...] = ("single_long", "uniform", "nobottleneck",
                                              "mse_uniform", "mse_skew")) -> None:
    """RankMe curves, one panel per variant (own axes): only the skew+bottleneck+CE reference
    shows the peak-then-decline; every control is monotone. `single_long` = the canonical
    clustered `single` spec run to 3000 steps, showing the decline AND its transience."""
    fig, axes = plt.subplots(1, len(variants), figsize=(3.6 * len(variants), 3.4))
    for ax, v in zip(axes, variants):
        steps, rm = _rankme(_results(v))
        ax.plot(steps, rm, lw=2, color="tab:blue")
        d = VARIANTS[v.removesuffix("_long")].d
        ax.set(xscale="log", xlabel="steps (log)", title=f"{v} (d={d})")
        if v.startswith("single"):
            ax.axvline(300, color="gray", lw=0.8, ls=":")   # the paper's Fig-4 window
    axes[0].set_ylabel("RankMe(features)")
    fig.suptitle("Only skew + bottleneck + CE compresses — controls are monotone "
                 "(single shown past the paper's 300-step window: the decline is transient)")
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


def fig_arch(depth: int = 6, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """Signature figure for the architecture-knob grid: stream RankMe and summed quality
    ledger term over training (one thin line per run), plus the final rogue check (min-write
    RankMe vs its ledger-w energy share). Color = norm cell, line style / marker = wiring."""
    fig, (rx, qx, gx) = plt.subplots(1, 3, figsize=(16, 4.6))
    for cell, norm in ARCH_CELLS.items():
        for wiring, ls, marker in (("par", "-", "o"), ("seq", "--", "s")):
            for seed in seeds:
                res = _results(f"{cell}_{wiring}_s{seed}")
                steps = np.array(sorted(res))
                rm = [res[s]["before_final_norm"]["acts_centered"]["rankme"] for s in steps]
                q = [sum(res[s][f"blk{k}"]["block_ledger"]["quality"] for k in range(depth))
                     for s in steps]
                rx.plot(steps, rm, ls, color=NORM_COLORS[norm], lw=1.2, alpha=0.75)
                qx.plot(steps, q, ls, color=NORM_COLORS[norm], lw=1.2, alpha=0.75)
                fin = steps[-1]
                writes = [f"blk{k}.{n}.out" for k in range(depth) for n in ("attn", "mlp")]
                wrm = {w: res[fin][w]["acts_centered"]["rankme"] for w in writes}
                mn = min(wrm, key=lambda w: wrm[w])
                blk, kind = mn.split(".")[:2]
                w = res[fin][blk]["block_ledger"]["w"]
                gx.scatter(w[1 if kind == "attn" else 2] / w[0], wrm[mn], s=50,
                           color=NORM_COLORS[norm], marker=marker, edgecolors="white", lw=0.8)
    rx.set(xscale="log", xlabel="training steps (log)",
           ylabel="RankMe (centered FINAL stream)",
           title="final-stream RankMe (post-last-block; centered, unlike the\nuncentered Li-facing figures)")
    qx.axhline(0, color="gray", lw=0.8)
    qx.set(xscale="log", xlabel="training steps (log)", ylabel="sum over blocks",
           title="quality ledger term (the Pythia carrier)")
    gx.set(xscale="log", xlabel="min write energy / stream energy (ledger $w$, log)",
           ylabel="min final write RankMe", title="rogue check (final step)")
    fig.legend(loc="lower center", ncol=6, handles=[
        *(Line2D([], [], color=c, lw=2, label=n) for n, c in NORM_COLORS.items()),
        Line2D([], [], color="gray", ls="-", marker="o", label="parallel"),
        Line2D([], [], color="gray", ls="--", marker="s", label="sequential")])
    fig.suptitle("architecture-knob grid: norm cell signatures (3 seeds each)")
    _save(fig, "fig_arch_grid")


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
    fig_arch()


if __name__ == "__main__":
    main()
