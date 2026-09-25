"""PDF versions of the toy appendix figures, for the thesis.

Three of them are the appendix's rx = -0.25 snapshots, which were screen captures of
analysis/fork_lag.ipynb's animation 1 at steps 190 / 216 / 300. The drawing code here is
that animation's frame function, unchanged, so the output is the same figure in vector
form; the trajectory is read from the saved sweep rather than recomputed (_init is seeded,
so the two agree, and the saved one is what the screenshots show: RankMe 1.9990 / 1.9992 /
1.9944 at the three steps).

The fourth is a Fig-4-style panel for the same rx = -0.25 run, built with toy.plots'
own helpers so it matches fig4_single. The unmodified reproduction, fig4_single itself,
comes from toy.plots.fig_single.

Writes to thesis-writing/thesis-tex/figures/. CPU, a couple of seconds.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
from matplotlib.lines import Line2D

from toy.plots import (CLASS_COLORS, CLASS_MARKERS, PHASE_STYLES, _phase_bounds,
                       _plot_phased)   # noqa: F401  (Agg is set by importing toy.plots)
import matplotlib.pyplot as plt

from toy.train import SKEW

SWEEPS = "data/results/toy/appendix_sweeps.pt"
OUT = "thesis-writing/thesis-tex/figures"
RX = "-0.25"
FIG4_RXS = ("0", "-0.25")                          # "0" = the init Li et al. reproduce
SNAPS = ((190, "rotation_start"), (216, "max_entropy"), (300, "end"))
COLS = ("magenta", "orange", "royalblue", "seagreen")
Y = torch.repeat_interleave(torch.arange(len(SKEW)), torch.tensor(SKEW)).tolist()


def _load(rx: str = RX) -> dict:
    r = torch.load(SWEEPS, weights_only=True)["rx"][rx]
    F, W = r["theta_path"].numpy(), r["W_path"].numpy()
    rm, cosw = r["rm"].numpy(), r["cos_w"].numpy()
    T = len(F)
    lams, us = np.zeros((T, 2)), np.zeros((T, 2, 2))
    for t in range(T):
        w_, V = np.linalg.eigh(F[t].T @ F[t])
        lams[t], us[t] = w_[::-1], V[:, ::-1]
    for t in range(1, T):                          # fix eigvec sign flips between frames
        for k in (0, 1):
            if us[t, :, k] @ us[t - 1, :, k] < 0:
                us[t, :, k] *= -1
    return dict(F=F, W=W, rm=rm, lam=r["lam"].numpy(), lams=lams, us=us, T=T,
                fork=int(np.argmax((cosw < 0.97) & (np.arange(T) > 10))),
                peak=int(np.argmax(rm)))


# --- animation 1's frame, verbatim from analysis/fork_lag.py -------------------------

def _panel(ax, title, L):
    ax.clear()
    ax.set(xlim=(-L, L), ylim=(-L, L), aspect='equal', title=title,
           xlabel='dim 1', ylabel='dim 2')
    ax.axhline(0, color='0.9', lw=0.6)
    ax.axvline(0, color='0.9', lw=0.6)


def _eigaxes(ax, d, t, L):
    for k, c in ((0, 'violet'), (1, 'yellowgreen')):
        u = d['us'][t, :, k]
        s = np.sqrt(d['lams'][t, k] / 6)           # RMS per-sample extent along u_k
        ax.plot([-L * u[0], L * u[0]], [-L * u[1], L * u[1]], ':', color=c, lw=1)
        ax.plot([0, s * u[0]], [0, s * u[1]], '-', color=c, lw=3.5)


def snapshot(d: dict, t: int, name: str, rx: str = RX) -> None:
    F, W = d['F'], d['W']
    lim, wlim = 1.2 * np.abs(F).max(), 1.2 * np.abs(W).max()
    fig, (ax_w, ax_f) = plt.subplots(1, 2, figsize=(11, 5.4))
    _panel(ax_w, '$W_i$', wlim)
    _panel(ax_f, r'$f_\theta(x)$', lim)
    t0 = max(0, t - 40)
    for i in range(4):
        ax_w.plot(W[t0:t + 1, 0, i], W[t0:t + 1, 1, i], '-', color=COLS[i], lw=0.7, alpha=0.5)
        ax_w.scatter(*W[t, :, i], s=70, color=COLS[i])
    for j, c in enumerate(Y):
        ax_f.plot(F[t0:t + 1, j, 0], F[t0:t + 1, j, 1], '-', color=COLS[c], lw=0.7, alpha=0.5)
        ax_f.scatter(*F[t, j], s=50, color=COLS[c])
    _eigaxes(ax_f, d, t, lim)
    fig.suptitle(f'rx = {rx}   step {t}   RankMe {d["rm"][t]:.4f}'
                 f'   {"(fork passed)" if t >= d["fork"] else ""}'
                 f'   {"(declining)" if t >= d["peak"] else ""}')
    _write(fig, f'toy_rx{rx.lstrip("-")}_{name}')


# --- Fig-4-style panel for the same run ---------------------------------------------

def fig4_rx(d: dict, rx: str = RX) -> None:
    """Fig 4 B (classifier weights), C (features), D (RankMe + top eigenvalues), as
    toy.plots.fig_single draws them. dup is 1 here, so the eigenvalues need no rescaling."""
    rm, ts = d['rm'], np.arange(d['T'])
    bounds = _phase_bounds(rm)
    fig, (bx, cx, dx) = plt.subplots(1, 3, figsize=(15, 4.2))
    for i in range(len(SKEW)):
        _plot_phased(bx, d['W'][:, :2, i], bounds, CLASS_COLORS[i], CLASS_MARKERS[i])
    for j, c in enumerate(Y):
        _plot_phased(cx, d['F'][:, j, :2], bounds, CLASS_COLORS[c], CLASS_MARKERS[c])
    bx.set(title="$W_i$", xlabel="dim 1", ylabel="dim 2", aspect="equal")
    cx.set(title=r"$f_\theta(x)$", xlabel="dim 1", ylabel="dim 2", aspect="equal")
    bx.legend(fontsize=7, handles=[
        Line2D([], [], color=CLASS_COLORS[i], marker=CLASS_MARKERS[i], ls="-",
               label=f"class {i} (n={n})") for i, n in enumerate(SKEW)])
    dx.set(title="RankMe & eigenvalues", xlabel="training steps", ylabel="RankMe")
    ex = dx.twinx()
    for lo, hi, (_, style) in zip((0, *bounds), (*bounds, d['T'] - 1), PHASE_STYLES):
        dx.plot(ts[lo:hi + 1], rm[lo:hi + 1], style, color="navy", lw=2)
        for i, color in enumerate(("violet", "yellowgreen")):
            ex.plot(ts[lo:hi + 1], d['lam'][lo:hi + 1, i], style, color=color, lw=2)
    ex.set_ylabel(r"eigenvalues ($\lambda_i = \sigma_i^2$)")
    for idx in bounds:
        dx.axvline(float(ts[idx]), color="gray", lw=0.8, alpha=0.6)
    dx.legend(fontsize=8, handles=[
        Line2D([], [], color="navy", lw=2, label="RankMe (phase-styled)"),
        Line2D([], [], color="violet", lw=2, label=r"$\lambda_1$"),
        Line2D([], [], color="yellowgreen", lw=2, label=r"$\lambda_2$")])
    fig.legend(loc="lower center", ncol=3,
               handles=[Line2D([], [], color="gray", ls=s, label=n) for n, s in PHASE_STYLES])
    fig.suptitle(f"rx = {rx}")
    _write(fig, f'toy_rx{rx.lstrip("-")}_fig4', tight=True)


def controls(variants=("single_long", "uniform", "nobottleneck", "mse_uniform", "mse_skew")
             ) -> None:
    """toy.plots.fig_controls without its suptitle, whose claim that the controls are
    monotone is not true of the mse_skew panel; the caption carries the reading instead."""
    from toy.plots import _phase_bounds, _rankme, _results
    from toy.train import VARIANTS
    fig, axes = plt.subplots(1, len(variants), figsize=(3.6 * len(variants), 3.4))
    for ax, v in zip(axes, variants):
        steps, rm = _rankme(_results(v))
        ax.plot(steps, rm, lw=2, color="tab:blue")
        d = VARIANTS[v.removesuffix("_long")].d
        ax.set(xscale="log", xlabel="steps (log)", title=f"{v} (d={d})")
        if v.startswith("single"):
            ax.axvline(300, color="gray", lw=0.8, ls=":")   # the paper's Fig-4 window
    axes[0].set_ylabel("RankMe(features)")
    _write(fig, "toy_controls", tight=True)


def _write(fig, name: str, tight: bool = False) -> None:
    """No bbox trim: the snapshots are the animation's canvas as it was captured, and
    tight=True is toy.plots._save's treatment, which fig4 panels are drawn for."""
    os.makedirs(OUT, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print(f"saved {OUT}/{name}.pdf")


def main() -> None:
    d = _load()
    print(f'rx={RX}: fork@{d["fork"]}, peak@{d["peak"]}')
    for t, name in SNAPS:
        snapshot(d, t, name)
    controls()
    for rx in FIG4_RXS:                            # rx = 0 is the unmodified reproduction
        fig4_rx(_load(rx) if rx != RX else d, rx)


if __name__ == "__main__":
    main()
