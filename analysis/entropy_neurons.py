# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: representation-geometry (3.14.0)
#     language: python
#     name: python3
# ---

# %%
# Entropy neurons across pretraining. Stolfo et al., "Confidence Regulation Neurons in
# Language Models" (arXiv:2406.16254), identify final-layer
# MLP neurons that write into the effective null space of the unembedding -- the span V0 of
# its bottom-k right singular vectors -- and so move the residual norm (hence the final-norm
# gain, hence logit temperature) without moving the logits. Their statistic is
#     rho_i = ||V0^T w_out^(i)|| / ||w_out^(i)||,
# paired with a high weight norm and a low LogitVar. They never ask when these form.
# Data: oneoff_scripts/entropy_neurons.py (weights only, log checkpoint schedule).
import os, sys
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import torch

from analysis.experiments_lib import get_model_label

CENTERED = "data/results/entropy_neurons.pt"        # W_U centred over the vocabulary
PAPER = "data/results/entropy_neurons_paper.pt"     # the paper's convention, uncentred
THRESHOLD = 0.5                                     # "most of the write is in the null space"
N_BLOCKS_SHOWN = 2      # blocks drawn per model, counting back from the last. 1 = final block
                        # only (no penultimate/mid-stack control curves)
# RANKME_HOOK = "before_final_norm"
RANKME_HOOK = "after_final_norm"

data = torch.load(CENTERED, weights_only=False)
paper = torch.load(PAPER, weights_only=False) if os.path.exists(PAPER) else {}
MODELS = [m for m in data if data[m]]
COLORS = dict(zip(MODELS, plt.cm.tab10.colors))
print({m: {b: len(v) for b, v in data[m].items()} for m in MODELS})


# %%
STYLES = ["-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]   # dotted is reserved for the chance line
STYLE_NAMES = ["last block", "penultimate", "mid-stack", "control"]


def style_legend(ax, n, **kw):
    """Second legend naming what each line style is, so the plot doesn't need the prose."""
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="k", ls=s, label=lab)
               for s, lab in list(zip(STYLES, STYLE_NAMES))[:n]]
    ax.add_artist(ax.legend(handles=handles, **kw))


def blocks_of(d, model):
    """Last block first, then descending: penultimate, mid-stack control, ...
    Truncated to N_BLOCKS_SHOWN, so the controls can be switched off from one place."""
    return sorted(d[model], reverse=True)[:N_BLOCKS_SHOWN]


def k_index(entry, frac=0.01):
    """Index into the stored ks of the paper's k ~ 0.01 d_model, and that k."""
    k = max(1, round(frac * entry["d_model"]))
    return entry["ks"].index(k), k


def series(per_step, fn, frac=0.01):
    """(tokens, values) over checkpoints, skipping zero-init blocks (nanochat steps 0-1,
    whose residual down-projection is identically zero, so rho is 0/0)."""
    steps = [s for s in sorted(per_step) if float(per_step[s]["norm"].max()) > 0]
    xs = np.array([per_step[s]["tokens"] for s in steps], dtype=float)
    ki = k_index(per_step[steps[0]], frac)[0]
    return xs, np.array([fn(per_step[s]["rho"][ki].numpy(), per_step[s]) for s in steps])


def frac_of_run(per_step, xs):
    return xs / max(e["tokens"] for e in per_step.values())


def chance(entry, q, frac=0.01, trials=10, seed=0):
    """The q-quantile of rho over a layer of randomly oriented neurons. sqrt(k/d) is the
    MEAN for one direction; the max over thousands of them is far above that, so any
    reference drawn against max/q99 has to be the matching order statistic."""
    d, n = entry["d_model"], entry["norm"].numel()
    k = k_index(entry, frac)[1]
    rng = np.random.default_rng(seed)
    qs = [np.quantile(np.sqrt((g := rng.standard_normal((n, d)) ** 2)[:, :k].sum(1) / g.sum(1)), q)
          for _ in range(trials)]
    return float(np.mean(qs))


# %%
# The RankMe peak of the final stream, per model, to mark on every training-time axis below.
# Rule: the curve dips at initialisation before the entropy-seeking rise, so take the trough
# within the first half of the run and the maximum after it -- scale-free, which matters
# because nanochat's run is two orders of magnitude shorter than OLMo's.
from analysis.experiments_lib import get_ys, get_xs_tokens

def rankme_curve(model, leaf=(RANKME_HOOK, "acts_centered")):
    cfg = "nanochat_samples" if model == "nanochat-d12" else "block_representations_samples"
    ys, steps = get_ys(cfg, model, leaf, "rankme")
    return (None, None) if ys is None else (np.asarray(get_xs_tokens(model, steps), float),
                                            np.asarray(ys, float))

def alpha_curve(model, leaf=(RANKME_HOOK, "acts_centered")):
    cfg = "nanochat_samples" if model == "nanochat-d12" else "block_representations_samples"
    ys, steps = get_ys(cfg, model, leaf, "alpha")
    return (None, None) if ys is None else (np.asarray(get_xs_tokens(model, steps), float),
                                            np.asarray(ys, float))


def rankme_peak(model):
    xs, ys = rankme_curve(model)
    if ys is None:
        return None
    i0 = int(np.argmin(ys[:max(2, len(ys) // 2)]))
    ip = i0 + int(np.argmax(ys[i0:]))
    return {"tokens": xs[ip], "frac": xs[ip] / xs[-1], "rankme": ys[ip], "idx": ip}

PEAKS = {m: p for m in MODELS if (p := rankme_peak(m))}
print({m: f"{p['tokens']:.2e} tokens (frac {p['frac']:.3f})" for m, p in PEAKS.items()})




def mark_peaks(ax, xvar="tokens"):
    """Thin vertical line at each model's RankMe peak, in that model's colour."""
    for model, peak in PEAKS.items():
        ax.axvline(peak[xvar], color=COLORS[model], lw=0.8, alpha=0.55, zorder=0)


# %% [markdown]
# ## 1. When does the null-space write appear?
# Top row: the most null-space-aligned neuron in the block (max rho). Bottom row: the 99th
# percentile. Line style keys the block (see the second legend); horizontal dotted lines are
# the chance level -- the SAME quantile over a layer of randomly oriented neurons, not
# sqrt(k/d), which is the mean of one draw and far below the max of several thousand.
# Left column against tokens seen, right column against fraction of the run, since the three
# families train on very different budgets.

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
for model in MODELS:
    for bi, b in enumerate(blocks_of(data, model)):
        per_step = data[model][b]
        style = STYLES[bi]
        for row, q in enumerate((1.0, 0.99)):
            xs, ys = series(per_step, lambda r, _e, q=q: np.quantile(r, q))
            label = get_model_label(model) if (bi == 0 and row == 0) else None
            axes[row, 0].plot(xs, ys, style, color=COLORS[model], lw=1.5, label=label)
            axes[row, 1].plot(frac_of_run(per_step, xs), ys, style, color=COLORS[model], lw=1.5)
        if bi == 0:
            e = per_step[max(per_step)]
            for row, q in enumerate((1.0, 0.99)):
                for col in (0, 1):
                    axes[row, col].axhline(chance(e, q), color=COLORS[model], ls=":", lw=0.8)
for col, xvar in enumerate(("tokens", "frac")):
    for row in (0, 1):
        axes[row, col].set_xscale("log")
        mark_peaks(axes[row, col], xvar)
axes[0, 0].set(ylabel=r"max $\rho$   ($k = 0.01\,d$)")
axes[1, 0].set(ylabel=r"$\rho$, q=0.99", xlabel="tokens")
axes[1, 1].set(xlabel="fraction of the run")
axes[0, 0].legend(fontsize=8, loc="upper left")
style_legend(axes[0, 1], max(len(blocks_of(data, m)) for m in MODELS), fontsize=8, loc="upper left")
fig.suptitle("Null-space composition of final-MLP write directions over pretraining")
fig.tight_layout()

# %% [markdown]
# ## 2. Is it a population or a single neuron?
# Count of neurons clearing rho > 0.5, i.e. the majority of the write direction inside the
# bottom-k subspace. Left: absolute count. Right: as a share of the block's neurons, which
# is the comparable quantity across widths.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for model in MODELS:
    for bi, b in enumerate(blocks_of(data, model)):
        per_step = data[model][b]
        style = STYLES[bi]
        xs, ns = series(per_step, lambda r, _e: float((r > THRESHOLD).sum()))
        _, fr = series(per_step, lambda r, _e: float((r > THRESHOLD).mean()))
        axes[0].plot(xs, ns, style, color=COLORS[model], lw=1.5,
                     label=get_model_label(model) if bi == 0 else None)
        axes[1].plot(xs, fr, style, color=COLORS[model], lw=1.5)
for ax, lab in zip(axes, (f"# neurons  " + r"$\rho > $" + f"{THRESHOLD}",
                          "share of neurons")):
    ax.set(xscale="log", xlabel="tokens", ylabel=lab)
    mark_peaks(ax)
axes[1].set_yscale("log")
axes[0].legend(fontsize=8)
style_legend(axes[1], max(len(blocks_of(data, m)) for m in MODELS), fontsize=8, loc="lower right")
fig.tight_layout()

# %% [markdown]
# ## 3. The paper's identification plane
# LogitVar (variance over the vocabulary of the row-normalised logit attribution) against
# rho, for the last block, at the first, middle and final checkpoint. Point size is the
# neuron's weight norm. An entropy neuron is the bottom-right corner: high null-space
# composition, near-zero direct logit effect, and a norm that survives weight decay.

# %%
fig, axes = plt.subplots(len(MODELS), 3, figsize=(11, 2.8 * len(MODELS)), squeeze=False)
for row, model in enumerate(MODELS):
    per_step = data[model][blocks_of(data, model)[0]]
    steps = [s for s in sorted(per_step) if float(per_step[s]["norm"].max()) > 0]
    for col, step in enumerate((steps[0], steps[len(steps) // 2], steps[-1])):
        e = per_step[step]
        ki = k_index(e)[0]
        ax = axes[row, col]
        ax.scatter(e["rho"][ki].numpy(), e["logitvar"].numpy(),
                   s=1 + 30 * (e["norm"] / e["norm"].max()).numpy() ** 2,
                   color=COLORS[model], alpha=0.35, edgecolors="none")
        ax.set(yscale="log", xlim=(0, 1), title=f"step {step}  ({e['tokens']:.1e} tokens)")
        if col == 0:
            ax.set_ylabel(f"{get_model_label(model)}\nLogitVar")
        if row == len(MODELS) - 1:
            ax.set_xlabel(r"$\rho$")
fig.tight_layout()

# %% [markdown]
# ## 4. Does the vocabulary centring matter?
# The paper folds the final norm into W_U but does not centre it. Centring removes the
# all-ones direction, which softmax ignores; it should land in the top singular vectors and
# leave the bottom-k span nearly untouched. Solid = centred, dotted = the paper's convention.

# %%
if paper:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in MODELS:
        if model not in paper or not paper[model]:
            continue
        b = blocks_of(data, model)[0]
        for d_, style in ((data, "-"), (paper, ":")):
            xs, ys = series(d_[model][b], lambda r, _e: r.max())
            ax.plot(xs, ys, style, color=COLORS[model], lw=1.5,
                    label=get_model_label(model) if style == "-" else None)
    mark_peaks(ax)
    ax.set(xscale="log", xlabel="tokens", ylabel=r"max $\rho$", title="centred vs paper convention")
    ax.legend(fontsize=8)
    fig.tight_layout()
else:
    print(f"{PAPER} not written yet")

# %% [markdown]
# ## 5. How much of the population moves
# Quantile fan per model, last block: median through max. If only the extreme quantiles
# lift off the chance line, the phenomenon is a tail of a few neurons rather than a
# reorganisation of the layer.

# %%
QS = (0.5, 0.9, 0.99, 0.999, 1.0)
fig, axes = plt.subplots(1, len(MODELS), figsize=(3.1 * len(MODELS), 3.4), sharey=True)
for ax, model in zip(np.atleast_1d(axes), MODELS):
    per_step = data[model][blocks_of(data, model)[0]]
    for q, shade in zip(QS, np.linspace(0.25, 1.0, len(QS))):
        xs, ys = series(per_step, lambda r, _e, q=q: np.quantile(r, q))
        ax.plot(xs, ys, color=COLORS[model], alpha=shade, lw=1.4, label=f"q={q:g}")
    e = per_step[max(per_step)]
    for q in QS:
        ax.axhline(chance(e, q), color="k", ls=":", lw=0.6)
    ax.axvline(PEAKS[model]["tokens"], color="k", lw=0.8, alpha=0.5, zorder=0)
    ax.set(xscale="log", xlabel="tokens", title=get_model_label(model))
np.atleast_1d(axes)[0].set_ylabel(r"$\rho$")
np.atleast_1d(axes)[-1].legend(fontsize=7)
fig.tight_layout()


# %% [markdown]
# ## 6. The two curves side by side
# RankMe of the centered final stream (left axis) against max rho in the last block (right
# axis), one panel per model. The vertical line is the RankMe peak. pythia-160m is left out;
# it is the family's edge case and the grid reads better as three Pythias over
# nanochat plus the two OLMos.

# %%
GRID = ["pythia-410m-deduped", "pythia-1b-deduped", "pythia-6.9b-deduped",
        "nanochat-d12", "OLMo-2-0425-1B", "OLMo-2-1124-7B"]

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, model in zip(axes.ravel(), GRID):
    xs, ys = rankme_curve(model)
    peak_rm = PEAKS[model]["rankme"] if model in PEAKS else None
    if ys is not None:
        ax.plot(xs, ys, color="0.35", lw=1.6, label="RankMe")
        if peak_rm is not None:
            ax.set_ylim(min(ys) - 0.05 * (peak_rm - min(ys)), peak_rm + 0.05 * (peak_rm - min(ys)))
    if model in data and data[model]:
        per_step = data[model][blocks_of(data, model)[0]]
        tx, ty = series(per_step, lambda r, _e: r.max())
        twin = ax.twinx()
        twin.plot(tx, ty, color=COLORS.get(model, "C3"), lw=1.6, label=r"max $\rho$")
        twin.set(ylim=(0, 1.02), ylabel=r"max $\rho$")
        twin.axhline(chance(per_step[max(per_step)], 1.0), color=COLORS.get(model, "C3"),
                     ls=":", lw=0.8)
    if model in PEAKS:
        ax.axvline(PEAKS[model]["tokens"], color="k", lw=0.9, alpha=0.6, zorder=0)
    ax.set(xscale="log", xlabel="tokens", ylabel="RankMe", title=get_model_label(model))
fig.suptitle(r"RankMe of the final stream vs null-space alignment of the last MLP")
fig.tight_layout()

# %% [markdown]
# The same grid with a third axis: what zero-ablating the last quarter of the blocks does to
# the final stream's RankMe (ablated − baseline, from data/results/ablate_blk*), on the
# checkpoints the two runs share. Each y-axis wears its curve's colour.

# %%
DEPTH = {"pythia-160m-deduped": 12, "pythia-410m-deduped": 24, "pythia-1b-deduped": 16,
         "pythia-6.9b-deduped": 32, "OLMo-2-0425-1B": 16, "OLMo-2-1124-7B": 32,
         "nanochat-d12": 12}
LAST_QUARTER = {12: "ablate_blk9-11", 16: "ablate_blk12-15",      # the quarter split of
                24: "ablate_blk18-23", 32: "ablate_blk24-31"}     # ledger_ablation_layers.py


def ablation_gap(model, yvar="rankme", leaf=(RANKME_HOOK, "acts_centered")):
    """(tokens, the metric with the last quarter of blocks zero-ablated − the baseline) over
    the checkpoints both runs carry; the ablation sweeps are the sparser of the two. yvar
    picks what is differenced — with matrix_entropy this is a difference of entropies, which
    is the log of the RankMe RATIO, not the log of the RankMe difference."""
    cfg = "nanochat_samples" if model == "nanochat-d12" else "block_representations_samples"
    base, steps = get_ys(cfg, model, leaf, yvar)
    abl, asteps = get_ys(LAST_QUARTER[DEPTH[model]], model, leaf, yvar)
    if base is None or abl is None:
        return None, None
    b = dict(zip(steps, base))
    shared = [(s, a) for s, a in zip(asteps, abl) if s in b]
    return (np.asarray(get_xs_tokens(model, [s for s, _ in shared]), float),
            np.asarray([a - b[s] for s, a in shared], float))


def color_axis(a, color, side="right"):
    """Tie one y-axis' ticks, label and spine to its curve — three scales need it."""
    a.tick_params(axis="y", colors=color)
    a.yaxis.label.set_color(color)
    a.spines[side].set_color(color)


def gap_floor(gaps, margin=0.05):
    """How far below zero the ablation axis may reach, as a fraction of a panel's own peak.
    One model (Pythia 410m) ends far below zero and drags its axis down; every other panel
    had good limits already. So take each panel's would-be autoscaled bottom (data range
    including the zero line, plus matplotlib's default 5% margin) over its peak, and return
    the second lowest: the floor is then exactly the deepest any OTHER model goes, and only
    the one outlier is clipped."""
    lows = sorted((min(g.min(), 0) - margin * (max(g.max(), 0) - min(g.min(), 0))) / g.max()
                  for g in gaps if g is not None and g.max() > 0)
    return lows[1] if len(lows) > 1 else lows[0]


GAPS = {m: ablation_gap(m) for m in GRID}
FLOOR = gap_floor([g for _, g in GAPS.values()])

fig, axes = plt.subplots(2, 3, figsize=(18, 7.5))
for ax, model in zip(axes.ravel(), GRID):
    xs, ys = rankme_curve(model)
    peak_rm = PEAKS[model]["rankme"] if model in PEAKS else None
    if ys is not None:
        ax.plot(xs, ys, color="0.35", lw=1.6, label="RankMe")
        if peak_rm is not None:
            ax.set_ylim(min(ys) - 0.05 * (peak_rm - min(ys)), peak_rm + 0.05 * (peak_rm - min(ys)))
    if model in data and data[model]:
        per_step = data[model][blocks_of(data, model)[0]]
        tx, ty = series(per_step, lambda r, _e: r.max())
        twin = ax.twinx()
        twin.plot(tx, ty, color=COLORS.get(model, "C3"), lw=1.6, label=r"max $\rho$")
        twin.set(ylim=(0, 1.02), ylabel=r"max $\rho$")
        twin.axhline(chance(per_step[max(per_step)], 1.0), color=COLORS.get(model, "C3"),
                     ls=":", lw=0.8)
        twin.grid(False)                       # one grid (the RankMe axis') for all three
        color_axis(twin, COLORS.get(model, "C3"))
    gx, gy = GAPS[model]
    if gy is None:
        print(f"{model}: no {LAST_QUARTER[DEPTH[model]]} run — ablation gap left out")
    else:
        gap = ax.twinx()
        gap.spines["right"].set_position(("axes", 1.22))          # its own scale, further out
        gap.plot(gx, gy, color="tab:red", ls="--", lw=1.6,   # dashed: one model's colour IS red
                 label="RankMe change under ablation")
        gap.axhline(0, color="tab:red", ls=":", lw=0.8)
        gap.set_ylabel("RankMe change (last quarter ablated)")
        gap.set_ylim(bottom=max(gap.get_ylim()[0], FLOOR * gy.max()))
        gap.grid(False)
        color_axis(gap, "tab:red")
    if model in PEAKS:
        ax.axvline(PEAKS[model]["tokens"], color="k", lw=0.9, alpha=0.6, zorder=0)
    ax.set(xscale="log", xlabel="tokens", ylabel="RankMe", title=get_model_label(model))
fig.suptitle(r"RankMe of the final stream vs null-space alignment of the last MLP, "
             r"with the last-quarter ablation gap")
fig.tight_layout()
fig.subplots_adjust(wspace=0.85, right=0.90)   # room for the offset spine tight_layout misses

# %% [markdown]
# The same grid with the null-space curve changed from the single most aligned neuron to the
# layer's whole write mass: rho of the block's MLP write matrix, ||V0^T W||_F / ||W||_F, which
# is rho_i weighted by each neuron's write norm instead of maximised over neurons. max rho asks
# whether ANY neuron has found the null space; this asks how much of the layer went there, so
# it sits an order of magnitude lower and gets the middle axis to itself, scaled to its own
# range. Its chance level is exactly sqrt(k/d) -- an isotropic W puts k/d of its energy in any
# k-dimensional subspace -- drawn dotted, as chance lines are throughout this notebook.

# %%
from matplotlib.lines import Line2D


def rho_mass(r, e):
    """||V0^T W||_F / ||W||_F for the block's whole MLP write matrix: rho computed on the
    layer's mass rather than on one neuron."""
    w = e["norm"].numpy()
    return float(np.sqrt(((r * w) ** 2).sum() / (w ** 2).sum()))


def mass_chance(entry):
    """sqrt(k/d): what rho_mass reads for an isotropic write matrix."""
    return float(np.sqrt(k_index(entry)[1] / entry["d_model"]))


LEGEND = [("0.35", "-", "RankMe"), ("k", "-", r"write mass in $\rho$"),
          ("tab:red", "--", "change under ablation")]

fig, axes = plt.subplots(2, 3, figsize=(18, 7.5))
for i, (ax, model) in enumerate(zip(axes.ravel(), GRID)):
    xs, ys = rankme_curve(model)
    peak_rm = PEAKS[model]["rankme"] if model in PEAKS else None
    if ys is not None:
        ax.plot(xs, ys, color="0.35", lw=1.6)
        if peak_rm is not None:
            ax.set_ylim(min(ys) - 0.05 * (peak_rm - min(ys)), peak_rm + 0.05 * (peak_rm - min(ys)))
    if model in data and data[model]:
        per_step = data[model][blocks_of(data, model)[0]]
        color = COLORS.get(model, "C3")
        twin = ax.twinx()
        twin.plot(*series(per_step, rho_mass), color=color, lw=1.6)
        twin.set(ylim=(0, None), ylabel=r"write mass in $\rho$")   # its own range, not max rho's
        twin.axhline(mass_chance(per_step[max(per_step)]), color=color, ls=":", lw=0.8)
        twin.grid(False)
        color_axis(twin, color)
    gx, gy = GAPS[model]
    if gy is None:
        print(f"{model}: no {LAST_QUARTER[DEPTH[model]]} run — ablation gap left out")
    else:
        gap = ax.twinx()
        gap.spines["right"].set_position(("axes", 1.22))
        gap.plot(gx, gy, color="tab:red", ls="--", lw=1.6)
        gap.axhline(0, color="tab:red", ls=":", lw=0.8)
        gap.set_ylabel("RankMe change (last quarter ablated)")
        gap.set_ylim(bottom=max(gap.get_ylim()[0], FLOOR * gy.max()))
        gap.grid(False)
        color_axis(gap, "tab:red")
    if model in PEAKS:
        ax.axvline(PEAKS[model]["tokens"], color="k", lw=0.9, alpha=0.6, zorder=0)
    if i == 0:
        ax.legend(handles=[Line2D([0], [0], color=c, ls=s, lw=1.6, label=lab)
                           for c, s, lab in LEGEND], fontsize=7, loc="upper left")
    ax.set(xscale="log", xlabel="tokens", ylabel="RankMe", title=get_model_label(model))
fig.suptitle(r"RankMe of the final stream vs the last MLP's write mass in the null space, "
             r"with the last-quarter ablation gap")
fig.tight_layout()
fig.subplots_adjust(wspace=0.85, right=0.90)

# %% [markdown]
# The same plot in matrix entropy, log(RankMe), which is the quantity the rank ledger is
# written in and the one that adds across independent directions. The ablation curve is then
# a difference of entropies -- log(RankMe ablated / RankMe baseline) -- not the log of the
# RankMe difference, so it reads as the log-factor the last quarter of blocks costs the
# final stream's effective rank.

# %%
LEGEND_H = [("0.35", "-", "matrix entropy"), ("k", "-", r"max $\rho$"),
            ("tab:red", "--", "change under ablation")]


def entropy_curve(model, leaf=(RANKME_HOOK, "acts_centered")):
    """log(RankMe) of the final stream, as stored by compute_metrics."""
    cfg = "nanochat_samples" if model == "nanochat-d12" else "block_representations_samples"
    ys, steps = get_ys(cfg, model, leaf, "matrix_entropy")
    return (None, None) if ys is None else (np.asarray(get_xs_tokens(model, steps), float),
                                            np.asarray(ys, float))


HGAPS = {m: ablation_gap(m, yvar="matrix_entropy") for m in GRID}
HFLOOR = gap_floor([g for _, g in HGAPS.values()])

fig, axes = plt.subplots(2, 3, figsize=(18, 7.5))
for i, (ax, model) in enumerate(zip(axes.ravel(), GRID)):
    xs, ys = entropy_curve(model)
    peak_h = np.log(PEAKS[model]["rankme"]) if model in PEAKS else None
    if ys is not None:
        ax.plot(xs, ys, color="0.35", lw=1.6)
        if peak_h is not None:
            ax.set_ylim(min(ys) - 0.05 * (peak_h - min(ys)), peak_h + 0.05 * (peak_h - min(ys)))
    if model in data and data[model]:
        per_step = data[model][blocks_of(data, model)[0]]
        color = COLORS.get(model, "C3")
        twin = ax.twinx()
        twin.plot(*series(per_step, lambda r, _e: r.max()), color=color, lw=1.6)
        twin.set(ylim=(0, 1.02), ylabel=r"max $\rho$")
        twin.axhline(chance(per_step[max(per_step)], 1.0), color=color, ls=":", lw=0.8)
        twin.grid(False)
        color_axis(twin, color)
    gx, gy = HGAPS[model]
    if gy is None:
        print(f"{model}: no {LAST_QUARTER[DEPTH[model]]} run — ablation gap left out")
    else:
        gap = ax.twinx()
        gap.spines["right"].set_position(("axes", 1.22))
        gap.plot(gx, gy, color="tab:red", ls="--", lw=1.6)
        gap.axhline(0, color="tab:red", ls=":", lw=0.8)
        gap.set_ylabel("entropy change (last quarter ablated)")
        gap.set_ylim(bottom=max(gap.get_ylim()[0], HFLOOR * gy.max()))
        gap.grid(False)
        color_axis(gap, "tab:red")
    if model in PEAKS:
        ax.axvline(PEAKS[model]["tokens"], color="k", lw=0.9, alpha=0.6, zorder=0)
    if i == 0:
        ax.legend(handles=[Line2D([0], [0], color=c, ls=s, lw=1.6, label=lab)
                           for c, s, lab in LEGEND_H], fontsize=7, loc="upper left")
    ax.set(xscale="log", xlabel="tokens", ylabel="matrix entropy = log RankMe",
           title=get_model_label(model))
fig.suptitle(r"Matrix entropy of the final stream vs null-space alignment of the last MLP, "
             r"with the last-quarter ablation gap")
fig.tight_layout()
fig.subplots_adjust(wspace=0.85, right=0.90)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, model in zip(axes.ravel(), GRID):
    xs, ys = alpha_curve(model)
    if ys is not None:
        ax.plot(xs, ys, color="0.35", lw=1.6, label=r"$\alpha$")
    if model in data and data[model]:
        per_step = data[model][blocks_of(data, model)[0]]
        tx, ty = series(per_step, lambda r, _e: r.max())
        twin = ax.twinx()
        twin.plot(tx, ty, color=COLORS.get(model, "C3"), lw=1.6, label=r"max $\rho$")
        twin.set(ylim=(0, 1.02), ylabel=r"max $\rho$")
        twin.axhline(chance(per_step[max(per_step)], 1.0), color=COLORS.get(model, "C3"),
                     ls=":", lw=0.8)
    if model in PEAKS:
        ax.axvline(PEAKS[model]["tokens"], color="k", lw=0.9, alpha=0.6, zorder=0)
    ax.set(xscale="log", xlabel="tokens", ylabel="RankMe", title=get_model_label(model))
fig.suptitle(r"$\alpha$ of the final stream vs null-space alignment of the last MLP")
fig.tight_layout()
