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

# %% [markdown]
# GRID = ["pythia-410m-deduped", "pythia-1b-deduped", "pythia-6.9b-deduped",
#         "nanochat-d12", "OLMo-2-0425-1B", "OLMo-2-1124-7B"]

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
