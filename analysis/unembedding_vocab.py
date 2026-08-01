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
#     display_name: representation-geometry (3.14.0.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The unembedding and the stream trade places
#
# **The object of this notebook.** Late in pretraining the model's internal representation
# *compresses*: its variance packs into fewer and fewer directions. Over the same window the
# unembedding — the matrix that turns an internal vector into one score per vocabulary word —
# does the opposite: its own spectrum *flattens*. This notebook asks whether that is a
# coincidence or a trade, and what is being held fixed by it.
#
# **Two words used throughout.**
#
# * **Spectrum.** Any of these objects can be written as a list of directions, each with an
#   amount of "size" attached. Sorted largest-first, that list is the spectrum.
#
# * **alphaReQ.** How fast the spectrum falls off, measured as the slope on a log-log plot.
#   A *large* alphaReQ means the size is concentrated in a few directions. A *small* alphaReQ
#   means it is spread evenly. It is the same information as RankMe (the effective number of
#   directions in use), read as a slope instead of a count, and unlike RankMe it does not
#   change when the model gets wider.
#
# **The arithmetic that motivates the whole notebook.** A word's score is
# `score = unembedding x representation`. Write both in the unembedding's own set of
# directions. Along direction *j* the unembedding multiplies by an amount `s_j`, and the
# representation carries an amount of variance `e_j`. So the variance in the *scores* along
# that direction is `s_j^2 * e_j` — a product. On a log-log plot a product of two power laws
# has the sum of their slopes:
#
#     alphaReQ(scores)  =  alphaReQ(unembedding)  +  alphaReQ(representation)
#
# So if the two slopes mirror each other, their **sum** is flat, and a flat sum means the
# *output* is holding still while the two factors trade work. That is the claim this notebook
# tests, in four rows: the two factors, their sum, the directly measured output, and the sum
# laid over the measurement.
#
# Data: `data/results/alpha_conservation.pt` (`oneoff_scripts/alpha_conservation.py`), which
# stores the three spectra per checkpoint so the slope window can be changed here without
# re-running anything.

# %%
import os, sys
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import torch

from analysis.experiments_lib import _alpha, get_model_label

R = "data/results"
LEAF = "after_final_norm"
WINDOW = (11, 100)        # the bulk window used throughout the unembedding notebook
FAMILY = {"pythia": "Pythia", "OLMo": "OLMo-2", "nanochat": "nanochat"}
COLORS = dict(zip(["pythia-160m-deduped", "pythia-410m-deduped", "pythia-1b-deduped",
                   "pythia-6.9b-deduped", "OLMo-2-0425-1B", "OLMo-2-1124-7B", "nanochat-d12"],
                  plt.cm.tab10.colors))


def load(name):
    p = f"{R}/{name}"
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def caption(text):
    print("\n".join(__import__("textwrap").wrap(text, 100)))


def family_of(model):
    return next(v for k, v in FAMILY.items() if k in model)


AC = load("alpha_conservation.pt")


def alpha(v, k0=WINDOW[0], k1=WINDOW[1]):
    """The slope, or nan where it is undefined.

    nanochat's unembedding is zero-initialised, so at its first checkpoint every singular
    value is exactly 0 and there is no spectrum to fit a slope to. That is a fact about the
    model, not a failure, so those points are dropped from the curves rather than crashing
    them or being silently filled in.
    """
    a = np.asarray(v, float)
    if (a > 0).sum() < 4:
        return np.nan
    try:
        return _alpha(a, k0, k1)
    except np.linalg.LinAlgError:
        return np.nan


def series(model, k0=WINDOW[0], k1=WINDOW[1], leaf=LEAF):
    """(tokens, unembedding slope, representation slope, measured score slope) for one model.

    Checkpoints where any of the three is undefined are dropped from all three together, so
    the sum in section 2 always adds curves measured at the same checkpoints."""
    per = AC[model]
    steps = sorted(s for s in per if leaf in per[s])
    x = np.array([per[s]["tokens"] for s in steps], float)
    head = np.array([alpha(per[s]["head"].numpy(), k0, k1) for s in steps])
    stream = np.array([alpha(per[s][leaf]["stream"].numpy(), k0, k1) for s in steps])
    logit = np.array([alpha(per[s][leaf]["logit"].numpy(), k0, k1) for s in steps])
    ok = (x > 0) & np.isfinite(head) & np.isfinite(stream) & np.isfinite(logit)
    return x[ok], head[ok], stream[ok], logit[ok]


# %% [markdown]
# ## 1. Do the two factors move in opposite directions?
#
# **Question.** As training proceeds, does the representation's spectrum steepen at the same
# time as the unembedding's spectrum flattens?
#
# | | |
# |---|---|
# | measured | alphaReQ over spectrum ranks 11–100, i.e. ignoring the largest ten directions and the deep tail |
# | blue | the unembedding's own spectrum (its squared singular values). Weights only — no data involved |
# | red | the representation entering the unembedding, at `after_final_norm`, measured over 262k tokens of each model's own training data |
# | x-axis | tokens the model has been trained on, not optimiser steps, so families are comparable |
#
# A rising line means "packing into fewer directions". The two axes are separate because the
# two quantities sit at different levels; only the *shapes* are being compared.

# %%
if AC is None:
    print("[skipped] alpha_conservation.pt not on disk yet — job pending")
else:
    models = list(AC)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, models):
        x, head, stream, _ = series(model)
        ln = [ax.plot(x, head, lw=1.8, color="tab:blue", label="unembedding")[0]]
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ex = ax.twinx()
        ln.append(ex.plot(x, stream, lw=1.8, color="tab:red",
                          label="representation entering it")[0])
        ex.tick_params(axis="y", labelcolor="tab:red")
        ax.set(xscale="log", xlabel="tokens seen", title=get_model_label(model))
        ax.legend(handles=ln, fontsize=7)
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    axes[0, 0].set_ylabel(r"$\alpha_{ReQ}$(unembedding)", color="tab:blue")
    axes[1, 0].set_ylabel(r"$\alpha_{ReQ}$(unembedding)", color="tab:blue")
    fig.suptitle(r"$\alpha_{ReQ}$ of each factor over training (higher = variance in fewer directions)")
    plt.tight_layout(); plt.show()
    caption("alphaReQ over ranks 11-100 of the unembedding spectrum (blue, left axis) and of the "
            "after_final_norm stream (red, right axis). Axes are independent: the levels are not "
            "comparable, the shapes are.")

# %% [markdown]
# ## 2. Is their sum flat?
#
# **Question.** If the two factors are trading work, their slopes should add up to something
# that barely moves. Does it?
#
# | | |
# |---|---|
# | plotted | `alphaReQ(unembedding) + alphaReQ(representation)`, the two lines from row 1 added |
# | grouped | one panel per model family, so within-family agreement is visible |
# | dotted | each model's own sum averaged over the last half-decade of training, as a flat reference |
#
# A flat sum is the whole point: it means whatever each factor does separately, the pair of
# them together is holding something constant.

# %%
if AC is not None:
    fams = sorted({family_of(m) for m in AC})
    fig, axes = plt.subplots(1, len(fams), figsize=(5.5 * len(fams), 4), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for model in [m for m in AC if family_of(m) == fam]:
            x, head, stream, _ = series(model)
            ax.plot(x, head + stream, lw=1.8, color=COLORS.get(model),
                    label=get_model_label(model))
            late = x >= x.max() / 3
            ax.axhline((head + stream)[late].mean(), color=COLORS.get(model), ls=":", lw=0.9)
        ax.set(xscale="log", xlabel="tokens seen", title=fam,
               ylabel=r"$\alpha_{ReQ}(\mathrm{unembed}) + \alpha_{ReQ}(\mathrm{representation})$")
        ax.legend(fontsize=7)
    fig.suptitle(r"Sum of the two $\alpha_{ReQ}$ slopes")
    plt.tight_layout(); plt.show()
    caption("alphaReQ(unembedding) + alphaReQ(representation), ranks 11-100. Dotted: each model's "
            "mean over the last third of its run.")

# %% [markdown]
# ## 3. What is the logit covariance doing?
#
# **Question.** The arithmetic at the top says the sum of the two slopes should equal the
# alphaReQ of the **logit covariance** — the spread of the model's word scores across tokens.
# Measured directly rather than inferred, what does that slope do over training?
#
# | | |
# |---|---|
# | measured | **alphaReQ of the logit covariance.** For each token the logit vector is `z = W_U h`; take the covariance of `z` across 262k tokens and fit the same log-log slope over ranks 11-100 as the other two curves |
# | how | the logit covariance is `W Sigma W'`, which is vocabulary-sized. Its non-zero eigenvalues are the same as those of a d x d stand-in (`B' G B`, with `Sigma = B B'` and `G = W'W`), so nothing vocabulary-sized is ever formed |
# | note | this is a genuinely separate measurement from row 2 — nothing about it was assumed |

# %%
if AC is not None:
    fig, axes = plt.subplots(1, len(fams), figsize=(5.5 * len(fams), 4), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for model in [m for m in AC if family_of(m) == fam]:
            x, _, _, logit = series(model)
            ax.plot(x, logit, lw=1.8, color=COLORS.get(model), label=get_model_label(model))
        ax.set(xscale="log", xlabel="tokens seen", title=fam,
               ylabel=r"$\alpha_{ReQ}$(logit covariance)")
        ax.legend(fontsize=7)
    fig.suptitle("alphaReQ of the logit covariance — how concentrated the spread of word scores is")
    plt.tight_layout(); plt.show()
    caption("alphaReQ of the logit covariance W_U Sigma W_U^T, ranks 11-100, computed from its "
            "d x d surrogate. Independent of rows 1 and 2.")

# %% [markdown]
# ## 4. Does the sum predict the logit covariance?
#
# **Question.** Rows 2 and 3 should be the same curve if the reasoning holds. Are they?
#
# | | |
# |---|---|
# | solid | the directly measured alphaReQ of the logit covariance (row 3) |
# | dashed | the sum of the two factors (row 2) |
# | what a gap means | the arithmetic assumed the representation's directions line up with the unembedding's. Where the two curves separate, that assumption is failing, and the size of the gap measures by how much |
#
# This is the test. Agreement means the mirroring in row 1 is a bookkeeping identity: the two
# factors are two halves of one conserved quantity. Disagreement localises where the two
# objects are misaligned, which is a finding in its own right.

# %%
if AC is not None:
    fig, axes = plt.subplots(1, len(fams), figsize=(5.5 * len(fams), 4), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for model in [m for m in AC if family_of(m) == fam]:
            x, head, stream, logit = series(model)
            c = COLORS.get(model)
            ax.plot(x, logit, lw=1.8, color=c, label=f"{get_model_label(model)} — logit covariance")
            ax.plot(x, head + stream, lw=1.4, ls="--", color=c,
                    label=f"{get_model_label(model)} — predicted by the sum")
        ax.set(xscale="log", xlabel="tokens seen", title=fam, ylabel=r"$\alpha_{ReQ}$")
        ax.legend(fontsize=6)
    fig.suptitle("Measured alphaReQ of the logit covariance (solid) against the sum of the two "
                 "(dashed)")
    plt.tight_layout(); plt.show()
    caption("Solid: measured alphaReQ of the logit covariance. Dashed: the sum from row 2. Ranks "
            "11-100. Separation between them bounds the error in assuming the stream's eigenbasis "
            "aligns with the unembedding's.")

# %% [markdown]
# ## 5. Which side moves first?
#
# **Question.** The two factors trade work. Does one of them lead — does the unembedding
# flatten first and the representation follow, or the other way round?
#
# The checkpoints used above are spaced by factors, far too coarse to see a lead of a few
# thousand steps. This section uses a dense late-training run instead
# (`configs/alpha_conservation_fine.yaml`): 113 consecutive nanochat checkpoints, 48 Pythia-1B
# and 26 OLMo-2-1B, all in the second half of training, where row 2 shows the sum has already
# settled.
#
# | | |
# |---|---|
# | method | remove each curve's slow trend, then slide one against the other and ask at which offset they agree best |
# | reading | a positive best-offset means the unembedding's move shows up in the representation *later*, i.e. the unembedding leads |
# | honesty check | the grey band is what the same procedure returns on shuffled data. A peak inside the band means the data cannot answer the question |

# %%
FINE = load("alpha_conservation_fine.pt")


def detrend(y, frac=0.25):
    """Subtract a moving average, so only the wiggles are compared, not the overall drift."""
    w = max(3, int(frac * len(y)) | 1)
    pad = np.pad(y, w // 2, mode="edge")
    return y - np.convolve(pad, np.ones(w) / w, mode="valid")[:len(y)]


def lag_profile(a, b, max_lag):
    """Correlation of a with b shifted by each offset; positive lag = b follows a."""
    a, b = detrend(a), detrend(b)
    a, b = (a - a.mean()) / (a.std() or 1), (b - b.mean()) / (b.std() or 1)
    return np.array([np.corrcoef(a[max(0, -k):len(a) - max(0, k)],
                                 b[max(0, k):len(b) - max(0, -k)])[0, 1]
                     for k in range(-max_lag, max_lag + 1)])


if FINE is None:
    print("[skipped] alpha_conservation_fine.pt not on disk yet — collection job pending")
else:
    # Only the models with a dense grid collected. The file carries an entry for every model
    # in the registry; most are empty, and an empty one is not a result to plot.
    MIN_POINTS = 8                       # fewer than this and a lead/lag is not measurable
    dense = {m: sorted(s for s in FINE[m] if LEAF in FINE[m][s]) for m in FINE}
    models = [m for m, s in dense.items() if len(s) >= MIN_POINTS]
    for m, s in dense.items():
        if 0 < len(s) < MIN_POINTS:
            print(f"[dropped] {m}: only {len(s)} dense checkpoints, need {MIN_POINTS}")
    fig, axes = plt.subplots(2, len(models), figsize=(5.5 * len(models), 7.5), squeeze=False)
    rng = np.random.default_rng(0)
    for j, model in enumerate(models):
        per, steps = FINE[model], dense[model]
        x = np.array([per[s]["tokens"] for s in steps], float)
        head = np.array([alpha(per[s]["head"].numpy()) for s in steps])
        stream = np.array([alpha(per[s][LEAF]["stream"].numpy()) for s in steps])
        keep = np.isfinite(head) & np.isfinite(stream)
        x, head, stream = x[keep], head[keep], stream[keep]

        ax = axes[0, j]
        ax.plot(x, detrend(head), lw=1.4, color="tab:blue", label="unembedding")
        ax.plot(x, detrend(stream), lw=1.4, color="tab:red", label="representation")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set(xlabel="tokens seen", ylabel=r"$\alpha_{ReQ}$, trend removed",
               title=f"{get_model_label(model)} — {len(steps)} consecutive checkpoints")
        ax.legend(fontsize=7)

        ax = axes[1, j]
        max_lag = min(12, len(steps) // 4)
        lags = np.arange(-max_lag, max_lag + 1)
        prof = lag_profile(head, stream, max_lag)
        null = np.array([lag_profile(head, rng.permutation(stream), max_lag)
                         for _ in range(200)])
        ax.fill_between(lags, np.quantile(null, 0.025, axis=0),
                        np.quantile(null, 0.975, axis=0), color="0.8",
                        label="what shuffled data gives (95%)")
        ax.plot(lags, prof, lw=1.8, color="tab:purple")
        ax.axvline(lags[np.argmax(np.abs(prof))], color="tab:purple", ls="--", lw=1,
                   label=f"best offset = {lags[np.argmax(np.abs(prof))]:+d} checkpoints")
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set(xlabel="offset (checkpoints); positive = the representation follows",
               ylabel="cross-correlation")
        ax.legend(fontsize=7)
    fig.suptitle("Lead/lag between the two slopes, late-training checkpoints")
    plt.tight_layout(); plt.show()
    caption("Top: both alphaReQ series over consecutive late-training checkpoints, moving-average "
            "trend removed. Bottom: cross-correlation against lag; grey band is the 95% interval "
            "from 200 shuffles. A peak inside the band is not resolvable.")
