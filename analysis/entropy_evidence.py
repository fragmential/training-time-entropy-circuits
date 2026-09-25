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
# # Entropy neurons: one figure per claim
#
# Every finding from the entropy-neuron thread that is not already in
# `entropy_neurons.ipynb` (which covers the rho onset, population size, the identification
# plane, the centring check, the quantile fan and the RankMe-vs-rho grid). Each section
# states a claim and plots the evidence for it. Sections whose run has not landed yet print
# what is missing instead of failing.

# %%
import os, sys, glob
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import torch

from analysis.experiments_lib import get_model_label
from utils.model_registry import get_token_count


def toks(model, steps):
    """Steps mean nothing across families; every x-axis in this notebook is tokens seen."""
    return np.array([get_token_count(model, int(s)) for s in steps], dtype=float)

R = "data/results"
THRESH = 0.5     # "most of the write lands in the null space", as section 2 uses
COLORS = dict(zip(["pythia-160m-deduped", "pythia-410m-deduped", "pythia-1b-deduped",
                   "pythia-6.9b-deduped", "OLMo-2-0425-1B", "OLMo-2-1124-7B", "nanochat-d12"],
                  plt.cm.tab10.colors))


def load(name):
    p = f"{R}/{name}"
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def npy(tag, model):
    p = f"{R}/{tag}/results_{model}.npy"
    return np.load(p, allow_pickle=True).item() if os.path.exists(p) else None


def missing(what):
    print(f"[skipped] {what}")


en = load("entropy_neurons.pt")        # read by several sections, so it loads once here


def k_index(e, frac=0.01):
    return e["ks"].index(k_of(e, frac))


def k_of(e, frac=0.01):
    """The null space's size for this model: rho is always measured at k = 0.01 d_model, which
    is 8 for nanochat-d12 and 41 for the 7B models, so it belongs in every label."""
    return max(1, round(frac * e["d_model"]))


def paper_set(e, norm_frac=0.05, n=20):
    """The n lowest-LogitVar neurons among the top norm_frac by weight norm.

    Stolfo et al., Confidence Regulation Neurons in Language Models (arXiv:2406.16254) detect entropy neurons by a high weight norm and a low
    LogitVar, then show those neurons write into the null space — rho is their evidence, not
    their filter. They pick a handful of visually identified outliers, hence the strictness
    here rather than a whole decile.
    """
    nm, lv = e["norm"].numpy(), e["logitvar"].numpy()
    hi = np.flatnonzero(nm > np.quantile(nm, 1 - norm_frac))
    return hi[np.argsort(lv[hi])[:n]]


# %% [markdown]
# ## 1. The null space is not a stable target until the transition
# Claim: neurons cannot align with `V0` before `V0` stops moving. `to_next` is the overlap of
# the bottom-k subspace at consecutive checkpoints, `to_final` its overlap with where it ends
# up (chance = k/d), `tail/med` the mean bottom-k singular value over the median.

# %%
st = load("nullspace_stability.pt")
if st is None:
    missing("nullspace_stability.pt")
else:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for m, r in st.items():
        x = np.array(r["tokens"], float)
        axes[0].plot(x[:-1], r["to_next"], marker=".", ms=3, color=COLORS.get(m), label=get_model_label(m))
        axes[1].plot(x, r["to_final"], marker=".", ms=3, color=COLORS.get(m),
                     label=get_model_label(m))
        axes[1].axhline(r["chance"], color=COLORS.get(m), ls=":", lw=0.7,
                        label="chance = k/d" if m == list(st)[0] else None)
        axes[2].plot(x, r["tail_ratio"], marker=".", ms=3, color=COLORS.get(m),
                     label=get_model_label(m))
    for ax, t in zip(axes, ("overlap with the next checkpoint",
                            "overlap with the final subspace (dotted = chance)",
                            "bottom-k singular values / median")):
        ax.set(xscale="log", xlabel="tokens", title=t)
        ax.legend(fontsize=6)
    fig.tight_layout()

# %% [markdown]
# ## 2. Which way of picking neurons finds null-space writers?
# Stolfo et al., Confidence Regulation Neurons in Language Models (arXiv:2406.16254) detect entropy neurons by a HIGH WEIGHT NORM and a LOW
# LOGITVAR, then show that those neurons write into the null space — rho is their evidence,
# not their filter — and they pick a handful of outliers rather than a broad quantile band.
# This compares three populations of the last block at
# the final checkpoint, by the median rho of the group over the layer's median rho — a ratio
# of 1 means the group is no more null-space-aligned than an average neuron:
#   20 highest rho                        selecting on the consequence (earlier sections)
#   top-decile norm AND bottom-decile      the norm/LogitVar criteria, loosely applied
#     LogitVar
#   top 5% norm, then 20 lowest LogitVar   the same criteria at outlier strictness. This is
#                                          the population used from here on.

# %%
if en is None:
    missing("entropy_neurons.pt")
else:
    names, groups = [], {"20 highest $\\rho$": [],
                         "top-decile weight norm AND bottom-decile LogitVar": [],
                         "top 5% weight norm, then the 20 lowest LogitVar": []}
    counts = []
    for m, blocks in en.items():
        b = max(blocks)
        e = blocks[b][max(blocks[b])]
        rho, lv, nm = e["rho"][k_index(e)].numpy(), e["logitvar"].numpy(), e["norm"].numpy()
        med = np.median(rho)
        sel = {"20 highest $\\rho$": np.argsort(-rho)[:20],
               "top-decile weight norm AND bottom-decile LogitVar":
                   np.flatnonzero((nm > np.quantile(nm, .9)) & (lv < np.quantile(lv, .1))),
               "top 5% weight norm, then the 20 lowest LogitVar": paper_set(e)}
        if len(sel["top-decile weight norm AND bottom-decile LogitVar"]) < 3:
            continue
        names.append(get_model_label(m))
        for k, idx in sel.items():
            groups[k].append(float(np.median(rho[idx]) / med))
        counts.append(int((rho[sel["top 5% weight norm, then the 20 lowest LogitVar"]] > 0.5).sum()))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    i_ = np.arange(len(names))
    for off, (k, v) in zip((-.27, 0, .27), groups.items()):
        axes[0].bar(i_ + off, v, .27, label=k)
    axes[0].axhline(1, color="k", lw=.8, ls=":")
    axes[0].set(xticks=i_, ylabel=r"median $\rho$ / layer median $\rho$",
                title="Which way of picking neurons finds null-space writers?\n"
                      r"1.0 = no better than an average neuron. $k = 0.01\,d_\mathrm{model}$")
    axes[1].bar(i_, counts)
    axes[1].set(xticks=i_, ylim=(0, 20), ylabel="neurons (out of 20)",
                title="Of the 20 neurons with the highest weight norm and the\n"
                      r"lowest LogitVar, how many have $\rho > 0.5$?")
    for ax in axes:
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    axes[0].legend(fontsize=7)
    fig.tight_layout()

# %% [markdown]
# ## 2a. "High weight norm and low LogitVar" is not one criterion, it is two
# Weight norm and LogitVar are different quantities on different scales, so there is no
# canonical way to combine them. Any rule that picks "the corner" of the scatter invents
# parameters. Three ways of doing it, and a check on whether the answer depends on which:
#   Pareto frontier   parameter-free: the neurons for which NO other neuron has both a
#                     higher weight norm and a lower LogitVar. This is what circling the
#                     outliers by eye approximates.
#   z-score sum       rank by z(norm) - z(log LogitVar), taking the top 20.
#   threshold rule    top q% by norm, then the 20 lowest LogitVar (what 2b uses).
# Left: the median rho each rule selects, as a multiple of the layer's median. Right: the
# threshold rule swept over q, to see whether the answer is an artefact of q = 5%.

# %%
def pareto_front(e):
    """Neurons not dominated on both axes: no other neuron has a higher weight norm AND a
    lower LogitVar. No thresholds, no free parameters."""
    nm, lv = e["norm"].numpy(), e["logitvar"].numpy()
    order = np.argsort(-nm)                      # walk from the largest norm downwards
    best, keep = np.inf, []
    for i in order:
        if lv[i] < best:                         # nothing bigger has a lower LogitVar
            keep.append(i)
            best = lv[i]
    return np.array(keep)


if en is None:
    missing("entropy_neurons.pt")
else:
    QS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    names, rules = [], {"Pareto frontier (no free parameters)": [],
                        "top 20 by z(norm) - z(log LogitVar)": [],
                        "top 5% by norm, then 20 lowest LogitVar": []}
    for m, blocks in en.items():
        b = max(blocks)
        e = blocks[b][max(blocks[b])]
        rho = e["rho"][k_index(e)].numpy()
        nm, lv = e["norm"].numpy(), np.log(e["logitvar"].numpy() + 1e-12)
        z = (nm - nm.mean()) / nm.std() - (lv - lv.mean()) / lv.std()
        med = np.median(rho)
        names.append(get_model_label(m))
        rules["Pareto frontier (no free parameters)"].append(np.median(rho[pareto_front(e)]) / med)
        rules["top 20 by z(norm) - z(log LogitVar)"].append(np.median(rho[np.argsort(-z)[:20]]) / med)
        rules["top 5% by norm, then 20 lowest LogitVar"].append(np.median(rho[paper_set(e)]) / med)
        axes[1].plot([100 * q for q in QS],
                     [np.median(rho[paper_set(e, norm_frac=q)]) / med for q in QS],
                     marker="o", ms=4, color=COLORS.get(m), label=get_model_label(m))
    i_ = np.arange(len(names))
    for off, (k, v) in zip((-.27, 0, .27), rules.items()):
        axes[0].bar(i_ + off, v, .27, label=k)
    axes[0].axhline(1, color="k", lw=.8, ls=":", label="no better than an average neuron")
    axes[0].set(xticks=i_, ylabel=r"median $\rho$ / layer median $\rho$",
                title="Does the answer depend on how the two criteria are combined?\n"
                      r"$\rho$ measured at $k = 0.01\,d_\mathrm{model}$")
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    axes[1].axhline(1, color="k", lw=.8, ls=":")
    axes[1].set(xscale="log", xlabel="q: kept top q% by weight norm before ranking on LogitVar",
                ylabel=r"median $\rho$ / layer median $\rho$",
                title="Is the result an artefact of the threshold?")
    for ax in axes:
        ax.legend(fontsize=6)
    fig.tight_layout()

# %% [markdown]
# ## 2b. Do high-weight-norm, low-LogitVar neurons become null-space writers?
# The 20 neurons with the highest weight norm and the lowest LogitVar are identified once, at
# the final checkpoint, then traced backwards through training.
# Solid: the median rho of those neurons. Dotted: the median rho of their whole layer, so the
# gap is what is specific to them rather than to the layer. Right: how many of the 20 clear
# rho > 0.5 at each checkpoint.

# %%
if en is None:
    missing("entropy_neurons.pt")
else:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for m, blocks in en.items():
        b = max(blocks)
        steps = sorted(blocks[b])
        idx = paper_set(blocks[b][steps[-1]])
        ki = k_index(blocks[b][steps[-1]])
        x = toks(m, steps)
        rhos = [blocks[b][s]["rho"][ki].numpy() for s in steps]
        axes[0].plot(x, [np.median(r[idx]) for r in rhos], color=COLORS.get(m), lw=1.6,
                     label=f"{get_model_label(m)}: the 20 high-norm, low-LogitVar neurons")
        axes[0].plot(x, [np.median(r) for r in rhos], color=COLORS.get(m), lw=1.0, ls=":",
                     label=f"{get_model_label(m)}: all neurons in the block")
        axes[1].plot(x, [int((r[idx] > 0.5).sum()) for r in rhos], color=COLORS.get(m), lw=1.6,
                     label=get_model_label(m))
    axes[0].set(xscale="log", xlabel="tokens",
                ylabel=r"median $\rho$  (share of the write in the null space)",
                title="Do high-weight-norm, low-LogitVar neurons\never become null-space writers?")
    axes[1].set(xscale="log", xlabel="tokens", ylim=(0, 20), ylabel="neurons (out of 20)",
                title=r"How many of those 20 have $\rho > 0.5$")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].legend(fontsize=6)
    fig.tight_layout()

# %% [markdown]
# ## 3. Ablation: dose-response against matched same-block controls
# Caveat: these ablations selected neurons by rho, not by weight norm and LogitVar, so they test
# "what do the null-space writers do", not "what do the paper's entropy neurons do".
# Claim: removing rho-ranked neurons moves the final stream, and matched random / weight-norm
# sets of the same size do not. Each panel is one measurement, x is how many neurons were
# mean-ablated, y the change against the untouched baseline at the final checkpoint.

# %%
ABL_MODEL, ABL_STEP = "OLMo-2-0425-1B", None
base = npy("entropy_ablate_blk-1_rho0", ABL_MODEL)
if base is None:
    missing("entropy_ablate_blk-1_rho0")
else:
    ABL_STEP = max(base)

    def readout(d):
        a = d["after_final_norm"]["acts_centered"]
        return dict(rankme=a["rankme"], alpha=a["alpha"],
                    trace=d["before_final_norm"]["acts_centered"]["trace"],
                    matrix_entropy=a["matrix_entropy"])

    b = readout(base[ABL_STEP])
    NS = (4, 16, 64, 256)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    for rank, c in (("rho", "C3"), ("wnorm", "C1"), ("random", "C0")):
        curves = {k: [] for k in b}
        ns = []
        for n in NS:
            r = npy(f"entropy_ablate_blk-1_{rank}{n}", ABL_MODEL)
            if r is None or ABL_STEP not in r:
                continue
            v = readout(r[ABL_STEP])
            ns.append(n)
            for k in b:
                curves[k].append(100 * (v[k] - b[k]) / abs(b[k]))
        for ax, k in zip(axes, ("rankme", "alpha", "trace", "matrix_entropy")):
            ax.plot(ns, curves[k], marker="o", ms=4, color=c, label=rank)
    for ax, k, unit in zip(axes, ("rankme", "alpha", "trace", "matrix_entropy"),
                           ("% change", "% change", "% change", "% change")):
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.set(xscale="log", xlabel="neurons ablated", ylabel=unit,
               title=f"{k.replace(chr(95),chr(32))} — {get_model_label(ABL_MODEL)}")
        ax.legend(fontsize=7)
    fig.tight_layout()

# %% [markdown]
# ## 4. The ablation effect over training
# Claim: the effect of removing the rho-ranked neurons is strongest before the end and decays.
# Solid = top-256 by rho, dashed = matched random 256, per checkpoint.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
any_data = False
for m in ("OLMo-2-0425-1B", "nanochat-d12", "pythia-1b-deduped", "pythia-410m-deduped"):
    bb = npy("entropy_ablate_blk-1_rho0", m)
    if bb is None:
        continue
    for tag, ls in (("rho256", "-"), ("random256", "--")):
        rr = npy(f"entropy_ablate_blk-1_{tag}", m)
        if rr is None:
            continue
        steps = sorted(set(bb) & set(rr))
        if len(steps) < 3:
            continue
        any_data = True
        for ax, key in zip(axes, ("rankme", "alpha")):
            y = [100 * (rr[s]["after_final_norm"]["acts_centered"][key]
                        - bb[s]["after_final_norm"]["acts_centered"][key])
                 / abs(bb[s]["after_final_norm"]["acts_centered"][key]) for s in steps]
            ax.plot(toks(m, steps), y, ls, color=COLORS.get(m), lw=1.3,
                    label=get_model_label(m) if (key == "rankme" and ls == "-") else None)
        y = [100 * (rr[s]["after_final_norm"]["acts_centered"]["matrix_entropy"]
                    - bb[s]["after_final_norm"]["acts_centered"]["matrix_entropy"])
             / abs(bb[s]["after_final_norm"]["acts_centered"]["matrix_entropy"]) for s in steps]
        axes[2].plot(toks(m, steps), y, ls, color=COLORS.get(m), lw=1.3)
if not any_data:
    missing("ablation trajectories")
for ax, t in zip(axes, ("RankMe change (%)", "alpha change (%)", "spectral entropy change (%)")):
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.set(xscale="log", xlabel="tokens", title=t)
    ax.legend(fontsize=6)
from matplotlib.lines import Line2D
axes[2].add_artist(axes[2].legend(
    handles=[Line2D([0], [0], color="k", ls="-", label="top-256 by rho"),
             Line2D([0], [0], color="k", ls="--", label="random 256 (control)")],
    fontsize=6, loc="lower left"))
fig.tight_layout()

# %% [markdown]
# ## 5. Taking the strongest null-space writers from the whole last quarter
# "Pooled" means the ranking is not done inside one block: the neurons of the final quarter
# of the model (blk-1..-4 for OLMo, blk-1..-3 for nanochat) are put in one list, ranked by rho
# together, and the global top-n taken — so n is a total across those blocks, and a block
# contributes only as many neurons as it has strong ones. The point is to ask whether the
# effect is bigger when you are free to take the strongest writers wherever they sit in the
# last quarter, instead of being confined to the final block.
# Solid = ranked by rho, dashed = a random set of the same total size over the same blocks.

# %%
QTAGS = {"OLMo-2-0425-1B": "blk-1..-4", "nanochat-d12": "blk-1..-3"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for m, q in QTAGS.items():
    bb = npy("entropy_ablate_blk-1_rho0", m)
    if bb is None:
        continue
    for rank, ls in (("rho", "-"), ("random", "--")):
        ns, ys = [], {k: [] for k in ("rankme", "alpha", "matrix_entropy")}
        for n in (64, 256, 1024):
            rr = npy(f"entropy_ablate_{q}_{rank}{n}", m)
            if rr is None:
                continue
            steps = sorted(set(bb) & set(rr))
            if not steps:
                continue
            s = steps[-1]
            ns.append(n)
            for k in ("rankme", "alpha"):
                a, c = rr[s]["after_final_norm"]["acts_centered"], bb[s]["after_final_norm"]["acts_centered"]
                ys[k].append(100 * (a[k] - c[k]) / abs(c[k]))
            a, c2 = rr[s]["after_final_norm"]["acts_centered"], bb[s]["after_final_norm"]["acts_centered"]
            ys["matrix_entropy"].append(100 * (a["matrix_entropy"] - c2["matrix_entropy"])
                                        / abs(c2["matrix_entropy"]))
        for ax, k in zip(axes, ("rankme", "alpha", "matrix_entropy")):
            if ns:
                ax.plot(ns, ys[k], ls, marker="o", ms=4, color=COLORS.get(m),
                        label=f"{get_model_label(m)} {rank}" if k == "rankme" else None)
for ax, t in zip(axes, ("RankMe change (%)", "alpha change (%)", "spectral entropy change (%)")):
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.set(xscale="log", xlabel="neurons ablated (pooled)", title=t)
axes[0].legend(fontsize=7)
fig.tight_layout()

# %% [markdown]
# ## 6. The direction the last blocks add is null-space aligned and large
# W_U is decomposed by SVD; its right singular vectors are directions in residual-stream
# space. The BOTTOM-k are the k with the smallest singular values — the directions the
# unembedding barely reads, i.e. the effective null space. The TOP-k are the k largest — the
# directions it reads hardest. k = 0.01 * d_model throughout (20 for OLMo-2-1B, d = 2048), the
# paper's choice. "Share" is the fraction of the unit direction's norm inside that subspace;
# a random direction gives sqrt(k/d) = 0.099, the dotted line.
# Claim: `gain` (top eigenvector of the covariance difference across the block) and `afn` (top
# eigenvector of the final stream) are nearly the same direction, carry ~10% of the stream's
# variance, and sit far outside chance in the unembedding's bottom-k subspace. `ratio`, the
# generalized eigenvector, is a low-variance artefact.

# %%
gd = load("grown_direction_neurons.pt")
if gd is None:
    missing("grown_direction_neurons.pt")
else:
    for model, per in gd.items():
        steps = sorted(per)
        x = toks(model, steps)
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
        for b in sorted({k for k in per[steps[0]] if isinstance(k, int)}, reverse=True)[:1]:
            for name, c in (("gain", "C3"), ("afn", "C0"), ("ratio", "0.6")):
                axes[0].plot(x, [per[s][b][name]["null_share"] for s in steps], marker="o", ms=4, color=c, label=name)
                axes[1].plot(x, [per[s][b][name]["top_share"] for s in steps], marker="o", ms=4, color=c)
                axes[2].plot(steps, [100 * per[s][b][name]["share_out"] for s in steps], marker="o", ms=4, color=c)
                axes[3].plot(x, [per[s][b][name]["split_half_cos"] for s in steps], marker="o", ms=4, color=c)
            axes[0].plot(x, [per[s]["chance"] for s in steps], "k:", lw=1, label="chance")
            axes[3].plot(x, [per[s][b]["cos_between"]["gain-afn"] for s in steps], "g-.", label="cos(gain, afn)")
        for ax, t in zip(axes, ("share in the null space", "share in the top-k subspace",
                                "% of the stream's variance", "stability / cos(gain,afn)")):
            ax.set(xscale="log", xlabel="tokens", title=f"{t}")
        axes[0].legend(fontsize=7); axes[3].legend(fontsize=7)
        fig.suptitle(f"{get_model_label(model)} — last block", y=1.02)
        fig.tight_layout()

# %% [markdown]
# ## 7. The direction is spread over tokens, not concentrated on a few
# Claim: it is not a sink — the participation ratio is a third to a half of all tokens. At the
# positions carrying `gain`, the model's entropy is far above its own average (hedging); for
# `afn` that stops being true after ~1e11 tokens, when the direction switches to markup and
# digits.

# %%
if gd is None:
    missing("grown_direction_neurons.pt")
else:
    for model, per in gd.items():
        steps = sorted(per)
        x = toks(model, steps)
        b = max(k for k in per[steps[0]] if isinstance(k, int))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for name, c in (("gain", "C3"), ("afn", "C0")):
            pr = [100 * per[s][b][name]["concentration"]["participation_ratio"]
                  / per[s][b][name]["concentration"]["n_tokens"] for s in steps]
            axes[0].plot(steps, pr, marker="o", ms=4, color=c, label=name)
            axes[1].plot(x, [per[s][b][name]["concentration"]["share_top_0.01"] for s in steps],
                         marker="o", ms=4, color=c)

        for ax, t in zip(axes, ("participation ratio (% of tokens)",
                                "energy share of the top 1% of positions")):
            ax.set(xscale="log", xlabel="tokens", title=t)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{get_model_label(model)} — last block", y=1.02)
        fig.tight_layout()

# %% [markdown]
# ## 8. Which neurons compress the representation
# Spectral entropy of the residual stream, H = -sum_i p_i log p_i over p_i = lambda_i / sum
# lambda of the centred covariance -- the representation's own entropy. Ablating a neuron of
# the LAST block changes the pre-final-norm stream by exactly a_tilde_i w_out and nothing
# else, so the ablated covariance is a rank-2 update of the baseline and every neuron's EXACT
# dH follows from one covariance and one cross-covariance. No linearisation, no forward pass
# per neuron. dH > 0 means removing the neuron RAISES entropy, i.e. it was compressing.
# Sample: 256 windows x 512 = 131k tokens per checkpoint, stored as n with each entry.
#
# This cell ranks by the CHANGE in dH across the run -- what shifted between 1e10 and 1e12.
# Section 8b ranks by dH at the end, i.e. which neurons actually compress most.

# %%
se = load("spectral_effect.pt")
det = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def spectral_panels(model, per, rank_by, subtitle):
    """The six-panel view of per-neuron compression, for a given way of picking the ten.

    rank_by(dh, d_dh) -> indices, so the only difference between the cells is the selection.
    """
    b = max(k for k in per if isinstance(k, int))
    steps = sorted(per[b])
    x = np.array([per[b][s]["tokens"] for s in steps], float)
    dh = [det(per[b][s]["d_spectral_entropy"]) for s in steps]
    rho = [det(per[b][s]["rho"]) for s in steps]
    wn = [det(per[b][s]["w_norm"]) for s in steps]
    d_dh, d_rho = dh[-1] - dh[0], rho[-1] - rho[0]
    top = rank_by(dh, d_dh)

    fig, axg = plt.subplots(2, 3, figsize=(17, 10))
    ax = axg.ravel()
    for k, i_ in enumerate(top):
        c = plt.cm.tab10(k % 10)
        lab = f"n{i_}   $\\Delta H$={dh[-1][i_]:+.3f}"
        ax[0].plot(x, [v[i_] for v in dh], lw=1.3, color=c, label=lab)
        ax[1].plot(x, [v[i_] for v in rho], lw=1.3, color=c, label=lab)
        ax[2].plot(x, [v[i_] for v in wn], lw=1.3, color=c)
    ax[1].plot(x, [np.median(v) for v in rho], "k:", lw=1.6, label="layer median")
    ax[2].plot(x, [np.median(v) for v in wn], "k:", lw=1.6, label="layer median")
    ax[0].axhline(0, color="k", lw=.8, ls=":")
    ax[0].set(xscale="log", xlabel="tokens", ylabel=r"$\Delta H$ (nats)", title=subtitle)
    ax[1].set(xscale="log", xlabel="tokens", ylabel=r"$\rho_i$",
              title="the same neurons' null-space share")
    ax[2].set(xscale="log", xlabel="tokens", ylabel=r"$\|w_{out}\|$",
              title="the same neurons' weight norm")

    ax[3].plot(x, [100 * (v > 0).mean() for v in dh], marker="o", ms=4, color="C0",
               label="% of neurons that compress")
    ax[3].set(xscale="log", xlabel="tokens", ylabel="% of the layer", ylim=(-2, 102),
              title="how much of the layer compresses, and how strongly")
    tw = ax[3].twinx()
    tw.plot(x, [v.max() for v in dh], marker="s", ms=4, color="C3", label=r"largest $\Delta H$")
    tw.set_ylabel(r"largest $\Delta H$ (nats)", color="C3")
    ax[3].legend(fontsize=7, loc="upper left"); tw.legend(fontsize=7, loc="lower right")

    for a, (xx, yy, xl, yl, ti) in zip(
            ax[4:], ((rho[-1], dh[-1], r"$\rho$", r"$\Delta H$ (nats)",
                      f"every neuron at {x[-1]:.0e} tokens"),
                     (d_rho, d_dh, r"$\Delta\rho$", r"$\Delta(\Delta H)$ (nats)",
                      f"change, {x[0]:.0e} to {x[-1]:.0e} tokens"))):
        sc = a.scatter(xx, yy, s=5, c=rho[-1], cmap="viridis", vmin=0, vmax=1, alpha=.55,
                       edgecolors="none")
        a.scatter(xx[top], yy[top], s=55, facecolors="none", edgecolors="crimson",
                  label="the 10 highlighted")
        a.axhline(0, color="k", lw=.8, ls=":")
        a.set(xlabel=xl, ylabel=yl, title=ti)
        a.legend(fontsize=7)
        plt.colorbar(sc, ax=a, label=r"final $\rho$")

    h, l = ax[0].get_legend_handles_labels()
    h2, l2 = ax[1].get_legend_handles_labels()
    fig.legend(h + h2[len(h):], l + l2[len(l):], fontsize=7, ncol=1,
               loc="center left", bbox_to_anchor=(0.005, 0.5), frameon=False)
    fig.suptitle(f"{get_model_label(model)} — blk{b}", y=1.02)
    fig.tight_layout(rect=(0.115, 0, 1, 1))


if se is None:
    missing("spectral_effect.pt")
else:
    for model, per in se.items():
        spectral_panels(model, per, lambda dh, d: d.argsort()[::-1][:10],
                        r"the 10 biggest gainers in compression")

# %% [markdown]
# ## 8b. The neurons that actually compress the most
# Same six panels, ranked by dH at the FINAL checkpoint rather than by how much it changed.
# Ranking by change is dominated by neurons that were strongly anti-compressing early and
# merely relaxed to zero -- a large gain that ends at no compression at all. This picks the
# neurons that are compressing at the end of training.

# %%
if se is None:
    missing("spectral_effect.pt")
else:
    for model, per in se.items():
        spectral_panels(model, per, lambda dh, d: dh[-1].argsort()[::-1][:10],
                        r"the 10 most compressing at the end")

# %% [markdown]
# ## 9. The frequency handover, the cone, and neural collapse
# Claim (to test): the unembedding stops carrying the frequency bias while the last MLP picks
# it up; the cone does not grow; the neural-collapse statistics do not move toward collapse.

# %%
cg = load("collapse_geometry.pt") or {}
# The heavy run carries NC2/NC3/UNC3/NC4/GNC2 at fewer checkpoints; the light one carries NC1,
# the cone and the frequency fit at roughly twice as many. Merge per (model, step) so every
# panel draws whatever actually measured it, at that metric's own time resolution.
for _f in sorted(glob.glob(f"{R}/collapse_geometry_heavy*.pt")):
    for _m, _steps in torch.load(_f, weights_only=False).items():
        for _s, _v in _steps.items():
            cg.setdefault(_m, {}).setdefault(_s, {}).update(_v)
if not cg:
    missing("collapse_geometry.pt — rerun with the frequency-ranked class selection")
else:
    from oneoff_scripts.plot_collapse_geometry import PANELS
    models = [m for m in cg if cg[m]]
    # Two figures: the geometry of the stream and the unembedding, then neural collapse.
    # Eight columns abreast was unreadable.
    TOP = ("frequency", "cone", "frequency in W_U", "equinorm / equiangular")
    # A coefficient of variation next to a cosine or an agreement rate flattens both; these two
    # get the right-hand axis of their panel rather than sharing its range.
    RIGHT_AXIS = {"afn_unc3_dual_cv", "afn_nc2_equiangle_cv"}
    for group in ([p for p in PANELS if p[0] in TOP], [p for p in PANELS if p[0] not in TOP]):
        fig, axes = plt.subplots(len(models), len(group), squeeze=False,
                                 figsize=(5.0 * len(group), 3.6 * len(models)))
        for row, model in enumerate(models):
            steps = sorted(cg[model])
            xs = np.array([cg[model][s]["tokens"] for s in steps], float)
            for col, (title, keys) in enumerate(group):
                ax, tw = axes[row, col], None
                for key, label in keys:
                    ys = np.array([cg[model][s].get(key, np.nan) for s in steps], float)
                    # each metric on its own checkpoints: the heavy neural-collapse run covers
                    # fewer than the light one, and keeping the gaps as NaN breaks every line
                    # into lone markers wherever a heavy step sits between two light-only ones
                    ok = np.isfinite(ys)
                    if not ok.any():
                        continue
                    if key in RIGHT_AXIS:
                        tw = ax.twinx() if tw is None else tw
                        tw.plot(xs[ok], ys[ok], marker=".", ms=3, lw=1.3, ls="--", color="C6",
                                label=label + " (right axis)")
                        tw.tick_params(axis="y", colors="C6")
                    else:
                        ax.plot(xs[ok], ys[ok], marker=".", ms=3, lw=1.3, label=label)
                ax.set(xscale="log", xlabel="tokens", title=title if row == 0 else "")
                h, l = ax.get_legend_handles_labels()
                if tw is not None:
                    h, l = [h + x for h, x in zip((h, l), tw.get_legend_handles_labels())]
                ax.legend(h, l, fontsize=6)
                if col == 0:
                    ax.set_ylabel(get_model_label(model))
        fig.tight_layout()


# %% [markdown]
# ## 9b. Self-duality against the rank the model actually carries
#
# **Question.** NC3 self-duality says each class mean lines up with its own unembedding row.
# Does it move with the representation's effective rank, or independently of it?
#
# | | |
# |---|---|
# | NC3 | mean cos(unembedding row, class mean), one per class, averaged. 1 = perfect self-duality |
# | UNC3 | Wu & Papyan's uniform-duality variant: the CoV of those cosines, so 0 = every class equally dual. A mean can rise while the spread does too |
# | RankMe | exp(matrix entropy) at `after_final_norm` from the main sweep (`block_representations_samples`, `nanochat_samples` for nanochat) — the same curve every other section uses |
# | rows | NC3 with UNC3 (top), NC3 with RankMe (bottom). Three quantities on three scales, so each row pairs NC3 with one of them on a right-hand axis |
# | x-axis | tokens seen |
#
# Data: `oneoff_scripts/collapse_geometry.py --heavy_metrics true`, merged over its per-model
# output files.

# %%
from analysis.experiments_lib import get_ys

CG6 = ["pythia-410m-deduped", "pythia-1b-deduped", "pythia-6.9b-deduped",
       "OLMo-2-0425-1B", "OLMo-2-1124-7B", "nanochat-d12"]


def series(model, key):
    """(tokens, values) for one collapse metric, on whichever checkpoints measured it."""
    st = [s for s in sorted(cg.get(model, {})) if np.isfinite(cg[model][s].get(key, np.nan))]
    return (np.array([cg[model][s]["tokens"] for s in st], float),
            np.array([cg[model][s][key] for s in st], float))


def rankme(model):
    src = "nanochat_samples" if model == "nanochat-d12" else "block_representations_samples"
    ys, steps = get_ys(src, model, ("after_final_norm", "acts_centered"), "rankme")
    return (toks(model, steps), np.asarray(ys, float)) if ys is not None else (None, None)


fig, axes = plt.subplots(2, len(CG6), figsize=(5.0 * len(CG6), 8.4), squeeze=False)
for j, m in enumerate(CG6):
    x3, y3 = series(m, "afn_nc3_dual")
    for row, (lab, col, get) in enumerate(
            ((r"UNC3: CoV of those cosines", "C6", lambda: series(m, "afn_unc3_dual_cv")),
             ("RankMe at after_final_norm", "C1", lambda: rankme(m)))):
        ax = axes[row, j]
        if len(x3):
            ax.plot(x3, y3, marker=".", ms=3, lw=1.3, color="C0",
                    label="NC3: mean cos(unembedding row, class mean)")
        xo, yo = get()
        if xo is not None and len(xo):
            tw = ax.twinx()
            tw.plot(xo, yo, marker=".", ms=3, lw=1.3, ls="--", color=col, label=lab + " (right axis)")
            tw.tick_params(axis="y", colors=col)
            if row == 1:
                tw.set_yscale("log")
            h, l = [a + b for a, b in zip(ax.get_legend_handles_labels(),
                                          tw.get_legend_handles_labels())]
            ax.legend(h, l, fontsize=6)
        elif len(x3):
            ax.legend(fontsize=6)
        if not len(x3) and xo is None:
            ax.text(.5, .5, "not on disk yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set(xscale="log", xlabel="tokens", ylabel="NC3 self-duality",
               title=f"{get_model_label(m)} — NC3 with {'UNC3' if row == 0 else 'RankMe'}")
        ax.title.set_fontsize(9)
fig.tight_layout()

# %% [markdown]
# ## 10. What the neurons actually write, in size
# Rows: (1) neurons sorted by write magnitude; (2) write magnitude against rho; (3) the
# activation ALONE, RMS(a_i); (4) the weight norm ALONE -- rows 3 and 4 split the write into
# its two factors, so it is visible which one carries the association; (5) the same
# correlation for all three quantities at every checkpoint.
# Sample size: the activation RMS is measured on the token count stored with each entry
# (32k tokens per checkpoint, 64 windows of 512, from the model's own training mix -- pile
# for Pythia, the OLMo mix for OLMo, fineweb-edu for nanochat), written by
# oneoff_scripts/entropy_neuron_activations.py. It is printed in every panel title.
# rho says WHERE a neuron writes, not how much. This is how much: per-neuron write magnitude
# RMS(a_i) * ||w_out^(i)||, where a_i is the neuron's post-nonlinearity activation, the only
# data-dependent part.
#
# One dot is one MLP neuron of the last block, so a panel holds d_ff of them. Rows 1-4 are the
# FINAL checkpoint; row 5 is every checkpoint. Row 2 answers "do the null-space writers write
# more or less than everyone else" -- yellow to the right means more. The box in rows 2-4 gives
# Pearson r on log x (the axes span orders of magnitude) and Spearman on the raw values as the
# rank-based check that does not assume the log is the right transform.

# %%
# two layouts exist: the per-checkpoint sweep, and an earlier single-checkpoint run under the
# old filename. Both hold the same quantity at the final checkpoint, so accept either.
from scipy.stats import spearmanr

# Merge the two sources PER MODEL rather than picking one file: the per-checkpoint sweep
# lands one model at a time, so choosing it wholesale would silently drop every model it has
# not reached yet. The sweep wins where it exists, the single-checkpoint run fills the rest.
ws = {**(load("entropy_neuron_activations.pt") or {}), **(load("neuron_write_stats.pt") or {})}
ws = ws or None


def final_rms(entry):
    """(RMS, n_tokens) for the last block at the final checkpoint, either layout. The token
    count is stored per entry by oneoff_scripts/entropy_neuron_activations.py, so the sample
    size travels with the data rather than living only in a submission script."""
    if "rms" in entry:
        return entry["rms"].numpy(), int(entry["n"])
    b = max(k for k in entry if isinstance(k, int))
    e = entry[b][max(entry[b])]
    return e["rms"].numpy(), int(e["n"])


def front(*xy, c):
    """Reorder points so the colour scale runs low-to-high in draw order: scatter paints in
    array order, so the bright high-value points end up on top of the dim majority instead of
    buried under it. Returns x, y, c ready to splat into scatter."""
    z = np.argsort(c)
    return (*(a[z] for a in xy), c[z])


if ws is None or en is None:
    missing("no activation statistics on disk")
else:
    models = [m for m in ws if m in en]
    fig, axes = plt.subplots(5, len(models), figsize=(5.4 * len(models), 21), squeeze=False)
    for j, m in enumerate(models):
        rms, n_tok = final_rms(ws[m])
        b = max(en[m])
        step = max(en[m][b])
        e = en[m][b][step]
        rho, nm = e["rho"][k_index(e)].numpy(), e["norm"].numpy()
        n = min(len(rms), len(nm))
        act, w, r = rms[:n], nm[:n], rho[:n]
        mag = act * w
        lgmag = np.log10(np.maximum(mag, 1e-12))
        ttl = f"{get_model_label(m)} — step {step}, {n_tok/1000:.0f}k tokens"

        o = np.argsort(-mag)
        fx, fy, fc = front(np.arange(n), mag[o], c=r[o])
        sc = axes[0, j].scatter(fx, fy, c=fc, s=4, cmap="viridis", vmin=0, vmax=1)
        axes[0, j].set(yscale="log", xlabel="neuron rank by write magnitude",
                       ylabel=r"RMS($a_i$)$\cdot\|w_{out}\|$", title=ttl)
        plt.colorbar(sc, ax=axes[0, j], label=r"$\rho$")

        # rows 1-3: the write split into its two factors, then each factor alone
        panels = ((1, mag, r"write magnitude  RMS($a_i$)$\cdot\|w_{out}\|$", r, r"$\rho$"),
                  (2, act, r"RMS($a_i$)  (activation only, no weights)", lgmag,
                   r"$\log_{10}$ write magnitude"),
                  (3, w, r"$\|w_{out}^{(i)}\|$  (weights only, no activation)", lgmag,
                   r"$\log_{10}$ write magnitude"))
        for row, xx, xl, cc, cl in panels:
            a = axes[row, j]
            fx, fy, fc = front(xx, r, c=cc)
            scx = a.scatter(fx, fy, c=fc, s=4, cmap="viridis", alpha=.7, edgecolors="none",
                            **({"vmin": 0, "vmax": 1} if row == 1 else {}))
            pe = float(np.corrcoef(np.log10(np.maximum(xx, 1e-12)), r)[0, 1])
            sp = float(spearmanr(xx, r).statistic)
            a.text(0.03, 0.95, f"Pearson $r$(log x, $\\rho$) = {pe:.2f}\n"
                               f"Spearman $\\rho_s$ = {sp:.2f}",
                   transform=a.transAxes, va="top", fontsize=8,
                   bbox=dict(fc="white", ec="0.7", alpha=.85, boxstyle="round,pad=0.3"))
            a.set(xscale="log", xlabel=xl, ylabel=r"$\rho$", title=ttl)
            plt.colorbar(scx, ax=a, label=cl)

        # row 4: the same correlations at every checkpoint, once the sweep has landed
        blocks = [k for k in ws[m] if isinstance(k, int)]
        if blocks and en[m]:
            bb = max(blocks)
            steps = sorted(set(ws[m][bb]) & set(en[m][bb]))
            series_ = {"write magnitude": [], "activation only": [], "weights only": []}
            ntk = []
            for st_ in steps:
                aa, q = ws[m][bb][st_], en[m][bb][st_]
                rr = q["rho"][k_index(q)].numpy()
                a_, w_ = aa["rms"].numpy(), q["norm"].numpy()
                nn = min(len(rr), len(a_), len(w_))
                for key, xx in (("write magnitude", a_[:nn] * w_[:nn]),
                                ("activation only", a_[:nn]), ("weights only", w_[:nn])):
                    series_[key].append(float(np.corrcoef(
                        np.log10(np.maximum(xx, 1e-12)), rr[:nn])[0, 1]))
                ntk.append(int(aa["n"]))
            for key, ys in series_.items():
                axes[4, j].plot(toks(m, steps), ys, marker="o", ms=3, label=key)
            axes[4, j].axhline(0, color="k", lw=.8, ls=":")
            axes[4, j].set(xscale="log", xlabel="tokens", ylim=(-0.6, 1),
                           ylabel=r"Pearson $r$(log x, $\rho$)",
                           title=f"{get_model_label(m)} — {ntk[0]/1000:.0f}k tokens/checkpoint")
            axes[4, j].legend(fontsize=6)
        else:
            axes[4, j].text(.5, .5, "per-checkpoint activation sweep\nnot on disk yet",
                            ha="center", va="center", transform=axes[4, j].transAxes, fontsize=8)
            axes[4, j].set_axis_off()
    fig.tight_layout()

# %% [markdown]
# ## 11. Where the layer's compression actually lives
#
# **OLMo-2 panels below are superseded.** The ablation here is algebraic,
# $h - (\tilde a \odot \varepsilon)W^\top$, which assumes the MLP reaches the residual
# linearly. OLMo-2's `post_feedforward_layernorm` sits in between, so its numbers are wrong
# at every $k$ (at $10^{12}$ they invert the sign). Pythia and nanochat have no such norm and
# reproduce exactly under the corrected run. Section 12b supersedes the OLMo columns.
#
# **Question.** Ablating the last quarter of blocks multiplies RankMe several-fold. Which
# units inside the last block do that?
#
# | | |
# |---|---|
# | metric | RankMe = exp(matrix entropy), at `after_final_norm` |
# | ablation | mean-ablate the activation: $a_i \rightarrow \bar a_i$, so the unit writes a constant $\bar a_i w_i$ instead of varying per token. Its write vector $w_i$ is untouched |
# | sample | 65,536 tokens per checkpoint |
# | ranking | each unit's effect measured in isolation |
# | null space | bottom $k$ right singular directions of $W_U$, $k = 0.01\,d_\text{model}$ |
# | y-axis | RankMe / unablated RankMe, so 1.0 = no effect |
#
# **Caveat.** The ranking is scored at `before_final_norm`, which favours units firing on
# massive-activation tokens that the norm divides away. These curves are a lower bound; the
# after-norm gradient ranking supersedes them.

# %%
curves = {"MLP neurons": load("neuron_ablation_curve.pt"),
          "attention heads": load("head_ablation_curve.pt")}
if not any(curves.values()):
    missing("neuron_ablation_curve.pt / head_ablation_curve.pt")
else:
    models = sorted({m for c in curves.values() if c for m in c})
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4.4 * len(models)), squeeze=False)
    for row, model in enumerate(models):
        for col, (unit, c) in enumerate(curves.items()):
            ax = axes[row, col]
            if not c or model not in c:
                ax.set_axis_off()
                continue
            b = max(k for k in c[model] if isinstance(k, int))
            for st, shade in zip(sorted(c[model][b]), np.linspace(.35, 1, len(c[model][b]))):
                d = c[model][b][st]
                ks = sorted(d["effect"])
                base = d["base_afn"]["rankme"]
                lab = f"{d['tokens']:.0e} tok"
                ax.plot(ks, [d["effect"][k]["afn"]["rankme"] / base for k in ks], "-o", ms=3,
                        color="C3", alpha=shade, label=f"{lab}, by effect")
                ax.plot(ks, [d["random"][k]["afn"]["rankme"] / base for k in ks], "--s", ms=3,
                        color="C0", alpha=shade, label=f"{lab}, random")
            ax.axhline(1, color="k", lw=.8, ls=":")
            ax.set(xscale="log", xlabel=f"{unit} ablated",
                   ylabel="RankMe / unablated RankMe",
                   title=f"{get_model_label(model)} — {unit}")
            ax.legend(fontsize=6)
    fig.tight_layout()

# %% [markdown]
# ## 12. The same, ranked after the final norm
#
# **OLMo-2 panels below are superseded.** The ablation here is algebraic,
# $h - (\tilde a \odot \varepsilon)W^\top$, which assumes the MLP reaches the residual
# linearly. OLMo-2's `post_feedforward_layernorm` sits in between, so its numbers are wrong
# at every $k$ (at $10^{12}$ they invert the sign). Pythia and nanochat have no such norm and
# reproduce exactly under the corrected run. Section 12b supersedes the OLMo columns.
#
# **Question.** Section 11 ranked units by their effect measured *before* the final norm.
# Does ranking them *after* it find better neurons?
#
# | | |
# |---|---|
# | metric | RankMe = exp(matrix entropy), at `after_final_norm` |
# | ablation | mean-ablate the activation: $a_i \rightarrow \bar a_i$, write vector $w_i$ untouched |
# | rankings | four, compared: the after-norm gradient $\partial\text{RankMe}/\partial a_i$; the mass a unit puts into the null space, RMS($a_i$)$\cdot\|V_0^\top w_i\|$; $\rho$, the same projection as a fraction of a unit write direction, so blind to size; and a random control |
# | rows | MLP mean-ablation, MLP zero-ablation ($a_i \rightarrow 0$ — identical before the norm, different after), attention heads |
# | null space | bottom $k$ right singular directions of $W_U$, $k = 0.01\,d_\text{model}$ |
# | sample | 65,536 tokens per checkpoint |
# | y-axis | RankMe / unablated RankMe, so 1.0 = no effect |
#
# Ranking before the norm favours units that fire on massive-activation tokens, which the
# norm then divides away. Ranking after it does not.

# %%
ROWS = (("MLP neurons, mean-ablation", "neuron_ablation_curve_afngrad.pt"),
        ("MLP neurons, zero-ablation", "neuron_ablation_curve_afngrad_zero.pt"),
        ("attention heads, mean-ablation", "head_ablation_curve_afngrad.pt"))
# (key, label, colour, linestyle, opacity). The "_corr" twin of each ranking is the same
# colour, lighter: same score, but it grows the CORRELATED set -- after the first unit it
# prefers writes that reinforce the running sum, so the top-k is one clique rather than a
# spread of directions. (A "_set" variant doing the opposite, skipping already-covered
# directions, is also stored; it changed nothing for the gradient and hurt rho.)
LINES = (("effect", r"$\partial$RankMe/$\partial a_i$ after the norm", "C3", "-", 1.),
         ("effect_corr", "the same, growing the correlated set (writes that reinforce each other)",
          "C3", "-", .45),
         ("void_mass", r"mass into the null space: RMS($a_i$)$\cdot\|V_0^\top w_i\|$", "C1", "-", 1.),
         ("void_mass_corr", "the same, growing the correlated set (writes that reinforce each other)",
          "C1", "-", .45),
         ("rho", r"$\rho$: null-space share of the unit write direction", "C2", "-", 1.),
         ("rho_corr", "the same, growing the correlated set (writes that reinforce each other)",
          "C2", "-", .45),
         ("logitvar", "lowest LogitVar first (Stolfo et al.'s detection half)", "C4", "-", 1.),
         ("random", "random control", "C0", "--", 1.))



PLAIN = ("effect", "void_mass", "rho", "logitvar", "random")   # score units directly


def ablation_grid(rows, columns=None, show=PLAIN):
    """rows are (label, {model: {block: {step: curve}}}); columns are (model, checkpoint step),
    defaulting to every model at its last checkpoint. show picks which LINES keys are drawn --
    the set-aware twins ("_corr", "_set") are off by default, since how they weight a unit's
    score against its alignment with the chosen set is arbitrary and not yet justified."""
    loaded = rows
    if not any(d for _, d in loaded):
        return missing("no after-norm gradient curves on disk")
    columns = columns or [(m, None) for m in sorted({m for _, d in loaded for m in d})]
    fig, axes = plt.subplots(len(loaded), len(columns), squeeze=False,
                             figsize=(5.2 * len(columns), 4.4 * len(loaded)))
    for row, (rlab, data_) in enumerate(loaded):
        for col, (model, step) in enumerate(columns):
            ax, blk = axes[row, col], data_.get(model, {})
            d = blk and blk[max(blk)].get(step or max(blk[max(blk)]))
            if not d:
                ax.set_axis_off()
                ax.text(.5, .5, "not on disk yet", ha="center", va="center", fontsize=8)
                continue
            ks, base = sorted(d["effect"]), d["base_afn"]["rankme"]
            for key, lab, c, ls, al in LINES:
                if key in d and key in show:
                    ax.plot(ks, [d[key][k]["afn"]["rankme"] / base for k in ks], ls, marker="o",
                            ms=4, color=c, alpha=al, label=lab)
            ax.axhline(1, color="k", lw=.8, ls=":")
            ax.set(xscale="log", xlabel=f"{rlab.split(',')[0]} ablated (all = {max(ks)})",
                   ylabel="RankMe / unablated RankMe",
                   title=f"{get_model_label(model)} — {rlab}\n{d['tokens']:.1e} tokens seen, "
                         f"{d['n']:,} sampled, baseline {base:.0f}")
            ax.title.set_fontsize(9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .04, 1, 1))


ablation_grid([(lab, load(f) or {}) for lab, f in ROWS])

# %% [markdown]
# ## 12b. The same at four times the sample, six models, two training stages each
#
# **Question.** Section 12 measured three models at their final checkpoint on 65,536 tokens.
# Does it hold across the model range, and does it look different mid-training?
#
# Same metric, units and rankings as section 12. What changes:
#
# | | |
# |---|---|
# | ablation | a real forward pass: a pre-hook clamps the selected units' activations to their dataset mean (or 0) and the model runs normally, so post-norms, residual adds and every downstream layer are the network's own. Section 12 subtracted the write algebraically instead |
# | ranking | the after-norm gradient is taken THROUGH the network, so it carries the same post-norm the ablation does. Section 12's contracted against the raw write matrix |
# | agreement | Pythia and nanochat reproduce section 12 to the digit at every checkpoint — the two methods only disagree where a post-norm intervenes, i.e. on OLMo-2 |
# | sample | 262,144 tokens per checkpoint — where a direct RankMe reproduces the main sweep's value; 65,536 is biased low |
# | columns | each Pythia at $10^{10.5}$ and at its final $3\times10^{11}$ tokens, each OLMo-2 at $10^{12}$ and at its final $\approx4\times10^{12}$, nanochat-d12 at its final $3.7\times10^{9}$ |
# | attention | the unit is the head, not the (head, head_dim) column — section 12 capped at 100 columns and so never removed all heads |
#
# Data: `oneoff_scripts/neuron_ablation_curve.py` → `data/results/ablation_fwd/results_<model>.pt`.

# %%
MODES = (("MLP neurons, mean-ablation", "mlp_mean"),
         ("MLP neurons, zero-ablation", "mlp_zero"),
         ("attention heads, mean-ablation", "attn_mean"))
fwd = {os.path.basename(f)[8:-3]: torch.load(f, weights_only=False)
       for f in glob.glob(f"{R}/ablation_fwd/results_*.pt")}
rows = [(lab, {m: c[t] for m, c in fwd.items() if t in c}) for lab, t in MODES]
PY_M = ["pythia-410m-deduped", "pythia-1b-deduped", "pythia-6.9b-deduped"]

for group in ([(m, 143_000) for m in PY_M],
              [("OLMo-2-0425-1B", 1_900_000), ("OLMo-2-1124-7B", 920_000),
               ("nanochat-d12", 7_080)],
              [(m, 15_000) for m in PY_M],
              [("OLMo-2-0425-1B", 480_000), ("OLMo-2-1124-7B", 240_000)]):
    ablation_grid(rows, group)

# %% [markdown]
# ## 12c. Why the first-order ranking loses to a structural one at large k
#
# **Question.** In section 12b the null-space-mass ranking overtakes the after-norm gradient
# ranking beyond a few hundred units, most starkly for pythia-6.9b at $10^{10.5}$. Why would
# a ranking by the exact derivative of the plotted quantity lose to a proxy?
#
# | | |
# |---|---|
# | predicted | $\sum_{i \in \text{top-}k} \partial H/\partial \varepsilon_i$ — what the individual gradients say ablating the set should do, if effects simply added |
# | measured | $\log(\text{RankMe}_k / \text{RankMe}_0)$ — the change actually produced by ablating that set |
# | units | MLP neurons of the last block, ranked by the gradient, mean-ablation |
# | y-axis | change in spectral entropy (nats), log scale |
#
# The gradient is a *marginal* quantity: it says what removing one unit does with all the
# others still present. Null-space mass is *additive* — separate units write into separate
# directions, so their masses accumulate over a set. Where the predicted curve flattens while
# the measured one keeps climbing, the top-gradient units are redundant with each other and
# the marginal ranking has stopped discriminating.

# %%
fig, axes = plt.subplots(1, len(fwd), figsize=(4.2 * len(fwd), 4.2), squeeze=False)
for ax, (model, curves) in zip(axes[0], sorted(fwd.items())):
    blocks = curves["mlp_mean"]
    blk = blocks[max(blocks)]
    for st, c in zip(sorted(blk), ("C0", "C3")):
        d = blk[st]
        ks = sorted(d["effect"])
        solo = np.sort(d["scores"]["effect"].numpy())[::-1] if "scores" in d else None
        act = [np.log(d["effect"][k]["afn"]["rankme"] / d["base_afn"]["rankme"]) for k in ks]
        ax.plot(ks, act, "-o", ms=3, color=c, label=f"measured, {d['tokens']:.0e} tokens")
        if solo is not None:
            ax.plot(ks, [solo[:k].sum() for k in ks], "--s", ms=3, color=c,
                    label=f"predicted by summing gradients, {d['tokens']:.0e} tokens")
    ax.set(xscale="log", yscale="log", xlabel="MLP neurons ablated, ranked by gradient",
           ylabel="change in spectral entropy (nats)", title=get_model_label(model))
    ax.title.set_fontsize(9)
    ax.legend(fontsize=6)
fig.tight_layout()

# %% [markdown]
# ## 13. Is the direction the last blocks build the same one the early blocks made?
#
# **Question.** All the token attribution in this thread was computed on $v_1$, the dominant
# direction defined at block 3, and then tracked down the stack. That never asked what the
# END of the model actually carries. Take $v_f$, the top eigendirection of
# `before_final_norm`, and ask which tokens carry it and whether it is $v_1$ at all.
#
# | | |
# |---|---|
# | $v_1$ | top eigendirection of block 3's output, the direction every earlier attribution used |
# | $v_f$ | top eigendirection of `before_final_norm`, i.e. what the model ends up with |
# | classes | position 0; the first newline of a document; every other newline; all remaining tokens ("bulk") |
# | bars | mean absolute projection onto the direction, per class — how strongly that class loads on it |
# | var share | share of the direction's total variance contributed by each class, so a class can load hard yet be rare |
#
# Data: `oneoff_scripts/final_direction_id.py`, final checkpoint, raw samples from
# `configs/block_rogue_id.yaml`.

# %%
fd = load("final_direction_id.pt")
if not fd:
    missing("final_direction_id.pt")
else:
    CLS = ["pos0", "first_nl", "other_nl", "bulk"]
    NICE = ["position 0", "first newline", "other newlines", "bulk (everything else)"]
    fig, axes = plt.subplots(1, 2 * len(fd), figsize=(5.0 * 2 * len(fd), 4.4), squeeze=False)
    for j, (m, e) in enumerate(sorted(fd.items())):
        ax = axes[0, 2 * j]
        x = np.arange(len(CLS))
        ax.bar(x - .2, [float(e["abs_by_class"][c]) for c in CLS], .4,
               label=r"$v_f$: top direction at before_final_norm")
        ax.bar(x + .2, [float(e["v1_abs_by_class"][c]) for c in CLS], .4,
               label=r"$v_1$: top direction at block 3")
        ax.set(xticks=x, ylabel="mean |projection onto the direction|",
               title=f"{get_model_label(m)} — who loads on each direction\n"
                     f"cos($v_f$, $v_1$) = {float(e['cos_with_v1']):.3f}")
        ax.set_xticklabels(NICE, rotation=20, ha="right", fontsize=7)
        ax.title.set_fontsize(9)
        ax.legend(fontsize=6)

        ax = axes[0, 2 * j + 1]
        ax.bar(x, [float(e["var_share_by_class"][c]) for c in CLS], .5, color="C2")
        ax.set(xticks=x, ylabel=r"share of $v_f$ variance", ylim=(0, 1),
               title=f"{get_model_label(m)} — where $v_f$'s variance comes from")
        ax.set_xticklabels(NICE, rotation=20, ha="right", fontsize=7)
        ax.title.set_fontsize(9)
    fig.tight_layout()

# %% [markdown]
# ## 14. Does the deduped Pythia behave like the one the paper used?
#
# **Question.** Stolfo et al. (arXiv:2406.16254) work on the standard Pythia suite; every
# measurement here uses the *deduped* suite. If deduplication changed the null-space picture,
# the comparison to their result would not be like for like.
#
# | | |
# |---|---|
# | $\rho$ | $\|V_0^\top w_{out}\|/\|w_{out}\|$, the null-space share of a neuron's write direction |
# | null space | bottom $k$ right singular directions of $W_U$, $k = 0.01\,d_\text{model}$ |
# | curves | neurons of the last block sorted by $\rho$, descending, deduped against non-deduped |
# | checkpoint | final (step 143,000) for both suites |
#
# Data: `oneoff_scripts/entropy_neurons.py` run on both suites.

# %%
nd = load("entropy_neurons_nondedup.pt")
dd = en or {}                              # the deduped suite, loaded in the setup cell
if not nd:
    missing("entropy_neurons_nondedup.pt")
else:
    pairs = [(n, n + "-deduped") for n in sorted(nd)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 4.6), squeeze=False)
    for j, (plain, dedup) in enumerate(pairs):
        ax = axes[0, j]
        for src, name, lab, c in ((nd, plain, "standard Pythia (the paper's suite)", "C1"),
                                  (dd, dedup, "deduped Pythia (used everywhere here)", "C0")):
            if name not in src:
                continue
            blk = src[name]
            e = blk[max(blk)]
            e = e[max(e)] if isinstance(next(iter(e.values())), dict) else e
            r = np.sort(e["rho"][k_index(e)].numpy())[::-1]
            ax.plot(np.arange(1, len(r) + 1), r, lw=1.4, color=c,
                    label=f"{lab} — {(r > THRESH).sum()} neurons with " + r"$\rho>0.5$")
        ax.axhline(THRESH, color="k", lw=.8, ls=":")
        ax.set(xscale="log", xlabel="neurons of the last block, sorted by $\\rho$",
               ylabel=r"$\rho$", title=f"{plain} — deduped against standard")
        ax.title.set_fontsize(9)
        ax.legend(fontsize=6)
    fig.tight_layout()

# %% [markdown]
# ## 15. The stream's own biggest directions, seen through the unembedding
#
# **Question.** $\rho$ and LogitVar have only ever been asked of *neuron write directions*.
# What if the same questions are put to the representation's own leading eigendirections —
# is the largest thing the model carries at the end something that reaches the logits at all?
#
# | | |
# |---|---|
# | direction | eigenvector $u_j$ of the centred `after_final_norm` covariance, $j=1$ the largest |
# | $\rho(k)$ | $\|V_0(k)^\top u_j\|$, share of $u_j$ in the bottom $k$ right singular directions of $W_U$ — the null space, same definition the neuron sections use |
# | top share | $\|V_{\text{top}}(k)^\top u_j\|$, share in the top $k$ instead: what does reach the logits |
# | LogitVar | variance over the vocabulary of $u_j$'s row-normalised logit attribution |
# | var share | $\lambda_j / \sum\lambda$, how much of the stream this one direction is |
# | $\Delta$RankMe | RankMe(spectrum$[j:]$) − RankMe(spectrum$[j-1:]$): removing direction $j$ once every larger one is already gone. Positive = the stream is higher-rank without it |
# | sample | 262,144 tokens, the same capture the ablation sections use |
#
# Data: `oneoff_scripts/direction_identity.py`, final checkpoint of each model.

# %%
DI = {}
for _f in sorted(glob.glob(f"{R}/direction_identity_*.pt")):
    if "_mc_" in _f:
        continue
    for _m, _st in torch.load(_f, weights_only=False).items():
        DI.setdefault(_m, {}).update(_st)

METRICS = (("var_share", "variance share"), ("rho", r"$\rho$"),
           ("top_share", r"top-$k$ share"), ("logitvar", "LogitVar"),
           ("d_rankme", r"$\Delta$RankMe"))


def di_value(e, key, j=0):
    """One metric for direction j+1. rho and top_share are dicts keyed by k, so take the
    k = 0.01 d_model entry -- the size every neuron section uses."""
    v = e[key][e["ks"][1]] if key in ("rho", "top_share") else e[key]
    return float(v[j])


if not DI:
    missing("direction_identity_*.pt")
else:
    from IPython.display import Markdown, display
    models = sorted(DI)
    finals = {m: DI[m][max(DI[m])] for m in models}
    hdr = ["model", "tokens", "$k$"] + [lab for _, lab in METRICS]
    rows = [[get_model_label(m), f"{finals[m]['tokens']:.1e}", str(k_of(finals[m]))]
            + [f"{di_value(finals[m], k):.3g}" for k, _ in METRICS] for m in models]
    display(Markdown("\n".join(["| " + " | ".join(hdr) + " |",
                                "|" + "---|" * len(hdr)]
                               + ["| " + " | ".join(r) + " |" for r in rows])))

    # the same table as a LaTeX tabular, to paste straight into the write-up
    tex = ["\\begin{tabular}{l" + "r" * (len(hdr) - 1) + "}", "\\toprule",
           " & ".join(hdr) + r" \\", "\\midrule"]
    tex += [" & ".join(r) + r" \\" for r in rows]
    tex += ["\\bottomrule", "\\end{tabular}"]
    print("```latex\n" + "\n".join(tex) + "\n```")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.9 * len(METRICS), 4.6), squeeze=False)
    for ax, (key, lab) in zip(axes[0], METRICS):
        ax.bar(range(len(models)), [di_value(finals[m], key) for m in models],
               color=[COLORS.get(m, "C0") for m in models])
        ax.axhline(0, color="k", lw=.8)
        ax.set(xticks=range(len(models)), ylabel=lab,
               title="largest direction, final checkpoint")
        ax.set_xticklabels([f"{get_model_label(m)}\n$k$={k_of(finals[m])}" for m in models],
                           rotation=35, ha="right", fontsize=7)
        ax.title.set_fontsize(9)
    fig.tight_layout()

# %% [markdown]
# ## 15b. The same five quantities for the first five directions
#
# **Question.** Is the null-space character special to the single largest direction, or does it
# run down the spectrum?
#
# One panel per quantity, one bar per direction (largest first), grouped by model. $\Delta$RankMe
# is marginal: direction $j$ removed with every larger direction already gone, so the bars are
# a decomposition rather than five independent measurements.

# %%
if not DI:
    missing("direction_identity_*.pt")
else:
    N_DIR = 5
    GROUPS = (("nanochat-d12", "OLMo-2-0425-1B", "OLMo-2-1124-7B"),
              ("pythia-410m-deduped", "pythia-1b-deduped", "pythia-6.9b-deduped"))
    finals = {m: DI[m][max(DI[m])] for m in DI}
    for group in GROUPS:
        models = [m for m in group if m in finals]
        if not models:
            continue
        fig, axes = plt.subplots(len(METRICS), 1, squeeze=False,
                                 figsize=(3.1 * len(models), 3.0 * len(METRICS)))
        x = np.arange(len(models))
        for ax, (key, lab) in zip(axes[:, 0], METRICS):
            for j in range(N_DIR):
                ax.bar(x + (j - (N_DIR - 1) / 2) * .17,
                       [di_value(finals[m], key, j) for m in models], .17,
                       color=plt.cm.viridis(j / N_DIR), label=f"direction {j + 1}")
            ax.axhline(0, color="k", lw=.8)
            ax.set(xticks=x, ylabel=lab)
            ax.set_xticklabels([f"{get_model_label(m)}\n$k$={k_of(finals[m])}" for m in models],
                               fontsize=9)
            ax.legend(fontsize=7, ncol=N_DIR)
        fig.suptitle("the five largest directions of after_final_norm, final checkpoint",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .97))

# %% [markdown]
# ## 15c. The same five quantities over training
#
# **Question.** Sections 15 and 15b are the final checkpoint only. Does the largest direction
# start out null-space aligned, or become so — and if it becomes so, is there an onset?
#
# Same quantities and definitions as section 15, measured at 8 log-spaced checkpoints per
# model. One panel per quantity, one line per model, x-axis tokens seen. The RankMe curve from
# the main sweep is drawn underneath for reference, since the claim to test is whether the
# null-space alignment moves with the effective rank rather than independently of it.
#
# Data: `oneoff_scripts/direction_identity.py --max_checkpoints 8`, one file per model.

# %%
MC = {}
for _f in sorted(glob.glob(f"{R}/direction_identity_mc_*.pt")):
    for _m, _st in torch.load(_f, weights_only=False).items():
        MC.setdefault(_m, {}).update(_st)

if not MC:
    missing("direction_identity_mc_*.pt")
else:
    panels = list(METRICS) + [("rankme", "RankMe")]
    models = sorted(MC)
    for key, lab in panels:                   # one figure per quantity, models 2 x 3 inside
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for ax, m in zip(axes.ravel(), models):
            st = sorted(MC[m])
            xs = np.array([MC[m][t]["tokens"] for t in st], float)
            ys = np.array([MC[m][t]["rankme"] if key == "rankme" else di_value(MC[m][t], key)
                           for t in st], float)
            ax.plot(xs, ys, marker="o", ms=4, lw=1.5, color=COLORS.get(m, "C0"))
            ax.set(xscale="log", xlabel="tokens seen", ylabel=lab,
                   title=f"{get_model_label(m)}  ($k$ = {k_of(MC[m][st[-1]])})")
            if key in ("logitvar", "rankme"):
                ax.set_yscale("log")
            ax.title.set_fontsize(9)
        for ax in axes.ravel()[len(models):]:
            ax.set_axis_off()
        fig.suptitle(f"{lab} — largest direction of after_final_norm, over training",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .95))
