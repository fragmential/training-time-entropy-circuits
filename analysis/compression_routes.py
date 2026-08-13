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
# # Two routes to the same compression
#
# **The object of this notebook.** Late in pretraining the model's final representation packs
# its variance into fewer directions. The last layer does a lot of that packing: switch it off
# and the number of directions in use jumps. The question here is *which* directions the
# packing goes into — because there is more than one way to do it, and the usual measure
# cannot tell them apart.
#
# **Terms, in plain form.**
#
# * **Unembedding.** The final matrix, turning the model's internal vector into one score per
#   vocabulary word. Its own directions come with sizes: some it reads loudly, some faintly.
#
# * **Two definitions of "dark", both used here.** The faintest **1%** of the unembedding's
#   directions is Stolfo et al's k ≈ 0.01·d; the faintest **5%** is Cancedda's band. Sections
#   1–3 use 1%, section 5 uses 5%, and section 4b shows both — plus a much wider 20% band — so
#   the choice can be seen rather than assumed. A random direction scores √(k/d) either way, and every figure states its own
#   k, so the two are read the same way despite the different thresholds.
#
# * **Dark directions.** The ~1% the unembedding reads most faintly. A neuron writing there
#   changes the internal vector without changing the word scores much — so it changes the
#   model's *confidence* without changing *what* it predicts. Named the effective null space
#   by Stolfo et al., *Confidence Regulation Neurons in Language Models* (arXiv:2406.16254);
#   shown to be where attention sinks live by Cancedda, *Spectral Filters, Dark Signals, and
#   Attention Sinks* (arXiv:2402.09221).
#
# * **Bright direction.** The single direction the unembedding reads loudest. Cancedda reports
#   its word-side image tracks how *common* each word is, so pushing along it slides the
#   model's output between "what the context says" and "what is common in general" — also a
#   confidence change, but out in the open rather than hidden.
#
# * **rho.** The share of a neuron's write that lands in the dark directions.
#
# * **Unit = one MLP neuron.** One entry of the MLP's hidden layer in the model's last block,
#   equivalently one column of that block's down-projection matrix — 8192 of them in
#   pythia-1b. Each neuron has an activation (how hard it fires on a given token) and a write
#   vector (the fixed direction it adds to the representation). "Unit" rather than "neuron"
#   only because the same machinery can also ablate attention heads; everything in this
#   notebook is neurons.
#
# * **Compressor / decompressor.** Switch one neuron off and re-measure how many directions
#   the final representation uses. If the number goes *up*, the neuron was packing (a
#   compressor). If it goes *down*, it was spreading (a decompressor). The measurement is
#   `solo` from `oneoff_scripts/neuron_ablation_curve.py`, taken through the whole network, so
#   the final normalisation and everything downstream are the network's own.
#
# **Why this notebook exists.** Ranking neurons by rho picks out the compressors well in
# nanochat and OLMo-2 and badly in Pythia. Either Pythia lacks dark writers, or it has them
# and they are doing two opposite jobs at once. Sections 1-3 settle that and then describe
# both groups across the *whole* unembedding spectrum instead of one end of it. Section 4 is
# the check that needs no neuron ranking at all.
#
# **Read section 5 before believing sections 1-3.** Those sections all measure the LAST block,
# and section 5 shows that in Pythia the dark direction is overwhelmingly present *earlier* in
# the stack and largely cancelled by the time the last block is reached. A quantity that comes
# out near zero in sections 1-3 is therefore evidence about the last block, not about the
# mechanism. Section 5 is the one that establishes the premise the rest of the notebook rests
# on, and it is measured where the thing actually lives.
#
# Data: `data/results/compression_routes_*.pt` (`oneoff_scripts/compression_routes.py`),
# `data/results/ablation_curves/mlp_mean.pt`, `data/results/stream_deflation{,_k05,_k20}.pt`,
# `data/results/sink_darkness_*.pt` (`oneoff_scripts/sink_darkness.py`).

# %%
# %load_ext autoreload
# %autoreload 2
import glob, os, sys
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import torch

from analysis.experiments_lib import get_model_label

R = "data/results"
ORDER = ["pythia-410m-deduped", "pythia-1b-deduped", "pythia-6.9b-deduped",
         "OLMo-2-0425-1B", "OLMo-2-1124-7B", "nanochat-d12"]
N_GROUP = 64          # neurons per group; the ablation curves have a k=64 rung too
COLORS = dict(zip(ORDER, plt.cm.tab10.colors))


def load(name):
    p = f"{R}/{name}"
    return torch.load(p, weights_only=False) if os.path.exists(p) else None


def caption(text):
    print("\n".join(__import__("textwrap").wrap(text, 100)))


def merged(pattern):
    """The per-model files the array job writes, gathered into one dict."""
    out = {}
    for f in sorted(glob.glob(f"{R}/{pattern}")):
        out.update(torch.load(f, weights_only=False))
    return out or None


ABL = load("ablation_curves/mlp_mean.pt")
CR = merged("compression_routes_*.pt")
DEF = load("stream_deflation.pt")
MODELS = [m for m in ORDER if CR and m in CR] if CR else []


def groups(model, step):
    """(compressor indices, decompressor indices, solo) for one model and checkpoint."""
    blocks = ABL[model]
    solo = blocks[max(blocks)][step]["solo"].float().numpy()
    return np.argsort(-solo)[:N_GROUP], np.argsort(solo)[:N_GROUP], solo


# %% [markdown]
# ## 1. Does the last layer's packing and spreading come from different neurons?
#
# **Question.** Switching off the whole last MLP changes how many directions the final
# representation uses. Is that one population pulling one way, or two populations partly
# cancelling?
#
# | | |
# |---|---|
# | measured | for every neuron, the change in directions-in-use when that neuron alone is switched off, taken through the whole network |
# | plotted | the sorted list of those per-neuron effects |
# | reading | a curve entirely above zero = one population. A curve crossing zero with mass on both sides = two opposed populations, and the layer's total is the difference between them |
# | shading | the two groups of 64 used everywhere below |

# %%
if ABL is None:
    print("[skipped] ablation_curves/mlp_mean.pt not on disk")
else:
    avail = [m for m in ORDER if m in ABL]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, model in zip(axes.flat, avail):
        blocks = ABL[model]
        for step, shade in zip(sorted(blocks[max(blocks)]), (0.45, 1.0)):
            e = blocks[max(blocks)][step]
            solo = np.sort(e["solo"].float().numpy())[::-1]
            ax.plot(np.arange(len(solo)), solo, lw=1.5, alpha=shade,
                    color=COLORS.get(model), label=f"{e['tokens']:.1e} tokens seen")
        ax.axhline(0, color="k", lw=0.9, ls=":")
        ax.axvspan(0, N_GROUP, color="tab:green", alpha=0.18)
        ax.axvspan(len(solo) - N_GROUP, len(solo), color="tab:purple", alpha=0.18)
        ax.set(xlabel="neurons of the last layer, sorted", title=get_model_label(model),
               ylabel="change in directions-in-use when this neuron is switched off")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(avail):]:
        ax.axis("off")
    fig.suptitle("Per-neuron effect on the final representation, sorted "
                 "(green = the 64 packers, purple = the 64 spreaders)")
    plt.tight_layout(); plt.show()
    caption("Every neuron of the last MLP, sorted by how much switching it off changes the "
            "number of directions the final representation uses; two checkpoints per model. "
            "Above zero the neuron was packing, below zero it was spreading. Shaded bands mark "
            "the two groups of 64 used in later sections. Descriptive only.")

# %% [markdown]
# ## 2. Are the packers darker than the spreaders?
#
# **Question.** If packing is done by writing into the directions the unembedding cannot see,
# then the packers should be much darker than the spreaders. Are they?
#
# | | |
# |---|---|
# | measured | rho, each group's median, divided by the median over the whole layer |
# | reading | 1.0 = no darker than an average neuron. A tall green bar next to a short purple one means darkness identifies the packers. Two bars of equal height mean it does not |
# | correctness note | for OLMo-2 the write direction is only right once the norm sitting at the end of its MLP branch is folded in; `oneoff_scripts/compression_routes.py` folds it. Values computed without it are not comparable |

# %%
if CR is None:
    print("[skipped] compression_routes_*.pt not on disk yet — array job pending")
else:
    labels, comp, decomp = [], [], []
    for model in MODELS:
        for step in sorted(CR[model]):
            if step not in ABL[model][max(ABL[model])]:
                continue
            rho = CR[model][step]["dark_share"].numpy()
            up, down, _ = groups(model, step)
            med = np.median(rho)
            labels.append(f"{get_model_label(model)}\n{CR[model][step]['tokens']:.0e} tok")
            comp.append(np.median(rho[up]) / med)
            decomp.append(np.median(rho[down]) / med)
    i = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.5 * len(labels) + 3, 4.4))
    ax.bar(i - 0.2, comp, 0.4, color="tab:green", label="the 64 packers")
    ax.bar(i + 0.2, decomp, 0.4, color="tab:purple", label="the 64 spreaders")
    ax.axhline(1, color="k", lw=0.9, ls=":", label="no darker than an average neuron")
    ax.set(xticks=i, ylabel="how dark the group is, relative to its layer")
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    fig.suptitle("Darkness of the packers against the darkness of the spreaders")
    plt.tight_layout(); plt.show()
    caption("Median rho of each group divided by the median over the whole last layer, per "
            "model and checkpoint. Equal bars mean darkness does not distinguish the two "
            "jobs. Descriptive only.")

# %% [markdown]
# ## 3. Where do the two groups write, across the whole unembedding?
#
# **Question.** Section 2 asks one yes/no question about one end of the unembedding's
# spectrum. Asked across the entire spectrum, do the two groups look different anywhere?
#
# | | |
# |---|---|
# | plotted | the unembedding's directions split into 20 equal bands, brightest on the left, darkest on the right; for each band, how much of a neuron's write direction lands in it |
# | lines | median over each group, and over the whole layer for reference |
# | why bands | this is the layout of Cancedda, arXiv:2402.09221, whose point is that both ends of the spectrum carry traffic and the middle is comparatively idle |
#
# The two candidate second routes each predict a specific band: the frequency route predicts
# the leftmost band, the sink-cancelling route predicts the rightmost. A group that separates
# in neither is telling us the route is something else again.

# %%
if CR is not None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, model in zip(axes.flat, MODELS):
        step = max(CR[model])
        bands = CR[model][step]["bands"].numpy()          # (20, neurons)
        up, down, _ = groups(model, step)
        x = np.arange(bands.shape[0]) + 1
        ax.plot(x, np.median(bands[:, up], axis=1), lw=1.8, color="tab:green",
                marker="o", ms=3, label="the 64 packers")
        ax.plot(x, np.median(bands[:, down], axis=1), lw=1.8, color="tab:purple",
                marker="o", ms=3, label="the 64 spreaders")
        ax.plot(x, np.median(bands, axis=1), lw=1.4, color="0.4", ls=":",
                label="all neurons of the layer")
        ax.set(yscale="log", xlabel="unembedding band (1 = read loudest, 20 = read faintest)",
               ylabel="share of the neuron's write in the band",
               title=f"{get_model_label(model)} — {CR[model][step]['tokens']:.1e} tokens seen")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(MODELS):]:
        ax.axis("off")
    fig.suptitle("Where each group writes, across the unembedding's whole spectrum")
    plt.tight_layout(); plt.show()
    caption("Median share of a neuron's write direction falling in each of 20 equal bands of "
            "the unembedding's spectrum, brightest band first, for the packers, the spreaders "
            "and the whole layer, at each model's final checkpoint. Descriptive only.")

# %% [markdown]
# ## 3b. The candidate signatures, side by side
#
# **Question.** Five scalars have been proposed to tell the two groups apart. Which of them
# actually does — and does any of them survive the obvious confound?
#
# | | |
# |---|---|
# | darkness | share of the write in the faintest 1% — the confidence-by-hiding route |
# | brightness | share of the write in the single loudest direction — the confidence-by-prior route |
# | how hard it fires | the neuron's activation size alone, no weights, no directions. **The control.** A neuron that is simply on more of the time moves the representation more whatever direction it points in, so any "mass" measure that multiplies by firing will separate the groups for a reason that has nothing to do with the unembedding |
# | weight size | the neuron's write vector length alone. Stolfo et al identify entropy neurons partly by a *large* weight; this asks whether the neurons that actually do the packing have one |
# | sink alignment | *signed* cosine with the direction of the model's most concentrated early write. Negative means the neuron writes *against* it, i.e. cancels it — and an absolute value would hide that. **Read section 5 first:** this comes out near zero for every model, but it is measured at the LAST block, where the project's ledger says the early direction has already been largely cancelled. A null here is uninformative about whether the mechanism exists earlier, and section 5 shows it plainly does |
#
# The early write is chosen without a per-model table: the early MLP write whose single
# largest direction holds the biggest share of that write's own variance. The share is printed
# in each panel label, so a model with no concentrated early write says so.

# %%
if CR is not None:
    KEYS = (("dark_share", "darkness (faintest 1%)"),
            ("bright_share", "brightness (loudest direction)"),
            ("act_rms", "how hard it fires — THE CONTROL"),
            ("w_norm", "weight size"),
            ("sink_cos", "alignment with the early write (signed)"))
    fig, axes = plt.subplots(len(KEYS), 1, figsize=(1.6 * len(MODELS) + 4, 3.5 * len(KEYS)))
    for ax, (key, title) in zip(axes, KEYS):
        labels, comp, decomp = [], [], []
        for model in MODELS:
            step = max(CR[model])
            v = CR[model][step][key].numpy()
            up, down, _ = groups(model, step)
            scale = np.median(np.abs(v)) or 1.0
            norm = 1.0 if key == "sink_cos" else scale
            labels.append(f"{get_model_label(model)}\nearly write = blk"
                          f"{CR[model][step]['sink_block']} "
                          f"({CR[model][step]['sink_share']:.2f})")
            comp.append(np.median(v[up]) / norm)
            decomp.append(np.median(v[down]) / norm)
        i = np.arange(len(labels))
        ax.bar(i - 0.2, comp, 0.4, color="tab:green", label="the 64 packers")
        ax.bar(i + 0.2, decomp, 0.4, color="tab:purple", label="the 64 spreaders")
        ax.axhline(1 if key != "sink_cos" else 0, color="k", lw=0.9, ls=":")
        ax.set(xticks=i, title=title,
               ylabel="signed cosine" if key == "sink_cos" else "relative to the layer median")
        ax.set_xticklabels(labels, fontsize=7)
        ax.legend(fontsize=8)
    fig.suptitle("Which signature separates the packers from the spreaders?")
    plt.tight_layout(); plt.show()
    caption("Five candidate signatures for the two groups, each at the model's final "
            "checkpoint. The first four are medians relative to the layer median; the last is "
            "a signed cosine, where negative means writing against the early write. Panel "
            "labels give which block the early write was taken from and how concentrated it "
            "is. Descriptive only.")

# %% [markdown]
# ## 3c. With the firing rate divided out
#
# **Question.** Section 3b shows the packers fire much harder than the spreaders. That alone
# would make them move the representation more, in any direction. With that removed, does
# either end of the unembedding still separate the two groups?
#
# | | |
# |---|---|
# | plotted | darkness x weight size, and brightness x weight size — how much *write magnitude* each group aims at each end of the spectrum, with the data-dependent firing removed |
# | reading | above 1 = the packers aim more there than the spreaders. Below 1 = the spreaders do |
# | why it matters | this is the version of the claim that is about the unembedding rather than about which neurons happen to be busy |

# %%
if CR is not None:
    labels, dark, bright = [], [], []
    for model in MODELS:
        step = max(CR[model])
        e = CR[model][step]
        up, down, _ = groups(model, step)
        w = e["w_norm"].numpy()
        for arr, out in ((e["dark_share"].numpy() * w, dark),
                         (e["bright_share"].numpy() * w, bright)):
            out.append(np.median(arr[up]) / max(np.median(arr[down]), 1e-30))
        labels.append(get_model_label(model))
    i = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3, 4.4))
    ax.bar(i - 0.2, dark, 0.4, color="tab:purple", label="aimed at the faintest 1%")
    ax.bar(i + 0.2, bright, 0.4, color="tab:red", label="aimed at the loudest direction")
    ax.axhline(1, color="k", lw=0.9, ls=":", label="the two groups aim there equally")
    ax.set(xticks=i, ylabel="packers' write magnitude / spreaders'")
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    fig.suptitle("Write magnitude aimed at each end of the unembedding, firing rate removed")
    plt.tight_layout(); plt.show()
    caption("Ratio of the packers' to the spreaders' median write magnitude aimed at the "
            "faintest 1% and at the loudest direction of the unembedding, with the "
            "data-dependent firing rate divided out, at each model's final checkpoint. "
            "Descriptive only.")

# %% [markdown]
# ## 4. The premise at the end of the model: is the dominant direction dark there?
#
# **Question.** Everything above assumes the biggest thing in the representation sits in the
# directions the unembedding barely reads. Cancedda (arXiv:2402.09221) reports exactly that for
# the attention sink. Before asking which subspace carries the compression, check that the
# subspace names mean what they are supposed to.
#
# | | |
# |---|---|
# | measured | the share of a direction's length lying in the faintest 1% of the unembedding's directions. 0 = none of it, 1 = all of it |
# | dotted | what a randomly chosen direction would score, √(k/d). A line sitting on the dotted line means "no darker than chance" |
# | three directions | the **mean** of the representation; its top **uncentered** direction; its top **centered** direction |
# | why three | a massive activation is largely a constant offset. Centring a covariance deletes constants by construction, so the centred direction can miss it entirely while the mean and uncentred ones find it. If the three disagree, centring is the reason |
# | two figures | `after_final_norm` first, then `before_final_norm`. The after-norm stream is the one that reaches the unembedding and the one every other measurement in this project uses; the before-norm stream is where a massive activation still has its full size, since the final normalisation divides it away |
#
# If all three sit at chance, the measurement is wrong rather than the literature — that is the
# reading, not a finding.

# %%
DIRS = (("mean", "the mean of the representation", "tab:green"),
        ("top_uncentered", "biggest direction, uncentred", "tab:red"),
        ("top_centered", "biggest direction, centred", "tab:blue"))


def where_grid(data, leaf):
    """Share of each of three stream directions inside the faintest k, over training."""
    if data is None:
        return print("[skipped] stream_deflation.pt not on disk yet — job pending")
    has = lambda m, st: leaf in data[m][st] and "where" in data[m][st][leaf]
    avail = [m for m in ORDER if m in data and any(has(m, st) for st in data[m])]
    pct = f"{100 * data[avail[0]][max(data[avail[0]])].get('k_frac', 0.01):g}%"
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, model in zip(axes.flat, avail):
        steps = sorted(st for st in data[model] if has(model, st))
        x = np.array([data[model][st]["tokens"] for st in steps], float)
        for key, lab, c in DIRS:
            ax.plot(x, [data[model][st][leaf]["where"][key] for st in steps],
                    lw=1.7, color=c, label=lab)
        ax.plot(x, [data[model][st][leaf]["where"]["chance"] for st in steps], "k:", lw=1.2,
                label="a random direction would score this")
        ax.set(xscale="log", xlabel="tokens seen", ylim=(0, 1.02),
               ylabel=f"share of the direction inside the faintest {pct}",
               title=get_model_label(model))
        ax.legend(fontsize=6)
    for ax in axes.flat[len(avail):]:
        ax.axis("off")
    fig.suptitle(f"How much of the dominant direction lies in the faintest {pct} of the "
                 f"unembedding — {leaf}")
    plt.tight_layout(); plt.show()
    caption(f"Share of three directions of the {leaf} representation lying in the faintest "
            f"{pct} of the unembedding's directions, over training. Dotted: what a random "
            f"direction scores. The centred and uncentred lines separate exactly when the "
            f"direction is a constant offset.")


where_grid(DEF, "after_final_norm")
where_grid(DEF, "before_final_norm")

# %% [markdown]
# ## 4b. Which directions carry the compression, with no neuron ranking at all?
#
# **Question.** Sections 1-3 depend on picking neurons. Removing that step: if the packing is
# into the unembedding's faintest directions, then deleting those directions from the
# representation should remove the late fall in directions-in-use. Does it?
#
# | | |
# |---|---|
# | measured | directions-in-use (RankMe) of the final representation, over training |
# | black | untouched |
# | coloured | the same after deleting a named set of the unembedding's directions from the representation: the faintest k, the single loudest, the loudest k, or a random k as the control |
# | three figures | k = 1% (Stolfo et al's band), k = 5% (Cancedda's), and k = 20% (no one's band — how far the picture survives widening). Each wider band deletes proportionally more directions, so the control line moves too — compare each removal against its own random control, not across the figures |
# | reading | the line that comes out *flat* names the carrier. A line that still falls means those directions were not where the packing went |
# | note | the unembedding is used **raw** — not vocabulary-centred, not gain-folded. The measured representation already carries the final norm's gain, and raw is also Cancedda's convention, so section 5 is directly comparable. An earlier version centred it; that recomputed the matrix from weights for a change worth <0.01 (see `utils/nullspace.head_subspace`) |

# %%
DEF_K05 = load("stream_deflation_k05.pt")
LINES = (("base", "untouched", "k", "-"),
         ("dark", "faintest {p} deleted", "tab:purple", "-"),
         ("top1", "loudest direction deleted", "tab:red", "-"),
         ("topk", "loudest {p} deleted", "tab:orange", "-"),
         ("random", "random {p} deleted (control)", "tab:blue", "--"))


def deflation_grid(data, leaf="after_final_norm"):
    """One panel per model; k is read from the data rather than assumed."""
    if data is None:
        return print("[skipped] not on disk yet — job pending")
    avail = [m for m in ORDER if m in data and data[m]]
    pct = f"{100 * data[avail[0]][max(data[avail[0]])].get('k_frac', 0.01):g}%"
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, model in zip(axes.flat, avail):
        steps = sorted(s for s in data[model] if leaf in data[model][s])
        x = np.array([data[model][s]["tokens"] for s in steps], float)
        for key, lab, c, ls in LINES:
            y = [data[model][s][leaf][f"rankme_{key}"] for s in steps]
            ax.plot(x, y, ls, lw=1.8 if key == "base" else 1.4, color=c, label=lab.format(p=pct))
        k = data[model][steps[-1]]["k"]
        ax.set(xscale="log", xlabel="tokens seen", ylabel="directions in use (RankMe)",
               title=f"{get_model_label(model)} — k = {k} of {data[model][steps[-1]]['d']}")
        ax.legend(fontsize=6)
    for ax in axes.flat[len(avail):]:
        ax.axis("off")
    fig.suptitle(f"Directions in use after deleting named parts of the unembedding "
                 f"(k = {pct} of the spectrum)")
    plt.tight_layout(); plt.show()
    caption(f"RankMe of the centred after_final_norm covariance over training, untouched and "
            f"with each named subspace of the raw unembedding projected out. k = {pct} of the "
            f"spectrum. Compare each removal against the random control in the same figure.")


deflation_grid(DEF)
deflation_grid(DEF_K05)
deflation_grid(load("stream_deflation_k20.pt"))

# %% [markdown]
# ## 4c. The same, as how much of the fall each removal accounts for
#
# **Question.** Section 4b read by eye. Quantified: how much of the late fall in
# directions-in-use does each removal take away?
#
# | | |
# |---|---|
# | measured | for each curve, the drop from its own highest point to its final value, as a fraction of that highest point |
# | reading | the untouched bar is the size of the compression phase. A removal whose bar is much shorter took the compression with it |
# | control | the random bar shows how much shrinkage comes from deleting 1% of anything |

# %%
if DEF is not None:
    def drawdown(model, key):
        steps = sorted(s for s in DEF[model] if "after_final_norm" in DEF[model][s])
        y = np.array([DEF[model][s]["after_final_norm"][f"rankme_{key}"] for s in steps])
        return float((y.max() - y[-1]) / y.max()) if len(y) > 1 and y.max() > 0 else np.nan

    keys = [k for k, *_ in LINES]
    i = np.arange(len(avail))
    fig, ax = plt.subplots(figsize=(1.7 * len(avail) + 4, 4.4))
    for off, (key, lab, c, _) in zip(np.linspace(-0.32, 0.32, len(keys)), LINES):
        ax.bar(i + off, [drawdown(m, key) for m in avail], 0.16, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.9)
    ax.set(xticks=i, ylabel="fraction of the peak lost by the end of training")
    ax.set_xticklabels([get_model_label(m) for m in avail], rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    fig.suptitle("How big is the compression phase, and how much of it does each removal take away?")
    plt.tight_layout(); plt.show()
    caption("Fall from peak to final value in directions-in-use, as a fraction of the peak, "
            "for the untouched representation and for each removal, per model. A short bar "
            "means that removal accounts for most of the compression. Descriptive only.")

# %% [markdown]
# ## 5. The premise, measured where the thing actually lives
#
# **Question.** Sections 1–5 all measure the last block. Cancedda (arXiv:2402.09221) reports
# that a model's attention sink sits almost entirely in the directions the unembedding barely
# reads. Is that true of these models — and is it true at the depth where the sink write has
# just happened, rather than at the end where the late blocks have been undoing it?
#
# | | |
# |---|---|
# | probe: after the write | the residual stream entering the block just after each model's known concentrated early write — pythia-410m blk9, pythia-1b blk4, pythia-6.9b blk6, nanochat blk5 |
# | probe: near the end | the residual stream entering the third-to-last block |
# | OLMo-2 | has no such early write, so its "after the write" probe is at a nominal depth only, and is expected to sit at chance. That makes it the negative control |
# | dark | Cancedda's band: the faintest **5%** of the unembedding's directions, raw matrix |
# | share | how much of a direction's length lies in that band. 0 = none, 1 = all. A random direction scores √(k/d) ≈ 0.223, drawn as the dotted line |
# | ratio | Cancedda's own per-token measure: energy inside the dark band divided by energy outside it. **1.0 = half in, half out**; above 1 means a token's vector is mostly dark |
# | three directions | the representation's **mean**, its largest direction **uncentred**, and its largest direction **centred**. A massive activation is largely a constant offset, and centring deletes constants, so a gap between the last two means centring is hiding it |
# | sample | 131,072 tokens per model per checkpoint |
#
# The token groups in the ratio panel are: every token; the tokens at position 0 of their
# window, which is the attention-sink slot; and the 1% of tokens with the largest vectors.

# %%
SINK = merged("sink_darkness_*.pt")
PROBES = (("post_sink", "just after the early write"), ("late", "third-to-last block"))


def sink_series(model, probe, field, key):
    """(tokens seen, value) over every checkpoint measured for one model."""
    steps = sorted(SINK[model])
    x = np.array([SINK[model][s]["tokens"] for s in steps], float)
    y = np.array([SINK[model][s]["at"][probe][field][key] for s in steps], float)
    return x, y


if SINK is None:
    print("[skipped] sink_darkness_*.pt not on disk")
else:
    avail = [m for m in ORDER if m in SINK]
    DIRS = (("mean", "the mean", "tab:green"),
            ("top_uncentered", "biggest, uncentred", "tab:red"),
            ("top_centered", "biggest, centred", "tab:blue"))
    for probe, plab in PROBES:
        fig, axes = plt.subplots(1, len(avail), figsize=(3.4 * len(avail), 3.8), squeeze=False)
        for ax, model in zip(axes[0], avail):
            for key, lab, c in DIRS:
                x, y = sink_series(model, probe, "share_0.05", key)
                ax.plot(x, y, lw=1.7, marker="o", ms=3, color=c, label=lab)
            chance = SINK[model][max(SINK[model])]["at"][probe]["chance_0.05"]
            ax.axhline(chance, color="k", lw=1.1, ls=":", label=f"random ({chance:.3f})")
            blk = SINK[model][max(SINK[model])]["probes"][probe]
            ax.set(xscale="log", xlabel="tokens seen", ylim=(0, 1.02),
                   title=f"{get_model_label(model)} — blk{blk}",
                   ylabel="share in the faintest 5%")
            ax.legend(fontsize=6)
        fig.suptitle(f"How dark is the dominant direction, over training — {plab}")
        plt.tight_layout(); plt.show()
        caption(f"Share of three directions of the residual stream lying in the faintest 5% of "
                f"the unembedding's directions, over training, measured {plab}. Dotted = what a "
                f"random direction scores. A gap between the uncentred and centred lines means "
                f"the direction is a constant offset that centring removes. Descriptive only.")

    RATIOS = (("all", "every token", "0.5"),
              ("pos0", "tokens at position 0 (the sink slot)", "tab:orange"),
              ("top1pct_norm", "the 1% largest vectors", "tab:purple"))
    for probe, plab in PROBES:
        fig, axes = plt.subplots(1, len(avail), figsize=(3.4 * len(avail), 3.8), squeeze=False)
        for ax, model in zip(axes[0], avail):
            for key, lab, c in RATIOS:
                x, y = sink_series(model, probe, "ratio_0.05", key)
                ax.plot(x, y, lw=1.7, marker="o", ms=3, color=c, label=lab)
            ax.axhline(1, color="k", lw=1.1, ls=":", label="half in, half out")
            ax.set(xscale="log", xlabel="tokens seen",
                   title=get_model_label(model),
                   ylabel="dark energy / non-dark energy")
            ax.legend(fontsize=6)
        fig.suptitle(f"Cancedda's per-token dark energy ratio, over training — {plab}")
        plt.tight_layout(); plt.show()
        caption(f"Energy inside the dark band divided by energy outside it, per token group, "
                f"over training, measured {plab}. Above the dotted line a token's vector is "
                f"more than half dark. Descriptive only.")

# %% [markdown]
# ## 6. The middle of the stack, which none of the above explains
#
# Everything so far is about the last layer. But the middle of the stack compresses too, on
# its own schedule, and in OLMo-2 it does so without any of the massive-activation machinery
# that explains Pythia's mid-stack behaviour. Two accounts:
#
# * **The output bottleneck** (Li et al., arXiv:2509.23024): skewed word frequencies plus
#   fewer dimensions than words force the model to reuse its biggest directions. This predicts
#   compression is strongest nearest the output and needs the bottleneck to bite.
#
# * **Shared features.** A useful mid-stack feature — register, topic, syntactic role,
#   position in the document — is one that *many different words* load on, so it is a
#   common-mode direction. Word identity is idiosyncratic and spread over many directions.
#   Learning better context therefore concentrates variance with no bottleneck involved.
#
# **Question.** Split the mid-stack representation into the part predictable from the current
# word and the part that is not. Which of the two compresses?
#
# | | |
# |---|---|
# | method | an analysis of variance on the residual stream, grouping rows by the current word. `identity` = how the group averages differ from each other; `context` = what is left, i.e. the same word in different surroundings |
# | plotted | directions-in-use for each part, and what share of the total variance each carries |
# | shared features predict | context becomes both the larger share and the lower rank, while identity stays high-rank |
# | bottleneck predicts | identity concentrates, as it collapses onto the frequent words' directions |

# %%
CI = merged("context_vs_identity_*.pt")
PARTS = (("total", "everything", "k", "-"),
         ("identity", "predictable from the word", "tab:orange", "-"),
         ("context", "the rest — the context part", "tab:cyan", "-"))

if CI is None:
    print("[skipped] context_vs_identity_*.pt not on disk yet — job pending")
else:
    for model in CI:
        steps = sorted(CI[model])
        blocks = CI[model][steps[-1]]["blocks"]
        names = [f"blk{b}" for b in blocks] + ["before_final_norm"]
        fig, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 7.5), squeeze=False)
        x = np.array([CI[model][s]["tokens"] for s in steps], float)
        for j, name in enumerate(names):
            for key, lab, c, ls in PARTS:
                y = [CI[model][s]["depths"][name][key]["rankme"] for s in steps]
                axes[0, j].plot(x, y, ls, lw=1.6, color=c, label=lab)
            share = [CI[model][s]["depths"][name]["context"]["trace"]
                     / max(CI[model][s]["depths"][name]["total"]["trace"], 1e-30) for s in steps]
            axes[1, j].plot(x, share, lw=1.8, color="tab:cyan")
            axes[1, j].axhline(1, color="k", lw=0.8, ls=":")
            depth = ("end of the model" if name == "before_final_norm"
                     else f"{100 * blocks[j] / CI[model][steps[-1]]['n_blocks']:.0f}% of the way up")
            axes[0, j].set(xscale="log", xlabel="tokens seen", title=depth,
                           ylabel="directions in use (RankMe)")
            axes[1, j].set(xscale="log", xlabel="tokens seen", ylim=(0, 1.05),
                           ylabel="share of the variance that is context")
            axes[0, j].legend(fontsize=7)
        fig.suptitle(f"{get_model_label(model)} — splitting the representation into "
                     f"'predictable from the word' and 'context'", y=1.01)
        plt.tight_layout(); plt.show()
        caption(f"Top: directions in use over training for the whole representation and for "
                f"each of its two parts, at three depths and at the end of the model, for "
                f"{get_model_label(model)}. Bottom: the share of the total variance carried by "
                f"the context part. Descriptive only.")
