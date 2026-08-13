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
#     display_name: representation-geometry (3.14.6.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Showcase: the compression phase, decomposed
#
# Companion to [docs/final_report.md](../docs/final_report.md). Every section has the same
# shape: a **Question**, the **Finding** that answers it, then the figures carrying the
# evidence — each caption says what is plotted and how far the claim reaches. Exemplar models
# per figure; the full per-model grids live in experiments.ipynb (Exp 4.x) and
# experiments_padded.ipynb. Methods: docs/ledger.md.
#
# Two repo-internal names, defined once here and used throughout:
#
# - **The decomposition** (repo name: "the rank ledger") — the exact per-block identity
#   ΔS = χ (overlap) + quality (block-intrinsic entropy) + interference, which sums across
#   blocks to the depth-differential change in log RankMe (docs/ledger.md).
#
# - **The sink write** (repo name: "the rogue write/direction") — pythia blk3.mlp's output,
#   whose top eigendirection is a sink direction carrying 94–99% of the write's variance (§4).

# %%
import os, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from analysis import experiments_lib as _lib
importlib.reload(_lib)
from analysis.experiments_lib import (grid_start, grid_show, plot_group, plot_heatmap,
                                      build_hooks, get_model_label, load_results)

def caption(txt):
    """Paper-style figure caption, rendered below the figure it follows."""
    from IPython.display import display, HTML
    display(HTML(f'<div style="font-size:0.97em; opacity:0.95; max-width:56em; '   # inherit theme
                 f'margin:0.2em 0 1.2em 0.5em"><b>Figure.</b> {txt}</div>'))      # text color

HK = build_hooks()
BLOCK_SAMPLES = "block_representations_samples"
TRAINSET_PACKED, FINEWEB_PADDED = "rankme_trainset_packed_4MT", "rankme_fineweb_padded"
ALL4 = ['pythia-1b-deduped', 'pythia-6.9b-deduped', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B']
# The presented model selection: ALL4 plus nanochat-d12 and pythia-410m, ordered so the two
# additions land in the top row of the 2-column grids.
ALL6 = ['nanochat-d12', 'pythia-410m-deduped', *ALL4]
PAIR = ['pythia-1b-deduped', 'OLMo-2-1124-7B']          # one exemplar per family
# §3 covers the full model set: four pythia scales, both OLMo-2 scales, nanochat-d12
LEDGER_MODELS = ['pythia-160m-deduped', 'pythia-410m-deduped', 'pythia-1b-deduped',
                 'pythia-6.9b-deduped', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'nanochat-d12']
SEQ = ['OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'nanochat-d12']   # sequential sub-blocks (Pythia's
                                                             # are parallel: no sub-step ΔS)
# §2's band-α maps: the ledger set minus pythia-160m, pythias → nanochat → OLMo-2s
SPEC6 = ['pythia-410m-deduped', 'pythia-1b-deduped', 'pythia-6.9b-deduped',
         'nanochat-d12', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B']
n_blocks = {'pythia-160m-deduped': 12, 'pythia-410m-deduped': 24,
            'pythia-1b-deduped': 16, 'pythia-6.9b-deduped': 32,
            'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32, 'nanochat-d12': 12}
d_model = {'pythia-160m-deduped': 768, 'pythia-410m-deduped': 1024,
           'pythia-1b-deduped': 2048, 'pythia-6.9b-deduped': 4096,
           'OLMo-2-0425-1B': 2048, 'OLMo-2-1124-7B': 4096, 'nanochat-d12': 768}
cfg_of = lambda m: 'nanochat_samples' if m == 'nanochat-d12' else BLOCK_SAMPLES
# tab10 for small sets of unordered series: mid-saturation and even in luminance, where
# base_colors' near-pure hues vibrate on white. Ordered sweeps keep the light-to-dark ramp.
QUAL = 'tab10'

# %% [markdown]
# ## 1. Reproduction: the Li et al phases
#
# **Question.** Does Li et al's warmup → entropy-seeking → compression RankMe trajectory
# appear in our models and data modes at all?
#
# **Finding.** Yes — across both families and both data modes (their setting: padded fineweb
# last-token; ours adds the packed training-mix), and in nanochat-d12 (packed only). This is
# the curve everything below decomposes.

# %%
# Packed curve from the all-model samples run (the old rankme_trainset_packed_4MT run only
# covered a model subset, which caused skipped-file noise here before). Full §3 model set:
# fineweb-padded exists for the pythias/OLMos; nanochat is packed-only (no padding support).
for model in LEDGER_MODELS:
    srcs = [(cfg_of(model), ('after_final_norm', 'acts_centered'), 'trainset packed')]
    if model != 'nanochat-d12':
        srcs.append((FINEWEB_PADDED, HK.AFN_AC, 'fineweb padded'))
    grid_start(ncols=2, savegroup='Li et al reproduction', savedir=get_model_label(model),
               title=f'Li et al reproduction — {get_model_label(model)}')
    plot_group('rankme', srcs, [model], color_palette=QUAL)
    plot_group('alpha', srcs, [model], color_palette=QUAL)
    grid_show()
    caption(f'RankMe (left) and power-law slope α (right) of the centered final-stream '
            f'covariance over training, {get_model_label(model)}, on the packed training mix '
            f'and (where collected) padded fineweb last-token — Li et al\'s setting. The '
            f'warmup → entropy-seeking → compression trajectory is the phase structure every '
            f'later figure decomposes. The small pythias are sanity checks for §3\'s 160m '
            f'edge case. Coloured from the tab10 cycle by source, so the two curves keep the '
            f'same two colours in every panel.')

#FIGURE A1

# %% [markdown]
# ## 2. What the scalar hides: the phases are spectrally local
#
# **Question.** Do the compression and expansion phases operate on the same parts of the
# spectrum — is the post-peak decline a contraction of the whole representation space?
#
# **Finding (universal, 4/4 models).** No: the measured compression lives in the top ~10–30
# eigendirections. Band alphas show a concentration front that reaches ranks 32–128 late,
# 128–512 barely, and NEVER ranks 512+ — that band flattens monotonically through all of
# training (falling α = entropy-seeking never ends there). RankMe is head-biased: it
# conflates the head event with a bulk that keeps expanding. Both band statistics are
# invariant to the head's growth.

# %%
KS, WINDOWS = (0, 32, 128), ((11, 100), (256, 512))#, (256, -50))
# Unordered pairs use QUAL (blue/orange) and stay solid: the most colour-vision-safe pair
# there is, so dashes are not needed to tell them apart.
C1_FS = 15
klabel = lambda k, m: f'k={k} ({100 * (d_model[m] - k) / d_model[m]:.0f}%)'   # % of dims kept
for model in ALL6:
    cfg = cfg_of(model)
    tails = [(cfg, ('after_final_norm', 'acts_centered'), klabel(k, model), ('tail_rankme', {'k': k})) for k in KS]
    alphas = [(cfg, ('after_final_norm', 'acts_centered'), f'{a}-{b}', ('alpha_window', {'k0': a, 'k1': b})) for a, b in WINDOWS]
    # both weightings of the same spectrum; the outer yvar names the left axis only
    variants = [(cfg, ('after_final_norm', 'acts_centered'), r'RankMe$_\lambda$', 'rankme'),
                (cfg, ('after_final_norm', 'acts_centered'), r'RankMe$_\sigma$', 'true_rankme')]
    grid_start(ncols=3, savegroup='Head-removed RankMe and band alpha',
               savedir=get_model_label(model),          # one dir per model under that group
               figsize=(16.5, 4.6),                     # smaller figure => larger text as shown
               title=f'Head-removed RankMe / band alpha — {get_model_label(model)}')
    plot_group('tail_rankme', tails, [model], color_palette='gradient', fs=C1_FS)
    plot_group('alpha_window', alphas, [model], color_palette=QUAL, fs=C1_FS)
    plot_group(r'RankMe$_\lambda$', variants, [model], color_palette=QUAL, fs=C1_FS,
               twin=1, twin_ylabel=r'RankMe$_\sigma$',
               ls_exceptions={'_': '-'}, hold_color_for_n=1)   # mathtext '_' is not a source tag
    grid_show()
caption(f'Left: RankMe with the top k eigendirections removed (light→dark; the bracket is '
        f'the share of d_model that survives the cut, and the retained spectrum is '
        f'renormalised before the entropy). '
        f'Middle: α fit inside fixed rank windows. The decline survives the small cuts but '
        f'is gone by k=128, and the deepest window moves the OPPOSITE way (flattening) '
        f'through all of training: a top-of-spectrum concentration front, not a global '
        f'contraction. '
        f'Right: the same spectrum under both RankMe weightings, on separate axes — entropy '
        f'of p ∝ λ (left, as used everywhere above) vs of p ∝ σ = √λ (right, the RankMe '
        f'paper\'s). The √ flattens the head, so RankMe$_\\sigma$ sits several times higher '
        f'and barely registers the event that RankMe$_\\lambda$ is dominated by; each axis is '
        f'tinted to its own series.')

#FIGURE C1

# %% [markdown]
# ### Baseline: what a pure spectrum-tilt compression would look like
# Not an experiment — a boring reference curve to overlay on the measured ones. The simplest
# caricature of "selection bias compresses the spectrum" is a power-law spectrum whose tilt
# steepens steadily: λ_i(t) = i^{−α(t)}, α increasing linearly in log-tokens (each unit of
# time multiplies λ_i by i^{−c} — relative growth proportional to relative rank, spectrum-
# wide). α's endpoints are calibrated so the synthetic RankMe matches the measured value at
# the RankMe peak and at the end of training — the overlay therefore isolates SHAPE and band
# behaviour, not level. Under the baseline everything moves together: a smooth, featureless
# decline, and every tail band falls in lockstep. The measured trajectory can then be judged
# against that: its decline is carried by the head while the deep band moves the OTHER way
# (band figure above) — i.e. real compression is not a uniform tilt.

# %%
from analysis.experiments_lib import _rankme as _rm

def _rm_of_alpha(a, d=2048, k=0):
    lam = np.arange(1, d + 1, dtype=float) ** -a
    return _rm(lam[k:])

def _alpha_for_rm(target, d=2048):
    lo, hi = 1e-3, 12.0                       # RankMe is monotone-decreasing in the tilt
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _rm_of_alpha(mid, d) > target else (lo, mid)
    return (lo + hi) / 2

def tilt_baseline(model='pythia-1b-deduped'):
    hook = ('after_final_norm', 'acts_centered')
    ys, steps = _lib.get_ys(BLOCK_SAMPLES, model, hook, 'rankme')
    xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
    pk = int(np.argmax(ys[2:])) + 2
    tr = pk + int(np.argmin(ys[pk:]))            # RankMe minimum = end of the compression phase
    a0, a1 = _alpha_for_rm(ys[pk]), _alpha_for_rm(ys[tr])
    sl = slice(pk, tr + 1)
    u = (np.log(xs[sl]) - np.log(xs[pk])) / (np.log(xs[tr]) - np.log(xs[pk]))
    alphas = a0 + (a1 - a0) * u
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Tilt baseline vs measured — {get_model_label(model)}')
    axes[0, 0].plot(xs, ys, lw=2, label='measured stream RankMe')
    axes[0, 0].plot(xs[sl], [_rm_of_alpha(a) for a in alphas], 'k--', lw=2,
                    label=f'tilt baseline (α {a0:.2f}→{a1:.2f})')
    axes[0, 0].set(xscale='log', xlabel='tokens', ylabel='RankMe',
                   title='headline curve: baseline calibrated peak → RankMe minimum')
    axes[0, 0].legend(fontsize=8)
    for k, c in ((32, 'tab:orange'), (512, 'tab:red')):
        mys, msteps = _lib.get_ys(BLOCK_SAMPLES, model, hook, 'tail_rankme', {'k': k})
        mx = np.asarray(_lib.get_xs_tokens(model, msteps), float)
        axes[0, 1].plot(mx[pk:], np.asarray(mys[pk:]) / mys[pk], lw=2, color=c,
                        label=f'measured tail k={k}')
        axes[0, 1].plot(xs[sl], [_rm_of_alpha(a, k=k) / _rm_of_alpha(a0, k=k) for a in alphas],
                        '--', lw=2, color=c, alpha=0.6, label=f'baseline tail k={k}')
    axes[0, 1].axhline(1, color='gray', lw=0.8)
    axes[0, 1].set(xscale='log', xlabel='tokens', ylabel='tail RankMe / value at peak',
                   title='bands: baseline drags every band down together')
    axes[0, 1].legend(fontsize=8)
    ays, asteps = _lib.get_ys(BLOCK_SAMPLES, model, hook, 'alpha')
    ax_ = np.asarray(_lib.get_xs_tokens(model, asteps), float)
    axes[1, 0].plot(ax_, ays, lw=2, label='measured α (global fit)')
    axes[1, 0].plot(xs[sl], alphas, 'k--', lw=2, label='baseline α(t)')
    axes[1, 0].set(xscale='log', xlabel='tokens', ylabel='α',
                   title='global power-law slope vs the baseline tilt')
    axes[1, 0].legend(fontsize=8)
    for (k0, k1), c in zip(WINDOWS, ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')):
        wys, wsteps = _lib.get_ys(BLOCK_SAMPLES, model, hook, 'alpha_window', {'k0': k0, 'k1': k1})
        wx = np.asarray(_lib.get_xs_tokens(model, wsteps), float)
        axes[1, 1].plot(wx, wys, lw=2, color=c, label=f'measured α {k0}–{k1}')
    axes[1, 1].plot(xs[sl], alphas, 'k--', lw=2, label='baseline α(t), (all bands)')
    axes[1, 1].set(xscale='log', xlabel='tokens', ylabel='band α',
                   title='Pure tilt does not vary by band of the spectrum')
    axes[1, 1].legend(fontsize=7)
    plt.tight_layout()
    plt.show()

tilt_baseline()
caption('The tilt baseline (dashed): a power-law spectrum whose slope steepens over time, '
        'calibrated to match the measured RankMe at the peak and at the minimum. Under it '
        'every band falls in lockstep and one α describes every rank. Measured instead: the '
        'deep band RISES through the compression phase and the band slopes fan out — real '
        'compression is a head event, not a uniform tilt.')
#FIGURE C2

# %% [markdown]
# ### The same band picture as a map: where the slope steepens, and when
# The band-α figure above tracks three hand-picked windows. Here the whole spectrum is chopped
# into 20 geometric rank bands (edges from rank 2 up to 3d/4; the top quarter is dropped
# because the spectrum falls off a cliff there and a band slope stops meaning anything), α is
# fitted inside each, and the result is one heatmap row per checkpoint.
#
# The bands are NOT uniform in log-rank: `C3_BANDS = (22, 4)` draws 22 bands with the rightmost
# 4× narrower than the leftmost (set the second number to 1 for a plain uniform split).
#
# The head bands come out ~2 ranks wide, which looks reckless for a two-parameter fit and is
# deliberate. The end-of-training head steepening — the bright block at the bottom left of the
# Pythia panels — is a NARROW feature living around ranks 4–8, and a wider head band averages
# it straight out: at ranks [4,6) pythia-1b reads α ≈ 1.9 over its last ten checkpoints, and
# widening that band to 4 ranks drops it to 0.74. It is not noise either. Over those last ten
# checkpoints α moves by 0.16 (1B) and 0.05 (6.9B) — the head band jitters 2–7× MORE before the
# RankMe peak than after it, so the speckle a narrow head band produces is confined to early
# training, where α genuinely is unstable, and does not touch the feature worth seeing.
# Widening the head to kill that speckle costs the finding. Raise the first number for more
# bands everywhere, the second to shift resolution from the head toward the tail.
#
# Both axes are log and the cells are drawn on real edges, so a row's height is its true share
# of log-token space — the densely-sampled late checkpoints come out as thin slivers rather
# than being spread evenly and lying about their spacing. The row at the far edge is the
# INITIALISATION checkpoint: it has real data but zero tokens, which has no place on a log
# axis, so it is drawn one geometric step before the first real checkpoint. That position is a
# display convention, not a token count — but dropping it hid OLMo-2 0425-1B's entire warmup,
# whose step 0 is the only row above the crash (RankMe 823 against 69 at the next checkpoint).
#
# Two dotted verticals mark ranks 11 and 100, the window the headline α number is fitted over,
# so every map shows how little of the spectrum that one number sees. Each panel also carries a
# narrow RankMe strip sharing the token axis — the scalar next to the decomposition of the
# scalar. The strip's own dash pattern carries the phases: dotted up to the warmup trough,
# solid through the entropy-seeking rise, dashed after the RankMe peak. Rules drawn across the
# map would read as data on a heatmap; the curve's own style does not.
#
# `vlim` couples or decouples the colour scale: a (vmin, vmax) pair puts every panel on one
# scale so cells are comparable ACROSS models, None autoscales each panel to its own 2–98th
# percentiles so structure is readable WITHIN one. Decoupled below, because on a shared scale
# the OLMo-2s (whose whole α range is ~0.6–1.8) sit in the middle greys while nanochat-d12
# clips to blocks of white. Swap in vlim=(0.2, 2.4) to compare models instead.
#
# Orientation, the RankMe strip's side, the palette and the figure size all come from the
# preset block below, so the whole section changes together rather than cell by cell.

# %%
# Layout presets for the §2 maps: (figsize, ncols, orient). VER = time down the panel (panels
# are tall, so the grid wants few columns); HOR = the transposed view, time across (panels are
# wide and short). CROP variants are for the entropy-peak-truncated figures, which have a
# handful of rows instead of ~60 and so need nothing like the height. Set one name and every
# call in the section follows it.
VER_2x3      = ((15, 38), 2, 'time_down')     # 6 models, 2 cols x 3 rows, full training
VER_3x2      = ((23, 26), 3, 'time_down')     # ditto, 3 cols x 2 rows
VER_2x3_CROP = ((15, 18), 2, 'time_down')     # truncated at the entropy peak
VER_3x2_CROP = ((25, 14), 3, 'time_down')
HOR_2x3      = ((30, 30), 2, 'time_right')    # transposed: time across, rank down
HOR_3x2      = ((35, 14), 3, 'time_right')
HOR_2x3_CROP = ((18, 20), 2, 'time_right')
HOR_3x2_CROP = ((23, 15), 3, 'time_right')

C3_LAYOUT      = HOR_3x2        # the three full-training maps
C3_LAYOUT_CROP = HOR_3x2_CROP   # the two entropy-peak-truncated maps
C3_BANDS = (10, 1)   # (how many bands, how many times NARROWER the tail band is than the
                     # head band). (10, 1) is ten equal log-rank bands, the plain version.
                     # 4x keeps the head bands ~2 ranks wide, which is what resolves the
                     # end-of-training head steepening — see the markdown above.
C3_FIT   = 'bar'     # alpha fit window: 'bar' (bracket above the map) | 'lines' | None
C3_PAD   = 3.5       # extra space between grid panels (grid only, not the panel PDFs)
C3_FS    = 22        # base font size for the maps (labels; ticks/title scale off it)
C3_CMAP  = 'magma'   # magma | inferno | plasma | viridis | cividis | mako | rocket | turbo
C3_SIDE  = 'tail'    # which end of the SPECTRUM the RankMe strip sits on: head | tail | None

# %%
grid_start(ncols=C3_LAYOUT_CROP[1], figsize=C3_LAYOUT_CROP[0], pad=C3_PAD,
           savegroup='Band alpha map', savedir='warmup to the entropy peak',
           title='Band alpha map — warmup to the entropy peak')
for model in SPEC6:
    _lib.plot_alpha_bands(model, cfg_of(model), include_init=False, upto=_lib.rankme_peak(cfg_of(model), model, include_init=False),
                          rows=slice(None), bands=slice(1, -1),      # (0.2, 2.4) to couple
                          vlim=None, n_bands=C3_BANDS[0], ratio=C3_BANDS[1],
                          orient=C3_LAYOUT_CROP[2], side=C3_SIDE, cmap=C3_CMAP, fs=C3_FS, fit_style=C3_FIT,
                          title=get_model_label(model))
grid_show()
caption('Band α of the final stream (after_final_norm, centered) from initialisation to the '
        'RankMe peak — the warmup and entropy-seeking phases only, and the middle 20 of the '
        '22 bands (the narrowest head band and the steep top band both crowd out the rest). '
        'Dotted verticals = ranks 11 and 100, the default α fit window; the side strip is '
        'dotted / solid / dashed across the warmup, entropy-seeking and compression phases. '
        'Rank bands across, '
        'pretraining tokens down, read like text. Bright = steep band, dark = flat. The peak '
        'is taken as the RankMe maximum after the initialisation crash, with the crash in the '
        'first third of training (a global minimum lands at the END for nanochat-d12, whose '
        'late decline drops below its own init crash).')

#FIGURE C3
# log-piecewise rankme showing how the rankme moves like a wave until entropy peak. heatmap vertical axis time horizontal axis spectrum

# %%
grid_start(ncols=C3_LAYOUT[1], figsize=C3_LAYOUT[0], pad=C3_PAD,
           savegroup='Band alpha map', savedir='full training, after_final_norm',
           title='Band alpha map — full training, after_final_norm')
for model in SPEC6:
    _lib.plot_alpha_bands(model, cfg_of(model), include_init=False, pct=(2,98), rows=slice(None), bands=slice(1,None),
                          vlim=None, n_bands=C3_BANDS[0], ratio=C3_BANDS[1],   # (0.2,2.4): couple
                          orient=C3_LAYOUT[2], side=C3_SIDE, cmap=C3_CMAP, fs=C3_FS, fit_style=C3_FIT,
                          title=get_model_label(model))
grid_show()
caption('The same map over all of training, after_final_norm. Tall on purpose: the late '
        'checkpoints are linearly spaced in steps and so pack into the last decade of the '
        'log-token axis, and squashing the panel would make them unreadable. Read against '
        'the truncated figure above, the rows past the peak are the compression phase. '
        'Colour scale per panel (vlim=None), so read structure within a model, not levels '
        'between them — each panel carries its own colourbar.')

# %%
grid_start(ncols=C3_LAYOUT[1], figsize=C3_LAYOUT[0], pad=C3_PAD,
           savegroup='Band alpha map', savedir='full training, before_final_norm',
           title='Band alpha map — full training, before_final_norm')
for model in SPEC6:
    _lib.plot_alpha_bands(model, cfg_of(model), hook=HK.BFN_AC, rows=slice(None),
                          bands=slice(1,None), vlim=None,                      # (0.2,2.4): couple
                          n_bands=C3_BANDS[0], ratio=C3_BANDS[1], include_init=False,
                          orient=C3_LAYOUT[2], side=C3_SIDE, cmap=C3_CMAP, fs=C3_FS, fit_style=C3_FIT,
                          title=get_model_label(model))
grid_show()
caption('The same full-training map one hookpoint earlier: before_final_norm, the residual '
        'stream as it leaves the last block, without the final normalisation. Same bands, '
        'same colour scale, so it reads directly against the figure above — what the final '
        'norm does to the band structure is the difference between the two.')

# %%
# The residual at the exact midpoint of the stack: with an even number of blocks that is the
# input to block L/2, i.e. what the first half wrote and the second half reads.
grid_start(ncols=C3_LAYOUT[1], figsize=C3_LAYOUT[0], pad=C3_PAD,
           savegroup='Band alpha map', savedir='full training, mid-stack residual',
           title='Band alpha map — full training, mid-stack residual')
for model in SPEC6:
    _lib.plot_alpha_bands(model, cfg_of(model),
                          hook=(f'blk{n_blocks[model] // 2}.attn.in', 'acts_centered'),
                          rows=slice(None), bands=slice(1,None), vlim=None,
                          n_bands=C3_BANDS[0], ratio=C3_BANDS[1], include_init=False,
                          orient=C3_LAYOUT[2], side=C3_SIDE, cmap=C3_CMAP, fs=C3_FS,
                          fit_style=C3_FIT, title=get_model_label(model))
grid_show()
caption('The same map at the midpoint of the residual stack — the input to block L/2, which '
        'is what the first half has written and the second half reads. Read against the two '
        'final-stream maps above: the band structure here is what the back half inherits. '
        'Note the RankMe strips: the pythias\' mid-stack stream is at single-digit RankMe by '
        'the end (the sink direction dominates the raw residual, before the final norm '
        'rescales it), where OLMo-2 stays in the hundreds.')

# %% [markdown]
# ### The same three maps without any banding: the spectrum itself
# The α maps above summarise each band by one fitted slope. These three plot the eigenvalues
# directly — one column PER EIGENVALUE, colour = λ on a log intensity scale — so no banding
# choice is baked in and the wave is a wave rather than a stripe. Everything else is
# unchanged: log rank across, log tokens down, cells on real edges, `upto` / `rows` / `ranks`
# / `vlim` the same knobs, same palette, the same dotted verticals at ranks 11 and 100, and
# the same RankMe side strip with the trough/peak horizontals.
# Grey cells are non-positive eigenvalues, which a log scale cannot place (there is exactly
# one, pythia-6.9b's last eigenvalue at step 32).
#
# **Read these knowing what is NOT yet corrected.** They show ABSOLUTE eigenvalues, so a model
# whose trace moves bodily paints that as a whole-row brightness change on top of the shape
# change — OLMo-2's trace drops ~2 orders of magnitude and Pythia's climbs ~2, which is most
# of what those panels show. Removing it needs a reference (trace? median? mid-range?) and
# each choice answers a different question; that is the next step, not this one.

# %%
# grid_start(ncols=C3_LAYOUT_CROP[1], figsize=C3_LAYOUT_CROP[0], pad=C3_PAD,
#            title='Spectrum map — warmup to the entropy peak')
# for model in SPEC6:
#     _lib.plot_spectrum_map(model, cfg_of(model), upto=_lib.rankme_peak(cfg_of(model), model, include_init=False)+2,
#                            rows=slice(None), ranks=slice(3,-100), vlim=None,   # (lo, hi) to couple
#                            pct=(2, 100), orient=C3_LAYOUT_CROP[2], side=C3_SIDE,
#                            cmap=C3_CMAP, fs=C3_FS, fit_style=C3_FIT, title=get_model_label(model))
# grid_show()
# caption('The raw centered spectrum of the final stream (after_final_norm) from initialisation '
#         'to the RankMe peak, one column per eigenvalue. Colour is λ itself, log-scaled, '
#         'autoscaled per panel from its 2nd percentile up to its maximum. Clipping the bottom '
#         'is what gives the figure its contrast: a few early checkpoints sit 2–3 decades below '
#         'everything else and otherwise push the bulk of the data into the top third of the '
#         'colourmap. The top is deliberately NOT clipped — the head is few eigenvalues but a '
#         'wide slice of a log-rank axis, so any ceiling flattens it. Against the banded α map '
#         'of the '
#         'same range: there the wave is a bright stripe of steep slope, here it is the shape '
#         'of the surface.')

# %%
# grid_start(ncols=C3_LAYOUT[1], figsize=C3_LAYOUT[0], pad=C3_PAD,
#            title='Spectrum map — full training, after_final_norm')
# for model in SPEC6:
#     _lib.plot_spectrum_map(model, cfg_of(model), rows=slice(None), ranks=slice(2,None),
#                            vlim=None, pct=(2, 100),                   # (lo, hi) to couple
#                            orient=C3_LAYOUT[2], side=C3_SIDE, cmap=C3_CMAP, fs=C3_FS, fit_style=C3_FIT,
#                            title=get_model_label(model))
# grid_show()
# caption('All of training, after_final_norm. The whole-row brightness drift is the trace '
#         'moving, not the shape changing — Pythia and nanochat brighten, both OLMo-2 scales '
#         'darken hard through the middle of training and partly recover. That confound is '
#         'exactly what a shift-invariant version would remove.')

# %%
# grid_start(ncols=C3_LAYOUT[1], figsize=C3_LAYOUT[0], pad=C3_PAD,
#            title='Spectrum map — full training, before_final_norm')
# for model in SPEC6:
#     _lib.plot_spectrum_map(model, cfg_of(model), hook=HK.BFN_AC, rows=slice(None),
#                            ranks=slice(None), vlim=None, pct=(2, 100),   # (lo, hi) to couple
#                            orient=C3_LAYOUT[2], side=C3_SIDE, cmap=C3_CMAP, fs=C3_FS, fit_style=C3_FIT,
#                            title=get_model_label(model))
# grid_show()
# caption('The same, one hookpoint earlier: before_final_norm, without the final normalisation. '
#         'The final norm is a per-token rescale, so the difference between this and the figure '
#         'above is where it changes the spectrum by more than an overall factor.')

# %% [markdown]
# ### The same span as plain spectra
# The maps above as ordinary log-log spectra, Pythia 1B, after_final_norm centered: every
# checkpoint from the 3rd up to just short of the RankMe peak, thinned onto a x2 token ladder
# so the lines are evenly spaced in log time. The trough is the middle of the run where RankMe
# sits below 10% of d_model, not its argmin — the bottom of the crash is flat and its argmin
# jitters.

# %%
SPEC_MODEL = 'pythia-1b-deduped'
grid_start(ncols=1, figsize=(10, 7), savegroup='Spectrum landmarks', savedir='overlaid',
           title='Pythia 1B — spectrum at the phase landmarks')
_lib.plot_landmark_spectra(SPEC_MODEL, cfg_of(SPEC_MODEL), hook=HK.AFN_AC,
                           normalise=False, drop_tail=10, keep=slice(None),
                           min_ratio=2.0,   # min token gap between drawn checkpoints
                           cmap='viridis',
                           title=get_model_label(SPEC_MODEL))
grid_show()
caption('Absolute centered eigenvalues of Pythia 1B\'s final stream, overlaid, coloured early '
        '(dark) to late (bright). Checkpoints run from the 3rd to just short of the RankMe '
        'peak, thinned onto a x2 token ladder so the spacing is even in log time. Bottom 10 '
        'ranks cut — that is the cliff, not shape.')

# %%
_N_LM = len(_lib.phase_landmarks(cfg_of(SPEC_MODEL), SPEC_MODEL))
grid_start(ncols=3, figsize=(18, 13), sharey=True,
           savegroup='Spectrum landmarks', savedir='per checkpoint, rest greyed',
           title='Pythia 1B — spectrum per checkpoint, rest greyed')
for i in range(_N_LM):
    _lib.plot_landmark_spectra(SPEC_MODEL, cfg_of(SPEC_MODEL), hook=HK.AFN_AC,
                               normalise=False, drop_tail=10, min_ratio=2.0,
                               cmap='viridis', highlight=i, title=None)
grid_show()
caption('The same nine spectra, one per panel, each drawn against the other eight in grey. '
        'Shared axes, so the highlighted line moves through a fixed frame: the head climbs '
        'monotonically while the tail drops and then recovers.')

# %% [markdown]
# ## 3. The decomposition: what carries the compression, per family
#
# **Question.** Is the final-stream RankMe trajectory formed by independent layer outputs, or
# emergently through layer interaction — and which term of the decomposition carries the
# post-peak decline?
#
# **Finding (7 models).** Emergent, and the carrier splits by architecture. The exact
# decomposition ΔS = χ + quality + interference (black line = the depth-differential
# log-RankMe: S at before_final_norm MINUS S of the stream entering blk0, per checkpoint)
# shows:
#
# - the decline is QUALITY-carried (red) in pythia-410m/1b/6.9b AND in nanochat-d12 (which
#   has OLMo-like sequential wiring and different data/tokenizer but, like Pythia, no
#   write-output normalisation — the norm-placement association holds out of family);
#
# - both OLMo-2 scales are INTERFERENCE-carried (purple), with quality recovering;
#
# - χ never carries the decline in any model.
#
# Honest edges: pythia-160m is MIXED (quality −3.9 but interference also −2.9, no
# zero-crossing) — the clean signature may emerge with scale; and the dominant quality block
# varies (blk3 at 1b/6.9b, blk5 at 410m, split between blk3 and the final block at
# 160m/nanochat).

# %% [markdown]
# ### Before the three terms: attention vs MLP
#
# **Question.** Of everything the blocks write into the residual stream, how much is
# attention and how much is the MLP — in energy, and in rank entropy?
#
# **Finding.** In energy the MLP dominates at every checkpoint of every model — attention's
# share never reaches 0.5 anywhere. It rises to a mid-training maximum (0.34–0.42 in the
# pythias, 0.26 in nanochat-d12) and then falls to 0.06–0.12 by the end. Both OLMo-2 scales
# start at a 50/50 tie, which is an artifact rather than a finding: OLMo-2 normalises each
# sub-block output, pinning every write's trace to d_model at init. They end at 0.21–0.23.
#
# In rank entropy the MLP sub-step carries the bulk of the negative ΔS: OLMo-2 1B mlp −1.36
# against attn +0.26 (total −1.10), OLMo-2 7B mlp −0.74 against attn −0.47 (total −1.22),
# nanochat-d12 mlp −2.59 against attn −0.09 (total −2.68). Only OLMo-2 7B has attention
# contributing a comparable share.
#
# **Why Pythia has no ΔS split.** Pythia's attention and MLP run in PARALLEL off the same
# block input, so there is no intermediate stream between them and no sub-step to score: the
# ledger's attention ΔS is identically zero and the whole block change lands on the MLP slot.
# The Pythia panels therefore show the block total, hatched, and the split figures below are
# restricted to the sequential families (OLMo-2, nanochat-d12), where
# ΔS_attn + ΔS_mlp = ΔS_block exactly.

# %%
_lib.plot_sub_bars(LEDGER_MODELS, cfg_of, n_blocks, seq=SEQ)
caption('Both currencies at the last checkpoint, all 7 models. Left: each family\'s share of '
        'the summed uncentered write trace, Σ over blocks (a share among the writes only — '
        'write traces do not sum to the residual trace, the cross terms are excluded). '
        'Right: the summed sub-step ΔS, attention against MLP. Pythia is parallel, so its '
        'sub-steps do not exist and the bar is the block total (hatched); for the sequential '
        'models the two colored bars sum to exactly that same block total.')

# %%
grid_start(ncols=2, savegroup='Attention vs MLP', savedir='share of write energy',
           title='Attention vs MLP — share of write energy')
for model in LEDGER_MODELS:
    _lib.plot_sub_energy(model, cfg_of(model), n_blocks[model], share=True,
                         title=get_model_label(model))
grid_show()
caption('Attention (blue) and MLP (orange) shares of the summed uncentered write trace over '
        'training, one panel per model; dotted line = 50/50. Attention peaks mid-training '
        '(0.34–0.42 in the pythias, 0.26 in nanochat-d12) and then gives up share for the '
        'rest of training. The OLMo-2 panels open at exactly 50/50 because OLMo-2 normalises '
        'each sub-block output, fixing every write\'s trace at d_model at init; the departure '
        'from 50/50 is what training adds.')

# %%
grid_start(ncols=2, savegroup='Attention vs MLP', savedir='write energy',
           title='Attention vs MLP — write energy')
for model in LEDGER_MODELS:
    _lib.plot_sub_energy(model, cfg_of(model), n_blocks[model], title=get_model_label(model))
grid_show()
caption('The unnormalized version of the figure above: summed uncentered write trace for '
        'attention and for the MLP, log-log. The vertical gap between the two curves is the '
        'share; the level is what the share hides. Pythia and nanochat-d12 writes GROW '
        'through training (nanochat by ~8 orders of magnitude — it norms neither write), '
        'while both OLMo-2 scales COLLAPSE by 1.5–2.5 orders before a late partial recovery. '
        'Within every model the two curves stay roughly parallel: the share moves slowly '
        'against a level that moves fast.')

# %%
grid_start(ncols=2, savegroup='Attention vs MLP', savedir='rank-entropy split',
           title='Attention vs MLP — rank-entropy split')
for model in SEQ:
    _lib.plot_sub_ds(model, cfg_of(model), n_blocks[model], title=get_model_label(model))
_lib.plot_sub_ds_lines(SEQ, cfg_of, n_blocks, title='all three, overlaid')
grid_show()
caption('Summed sub-step ΔS for the three sequential models: attention (blue) and MLP '
        '(orange) sign-stacked, black = their sum, which is the same block-summed ΔS the '
        'ledger stacks below decompose into χ/quality/interference. Bottom right: the same '
        'six series overlaid as lines (solid = attn, dashed = mlp). The MLP sub-step carries '
        'the end-of-training decline in all three; attention ends positive in OLMo-2 1B and '
        'negative in OLMo-2 7B. Pythia is absent by construction — parallel sub-blocks have '
        'no intermediate stream to measure a sub-step against.')

# %% [markdown]
# #### Curiosity: Pythia under a pretended write order — NOT A MEASUREMENT
#
# Pythia has no post-attention stream, so the two figures below invent one. The write that
# goes "first" is charged its own 2-component ledger against the block input (χ + quality =
# S(mix) − S(in) — everything attributable before the second write exists); its interference
# term does not exist to be charged, because that would need S(in + first), a tensor only a
# sequential model forms. The second write takes the residue, so the pair still sums to the
# measured block ΔS and the black total stays real. Only the SPLIT is invented.
#
# Two reasons not to believe the split. **(1) The orders disagree wildly.** Pythia 1B ends at
# attn −0.03 / mlp −1.50 read attention-first, and mlp −5.08 / attn +3.55 read MLP-first — the
# same block, the same measured total −1.52, an attribution that moves by ~5 nats depending on
# a fiction. **(2) Where truth exists, the construction misses it.** The dashed blue line on
# the sequential panels is the measured attention sub-step: attention-first predicts +1.04 at
# OLMo-2 1B against a true +0.26, and +1.28 at OLMo-2 7B against a true −0.47 (wrong sign).
# That is the dropped interference term, and OLMo-2 is exactly the family whose decline the
# interference term carries (§3 header) — so this is the worst case, not a fluke.

# %%
grid_start(ncols=2, savegroup='Attention vs MLP', savedir='pretended order, attn first',
           title='Attention vs MLP — pretended order, attn first')
for model in LEDGER_MODELS:
    _lib.plot_sub_ds(model, cfg_of(model), n_blocks[model], first='attn', truth=True,
                     title=get_model_label(model))
grid_show()
caption('Counterfactual ΔS split with attention pretended to be written first, all 7 models. '
        'Attention (blue) carries χ + quality of its own 2-component mix against the block '
        'input; the MLP (orange) takes the residue, which absorbs every cross term. Black = '
        'the measured block ΔS, unchanged and real. Dashed blue on the OLMo-2 and nanochat '
        'panels = the MEASURED attention sub-step, which those models genuinely have: the gap '
        'to the blue fill is what dropping the interference term costs.')

# %%
grid_start(ncols=2, savegroup='Attention vs MLP', savedir='pretended order, mlp first',
           title='Attention vs MLP — pretended order, mlp first')
for model in LEDGER_MODELS:
    _lib.plot_sub_ds(model, cfg_of(model), n_blocks[model], first='mlp', truth=True,
                     title=get_model_label(model))
grid_show()
caption('The same construction with the order reversed — the MLP charged its own χ + quality, '
        'attention taking the residue. Against the figure above, on identical data and with '
        'an identical black total: this is the order-dependence of a split that the parallel '
        'architecture leaves unidentifiable. For the sequential models the reversed order is '
        'wrong by construction (they really are attention-first), and the dashed measured '
        'attention line shows how far it lands.')

# %%
grid_start(ncols=3, title='Ledger contributions')
for model in SPEC6:
    _lib.plot_ledger_stack(model, cfg_of(model), n_blocks[model],
                           title=get_model_label(model))
grid_show()
caption('The decomposition summed over blocks, one panel per model: ΔS = χ (green) '
        '+ quality (red) + interference (purple); black = the total, the '
        'depth-differential log RankMe (stack output entropy MINUS input entropy per '
        'checkpoint — negative means output below input, not negative entropy). '
        'Positive parts stack up from zero, negative down. Dotted gray = the embedding '
        'stream\'s own entropy, the base of the black difference. The 7-model split '
        'and its edge cases: section header.')

#FIGURE B3

# %%
# Thesis rendering of the same stack: prose term names, one shared legend, no figure title.
LEDGER_TERMS = (('chi', 'overlap', 'tab:green'),
                ('quality', 'block-intrinsic entropy', 'tab:red'),
                ('interference', 'interference', 'tab:purple'))
LS_FS, LS_FIG = 15, (12.5, 6.4)

def ledger_panel(ax, model, fs=LS_FS, emb=True, xlabel=True, ylabel=True, total_lw=2.5,
                 legend=None):
    """One model's block-summed ledger stack on `ax`. legend: fontsize, for standalone use —
    in the grid the legend is shared and drawn once under the whole figure."""
    plt.sca(ax)
    src, L = cfg_of(model), n_blocks[model]
    steps = _lib.get_ys(src, model, ('blk0', 'block_ledger'), 'chi')[1]
    xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
    keep = xs > 0
    stack = [(lab, np.asarray(np.sum([_lib.get_ys(src, model, (f'blk{l}', 'block_ledger'), t)[0]
                                      for l in range(L)], 0), float)[keep], c)
             for t, lab, c in LEDGER_TERMS]
    gx, tot = _lib.signed_stack(xs[keep], stack)
    ax.plot(gx, tot, 'k', lw=total_lw, label=r'$\Delta S$ total')
    if emb:
        es, esteps = _lib.get_ys(src, model, ('blk0.attn.in', 'acts_centered'), 'matrix_entropy')
        exs = np.asarray(_lib.get_xs_tokens(model, esteps), float)
        ek = exs > 0
        ax.plot(exs[ek], np.asarray(es, float)[ek], color='0.4', ls=':', lw=1.5,
                label=r'$S_0$ (embedding)')
    ax.set_xscale('log')
    ax.set_title(get_model_label(model), fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if ylabel: ax.set_ylabel(r'$\Delta S$', fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)
    if legend: ax.legend(fontsize=legend)

def ledger_stack_fig(models, ncols=3, figsize=LS_FIG, fs=LS_FS, legend_ncol=5,
                     panel_legend=False, savegroup='Ledger contributions', savedir='thesis'):
    """Grid of ledger_panel with one horizontal legend under the whole figure; also writes
    each panel out on its own. Outer labels only — inner panels stay clean."""
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, model in enumerate(models):
        ledger_panel(axes.flat[i], model, fs=fs, xlabel=i >= len(models) - ncols,
                     ylabel=i % ncols == 0)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=legend_ncol, fontsize=fs,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.show()
    if savedir:
        from pathlib import Path
        outdir = Path('analysis/figures') / _lib._safe_name(savegroup) / _lib._safe_name(savedir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / '_grid.pdf', bbox_inches='tight')
        panel = (figsize[0] / ncols, figsize[1] / nrows)
        for i, model in enumerate(models):
            sf, sax = plt.subplots(figsize=panel)
            ledger_panel(sax, model, fs=fs, legend=fs - 4 if panel_legend else None)
            sf.tight_layout()
            sf.savefig(outdir / f'{i}_{_lib._safe_name(model, "_")}.pdf', bbox_inches='tight')
            plt.close(sf)

# %%
ledger_stack_fig(SPEC6)
caption('The decomposition summed over blocks, one panel per model: ΔS = overlap (green) + '
        'block-intrinsic entropy (red) + interference (purple); black = the total, the '
        'depth-differential log RankMe. Positive parts stack up from zero, negative down. '
        'Dotted gray = S₀, the embedding stream\'s own entropy, the base the black difference '
        'is measured from. Same data as the figure above, laid out for the thesis.')

# %% [markdown]
# ### Layer decomposition of the quality term
# The "carrying blocks" row of the ledger claim, shown rather than asserted: one line per block's own
# quality contribution over training. Numbers table: dig_findings "Per-block quality
# decomposition".

# %%
def term_per_block(models, term='quality', cfg_fn=None, suptitle=None):
    from matplotlib.cm import ScalarMappable
    cfg_fn = cfg_fn or cfg_of
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, models):
        L = n_blocks[model]
        for l in range(L):
            ys, steps = _lib.get_ys(cfg_fn(model), model, (f'blk{l}', 'block_ledger'), term)
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            keep = xs > 0
            ax.plot(xs[keep], np.asarray(ys, float)[keep], lw=1.5,
                    color=plt.get_cmap('viridis')(l / (L - 1)))
        ax.axhline(0, color='gray', lw=0.8)
        ax.set(xscale='log', xlabel='tokens', title=get_model_label(model))
        plt.colorbar(ScalarMappable(cmap='viridis'), ax=ax, pad=0.01,
                     ticks=[]).set_label(f'block (0 → {L - 1})', fontsize=8)
    for ax in axes.flat[len(models):]:
        ax.axis('off')
    axes[0, 0].set_ylabel(f'per-block {term} term')
    axes[1, 0].set_ylabel(f'per-block {term} term')
    fig.suptitle(suptitle or f'{term} ledger term per block')
    plt.tight_layout(); plt.show()

term_per_block(LEDGER_MODELS, 'quality')
caption('One line per block\'s own quality contribution (colorbar = block index); the §3 '
        'stacks are the sums of these lines. At pythia-1b/6.9b one early block (blk3) '
        'carries essentially the whole collapse; at 410m it is blk5; at 160m and nanochat '
        'it splits between an early and the FINAL block. In OLMo-2 no block dominates. '
        'Full table: dig_findings.')

# %% [markdown]
# ### Per-block ΔS: is a block's entropy-change sign a fixed property of the block?
#
# **Question.** Is a block's add/remove role a fixed property of the block (RQ1c.0)?
# (These figures also show the depth-disproportionality of the change directly — that is
# H1.2's question.)
#
# **Finding.** Not fixed: comparing packed with padded last-token geometry, pythia's
# final layers flip from ADDING entropy to REMOVING it. The flip is the evidence for the conditional-final-layers interpretation
# (final_report §2.3): a layer's entropy effect is conditional on its input and on what
# the output must become, not an intrinsic role. NB the attribution here is compositional
# (the exact decomposition), not counterfactual — the counterfactual program is plan.md
# item 2.

# %%
term_per_block(LEDGER_MODELS, 'delta_s', suptitle='Per-block ΔS — packed (all-token)')
caption('Each block\'s total ΔS_k over training (colorbar = block index), packed geometry, '
        'all 7 models. The §3 stacks are the sums of these lines, term-split; here the net '
        'per-block change is shown directly.')
#FIGURE B1
# but I can't figure out why olmo has significant early layer bump here.

# %%
PADDED_BS = 'block_representations_samples_padded'
term_per_block(ALL4, 'delta_s', cfg_fn=lambda m: PADDED_BS,
               suptitle='Per-block ΔS — padded (last-token)')
caption('The same per-block ΔS_k in padded last-token geometry (the 4 padded models). '
        'Against the packed figure above: pythia\'s final layers flip sign — the same '
        'layers add entropy packed and remove it padded (final_report §2.3).')

# %% [markdown]
# ## 4. Pythia's sink write (repo codename: the rogue write)
#
# **Question.** What is the single write that carries Pythia's quality-term collapse (§3),
# and what does its dominant direction do?
#
# **Definition.** The sink direction = the top eigenvector v₁ of the CENTERED covariance of
# blk3.mlp's output (final ckpt) — the axis of maximal variance of that write. It earns a
# name because it alone carries 94–99% of the write's variance (that is what "rank ≈ 1
# write" means). "Projection onto it" = (x − μ)·v₁ per token row.
#
# **Finding (position-split analysis, Jul 13).** The direction is a SINK direction with two
# carriers per 511-token window:
#
# - POSITION 0, unconditionally — signed proj +1657 on arbitrary mid-text content tokens,
#   510/512 windows (the packed-stream form of the attention-sinks papers' "BOS" slot, which
#   is positional — our mix contains no BOS at all);
#
# - the FIRST NEWLINE of the window (+2135; 478/486 spiking newlines are the window's first,
#   median position 25). The other 96% of newlines are bulk-ordinary (−9.5 vs bulk −7) — the
#   earlier headline "mean |proj| 98 on newlines" was that 4%/96% mixture.
#
# Variance shares: first-newline slots 60.4%, pos-0 slots 38.7%, all remaining 249k rows
# 0.9%. The write collapses to rank ≈ 1 at the RankMe peak, grows enormous relative to its
# own incoming stream (a block-local ratio; definitional numbers in dig_findings), and the
# late blocks increasingly write against it (signed trace → −0.65; pos-0 output norm 70.7 <
# bulk 131 — the spike is largely cancelled by the end). OLMo-2 shows none of this
# (top-eigval share ~0.01 vs 0.99). Preimage check: the input direction that best predicts
# the score is UNALIGNED with the raw '\n' embedding (cos −0.013) — the spike is constructed
# by the MLP from processed features, not echoed from the embedding.
#
# **Scope.** Correlational and final-checkpoint: this locates WHERE the direction activates,
# not what it is for. Follow-ups below: an explicit <|endoftext|> at position 0 does NOT
# absorb the first-newline slot (both slots are structural), and the depth profile shows
# deposit at blk3 → unchanged ride through blk12 → scrubbed by the late blocks.

# %%
for model in ('pythia-1b-deduped',):
    half = n_blocks[model] // 2
    outs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'acts_centered'), f'mlp {l}')
            for l in range(half)]
    grid_start(ncols=2, savegroup='Sink write trajectory', savedir=get_model_label(model),
               title=f'Sink write trajectory — {get_model_label(model)}')
    plot_group('rankme', outs, [model], color_palette='gradient', ylog=True, title='write RankMe (centered)')
    plot_group('trace', outs, [model], color_palette='gradient', ylog=True, title='write trace')
    grid_show()
    caption(f'RankMe (left) and trace (right) of the centered mlp-output covariance for the '
            f'first half of the writes, {get_model_label(model)} (light→dark = deeper; the '
            f'late half is omitted — late writes are legitimately large). Exactly one early '
            f'write departs from the family: blk3 collapses to rank ≈ 1 at the stream '
            f'RankMe peak while its trace keeps growing.')

#FIGURE B2.0

# %%
for model in ('pythia-1b-deduped',):
    half = n_blocks[model] // 2
    outs = [(BLOCK_SAMPLES, (f'blk{l}.attn.out', 'acts_centered'), f'attn {l}')
            for l in range(half)]
    grid_start(ncols=2, savegroup='Sink write trajectory', savedir=get_model_label(model),
               title=f'Sink write trajectory — {get_model_label(model)}')
    plot_group('rankme', outs, [model], color_palette='gradient', ylog=True, title='write RankMe (centered)')
    plot_group('trace', outs, [model], color_palette='gradient', ylog=True, title='write trace')
    grid_show()
    caption(f'RankMe (left) and trace (right) of the centered attn-output covariance for the '
            f'first half of the writes, {get_model_label(model)} (light→dark = deeper; the '
            f'late half is omitted — late writes are legitimately large).')

#FIGURE B2.0

# %%
# The OLMo-2 control for the pythia figures above. Shared by the figures below: the write
# sources and the embedding-output reference (blk0.attn.in = the stream they are added into),
# thick dotted and held out of the depth gradient.
write_srcs = lambda subs, blocks: [(BLOCK_SAMPLES, (f'blk{l}.{sub}.out', 'acts_centered'),
                                    f'{sub} {l}') for l in blocks for sub in subs]
EMB_SRC = (BLOCK_SAMPLES, ('blk0.attn.in', 'acts_centered'), 'embedding out')
emb_style = lambda n: dict(src_colors={n: '0.25'}, src_ls={n: ':'}, src_lw={n: 4.0},
                           key_didx=[n])

# %%
for model in ('OLMo-2-0425-1B',):
    half = n_blocks[model] // 2
    for sub in ('mlp', 'attn'):
        srcs = [*write_srcs((sub,), range(half)), EMB_SRC]
        style = emb_style(len(srcs) - 1)
        grid_start(ncols=3, savegroup='Sink write trajectory',
                   savedir=f'{get_model_label(model)} — {sub}',
                   title=f'Sink write trajectory ({sub}) — {get_model_label(model)}')
        plot_group('rankme', srcs, [model], color_palette='gradient', ylog=True,
                   title='write RankMe (centered)', **style)
        plot_group('trace', srcs, [model], color_palette='gradient', ylog=True,
                   title='write trace', **style)
        _lib.plot_sub_ds_depth(model, BLOCK_SAMPLES, range(half), subs=(sub,), lines=False,
                               title=f'total ΔS over these {half} writes')
        grid_show()
    caption(f'RankMe, trace and summed sub-step ΔS of the centered write covariance over the '
            f'first half of {get_model_label(model)}\'s writes — mlp (top) and attn (bottom); '
            f'the late half is omitted, late writes are legitimately large. Dotted: the '
            f'embedding output, as a level in the first two panels and as its change from the '
            f'first checkpoint in the third. No early write leaves the family here — the '
            f'contrast with pythia\'s blk3 above.')

    srcs = [*write_srcs(('attn', 'mlp'), range(half)), EMB_SRC]
    style = emb_style(len(srcs) - 1) | dict(ls_exceptions={'attn': ':'})
    grid_start(ncols=3, savegroup='Sink write trajectory',
               savedir=f'{get_model_label(model)} — both sublayers',
               title=f'Sink write trajectory, both sublayers — {get_model_label(model)}')
    plot_group('rankme', srcs, [model], color_palette='gradient', ylog=True,
               title='write RankMe (centered)', **style)
    plot_group('trace', srcs, [model], color_palette='gradient', ylog=True,
               title='write trace', **style)
    _lib.plot_sub_ds_depth(model, BLOCK_SAMPLES, range(half), lines=False,
                           title=f'total ΔS over these {2 * half} writes')
    grid_show()
    caption('The two figures above on one axis: colour is the block, attn dotted, mlp solid.')

# %%
# The second version drops block 1 (code blk0), whose two writes carry most of the range, and
# takes its output stream as the reference in place of the embedding.
for model in ('OLMo-2-0425-1B',):
    half = n_blocks[model] // 2
    for start in (0, 1):
        grid_start(ncols=3, sharey=True, savegroup='Sink write trajectory',
                   savedir=f'{get_model_label(model)} — sublayer ΔS from blk{start}',
                   title=f'Per-write sub-step ΔS — {get_model_label(model)}'
                         + (f', from blk{start}' if start else ''))
        for subs in (('attn',), ('mlp',), ('attn', 'mlp')):
            _lib.plot_sub_ds_depth(model, BLOCK_SAMPLES, range(start, half), subs=subs,
                                   title=' + '.join(subs) + ' writes')
        grid_show()
    caption('The same blocks\' ΔS write by write, shared y; black is the sum over the panel\'s '
            'writes. Top: all of them, dotted = S(embedding stream). Bottom: blk0 dropped — off '
            'the scale of the rest — with the stream it writes out (blk1\'s input) as the '
            'dotted reference instead. Both dotted lines are drawn against their own first '
            'checkpoint: a stream the writes are added into has no ΔS of its own.')

# %%
# rogue_hist, step by step: (1) load the RAW per-token activations of blk3.mlp's output
# (block_rogue_id run, final ckpt; rows = the 262k packed-mix token positions, in order);
# (2) project each centered row onto the write's TOP eigendirection -> one scalar per token
# ("how strongly this token activates the sink direction", i.e. (x − μ)·v1 with v1 the top
# centered eigenvector of the write — defined above); (3) recover each row's token id
# from the packed mix (deterministic order); (4) histogram |projection|, newline-bearing
# tokens vs all others. The separated red tail IS the token-attribution evidence.
# (First run downloads the pythia tokenizer -> the HF warning; afterwards it's cached.)
import torch
from utils.accessor import DataAccessor
from transformers import AutoTokenizer

def rogue_hist(model, step=143000):
    acc = DataAccessor(f'data/inferences/block_rogue_id/{model}/step{step}.pt')
    v = acc['blk3.mlp.out'].acts
    score = (v.samples.float() - v.mean.float()) @ v.eigvecs_centered[:, 0].float()
    ids = torch.load('data/mixes/pile_30M_512.pt')[:, :511].reshape(-1)[:len(score)]
    tok = AutoTokenizer.from_pretrained(f'EleutherAI/{model}')
    uniq = torch.unique(ids)
    toks = tok.convert_ids_to_tokens(uniq.tolist())     # one vectorized call, not 50k decodes
    nl = {int(i) for i, t in zip(uniq, toks) if 'Ċ' in t or '\n' in t}   # 'Ċ' = byte-level \n
    is_nl = torch.tensor([int(t) in nl for t in ids])
    is_p0 = (torch.arange(len(score)) % 511) == 0                        # window-start rows
    plt.figure(figsize=(7, 4))
    for m, lbl, c in ((~is_nl & ~is_p0, 'other tokens', 'tab:gray'),
                      (is_nl & ~is_p0, 'newline tokens', 'tab:red'),
                      (is_p0, 'position 0 (any token)', 'tab:blue')):
        plt.hist(score[m].abs().numpy(), bins=120, log=True, alpha=0.6, color=c, label=lbl)
    plt.xlabel('|projection onto sink direction|'); plt.ylabel('token count (log)')
    plt.title(f'Who carries the sink direction — {get_model_label(model)}'); plt.legend(); plt.show()

rogue_hist('pythia-1b-deduped')
caption('Histogram of each token row\'s |projection onto the sink direction| (log count '
        'axis): window-start rows (blue, any token), newline tokens elsewhere (red), all '
        'others (gray). Both sink slots separate from the bulk, and the red tail is almost '
        'entirely each window\'s FIRST newline — 96% of newlines sit in the bulk.')

#FIGURE B2.2
# but the labelling is not clear. plot does not even say what layer this is. Purpose of this plot is to confirm what's driving blk3 magnitude

# %%
# Systematic version: mean |projection| per token TYPE (all tokens, n>=20, no prior grouping) —
# the top of the ranking, newline variants highlighted. The complementary preimage result
# (text, docs/dig_findings.md): the best linear INPUT predictor of the score is unaligned with
# the raw newline embedding (cos −0.013) — the spike is constructed by the MLP, not echoed
# from the token embedding.
def rogue_ranking(model, step=143000, top=20):
    acc = DataAccessor(f'data/inferences/block_rogue_id/{model}/step{step}.pt')
    v = acc['blk3.mlp.out'].acts
    score = (v.samples.float() - v.mean.float()) @ v.eigvecs_centered[:, 0].float()
    ids = torch.load('data/mixes/pile_30M_512.pt')[:, :511].reshape(-1)[:len(score)]
    tok = AutoTokenizer.from_pretrained(f'EleutherAI/{model}')
    uniq, inv, counts = torch.unique(ids, return_inverse=True, return_counts=True)
    s_abs = score.abs()
    mean_abs = torch.zeros(len(uniq)).index_add_(0, inv, s_abs) / counts
    var = (torch.zeros(len(uniq)).index_add_(0, inv, s_abs.square()) / counts - mean_abs.square())
    keep = counts >= 20
    order = torch.argsort(mean_abs.masked_fill(~keep, -1), descending=True)[:top].tolist()
    decoded = tok.batch_decode([[int(uniq[i])] for i in order])
    vals = [float(mean_abs[i]) for i in order]
    ns = [int(counts[i]) for i in order]
    ci = [1.96 * float(var[i].clamp(min=0).sqrt()) / n ** 0.5 for i, n in zip(order, ns)]
    colors = ['tab:red' if '\n' in t else 'tab:gray' for t in decoded]   # test the RAW token: repr() escapes \n
    ypos = range(top)[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(ypos, vals, color=colors)
    lo = [max(v - c, 0) for v, c in zip(vals, ci)]                       # tick at the CI lower edge
    plt.scatter(lo, ypos, marker='|', color='k', s=80, zorder=3)
    plt.yticks(ypos, [f'{t!r}  (n={c})' for t, c in zip(decoded, ns)], fontsize=7)
    plt.xlabel('mean |projection onto sink direction|')
    plt.title(f'Token types ranked by sink-direction activation — {get_model_label(model)}\n(red = newline-bearing; small n = noisy)')
    plt.show()

rogue_ranking('pythia-1b-deduped')
caption('Mean |projection| per token TYPE (n ≥ 20), top 20; red = newline-bearing; black '
        'tick = lower edge of the 95% CI (a far-left tick = noise-dominated mean). Only the '
        'newline rows are statistically solid. This per-type view structurally hides the '
        'position-0 slot (spread over ~500 types) — see the histogram above.')

#FIGURE B2.1
# this should go before B2.2. Story goes: identify token -> compare with first pos

# %% [markdown]
# ### Follow-ups: is the newline slot just a missing resting token? and where does the
# ### spike live over depth?
# Two targeted re-collections (pythia-1b, final ckpt; `oneoff_scripts/rogue_id_followups.py`):
# - **EOT-twin:** same windows with an explicit `<|endoftext|>` prepended at position 0 —
#   the attention-sinks papers give models a resting token per prompt, our packed stream has
#   none, so maybe the first newline only spikes as a stand-in. It does NOT: with the EOT
#   present (and itself taking the position-0 slot at 1583), the window's first newline
#   still spikes at 2178 (original run: 2132). Both slots are structural.
# - **Mid-stack depth profile:** the same projection measured on the STREAM at several
#   depths. The sink-slot spike is deposited at blk3 and rides the stream essentially
#   unchanged through blk12 (~99% of those rows' centered norm), halves at blk15, and is
#   almost fully removed by before_final_norm. Ordinary tokens DO carry sink-direction
#   content mid-stack — ~27% of their centered row norm at blk4, decaying monotonically to
#   ~3.6% at the final stream — confirming the mid-stack route by which sink-direction variance can
#   reach ordinary (e.g. padded last) tokens even though the final stream looks clean.

# %%
def rogue_depth_profile():
    r = torch.load('data/results/rogue_id_followups.pt')
    ms = r['midstack']
    leaves = ['blk3.mlp.out (write)', 'blk4.attn.in', 'blk6.attn.in', 'blk9.attn.in',
              'blk12.attn.in', 'blk15.attn.in', 'before_final_norm']
    slots = (('pos0', 'tab:blue'), ('first_nl', 'tab:red'), ('other_nl', 'tab:orange'),
             ('bulk', 'tab:gray'))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('The sink spike over depth — pythia-1b, final ckpt (midstack run)')
    for slot, c in slots:
        absv = [ms['write'][slot]] + [ms[l]['abs'][slot] for l in leaves[1:]]
        frac = [ms[l]['frac'][slot] for l in leaves[1:]]
        axes[0].plot(range(len(leaves)), absv, marker='o', lw=2, color=c, label=slot)
        axes[1].plot(range(1, len(leaves)), frac, marker='o', lw=2, color=c, label=slot)
    axes[0].set(yscale='log', ylabel='mean |projection onto sink direction|',
                title='absolute (log)')
    axes[1].set(ylabel='mean |projection| / centered row norm', title='fraction of the row')
    for ax in axes:
        ax.set_xticks(range(len(leaves)))
        ax.set_xticklabels(leaves, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

rogue_depth_profile()
caption('Mean |projection onto the sink direction| per token class over depth: absolute '
        '(left, log) and as a fraction of row norm (right). The blk3 write deposits the '
        'spike; the slots carry it near full strength through blk12; the late blocks scrub '
        'it. Ordinary tokens are not clean mid-stack (~27% at blk4 → ~4% at the end). EOT '
        'control (not plotted): an explicit <|endoftext|> does not absorb the first-newline '
        'slot.')

# FIGURE B2.?
# I don't really know what the purpose is of this plot and don't understand what it's showing, if I understood I would maybe put it in,
# but as of now it makes zero sense to me. I don't understand the purpose. purpose should be one short sentence.

# %%
def cancellation_headmass(model):
    L, cfg = n_blocks[model], cfg_of(model)
    mats, steps = _lib.get_ys(cfg, model, ('', 'block_block_coupling'), 'signed_trace')
    leaves = _lib.get_y(cfg, model, ('', 'block_block_coupling'), 'leaves', steps[0])
    xs, last = _lib.get_xs_tokens(model, steps), leaves.index(f'blk{L - 1}.mlp.out')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'Cancellation / head-mass / concentration — {get_model_label(model)}')
    for l in (1, 3, 5, L // 2):
        i = leaves.index(f'blk{l}.mlp.out')
        axes[0].plot(xs, [m[i, last] for m in mats], marker='o', lw=2, label=f'mlp {l} ~ mlp {L-1}')
    axes[0].set(xscale='log', title='signed trace vs last write', xlabel='tokens'); axes[0].legend(fontsize=8)
    for l in (0, L // 2, L - 1):
        cs, ss = _lib.get_ys(cfg, model, (f'blk{l}.mlp.out', 'eigendirection_attrib'), 'contrib')
        axes[1].plot(_lib.get_xs_tokens(model, ss), [float(np.abs(c[:32]).sum() / np.abs(c).sum()) for c in cs],
                     marker='o', lw=2, label=f'mlp {l}')
    axes[1].set(xscale='log', ylim=(0, 1), title='head-mass (top-32 share)', xlabel='tokens'); axes[1].legend(fontsize=8)
    share = [_lib.get_y(cfg, model, (f'blk{l}.mlp.out', 'acts_centered'), 'eigenspectrum', None)[0]
             for l in range(L)]
    axes[2].bar(range(L), share, color={'pythia': 'tab:red', 'OLMo': 'tab:blue'}.get(
        next((f for f in ('pythia', 'OLMo') if f in model), ''), 'tab:green'))
    axes[2].set(ylim=(0, 1), title='top-eigval share per mlp write (final)', xlabel='block')
    plt.show()
    caption(f'Left: signed-trace coupling of early/mid writes against the LAST write '
            f'(negative = cancellation). Middle: share of each write\'s attribution mass in '
            f'its top 32 directions. Right: top-eigenvalue share per write at the final '
            f'checkpoint — Pythia\'s bar near 1.0 at blk3 IS the sink write; OLMo-2 has '
            f'none above ~0.1.')

for model in ALL6:
    cancellation_headmass(model)

# FIGURE B2.?
# third subplot (top eig-val share) looks cool but genuinely idk how to understand it.
# what does "top-eigval share" mean? furthermore, maybe we have too many plots on this point. it feels unbalanced
# also, middle plot I don't understand what it is at all? another thing where maybe if I understand what question it's answering it would be okay.
# subplot 1 seems to make sense, it's answering "what layers are being deleted by final layer?"
# however, I would like to note that the mlp layer choices are pretty random. Skipping layers is a critical oversight.


# %% [markdown]
# ## 5. OLMo-2's mechanism: distributed aligned reinforcement
#
# **Question.** OLMo-2's compression is interference-carried (§3) and it has no sink write
# (§4's final panel). What structure among its writes carries the compression instead?
#
# **Finding.** The late writes are mutually ALIGNED (positive signed trace between neighbors
# — reinforcement, the opposite of Pythia's cancellation), diffuse in token space (top-eigval
# share ~10%, no dominant token class), and they carry the interference-driven compression.

# %%
# The write-ensemble structure that carries §5's claim: the mlp×mlp signed-trace coupling.
# (The mlp×attn view — a separate memory-management observation — lives in §8.)
from analysis.experiments_lib import submatrix

def coupling_grid(models, rows='mlp', cols='mlp', step=None):
    nrows = (len(models) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.5 * nrows), squeeze=False)
    fig.suptitle(f'{rows} × {cols} signed-trace coupling, final checkpoint')
    for ax, model in zip(axes.flat, models):
        bb = lambda y: _lib.get_y(cfg_of(model), model, ('', 'block_block_coupling'), y, step)
        M, rl, cl = submatrix(bb('signed_trace'), bb('leaves'), rows=rows, cols=cols)
        if rows == cols:
            M = M - np.diag(np.diag(M))
        v = np.abs(M).max()
        im = ax.imshow(M, cmap='coolwarm', vmin=-v, vmax=v)
        ax.set(title=get_model_label(model), xlabel=f'{cols} block', ylabel=f'{rows} block')
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()


# %%

coupling_grid(ALL6)
caption('Signed trace between every pair of mlp writes, final checkpoint (red = '
        'reinforcing, blue = cancelling; diagonal removed), across the 6-model selection '
        '(top row: nanochat-d12, pythia-410m). Both OLMo-2 panels show the '
        'late-block red square — every late write aligned with every other; the Pythia '
        'panels instead show a blue band through the sink-write block. This is '
        '"distributed aligned reinforcement", operationally.')
#FIGURE B4.0

# %% [markdown]
# ### Does the alignment actually carry the compression?
# The chain this figure closes: §3's ledger says OLMo-2's rank-entropy decline sits in the
# INTERFERENCE term — the part of ΔS attributable to cross-covariances between DIFFERENT
# blocks' writes (negative interference = the cross-terms pile variance onto shared
# directions, concentrating the stream spectrum). §5's heatmap says the late writes are
# mutually aligned. Neither alone shows the aligned writes are the thing compressing. The
# test, in two halves:
# - **When (top two panels, shared token axis):** "late-write alignment" = the mean signed
#   trace over all pairs of distinct mlp writes in the deeper half of the stack (one number
#   per checkpoint, from the coupling matrix above). If alignment causes the interference
#   compression, its rise and the interference term's fall must begin together.
# - **Which blocks (right panel):** per block, x = that block's mean signed-trace coupling to
#   the late-half writes, y = that block's own interference ledger term; both averaged over
#   the final third of checkpoints, where the compression phase is in full swing. If aligned
#   writes carry it, the blocks with the most negative interference must be exactly the
#   most-coupled (rightmost) ones — points should run diagonally down-right for OLMo-2.
#   For Pythia the prediction is the opposite corner: its negative interference sits at
#   blocks ANTI-aligned with the ensemble (the cancellation of the sink write).

# %%
def alignment_vs_interference(model):
    L, cfg = n_blocks[model], cfg_of(model)
    mats, csteps = _lib.get_ys(cfg, model, ('', 'block_block_coupling'), 'signed_trace')
    leaves = _lib.get_y(cfg, model, ('', 'block_block_coupling'), 'leaves', csteps[0])
    cxs = _lib.get_xs_tokens(model, csteps)
    idx = [leaves.index(f'blk{k}.mlp.out') for k in range(L // 2, L)]
    late = [float(np.mean([m[i, j] for i in idx for j in idx if i != j])) for m in mats]
    intf = {k: np.asarray(_lib.get_ys(cfg, model, (f'blk{k}', 'block_ledger'), 'interference')[0])
            for k in range(L)}
    isteps = _lib.get_ys(cfg, model, ('blk0', 'block_ledger'), 'interference')[1]
    ixs = _lib.get_xs_tokens(model, isteps)
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=(1.3, 1), hspace=0.15, wspace=0.25)
    fig.suptitle(f'Does the late-write alignment carry the compression? — {get_model_label(model)}')
    top = fig.add_subplot(gs[0, 0])
    bot = fig.add_subplot(gs[1, 0], sharex=top)
    top.plot(cxs, late, color='tab:red', lw=2)
    top.axhline(0, color='gray', lw=0.8)
    top.set_ylabel('late-write\nalignment', fontsize=9)
    top.set(xscale='log', title='when: alignment (top) vs interference (bottom), same token axis')
    plt.setp(top.get_xticklabels(), visible=False)
    bot.plot(ixs, np.sum(list(intf.values()), axis=0), color='tab:purple', lw=2)
    bot.axhline(0, color='gray', lw=0.8)
    bot.set(xscale='log', xlabel='tokens')
    bot.set_ylabel('Σ interference\n(rank entropy)', fontsize=9)
    sc = fig.add_subplot(gs[:, 1])
    seg = slice(2 * len(isteps) // 3, None)                 # final third of checkpoints
    cseg = mats[2 * len(mats) // 3:]
    for k in range(L):
        i = leaves.index(f'blk{k}.mlp.out')
        coup = float(np.mean([m[i, j] for m in cseg for j in idx if j != i]))
        sc.scatter(coup, float(np.mean(intf[k][seg])), s=45,
                   color=plt.get_cmap('viridis')(k / (L - 1)))
        sc.annotate(str(k), (coup, float(np.mean(intf[k][seg]))), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    sc.axhline(0, color='gray', lw=0.8)
    sc.axvline(0, color='gray', lw=0.8)
    sc.set(xlabel='coupling to the late writes',
           ylabel='own interference term',
           title='which blocks: coupling vs interference\n(each avg over final third of ckpts)')
    plt.show()
    caption(f'Left pair (shared token axis), {get_model_label(model)}: late-write alignment '
            f'(red) above the total interference term (purple). Right: per block, coupling '
            f'to the late writes vs its own interference term, averaged over the final '
            f'third of checkpoints. '
            + ('The onsets coincide and the compressing blocks are exactly the '
               'most-coupled ones (down-right): the aligned ensemble IS the compressor.'
               if 'OLMo' in model else
               'No positive alignment appears, and the negative-interference blocks sit at '
               'NEGATIVE coupling: Pythia\'s interference is the sink-write cancellation, '
               'not reinforcement.' if 'pythia' in model else
               'nanochat-d12 is quality-carried (§3), so this panel is the negative case: '
               'no aligned late ensemble to carry an interference-driven compression.'))

for model in ALL6:
    alignment_vs_interference(model)

#FIGURE B4.?
# Very confused about any plot in here. Doesn't make sense. I don't think I will use it.

# %% [markdown]
# ## 6. The minimal model (RQ3)
#
# **Question.** What is the minimal model that shows the phase trajectory — and does the
# minimal model share the LLM's decomposition signature?
#
# This section answers it in four steps: (6.1) the single-layer reproduction and its
# controls; (6.2) the conditions under which the pattern survives depth; (6.3) the
# decomposition signature in the deep toy; (6.4) the architecture knobs. Full studies:
# docs/toy.md (§3 single-layer mechanism, §4 depth) + analysis/toy_multilayer.ipynb
# (every depth ablation).
#
# **Reading note.** The Li-facing toy figures (6.1, 6.2) use UNCENTERED RankMe, matching Li
# et al's definition on raw features; 6.3 and 6.4 use the centered LLM-side pipeline — their
# axes say so.
#
# ### 6.1 Single-layer reproduction: the phases need a fourth, unstated condition
#
# **Finding.** Li et al's Fig 4 reproduces (phases, panel B/C geometry, all four negative
# controls) through this repo's unmodified metric pipeline — but only under their
# CONSTRUCTED init, readable off the gray t=0 markers in their figure (frequent-class
# features clustered on separated directions with aligned W columns; both rare classes
# coincident at the origin, split only by a tiny jitter δ). iid init at any scale fails.
#
# **Mechanism (docs/toy.md §3).** The co-traveling rare pair's off-diagonal
# covariance CANCELS the frequent classes' tilt; the fork collapses that cancellation,
# reopening the eigengap — a kick scaling with the squared fork amplitude (which sets the
# fork→decline lag: zero when the pair starts at the exact origin, as in the paper; ~25
# steps at a −0.25 offset). The kick always happens; whether it PRINTS is a visibility
# condition — the spectrum must have saturated before the fork, which the constructed init
# controls via δ. The toy's compression is a transient (RankMe back to ~2.0 by step ~3000).

# %%
lifig = 'reference/li et al reference tex/figures/Fig4_top.png'
if os.path.exists(lifig):
    plt.figure(figsize=(12, 5)); plt.imshow(mpimg.imread(lifig)); plt.axis('off')
    plt.title('Li et al, Fig 4 (original)'); plt.show()
    caption('The ORIGINAL Li et al Fig 4, top row, for side-by-side comparison: (A) model '
            'schematic, (B) classifier weight rows W_i in 2D over training, (C) feature '
            'vectors f(x), (D) RankMe and top singular values — phases drawn as dotted '
            '(warmup) / solid (entropy-seeking) / dashed (compression) segments.')
#FIGURE F0

# %%
TOY_CAPS = {
    'fig4_single.png':
        'Reproduction of Li et al Fig 4 B–D under their constructed init: classifier rows '
        '(B) and features (C), one color+marker per class, line style = phase; RankMe with '
        'the top two eigenvalues (D). The rare pair co-travels and splits late onto the '
        'frequent classes\' axes; RankMe peaks at the fork and genuinely declines. With '
        'the pair starting at the origin, decline onset coincides with the fork, as in the '
        'paper.',
    'fig_controls.png':
        'The paper\'s negative controls, one panel each; the first panel is the '
        'skew+bottleneck+CE reference run to 3000 steps (dotted vertical = the paper\'s '
        '300-step window) — it alone shows the post-peak decline, itself a transient. The '
        'CE controls are monotone after warmup; mse_skew\'s small dip is addressed below.',
}

for f, cap_txt in TOY_CAPS.items():
    p = f'toy/figures/{f}'
    if os.path.exists(p):
        plt.figure(figsize=(11, 6)); plt.imshow(mpimg.imread(p)); plt.axis('off'); plt.show()
        caption(cap_txt)
#FIGURE F1
#FIGURE F2

# %% [markdown]
# **On mse_skew's small dip.** mse_skew cannot fit its task: with logits = FW at d = 2,
# MSE's global optimum is the best rank-2 approximation of the one-hot targets, which
# keeps the two frequent classes and maps every rare row to zero (per-rare-row MSE 0.25,
# rare weight columns → 0 — appendix B4 shows both, in the sweep seeds and this control
# run alike).
#
# Abandoning the rare classes is therefore the optimum, not a failed optimization: no
# fork is possible, hence no compression — the loss removes the mechanism, which is what
# makes this a principled negative control. (CE solves the same task at the same rank
# because it needs argmax margins, not one-hot values.) The dip itself is not
# interpreted; observationally it aligns with the rare weight columns' peak, the onset
# of their decay.

# %% [markdown]
# ### 6.2 Depth: when does the phase pattern survive it?
#
# **Question.** Our first multi-layer toys (the 32-class task) never showed the phase
# pattern at any depth or init. Is depth what kills it?
#
# **Finding (docs/toy.md §4; every ablation in analysis/toy_multilayer.ipynb).** No —
# depth is innocent. The pattern shows iff three conditions hold: (T) the compression event
# lands after spectrum saturation, (K) the kick is large enough, and (I) the map from the
# features to the measured representation starts near the identity. The figure shows the
# three decisive positives; note the third panel's toggle, where ONLY the fork delay δ
# changes. The 32-class task fails (T) by construction — its 26 rare-class forks smear
# across the entropy-seeking rise (task structure, not depth; toy.md §4) — which is
# why the early multi-layer attempts came up empty.
#
# The residual connection is NOT the requirement: a PLAIN network (He et al's term — no skip
# connections) with identity-initialised layers shows the full pattern (middle panel). The
# residual merely provides near-identity transmission by default, which is how LLMs get (I).

# %%
def _toy_rm(tag):
    r = np.load(f'data/results/toy_{tag}/results_toy-{tag}.npy', allow_pickle=True).item()
    steps = sorted(r)
    return steps, [r[s]['before_final_norm']['acts_uncentered']['rankme'] for s in steps]

def _toy_fork(tag):
    import torch
    import torch.nn.functional as Fn
    t = torch.load(f'data/results/toy/trajectories_{tag}.pt')
    ts, W = t['steps'].tolist(), t['W']
    cos = [float(Fn.cosine_similarity(W[i][:, -2], W[i][:, -1], dim=0)) for i in range(len(ts))]
    shared = next((ts[i] for i in range(len(ts)) if cos[i] > 0.97), 0)
    return next((ts[j] for j in range(len(ts)) if ts[j] >= shared and cos[j] < 0.9), None)

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
for ax, (title, tags) in zip(axes, (
        ('residual, 6 blocks, small-init writes\n(single_deep, 3 seeds)',
         [f'single_d6_bs_s{s}' for s in (0, 1, 2)]),
        ('NO residual, identity-init blocks\n(3 seeds)',
         [f'single_d6_id_plain_s{s}' for s in (0, 1, 2)]),
        ('skewed 6-class bottleneck task:\nfork mid-rise (δ=1e-3) vs after saturation (δ=1e-8)',
         ['skewpair_deep_s0', 'skewpair_deep_d8_s0']))):
    for tag in tags:
        st, rm = _toy_rm(tag)
        ln, = ax.plot(st, rm, lw=1.6,
                      label=('δ=1e-3 (masked)' if tag == 'skewpair_deep_s0'
                             else 'δ=1e-8 (pattern)' if tag == 'skewpair_deep_d8_s0' else None))
        fk = _toy_fork(tag)
        if fk:
            ax.axvline(fk, color=ln.get_color(), lw=0.9, ls=':')
    ax.set(xscale='log', xlabel='step', title=title)
    if ax is axes[2]:
        ax.legend(fontsize=8)
axes[0].set_ylabel('RankMe (uncentered features)')
fig.suptitle('Li et al\'s phase curve through multi-layer stacks — decline onset = the rare-pair fork (dotted)')
plt.tight_layout(); plt.show()
caption('The three decisive positives (uncentered RankMe; dotted = the measured fork). '
        'Left: 6-block residual stack on the constructed init — full pattern, decline at '
        'the fork, 3/3 seeds. Middle: a plain stack with identity-init layers — the same, '
        '3/3. Right: only the fork delay δ moves — a mid-rise fork is masked, a '
        'post-saturation fork shows. Open items: toy.md §4.3.')

# %% [markdown]
# ### 6.3 Does the minimal deep model share the LLM's decomposition signature?
#
# **Question.** Independently of whether the phase pattern shows, does a deep toy's
# decomposition look like a transformer family's (§3)?
#
# **Finding.** Yes — Pythia's. The 6-layer residual toy on the 32-class task is
# QUALITY-carried with χ stable (6/6 no-norm seeds), in a model with no attention, no
# norms, no tokens. Interference-carried compression (OLMo's mode) has never appeared in
# any toy run — the toy program's largest known gap.
#
# Trajectory and signature are independent facts: this task never shows the phase pattern
# (it fails 6.2's timing condition), yet its decomposition signature is clean. The right
# panel is the representativeness guard: no single write dominates the energy shares, so
# the summed terms describe the whole stack, not one layer.

# %%
def toy_ledger(tag='multi_residual_nonlinear', depth=6):
    r = np.load(f'data/results/toy_{tag}/results_toy-{tag}.npy', allow_pickle=True).item()
    steps = sorted(r)
    fig, (lx, wx) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for term, c in (('chi', 'tab:green'), ('quality', 'tab:red'),
                    ('interference', 'tab:purple'), ('delta_s', 'k')):
        ys = [sum(float(r[s][f'blk{k}']['block_ledger'][term]) for k in range(depth)) for s in steps]
        lx.plot(steps, ys, lw=2.2 if term == 'delta_s' else 1.8, color=c,
                label='ΔS total' if term == 'delta_s' else term)
    lx.axhline(0, color='gray', lw=0.8)
    lx.set(xscale='log', xlabel='training step', ylabel='sum over blocks (rank entropy)',
           title='decomposition terms (centered)')
    lx.legend(fontsize=8)
    w = np.stack([r[s]['']['overlap_chi']['w'] for s in steps])
    for k in range(w.shape[1]):
        wx.plot(steps, w[:, k], lw=1, color=plt.get_cmap('viridis')(k / max(w.shape[1] - 1, 1)))
    wx.plot(steps, w.max(1), color='crimson', lw=2, label=r'$\max_k w_k$')
    wx.set(xscale='log', xlabel='training step', ylabel='trace weight $w_k$', ylim=(0, 1.05),
           title='write energy shares (representativeness guard)')
    wx.legend(fontsize=8)
    fig.suptitle('The deep toy\'s decomposition signature — 6-layer residual, 32-class task')
    plt.tight_layout(); plt.show()

toy_ledger()
caption('Left: the decomposition summed over the 6 layers of the deep toy (centered): '
        'quality carries the decline while χ stays stable — Pythia\'s §3 signature in a '
        'model with no attention, no norms, no tokens. Right: write energy shares — no '
        'single write dominates, so the sums are representative. This task shows no phase '
        'pattern (6.2); the claim is about the terms, not the trajectory.')

# %% [markdown]
# ### 6.4 Architecture knobs: write-norm removes the Pythia mechanism
#
# **Question.** Which architectural knob removes the sink-write/quality mechanism — and does
# any knob produce OLMo's interference mode?
#
# **Finding.** WRITE-norm (normalising the sublayer output before the residual add — the
# OLMo-2 reordered-norm analog) suppresses the sink-write mechanism completely: 6/6 seeds,
# both wirings, min write RankMe 5.6–7.5 vs 1.7–2.6 un-normed. Read-norm also suppresses
# the toy's sink write — a toy/LLM discrepancy, since real Pythia has read-norm and a sink
# write anyway. No knob in the 24-run grid produces interference-carried compression, and
# parallel-vs-sequential wiring selects nothing.
#
# **Scope (⚠️).** The grid runs on the 32-class task, which cannot show the phase pattern
# (6.2), and predates the pattern conditions — the suppression and decomposition-carrier
# results stand (centered quantities, trajectory-independent), but any trajectory-shape
# reading is void until the planned rerun under the constructed-timing setup (docs/plan.md).

# %%
p = 'toy/figures/fig_arch_grid.png'
if os.path.exists(p):
    plt.figure(figsize=(11, 6)); plt.imshow(mpimg.imread(p)); plt.axis('off'); plt.show()
    caption('The architecture-knob grid (color = norm cell; solid/dashed = wiring; 3 seeds '
            'each). Left: centered final-stream RankMe. Middle: summed quality term. Right: '
            'each run\'s minimum write RankMe vs its energy share — write-norm (green) '
            'abolishes the collapsed write in all its runs. Trajectory-shape readings void '
            'pending the rerun (section note).')

# %% [markdown]
# ## 7. RQ2: task geometry beyond the general population — PARKED
#
# **Question.** Does maths/numerical data have more shared geometric structure than
# quotes/memorised data, beyond the general population's?
#
# **Status (parked Jul 13; do not cite).** The documented excess-mass results mixed
# ACTIVATION and GRADIENT quantities (acts-only recomputation inverts the quotes/memorized
# ordering and weakens math), the ref pairing of the stored gen_vs_ref results is
# unverified, and several comparisons straddled incompatible geometries (packed all-token
# vs padded last-token covariances are different measured objects). The contaminated
# figures that used to render here are removed; the redo plan is docs/plan.md item 10, the
# full diagnosis is final_report §4, and the machinery (methods + notebook) is
# docs/rq2.md and analysis/rq2_results.ipynb.
#
# One sub-result is a clean negative independent of the contamination: H2.1
# (gradient–activation eigenstructure alignment indicates structured data) is refuted in
# all three variants tested.

# %% [markdown]
# ## 8. Side findings
#
# Real results that are not load-bearing for RQ1–RQ3. Two former subsections are removed
# rather than shown-with-warnings:
#
# - **Write means (mean_frac):** the claimed family split was retracted as a
#   misinterpretation (the panels contradicted its cross-model form) — nothing
#   mean_frac-related is citable until re-derived from scratch (final_report appendix).
#
# - **Drift (raw consecutive-checkpoint CKA):** confounded by uneven checkpoint spacing,
#   and the stored metric additionally subsamples inconsistently at late checkpoints; the
#   gap-corrected re-derivation is planned (docs/plan.md item 7).

# %%
# mlp×attn coupling (moved out of §5 — a separate observation from the write-ensemble claim)
coupling_grid(ALL6, rows='attn', cols='mlp')
caption('Signed trace between mlp writes (rows) and attention writes (cols), final '
        'checkpoint. The near-diagonal negatives show MLPs partially consuming their '
        'neighbouring attention outputs — the "memory management" pattern of the GELU-4L '
        'literature. Side observation, independent of the family mechanisms.')
#FIGURE G

# %% [markdown]
# ### Compression valleys: the depth-axis view (one object, three shadows)
#
# **Question.** Does the sink write's training-time story have a depth-axis counterpart —
# the "compression valleys" of the sinks literature (arXiv:2510.06477)?
#
# **Finding.** Yes, and it splits by family exactly as §3–§5 predict: stream RankMe per
# block at the final checkpoint crashes to rank ~2 right after blk3's write enters and
# recovers via the late-block cancellation in Pythia and nanochat (no write-norm); OLMo-2
# (write-norm) has no valley at all. The massive activation (feature axis), the compression
# valley (depth axis), and the compression phase (training axis) are the same sink
# direction seen three ways.

# %%
VALLEY = {('block_representations_samples', 'pythia-1b-deduped'): 16,
          ('block_representations_samples', 'pythia-6.9b-deduped'): 32,
          ('block_representations_samples', 'OLMo-2-0425-1B'): 16,
          ('block_representations_samples', 'OLMo-2-1124-7B'): 32,
          ('nanochat_samples', 'nanochat-d12'): 12}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for (cfg, model), L in VALLEY.items():
    res, steps = load_results(cfg, model)
    leaves = [f'blk{k}.attn.in' for k in range(L)] + ['before_final_norm']
    fin = res[steps[-1]]
    rank = [fin.get(lf, {}).get('acts_centered', {}).get('rankme', np.nan) for lf in leaves]
    hbits = [fin.get(lf, {}).get('acts_uncentered', {}).get('matrix_entropy', np.nan) / np.log(2)
             for lf in leaves]                       # sinks-paper metric: H(p=σ²/‖X‖²) in bits
    x, ls = np.linspace(0, 1, len(leaves)), '-' if 'OLMo' in model else '--'
    axes[0].semilogy(x, rank, marker='o', lw=2, ls=ls, label=get_model_label(model))
    axes[1].plot(x, hbits, marker='o', lw=2, ls=ls, label=get_model_label(model))
axes[0].set(xlabel='relative depth', ylabel='stream RankMe (centered, log)',
            title='RankMe view')
axes[1].set(xlabel='relative depth', ylabel='matrix-based entropy (bits, uncentered)',
            title='sinks-paper metric: matrix entropy $H(\\sigma_i^2/\\|X\\|_F^2)$')
axes[0].legend(fontsize=8)
fig.suptitle('Compression valleys at the final checkpoint — dashed = no write-norm')
plt.show()
caption('Stream RankMe per block at the final checkpoint (left, centered, log) and the '
        'sinks-paper metric (right, uncentered matrix entropy in bits, for numeric '
        'comparison with their layer profiles). Dashed = families without write-norm: the '
        'valley right after blk3–4; solid = OLMo-2: no valley. Protocol caveat: theirs is '
        'per-example, ours a pooled population covariance — shapes comparable, absolute '
        'values approximate.')

# %% [markdown]
# Appendix material (the dataset-swap control, the toy edge-case maps) lives in its own
# notebook: [showcase_appendix.ipynb](showcase_appendix.ipynb).
