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
# # Animated unembedding singular spectra
#
# The weights-side companion to the experiments_anim split (1: boundary mean spectra,
# 2: eigval spectra, 3: coupling heatmaps, 4: layerwise RankMe, 5: layerwise mean norm,
# 6: entropy-neuron plane). Same engine (`analysis/spectrum_anim.py`), one frame per
# weight checkpoint, but the spectra here come from the unembedding matrix $W_U$ rather
# than from activation covariances.
#
# Data: `data/results/unembedding_spectra.pt` — singular values of $W_U$ per model and
# checkpoint, written by `oneoff_scripts/unembedding_spectra.py` (selective weight
# loading, log checkpoint schedule); `data/results/unembedding_spectra_OLMo-2-0425-1B.pt`
# adds the dense late-training grid for that model. The corrected variants come from
# `data/results/unembedding_variant_spectra.pt`
# (`oneoff_scripts/unembedding_variant_spectra.py`). The static versions of these figures
# are section 3 of `analysis/unembedding.py`.

# %%
import os, sys
import importlib
if os.path.basename(os.getcwd()) == 'analysis':
    os.chdir('..')
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils.model_registry, utils.accessor
importlib.reload(utils.model_registry)   # deps first: reload(_lib) alone re-imports cached modules
importlib.reload(utils.accessor)
from analysis import experiments_lib as _lib
importlib.reload(_lib)
from analysis.experiments_lib import (
    build_hooks, get_ys, get_series_y, panel_palettes, smooth_spectrum, submatrix,
    block_mean_cos, model_name_options, get_model_label, get_xs_tokens,
    YVAR_LABELS, XVAR_FNS)

# %%
# Config this notebook needs (kept out of the generic lib):
SPECTRA = 'data/results/unembedding_spectra.pt'            # sigma(W_U) per model/checkpoint
EXTRA = 'data/results/unembedding_spectra_OLMo-2-0425-1B.pt'   # dense late-training grid
VARIANTS = 'data/results/unembedding_variant_spectra.pt'
ALPHA_WINDOW = (11, 100)                  # the alphaReQ rank window used throughout §3

US = torch.load(SPECTRA, weights_only=False)
if os.path.exists(EXTRA):
    for m, per_step in torch.load(EXTRA, weights_only=False).items():
        US.setdefault(m, {}).update(per_step)
VS = torch.load(VARIANTS, weights_only=False) if os.path.exists(VARIANTS) else {}
MODELS = [m for m in US if US[m]]
COLORS = dict(zip(MODELS, plt.cm.tab10.colors))
print({m: len(US[m]) for m in MODELS})

# %%
# Animated engine (analysis/spectrum_anim.py): inject the data backend as the other anim
# notebooks do. Every panel here is caller-supplied ('series' / 'curve' frames), so the
# backend is only used for the frame labels' token counts.
import analysis.spectrum_anim as sa
importlib.reload(sa)   # pick up engine edits without restarting the kernel
sa.configure(get_series_y=get_series_y, get_ys=get_ys,
             panel_palettes=panel_palettes, smooth_spectrum=smooth_spectrum,
             submatrix=submatrix, block_mean_cos=block_mean_cos,
             model_name_options=model_name_options, YVAR_LABELS=YVAR_LABELS, XVAR_FNS=XVAR_FNS)
animate_spectra = sa.animate_spectra

# %%
from IPython.display import display


def rankme(sv):
    """RankMe of the squared spectrum — the effective number of singular directions."""
    lam = np.asarray(sv, float) ** 2
    p = lam / lam.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def alpha(sv, k0=ALPHA_WINDOW[0], k1=ALPHA_WINDOW[1]):
    """alphaReQ: weighted log-log slope of the sigma^2 decay over ranks [k0, k1)."""
    try:
        return _lib._alpha(np.asarray(sv, float) ** 2, k0=k0, k1=k1)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan


def svals(model, variant=None):
    """(steps, sigma per step). variant=None reads the raw spectra; a variant name reads
    the corrected heads (freq_centered, mean_deflated, freq_mean) from the variants file.
    All-zero checkpoints are dropped — nanochat zero-inits its head, so its step 0 has no
    spectrum to draw and no finite RankMe or alpha."""
    per_step = US[model] if variant is None else VS[model]
    get = (lambda s: per_step[s]) if variant is None else (lambda s: per_step[s][variant])
    live = [(s, get(s).numpy().astype(float)) for s in sorted(per_step)]
    live = [(s, a) for s, a in live if a.max() > 0]
    return [s for s, _ in live], [a for _, a in live]


def share(sv):
    """Energy share sigma_i^2 / sum_j sigma_j^2 — the spectrum's shape with its scale divided out."""
    lam = np.asarray(sv, float) ** 2
    return lam / lam.sum()


def frames_with_ghosts(seq, color):
    """One frame per checkpoint: the checkpoint's own spectrum over grey copies of the
    first and last checkpoints, so growth and reshaping are both visible in place."""
    ghosts = [(seq[0], '0.78', 'first checkpoint'), (seq[-1], '0.45', 'last checkpoint')]
    return [[*ghosts, (a, color, 'this checkpoint')] for a in seq]


def frame_labels(model, steps):
    return [f'step {s}  ({t:.2e} tokens)'
            for s, t in zip(steps, get_xs_tokens(model, list(steps)))]


# %% [markdown]
# ## Raw unembedding spectra
#
# Left: the singular values themselves (absolute scale, log-log). Middle: the same
# spectrum as an energy share $\sigma_i^2/\sum_j\sigma_j^2$, which removes the overall
# growth of the weight norm and leaves only the change of shape. Right: RankMe of
# $\sigma^2$ and the tail slope $\alpha$ over ranks 11–100, with a red line scanning to
# the frame's token count.

# %%
def anim_unembedding(model, save_dir=None, fps=4, figsize=(19.5, 5.4)):
    steps, sv = svals(model)
    toks = [float(t) for t in get_xs_tokens(model, list(steps))]
    sh = [share(a) for a in sv]
    color = COLORS[model]
    panels = [
        ('sigma', None, [model], {
            'kind': 'series', 'steps': steps, 'frames': frames_with_ghosts(sv, color),
            'xlabel': 'rank $i$', 'ylabel': r'$\sigma_i$',
            'title': r'unembedding singular values — absolute'}),
        ('share', None, [model], {
            'kind': 'series', 'steps': steps, 'frames': frames_with_ghosts(sh, color),
            'xlabel': 'rank $i$', 'ylabel': r'$\sigma_i^2 / \sum_j \sigma_j^2$',
            'title': 'spectrum shape — energy share per rank'}),
        ('curves', None, [model], {
            'kind': 'curve', 'scan_x': toks,
            'series': [(toks, [rankme(a) for a in sv], '0.35', 'RankMe', False),
                       (toks, [alpha(a) for a in sv], 'tab:purple',
                        r'$\alpha$ (ranks %d–%d)' % ALPHA_WINDOW, True)],
            'xlabel': 'pretraining tokens', 'ylabel': r'RankMe of $\sigma^2$',
            'twin_ylabel': r'$\alpha$', 'title': 'effective rank and tail slope'})]
    save = f'{save_dir}/Unembedding_spectrum_{model}.mp4' if save_dir else None
    # progress bar on top: with three columns the frame label would sit on the right panel's
    # x-axis label at the bottom
    display(animate_spectra(panels, ncols=3, fps=fps, figsize=figsize, model=model, save=save,
                            frame_labels=frame_labels(model, steps), prog_bar='top',
                            suptitle=f'Unembedding singular spectrum over pretraining — '
                                     f'{get_model_label(model)}'))


# %%
anim_unembedding('pythia-160m-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding('pythia-410m-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding('pythia-1b-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding('pythia-6.9b-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding('OLMo-2-0425-1B', save_dir='analysis/figures/animations')

# %%
anim_unembedding('OLMo-2-1124-7B', save_dir='analysis/figures/animations')

# %%
anim_unembedding('nanochat-d12', save_dir='analysis/figures/animations')

# %% [markdown]
# ## Corrected heads
#
# The same animation for the softmax-invariant corrections of $W_U$, overlaid:
# standard = the raw head; freq_centered = $W - \mathbf{1}(W^\top f)$ with $f$ the
# empirical unigram frequency of the collection mix; mean_deflated = $W(I-\hat\mu
# \hat\mu^\top)$ with $\mu$ the after-final-norm stream mean at that checkpoint;
# freq_mean = both. Left is absolute, right is the energy share.

# %%
VARIANT_COLORS = {'standard': '0.35', 'freq_centered': 'tab:blue',
                  'mean_deflated': 'tab:orange', 'freq_mean': 'tab:green'}
# W(I - mu mu^T) nulls one stream direction by construction, so those spectra end in a
# numerical zero (~1e-5) that would stretch the log axis over 15 decades. Drop it — it is
# the correction's own arithmetic, not the head's. The frequency subtraction acts on the
# vocabulary side and takes no rank, so its spectrum stays full length.
DEFLATED_RANKS = {'standard': 0, 'freq_centered': 0, 'mean_deflated': 1, 'freq_mean': 1}


def anim_unembedding_variants(model, save_dir=None, fps=4, figsize=(14, 5.4)):
    if model not in VS:
        return print(f'[skipped] {model}: no entry in {VARIANTS}')
    per_step = VS[model]
    variants = [v for v in VARIANT_COLORS if any(v in per_step[s] for s in per_step)]
    live, _ = svals(model, 'standard')        # drop the zero-init checkpoints, then the
    steps = [s for s in live                  # ones a correction is missing at (needs the
             if all(per_step[s].get(v) is not None for v in variants)]   # stream mean there)
    if len(steps) < len(live):
        print(f'{model}: dropped {sorted(set(live) - set(steps))} — not every correction stored')
    keep = lambda a, v: a[:len(a) - DEFLATED_RANKS[v]] if DEFLATED_RANKS[v] else a
    absolute = [[(keep(per_step[s][v].numpy().astype(float), v), VARIANT_COLORS[v], v)
                 for v in variants] for s in steps]
    shares = [[(share(a), c, lab) for a, c, lab in f] for f in absolute]
    panels = [
        ('sigma', None, [model], {
            'kind': 'series', 'steps': steps, 'frames': absolute,
            'xlabel': 'rank $i$', 'ylabel': r'$\sigma_i$',
            'title': 'corrected unembeddings — absolute'}),
        ('share', None, [model], {
            'kind': 'series', 'steps': steps, 'frames': shares,
            'xlabel': 'rank $i$', 'ylabel': r'$\sigma_i^2 / \sum_j \sigma_j^2$',
            'title': 'corrected unembeddings — energy share per rank'})]
    save = f'{save_dir}/Unembedding_variants_{model}.mp4' if save_dir else None
    display(animate_spectra(panels, ncols=2, fps=fps, figsize=figsize, model=model, save=save,
                            frame_labels=frame_labels(model, steps), prog_bar='bottom',
                            suptitle=f'Unembedding spectra under softmax-invariant '
                                     f'corrections — {get_model_label(model)}'))


# %%
anim_unembedding_variants('pythia-160m-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_variants('pythia-410m-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_variants('pythia-1b-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_variants('pythia-6.9b-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_variants('OLMo-2-0425-1B', save_dir='analysis/figures/animations')

# %%
anim_unembedding_variants('OLMo-2-1124-7B', save_dir='analysis/figures/animations')

# %%
anim_unembedding_variants('nanochat-d12', save_dir='analysis/figures/animations')

# %% [markdown]
# ## The tail, read from the bottom
#
# The same raw spectra on a linear y-axis pinned to 0–20, so the bulk keeps its scale
# across models and checkpoints. Left: the forward index, zoomed to the last quarter of
# the ranks. Middle: the whole spectrum counted from the *smallest* singular value upward
# on a log axis, so the bottom gets the resolution the forward index spends on the head.
# Both drop the leading `drop_top` singular values. Right: four training curves, each on
# its own scale and coloured to match it — $\alpha$ of the unembedding, $\alpha$ and
# RankMe of the final stream (`after_final_norm`, centered, samples sweep), and what
# zero-ablating the last quarter of blocks does to that RankMe (ablated − baseline, on
# the checkpoints both runs share). Both alphas use the same rank window.

# %%
STREAM_HOOK = 'after_final_norm'   # the stream the unembedding reads
SAMPLES_SRC = {m: 'nanochat_samples' if m == 'nanochat-d12' else 'block_representations_samples'
               for m in MODELS}
DEPTH = {'pythia-160m-deduped': 12, 'pythia-410m-deduped': 24, 'pythia-1b-deduped': 16,
         'pythia-6.9b-deduped': 32, 'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32,
         'nanochat-d12': 12}
# The zero-ablation run whose ablated group is the model's last quarter of blocks
# (the quarter split of analysis/ledger_ablation_layers.py).
LAST_QUARTER = {12: 'ablate_blk9-11', 16: 'ablate_blk12-15',
                24: 'ablate_blk18-23', 32: 'ablate_blk24-31'}


def stream_curve(model, yvar, src=None, kwargs={}):
    """(steps, values) of a final-stream metric; src defaults to the model's samples sweep."""
    ys, steps = get_ys(src or SAMPLES_SRC[model], model, (STREAM_HOOK, 'acts_centered'),
                       yvar, kwargs)
    return (None, None) if ys is None else (list(steps), np.asarray(ys, float))


def stream_alpha(model, window=ALPHA_WINDOW):
    """(tokens, alphaReQ) of the final stream's centered eigenspectrum, in the same rank
    window as the unembedding's alpha."""
    steps, ys = stream_curve(model, 'alpha_window', kwargs={'k0': window[0], 'k1': window[1]})
    return (None, None) if steps is None else (np.asarray(get_xs_tokens(model, steps), float), ys)


def stream_rankme(model):
    """(tokens, RankMe) of the final stream, and (tokens, ablated − baseline) for the run
    with the last quarter of blocks zero-ablated, on the checkpoints the two runs share."""
    steps, base = stream_curve(model, 'rankme')
    if steps is None:
        return (None, None), (None, None)
    xs = np.asarray(get_xs_tokens(model, steps), float)
    asteps, abl = stream_curve(model, 'rankme', src=LAST_QUARTER[DEPTH[model]])
    if asteps is None:
        return (xs, base), (None, None)
    shared = [s for s in asteps if s in set(steps)]
    b, a = dict(zip(steps, base)), dict(zip(asteps, abl))
    return (xs, base), (np.asarray(get_xs_tokens(model, shared), float),
                        np.asarray([a[s] - b[s] for s in shared], float))


def anim_unembedding_tail(model, drop_top=1, save_dir=None, fps=4, figsize=(20.5, 5.4),
                          ylim=(0, 20)):
    """drop_top: how many leading singular values to slice off both spectrum panels."""
    steps, sv = svals(model)
    toks = [float(t) for t in get_xs_tokens(model, list(steps))]
    bulk = [a[drop_top:] for a in sv]
    quarter = 3 * len(bulk[0]) // 4                 # the forward panel shows only the tail quarter
    label = get_model_label(model)
    frames = lambda seq: [[(a, COLORS[model], label)] for a in seq]
    (rx, rme), (dx, drme) = stream_rankme(model)
    sax, say = stream_alpha(model)
    abl = LAST_QUARTER[DEPTH[model]]
    # (x, y, colour, legend label, axis label) — each gets its own scale, in this order
    extra = [(sax, say, 'tab:gray', r'$\alpha$ final stream', r'$\alpha$ of the final stream'),
             (rx, rme, 'tab:blue', 'RankMe final stream', 'RankMe of the final stream'),
             (dx, drme, 'tab:red', 'RankMe change under ablation', f'RankMe({abl}) − RankMe')]
    for xs, _, _, lab, _ in extra:
        if xs is None:
            print(f'{model}: {lab} unavailable — left out')
    extra = [e for e in extra if e[0] is not None]
    series = [(toks, [alpha(a) for a in sv], COLORS[model], r'$\alpha$ unembedding', 0)]
    series += [(xs, ys, c, lab, k + 1) for k, (xs, ys, c, lab, _) in enumerate(extra)]
    panels = [
        ('sigma', None, [model], {
            'kind': 'series', 'steps': steps, 'frames': frames([a[quarter:] for a in bulk]),
            'xlog': False, 'ylog': False, 'ylim': ylim, 'x0': drop_top + 1 + quarter,
            'xlabel': 'rank $i$', 'ylabel': r'$\sigma_i$',
            'title': f'singular values, last quarter of the ranks'}),
        ('reversed', None, [model], {
            'kind': 'series', 'steps': steps, 'frames': frames([a[::-1] for a in bulk]),
            'xlog': True, 'ylog': False, 'ylim': ylim, 'x0': 1,
            'xlabel': r'index from the smallest $\sigma$', 'ylabel': r'$\sigma_i$',
            'title': f'the same spectrum counted from the tail, top {drop_top} dropped'}),
        ('curves', None, [model], {
            'kind': 'curve', 'scan_x': toks, 'series': series, 'color_axes': True,
            'xlabel': 'pretraining tokens',
            'ylabel': r'$\alpha$ of $\sigma^2$(unembedding), ranks %d–%d' % ALPHA_WINDOW,
            'twin_ylabel': [e[4] for e in extra],
            'title': 'tail slope and effective rank'})]
    save = f'{save_dir}/Unembedding_tail_{model}.mp4' if save_dir else None
    display(animate_spectra(panels, ncols=3, fps=fps, figsize=figsize, model=model, save=save,
                            frame_labels=frame_labels(model, steps), prog_bar='top',
                            suptitle=f'Unembedding spectrum tail over pretraining — {label}'))


# %%
anim_unembedding_tail('pythia-160m-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_tail('pythia-410m-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_tail('pythia-1b-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_tail('pythia-6.9b-deduped', save_dir='analysis/figures/animations')

# %%
anim_unembedding_tail('OLMo-2-0425-1B', save_dir='analysis/figures/animations')

# %%
anim_unembedding_tail('OLMo-2-1124-7B', save_dir='analysis/figures/animations')

# %%
anim_unembedding_tail('nanochat-d12', save_dir='analysis/figures/animations')
