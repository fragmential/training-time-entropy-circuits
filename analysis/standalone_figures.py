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
# # The warmup→peak phase, as one figure
#
# A composite for the thesis: the band-α map of Pythia 1B's first phase down the left with its
# RankMe trace beside it, and the raw spectrum at each of a handful of checkpoints stacked down
# the right, each tied by a leader line to the moment on the map it was taken from. Everything
# is read from `analysis/experiments_lib` — this notebook only does layout.

# %%
import os, re, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import LogLocator, NullFormatter
from pathlib import Path
from analysis import experiments_lib as _lib
importlib.reload(_lib)
from analysis.experiments_lib import build_hooks, get_model_label

# 'Pythia 1BD' -> 'Pythia 1B', 'OLMo-2 1B' -> 'OLMo 2 1B': the deduped marker and the family
# hyphen are both noise in a figure
model_label = lambda m: re.sub(r'([MB])D\b', r'\1', get_model_label(m)).replace('OLMo-2', 'OLMo 2')

HK = build_hooks()
MODEL = 'pythia-1b-deduped'
SRC = 'block_representations_samples'
HOOK = HK.AFN_AC

MAP_CMAP, SPEC_CMAP = 'magma', 'viridis'
SPEC_SPAN = (0.0, 0.81)  # viridis slice: stop short of the pale yellow end
BANDS = (10, 1)          # (n bands, head:tail width ratio) — see alpha_band_edges
INK, MUTE, RULE = '0.15', '0.45', '0.80'
MARK = '0.65'            # timing lines: light enough to read over magma's dark cells
FS = 34                  # base font size (scales with figsize, not fixed points)
PANEL_BG = '#F4F4F7'     # seaborn-ish tint for the line panels; the map keeps white


# %%
def _phase_data(model=MODEL, source=SRC, hook=HOOK, bands=slice(1, None), drop_tail=10,
                every=2):
    """Everything the figure draws: the α map cropped to the peak, the RankMe trace, and one
    spectrum per landmark checkpoint."""
    pk = _lib.rankme_peak(source, model, hook)
    xs, e, A = _lib.alpha_bands(source, model, hook, n_bands=BANDS[0], ratio=BANDS[1])
    xs, A = xs[:pk + 1], A[:pk + 1]
    b = range(*bands.indices(A.shape[1]))
    A, e = A[:, b.start:b.stop], e[b.start:b.stop + 1]
    rx, rv = _lib.rankme_series(source, model, hook)
    lm = _lib.phase_landmarks(source, model, hook)
    spec = []
    for _lbl, _row, step, tok in lm:
        v = np.asarray(_lib.get_y(source, model, hook, 'eigvals', step), float)
        spec.append((tok, v[:-drop_tail] if drop_tail else v))
    spec = spec[::every]
    return dict(xs=xs, e=e, A=A, rx=rx[:pk + 1], rv=rv[:pk + 1], spec=spec,
                marks=[t for t, _v in spec])


def _style(ax, spine=RULE, bg=None):
    ax.grid(False)
    if bg:                                   # seaborn-style: tinted panel, white gridlines
        ax.set_facecolor(bg)
        ax.grid(True, which='major', color='white', lw=1.4)
        ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(spine); s.set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=INK, length=4, width=0.9, labelsize=FS - 5,
                   pad=10)          # clear of the axes: log exponents ride high
    ax.tick_params(which='minor', length=0)


def phase_figure(model=MODEL, source=SRC, hook=HOOK, figsize=(42.5, 19.1), dpi=120,
                 vlim=None, pct=(2, 98), save=None):
    """Band-α map + RankMe trace on the left, one spectrum per landmark down the right, joined
    by leader lines at the token position each was taken from."""
    d = _phase_data(model, source, hook)
    n = len(d['spec'])
    cols = plt.get_cmap(SPEC_CMAP)(np.linspace(*SPEC_SPAN, n))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    outer = fig.add_gridspec(1, 2, width_ratios=[0.59, 1.0], wspace=0.11,
                             left=0.065, right=0.955, top=0.94, bottom=0.085)
    left = outer[0].subgridspec(2, 2, width_ratios=[1, 0.63], height_ratios=[1, 0.016],
                               wspace=0.05, hspace=0.25)
    ax_map = fig.add_subplot(left[0, 0])
    ax_rk = fig.add_subplot(left[0, 1], sharey=ax_map)
    ax_cb = fig.add_subplot(left[1, 0])
    last_left = max(i for i in range(n) if i % 2 == 0)

    # --- the map -----------------------------------------------------------------------
    vmin, vmax = vlim if vlim is not None else np.percentile(d['A'], pct)
    # edgecolors='face' covers the hairline seams pcolormesh leaves between cells in a PDF
    im = ax_map.pcolormesh(d['e'], _lib._log_edges(d['xs']), d['A'], cmap=MAP_CMAP,
                           vmin=vmin, vmax=vmax, edgecolors='face', linewidth=0.4)
    ax_map.set(xscale='log', yscale='log')
    ax_map.invert_yaxis()
    ax_map.set_xlabel('Eigenvalue index', fontsize=FS, color=INK, labelpad=16)
    ax_map.set_ylabel('Pretraining tokens', fontsize=FS, color=INK, labelpad=14)
    _style(ax_map)

    # --- the spectra block -------------------------------------------------------------
    # Double-stacked: panel i sits in column i%2, offset half a panel height from its
    # neighbours, so each is 1/3 of the block tall instead of 1/5. Its own gridspec rather
    # than a nested one, so it can run taller than the left side (which loses height to the
    # map's labels) and be centred independently: the block is placed so the MIDDLE panel's
    # centre lands on its own token row, which makes that leader exactly horizontal and the
    # others symmetric about it.
    cell = outer[0, 1].get_position(fig)
    ycen = fig.transFigure.inverted().transform(
        ax_map.transData.transform((ax_map.get_xlim()[0], d['spec'][n // 2][0])))[1]
    half = min(ycen - 0.005, 0.995 - ycen)
    right = fig.add_gridspec(n + 1, 2, left=cell.x0, right=cell.x1,
                             top=ycen + half, bottom=ycen - half,
                             hspace=0.135, wspace=0.05)
    ax_sp = [fig.add_subplot(right[i:i + 2, i % 2]) for i in range(n)]

    # --- RankMe beside it, sharing the token axis --------------------------------------
    ax_rk.plot(d['rv'], d['rx'], color=INK, lw=3.5)
    ax_rk.set_xlabel('RankMe', fontsize=FS, color=INK, labelpad=16)
    ax_rk.tick_params(labelleft=False)
    ax_rk.locator_params(axis='x', nbins=3)
    _style(ax_rk, bg=PANEL_BG)

    cb = fig.colorbar(im, cax=ax_cb, orientation='horizontal')
    cb.set_label(r'$\alpha$', fontsize=FS + 2, color=INK, labelpad=12)
    cb.ax.tick_params(labelsize=FS - 5, colors=MUTE, labelcolor=INK, pad=8)
    cb.outline.set_edgecolor(RULE)

    # --- the spectra, newest at the bottom to match the map's direction ------------------
    lo = min(v.min() for _t, v in d['spec']) * 0.6
    hi = max(v.max() for _t, v in d['spec']) * 1.6
    for i, (ax, (tok, v)) in enumerate(zip(ax_sp, d['spec'])):
        for j, (_t2, v2) in enumerate(d['spec']):                    # context, greyed
            if j != i:
                ax.plot(np.arange(1, len(v2) + 1), v2, color='0.73', lw=2.0, zorder=1)
        ax.plot(np.arange(1, len(v) + 1), v, color=cols[i], lw=5.0, zorder=3)
        ax.set(xscale='log', yscale='log', ylim=(lo, hi))
        ax.xaxis.set_major_locator(LogLocator(numticks=4))
        ax.yaxis.set_major_locator(LogLocator(numticks=3))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.text(0.97, 0.92, r'${\mathbf{' + f'{np.log10(tok):.1f}' + r'}}_{\log_{10}}$', ha='right',
                va='top', transform=ax.transAxes, fontsize=FS - 2, color=cols[i] * 0.75)
        ax.text(0.045, 0.08, 'abcdefgh'[i], transform=ax.transAxes, ha='left', va='bottom',
                fontsize=FS + 8, weight='bold', color=INK)
        _style(ax, bg=PANEL_BG)
        ax.tick_params(left=False, right=False, labelleft=False, labelright=False)
        if i != last_left:                      # one x axis for the whole block, bottom-left
            ax.set_xticklabels([])
            ax.tick_params(bottom=False)
    ax_sp[last_left].set_xlabel('Eigenvalue index', fontsize=FS, color=INK, labelpad=16)

    # --- leaders: token position -> its spectrum -----------------------------------------
    # Anchored on the RankMe strip's right edge, the last thing in the left block, so a leader
    # never crosses a panel it is not pointing at.
    xr = ax_rk.get_xlim()[1]
    for i, (ax, (tok, _v)) in enumerate(zip(ax_sp, d['spec'])):
        for a in (ax_map, ax_rk):                       # when this spectrum was taken
            _lib._mark_lines(a, [tok], 'y', MARK, ls='--', lw=2.6, alpha=1.0)
        ax_rk.plot([xr], [tok], marker='_', ms=10, mew=1.8, color=cols[i],
                   clip_on=False, zorder=6)
        # right-column targets: nudge the landing point off centre so the leader threads the
        # gap between the left panels instead of grazing their edges
        yb = 0.5 + (0.03 if i == 1 else -0.03 if i == 3 else 0.0)
        for zo, al in ((0, 0.85), (12, 0.14)):    # behind, plus a ghost so covered stretches
            fig.add_artist(ConnectionPatch(                       # stay faintly traceable
                xyA=(xr, tok), coordsA=ax_rk.transData,
                xyB=(-0.02, yb), coordsB=ax.transAxes,
                color=INK, lw=1.0, alpha=al, zorder=zo, clip_on=False))

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
        fig.savefig(str(save).replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.show()
    return fig


# %%
_ = phase_figure(save='analysis/figures/standalone/pythia-1b_warmup_to_peak.pdf')


# %% [markdown]
# ## Two models, map beside band alpha
#
# Pythia 1B and OLMo-2 1B, a row each: the band-alpha map as showcase section 2 draws it (time
# across), and beside it alpha fitted inside two fixed rank windows — the head-removed reading
# of the same spectrum. One legend for the line column.

# %%
CMP_MODELS = ('pythia-1b-deduped', 'OLMo-2-0425-1B')
cfg_of = lambda m: 'nanochat_samples' if m == 'nanochat-d12' else SRC
WINDOWS = ((11, 100), (256, 512))        # the rank windows alpha is fitted in
CMP_FS = 15


def alpha_vlim(models, bands=slice(1, None), pct=(2, 98)):
    """One colour range over both models' band alpha, so the shared colourbar means the same
    thing in each map."""
    vals = []
    for m in models:
        _xs, _e, A = _lib.alpha_bands(cfg_of(m), m, HOOK, n_bands=BANDS[0], ratio=BANDS[1],
                                      include_init=False)
        b = range(*bands.indices(A.shape[1]))
        vals.append(A[:, b.start:b.stop].ravel())
    return tuple(np.nanpercentile(np.concatenate(vals), pct))


def phase_compare(models=CMP_MODELS, windows=WINDOWS, figsize=(15.0, 8.6), fs=CMP_FS,
                  map_fs=None, bands=slice(1, None), cmap=MAP_CMAP, cb_y=0.045, cb_h=0.018,
                  save=None):
    """Per model: the band-alpha map (showcase settings, time across) beside its windowed
    alpha. One colour scale over both maps, its bar on the bottom next to the line legend.
    map_fs: the maps carry more ticks, so they take their own font size."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.17,
                             left=0.05, right=0.985, top=0.94, bottom=0.185)
    gl = outer[0].subgridspec(len(models), 1, hspace=0.5)
    gr = outer[1].subgridspec(len(models), 1, hspace=0.5)
    vlim = alpha_vlim(models, bands)
    maps = []
    for i, m in enumerate(models):
        maps.append(fig.add_subplot(gl[i, 0]))
        _lib.plot_alpha_bands(m, cfg_of(m), _ax=maps[-1], hook=HOOK,
                              n_bands=BANDS[0], ratio=BANDS[1], bands=bands, vlim=vlim,
                              cbar=False, include_init=False, orient='time_right',
                              side='tail', cmap=cmap, fs=map_fs or fs, fit_style='bar',
                              title=model_label(m))
        ax = fig.add_subplot(gr[i, 0])
        for (k0, k1), c in zip(windows, plt.get_cmap('tab10').colors):
            ys, steps = _lib.get_ys(cfg_of(m), m, HOOK, 'alpha_window', {'k0': k0, 'k1': k1})
            xs = np.asarray(_lib.get_xs_tokens(m, steps), float)
            k = xs > 0
            ax.plot(xs[k], np.asarray(ys, float)[k], color=c, lw=2.0, label=f'{k0}-{k1}')
        ax.set_xscale('log')
        ax.set_title(model_label(m), fontsize=fs)
        ax.set_ylabel(r'$\alpha$', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if i == len(models) - 1:
            ax.set_xlabel('Pretraining tokens', fontsize=fs)
    handles, labels = ax.get_legend_handles_labels()
    p = ax.get_position()                        # centre the legend under the line column only
    fig.legend(handles, labels, loc='center', ncol=len(windows), fontsize=fs,   # on the bar's
               frameon=False, bbox_to_anchor=((p.x0 + p.x1) / 2, cb_y + cb_h / 2))  # midline
    q = maps[-1].get_position()                  # the bar spans the map column, level with it
    sm = ScalarMappable(cmap=cmap, norm=Normalize(*vlim)); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes((q.x0, cb_y, q.width, cb_h)),
                      orientation='horizontal', extend='both')
    cb.set_label(r'$\alpha$', fontsize=fs, labelpad=2)
    cb.ax.tick_params(labelsize=fs - 3)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = phase_compare(save='analysis/figures/standalone/compare_map_alpha.pdf')


# %% [markdown]
# ## The ledger, summed over blocks
#
# ΔS = overlap + block-intrinsic entropy + interference, summed over every block, one panel per
# model: positive terms stack up from zero, negative down, black is the measured total. The
# showcase carries the same figure with the repo's own term names (χ / quality); this is the
# thesis rendering — prose names, one shared legend, no title. Each panel is also written out
# on its own next to the composite.

# %%
LEDGER_SPEC6 = ['pythia-410m-deduped', 'pythia-1b-deduped', 'pythia-6.9b-deduped',
                'nanochat-d12', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B']
N_BLOCKS = {'pythia-410m-deduped': 24, 'pythia-1b-deduped': 16, 'pythia-6.9b-deduped': 32,
            'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32, 'nanochat-d12': 12}
LEDGER_TERMS = (('chi', 'Overlap', 'tab:green'),
                ('quality', 'Block-intrinsic', 'tab:red'),
                ('interference', 'Interference', 'tab:purple'))
LS_FS, LS_FIG = 15, (15.3, 7.6)   # 2/3 of the way from the showcase grid's 21x10 to 12.5x6.4


def ledger_panel(ax, model, blocks=None, src=None, fs=LS_FS, emb=True, xlabel=True,
                 ylabel=True, total_lw=2.5, legend=None, title=None, zero_line=False):
    """The ledger stack summed over `blocks` (default every block) on `ax`, read from results
    dir `src` (default the model's baseline sweep). legend: fontsize, for standalone use — in
    the grid the legend is shared and drawn once under the whole figure."""
    plt.sca(ax)
    src = src or cfg_of(model)
    L = list(range(N_BLOCKS[model])) if blocks is None else list(blocks)
    steps = _lib.get_ys(src, model, (f'blk{L[0]}', 'block_ledger'), 'chi')[1]
    xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
    keep = xs > 0
    stack = [(lab, np.asarray(np.sum([_lib.get_ys(src, model, (f'blk{l}', 'block_ledger'), t)[0]
                                      for l in L], 0), float)[keep], c)
             for t, lab, c in LEDGER_TERMS]
    gx, tot = _lib.signed_stack(xs[keep], stack)
    ax.plot(gx, tot, 'k', lw=total_lw, label=r'$\Delta S$ total')
    if emb:
        es, esteps = _lib.get_ys(src, model, ('blk0.attn.in', 'acts_centered'), 'matrix_entropy')
        exs = np.asarray(_lib.get_xs_tokens(model, esteps), float)
        ek = exs > 0
        ax.plot(exs[ek], np.asarray(es, float)[ek], color='0.4', ls=':', lw=1.5,
                label=r'$S_0$ (embedding)')
    if zero_line: ax.axhline(0, color='0.35', lw=0.8)
    ax.set_xscale('log')
    ax.set_title(model_label(model) if title is None else title, fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if ylabel: ax.set_ylabel(r'$\Delta S$', fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)
    if legend: ax.legend(fontsize=legend)


def panel_grid(models, panel, ncols=3, figsize=LS_FIG, fs=LS_FS, legend_ncol=5,
               legend_from=0, panel_legend=False, save=None):
    """The house layout for this notebook: a grid of `panel(ax, model, fs=, xlabel=, ylabel=)`
    with outer labels only and one horizontal legend under the whole figure (legend_ncol=None
    for a panel that carries its own key). save: the composite's path; each panel also goes
    beside it as <stem>_<model>.pdf. legend_from: which panel's handles the legend copies."""
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, model in enumerate(models):
        panel(axes.flat[i], model, fs=fs, xlabel=i >= len(models) - ncols,
              ylabel=i % ncols == 0)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    handles, labels = axes.flat[legend_from].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.48 / figsize[1] if legend_ncol else 0, 1, 1))  # legend strip
    if legend_ncol:
        fig.legend(handles, labels, loc='lower center', ncol=legend_ncol, fontsize=fs,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight')
        size = (figsize[0] / ncols, figsize[1] / nrows)
        for model in models:
            sf, sax = plt.subplots(figsize=size)
            panel(sax, model, fs=fs, xlabel=True, ylabel=True)
            if panel_legend: sax.legend(fontsize=fs - 4)
            sf.tight_layout()
            sf.savefig(out.with_name(f'{out.stem}_{model}{out.suffix}'), bbox_inches='tight')
            plt.close(sf)
    return fig


def ledger_stack(models=LEDGER_SPEC6, **kw):
    """The ledger stack for each model, block-summed."""
    return panel_grid(models, ledger_panel, **kw)


# %%
_ = ledger_stack(save='analysis/figures/standalone/ledger_stack.pdf')

# %% [markdown]
# ## Ablating the carrier layer
#
# Pythia 1B with Layer 4's writes zeroed for the whole of inference, against the unablated
# sweep. Left 2x2: the ledger stack over all sixteen layers (top) and over the last quarter
# (bottom), baseline beside ablated. Right: RankMe at the two readout points, after the final
# norm (top) and at residual 16 (bottom), for the baseline and three interventions.
#
# **Indexing.** Figures are one-indexed and the code is not, so `blk3` is drawn as Layer 4.
# Layers run 1..L; residual points run 0 (the embedding output), 1 (after Layer 1), ... L
# (after Layer L), then afn (after the final norm), so `before_final_norm` is residual L.

# %%
ABL_MODEL = 'pythia-1b-deduped'
ABL_CARRIER = 'blk3'                              # the sink-write carrier; Layer 4 in the figure
# carrier, last quarter, the two together, middle half — the composite sits beside its parts
ABL_RUNS = ('blk3', 'blk12-15', 'blk3+12-15', 'blk4-11')
ABL_DROP = ('blk3+12-15')      # run(s) to leave out of the RankMe panels: one tag, or several
                               # as a list. '' or () keeps them all
ABL_FIG, ABL_FS = (17.0, 7.8), LS_FS
ABL_C = ('tab:blue', 'tab:orange', 'tab:purple', 'tab:green')
# 0-indexed run tag -> the name drawn in the figure; '<name> ablated' in the RankMe legend
ABL_NAMES = {'blk3': 'Layer 4', 'blk12-15': 'Last quarter', 'blk4-11': 'Middle half',
             'blk3+12-15': 'Layer 4 + Last quarter'}
ABL_SPANS = ('All layers', 'Last quarter')        # the two block groups the stacks sum over


def ledger_ablation(model=ABL_MODEL, carrier=ABL_CARRIER, runs=ABL_RUNS, drop=ABL_DROP,
                    figsize=ABL_FIG, fs=ABL_FS, legend_ncol=4, rk_legend_ncol=2,
                    rk_ylim=None, save=None):
    """2x3: the ledger stack baseline vs ablated over all layers (top) and the last quarter
    (bottom) on the left, RankMe at afn / residual L on the right. The two halves are separate
    figures sharing a canvas — a rule divides them, legends included. drop: runs to omit."""
    from matplotlib.lines import Line2D
    drop = [drop] if isinstance(drop, str) else list(drop)   # a bare tag must not match by
                                                            # substring ('blk3' vs 'blk3+12-15')
    L = N_BLOCKS[model]
    fig = plt.figure(figsize=figsize)
    # nested so the ΔS pair (sharing a y axis, so no labels between them) can sit close
    # together while the RankMe column keeps the room its own y label needs
    outer = fig.add_gridspec(1, 2, width_ratios=[2, 1.06], wspace=0.19,
                             left=0.052, right=0.988, top=0.935, bottom=0.20)
    gl = outer[0].subgridspec(2, 2, wspace=0.09, hspace=0.40)
    gr = outer[1].subgridspec(2, 1, hspace=0.40)
    cols = (('Baseline', cfg_of(model)),
            (f'{ABL_NAMES[carrier]} ablated', f'ablate_{carrier}'))
    left = []
    for i, blocks in enumerate((list(range(L)), list(range(3 * L // 4, L)))):
        base = fig.add_subplot(gl[i, 0])
        for j, (clabel, src) in enumerate(cols):
            ax = base if j == 0 else fig.add_subplot(gl[i, j], sharey=base)
            left.append(ax)
            ledger_panel(ax, model, blocks=blocks, src=src, fs=fs, emb=False, zero_line=True,
                         xlabel=i == 1, ylabel=j == 0,
                         title=f'{clabel}, {ABL_SPANS[i]}')
            if j: ax.tick_params(labelleft=False)
    handles, labels = base.get_legend_handles_labels()

    rk = []
    for i, (leaf, point) in enumerate((('after_final_norm', 'after final norm'), ('before_final_norm', 'before final norm'))):
        ax = fig.add_subplot(gr[i, 0], sharey=rk[0] if rk else None)
        rk.append(ax)
        # filtered as (tag, colour) pairs, so dropping a run leaves the others' colours put
        for tag, color, lw in ([(None, 'k', 2.4)]
                               + [(t, c, 1.7) for t, c in zip(runs, ABL_C) if t not in drop]):
            src = cfg_of(model) if tag is None else f'ablate_{tag}'
            ys, steps = _lib.get_ys(src, model, (leaf, 'acts_centered'), 'rankme')
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            k = xs > 0
            ax.plot(xs[k], np.asarray(ys, float)[k], color=color, lw=lw,
                    label='Baseline' if tag is None else f'{ABL_NAMES[tag]} abl.')
        ax.set_xscale('log')
        ax.set_title(f'Residual {point}', fontsize=fs)
        ax.set_ylabel('RankMe', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if i == 1: ax.set_xlabel('Pretraining tokens', fontsize=fs)
    if rk_ylim: rk[0].set_ylim(*rk_ylim)

    # one legend under each half, each centred on its own block, with a rule between them:
    # 'ΔS total' and 'Baseline' are both black lines and would otherwise read as one key
    span = lambda axs: (min(a.get_position().x0 for a in axs),
                        max(a.get_position().x1 for a in axs))
    (lx0, lx1), (rx0, rx1) = span(left), span(rk)
    fig.legend(handles, labels, loc='lower center', ncol=legend_ncol, fontsize=fs,
               frameon=False, bbox_to_anchor=((lx0 + lx1) / 2, 0.0))
    fig.legend(*rk[0].get_legend_handles_labels(), loc='lower center', ncol=rk_legend_ncol,
               fontsize=fs - 2, frameon=False, bbox_to_anchor=((rx0 + rx1) / 2, 0.0))
    xd = (lx1 + rx0) / 2
    fig.add_artist(Line2D([xd, xd], [0.01, 0.97], transform=fig.transFigure,
                          color='0.72', lw=1.2))
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight')
    return fig


# %%
_ = ledger_ablation(save='analysis/figures/standalone/ledger_ablation_pythia-1b.pdf')


# %% [markdown]
# ## Reproduction: the Li et al phases
#
# RankMe and α_ReQ after the final norm, in both data modes: the packed training mix and
# fineweb padded to last-token. nanochat has no padded run (no padding-mask support), so its
# panel carries the packed line only.

# %%
REPRO_HOOK = ('after_final_norm', 'acts_centered')
REPRO_SRCS = (('Trainset packed', None), ('FineWeb padded', 'rankme_fineweb_padded'))
REPRO_TRIO = ('pythia-1b-deduped', 'OLMo-2-0425-1B', 'nanochat-d12')   # one per family


def repro_panel(ax, model, yvar='rankme', ylab='RankMe', fs=LS_FS, xlabel=True, ylabel=True,
                title=None):
    """One model's after-final-norm curve in each data mode; a mode it lacks is skipped."""
    for (lbl, src), c in zip(REPRO_SRCS, plt.get_cmap('tab10').colors):
        ys, steps = _lib.get_ys(src or cfg_of(model), model, REPRO_HOOK, yvar)
        if ys is None:
            continue
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        k = xs > 0
        ax.plot(xs[k], np.asarray(ys, float)[k], color=c, lw=2.0, label=lbl)
    ax.set_xscale('log')
    ax.set_title(model_label(model) if title is None else title, fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if ylabel: ax.set_ylabel(ylab, fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)


def repro_compare(models=REPRO_TRIO, rows=(('rankme', 'RankMe'), ('alpha', r'$\alpha$')),
                  figsize=LS_FIG, fs=LS_FS, legend_ncol=2, save=None):
    """Columns = model, rows = metric. The model names title the top row only, and the two
    rows share a token axis per column."""
    fig, axes = plt.subplots(len(rows), len(models), figsize=figsize, squeeze=False,
                             sharex='col')
    for i, (yvar, ylab) in enumerate(rows):
        for j, m in enumerate(models):
            repro_panel(axes[i, j], m, yvar=yvar, ylab=ylab, fs=fs, ylabel=j == 0,
                        xlabel=i == len(rows) - 1, title=None if i == 0 else '')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=legend_ncol, fontsize=fs,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = panel_grid(LEDGER_SPEC6, repro_panel, legend_ncol=2,
               save='analysis/figures/standalone/repro_rankme.pdf')

# %%
_ = panel_grid(LEDGER_SPEC6,
               lambda ax, m, **kw: repro_panel(ax, m, yvar='alpha', ylab=r'$\alpha$', **kw),
               legend_ncol=2, save='analysis/figures/standalone/repro_alpha.pdf')

# %%
_ = repro_compare(save='analysis/figures/standalone/repro_compare.pdf')


# %% [markdown]
# ## Per-block ΔS
#
# Each block's own ΔS over training, packed geometry — the ledger stacks summed over blocks
# are the sums of these lines. Colour runs with depth, Layer 1 to Layer L.

# %%
def per_block_panel(ax, model, term='delta_s', fs=LS_FS, xlabel=True, ylabel=True, cbar=True):
    """One line per block, coloured by depth, with a depth key on the right of the panel."""
    from matplotlib.cm import ScalarMappable
    L = N_BLOCKS[model]
    for l in range(L):
        ys, steps = _lib.get_ys(cfg_of(model), model, (f'blk{l}', 'block_ledger'), term)
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        k = xs > 0
        ax.plot(xs[k], np.asarray(ys, float)[k], lw=1.5,
                color=plt.get_cmap('viridis')(l / (L - 1)))
    ax.axhline(0, color='0.35', lw=0.8)
    ax.set_xscale('log')
    ax.set_title(model_label(model), fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if cbar:
        cb = plt.colorbar(ScalarMappable(cmap='viridis'), ax=ax, pad=0.02, ticks=[])
        cb.set_label(f'Layer 1 → {L}', fontsize=fs - 4)
    if ylabel: ax.set_ylabel(r'$\Delta S$', fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)


# %%
_ = panel_grid(LEDGER_SPEC6, per_block_panel, legend_ncol=None,
               save='analysis/figures/standalone/per_block_delta_s.pdf')


# %% [markdown]
# ## Block write RankMe
#
# The showcase's write-RankMe figure over the full model set, from `block_write_compound`.
# The compound write is the block's two writes taken together as one covariance — the
# experimental one, so the MLP and attention writes get their own figures to read it against.
# One line per block, colour running with depth; trace is a separate figure below.

# %%
WRITE_SRC = 'block_write_compound'
# kind -> (node suffix, quantity family, column label). '' is the two writes combined.
WRITE_KINDS = {'mlp':      ('.mlp.out',  'acts_centered',        'MLP writes'),
               'attn':     ('.attn.out', 'acts_centered',        'Attention writes'),
               'compound': ('',          'block_write_compound', 'Compound writes')}


def write_panel(ax, model, kind='compound', yvar='rankme', ylab='RankMe', ylog=False,
                first_as_output=False, drop=(), fs=LS_FS, xlabel=True, ylabel=True,
                cbar=True, title=None):
    """One line per block's write, coloured by depth, with a depth key beside the panel.
    first_as_output: draw Layer 1 as the stream LEAVING it (embedding output + its write,
    i.e. blk1.attn.in from the baseline sweep) instead of the write on its own.
    drop: 0-indexed blocks to leave out; the rest keep their depth colour."""
    from matplotlib.cm import ScalarMappable
    suffix, fam, _ = WRITE_KINDS[kind]
    L = N_BLOCKS[model]
    for l in range(L):
        if l in drop:
            continue
        src, node = ((cfg_of(model), ('blk1.attn.in', 'acts_centered'))
                     if first_as_output and l == 0
                     else (WRITE_SRC, (f'blk{l}{suffix}', fam)))
        ys, steps = _lib.get_ys(src, model, node, yvar)
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        vs = np.asarray(ys, float)
        k = (xs > 0) & (vs > 0 if ylog else np.ones_like(xs, bool))
        ax.plot(xs[k], vs[k], lw=1.5, color=plt.get_cmap('viridis')(l / (L - 1)))
    ax.set_xscale('log')
    if ylog: ax.set_yscale('log')
    ax.set_title(model_label(model) if title is None else title, fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if cbar:
        cb = plt.colorbar(ScalarMappable(cmap='viridis'), ax=ax, pad=0.02, ticks=[])
        cb.set_label(f'Layer 1 → {L}', fontsize=fs - 4)
    if ylabel: ax.set_ylabel(ylab, fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)


def write_kind(kind, **fixed):
    """write_panel bound to one kind, for panel_grid."""
    return lambda ax, m, **kw: write_panel(ax, m, kind=kind, **fixed, **kw)


# %%
_ = panel_grid(LEDGER_SPEC6, write_kind('compound', first_as_output=True), legend_ncol=None,
               save='analysis/figures/standalone/write_rankme_compound.pdf')

# %%
_ = panel_grid(LEDGER_SPEC6, write_kind('mlp'), legend_ncol=None,
               save='analysis/figures/standalone/write_rankme_mlp.pdf')

# %%
_ = panel_grid(LEDGER_SPEC6, write_kind('attn'), legend_ncol=None,
               save='analysis/figures/standalone/write_rankme_attn.pdf')

# %%
_ = panel_grid(LEDGER_SPEC6, write_kind('compound', yvar='trace', ylab='Trace', ylog=True),
               legend_ncol=None,      # trace spans decades; RankMe stays linear everywhere
               save='analysis/figures/standalone/write_trace_compound.pdf')


# %%
def write_compare(models=CMP_MODELS, kinds=('mlp', 'attn', 'compound'), yvar='rankme',
                  ylab='RankMe', first_as_output=False, ylog=False, figsize=LS_FIG,
                  fs=LS_FS, save=None):
    """Rows = model, columns = which write. sharey per row, so the three are comparable
    within a model but each model keeps its own range."""
    fig, axes = plt.subplots(len(models), len(kinds), figsize=figsize, squeeze=False,
                             sharey='row')
    for i, m in enumerate(models):
        for j, kind in enumerate(kinds):
            write_panel(axes[i, j], m, kind=kind, yvar=yvar, ylab=ylab, fs=fs, ylog=ylog,
                        first_as_output=first_as_output,
                        xlabel=i == len(models) - 1, ylabel=j == 0,
                        title=f'{model_label(m)}, {WRITE_KINDS[kind][2]}')
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = write_compare(save='analysis/figures/standalone/write_rankme_compare.pdf', first_as_output=True)


# %%
# 0-indexed blocks to drop: pythia-1b loses Layer 4 (the spike write) and Layers 12, 14-16;
# OLMo-2 1B loses Layers 15-16. y is NOT shared across the pair — rescaling is the point.
WRITE_OUTLIERS = {'pythia-1b-deduped': (3, 11, 12, 13, 14, 15),
                  'OLMo-2-0425-1B': (14, 15)}
WRITE_OUT_FIG = (17.5, 8.2)   # wide enough for the model names to sit horizontal on the left
WRITE_OUT_COLS = ('All layers', 'Common layers', 'Layers with compression phase')


def layer_ranges(blocks):
    """0-indexed blocks as compact one-indexed text: (3, 11, 13, 14, 15) -> '4, 12, 14-16'."""
    runs = []
    for x in sorted(l + 1 for l in blocks):
        if runs and x == runs[-1][1] + 1: runs[-1][1] = x
        else: runs.append([x, x])
    return ', '.join(f'{a}-{z}' if a != z else f'{a}' for a, z in runs)


def write_outliers(models=tuple(WRITE_OUTLIERS), kind='compound', yvar='rankme',
                   ylab='RankMe', outliers=WRITE_OUTLIERS, first_as_output=True, ylog=False,
                   afn_bg=True, peak_line=True, table_x=(0.075, 0.145),
                   figsize=WRITE_OUT_FIG, fs=LS_FS, save=None):
    """Rows = model, columns = every block / the common ones / the outliers on their own.
    afn_bg: the after-final-norm RankMe greyed in behind (same units as the writes).
    peak_line: dotted rule at that curve's canonical peak."""
    from matplotlib.cm import ScalarMappable
    fig, axes = plt.subplots(len(models), 3, figsize=figsize, squeeze=False, sharey='row')
    for i, m in enumerate(models):
        src, out = cfg_of(m), outliers.get(m, ())
        keep_out = tuple(l for l in range(N_BLOCKS[m]) if l not in out)   # all but the outliers
        for j, dropped in enumerate(((), out, keep_out)):
            ax = axes[i, j]
            write_panel(ax, m, kind=kind, yvar=yvar, ylab=ylab, fs=fs, drop=dropped,
                        first_as_output=first_as_output, cbar=False, ylog=ylog,
                        xlabel=i == len(models) - 1, ylabel=j == 0,
                        title=WRITE_OUT_COLS[j] if i == 0 else '')
            # include_init=False on both, or the peak index lands a row off — and step 0 puts
            # a spurious spike on the left edge of the grey curve
            rx, rv = _lib.rankme_series(src, m, HOOK, include_init=False)
            if afn_bg and yvar == 'rankme':          # only comparable in RankMe's own units
                ax.plot(rx, rv, color='0.75', lw=3.5, zorder=1,
                        label='RankMe after final norm')
            if peak_line:
                ax.axvline(rx[_lib.rankme_peak(src, m, HOOK, include_init=False)],
                           color='0.45', ls=':', lw=1.6, zorder=1)
    depths = {N_BLOCKS[m] for m in models}
    fig.tight_layout(rect=(0.10, 0, 0.93, 0.88))   # room for the header block
    for i, m in enumerate(models):            # the model names as row labels, outside the axes
        p = axes[i, 0].get_position()
        fig.text(0.012, (p.y0 + p.y1) / 2, model_label(m), va='center', ha='left', fontsize=fs)
    # the header table, laid out by position in the figure's own font — padding a monospace
    # block aligns too, but the fallback mono face sits oddly against everything else
    x, line, y0 = axes[0, 0].get_position().x0, 0.045, 0.99
    head = fig.text(x, y0, 'Layers with their own compression phase:', fontsize=fs + 1,
                    va='top')
    cond = fig.text(x, y0 - line, "(condition used: more than 10% entropy drop from the "
                    "layer's global peak)", fontsize=fs - 3, color='0.35', va='top')
    fig.canvas.draw()                    # the table sits beside the header, clear of BOTH lines
    xt = max(t.get_window_extent().transformed(fig.transFigure.inverted()).x1
             for t in (head, cond)) + 0.025
    for k, m in enumerate(models):
        for dx, txt, c in ((0.0, model_label(m), '0.15'),
                           (table_x[0], layer_ranges(outliers.get(m, ())), '0.15'),
                           (table_x[1], f'{100 * len(outliers.get(m, ())) / N_BLOCKS[m]:.1f}%',
                            '0.35')):
            fig.text(xt + dx, y0 - line * k, txt, fontsize=fs - 1, color=c, va='top')
    cb = fig.colorbar(ScalarMappable(cmap='viridis'),
                      cax=fig.add_axes((0.945, 0.12, 0.016, 0.74)), ticks=[])
    cb.set_label(f'Layer 1 → {depths.pop()}' if len(depths) == 1 else 'Relative depth',
                 fontsize=fs - 2)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = write_outliers(save='analysis/figures/standalone/write_rankme_outliers.pdf', afn_bg=False)
