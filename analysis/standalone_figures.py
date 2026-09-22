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
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from pathlib import Path
from matplotlib.colors import to_rgb
from matplotlib.transforms import offset_copy
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
               legend_from=0, handles=None, panel_legend=False, save=None):
    """The house layout for this notebook: a grid of `panel(ax, model, fs=, xlabel=, ylabel=)`
    with outer labels only and one horizontal legend under the whole figure (legend_ncol=None
    for a panel that carries its own key). save: the composite's path; each panel also goes
    beside it as <stem>_<model>.pdf. legend_from: which panel's handles the legend copies;
    `handles` overrides that, for keys a panel's own axes cannot supply (e.g. a twin's line)."""
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, model in enumerate(models):
        panel(axes.flat[i], model, fs=fs, xlabel=i >= len(models) - ncols,
              ylabel=i % ncols == 0)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    hs, ls = ((handles, [h.get_label() for h in handles]) if handles else
              axes.flat[legend_from].get_legend_handles_labels())
    fig.tight_layout(rect=(0, 0.48 / figsize[1] if legend_ncol else 0, 1, 1))  # legend strip
    if legend_ncol:
        fig.legend(hs, ls, loc='lower center', ncol=legend_ncol, fontsize=fs,
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
    # measure the RankMe column's INK, not its axes box: its y label and ticks stick out to
    # the left of x0, and the plain midpoint runs the rule straight through them
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    ink = min(inv.transform(a.get_tightbbox(fig.canvas.get_renderer()))[0][0] for a in rk)
    xd = (lx1 + ink) / 2
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
# ### The same, widened to the token filter and the other two families
#
# 2x5: the stacks gain a third column read on interior tokens only, and the last column carries
# an after-final-norm RankMe panel for OLMo 2 1B and nanochat D12. There is no ablated
# content-token run, so the ablation and the token filter are each read against the packed
# baseline. The other two families have last-quarter and middle-half runs only, so their panels
# carry three lines.

# %%
ABL_WIDE_FIG = (24.5, 8.4)
ABL_PAD = 0.006                  # gap around the margin labels and the group rule, figure units
ABL_CONTENT_SRC = 'content_tokens_full'
ABL_ROLE_C = {ABL_NAMES[t]: c for t, c in zip(ABL_RUNS, ABL_C)}   # colour by role, not by tag
# same two roles as Pythia's, at each model's own depth
ABL_OTHERS = {'OLMo-2-0425-1B': {'Last quarter': 'blk12-15', 'Middle half': 'blk4-11'},
              'nanochat-d12': {'Last quarter': 'blk9-11', 'Middle half': 'blk3-8'}}


def ledger_ablation_wide(model=ABL_MODEL, carrier=ABL_CARRIER, runs=ABL_RUNS, drop=ABL_DROP,
                         content_src=ABL_CONTENT_SRC, others=ABL_OTHERS, figsize=ABL_WIDE_FIG,
                         fs=ABL_FS, legend_ncol=4, rk_legend_ncol=4, rk_ylim=None, save=None):
    """ledger_ablation with two more columns: the stacks read on interior tokens, and an
    after-final-norm RankMe panel for each of the other two families. Three blocks, each
    divided off by a rule, as in ledger_ablation."""
    from matplotlib.lines import Line2D
    drop = [drop] if isinstance(drop, str) else list(drop)
    L = N_BLOCKS[model]
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, 3, width_ratios=[3, 1.06, 1.06], wspace=0.15,
                             left=0.037, right=0.99, top=0.915, bottom=0.19)
    gl = outer[0].subgridspec(2, 3, wspace=0.09, hspace=0.46)
    gm = outer[1].subgridspec(2, 1, hspace=0.46)
    gr = outer[2].subgridspec(2, 1, hspace=0.46)
    cols = (('Baseline', cfg_of(model)),
            (f'{ABL_NAMES[carrier]} ablated', f'ablate_{carrier}'),
            ('No delimiters or first tokens', content_src))
    left = []
    for i, blocks in enumerate((list(range(L)), list(range(3 * L // 4, L)))):
        base = fig.add_subplot(gl[i, 0])
        for j, (clabel, src) in enumerate(cols):
            ax = base if j == 0 else fig.add_subplot(gl[i, j], sharey=base)
            left.append(ax)
            ledger_panel(ax, model, blocks=blocks, src=src, fs=fs, emb=False, zero_line=True,
                         xlabel=i == 1, ylabel=j == 0, title=clabel)
            if j: ax.tick_params(labelleft=False)
    handles, labels = base.get_legend_handles_labels()

    rk = []
    for i, (leaf, point) in enumerate((('after_final_norm', 'afn'),
                                       ('before_final_norm', 'bfn'))):
        ax = fig.add_subplot(gm[i, 0], sharey=rk[0] if rk else None)
        rk.append(ax)
        for tag, color, lw in ([(None, 'k', 2.4)]
                               + [(t, c, 1.7) for t, c in zip(runs, ABL_C) if t not in drop]):
            src = cfg_of(model) if tag is None else f'ablate_{tag}'
            ys, steps = _lib.get_ys(src, model, (leaf, 'acts_centered'), 'rankme')
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            k = xs > 0
            ax.plot(xs[k], np.asarray(ys, float)[k], color=color, lw=lw,
                    label='Baseline' if tag is None else f'{ABL_NAMES[tag]} abl.')
        ax.set_xscale('log')
        ax.set_title(f'{model_label(model)}, {point}', fontsize=fs)
        ax.set_ylabel('RankMe', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if i == 1: ax.set_xlabel('Pretraining tokens', fontsize=fs)
    if rk_ylim: rk[0].set_ylim(*rk_ylim)

    oth = []                                      # own y each: the families differ in scale
    for i, (m, tags) in enumerate(others.items()):
        ax = fig.add_subplot(gr[i, 0])
        oth.append(ax)
        for role, tag, color, lw in ([('Baseline', None, 'k', 2.4)]
                                     + [(r, t, ABL_ROLE_C[r], 1.7) for r, t in tags.items()]):
            src = cfg_of(m) if tag is None else f'ablate_{tag}'
            ys, steps = _lib.get_ys(src, m, ('after_final_norm', 'acts_centered'), 'rankme')
            xs = np.asarray(_lib.get_xs_tokens(m, steps), float)
            k = xs > 0
            ax.plot(xs[k], np.asarray(ys, float)[k], color=color, lw=lw,
                    label=role if tag is None else f'{role} abl.')
        ax.set_xscale('log')
        ax.set_title(f'{model_label(m)}, afn', fontsize=fs)
        ax.set_ylabel('RankMe', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if i == 1: ax.set_xlabel('Pretraining tokens', fontsize=fs)

    span = lambda axs: (min(a.get_position().x0 for a in axs),
                        max(a.get_position().x1 for a in axs))
    (lx0, lx1), (ox0, ox1) = span(left), span(oth)
    fig.canvas.draw()
    inv, rend = fig.transFigure.inverted(), fig.canvas.get_renderer()
    ink = lambda axs: min(inv.transform(a.get_tightbbox(rend))[0][0] for a in axs)
    fig.legend(handles, labels, loc='lower center', ncol=legend_ncol, fontsize=fs,
               frameon=False, bbox_to_anchor=((lx0 + lx1) / 2, 0.0))
    # one row under every RankMe panel: the other families draw a subset of the same lines
    fig.legend(*rk[0].get_legend_handles_labels(), loc='lower center', ncol=rk_legend_ncol,
               fontsize=fs - 2, frameon=False, bbox_to_anchor=((ink(rk) + ox1) / 2, 0.0))
    # the row label carries the layers, so the column titles stay one line and the model is
    # named once, over the stacks it owns
    rows = [fig.text(0.0, sum(left[3 * i].get_position().intervaly) / 2, lbl, rotation=90,
                     va='center', ha='left', fontsize=fs) for i, lbl in enumerate(ABL_SPANS)]
    lab = min(left[3 * i].yaxis.label.get_window_extent().transformed(inv).x0 for i in (0, 1))
    shift = lab - ABL_PAD - max(t.get_window_extent().transformed(inv).x1 for t in rows)
    for t in rows:
        t.set_x(t.get_position()[0] + shift)
    hx0, hy = min(t.get_window_extent().transformed(inv).x0 for t in rows), 0.972
    head = fig.text((hx0 + lx1) / 2, hy, model_label(model), ha='center', va='center',
                    fontsize=fs + 1)
    fig.canvas.draw()
    b = head.get_window_extent().transformed(inv)
    # the vertical rule sits midway between the stacks and the leftmost INK of the RankMe
    # panels: their y labels and ticks stick out past the axes box
    xd = (lx1 + ink(rk)) / 2
    for xs_, ys_ in (([hx0, b.x0 - ABL_PAD], [hy, hy]), ([b.x1 + ABL_PAD, lx1], [hy, hy]),
                     ([xd, xd], [0.01, 0.94])):
        fig.add_artist(Line2D(xs_, ys_, transform=fig.transFigure, color='0.72', lw=1.2))
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight')
    return fig


# %%
_ = ledger_ablation_wide(save='analysis/figures/standalone/ledger_ablation_wide_pythia-1b.pdf')


# %% [markdown]
# ## Reproduction: the Li et al phases
#
# RankMe and α_ReQ after the final norm, in both data modes: the packed training mix and
# fineweb padded to last-token. nanochat has no padded run (no padding-mask support), so its
# panel carries the packed line only.

# %%
from matplotlib.lines import Line2D

REPRO_HOOK = ('after_final_norm', 'acts_centered')
# padded source per model, all 50 log-spaced checkpoints: Pythia at max length 128, OLMo at 512.
# Pythia 1B's run is the layerwise one, which supersets the after-final-norm sweep.
PADDED_SRC = {'pythia-1b-deduped': 'fineweb_padded_layerwise_len128',
              'OLMo-2-0425-1B': 'rankme_fineweb_padded_log',
              'OLMo-2-1124-7B': 'rankme_fineweb_padded_log'}
padded_of = lambda m: PADDED_SRC.get(m, 'rankme_fineweb_padded_len128')
REPRO_SRCS = (('Trainset packed (ours)', cfg_of), ('FineWeb padded (Li et al.)', padded_of))
REPRO_TRIO = ('pythia-1b-deduped', 'OLMo-2-0425-1B', 'nanochat-d12')   # one per family
REPRO_LS = ((':', 'Warm-up'), ('-', 'Expansion'), ('--', 'Compression'))
_repro_key = lambda: [Line2D([], [], color='0.35', lw=2.0, ls=ls, label=lab)
                      for ls, lab in REPRO_LS]


def phase_bounds(src, model, hook=REPRO_HOOK):
    """(trough, peak) in tokens, read off the RankMe curve of that same run, so each data
    mode is split at its own landmarks. None where the run cannot supply them."""
    try:
        rx, _ = _lib.rankme_series(src, model, hook, include_init=False)
        tr, pk = _lib.rankme_phases(src, model, hook, include_init=False)
        return float(rx[tr]), float(rx[pk])
    except (TypeError, ValueError, IndexError):
        return None


def plot_phased(ax, xs, ys, bounds, color, lw=2.0, label=None):
    """One curve in three linestyles, split at the trough and the peak. Neighbouring segments
    share their boundary point so the line does not break, and the legend proxy stays solid."""
    if bounds is None:
        ax.plot(xs, ys, color=color, lw=lw, label=label)
        return
    lo, hi = bounds
    for (ls, _), k in zip(REPRO_LS, (xs <= lo, (xs >= lo) & (xs <= hi), xs >= hi)):
        if k.sum() > 1:
            ax.plot(xs[k], ys[k], color=color, lw=lw, ls=ls)
    ax.plot([], [], color=color, lw=lw, label=label)


def repro_panel(ax, model, yvar='rankme', ylab='RankMe', fs=LS_FS, xlabel=True, ylabel=True,
                title=None):
    """One model's after-final-norm curve in each data mode; a mode it lacks is skipped.
    Linestyle marks the phase, taken from that mode's own RankMe landmarks whatever `yvar`
    is plotted."""
    for (lbl, src), c in zip(REPRO_SRCS, plt.get_cmap('tab10').colors):
        ys, steps = _lib.get_ys(src(model), model, REPRO_HOOK, yvar)
        if ys is None:
            continue
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        k = xs > 0
        plot_phased(ax, xs[k], np.asarray(ys, float)[k], phase_bounds(src(model), model),
                    color=c, label=lbl)
    ax.set_xscale('log')
    ax.set_title(model_label(model) if title is None else title, fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if ylabel: ax.set_ylabel(ylab, fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)


def repro_compare(models=REPRO_TRIO, rows=(('rankme', 'RankMe'), ('alpha', r'$\alpha$')),
                  figsize=LS_FIG, fs=LS_FS, legend_ncol=5, save=None):
    """Columns = model, rows = metric. The model names title the top row only, and the two
    rows share a token axis per column."""
    fig, axes = plt.subplots(len(rows), len(models), figsize=figsize, squeeze=False,
                             sharex='col')
    for i, (yvar, ylab) in enumerate(rows):
        for j, m in enumerate(models):
            repro_panel(axes[i, j], m, yvar=yvar, ylab=ylab, fs=fs, ylabel=j == 0,
                        xlabel=i == len(rows) - 1, title=None if i == 0 else '')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += _repro_key()                         # colour keys the data mode, style the phase
    labels += [lab for _ls, lab in REPRO_LS]
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
    rx, _ = _lib.rankme_series(cfg_of(model), model, HOOK, include_init=False)
    ax.axvline(rx[_lib.rankme_peak(cfg_of(model), model, HOOK, include_init=False)],
               color='0.45', ls=':', lw=1.6, zorder=1)   # start of the compression phase
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


# %% [markdown]
# ## What drives the sink write
#
# Pythia 1B, blk3.mlp's output at the final checkpoint, each token row projected onto the
# write's top centered eigendirection. Left: the token types with the largest mean projection.
# Right: the distribution of that projection over all 262k rows of the packed mix, split by
# what the row is. Precomputed by `oneoff_scripts/rogue_id_attribution.py`.

# %%
import torch

ROGUE_ID = torch.load('data/results/rogue_id_attribution.pt', weights_only=False)
ROGUE_MODEL = 'pythia-1b-deduped'
PROJ = r'$\|\text{proj}_\text{spike}\|$'
# drawn bulk-first so the small groups sit on top; the legend is reordered to read newline /
# position 0 / both / other, since "other" only means anything after the named groups. Purple
# for the overlap: a newline that IS the window's first token, between its two parents' hues
ROGUE_GROUPS = (('Other tokens', 'tab:gray'), ('Tokens at Position 0', 'tab:blue'),
                ('Newline tokens', 'tab:red'), ('Both', 'tab:purple'))
ROGUE_LEGEND = (2, 1, 3, 0)
ROGUE_FIG = (15.4, 5.2)
ROGUE_WR = (1, 1)   # the legend is cleared by the ranking's x headroom, not by width


def rogue_ranking_panel(ax, model=ROGUE_MODEL, top=20, min_n=20, headroom=1.08, fs=LS_FS):
    """Token types ranked by mean |projection|; red = newline-bearing, tick = 95% CI floor."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ty = ROGUE_ID[model]['types']
    mean_abs, var, counts = ty['mean_abs'], ty['var_abs'], ty['count']
    order = torch.argsort(mean_abs.masked_fill(counts < min_n, -1), descending=True)[:top].tolist()
    decoded = [ty['token'][i] for i in order]
    vals = [float(mean_abs[i]) for i in order]
    ci = [1.96 * float(var[i].clamp(min=0).sqrt()) / int(counts[i]) ** 0.5 for i in order]
    ypos = list(range(top))[::-1]
    ax.barh(ypos, vals, color=['tab:red' if '\n' in t else 'tab:gray' for t in decoded])
    ax.scatter([max(v - c, 0) for v, c in zip(vals, ci)], ypos, marker='|', color='k',
               s=80, zorder=3)
    ax.set_yticks(ypos, [f'{t!r}  ({int(counts[i])})' for t, i in zip(decoded, order)],
                  fontsize=fs - 5)
    ax.set_ylabel('Token (count)', rotation=0, ha='right', va='top', fontsize=fs - 4)
    ax.yaxis.set_label_coords(-0.015, -0.012)      # under the tick labels, not beside them
    # every bar starts at 0 and they are all a similar length, so there is no free corner
    # inside the axes: clearing the key by headroom or by width costs far more space than
    # putting it under the axis does
    ax.set_xlim(0, max(vals) * headroom)
    ax.set_xlabel(f'Mean {PROJ}', fontsize=fs)
    ax.tick_params(axis='x', labelsize=fs - 3)
    # same criterion as the histogram's split: the token's text contains a newline character
    ax.legend(handles=[Patch(color='tab:red', label='Newline tokens'),
                       Patch(color='tab:gray', label='Other tokens'),
                       Line2D([], [], color='k', marker='|', ls='none', markersize=9,
                              label='95% CI lower edge')],
              fontsize=fs - 4, loc='lower right')


def rogue_hist_panel(ax, model=ROGUE_MODEL, bins=120, fs=LS_FS):
    """|projection| over every row of the packed mix, split by what the row is."""
    from matplotlib.patches import Patch
    a = ROGUE_ID[model]
    score, is_nl, is_p0 = a['score'], a['is_newline'], a['is_pos0']
    # the overlap gets its own group, so the four are disjoint and nothing is absorbed
    masks = (~is_nl & ~is_p0, is_p0 & ~is_nl, is_nl & ~is_p0, is_nl & is_p0)
    v = score.abs().numpy()
    edges = np.histogram_bin_edges(v, bins=bins)
    cts = np.stack([np.histogram(v[m.numpy()], bins=edges)[0] for m in masks])
    # opaque, shortest bar in front: within a bin the tallest is drawn first and the shorter
    # ones over it, so each group reads as its own colour instead of a blend
    rank = np.argsort(np.argsort(-cts, axis=0, kind='stable'), axis=0)
    mid, w = (edges[:-1] + edges[1:]) / 2, np.diff(edges)
    for g, (_lbl, c) in enumerate(ROGUE_GROUPS):
        for r in range(len(masks)):
            k = (rank[g] == r) & (cts[g] > 0)
            ax.bar(mid[k], cts[g][k], width=w[k], color=c, zorder=2 + r)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.7)                        # so single-count bins stay visible
    ax.set_xlabel(PROJ, fontsize=fs)
    ax.set_ylabel('Token count', fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    # keys built from the group colours, not from the drawn bars: a group that is never the
    # tallest in any bin contributes no rank-0 container and would shift the handle list
    ax.legend(handles=[Patch(color=ROGUE_GROUPS[i][1], label=ROGUE_GROUPS[i][0])
                       for i in ROGUE_LEGEND], fontsize=fs - 3)


def rogue_figure(model=ROGUE_MODEL, figsize=ROGUE_FIG, fs=LS_FS, save=None):
    """Ranking left, histogram right. save: the pair's path; each panel also goes beside it
    as <stem>_ranking.pdf / <stem>_hist.pdf, for using them as separate subfigures."""
    panels = (rogue_ranking_panel, rogue_hist_panel)
    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={'width_ratios': list(ROGUE_WR)})
    for ax, panel in zip(axes, panels):
        panel(ax, model, fs=fs)
    fig.tight_layout()
    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight')
        for panel, name, r in zip(panels, ('ranking', 'hist'), ROGUE_WR):
            sf, sax = plt.subplots(figsize=(figsize[0] * r / sum(ROGUE_WR), figsize[1]))
            panel(sax, model, fs=fs)
            sf.tight_layout()
            sf.savefig(out.with_name(f'{out.stem}_{name}{out.suffix}'), bbox_inches='tight')
            plt.close(sf)
    return fig


# %%
_ = rogue_figure(save='analysis/figures/standalone/sink_attribution.pdf')


# %% [markdown]
# ## The unembedding and the stream trade off
#
# α_ReQ over spectrum ranks 11–100. The unembedding's own spectrum against the stream that
# enters it, and then the test: their sum against the directly measured α_ReQ of the logit
# covariance. Data from `oneoff_scripts/alpha_conservation.py`.

# %%
AC = torch.load('data/results/alpha_conservation.pt', weights_only=False)
AC_WINDOW = (11, 100)
from matplotlib.lines import Line2D

AC_C = {'unembedding': 'tab:blue', 'stream': 'tab:red'}
AC_KEY = (('Final state representations', 'stream'), ('Unembedding weights', 'unembedding'))
AC_FAMILIES = ('Pythia', 'OLMo 2', 'Nanochat')     # panel per family, in this order
AC_FIG, AC_SUM_FIG = LS_FIG, (15.3, 4.6)


def ac_alpha(v, k0=AC_WINDOW[0], k1=AC_WINDOW[1]):
    """The slope, or nan where the spectrum cannot carry one (nanochat's zero-init head)."""
    a = np.asarray(v, float)
    if (a > 0).sum() < 4:
        return np.nan
    try:
        return _lib._alpha(a, k0, k1)
    except np.linalg.LinAlgError:
        return np.nan


def ac_series(model, leaf='after_final_norm'):
    """(tokens, unembedding α, stream α, logit-covariance α), dropping any checkpoint where
    one of the three is undefined so the sum always adds curves measured at the same steps."""
    per = AC[model]
    steps = sorted(s for s in per if leaf in per[s])
    x = np.array([per[s]['tokens'] for s in steps], float)
    cols = [np.array([ac_alpha(per[s]['head'].numpy()) for s in steps])]
    cols += [np.array([ac_alpha(per[s][leaf][k].numpy()) for s in steps])
             for k in ('stream', 'logit')]
    ok = (x > 0) & np.all(np.isfinite(cols), axis=0)
    return (x[ok], *(c[ok] for c in cols))


def ac_panel(ax, model, fs=LS_FS, xlabel=True, ylabel=True):
    """One model's two factors on twin axes: they sit at different levels, so only the shapes
    are comparable. Which curve is which is in the shared key, so the axis carries a bare α.
    The bare alpha labels the left axis; the twin carries only its own ticks."""
    x, head, stream, _logit = ac_series(model)
    ax.plot(x, head, lw=2.0, color=AC_C['unembedding'])
    tw = ax.twinx()
    tw.plot(x, stream, lw=2.0, color=AC_C['stream'])
    # the two axes are offset but carry the SAME span, so a vertical distance is the same
    # change in alpha on either curve — the levels stay incomparable, the slopes do not
    span = max(np.ptp(head), np.ptp(stream)) * 1.15
    for a, v, side in ((ax, head, 'unembedding'), (tw, stream, 'stream')):
        mid = (v.min() + v.max()) / 2
        a.set_ylim(mid - span / 2, mid + span / 2)
        a.tick_params(axis='y', labelcolor=AC_C[side], labelsize=fs - 3)
    tw.grid(False)
    ax.set_xscale('log')
    ax.set_title(model_label(model), fontsize=fs)
    ax.tick_params(axis='x', labelsize=fs - 3)
    if ylabel: ax.set_ylabel(r'$\alpha$', fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)


def ac_sum(models=tuple(AC), families=AC_FAMILIES, figsize=AC_SUM_FIG, fs=LS_FS, save=None):
    """Per family: the measured α of the logit covariance (solid) against the sum of the two
    factors (dashed). Where they separate, the stream's eigenbasis is not aligned with the
    unembedding's, and the gap measures by how much."""
    from matplotlib.lines import Line2D
    fam_of = lambda m: next(f for f in families if f.split()[0].lower() in m.lower())
    fig, axes = plt.subplots(1, len(families), figsize=figsize, squeeze=False)
    for ax, fam in zip(axes[0], families):
        for model in [m for m in models if fam_of(m) == fam]:
            x, head, stream, logit = ac_series(model)
            c = plt.get_cmap('tab10')(list(models).index(model) % 10)
            ax.plot(x, logit, lw=2.0, color=c, label=model_label(model))
            ax.plot(x, head + stream, lw=1.6, ls='--', color=c)
        ax.set_xscale('log')
        ax.set_title(fam, fontsize=fs)
        ax.set_xlabel('Pretraining tokens', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        ax.legend(fontsize=fs - 4)
    axes[0, 0].set_ylabel(r'$\alpha$', fontsize=fs)
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles=[Line2D([], [], color='0.25', lw=2.0, label='Logit covariance'),
                        Line2D([], [], color='0.25', lw=1.6, ls='--',
                               label='Unembedding + representation')],
               loc='lower center', ncol=2, fontsize=fs, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = panel_grid(LEDGER_SPEC6, ac_panel, legend_ncol=2, figsize=AC_FIG,
               handles=[Line2D([], [], color=AC_C[k], lw=2.0, label=lbl)
                        for lbl, k in AC_KEY],   # the twin's line is not on the panel's axes
               save='analysis/figures/standalone/alpha_factors.pdf')

# %%
_ = ac_sum(save='analysis/figures/standalone/alpha_sum_vs_logit.pdf')


# %% [markdown]
# ## Entropy neurons against the final stream
#
# Three scales per panel: RankMe of the centered final stream, the most null-space-aligned
# neuron in the last block (max ρ, Stolfo et al's statistic), and what zero-ablating the last
# quarter of the blocks does to that RankMe (ablated − baseline, on the checkpoints the two
# runs share). Each y-axis wears its curve's colour. Data from
# `oneoff_scripts/entropy_neurons.py`.

# %%
from matplotlib.lines import Line2D

EN_DATA = torch.load('data/results/entropy_neurons.pt', weights_only=False)
EN_HOOK = ('after_final_norm', 'acts_centered')
EN_LAST_Q = {12: 'ablate_blk9-11', 16: 'ablate_blk12-15',       # the quarter split used by
             24: 'ablate_blk18-23', 32: 'ablate_blk24-31'}      # ledger_ablation_layers
# one colour per QUANTITY, not per model: the three scales have to be told apart within a
# panel, and a per-model hue made max rho mean something different in every one
EN_C = {'rankme': '0.66', 'rho': 'tab:blue', 'gap': 'tab:red'}   # RankMe reads as ground
EN_KEY = (('RankMe', 'rankme', '-'), (r'Max $\rho$', 'rho', '-'),
          (r'$\text{RankMe}_\text{Q4 ablated} - \text{RankMe}_\text{baseline}$ ($\Delta$RankMe)', 'gap', '--'))
EN_FIG, EN_OFFSET = (17.5, 8.0), 1.14   # a tighter offset lets the columns sit closer
EN_OFFSET_LAST = 1.18   # the last column also fits the rho label between the two spines


def en_rho(model, frac=0.01):
    """(tokens, max rho in the last block) over checkpoints, skipping zero-init blocks —
    nanochat's first two have an identically zero down-projection, so rho is 0/0 there."""
    per = EN_DATA[model][max(EN_DATA[model])]
    steps = [s for s in sorted(per) if float(per[s]['norm'].max()) > 0]
    ki = per[steps[0]]['ks'].index(max(1, round(frac * per[steps[0]]['d_model'])))
    return (np.array([per[s]['tokens'] for s in steps], float),
            np.array([per[s]['rho'][ki].numpy().max() for s in steps]))


def en_chance(model, q=1.0, frac=0.01, trials=10, seed=0):
    """The q-quantile of rho over a layer of randomly oriented neurons. sqrt(k/d) is the mean
    for ONE direction; the max over thousands sits far above it, so the reference has to be
    the matching order statistic."""
    per = EN_DATA[model][max(EN_DATA[model])]
    e = per[max(per)]
    d, n = e['d_model'], e['norm'].numel()
    k = max(1, round(frac * d))
    rng = np.random.default_rng(seed)
    return float(np.mean([np.quantile(np.sqrt((g := rng.standard_normal((n, d)) ** 2)[:, :k]
                                              .sum(1) / g.sum(1)), q) for _ in range(trials)]))


def en_gap(model, yvar='rankme'):
    """(tokens, metric with the last quarter of blocks zeroed − baseline) over the checkpoints
    both runs carry; the ablation sweeps are the sparser of the two."""
    base, steps = _lib.get_ys(cfg_of(model), model, EN_HOOK, yvar)
    abl, asteps = _lib.get_ys(EN_LAST_Q[N_BLOCKS[model]], model, EN_HOOK, yvar)
    if base is None or abl is None:
        return None, None
    b = dict(zip(steps, base))
    shared = [(s, a) for s, a in zip(asteps, abl) if s in b]
    return (np.asarray(_lib.get_xs_tokens(model, [s for s, _ in shared]), float),
            np.asarray([a - b[s] for s, a in shared], float))


def en_floor(gaps, margin=0.05):
    """How far below zero the ablation axis may reach, as a fraction of a panel's own peak.
    Pythia 410M ends far below zero and drags its axis down while every other panel is already
    well scaled, so take the SECOND lowest would-be autoscaled bottom: the floor is then
    exactly the deepest any other model goes, and only that one outlier is clipped."""
    lows = sorted((min(g.min(), 0) - margin * (max(g.max(), 0) - min(g.min(), 0))) / g.max()
                  for g in gaps if g is not None and g.max() > 0)
    return lows[1] if len(lows) > 1 else lows[0]


def _en_axis(a, color, side='right'):
    """Tie one y-axis' ticks, label and spine to its curve — three scales in a panel need it."""
    a.tick_params(axis='y', colors=color, labelsize=LS_FS - 4)
    a.yaxis.label.set_color(color)
    a.spines[side].set_color(color)


def entropy_neurons(models=LEDGER_SPEC6, figsize=EN_FIG, fs=LS_FS, floor=None, save=None):
    """RankMe / max rho / ablation gap on three scales, one panel per model."""
    gaps = {m: en_gap(m) for m in models}
    floor = en_floor([g for _, g in gaps.values()]) if floor is None else floor
    nrows = -(-len(models) // 3)
    fig, axes = plt.subplots(nrows, 3, figsize=figsize, squeeze=False)
    for i, model in enumerate(models):
        ax = axes.flat[i]
        last = i % 3 == 2
        vals, steps = _lib.get_ys(cfg_of(model), model, EN_HOOK, 'rankme')
        ax.plot(np.asarray(_lib.get_xs_tokens(model, steps), float),
                np.asarray(vals, float), color=EN_C['rankme'], lw=3.0, zorder=1)
        rx, ry = en_rho(model)
        tw = ax.twinx()
        tw.plot(rx, ry, color=EN_C['rho'], lw=1.8)
        tw.axhline(en_chance(model), color=EN_C['rho'], ls=':', lw=0.9)
        tw.set(ylim=(0, 1.02))
        tw.grid(False)
        if last: tw.set_ylabel(r'Max $\rho$', fontsize=fs)
        _en_axis(tw, EN_C['rho'])
        gx, gy = gaps[model]
        if gy is not None:
            gp = ax.twinx()
            gp.spines['right'].set_position(                       # its own scale, further out
                ('axes', EN_OFFSET_LAST if last else EN_OFFSET))
            gp.plot(gx, gy, color=EN_C['gap'], ls='--', lw=1.8)
            gp.axhline(0, color=EN_C['gap'], ls=':', lw=0.9)
            gp.set_ylim(bottom=max(gp.get_ylim()[0], floor * gy.max()))
            gp.grid(False)
            if last: gp.set_ylabel(r'$\Delta$RankMe', fontsize=fs)
            _en_axis(gp, EN_C['gap'])
        ax.set_xscale('log')
        ax.set_title(model_label(model), fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if i % 3 == 0: ax.set_ylabel('RankMe', fontsize=fs)
        if i >= len(models) - 3: ax.set_xlabel('Pretraining tokens', fontsize=fs)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.045, right=0.905, top=0.94, bottom=0.15,  # the offset spine
                        wspace=0.42, hspace=0.42)                       # needs room tight_layout misses
    fig.legend(handles=[Line2D([], [], color=EN_C[k], ls=s,
                           lw=3.0 if k == 'rankme' else 1.8, label=lbl)
                        for lbl, k, s in EN_KEY],
               loc='lower center', ncol=len(EN_KEY), fontsize=fs, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = entropy_neurons(save='analysis/figures/standalone/entropy_neurons.pdf')


# %% [markdown]
# ## ΔS of one layer at the warmup trough
#
# A single slice of the ledger stack rather than the whole trajectory: at the trough of the
# warmup crash, what Layer 9 does, what it does with Layers 5-8 zeroed, and what Layer 5 does
# beside it. One panel per model, each bar the three terms stacked — positives up from zero,
# negatives down — with the black rule at the measured total.

# %%
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

L9_MODELS = ('pythia-1b-deduped', 'OLMo-2-0425-1B')
# (0-indexed block, results dir or None for the baseline, bar label)
L9_COLS = ((8, None, 'Layer 9'), (8, 'ablate_blk4-7', 'Layer 9\n(Layers 5-8 ablated)'),
           (4, None, 'Layer 5'))
L9_FIG = (13.0, 4.8)
L9_ALPHA = 0.55   # signed_stack's fill alpha: the bars must read as slices of that stack


def ledger_at(model, blk, src, step):
    """(the three ledger terms, their total) for one block at one checkpoint."""
    v = [float(_lib.get_y(src, model, (f'blk{blk}', 'block_ledger'), t, step))
         for t, _lbl, _c in LEDGER_TERMS]
    return v, sum(v)


def ledger_bars(models=L9_MODELS, cols=L9_COLS, at='trough', figsize=L9_FIG, fs=LS_FS,
                width=0.62, legend_ncol=4, save=None):
    """One panel per model, one stacked bar per (layer, run). at: 'trough' | 'peak' | 'end'."""
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    for i, model in enumerate(models):
        ax = axes[0, i]
        base = cfg_of(model)
        xs, _rv = _lib.rankme_series(base, model, HOOK, include_init=False)
        lo, hi = _lib.rankme_phases(base, model, HOOK, include_init=False)
        tok = xs[{'trough': lo, 'peak': hi, 'end': len(xs) - 1}[at]]
        for j, (blk, src, label) in enumerate(cols):
            src = src or base
            # each run has its own checkpoint grid, so match the moment on tokens
            _v, steps = _lib.get_ys(src, model, (f'blk{blk}', 'block_ledger'), 'chi')
            sx = np.asarray(_lib.get_xs_tokens(model, steps), float)
            step = steps[int(np.argmin(np.abs(np.log(np.maximum(sx, 1)) - np.log(tok))))]
            vals, tot = ledger_at(model, blk, src, step)
            pos = neg = 0.0
            for v, (_t, _lbl, c) in zip(vals, LEDGER_TERMS):
                ax.bar(j, v, width, bottom=pos if v >= 0 else neg, color=c,
                       alpha=L9_ALPHA, lw=0)
                if v >= 0: pos += v
                else: neg += v
            ax.plot([j - width / 2, j + width / 2], [tot, tot], color='k', lw=2.5, zorder=3)
        ax.axhline(0, color='0.35', lw=0.8)
        ax.set_xticks(range(len(cols)), [c[2] for c in cols], fontsize=fs - 3)
        ax.set_title(model_label(model), fontsize=fs)
        ax.tick_params(axis='y', labelsize=fs - 3)
        if i == 0: ax.set_ylabel(r'$\Delta S$', fontsize=fs)
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles=[Patch(color=c, alpha=L9_ALPHA, label=lbl) for _t, lbl, c in LEDGER_TERMS]
               + [Line2D([], [], color='k', lw=2.5, label=r'$\Delta S$ total')],
               loc='lower center', ncol=legend_ncol, fontsize=fs, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = ledger_bars(save='analysis/figures/standalone/ledger_layer9_trough.pdf')


# %% [markdown]
# ## Layer-wise entropy through training
#
# Entropy against depth at four moments — before the warmup drop, at the drop, at the entropy
# peak, and at the end — one row of snapshots per model, with that model's RankMe trace beside
# them and a leader line from each snapshot to the moment it was taken. OLMo 2 1B has only one
# pre-warmup checkpoint, its step 0, which has no place on a log axis; its first leader
# therefore points at the left edge of the trace.

# %%
# nanochat between the two 1Bs: it has a compression valley and OLMo does not
DEPTH_MODELS = ('pythia-1b-deduped', 'nanochat-d12', 'OLMo-2-0425-1B')
# every model but OLMo 2 7B carries a step-0 checkpoint, so the first snapshot IS the
# initialisation; OLMo 2 7B's is its earliest available instead
DEPTH_SNAPS = ('Init', 'Entropy trough', 'Entropy peak', 'End of pretraining')
# DEPTH_CMAP, DEPTH_SPAN = 'viridis', (0.0, 0.88)   # stop just short of pale yellow
DEPTH_CMAP, DEPTH_SPAN = 'viridis', (0.0, 0.94)   # stop just short of pale yellow
DEPTH_BG = PANEL_BG                               # the spectra panels' tint
DEPTH_MUTE = '0.72'
DEPTH_RK = '0.62'      # the locator trace: present, not competing with the profiles                               # the snapshots a panel is not showing
DEPTH_FIG = (16.5, 12.0)
DEPTH_RATIO = 0.42                                # RankMe row height, relative to a snapshot row


def depth_nodes(model):
    """The residual read-points in order: 0 (embedding out), 1..L (after each layer), afn."""
    L = N_BLOCKS[model]
    return ([f'blk{l}.attn.in' for l in range(L)] + ['before_final_norm', 'after_final_norm'],
            [*range(L + 1), 'afn'])


def depth_pct(n):
    """Read-point index -> per cent of depth, so models of different L share one axis and one
    set of ticks (a raw index axis gave nanochat 4 labels and the 7Bs 9)."""
    return np.linspace(0, 100, n)


def depth_landmarks(model, base=None):
    """(aligned steps, tokens, the four landmark row indices). map_tokens gives the step-0 row
    a DISPLAY position, not a token count, so the steps must be carried alongside — matching a
    landmark back by tokens snaps the init onto the first real checkpoint. The trough is the
    minimum of the final stream over the first half; the peak is the MIDDLE layer's maximum
    after it, which is the shape the figure is about (the final stream peaks later)."""
    base = base or cfg_of(model)
    rv, steps = _lib.get_ys(base, model, HOOK, 'rankme')
    xs, keep = _lib.map_tokens(model, steps, True)
    steps, xs, rv = np.asarray(steps)[keep], xs[keep], np.asarray(rv, float)[keep]
    mid = (f'blk{N_BLOCKS[model] // 2}.attn.in', 'acts_centered')
    rm = np.asarray(_lib.get_ys(base, model, mid, 'rankme')[0], float)[keep]
    lo = int(np.argmin(rv[:max(2, len(rv) // 2)]))
    hi = lo + int(np.argmax(rm[lo:]))
    return steps, xs, (0, lo, hi, len(xs) - 1)


def _even_log(cands, xs, lo_t, hi_t, max_n, tol):
    """Up to max_n of `cands` sitting as evenly as possible in log tokens between lo_t and
    hi_t. Returns [] rather than an uneven set: a ghost row that is not log-spaced misreads
    as a rate, so it is better to draw none."""
    for n in range(min(max_n, len(cands)), 0, -1):
        tgt = np.geomspace(lo_t, hi_t, n + 2)[1:-1]
        pick = sorted({min(cands, key=lambda c: abs(np.log(xs[c]) - np.log(t))) for t in tgt})
        if len(pick) < n:
            continue
        edges = np.log([lo_t, *xs[pick], hi_t])
        g = np.diff(edges)
        if g.min() > 0 and g.max() / g.min() <= tol:
            return pick
    return []


def depth_ghosts(model, marks, xs, max_n=3, tol=2.6):
    """Per landmark, the in-between checkpoints to ghost in behind it: those lying between the
    previous landmark and this one. The first landmark has none by construction, and a model
    with nothing in a gap (OLMo 2 1B, between its step 0 and the trough) gets none there."""
    out = [[]]
    for a, b in zip(marks, marks[1:]):
        cands = [c for c in range(a + 1, b)]
        out.append(_even_log(cands, xs, xs[a], xs[b], max_n, tol) if cands else [])
    return out


def depth_all_profiles(model, yvar='matrix_entropy', src=None):
    """(tokens, S over depth) for EVERY checkpoint, plus the landmark row indices — the run as
    a continuum, for drawing faintly behind the landmarks."""
    base = cfg_of(model)
    steps, xs, marks = depth_landmarks(model, base)
    nodes, _labels = depth_nodes(model)
    if src and src != base:
        _v, ss = _lib.get_ys(src, model, (nodes[0], 'acts_centered'), yvar)
        pick = lambda st: min(ss, key=lambda t: abs(np.log(max(t, 1)) - np.log(max(st, 1))))
    else:
        pick = lambda st: st
    prof = [(xs[i], np.array([float(_lib.get_y(src or base, model, (n, 'acts_centered'), yvar,
                                               pick(int(steps[i])))) for n in nodes]))
            for i in range(len(steps))]
    return prof, marks


def depth_snapshots(model, yvar='matrix_entropy', src=None):
    """(x positions, tick labels, [(tokens, S over depth)] at the four landmarks). src reads
    the profiles from another run at the same checkpoints; the landmarks stay the baseline's."""
    base = cfg_of(model)
    steps, xs, marks = depth_landmarks(model, base)
    nodes, labels = depth_nodes(model)
    if src and src != base:                  # another run: match on the STEP, not on tokens
        _v, ss = _lib.get_ys(src, model, (nodes[0], 'acts_centered'), yvar)
        pick = lambda st: min(ss, key=lambda t: abs(np.log(max(t, 1)) - np.log(max(st, 1))))
    else:
        pick = lambda st: st
    prof = lambda i: np.array([float(_lib.get_y(src or base, model, (n, 'acts_centered'),
                                               yvar, pick(int(steps[i])))) for n in nodes])
    out = [(xs[i], prof(i)) for i in marks]
    ghosts = [[(xs[g], prof(g)) for g in row] for row in depth_ghosts(model, marks, xs)]
    return np.arange(len(nodes)), labels, out, ghosts


def _depth_style(ax, fs, bg=DEPTH_BG):
    """The spectra panels' look: tinted ground, white gridlines, quiet spines."""
    ax.set_facecolor(bg)
    ax.grid(True, which='major', color='white', lw=1.2)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(RULE); sp.set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelcolor=INK, labelsize=fs - 4, length=3)


def _rk_strip(ax, model, fs, marks=(), cols=None, lw=1.6, labeltop=False,
              edge_ticks=True, rule_alpha=0.71):
    """The locator trace: RankMe over tokens, kept quiet. Ticks are capped so a long run does
    not end up with twice the labels of a short one. edge_ticks: the marks on the strip's two
    edges — the rules alone locate the moment, and at small strip heights the ticks are the
    loudest thing in the panel."""
    rx, rv = _lib.rankme_series(cfg_of(model), model, HOOK, include_init=True)
    ax.plot(rx, rv, color=DEPTH_RK, lw=lw)
    ax.set(xscale='log', xlim=(rx[0] * 0.92, rx[-1] * 1.08))
    ax.xaxis.set_major_locator(LogLocator(numticks=5))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel('RankMe', fontsize=fs - 4, color=MUTE)
    ax.locator_params(axis='y', nbins=2)
    _depth_style(ax, fs - 1)
    if labeltop:                     # the row below is flush against the strip
        ax.xaxis.set_ticks_position('top'); ax.xaxis.set_tick_params(labeltop=True,
                                                                     labelbottom=False)
    ylim = ax.get_ylim()          # the edge ticks sit ON the limits; plotting them would
    for i, tok in enumerate(marks):   # otherwise autoscale the axis outward around them
        # as in the phase figure: a dashed rule across the strip at the moment, and a tick on
        # each edge. Leaders land on the EDGE, not on the curve, so they never cross the trace
        _lib._mark_lines(ax, [tok], 'x', MARK, ls='--', lw=1.6, alpha=rule_alpha)
        c = cols[i] if cols is not None else INK
        for y in (ylim if edge_ticks else ()):
            ax.plot([tok], [y], marker='|', ms=7, mew=1.3, color=c, clip_on=False, zorder=6)
    ax.set_ylim(*ylim)
    return rx, rv


def _ghost_color(tok, snaps, i, cols, span=None):
    """A ghost is coloured as if it were part of the full-training sweep: interpolate the two
    landmark colours by where it sits in log tokens between them."""
    t0, t1 = snaps[i - 1][0], snaps[i][0]
    f = np.clip((np.log(tok) - np.log(t0)) / max(np.log(t1) - np.log(t0), 1e-9), 0, 1)
    span = span or DEPTH_SPAN
    lo = span[0] + (span[1] - span[0]) * (i - 1) / (len(snaps) - 1)
    hi = span[0] + (span[1] - span[0]) * i / (len(snaps) - 1)
    return plt.get_cmap(DEPTH_CMAP)(lo + (hi - lo) * f)


def depth_grid(models=DEPTH_MODELS, side='top', figsize=(16.0, 12.5), ratio=0.42,
               fs=LS_FS, ylab=r'$S_\ell$', others=True, ghosts=True, shared_y=True,
               stacking_dir='vertical', save=None):
    """Per model: a RankMe locator and a row of depth profiles, joined by leader lines. The
    profiles are the figure; the model name sits in the left margin so the panels can grow.
    others: grey in the OTHER landmarks behind each panel. ghosts: the in-between checkpoints
    on the way to this one, coloured along the same sweep. shared_y: True gives every model
    one y range, so a fall of n nats is the same drop everywhere — per-model limits made
    OLMo 2 1B's shallow fall fill its panel and read as deep as Pythia's. 'span' keeps one
    range WIDTH but centres it on each model, which costs the absolute comparison and buys
    back the vertical space; False is per model."""
    prof = {m: depth_snapshots(m) for m in models}
    if shared_y:
        spans = [(min(v.min() for _t, v in p[2]), max(v.max() for _t, v in p[2]))
                 for p in prof.values()]
        width_y = max(b - a for a, b in spans)
    n = len(DEPTH_SNAPS)
    cols = plt.get_cmap(DEPTH_CMAP)(np.linspace(*DEPTH_SPAN, n))
    if stacking_dir == 'horizontal':
        figsize = (figsize[0] * len(models), figsize[1])
        heights = [h for _ in models for h in ((ratio, 1) if side == 'top' else (1, ratio))]
        fig = plt.figure(figsize=figsize)
        left, right = 0.02, 0.995
        gs = fig.add_gridspec(2, len(models) * n,
                              height_ratios=((ratio, 1) if side == 'top' else (1, ratio)),
                              hspace=0.42, wspace=0.05,
                              left=left, right=right, top=0.955, bottom=0.05)
        for mi, model in enumerate(models):
            x, _labels, snaps, ghost_rows = prof[model]
            pct = depth_pct(len(x))
            lo, hi = min(v.min() for _t, v in snaps), max(v.max() for _t, v in snaps)
            if shared_y == 'span':
                mid = (lo + hi) / 2
                lo, hi = mid - width_y / 2, mid + width_y / 2
            elif shared_y:
                lo, hi = min(a for a, _b in spans), max(b for _a, b in spans)
            lo, hi = lo - 0.15, hi + 0.15
            c0, c1 = mi * n, (mi + 1) * n
            r_rk, r_sn = (0, 1) if side == 'top' else (1, 0)
            rk = fig.add_subplot(gs[r_rk, c0:c1])
            rx, rv = _rk_strip(rk, model, fs, [t for t, _v in snaps], cols)
            axes_row = []
            for i, (tok, v) in enumerate(snaps):
                ax = fig.add_subplot(gs[r_sn, c0 + i])
                axes_row.append(ax)
                if others:
                    for j, (_t2, v2) in enumerate(snaps):
                        if j != i: ax.plot(pct, v2, color=DEPTH_MUTE, lw=1.4, zorder=2)
                if ghosts:
                    for gt, gv in ghost_rows[i]:
                        ax.plot(pct, gv, color=_ghost_color(gt, snaps, i, cols), lw=1.8,
                                alpha=0.42, zorder=2.5)
                ax.plot(pct, v, color=cols[i], lw=4.0, zorder=3)
                ax.set(ylim=(lo, hi), xlim=(0, 100))
                ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                _depth_style(ax, fs)
                if i: ax.tick_params(labelleft=False)
                else: ax.set_ylabel(ylab, fontsize=fs)
                if mi == len(models) - 1: ax.set_xlabel('Depth (%)', fontsize=fs)
                else: ax.tick_params(labelbottom=False)
                edge = rk.get_ylim()[0 if side == 'top' else 1]
                fig.add_artist(ConnectionPatch(
                    xyA=(0.5, 1.0 if side == 'top' else 0.0), coordsA=ax.transAxes,
                    xyB=(tok, edge), coordsB=rk.transData,
                    color=cols[i], lw=1.0, alpha=0.65, zorder=0))
            p0, p1 = axes_row[0].get_position(), rk.get_position()
            fig.text((p0.x0 + p0.x1) / 2, max(p0.y1, p1.y1) + 0.22 / figsize[1], model_label(model),
                     va='bottom', ha='center', fontsize=fs + 1)
    else:
        heights = [h for _ in models for h in ((ratio, 1) if side == 'top' else (1, ratio))]
        fig = plt.figure(figsize=figsize)
        left, right = 0.075, 0.995
        gs = fig.add_gridspec(2 * len(models), n, height_ratios=heights, hspace=0.42, wspace=0.05,
                              left=left, right=right, top=0.955, bottom=0.05)
        for mi, model in enumerate(models):
            r_rk, r_sn = (2 * mi, 2 * mi + 1) if side == 'top' else (2 * mi + 1, 2 * mi)
            x, _labels, snaps, ghost_rows = prof[model]
            pct = depth_pct(len(x))
            lo, hi = min(v.min() for _t, v in snaps), max(v.max() for _t, v in snaps)
            if shared_y == 'span':
                mid = (lo + hi) / 2
                lo, hi = mid - width_y / 2, mid + width_y / 2
            elif shared_y:
                lo, hi = min(a for a, _b in spans), max(b for _a, b in spans)
            lo, hi = lo - 0.15, hi + 0.15
            rk = fig.add_subplot(gs[r_rk, :])
            rx, rv = _rk_strip(rk, model, fs, [t for t, _v in snaps], cols)
            axes_row = []
            for i, (tok, v) in enumerate(snaps):
                ax = fig.add_subplot(gs[r_sn, i])
                axes_row.append(ax)
                if others:
                    for j, (_t2, v2) in enumerate(snaps):
                        if j != i: ax.plot(pct, v2, color=DEPTH_MUTE, lw=1.4, zorder=2)
                if ghosts:
                    for gt, gv in ghost_rows[i]:
                        ax.plot(pct, gv, color=_ghost_color(gt, snaps, i, cols), lw=1.8,
                                alpha=0.42, zorder=2.5)
                ax.plot(pct, v, color=cols[i], lw=4.0, zorder=3)
                ax.set(ylim=(lo, hi), xlim=(0, 100))
                ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
                _depth_style(ax, fs)
                if i: ax.tick_params(labelleft=False)
                else: ax.set_ylabel(ylab, fontsize=fs)
                if mi == len(models) - 1: ax.set_xlabel('Depth (%)', fontsize=fs)
                else: ax.tick_params(labelbottom=False)
                edge = rk.get_ylim()[0 if side == 'top' else 1]
                fig.add_artist(ConnectionPatch(
                    xyA=(0.5, 1.0 if side == 'top' else 0.0), coordsA=ax.transAxes,
                    xyB=(tok, edge), coordsB=rk.transData,
                    color=cols[i], lw=1.0, alpha=0.65, zorder=0))
            p0, p1 = axes_row[0].get_position(), rk.get_position()
            fig.text((left + right) / 2, max(p0.y1, p1.y1) + 0.22 / figsize[1], model_label(model),
                     va='bottom', ha='center', fontsize=fs + 1)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig



# %%
_ = depth_grid(others=False, ghosts=True,
               save='analysis/figures/standalone/depth_entropy_snapshots.pdf')


# %%
_ = depth_grid(others=False, ghosts=True, stacking_dir='horizontal', figsize=(10.0, 4.17), save='analysis/figures/standalone/depth_entropy_snapshots_hor.pdf')

# %% [markdown]
# ## The same snapshots merged, under three interventions
#
# Pythia 1B only, the four snapshots overlaid in one panel: the baseline, the run with Layer 4
# zeroed, and the content-tokens-only measurement. Dotted segments follow the animation
# notebook's convention — the stretch of depth an ablation skips, and the final norm's step,
# are bridges rather than measured transitions.

# %%
from matplotlib.lines import Line2D

ALL_TOK, NO_DELIM = 'All tokens', 'No delimiters or first tokens'
MERGE_RUNS = ((ALL_TOK, None), ('Layer 4 ablated', 'ablate_blk3'),
              (NO_DELIM, 'content_tokens_full'))
PAIR_RUNS = ((ALL_TOK, None), (NO_DELIM, 'content_tokens_full'))
MERGE_MODEL = 'pythia-1b-deduped'
MERGE_ABLATED = {'ablate_blk3': (3, 4)}     # residual span a zeroed write leaves untouched
MERGE_FIG = (15.5, 4.6)


def _bridged(ax, x, y, bridges, color, lw, fs=None):
    """Draw y over depth with `bridges` (index pairs) dotted and the rest solid."""
    for a, b in bridges:
        ax.plot(x[a:b + 1], y[a:b + 1], ls=':', color=color, lw=lw, zorder=3)
    cuts = sorted(bridges)
    edges = [0] + [i for ab in cuts for i in ab] + [len(x) - 1]
    for a, b in zip(edges[::2], edges[1::2]):
        if b > a: ax.plot(x[a:b + 1], y[a:b + 1], color=color, lw=lw, zorder=3)


def merged_depth(model=MERGE_MODEL, runs=MERGE_RUNS, figsize=MERGE_FIG, fs=LS_FS,
                 ylab=r'$S_\ell$', legend_ncol=4, save=None):
    """One panel per run, the four snapshots overlaid; the token axis is gone, so the key
    names the moments instead."""
    n = len(DEPTH_SNAPS)
    cols = plt.get_cmap(DEPTH_CMAP)(np.linspace(*DEPTH_SPAN, n))
    fig, axes = plt.subplots(1, len(runs), figsize=figsize, squeeze=False, sharey=True)
    L = N_BLOCKS[model]
    for j, (label, src) in enumerate(runs):
        ax = axes[0, j]
        x, _labels, snaps, ghosts = depth_snapshots(model, src=src)
        pct = depth_pct(len(x))
        bridges = [MERGE_ABLATED[src]] if src in MERGE_ABLATED else []
        for i, (_tok, v) in enumerate(snaps):
            _bridged(ax, pct, v, bridges, cols[i], 3.0)
        ax.set_xlim(0, 100)
        ax.set_title(label, fontsize=fs)
        ax.set_xlabel('Depth (%)', fontsize=fs)
        _depth_style(ax, fs)
        if j == 0: ax.set_ylabel(ylab, fontsize=fs)
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles=[Line2D([], [], color=cols[i], lw=3.0, label=DEPTH_SNAPS[i])
                        for i in range(n)],
               loc='lower center', ncol=legend_ncol, fontsize=fs, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


def merged_depth_combo(top=(MERGE_MODEL, MERGE_RUNS),
                       bottom=(('OLMo-2-0425-1B', PAIR_RUNS), ('nanochat-d12', PAIR_RUNS)),
                       figsize=(15.0, 8.4), fs=LS_FS, ylab=r'$S_\ell$', legend_ncol=4,
                       ratio=0.85, save=None):
    """Pythia's three interventions across the top, then one pair per model beneath — 3 over 4.
    The lower row is a little smaller: it is the control, not the result."""
    n = len(DEPTH_SNAPS)
    cols = plt.get_cmap(DEPTH_CMAP)(np.linspace(*DEPTH_SPAN, n))
    nb = sum(len(r) for _m, r in bottom)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, np.lcm(len(top[1]), nb), height_ratios=[1, ratio],
                          hspace=0.62, wspace=0.32, left=0.06, right=0.995,
                          top=0.90, bottom=0.135)
    span = gs.ncols // len(top[1])
    base = None
    for j, (label, src) in enumerate(top[1]):        # one y per model: the runs are the
        ax = fig.add_subplot(gs[0, j * span:(j + 1) * span], sharey=base)   # comparison
        base = base or ax
        _merged_panel(ax, top[0], src, label, cols, fs, ylab if j == 0 else None)
        if j: ax.tick_params(labelleft=False)
    q0, q1 = fig.axes[0].get_position(), fig.axes[-1].get_position()
    fig.text((q0.x0 + q1.x1) / 2, q0.y1 + 0.30 / figsize[1], model_label(top[0]),
             va='bottom', ha='center', fontsize=fs + 1)
    span = gs.ncols // nb
    c = 0
    for model, runs in bottom:
        first, base = len(fig.axes), None
        for j, (label, src) in enumerate(runs):
            ax = fig.add_subplot(gs[1, c * span:(c + 1) * span], sharey=base)
            base = base or ax
            _merged_panel(ax, model, src, label, cols, fs - 1, ylab if c == 0 else None)
            if j: ax.tick_params(labelleft=False)
            c += 1
        q0, q1 = fig.axes[first].get_position(), fig.axes[-1].get_position()
        fig.text((q0.x0 + q1.x1) / 2, q0.y1 + 0.30 / figsize[1], model_label(model),
                 va='bottom', ha='center', fontsize=fs + 1)
    fig.legend(handles=[Line2D([], [], color=cols[i], lw=2.8, label=DEPTH_SNAPS[i])
                        for i in range(n)],
               loc='lower center', ncol=legend_ncol, fontsize=fs, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


def _merged_panel(ax, model, src, label, cols, fs, ylab):
    """One run's four snapshots overlaid against per-cent depth."""
    x, _labels, snaps, _gh = depth_snapshots(model, src=src)
    pct = depth_pct(len(x))
    bridges = [MERGE_ABLATED[src]] if src in MERGE_ABLATED else []
    for i, (_tok, v) in enumerate(snaps):
        _bridged(ax, pct, v, bridges, cols[i], 2.8)
    ax.set(xlim=(0, 100))
    ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
    ax.set_title(label, fontsize=fs - 2)
    ax.set_xlabel('Depth (%)', fontsize=fs)
    _depth_style(ax, fs)
    if ylab: ax.set_ylabel(ylab, fontsize=fs)


# %%
_ = merged_depth(save='analysis/figures/standalone/depth_entropy_merged.pdf')

# %%
_ = merged_depth_combo(save='analysis/figures/standalone/depth_entropy_combo.pdf')


# %% [markdown]
# ## The merged snapshots, every model
#
# One panel per model, the same four snapshots overlaid: once on the packed baseline and once
# on the content-tokens-only measurement.

# %%
def merged_depth_panel(ax, model, src=None, fs=LS_FS, xlabel=True, ylabel=True,
                       ylab=r'$S_\ell$'):
    """The four depth profiles of one model in one panel, against per-cent depth."""
    cols = plt.get_cmap(DEPTH_CMAP)(np.linspace(*DEPTH_SPAN, len(DEPTH_SNAPS)))
    x, _labels, snaps, ghosts = depth_snapshots(model, src=src)
    pct = depth_pct(len(x))
    for i, (_tok, v) in enumerate(snaps):
        _bridged(ax, pct, v, [], cols[i], 2.8)
    ax.set_xlim(0, 100)
    ax.set_title(model_label(model), fontsize=fs)
    _depth_style(ax, fs)
    if ylabel: ax.set_ylabel(ylab, fontsize=fs)
    if xlabel: ax.set_xlabel('Depth (%)', fontsize=fs)


DEPTH_KEY = [Line2D([], [], lw=2.8, label=lbl,
                    color=plt.get_cmap(DEPTH_CMAP)(np.linspace(*DEPTH_SPAN, len(DEPTH_SNAPS))[i]))
             for i, lbl in enumerate(DEPTH_SNAPS)]

def depth_pair_grid(models=LEDGER_SPEC6, runs=PAIR_RUNS,
                    groups=2, figsize=(17.0, 9.0), fs=LS_FS, ylab=r'$S_\ell$', legend_ncol=4,
                    name_pad=0.026, ghosts=True, span=(0.0, 0.97), alpha=0.55,
                    save=None):
    """Rows = model, columns = run, with the models split into `groups` side-by-side blocks so
    the figure is not one tall column. y is shared across each model's pair, which is the
    point: read at its own scale each panel looks similar, and the comparison against the
    baseline is what the figure is for. The model name sits centred above its pair.
    ghosts draws the in-between checkpoints faintly — the same ones depth_ghosts picks for
    the 4-snapshot figure, so the two agree — which makes the landmarks read as highlights on
    a run rather than as four isolated states. span is the sweep\'s slice of viridis, wider
    here because the faint lines need the extra separation. The legend names only the
    landmarks."""
    n, nr = len(DEPTH_SNAPS), -(-len(models) // groups)
    cmap = plt.get_cmap(DEPTH_CMAP)
    nrun = len(runs)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nr, groups * nrun + (groups - 1),
                          width_ratios=[w for g in range(groups)
                                        for w in ([1] * nrun if g == 0 else [0.34] + [1] * nrun)],
                          hspace=0.52, wspace=0.10, left=0.045, right=0.995,
                          top=0.93, bottom=0.11)
    for mi, model in enumerate(models):
        g, i = divmod(mi, nr)
        c0 = g * (nrun + 1)
        base = None
        for j, (label, src) in enumerate(runs):
            ax = fig.add_subplot(gs[i, c0 + j], sharey=base)
            base = base or ax
            x, _labels, snaps, ghost_rows = depth_snapshots(model, src=src)
            pct = depth_pct(len(x))
            cols = [cmap(span[0] + (span[1] - span[0]) * si / (n - 1)) for si in range(n)]
            if ghosts:
                for si, row in enumerate(ghost_rows):     # si, not i: i is the grid row
                    for gt, gv in row:
                        ax.plot(pct, gv, color=_ghost_color(gt, snaps, si, cols, span),
                                lw=1.5, alpha=alpha, zorder=2)
            for si, (_tok, v) in enumerate(snaps):
                _bridged(ax, pct, v, [], cols[si], 2.8)
            ax.set(xlim=(0, 100))
            ax.xaxis.set_major_locator(MaxNLocator(4, prune='upper'))
            _depth_style(ax, fs)
            if j: ax.tick_params(labelleft=False)
            else: ax.set_ylabel(ylab, fontsize=fs)
            if i == nr - 1: ax.set_xlabel('Depth (%)', fontsize=fs)
            ax.set_title(label, fontsize=fs - 3, pad=3)
        p0 = fig.axes[-nrun].get_position()                 # centred over the model's pair
        p1 = fig.axes[-1].get_position()
        # just clear of its own run titles: any higher and it reads as belonging to the row
        # above, which the generous hspace makes very easy to do
        fig.text((p0.x0 + p1.x1) / 2, p1.y1 + name_pad, model_label(model),
                 va='bottom', ha='center', fontsize=fs)
    key = [cmap(span[0] + (span[1] - span[0]) * i / (n - 1)) for i in range(n)]
    fig.legend(handles=[Line2D([], [], color=key[i], lw=2.8, label=DEPTH_SNAPS[i])
                        for i in range(n)],
               loc='lower center', ncol=legend_ncol, fontsize=fs, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = depth_pair_grid(save='analysis/figures/standalone/depth_entropy_pair.pdf', span=(0, 0.94))


# %%
_ = panel_grid(LEDGER_SPEC6, merged_depth_panel, legend_ncol=4, handles=DEPTH_KEY,
               save='analysis/figures/standalone/depth_entropy_all.pdf')

# %%
_ = panel_grid(LEDGER_SPEC6,
               lambda ax, m, **kw: merged_depth_panel(ax, m, src='content_tokens_full', **kw),
               legend_ncol=4, handles=DEPTH_KEY,
               save='analysis/figures/standalone/depth_entropy_all_content.pdf')


# %% [markdown]
# ## Block↔block coupling through training
#
# Per model: the RankMe trace, then the pairwise coupling of the block writes at a series of
# checkpoints — CKA (shared subspace, unsigned) above, cross-covariance trace (net reinforce
# or cancel, energy-weighted) below. Checkpoints are picked to sit as evenly as possible in
# log tokens; past ~16 the late ones start bunching, so that is the resolution ceiling.

# %%
CPL_KINDS = (('cka', 'CKA', 'Blues', False),
             ('signed_trace', 'Cross-cov\ntrace', 'coolwarm', True))
CPL_FIG3, CPL_FIG6 = (17.0, 11.0), (19.0, 21.0)


def _even_pick(lg, k):
    """Indices of k points out of the sorted log positions `lg`, as evenly spaced as that set
    allows, always including both ends. Two candidates are tried and the one with the better
    max/min gap ratio wins: snapping to evenly spaced targets, and a bottleneck sweep that
    maximises the smallest gap (anchored at each end in turn — anchoring only on the left
    leaves the last pick short of the end, and dragging it there opens a double-width gap).
    Neither dominates: the sweep is far better when the checkpoints are dense enough to
    approximate the target spacing, the snap when they are not."""
    n = len(lg)
    k = min(k, n)
    cands = []
    idx = []                                             # (a) nearest to even targets, monotone
    for i, t in enumerate(np.linspace(lg[0], lg[-1], k)):
        lo = idx[-1] + 1 if idx else 0
        idx.append(lo + int(np.argmin(np.abs(lg[lo:n - (k - i) + 1] - t))))
    cands.append(idx)
    for rev in (False, True):                            # (b) bottleneck sweep from each end
        v = (lg[-1] - lg[::-1]) if rev else lg
        def take(g):
            got = [0]
            for j in range(1, n):
                if v[j] - v[got[-1]] >= g: got.append(j)
            return got
        lo, hi = 0.0, lg[-1] - lg[0]
        for _ in range(60):
            mid = (lo + hi) / 2
            if len(take(mid)) >= k: lo = mid
            else: hi = mid
        got = take(lo)[:k]
        got = sorted(n - 1 - j for j in got) if rev else got
        got[0], got[-1] = 0, n - 1
        cands.append(sorted(set(got)))
    ratio = lambda ix: (lambda d: d.max() / max(d.min(), 1e-12))(np.diff(lg[ix]))
    return min(cands, key=ratio)


def log_steps(model, k, node=('', 'block_block_coupling'), yvar='cka'):
    """k checkpoints spaced as evenly in log tokens as the run allows, from the ones THIS
    quantity carries. How even that can be is set by the collection: the early checkpoints are
    a doubling ladder (0.30 decades), so a target spacing that is not near a multiple of it
    has to alternate 0.30 and 0.60."""
    _v, steps = _lib.get_ys(cfg_of(model), model, node, yvar)
    xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
    steps, xs = np.asarray(steps)[xs > 0], xs[xs > 0]
    keep = np.concatenate([[True], np.diff(xs) > 0])     # tied token counts: keep one
    steps, xs = steps[keep], xs[keep]
    idx = _even_pick(np.log(xs), k)
    return steps[idx], xs[idx]


# bare number: the unit is stated once, as an axis label under the block
TOK_LABEL = lambda t: r'$10^{' + f'{np.log10(t):.1f}' + r'}$'
TOK_AXIS = 'Pretraining tokens'


def _hm_ticks(ax, n, fs, xaxis, yaxis, want=None):
    """First / middle / last layer index. Only the first panel of a row is labelled — every
    panel shares the layer axis — so there is no neighbour for the last label to run into.
    Labels only: tick marks between flush cells would read as gridlines."""
    pos = [0, n // 2, n - 1]
    lab = [str(q + 1) for q in pos]                      # layers are one-indexed in figures
    ax.set_xticks(pos, lab if xaxis else [''] * len(pos), fontsize=fs - 3, rotation=0)
    ax.set_yticks(pos, lab if yaxis else [''] * len(pos), fontsize=fs - 3, rotation=0)
    ax.tick_params(length=0, colors=MUTE, labelcolor=INK, pad=2)
    # the outer labels sit INSIDE the panel: centred on the edge cell they hang over the seam
    # with the next heatmap and read as if they belonged to both
    for t, ha in zip(ax.get_xticklabels(), ('left', 'center', 'right')):
        t.set_ha(ha)
    for t, va in zip(ax.get_yticklabels(), ('top', 'center', 'bottom')):
        t.set_va(va)


def coupling_grid(models=DEPTH_MODELS, k=12, split=True, interleave=False, side='top',
                  across=False, pair=('mlp', 'mlp'), width=12.0, fs=LS_FS, ratio=0.80,
                  cbar=0.13, gap=2.4, colgap=3.0, pad=0.26, edge_ticks=False, strip=True,
                  name_pad=None, save=None):   # colgap: the across gutter holds a
                                # colourbar AND the next model's row labels
    """Rows per model: heatmap rows and a RankMe locator, one colourbar per kind on the right.
    split=True halves the checkpoints — above the locator and below it, each column tied to
    its moment by a leader — which fits k of them in k/2 columns: the first half above and the
    second below. interleave=True instead alternates them (even above, odd below), which makes
    every leader near-vertical at the cost of the simple first/second reading. The moments are
    named on the locator, above their own rule, so the columns carry only layer indices.
    name_pad is the model name's height above its block, in inches; None takes the default
    for the layout, which differ because their inter-model gaps do.
    strip=False drops the RankMe locator entirely and names the moments on the heatmap
    columns instead, which is what lets the models stack tightly enough to stay readable at a
    width where laying them out across would be too small.
    across=True lays the models out left to right instead of stacked, each with its own
    colourbars in the gutter after it. pair selects the submatrix (default mlp x mlp; an
    all x all map interleaves two different objects and reads as a chequerboard). Spec and
    drawing come from experiments_lib's matrix helpers, the same ones the animations use."""
    nk, left, right = len(CPL_KINDS), 0.075, 0.995
    hspace, wspace, top, bottom = 0.02, 0.004, 0.965, 0.01
    picked = {m: log_steps(m, k) for m in models}          # <= k: near-duplicates are dropped
    nmax = max(len(t) for _s, t in picked.values())
    ncol = -(-nmax // 2) if split else nmax          # each block gets half the columns
    cell = width * (right - cbar - left) / (ncol + wspace * (ncol - 1))
    fs = fs * float(np.clip(cell / 1.5, 1.0, 1.3))    # a mild lift where the cells are big
    # in ROW-HEIGHT units: on a figure this tall a couple of mm is imperceptible, so the
    # clearance is set as a fraction of a heatmap cell. It applies in BOTH layouts — without
    # it the leaders have no room to be seen between the strip and the row beside it
    per = ([nk * [1]] if not strip else
           [nk * [1], [pad], [ratio], [pad], nk * [1]] if split else
           [[ratio], [pad], nk * [1]] if side == 'top' else [nk * [1], [pad], [ratio]])
    per = [h for grp in per for h in grp]
    nm = len(models)
    if across:                              # models side by side, each block + a bar gutter
        heights, widths = per, [w for mi in range(nm)
                                for w in ([1] * ncol if mi == 0 else [colgap] + [1] * ncol)]
        # `width` sizes ONE model's block, as in the stacked layout, and the figure grows to
        # fit the rest: holding the total fixed would shrink every cell by a factor of nm
        width = cell * (sum(widths) + wspace * (len(widths) - 1)) / (right - left)
    else:
        heights = [h for mi in range(nm) for h in (per if mi == 0 else [gap] + per)]
        widths = [1] * ncol
    nrows, ncols_t, R = len(heights), len(widths), sum(heights)
    figsize = (width, cell * R * (1 + hspace * (nrows - 1) / nrows) / (top - bottom))
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols_t, height_ratios=heights, width_ratios=widths,
                          hspace=hspace, wspace=wspace, left=left,
                          right=right if across else right - cbar, top=top, bottom=bottom)
    for mi, model in enumerate(models):
        base = 0 if across else mi * len(per) + mi   # non-across: each model past the
        cbase = mi * (ncol + 1) if across else 0     # first follows a spacer row
        r_rk = base if (not split and side == 'top') else base + nk + 1   # unused w/o strip
        steps, toks = picked[model]
        n = len(steps)
        half = -(-n // 2)
        top_i = range(0, n, 2) if interleave else range(0, half)
        bot_i = range(1, n, 2) if interleave else range(half, n)
        blocks = ([(range(base, base + nk), range(0, n), 'above')] if not strip else
                  [(range(base, base + nk), top_i, 'above'),
                   (range(r_rk + 2, r_rk + 2 + nk), bot_i, 'below')] if split else
                  [(range(base + 2, base + 2 + nk) if side == 'top' else range(base, base + nk),
                    range(0, n), 'below' if side == 'top' else 'above')])
        cpl = lambda y, st: _lib.get_y(cfg_of(model), model, ('', 'block_block_coupling'),
                                       y, int(st))
        leaves = cpl('leaves', steps[0])
        specs = [_lib.matrix_spec([np.asarray(cpl(key, st), float) for st in steps], leaves,
                                  dict(pair=pair, cmap=cmap, dynamic=sym, title=None,
                                       **({} if sym else dict(vmin=0, vmax=1))))
                 for key, _lbl, cmap, sym in CPL_KINDS]
        nlay = len(specs[0]['frames'][0])
        rk = None
        every = max(1, int(np.ceil(0.62 * fs / 15 / cell)))     # as many as the width holds
        if strip:
            rk = fig.add_subplot(gs[r_rk, cbase:cbase + ncol])
            rx, rv = _rk_strip(rk, model, fs, toks, labeltop=False, edge_ticks=edge_ticks)
            rk.tick_params(labelbottom=False)            # the moments are named on the rules
            tr = offset_copy(rk.transData, fig=fig, y=6, units='points')   # clear of the tick
            for tok in toks[::every]:                        # above its own rule, outside the strip
                rk.text(tok, rk.get_ylim()[1], TOK_LABEL(tok), fontsize=fs - 3, color=INK,
                        ha='center', va='bottom', rotation=0, clip_on=False, transform=tr)
        for rows, cols_i, where in blocks:
            rows = list(rows)
            for j, r in enumerate(rows):
                for c, i in enumerate(cols_i):
                    ax = fig.add_subplot(gs[r, cbase + c])
                    im = _lib.draw_matrix(ax, specs[j], i)
                    # x and y are the same axis, so the y labels carry the range on their
                    # own once the cells get too narrow to hold a readable row of numbers
                    _hm_ticks(ax, nlay, fs, xaxis=(j == len(rows) - 1 and c == 0),
                              yaxis=(c == 0))
                    if rk is None and j == 0 and i % every == 0:   # no locator: label here
                        ax.set_title(TOK_LABEL(toks[i]), fontsize=fs - 3, pad=4)
                    if c == 0:                          # horizontal, clear of the tick labels
                        ax.set_ylabel(CPL_KINDS[j][1], fontsize=fs - 2, rotation=0,
                                      ha='center', va='center', labelpad=30)
                    near = (j == len(rows) - 1) if where == 'above' else (j == 0)
                    if near and rk is not None:   # leader from the row beside the strip
                        edge = rk.get_ylim()[1 if where == 'above' else 0]
                        fig.add_artist(ConnectionPatch(
                            xyA=(0.5, 0.0 if where == 'above' else 1.0), coordsA=ax.transAxes,
                            xyB=(toks[i], edge), coordsB=rk.transData,
                            color=MUTE, lw=1.0, alpha=0.28, zorder=0))
                    if c == len(list(cols_i)) - 1:
                        q = ax.get_position()             # one bar per kind, after the block
                        x0 = (q.x1 + 0.008) if across else (right - cbar + 0.012)
                        cax = fig.add_axes((x0, q.y0, 0.010, q.height))
                        cb = fig.colorbar(im, cax=cax)
                        cb.ax.tick_params(labelsize=fs - 5, colors=MUTE, labelcolor=INK)
                        cb.outline.set_edgecolor(RULE)
        q = (rk or fig.axes[-1]).get_position()
        blk0 = gs[base, cbase:cbase + ncol].get_position(fig)
        fig.text(blk0.x0, (q.y1 if rk else blk0.y1) + (0.32 if rk else 0.30) / figsize[1],
                 TOK_AXIS, va='bottom', ha='left', fontsize=fs - 2, color=MUTE)
        off = (name_pad if name_pad is not None else
               0.34 if split else 0.56 if strip else 0.40) / figsize[1]
        blk = gs[base, cbase:cbase + ncol].get_position(fig)
        fig.text((blk.x0 + blk.x1) / 2, blk.y1 + off, model_label(model),
                 va='bottom', ha='center', fontsize=fs + 1)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = coupling_grid(save='analysis/figures/standalone/coupling_snapshots.pdf', split=False, k=8)

# %%
_ = coupling_grid(save='analysis/figures/standalone/coupling_snapshots_across.pdf',
                  split=False, k=8, across=True)

# %%
# No locator: the moments label the columns, so the models stack tightly enough to stay
# readable at a width where laying them out across would be too small.
_ = coupling_grid(save='analysis/figures/standalone/coupling_snapshots_nostrip.pdf',
                  split=False, k=8, strip=False, gap=0.55, width=13.0, name_pad=0.3)

# %%
_ = coupling_grid(LEDGER_SPEC6, k=16, split=False, ratio=1.05,
                  save='analysis/figures/standalone/coupling_snapshots_all.pdf')


# %% [markdown]
# ## Which directions carry the compression
#
# RankMe of the final stream, untouched and after projecting a named part of the unembedding
# out of the representation: the faintest k directions, the single loudest, the loudest k, and
# a random k as the control. A line that comes out flat names the carrier. Each k is its own
# figure — a wider band deletes proportionally more, so the control moves with it and the
# comparison is only valid within a figure.

# %%
DEF_RUNS = (('1%', 'stream_deflation.pt'), ('5%', 'stream_deflation_k05.pt'),
            ('20%', 'stream_deflation_k20.pt'))
DEF_LINES = (('base', 'Untouched', 'k', '-', 2.2),
             ('dark', 'Faintest {p} deleted', 'tab:purple', '-', 1.7),
             ('top1', 'Loudest direction deleted', 'tab:red', '-', 1.7),
             ('topk', 'Loudest {p} deleted', 'tab:orange', '-', 1.7),
             ('random', 'Random {p} deleted (control)', 'tab:blue', '--', 1.7))
DEF_CACHE = {}


def deflation_panel(ax, model, data=None, pct='1%', leaf='after_final_norm', fs=LS_FS,
                    xlabel=True, ylabel=True):
    """One model's RankMe under each named deletion; k is read from the data, not assumed."""
    per = data[model]
    steps = sorted(s for s in per if leaf in per[s])
    x = np.array([per[s]['tokens'] for s in steps], float)
    for key, lab, c, ls, lw in DEF_LINES:
        ax.plot(x, [per[s][leaf][f'rankme_{key}'] for s in steps], ls, color=c, lw=lw,
                label=lab.format(p=pct))
    e = per[steps[-1]]
    ax.set_xscale('log')
    ax.set_title(f'{model_label(model)}  (k = {e["k"]} of {e["d"]})', fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if ylabel: ax.set_ylabel('RankMe', fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)


def deflation_grid(pct, path, models=LEDGER_SPEC6, **kw):
    """The deletion grid for one band width; nanochat sits bottom-left as everywhere else."""
    data = DEF_CACHE.setdefault(path, torch.load(f'data/results/{path}', weights_only=False))
    avail = [m for m in models if m in data and data[m]]
    return panel_grid(avail, lambda ax, m, **k2: deflation_panel(ax, m, data=data, pct=pct, **k2),
                      legend_ncol=5, **kw)


# %%
for _pct, _path in DEF_RUNS:
    _ = deflation_grid(_pct, _path,
                       save=f'analysis/figures/standalone/deflation_k{_pct.rstrip("%")}.pdf')


# %% [markdown]
# ## Dataset-swap control
#
# Each 1B model re-run on the OTHER family's pretraining mix — same texts, re-tokenized.
# Solid is a model on its own mix, dashed the same model on the other's. Every ledger term
# and the final stream's RankMe, so the question "does the split follow the model or the
# data?" is answered on all of them rather than on Σquality alone.

# %%
SWAP_SRC = 'block_representations_samples_swap'
SWAP_MODELS = (('pythia-1b-deduped', 'tab:red'), ('OLMo-2-0425-1B', 'tab:blue'))
SWAP_RUNS = (('Own mix', None, '-'), ('Swapped mix', SWAP_SRC, '--'))
# (ledger term or None for the stream metric, panel label)
SWAP_PANELS = ((None, 'RankMe (afn)'), ('quality', r'$\Sigma$ block-intrinsic'),
               ('chi', r'$\Sigma$ overlap'), ('interference', r'$\Sigma$ interference'))
SWAP_FIG = (13.0, 7.4)


def swap_series(model, src, term):
    """(tokens, values): the ledger term summed over blocks, or the final stream's RankMe."""
    src = src or cfg_of(model)
    if term is None:
        ys, steps = _lib.get_ys(src, model, HOOK, 'rankme')
        v = np.asarray(ys, float)
    else:
        steps = _lib.get_ys(src, model, ('blk0', 'block_ledger'), 'chi')[1]
        v = np.sum([_lib.get_ys(src, model, (f'blk{l}', 'block_ledger'), term)[0]
                    for l in range(N_BLOCKS[model])], axis=0)
    xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
    k = xs > 0
    return xs[k], np.asarray(v, float)[k]


def swap_grid(models=SWAP_MODELS, panels=SWAP_PANELS, figsize=SWAP_FIG, fs=LS_FS,
              ncols=2, legend_ncol=4, save=None):
    """One panel per quantity; within a panel, each model solid on its own mix and dashed on
    the other's. Dashed tracking solid = the split follows the model, not the data."""
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, (term, label) in enumerate(panels):
        ax = axes.flat[i]
        for model, c in models:
            for run, src, ls in SWAP_RUNS:
                x, v = swap_series(model, src, term)
                ax.plot(x, v, color=c, ls=ls, lw=2.0 if ls == '-' else 1.6,
                        label=f'{model_label(model)}, {run.lower()}')
        ax.set_xscale('log')
        ax.set_ylabel(label, fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if i >= len(panels) - ncols: ax.set_xlabel('Pretraining tokens', fontsize=fs)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=legend_ncol, fontsize=fs,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = swap_grid(save='analysis/figures/standalone/dataset_swap.pdf')


# %%
def swap_deviation(models=SWAP_MODELS, panels=SWAP_PANELS, src=SWAP_SRC):
    """How far each swapped run sits from its own-mix reference, printed rather than drawn.
    The two runs are on different checkpoint grids, so the swapped curve is interpolated onto
    the reference's tokens over the overlapping range. |dev| is scale-free as a share of the
    reference's own range, which is what makes the four quantities comparable."""
    print(f'{"quantity":>22} {"model":>12} {"mean dev":>10} {"mean |dev|":>11} '
          f'{"|dev| / range":>14}')
    for term, label in panels:
        for model, _c in models:
            x0, v0 = swap_series(model, None, term)
            x1, v1 = swap_series(model, src, term)
            lo, hi = max(x0.min(), x1.min()), min(x0.max(), x1.max())
            k = (x0 >= lo) & (x0 <= hi)
            d = np.interp(np.log(x0[k]), np.log(x1), v1) - v0[k]
            rng = v0[k].max() - v0[k].min()
            print(f'{label.replace("$", "").replace(chr(92) + "Sigma", "sum"):>22} '
                  f'{model_label(model):>12} {d.mean():>10.3f} {np.abs(d).mean():>11.3f} '
                  f'{np.abs(d).mean() / rng:>13.1%}')


swap_deviation()


# %% [markdown]
# ## How many samples the spectrum needs
#
# The centered after-final-norm spectrum of Pythia 1B at its final checkpoint, measured from
# a range of sample counts. Packed against padded last-token, on one pair of axes: the packed
# spectrum is invariant from 16k rows up, the padded one is not.

# %%
SC_DATA = torch.load('data/results/sample_count_spectra.pt', weights_only=True)
SC_NS = {'packed': (16_384, 32_768, 65_536, 131_072, 262_144),
         'padded': (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)}
SC_STD = {'packed': 262_144, 'padded': 16_384}      # the budget every other figure uses
# Tokens per sequence, a sequence being a 512-token window: packed reads the whole window, so
# N tokens is N/512 sequences; padded keeps one last token each, so there N IS the sequence
# count. The realised count is 511/512 x N either way — the last token of a window has no
# next-token target (data_utils.select_token_mask).
SC_TOK_PER_SEQ = {'packed': 512, 'padded': 1}
SC_FIG = (12.5, 5.0)


def sample_count_spectra(leaf='after_final_norm', geoms=('packed', 'padded'), figsize=SC_FIG,
                         fs=LS_FS, cmap='plasma', save=None):
    """One panel per data mode, sharing both axes so the two are directly comparable."""
    fig, axes = plt.subplots(1, len(geoms), figsize=figsize, squeeze=False,
                             sharex=True, sharey=True)
    for j, geom in enumerate(geoms):
        ax = axes[0, j]
        ns = SC_NS[geom]
        for i, n in enumerate(ns):
            lam = SC_DATA[f'{geom}.{leaf}.{n}'].numpy()
            lam = lam[lam > 0]
            p = lam / lam.sum()
            rm = np.exp(-(p * np.log(p)).sum())
            std, per = n == SC_STD[geom], SC_TOK_PER_SEQ[geom]
            head = (rf'$N = |\mathcal{{S}}| = {n:,}$' if per == 1 else
                    rf'$|\mathcal{{S}}| = {n // per:,}$, $N = {n:,}$')
            # the standard budget is named in the key rather than drawn heavier
            ax.loglog(np.arange(1, len(lam) + 1), lam, lw=1.6,
                      color=plt.get_cmap(cmap)(i / (len(ns) - 1)),
                      label=f'{head} (RankMe {rm:.1f})' + (' — standard' if std else ''))
        ax.set_title(geom.capitalize(), fontsize=fs)
        ax.set_xlabel('Eigenvalue index', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        ax.legend(fontsize=fs - 5, loc='lower left')
        if j == 0: ax.set_ylabel('Eigenvalue', fontsize=fs)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = sample_count_spectra(save='analysis/figures/standalone/sample_count_spectra.pdf')


# %% [markdown]
# ## Final-norm robustness
#
# RankMe before and after the final norm, both 1B models, on all tokens and on content tokens
# only. The two read-points track each other; what separates them is the outlier tokens, so
# removing those closes most of the gap.

# %%
FN_MODELS = ('pythia-1b-deduped', 'OLMo-2-0425-1B')
# (leaf, run source, label, colour, linestyle)
# the read-points are named against each other here, not by residual index: the figure is
# about the norm, so "before"/"after" is what carries the meaning
FN_LINES = (('before_final_norm', None, 'Before final norm, all tokens', 'tab:blue', '-'),
            ('after_final_norm', None, 'After final norm, all tokens', 'tab:red', '-'),
            ('before_final_norm', 'content_tokens_full', 'Before final norm, content only',
             'tab:blue', '--'),
            ('after_final_norm', 'content_tokens_full', 'After final norm, content only',
             'tab:red', '--'))
FN_FIG = (12.5, 4.8)


def final_norm_grid(models=FN_MODELS, lines=FN_LINES, content=True, figsize=FN_FIG,
                    fs=LS_FS, legend_ncol=None, save=None):
    """One panel per model, every curve in each. content=False drops the content-tokens-only
    pair, leaving just the two read-points on all tokens."""
    lines = [ln for ln in lines if content or ln[1] is None]
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        for leaf, src, label, c, ls in lines:
            ys, steps = _lib.get_ys(src or cfg_of(model), model, (leaf, 'acts_centered'),
                                    'rankme')
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            k = xs > 0
            ax.plot(xs[k], np.asarray(ys, float)[k], color=c, ls=ls,
                    lw=2.0 if ls == '-' else 1.6, label=label)
        ax.set_xscale('log')
        ax.set_title(model_label(model), fontsize=fs)
        ax.set_xlabel('Pretraining tokens', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if j == 0: ax.set_ylabel('RankMe', fontsize=fs)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles, labels, loc='lower center', ncol=legend_ncol or len(lines),
               fontsize=fs, frameon=False, bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = final_norm_grid(content=False,
                    save='analysis/figures/standalone/final_norm_robustness.pdf')


# %% [markdown]
# ## Packed against padded, layer by layer
#
# The same four snapshots for Pythia 1B, read on the packed trainset stream and on the padded
# last-token stream. NB the padded sweep is the model's OWN mix padded (`dataset_name: native`,
# `token_selection: last`), not fineweb — the fineweb padded run stores only the final stream,
# no per-layer nodes, so a fineweb version of this figure is not available from disk.

# %%
PADDED_BS = 'fineweb_padded_layerwise_len128'

# %%
_ = merged_depth(runs=(('Packed, trainset', None), ('Padded, last token', PADDED_BS)),
                 figsize=(10.5, 4.6),
                 save='analysis/figures/standalone/depth_packed_vs_padded.pdf')


# %% [markdown]
# ## ΔS of the non-final layers
#
# The ledger summed over the first three quarters of the stack — everything but the final
# layers, which are the ones that flip sign between data modes. The baseline trajectory is the
# stack's own `exp(ΔS total + Δ embedding)`; from the entropy peak onward three more are
# stitched onto it, each the same quantity with one ledger term contributing nothing. The gap
# that opens after the peak is what that term was worth. Two versions of the one figure: with
# the stacks, and with the trajectories alone.

# %%
from matplotlib.lines import Line2D
from analysis.experiments_lib import EXP_C, EMB_DELTA_C

NF_MODELS = ('pythia-1b-deduped', 'OLMo-2-0425-1B', 'nanochat-d12')
NF_FRAC = 0.75                       # share of the stack counted, from the bottom
NF_FIG = (15.5, 4.8)
# (term contributing nothing, label, colour) — matched to the stack's own term colours
NF_DROPS = (('chi', 'Without overlap', 'tab:green'),
            ('quality', 'Without block-intrinsic', 'tab:red'),
            ('interference', 'Without interference', 'tab:purple'))
# 'emb' is not a ledger term: it holds the embedding stream still from the branch on, so the
# trajectory is the blocks' own ΔS with the ground they write into frozen
NF_DROPS_EMB = NF_DROPS + (('emb', 'Without Δ embedding', EMB_DELTA_C),)


def nonfinal_blocks(model, frac=NF_FRAC):
    return list(range(int(N_BLOCKS[model] * frac)))


def nonfinal_exp(model, blocks=None, src=None, anchor='peak', shift=0.0, drops=NF_DROPS,
                 only=False, xvar='tokens'):
    """(x, baseline, {term: stitched trajectory}, anchor index). The baseline is
    exp(S(embedding) + ΣΔS) over `blocks`, which IS the measured RankMe of the residual after
    them — the ledger telescopes, verified to 1e-6. exp_axis' own curve is this divided by
    S(embedding) at the first checkpoint, a constant ~1e3 offset, so it is not reusable here.
    Each trajectory leaves the baseline at the anchor and runs on without that term.
    shift moves the branch point along the token axis in decades — -0.5 is half a major tick
    earlier, which starts the split before the curve has turned over. only=True inverts each
    variant: instead of everything except that term, the term ALONE moves and the rest are
    frozen at the branch — what the term would have done on its own."""
    src = src or cfg_of(model)
    blocks = nonfinal_blocks(model) if blocks is None else list(blocks)
    terms = {t: np.sum([_lib.get_ys(src, model, (f'blk{l}', 'block_ledger'), t)[0]
                        for l in blocks], axis=0)
             for t in ('chi', 'quality', 'interference')}
    steps = _lib.get_ys(src, model, (f'blk{blocks[0]}', 'block_ledger'), 'chi')[1]
    x = np.asarray(_lib.XVAR_FNS[xvar](model, steps), float)
    k = x > 0
    x, terms = x[k], {t: np.asarray(v, float)[k] for t, v in terms.items()}
    es, esteps = _lib.get_ys(src, model, ('blk0.attn.in', 'acts_centered'), 'matrix_entropy')
    ex = np.asarray(_lib.XVAR_FNS[xvar](model, esteps), float)
    ek = ex > 0
    semb = np.interp(np.log(x), np.log(ex[ek]), np.asarray(es, float)[ek])
    total = sum(terms.values()) + semb           # S of the residual after `blocks`
    base = np.exp(total)
    # the anchor is a landmark of THIS curve, not of the final stream: the residual after the
    # non-final layers peaks well before afn does (Pythia 1B: 4.2e9 against 2.1e10). The warmup
    # trough is sought over the first half in LOG TOKENS — by index it lands past the peak,
    # since the checkpoints are log-spaced early and linear late
    early = x <= np.sqrt(x[0] * x[-1])
    lo = int(np.argmin(np.where(early, base, np.inf)))
    i = lo + int(np.argmax(base[lo:])) if anchor == 'peak' else lo
    if shift:
        i = int(np.argmin(np.abs(np.log10(x) - (np.log10(x[i]) + shift))))
    out = {}
    for t, _lbl, _c in drops:
        part = semb if t == 'emb' else terms[t]
        mod = part if only else total - part               # only: that term is the whole move
        out[t] = base[i] * np.exp(mod - mod[i])            # stitched onto the baseline
    return x, base, out, i


def _traj_on(ax, model, frac, anchor, src=None, shift=0.0, drops=NF_DROPS, only=False,
             baseline=True, lw=1.8):
    """Draw the stitched trajectories on `ax`; baseline=False when the stack already drew it."""
    blocks = nonfinal_blocks(model, frac)
    x, base, mods, i = nonfinal_exp(model, blocks, src=src, anchor=anchor, shift=shift,
                                    drops=drops, only=only)
    if baseline:
        ax.plot(x, base, color=EXP_C, ls=':', lw=2.2)
    for t, _lbl, c in drops:
        ax.plot(x[i:], mods[t][i:], color=c, lw=lw)
    ax.plot([x[i]], [base[i]], 'o', ms=5, color=EXP_C, zorder=4)
    return blocks


NF_YLAB = 'RankMe of the residual after these layers'
def nf_key(drops, only=False):
    """only=True relabels each variant: the term is the whole move, not the one left out."""
    lab = lambda l: l.replace('Without ', 'Only ') if only else l
    return ([Line2D([], [], color=EXP_C, ls=':', lw=2.2,
                    label=r'Measured, $\exp(S_0 + \Sigma\,\Delta S)$')]
            + [Line2D([], [], color=c, lw=1.8, label=lab(l)) for _t, l, c in drops])


def nonfinal_stacks(models=NF_MODELS, frac=NF_FRAC, anchor='peak', src=None, shift=0.0,
                    drops=NF_DROPS, only=False, figsize=NF_FIG, fs=LS_FS, legend_ncol=4,
                    save=None):
    """Version 1: the stack per model, with the trajectories on its own exp twin axis."""
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    for j, model in enumerate(models):
        blocks = nonfinal_blocks(model, frac)
        before = list(fig.axes)
        _lib.plot_ledger_stack(model, src or cfg_of(model), blocks=blocks, _ax=axes[0, j],
                               emb=False, emb_delta=True, exp_axis=False, zero_line=True,
                               ylabel=r'$\Delta S$' if j == 0 else None,
                               legend=fs - 6 if j == 0 else None,
                               title=f'{model_label(model)}, Layers 1-{blocks[-1] + 1}')
        tw = axes[0, j].twinx()                  # our own: exp_axis' curve is offset by a
        tw.grid(False)                           # constant and would not be the RankMe
        _traj_on(tw, model, frac, anchor, src=src, shift=shift, drops=drops, only=only)
        tw.tick_params(axis='y', colors=EXP_C, labelsize=fs - 4)
        if j == len(models) - 1:
            tw.set_ylabel(NF_YLAB, color=EXP_C, fontsize=fs - 2)
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles=nf_key(drops, only), loc='lower center', ncol=legend_ncol,
               fontsize=fs,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


def nonfinal_trajectories(models=NF_MODELS, frac=NF_FRAC, anchor='peak', src=None,
                          shift=0.0, drops=NF_DROPS, only=False, figsize=NF_FIG, fs=LS_FS,
                          legend_ncol=4, save=None):
    """Version 2: the same trajectories with no stack behind them."""
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    for j, model in enumerate(models):
        ax = axes[0, j]
        blocks = _traj_on(ax, model, frac, anchor, src=src, shift=shift, drops=drops,
                          only=only)
        ax.set_xscale('log')
        ax.set_title(f'{model_label(model)}, Layers 1-{blocks[-1] + 1}', fontsize=fs)
        ax.set_xlabel('Pretraining tokens', fontsize=fs)
        ax.tick_params(labelsize=fs - 3)
        if j == 0: ax.set_ylabel(NF_YLAB, fontsize=fs - 2)   # linear: the point of the
                                                            # exponential is a rank scale
    fig.tight_layout(rect=(0, 0.48 / figsize[1], 1, 1))
    fig.legend(handles=nf_key(drops, only), loc='lower center', ncol=legend_ncol,
               fontsize=fs,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = nonfinal_stacks(save='analysis/figures/standalone/nonfinal_stacks.pdf')

# %%
_ = nonfinal_trajectories(save='analysis/figures/standalone/nonfinal_trajectories.pdf')


# %% [markdown]
# ### The same, on content tokens only
#
# Identical construction on the content-tokens-only measurement — delimiters and first tokens
# dropped — so the term that carries the fall can be read with the outlier positions gone.

# %%
_ = nonfinal_stacks(src='content_tokens_full',
                    save='analysis/figures/standalone/nonfinal_stacks_content.pdf')

# %%
_ = nonfinal_trajectories(src='content_tokens_full',
                          save='analysis/figures/standalone/nonfinal_trajectories_content.pdf')


# %% [markdown]
# ### Content tokens only, branching half a decade earlier
#
# The same again, with the split moved back half a major tick on the token axis, so the
# trajectories part before the curve has turned over rather than at its top.

# %%
_ = nonfinal_stacks(src='content_tokens_full', shift=-0.5,
                    save='analysis/figures/standalone/nonfinal_stacks_content_early.pdf')

# %%
_ = nonfinal_trajectories(
    src='content_tokens_full', shift=-0.5,
    save='analysis/figures/standalone/nonfinal_trajectories_content_early.pdf')


# %% [markdown]
# ### Content tokens only, branching half a decade later
#
# And the same with the split half a major tick the other way, after the turn.

# %%
_ = nonfinal_stacks(src='content_tokens_full', shift=0.5,
                    save='analysis/figures/standalone/nonfinal_stacks_content_late.pdf')

# %%
_ = nonfinal_trajectories(
    src='content_tokens_full', shift=0.5,
    save='analysis/figures/standalone/nonfinal_trajectories_content_late.pdf')


# %% [markdown]
# ### Content tokens only, with the embedding as a fourth branch
#
# The same early split, plus one more trajectory: the embedding stream held still from the
# branch on, so the curve is the blocks' own ΔS with the ground they write into frozen.

# %%
_ = nonfinal_stacks(
    src='content_tokens_full', shift=-0.5, drops=NF_DROPS_EMB, legend_ncol=5,
    save='analysis/figures/standalone/nonfinal_stacks_content_early_emb.pdf')

# %%
_ = nonfinal_trajectories(
    src='content_tokens_full', shift=-0.5, drops=NF_DROPS_EMB, legend_ncol=5,
    save='analysis/figures/standalone/nonfinal_trajectories_content_early_emb.pdf')


# %% [markdown]
# ### The same, but one term at a time
#
# The inverse reading of the figure above: instead of everything except one term, each
# trajectory is that term ALONE moving from the branch point, with the rest held still.

# %%
_ = nonfinal_stacks(
    src='content_tokens_full', shift=-0.5, drops=NF_DROPS_EMB, only=True, legend_ncol=5,
    save='analysis/figures/standalone/nonfinal_stacks_content_early_only.pdf')

# %%
_ = nonfinal_trajectories(
    src='content_tokens_full', shift=-0.5, drops=NF_DROPS_EMB, only=True, legend_ncol=5,
    save='analysis/figures/standalone/nonfinal_trajectories_content_early_only.pdf')


# %% [markdown]
# ## Per-layer contribution since the branch
#
# The depth stack of experiments.ipynb, re-based: each of the first three quarters of the
# layers contributes its own ΔS measured from the branch point rather than from step 0, and
# the embedding stream is one more band on the same footing. Every band therefore starts at
# zero, so the embedding's several-nat offset cannot swamp the layers, and the black total is
# log of the RankMe ratio since the branch.

# %%
from matplotlib.patches import Patch

NF_STACK_FIG = (15.5, 5.0)


def layer_stack_panel(ax, model, frac=NF_FRAC, src=None, anchor='peak', shift=0.0, fs=LS_FS,
                      cmap='viridis', title=None, xlabel=True, legend=False, tick_right=False,
                      cb_frac=0.15, cbar=True, n_colors=None):
    """One model's per-layer ΔS since the branch, signed-stacked, embedding included."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap
    plt.sca(ax)
    blocks = nonfinal_blocks(model, frac)
    cfg = src or cfg_of(model)
    x, _base, _mods, i = nonfinal_exp(model, blocks, src=src, anchor=anchor, shift=shift)
    per = [np.asarray(_lib.get_ys(cfg, model, (f'blk{l}', 'block_ledger'), 'delta_s')[0],
                      float) for l in blocks]
    steps = _lib.get_ys(cfg, model, (f'blk{blocks[0]}', 'block_ledger'), 'chi')[1]
    xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
    k = xs > 0
    per = [p[k] for p in per]
    es, esteps = _lib.get_ys(cfg, model, ('blk0.attn.in', 'acts_centered'), 'matrix_entropy')
    ex = np.asarray(_lib.get_xs_tokens(model, esteps), float)
    semb = np.interp(np.log(x), np.log(ex[ex > 0]), np.asarray(es, float)[ex > 0])
    # n_colors fixes the depth scale across models, so one key can serve every panel
    cols = plt.get_cmap(cmap)(np.linspace(0, 1, n_colors or len(blocks)))[:len(blocks)]
    # every band is measured from the branch, so they all start at 0 and share a scale
    series = ([('Embedding', semb[i:] - semb[i], EMB_DELTA_C)]
              + [(f'Layer {l + 1}', p[i:] - p[i], cols[n]) for n, (l, p)
                 in enumerate(zip(blocks, per))])
    gx, tot = _lib.signed_stack(x[i:], series, alpha=0.9)
    ax.plot(gx, tot, 'k', lw=2.2, label=r'Total $\Delta S$ since entropy peak')
    ax.axhline(0, color='0.35', lw=0.8)
    ax.set_xscale('log')
    ax.set_title(f'{model_label(model)}, Layers 1-{blocks[-1] + 1}' if title is None else title,
                 fontsize=fs)
    if xlabel: ax.set_xlabel('Pretraining tokens', fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    if tick_right: ax.yaxis.tick_right()
    # discrete: one swatch per layer, so a band can be read off by name
    n = len(blocks)
    every = 1 if n <= 12 else 2
    if cbar:
        sm = ScalarMappable(cmap=ListedColormap(list(cols)),
                            norm=BoundaryNorm(np.arange(n + 1) - 0.5, n))
        cb = plt.colorbar(sm, ax=ax, pad=0.02, fraction=cb_frac, ticks=np.arange(0, n, every))
        cb.set_ticklabels([str(blocks[t] + 1) for t in range(0, n, every)])
        cb.ax.tick_params(labelsize=fs - 5, length=0)
        cb.set_label('Layer', fontsize=fs - 4)
        cb.outline.set_edgecolor(RULE)
    if legend:
        ax.legend(handles=[Patch(color=EMB_DELTA_C, alpha=0.9, label='Embedding'),
                           Line2D([], [], color='k', lw=2.2, label='Total')],
                  fontsize=fs - 4, loc='lower left')


def nonfinal_layer_stack(models=NF_MODELS, frac=NF_FRAC, src=None, anchor='peak', shift=0.0,
                         figsize=NF_STACK_FIG, fs=LS_FS, cmap='viridis', save=None):
    """One panel per model: per-layer ΔS since the branch, signed-stacked, embedding included."""
    fig, axes = plt.subplots(1, len(models), figsize=figsize, squeeze=False)
    for j, model in enumerate(models):
        layer_stack_panel(axes[0, j], model, frac=frac, src=src, anchor=anchor, shift=shift,
                          fs=fs, cmap=cmap, legend=j == 0)
        if j == 0:
            axes[0, j].set_ylabel(r'$\Delta S$ since entropy peak', fontsize=fs)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = nonfinal_layer_stack(src='content_tokens_full', shift=0.0,
                         save='analysis/figures/standalone/nonfinal_layer_stack.pdf')

# %%
_ = nonfinal_layer_stack(src='content_tokens_full', shift=-0.5,
                         save='analysis/figures/standalone/nonfinal_layer_stack_early.pdf')


# %% [markdown]
# ## Each layer's components since the branch
#
# The stack above, opened up one layer at a time: for every layer in the band its ΔS splits
# into overlap + block-intrinsic + interference, all measured from the branch and over the
# same token range, so a panel says which term moved that layer. Layer 1 and the last of the
# group are left out — the first sits on the embedding, the last is the group's boundary.

# %%
NF_COMP_ALPHA = 0.55                 # signed_stack's fill alpha, as the other stacks use
NF_COMP_RANGE = (1, 12)                                  # one-indexed, inclusive
NF_COMP_BY_MODEL = {'nanochat-d12': (1, 8)}              # shallower stack, shorter band
NF_TARGET_C = 'tab:blue'             # what is still owed against the embedding's rise
NF_GAP, NF_KEY_W = 0.004, 0.13       # layer-column gap and floor colour key width, figure units


def comp_layers(model, rng=NF_COMP_RANGE, per_model=NF_COMP_BY_MODEL):
    lo, hi = per_model.get(model, rng)
    return list(range(lo - 1, hi))                       # -> 0-indexed blocks


def nonfinal_component_grid(models=NF_MODELS, src=None, anchor='peak', shift=0.0,
                            frac=NF_FRAC, hide_paid=True, figsize=None, fs=LS_FS,
                            legend_ncol=5, stack_col=False, save=None):
    """Rows = model, columns = layer. Each panel is that layer's own three-term stack, from
    the branch on and on the row's shared y, so layers are comparable within a model.
    The dotted line is what the stack still owes to cancel the embedding stream's own rise:
    -Δ embedding for layer 1, then less whatever the layers before it already returned.
    hide_paid=True draws it only while that is still negative, so it fades out at the depth
    where the rise has been paid off and never appears at all where the embedding did not
    rise; False keeps the whole curve, including the part above zero that says by how much a
    layer over-paid.
    stack_col=True appends the model's whole layer stack as a last column, on its own y so the
    layer panels keep their scale."""
    cols_n = max(len(comp_layers(m)) for m in models)
    ncol = cols_n + bool(stack_col)
    stack_n = max(len(nonfinal_blocks(m, frac)) for m in models)   # one depth scale for all
    figsize = figsize or (1.55 * ncol + 1.2, 2.5 * len(models) + 0.9)
    # gridspec rather than subplots(sharey='row'): the stack column must stay off the row's y
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(models), ncol)
    axes = np.empty((len(models), ncol), dtype=object)
    for r in range(len(models)):
        axes[r, 0] = fig.add_subplot(gs[r, 0])
        for c in range(1, cols_n):
            axes[r, c] = fig.add_subplot(gs[r, c], sharex=axes[r, 0], sharey=axes[r, 0])
        if stack_col:
            axes[r, ncol - 1] = fig.add_subplot(gs[r, ncol - 1])
    for r, model in enumerate(models):
        cfg = src or cfg_of(model)
        x, _base, _mods, i = nonfinal_exp(model, nonfinal_blocks(model, frac), src=src,
                                          anchor=anchor, shift=shift)
        layers = comp_layers(model)
        steps0 = _lib.get_ys(cfg, model, ('blk0', 'block_ledger'), 'chi')[1]
        xs0 = np.asarray(_lib.get_xs_tokens(model, steps0), float)
        k0 = xs0 > 0
        es, esteps = _lib.get_ys(cfg, model, ('blk0.attn.in', 'acts_centered'),
                                 'matrix_entropy')
        ex = np.asarray(_lib.get_xs_tokens(model, esteps), float)
        semb = np.interp(np.log(x), np.log(ex[ex > 0]), np.asarray(es, float)[ex > 0])
        owed = -(semb[i:] - semb[i])                     # layer 1 owes the whole rise
        pending, span = [], []                           # targets drawn after the row is scaled
        for c, l in enumerate(layers):
            ax = axes[r, c]
            plt.sca(ax)
            series = []
            for term, label, colour in LEDGER_TERMS:
                v = np.asarray(_lib.get_ys(cfg, model, (f'blk{l}', 'block_ledger'), term)[0],
                               float)
                steps = _lib.get_ys(cfg, model, (f'blk{l}', 'block_ledger'), 'chi')[1]
                xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
                v = v[xs > 0]
                series.append((label, v[i:] - v[i], colour))
            gx, tot = _lib.signed_stack(x[i:], series, alpha=NF_COMP_ALPHA)
            ax.plot(gx, tot, 'k', lw=1.6)
            ax.axhline(0, color='0.35', lw=0.7)
            span.append(ax.dataLim.intervaly)            # this panel's stack, before the target
            pending.append((ax, np.where(owed < 0, owed, np.nan) if hide_paid else owed))
            ds = np.asarray(_lib.get_ys(cfg, model, (f'blk{l}', 'block_ledger'),
                                        'delta_s')[0], float)[k0]
            owed = owed - (ds[i:] - ds[i])               # this layer's share, paid forward
            ax.set_xscale('log')
            ax.set_title(f'Layer {l + 1}', fontsize=fs - 4, pad=3)
            ax.tick_params(labelsize=fs - 6)
            ax.xaxis.set_major_locator(LogLocator(numticks=3))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if c: ax.tick_params(labelleft=False)
            if r < len(models) - 1 or c: ax.set_xlabel('')
        for ax in axes[r, len(layers):cols_n]:
            ax.set_visible(False)
        # scale the row to hold EVERY layer's stack, then draw the targets into it: they are a
        # reference, and a deep one at layer 1 would otherwise set the range for the whole row
        lo, hi = min(v[0] for v in span), max(v[1] for v in span)
        pad = 0.05 * (hi - lo)
        axes[r, 0].set_ylim(lo - pad, hi + pad)          # shared: applies across the row
        for ax, owed_c in pending:
            ax.plot(x[i:], owed_c, color=NF_TARGET_C, ls=':', lw=1.6, zorder=4,
                    clip_on=True)
        axes[r, 0].set_ylabel(r'$\Delta S$ since entropy peak', fontsize=fs - 3)
        if stack_col:
            blocks = nonfinal_blocks(model, frac)
            sax = axes[r, ncol - 1]
            layer_stack_panel(sax, model, frac=frac, src=src, anchor=anchor, shift=shift,
                              fs=fs - 4, xlabel=False, cbar=False, n_colors=stack_n,
                              title=f'Layers 1-{blocks[-1] + 1}')
            sax.xaxis.set_major_locator(LogLocator(numticks=3))
            sax.xaxis.set_minor_formatter(NullFormatter())
    axes[-1, 0].set_xlabel('Pretraining tokens', fontsize=fs - 2)
    fig.tight_layout(rect=(0.035, (0.95 if stack_col else 0.40) / figsize[1], 1, 1),
                     **({'w_pad': 0.15, 'h_pad': 0.6} if stack_col else {}))
    if stack_col:
        # tight_layout gives the whole grid one wspace, sized by the only column that carries
        # y tick labels. Repack the layer columns into their span at a hairline gap instead.
        for r in range(len(models)):
            row = [axes[r, c] for c in range(cols_n)]
            x0, x1 = row[0].get_position().x0, row[-1].get_position().x1
            w = (x1 - x0 - NF_GAP * (cols_n - 1)) / cols_n
            for c, ax in enumerate(row):
                q = ax.get_position()
                ax.set_position((x0 + c * (w + NF_GAP), q.y0, w, q.height))
    names = []
    for r, model in enumerate(models):                   # model names in the left margin
        q = axes[r, 0].get_position()
        names.append(fig.text(0.004, (q.y0 + q.y1) / 2, model_label(model), rotation=90,
                              va='center', ha='left', fontsize=fs - 1))
    # halve the gap the layout leaves between a name and the y label beside it
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    lab = min(axes[r, 0].yaxis.label.get_window_extent().transformed(inv).x0
              for r in range(len(models)))
    edge = max(t.get_window_extent().transformed(inv).x1 for t in names)
    for t in names:
        t.set_x(t.get_position()[0] + (lab - edge) / 2)
    leg = fig.legend(handles=[Patch(color=c, alpha=NF_COMP_ALPHA, label=l)
                              for _t, l, c in LEDGER_TERMS]
                     + [Line2D([], [], color='k', lw=1.6, label=r'$\Delta S$'),
                        Line2D([], [], color=NF_TARGET_C, ls=':', lw=1.6,
                               label='Required to compensate for embedding change')],
                     loc='lower center', ncol=legend_ncol, fontsize=fs - 1, frameon=False,
                     bbox_to_anchor=(0.5, 0.03 if stack_col else 0.0))
    if stack_col:
        # the stack's two colour keys close the legend row: the depth bar, then the embedding
        # swatch, each followed by its label. Widths are measured, then the legend is shifted
        # left by half of them so the whole row stays centred.
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import BoundaryNorm, ListedColormap
        inv = fig.transFigure.inverted()
        t_emb = fig.text(0, 0, 'Embedding', va='center', ha='left', fontsize=fs - 1)
        t_lay = fig.text(0, 0, 'Layer', va='center', ha='left', fontsize=fs - 1)
        fig.canvas.draw()
        we = t_emb.get_window_extent().transformed(inv).width
        wl = t_lay.get_window_extent().transformed(inv).width
        h = 0.7 * (fs - 1) / 72 / figsize[1]             # a legend swatch's height
        ws = 2.0 * (fs - 1) / 72 / figsize[0]            # ... and its width
        # the legend's own metrics, so the hand-placed keys sit on the same rhythm
        htp = plt.rcParams['legend.handletextpad'] * (fs - 1) / 72 / figsize[0]
        csp = plt.rcParams['legend.columnspacing'] * (fs - 1) / 72 / figsize[0]
        leg.set_bbox_to_anchor(
            (0.5 - (2 * csp + 2 * htp + NF_KEY_W + wl + ws + we) / 2, 0.03),
            transform=fig.transFigure)
        fig.canvas.draw()
        box = leg.get_window_extent().transformed(inv)
        cy, x = box.y0 + box.height / 2, box.x1 + csp
        cols = plt.get_cmap('viridis')(np.linspace(0, 1, stack_n))
        sm = ScalarMappable(cmap=ListedColormap(list(cols)),
                            norm=BoundaryNorm(np.arange(stack_n + 1) - 0.5, stack_n))
        cb = fig.colorbar(sm, cax=fig.add_axes((x, cy - h / 2, NF_KEY_W, h)),
                          orientation='horizontal', ticks=np.arange(stack_n))
        cb.set_ticklabels([str(t + 1) for t in range(stack_n)])
        cb.ax.tick_params(labelsize=fs - 7, length=0, pad=1)
        cb.outline.set_edgecolor(RULE)
        x += NF_KEY_W + htp
        t_lay.set_position((x, cy))
        x += wl + csp
        sw = fig.add_axes((x, cy - h / 2, ws, h))
        sw.set(xticks=[], yticks=[], facecolor=EMB_DELTA_C)
        sw.patch.set_alpha(0.9)
        for s in sw.spines.values():
            s.set_visible(False)
        t_emb.set_position((x + ws + htp, cy))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches='tight')
    return fig


# %%
_ = nonfinal_component_grid(
    src='content_tokens_full', shift=0.0, hide_paid=False,
    save='analysis/figures/standalone/nonfinal_components.pdf')

# %%
_ = nonfinal_component_grid(
    src='content_tokens_full', shift=-0.5, hide_paid=False,
    save='analysis/figures/standalone/nonfinal_components_early.pdf')

# %% [markdown]
# The same grid with each model's whole layer stack appended on the right, on its own y.

# %%
_ = nonfinal_component_grid(
    src='content_tokens_full', shift=0.0, hide_paid=False, stack_col=True, legend_ncol=6,
    save='analysis/figures/standalone/nonfinal_components_stack.pdf')
