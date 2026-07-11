"""Animated spectrum plots, swept across training checkpoints.

`animate_spectra(panels, ..., save=...)`:
  save=None    -> inline jshtml player (shown in the notebook)
  save='*.gif' -> Pillow (no ffmpeg)
  save='*.mp4' -> ffmpeg. If ffmpeg isn't on PATH (e.g. the VSCode kernel), the
                  frames are pickled and a staging job is sbatch'd (fire-and-forget,
                  `module load 2025 FFmpeg`) to render them. Returns the path the mp4
                  will appear at; view it once the job finishes.

The data backend (get_series_y etc.) is injected once via configure(); the kernel
resolves every frame to plain arrays, so the render side needs only matplotlib+ffmpeg.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')

import sys
import shutil
import pickle
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from pathlib import Path

_FFMPEG_MODULE = '2025 FFmpeg/7.1.1-GCCcore-14.2.0'
_BK = {}  # get_series_y, get_ys, panel_palettes, smooth_spectrum, submatrix, block_mean_cos, model_name_options, YVAR_LABELS, XVAR_FNS


def configure(**backend):
    _BK.update(backend)


def _bk(name):
    if name not in _BK:
        raise RuntimeError(f"spectrum_anim.configure(...) needed: missing '{name}'")
    return _BK[name]


def _frame_label(model, step, xvar):
    if xvar == 'steps':
        return f'step {step}'
    val = _bk('XVAR_FNS')[xvar](model, [step])[0]
    unit = {'tokens': 'tokens', 'flops': 'FLOPs'}.get(xvar, xvar)
    return f'{val:.2e} {unit}'


def _draw_progress(bar_ax, frac, label=None):
    bar_ax.clear()
    bar_ax.set_xlim(0, 1); bar_ax.set_ylim(0, 1); bar_ax.axis('off')
    bar_ax.add_patch(plt.Rectangle((0, 0.35), 1.0,  0.3, color='0.88', ec='none'))
    bar_ax.add_patch(plt.Rectangle((0, 0.35), frac, 0.3, color='0.30', ec='none'))
    if label:
        bar_ax.text(1.0, 0.85, label, ha='right', va='bottom', fontsize=10, color='0.3')


# ---- kernel side: resolve panels to a backend-free, picklable spec ----------

def _panel_palettes(data_sources, opts):
    return _bk('panel_palettes')(len(data_sources), opts.get('color_palette', 'hue_shift'),
                                 opts.get('color_kwargs', {}))


def _ylim(lo, hi, ylog):
    if not np.isfinite(lo):
        return None
    return (lo * 0.8, hi * 1.25) if ylog else (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))


def _strip_panel(yvar, data_sources, model_names, opts, palettes):
    """Thin per-checkpoint readout: each layer's metric value as a horizontal mark
    (color matches the spectrum). Resolves through get_ys — no plot_group duplication."""
    name = yvar if isinstance(yvar, str) else yvar[0]
    ylog, mn = opts.get('ylog', False), model_names[0]
    series, lo, hi = [], np.inf, -np.inf
    for didx, dsrc in enumerate(data_sources):
        ys = _bk('get_ys')(dsrc[0], mn, dsrc[1], name)[0]
        if ys is None: continue
        ys = [float(y) for y in ys]
        series.append((ys, palettes[didx][_bk('model_name_options').index(mn)]))
        pos = [y for y in ys if y > 0] if ylog else ys
        if pos: lo, hi = min(lo, min(pos)), max(hi, max(pos))
    return dict(kind='strip', ylog=ylog, ylim=_ylim(lo, hi, ylog), series=series,
                label=opts.get('title') or _bk('YVAR_LABELS').get(name, name))


def _heatmap_panel(data_sources, model_names, opts):
    """Block×block cosine-of-means matrix per checkpoint (block_mean_cos per step)."""
    mn, ds0 = model_names[0], data_sources[0]
    labels = [s[2] if len(s) > 2 else '' for s in data_sources]
    steps = _bk('get_ys')(ds0[0], mn, (ds0[1][0],), 'acts_mean_vec')[1]
    frames = [np.asarray(_bk('block_mean_cos')(mn, data_sources, s), dtype=float) for s in steps]
    return _matrix_spec(frames, labels, opts), steps


def _matrix_panel(yvar, data_sources, model_names, opts):
    """A stored matrix-valued metric series, animated directly (e.g. the samples runs'
    block_block_coupling matrices: cka / signed_trace / mean_cos at the root node)."""
    (src, hook, *_), mn = data_sources[0], model_names[0]
    frames, steps = _bk('get_ys')(src, mn, hook, yvar)
    return _matrix_spec([np.asarray(f, dtype=float) for f in frames], opts.get('labels'), opts), steps


def _matrix_spec(frames, labels, opts):
    """dynamic=True -> symmetric range from the off-diagonal max; per_frame picks whether
    that range is recomputed each frame (full contrast) or pinned once (stable colorbar).
    pair=(rows, cols) label-substring-selects a submatrix (e.g. ('attn', 'mlp')); the
    diagonal is only hidden (self-pairs) when rows == cols."""
    rows, cols = opts.get('pair') or ('', '')
    labels2 = labels
    if labels and (rows or cols):
        sub = [_bk('submatrix')(f, labels, rows, cols) for f in frames]
        frames, labels, labels2 = [s[0] for s in sub], sub[0][1], sub[0][2]
    hide_diag = opts.get('hide_diag', rows == cols)
    dynamic, per_frame = opts.get('dynamic', True), opts.get('per_frame', False)
    vmin, vmax = opts.get('vmin', -1), opts.get('vmax', 1)
    if dynamic and not per_frame:                       # pin one global symmetric range
        off = np.concatenate([f[~np.eye(len(f), dtype=bool)] if hide_diag else f.ravel()
                              for f in frames])
        v = np.nanmax(np.abs(off)); vmin, vmax = -v, v
    return dict(kind='heatmap', frames=frames, labels=labels, labels2=labels2,
                hide_diag=hide_diag, per_frame=dynamic and per_frame,
                cmap=opts.get('cmap', 'coolwarm'), vmin=vmin, vmax=vmax,
                title=opts.get('title'))


def _materialize(panels, ncols, fps, model, xvar, prog_bar, suptitle, figsize, smooth=0, peak=3):
    spec_panels, steps = [], None
    for yvar, data_sources, model_names, opts in panels:
        if opts.get('kind') in ('heatmap', 'matrix'):
            panel, steps = (_heatmap_panel(data_sources, model_names, opts) if opts['kind'] == 'heatmap'
                            else _matrix_panel(yvar, data_sources, model_names, opts))
            spec_panels.append(panel)
            continue
        palettes = _panel_palettes(data_sources, opts)
        ds0 = data_sources[0]
        yv = ds0[3] if len(ds0) > 3 else yvar
        steps = _bk('get_ys')(ds0[0], model_names[0], ds0[1], yv if isinstance(yv, str) else yv[0])[1]
        if opts.get('kind') == 'strip':
            spec_panels.append(_strip_panel(yvar, data_sources, model_names, opts, palettes))
            continue
        if opts.get('kind') == 'profile':
            # depth profile per checkpoint: one scalar yvar per data_source (= per layer),
            # assembled into a per-frame line over source order; reuses the spectrum drawer.
            mn = model_names[0]
            cols = [_bk('get_ys')(d[0], mn, d[1], d[3] if len(d) > 3 else yvar)[0] for d in data_sources]
            frames = [[(np.array([c[i] for c in cols], dtype=float),
                        palettes[0][_bk('model_name_options').index(mn)], None)] for i in range(len(steps))]
            vals = np.array([[c[i] for c in cols] for i in range(len(steps))], dtype=float)
            ylog = opts.get('ylog', True)
            pos = vals[vals > 0] if ylog else vals
            spec_panels.append(dict(kind='spectrum', title=opts.get('title'), mode='line',
                                    xlog=False, ylog=ylog, ylim=_ylim(pos.min(), pos.max(), ylog),
                                    frames=frames, xlabel=opts.get('xlabel', 'block'),
                                    ylabel=_bk('YVAR_LABELS').get(yvar, yvar)))
            continue

        xlog, ylog = opts.get('xlog', True), opts.get('ylog', True)
        mode, title = opts.get('mode', 'default'), opts.get('title')
        ddidx, dmidx = opts.get('disable_didx', []), opts.get('disable_midx', [])
        yvarname = yvar if isinstance(yvar, str) else yvar[0]
        labels = len(model_names) > 1, len(data_sources) > 1 or len(model_names) <= 1
        if mode == 'default':
            mode = 'hist' if yvarname == 'histogram' else 'line'
        if yvarname == 'histogram' and xlog:
            yvarname = 'loghistogram'

        frames, lo, hi = [], np.inf, -np.inf
        for s in steps:
            series = []
            for midx, mn in enumerate(model_names):
                if midx in dmidx: continue
                for didx, dsrc in enumerate(data_sources):
                    if didx in ddidx: continue
                    arr, lbl = _bk('get_series_y')(dsrc, mn, yvar, s, labels)
                    if arr is None: continue
                    a = np.asarray(arr, dtype=float)
                    if mode == 'line':
                        a = _bk('smooth_spectrum')(a, smooth, peak)
                    series.append((a, palettes[didx][_bk('model_name_options').index(mn)], lbl))
                    p = a[a > 0] if ylog else a
                    if p.size: lo, hi = min(lo, p.min()), max(hi, p.max())
            frames.append(series)

        spec_panels.append(dict(
            kind='spectrum', title=title, mode=mode, xlog=xlog, ylog=ylog,
            ylim=_ylim(lo, hi, ylog), frames=frames,
            xlabel=r'$\nu$' if mode in ('hist', 'histogram') else 'Component index',
            ylabel=_bk('YVAR_LABELS').get(yvarname, yvarname)))

    flabels = [_frame_label(model, s, xvar) for s in steps] if prog_bar in ('top', 'bottom') else None
    return dict(ncols=ncols, fps=fps, prog_bar=prog_bar, suptitle=suptitle, figsize=figsize,
                n=len(steps), frame_labels=flabels, panels=spec_panels)


# ---- renderer: pure matplotlib, no backend (runs in kernel or under srun) ----

def _layout(kinds, ncols, prog_bar, figsize):
    n = len(kinds)
    bar = prog_bar in ('top', 'bottom')
    if any(k == 'strip' for k in kinds):
        # group consecutive strips into one tight band (nested gridspec) between spectra
        groups, i = [], 0
        while i < n:
            if kinds[i] == 'strip':
                j = i
                while j < n and kinds[j] == 'strip': j += 1
                groups.append(('band', list(range(i, j)))); i = j
            else:
                groups.append(('spec', i)); i += 1
        # Per-boundary gaps via spacer columns (wspace can't vary per boundary): every panel
        # carries y-labels on its LEFT, so a gap needs width iff its RIGHT neighbour is a
        # labelled spec panel; a bare spec right edge -> band needs almost none. Strips' own
        # label room comes from the band's inner wspace.
        wr = [1.0 if g[0] == 'spec' else 0.085 * len(g[1]) for g in groups]
        gaps = [0.19 if groups[i + 1][0] == 'spec' else 0.06 for i in range(len(groups) - 1)]
        cols = wr[:1] + [x for gap, w in zip(gaps, wr[1:]) for x in (gap, w)]
        figw = 7 * sum(g[0] == 'spec' for g in groups) + 1.7 * sum(g[0] == 'band' for g in groups) + 1.2
        fig = plt.figure(figsize=figsize or (figw, 5.4))
        if bar:
            hr = [1, 28] if prog_bar == 'top' else [28, 1]
            gso = fig.add_gridspec(2, len(cols), width_ratios=cols, height_ratios=hr, wspace=0)
            prow = 1 if prog_bar == 'top' else 0
            bar_ax = fig.add_subplot(gso[0 if prog_bar == 'top' else 1, :])
        else:
            gso = fig.add_gridspec(1, len(cols), width_ratios=cols, wspace=0)
            prow, bar_ax = 0, None
        axes = [None] * n
        for gc, g in enumerate(groups):
            if g[0] == 'spec':
                axes[g[1]] = fig.add_subplot(gso[prow, 2 * gc])
            else:
                sub = gso[prow, 2 * gc].subgridspec(1, len(g[1]), wspace=0.9)
                for k, idx in enumerate(g[1]):
                    axes[idx] = fig.add_subplot(sub[0, k])
        return fig, axes, bar_ax
    nrows = (n + ncols - 1) // ncols
    square = all(k == 'heatmap' for k in kinds)        # heatmaps are square -> avoid a wide figure
    fig = plt.figure(figsize=figsize or ((5.6 * ncols, 5.5 * nrows + 0.3) if square
                                         else (7 * ncols, 5 * nrows + 0.3)))
    if bar:
        ratios = ([1] + [40] * nrows) if prog_bar == 'top' else ([40] * nrows + [1])
        gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=ratios)
        off = 1 if prog_bar == 'top' else 0
        bar_ax = fig.add_subplot(gs[0 if prog_bar == 'top' else nrows, :])
        axes = [fig.add_subplot(gs[r + off, c]) for r in range(nrows) for c in range(ncols)]
    else:
        gs = fig.add_gridspec(nrows, ncols)
        axes = [fig.add_subplot(gs[r, c]) for r in range(nrows) for c in range(ncols)]
        bar_ax = None
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n], bar_ax


def _draw_spectrum(ax, p, i):
    for arr, color, label in p['frames'][i]:
        xs = np.arange(len(arr))
        if p['mode'] == 'line':
            ax.plot(xs, arr, color=color, lw=2, label=label)
        else:
            ax.bar(xs, arr, color=color, label=label, alpha=0.6)
    if p['xlog']: ax.set_xscale('log')
    if p['ylog']: ax.set_yscale('log')
    ax.set_xlabel(p['xlabel'], fontsize=14); ax.set_ylabel(p['ylabel'], fontsize=14)
    if len(p['frames'][i]) <= 12:                          # >12: static colorbar drawn once at setup
        ax.legend(loc='lower left', ncol=2, fontsize=8)    # pinned -> no per-frame jumping
    if p['title'] is not None: ax.set_title(p['title'])
    if p['ylim'] is not None: ax.set_ylim(*p['ylim'])


def _draw_strip(ax, p, i):
    for ys, color in p['series']:
        ax.axhline(ys[i], color=color, lw=4, solid_capstyle='butt')   # thin mark per layer
    if p['ylog']: ax.set_yscale('log')
    if p['ylim']: ax.set_ylim(*p['ylim'])
    ax.set_xticks([])
    ax.tick_params(axis='y', which='both', length=2, labelleft=False)  # small ticks, no side labels
    lo, hi = p['ylim'] if p['ylim'] else (0.0, 1.0)
    fmt = (lambda v: f'$10^{{{np.log10(v):.1f}}}$') if p['ylog'] else (lambda v: f'{v:.2g}')
    ax.text(0.5, 1.012, fmt(hi), transform=ax.transAxes, ha='center', va='bottom', fontsize=6.5)
    ax.text(0.5, -0.012, fmt(lo), transform=ax.transAxes, ha='center', va='top', fontsize=6.5)
    ax.set_ylabel(p['label'], fontsize=7.5, rotation=90, labelpad=2)         # metric name, vertical


def _draw_heatmap(ax, p, i):
    M = np.array(p['frames'][i], dtype=float)
    if p['hide_diag']: np.fill_diagonal(M, np.nan)     # self-pairs -> white
    vmin, vmax = p['vmin'], p['vmax']
    if p['per_frame']:                                   # recompute symmetric range this frame
        v = np.nanmax(np.abs(M)); vmin, vmax = -v, v     # NaN diagonal already excluded
    cmap = plt.get_cmap(p['cmap']).copy(); cmap.set_bad('white')
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)              # seaborn's grid (kernel-side) draws through cell centres
    ax.set_anchor('C')          # square aspect shrinks the box -> centre it (else pins right)
    cols = p.get('labels2') or p['labels']
    if cols:
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90, fontsize=6)
    if p['labels']:
        ax.set_yticks(range(len(p['labels']))); ax.set_yticklabels(p['labels'], fontsize=6)
    if p['title'] is not None: ax.set_title(p['title'])


_DRAW = {'strip': _draw_strip, 'spectrum': _draw_spectrum, 'heatmap': _draw_heatmap}


def render(spec, save=None):
    kinds = [p['kind'] for p in spec['panels']]
    fig, axes, bar_ax = _layout(kinds, spec['ncols'], spec['prog_bar'], spec['figsize'])
    if spec['suptitle'] is not None:
        fig.suptitle(spec['suptitle'])
    right = 0.88 if any(p['kind'] == 'heatmap' or (p['kind'] == 'spectrum' and len(p['frames'][0]) > 12)
                        for p in spec['panels']) else 0.97   # room for colorbar labels
    fig.subplots_adjust(left=0.08, right=right, top=0.88, bottom=0.12, hspace=0.35, wspace=0.30)
    n = spec['n']

    from matplotlib.cm import ScalarMappable          # static colorbars for pinned heatmaps and
    from matplotlib.colors import Normalize, ListedColormap   # depth-gradient legends (survive per-frame clear)
    for ax, p in zip(axes, spec['panels']):
        if p['kind'] == 'heatmap' and not p['per_frame']:
            cmap = plt.get_cmap(p['cmap']).copy(); cmap.set_bad('white')
            sm = ScalarMappable(cmap=cmap, norm=Normalize(p['vmin'], p['vmax'])); sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        elif p['kind'] == 'spectrum' and len(p['frames'][0]) > 12:  # legend wall -> slim colorbar
            colors, labels = [s[1] for s in p['frames'][0]], [s[2] for s in p['frames'][0]]
            sm = ScalarMappable(cmap=ListedColormap(colors)); sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02,
                              ticks=[0.5 / len(labels), 0.5, 1 - 0.5 / len(labels)])
            cb.ax.set_yticklabels([labels[0], labels[len(labels) // 2], labels[-1]], fontsize=8)

    def update(i):
        for ax, p in zip(axes, spec['panels']):
            ax.clear(); ax.figure.sca(ax)  # not plt.sca: manager is None mid-save
            _DRAW[p['kind']](ax, p, i)
        if bar_ax is not None:
            _draw_progress(bar_ax, i / (n - 1) if n > 1 else 1.0, spec['frame_labels'][i])

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / spec['fps'])
    if save is None:
        html = HTML(anim.to_jshtml()); plt.close(fig); return html
    Path(save).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save, writer='pillow' if save.endswith('.gif') else 'ffmpeg', fps=spec['fps'])
    plt.close(fig)
    return save


def _save_mp4_via_sbatch(spec, out):
    """No ffmpeg here: pickle the frames and sbatch a staging job to render them
    (fire-and-forget). The render job removes the pickle when done."""
    out = os.path.abspath(out)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pkl = out + '.spec.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(spec, f)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(f'{repo}/slurm/logs', exist_ok=True)
    # sbatch --wrap runs under sh, where `module` is undefined; bash -lc loads the env.
    cmd = (f"cd {repo} && module load {_FFMPEG_MODULE} && "
           f"uv run python {os.path.abspath(__file__)} {pkl} {out}")
    subprocess.run(["sbatch", "--partition=staging", "--ntasks=1", "--cpus-per-task=4",
                    "--time=00:20:00", "--job-name=anim_mp4",
                    f"--output={repo}/slurm/logs/anim_mp4_%j.out", f"--wrap=bash -lc '{cmd}'"],
                   check=True, capture_output=True, text=True)
    print(f"[spectrum_anim] rendering in background -> {out}")
    return out


def animate_spectra(panels, ncols=2, fps=4, figsize=None, model=None,
                    xvar='tokens', prog_bar='bottom', suptitle=None, save=None,
                    smooth=0, peak=3, **common):
    """One animation, several spectrum subplots synced across checkpoints.
    panels: list of (yvar, data_sources, model_names[, opts_dict]). `common` kwargs
    (xlog, ylog, title, ...) apply to every panel; per-panel opts override.
    smooth: window (odd; 0=off) — denoises line spectra, keeps peaks/edges (shared
            smooth_spectrum from experiments_lib, injected via configure()).
    peak:   >1 biases the smoothing toward the window max (upper envelope).
    save:   optional path to ALSO write the animation to — a pure side effect; the inline
            render returned for the notebook is identical whether or not save is given.
            '*.mp4' with no ffmpeg on PATH sbatches a background render; '*.gif' / ffmpeg
            '*.mp4' are written in-process."""
    norm = []
    for p in panels:
        yvar, data_sources, model_names, *rest = p
        norm.append((yvar, data_sources, model_names, {**common, **(rest[0] if rest else {})}))
    spec = _materialize(norm, ncols, fps, model or norm[0][2][0], xvar, prog_bar, suptitle, figsize, smooth, peak)
    if save:                                            # write a file too (does NOT replace inline render)
        if save.endswith('.mp4') and shutil.which('ffmpeg') is None:
            _save_mp4_via_sbatch(spec, save)
        else:
            render(spec, save)
    return render(spec, save=None)                      # always render inline for the notebook


if __name__ == '__main__':  # render side: python spectrum_anim.py <spec.pkl> <out>
    with open(sys.argv[1], 'rb') as f:
        render(pickle.load(f), sys.argv[2])
    os.remove(sys.argv[1])  # drop the intermediate spec pickle
    print('wrote', sys.argv[2])
