"""Shared analysis infrastructure for the experiments notebooks.

Extracted from experiments_no_output.py (pre-plot setup): data access (get_ys,
get_series_y, load_results + caches), registries (XVAR_FNS, YVAR_LABELS, palettes,
model_name_options), the grid/griddable plotting machinery, and build_hooks /
HookConsts. Generic + dataset-agnostic; the notebook builds HK and owns its dataset
constants. Import it (repo root on sys.path) instead of duplicating these cells.
"""
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from collections.abc import Callable
import seaborn
from functools import reduce
from contextlib import contextmanager
from operator import getitem
from functools import wraps
from analysis.color_palettes import make_alternate_versions as make_color_palettes
seaborn.set_style('whitegrid')

# Always run from repo root regardless of notebook location
from utils.model_registry import get_token_count

@contextmanager
def disable(obj, attr_name):
    temp = getattr(obj, attr_name)
    setattr(obj, attr_name, lambda *a, **kw: None)
    yield
    setattr(obj, attr_name, temp)

@contextmanager
def disable_show():
    temp = plt.show
    plt.show = lambda *a, **kw: None
    yield
    plt.show = temp



# %%
# Do not clear cache on "run all"
try:
    _ = res_cache # type: ignore
except NameError:
    res_cache = {}
try:
    _ = hook_cache # type: ignore
except NameError:
    hook_cache = {}

def load_results(dataset_name: str, model_name: str):
    path = os.path.join('data', 'results', dataset_name, f'results_{model_name}.npy')
    if path in res_cache:
        return res_cache[path]
    try:
        res = np.load(path, allow_pickle=True).item()
    except FileNotFoundError as e:
        print(f"skipped: {path}", file=sys.stderr)
        return None, None
    step_nums = sorted(res.keys())
    # print(res)
    res_cache[path] = (res, step_nums)
    return res, step_nums

def make_histogram(
    vals: np.ndarray,
    bins: int = 512,
    log_bins: bool = False,
    normalise_as_pdf: bool = True,
) -> np.ndarray:
    edges = (np.logspace(-10, np.log10(vals.max()), bins + 1) if log_bins
             else np.linspace(0, vals.max(), bins + 1))
    h, _ = np.histogram(vals, bins=edges)
    return h / (h.sum() * np.diff(edges)) if normalise_as_pdf else h

def rebin_hist(hist: np.ndarray, bins=None, normalise_as_pdf=True):
    if bins is not None:
        assert not hist.size % bins, "old histogram bincount must be divisible by new histogram bincount"
        hist = hist.reshape(-1,hist.size//bins)
        hist = hist.mean(axis=1) if normalise_as_pdf else hist.sum(axis=1)
    return hist

adjust_hook = {
    "histogram": lambda hist, bins=None: rebin_hist(hist, bins=bins),
    "loghistogram": lambda hist, bins=None: rebin_hist(hist, bins=bins),
}

def _cos(a, b):                                         # acts_mean_vec is a torch tensor -> float() to plot cleanly
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
def _joined_residual(node, model):                      # the sub-block input a block output adds into;
    inp = node.replace('.out', '.in')                   # Pythia's parallel residual: mlp.in === attn.in
    return inp.replace('.mlp.in', '.attn.in') if 'pythia' in model else inp
def _drift(V, lag=1):                                   # series-mode: self-cosine across checkpoints
    return [np.nan]*lag + [_cos(V[t], V[t-lag]) for t in range(lag, len(V))]
def _drift_ema(V, beta=0.7):                            # series-mode: cosine vs EMA of own history (h reassigned, not mutated)
    out, h = [np.nan], V[0]
    for t in range(1, len(V)): out.append(_cos(V[t], h)); h = beta*h + (1-beta)*V[t]
    return out

def _rankme(evs: np.ndarray) -> float:
    p = evs[evs > 0]
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def _alpha(evs: np.ndarray, k0: int = 32, k1: int = 300) -> float:
    """Stringer-style weighted log-log slope over eigenvalue ranks [k0, k1) (0-based).
    k1 <= 0 counts back from the end of the positive spectrum (e.g. -20 = drop last 20)."""
    lam = np.asarray(evs, float); lam = lam[lam > 0]
    k1 = min(k1, len(lam)) if k1 > 0 else len(lam) + k1
    r = np.arange(k0, k1) + 1.0
    x = np.stack([-np.log(r), np.ones_like(r)], 1)
    w = (1.0 / r)[:, None]
    return float(np.linalg.solve(x.T @ (x * w), (w * x).T @ np.log(lam[k0:k1]))[0])

virtual_hooks = {
    "peak_eigval": (["eigenspectrum", "trace"], (lambda evs, tr: evs[0].item()*tr)),
    "tail_rankme": (["eigenspectrum"], (lambda evs, k=32: _rankme(np.asarray(evs)[k:]))),
    "alpha_window": (["eigenspectrum"], _alpha),   # kwargs k0/k1: bulk alphaReQ, head excluded
    "top1_dominance": (["eigenspectrum"], (lambda evs: float(evs[0] / evs[1]))),   # λ1/λ2
    "eigvals": (["eigenspectrum", "trace"], (lambda evs, tr: evs*tr)),
    "histogram":    (["eigvals"], (lambda evs, bins=512: make_histogram(evs, bins))),
    "loghistogram": (["eigvals"], (lambda evs, bins=512: make_histogram(evs, bins, log_bins=True))),
    "mean_norm": ([{'key': lambda k: f"{k.split('_')[0]}_mean_metrics", 'yvar': 'mean_norm'}], lambda x: x),
    "mean_frac": (["trace", "mean_norm"], lambda tr, mu: mu**2 / tr),
    "mean_ratio": (["trace", "mean_norm"], lambda tr, mu: mu**2 / (tr-mu**2)),
    "rankme_center_diff": (["rankme", {'key': lambda k: k.replace("_centered", "_uncentered"), 'yvar': 'rankme'}], lambda c, u: c - u),
    "rankme_center_prop": (["rankme", {'key': lambda k: k.replace("_centered", "_uncentered"), 'yvar': 'rankme'}], lambda c, u: (c - u)/c),
    "cos_to_res":   ([{'yvar': 'acts_mean_vec'}, {'node': _joined_residual, 'yvar': 'acts_mean_vec'}], _cos),
    "magratio_res": ([{'yvar': 'acts_mean_vec'}, {'node': _joined_residual, 'yvar': 'acts_mean_vec'}], lambda a, b: float(np.linalg.norm(a) / np.linalg.norm(b))),
    "cos_drift":     ([{'yvar': 'acts_mean_vec'}], _drift, 'series'),
    "cos_drift_ema": ([{'yvar': 'acts_mean_vec'}], _drift_ema, 'series'),
    # Per-sub-step ledger, assembled from incremental_overlap at blk*.{attn,mlp}.out nodes
    # (sequential families only — Pythia's parallel sub-steps are fictitious). s_next comes
    # from the sibling mlp's s_in (attn step) or the block ledger's s_next (mlp step).
    "sub_s_next":       ([{'node': lambda n, m: (f"{n.split('.')[0]}.mlp.out", 'incremental_overlap', 's_in')
                           if '.attn.' in n else (n.split('.')[0], 'block_ledger', 's_next')}], lambda x: x),
    "sub_delta_s":      (["sub_s_next", "s_in"],  lambda nxt, si: nxt - si),
    "sub_quality":      (["w", "s_out", "s_in"],  lambda w, so, si: (1 - w) * (so - si)),
    "sub_interference": (["sub_s_next", "s_mix"], lambda nxt, smix: nxt - smix),
}

def _try_get(d, path):
    try:
        return reduce(getitem, path, d)
    except (TypeError, KeyError):
        return None

def _operand(req, hook, model):
    if isinstance(req, str): return hook, req
    node, k = req.get('node', hook[0]), req.get('key')
    if callable(node): node = node(hook[0], model)   # node selector may depend on the model (family aliases)
    if isinstance(node, tuple): return node[:-1], node[-1]   # selector may return a full (node, key, yvar) redirect
    return ((node,) if k is None else (node, k(hook[1]) if callable(k) else k)), req['yvar']

def get_ys(source_file, model_name, hook, yvar, yvar_kwargs={}):
    hookpath = (*hook, yvar)
    step_nums = None

    cache_id = '\\'.join((source_file, model_name, *hookpath, *yvar_kwargs.keys(), *map(str, yvar_kwargs.values())))
    if cache_id in hook_cache:
        return hook_cache[cache_id]

    try:
        res, step_nums = load_results(source_file, model_name)
        if step_nums is None:
            return None, None
        # keep only steps carrying the hook (e.g. drift metrics start at the 2nd checkpoint);
        # a hook present at no step falls through to the virtual-hook path as before
        pairs = [(y, s) for s in step_nums if (y := _try_get(res[s], hookpath)) is not None]
        if not pairs:
            raise KeyError(hookpath)
        ys, step_nums = map(list, zip(*pairs))

    except (TypeError, KeyError) as e:
        if step_nums is None: raise e
        if yvar not in virtual_hooks:
            print("broken hook:", hookpath, file=sys.stderr)
            raise e

        required_yvars, hook_transform, *mode = virtual_hooks[yvar]   # 'series' => whole-series transform
        try:
            operands = [get_ys(source_file, model_name, *_operand(req, hook, model_name))[0] for req in required_yvars]
        except KeyError:
            print(f"hook: {hook}\nrequired vars: {required_yvars}\nhook transform: {hook_transform}")
            raise e
        if None in operands:
            return None, None
        ys = (hook_transform(*operands, **yvar_kwargs) if mode and mode[0] == 'series'
              else [hook_transform(*per, **yvar_kwargs) for per in zip(*operands)])

    hook_cache[cache_id] = (ys, step_nums)
    return ys, step_nums

# get y but abstracted for only one step, not faster lol, I'm lazy
def get_y(source_file, model_name, hook, yvar, step_num, yvar_kwargs={}):
    ys, step_nums = get_ys(source_file, model_name, hook, yvar, yvar_kwargs)
    if ys is None or step_nums is None: return None
    # print(step_num)
    # print(ys)
    return ys[step_nums.index(step_num)] if step_num is not None else ys[-1]

def get_xs_steps(model_name, step_nums):
    return step_nums

def get_xs_tokens(model_name, step_nums):
    return [get_token_count(model_name.split('_')[0], s) for s in step_nums]


def _parse_param_count(model_name):
    """Extract parameter count from model name (e.g. 'pythia-70m' -> 70e6, 'OLMo-2-0425-1B' -> 1e9)."""
    name = model_name.lower()
    if 'pythia' in name:
        size_str = model_name.split('pythia-')[-1].split('-deduped')[0]
    elif 'olmo' in name:
        size_str = model_name.split('-')[-1]
    elif 'nanochat' in name:
        return 185e6   # d12; ~185.6M params (nanochat names carry depth tags, not counts)
    else:
        raise ValueError(f"Cannot extract param count from {model_name}")
    multiplier = 1e6 if size_str[-1].lower() == 'm' else 1e9
    return float(size_str[:-1]) * multiplier


def get_xs_flops(model_name, step_nums):
    num_params = _parse_param_count(model_name)
    return [num_params * s for s in step_nums]


XVAR_FNS = {
    'steps': get_xs_steps,
    'tokens': get_xs_tokens,
    'flops': get_xs_flops,
}

XVAR_LABELS = {
    'steps': 'Steps',
    'tokens': 'Pretraining tokens',
    'flops': 'Flops',
    'isoflops': 'Parameters',
}

YVAR_LABELS = {
    'alpha': r'$\alpha$',
    'rankme': 'ev RankMe',
    'matrix_entropy': 'Matrix (ev) Entropy',
    'sv_entropy': 'sv Entropy',
    'true_rankme': 'sv RankMe',
    'trace': 'Trace',
    'log_det': 'Log-det',
    'log_det_a': 'Log-det of A',
    'log_det_b': 'Log-det of B',
    'log_det_g': 'Log-det of G',
    'peak_eigval': r'Peak $\nu$',
    'histogram': r'$p(\nu)$',
    'loghistogram': r'$p(\nu)$',
    'eigenspectrum': r'$\nu$ Relative',
    'eigvals': r'$\nu$ Absolute',
    'mean_norm': r'$\|\mu\|_2$',
    'mean_frac': r'Mean fraction of total energy — $\rho_\mu$',
    'mean_ratio': r'Mean-to-fluctuation ratio — $\text{SNR}_\mu$',
    'rankme_center_diff': r'$\text{RankMe}_c$ - $\text{RankMe}_u$',
    'rankme_center_prop': r'Effect of mean on $\text{RankMe}_c$',
    'max_overlap':           r'$\max_j\,|\langle\hat\mu, v_j\rangle|$',
    'max_overlap_idx':       r'$\arg\max_j$ overlap (eig index)',
    'top_overlap':           r'$|\langle\hat\mu, v_0\rangle|$ (top dir)',
    'pr':                    r'Participation ratio of $\mu$',
    'pr_weighted':           r'PR of energy-weighted $\mu$',
    'rayleigh':              r'$\hat\mu^\top\Sigma\hat\mu$ (var. along $\mu$)',
    'rayleigh_normed':       r'$\hat\mu^\top\Sigma\hat\mu/\lambda_{\max}$',
    'mahalanobis':           r'$\hat\mu^\top\Sigma^{-1}\hat\mu$ (damped)',
    'centroid_idx':          r'Spectral centroid of $\mu$',
    'centroid_idx_weighted': r'Energy-weighted centroid of $\mu$',
    'profile':               r'$p_j=|\langle\hat\mu, v_j\rangle|^2$',
    'profile_weighted':      r'$\tilde p_j\propto\lambda_j p_j$',
    'cos_to_res':            r'$\cos(\mu_{out}, \mu_{in})$ (vs joined residual)',
    'magratio_res':          r'$\|\mu_{out}\|/\|\mu_{in}\|$',
    'cos_drift':             r'$\cos(\mu_t, \mu_{t-1})$ (drift)',
    'cos_drift_ema':         r'$\cos(\mu_t, \overline{\mu}_{<t})$ (EMA drift)',
    # Block-composition / ledger family (samples runs)
    'tail_rankme':           r'RankMe of $\lambda_{>k}$ (head removed)',
    'alpha_window':          r'$\alpha_{[k_0,k_1)}$ (windowed)',
    'top1_dominance':        r'$\lambda_1/\lambda_2$ (head top-1 dominance)',
    'delta_s':               r'$\Delta S$ (block rank-entropy change)',
    'chi':                   r'$\chi$ (overlap)',
    'quality':               r'quality $\sum_i w_i(S_i - S_{in})$',
    'interference':          r'$I$ (interference)',
    'chi_frac':              r'$\chi/H(w)$',
    'sub_delta_s':           r'$\Delta S$ (sub-step)',
    'sub_quality':           r'quality (sub-step)',
    'sub_interference':      r'$I$ (sub-step)',
    'delta_rankme':          r'$\Delta$RankMe (leave-one-out)',
    'rankme_ablated':        r'RankMe$(\Sigma_{r\setminus k})$',
    'cka_drift':             r'CKA$(c^{(t)}, c^{(t-1)})$',
    'tr_P':                  r'$\mathrm{tr}\,P_k$ (reinforce/cancel)',
    'tr_R':                  r'$\mathrm{tr}\,R_k$',
    'R_over_r':              r'$\mathrm{tr}\,R_k/\mathrm{tr}\,\Sigma_r$',
    'R_over_ck':             r'$\mathrm{tr}\,R_k/\mathrm{tr}\,\Sigma_{c_k}$',
    'cka_cr':                r'CKA$(c_k, r)$',
    'cos_cr':                r'mean $\cos(c_i, r_i)$ (per-token)',
    'migration':             r'tail$\to$head migration',
}

base_colors = [
    'turquoise', 'turquoise',
    'cornflowerblue', 'cornflowerblue',
    'lime', 'lime',
    'darkgreen', 'darkgreen',
    'gold', 'gold',
    'dodgerblue', 'dodgerblue',
    'magenta', 'magenta',
    'purple', 'purple',
    'deeppink', 'deeppink',
    'brown', 'brown',
    'blue',
    'red',
    'darkorange',   # nanochat-d12
]

model_name_options = [
    'pythia-14m', 'pythia-14m-deduped',
    'pythia-31m', 'pythia-31m-deduped',
    'pythia-70m', 'pythia-70m-deduped',
    'pythia-160m', 'pythia-160m-deduped',
    'pythia-410m', 'pythia-410m-deduped',
    'pythia-1b', 'pythia-1b-deduped',
    'pythia-1.4b', 'pythia-1.4b-deduped',
    'pythia-2.8b', 'pythia-2.8b-deduped',
    'pythia-6.9b', 'pythia-6.9b-deduped',
    'pythia-12b', 'pythia-12b-deduped',
    'OLMo-2-0425-1B',
    'OLMo-2-1124-7B',
    'nanochat-d12',
]

# def get_ls(model_name: str, xvar=None):
#     if xvar == 'isoflops':
#         return '*' if 'deduped' in model_name else 's'

#     ls_exceptions = {'_': '-.', 'deduped': '--'}
#     for substring, ls in ls_exceptions.items():
#         if substring in model_name:
#             return ls
#     return '-'
def get_ls(label: str, xvar: str = None, exceptions: dict = {}):
    ls_exceptions = {'_': '-.'} | exceptions
    for substring, ls in ls_exceptions.items():
        if substring.lower() in label.lower():
            return ls
    return '-'

def get_model_label(model_name: str):
    parts = model_name.replace('_','-').split('-')

    if parts[0] == 'nanochat':
        return ' '.join(['Nanochat', *[p.upper() for p in parts[1:]]])

    if parts[0] == 'pythia':
        if len(parts) > 2:
            if parts[2] == 'deduped':
                parts.pop(2)
                parts[1] += 'd'
        parts = [parts[0].capitalize(), parts[1].upper(), *parts[2:]]

    if parts[0] == 'OLMo' and parts[1] == '2':
        parts = ['OLMo-2', parts[3], *parts[4:]]

    return ' '.join(parts)

def make_label(model_name, source_label, label_model, label_source):
    return (
        f'{get_model_label(model_name)}' * label_model
      + f' - '                           * label_model * label_source
      + f'{source_label}'                * label_source
    )


# %%
def _export_transcription(title, calls, extra_kwargs):
    os.makedirs('analysis/transcriptions', exist_ok=True)
    safe = re.sub(r'[^\w\-]', '_', title or 'untitled').strip('_')
    path = f'analysis/transcriptions/{safe}.txt'

    lines = [f"Figure: {title}", "=" * len(f"Figure: {title}"), ""]

    for func, args, kwargs in calls:
        if func.__name__ != 'plot_group':
            continue

        effective     = extra_kwargs | kwargs
        yvar          = args[0]
        data_sources  = args[1]
        model_names   = args[2]
        xvar          = effective.get('xvar', 'tokens')
        subplot_title = effective.get('title') or (yvar if isinstance(yvar, str) else yvar[0])
        disable_didx  = effective.get('disable_didx', [])
        disable_midx  = effective.get('disable_midx', [])
        omit_step0    = effective.get('omit_step0', False)

        yvar_str    = yvar if isinstance(yvar, str) else yvar[0]
        yvar_kwargs = {} if isinstance(yvar, str) else yvar[1]

        lines.append(f"## {subplot_title} ({yvar_str})")

        for midx, model_name in enumerate(model_names):
            if midx in disable_midx:
                continue
            for didx, source in enumerate(data_sources):
                if didx in disable_didx:
                    continue
                source_file, hook, label = source[0], source[1], source[2]
                ys, step_nums = get_ys(source_file, model_name, hook, yvar_str, yvar_kwargs)
                if ys is None or step_nums is None:
                    continue
                xs = XVAR_FNS[xvar](model_name, step_nums)

                lines.append(f"  {label} | {model_name}:")
                lines.append(f"  {'tokens':>14}  {'value':>14}  step")
                for x, y, s in zip(xs[int(omit_step0):], ys[int(omit_step0):], step_nums[int(omit_step0):]):
                    if hasattr(y, '__len__'):
                        continue  # skip array-valued metrics (eigenspectrum etc.)
                    lines.append(f"  {x:>14.4e}  {float(y):>14.6g}  {s}")
                lines.append("")

        lines.append("")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


# %%
def plot_model_training(model_name: str, source: tuple[str], xvar: str = 'tokens',
                        yvar: str = 'alpha', color: str = 'k', marker: str = '',
                        label_source: bool = False, label_model: bool = True,
                        ls_exceptions: dict = {}, yvar_kwargs: dict = {},
                        omit_step0: bool = False, lw: float = 3, alpha: float = 1.0, zorder=None, transform=None):
    source_file, hook, source_label, *custom_yvar = source
    if custom_yvar:
        yvar = custom_yvar[0]
    if not isinstance(yvar, str):
        yvar, yvar_kwargs = yvar

    ys, step_nums = get_ys(source_file, model_name, hook, yvar, yvar_kwargs)
    if ys is None: return
    if transform is not None: ys = transform(np.asarray(ys, float))
    xs = XVAR_FNS[xvar](model_name, step_nums)
    ls = get_ls(source_label+model_name, exceptions=ls_exceptions)

    # step-0 has 0 tokens: unplottable on the log token axis (its clipped segment renders as a
    # spurious horizontal line entering from the left edge)
    skip = max(int(omit_step0), int(xvar == 'tokens' and xs[0] <= 0))
    label = make_label(model_name, source_label, label_model, label_source)
    plt.plot(xs[skip:], ys[skip:], marker=marker, color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder, label=label)
    return xs[skip:], ys[skip:]

def get_series_y(data_source, model_name, yvar, step, labels):
    yvar_kwargs = {}
    source_file, hook, source_label, *custom_yvar = data_source
    if custom_yvar:
        yvar = custom_yvar[0]
    if not isinstance(yvar, str):
        yvar, yvar_kwargs = yvar
    label = make_label(model_name, source_label, *labels)
    return get_y(source_file, model_name, hook, yvar, step, yvar_kwargs=yvar_kwargs), label

def smooth_spectrum(a: np.ndarray, window: int = 22, peak: int = 3) -> np.ndarray:
    """window: smoothing width (odd; 0=off). Endpoints always pinned so edge peaks
    survive. peak<=1 -> Savitzky-Golay (denoise around the mean). peak>1 -> bias toward
    the window's upper values via a rolling quantile q=peak/(peak+1) (peak=3->0.75,
    7->0.875, larger->max), then a savgol pass so it stays smooth (no staircase/plateaus).
    Robust on the wide-dynamic-range spectra: no powers/underflow."""
    a = np.asarray(a, dtype=float)
    if not window or window < 3 or a.size < 3:
        return a
    from scipy.signal import savgol_filter
    w = int(window) | 1                          # force odd
    w = min(w, a.size if a.size % 2 else a.size - 1)
    if w < 3:
        return a
    po = min(3, w - 1)
    if peak and peak > 1:
        from numpy.lib.stride_tricks import sliding_window_view
        q = peak / (peak + 1.0)
        win = sliding_window_view(np.pad(a, w // 2, mode='edge'), w)   # (n, w)
        sm = savgol_filter(np.quantile(win, q, axis=1), w, po, mode='interp')
    else:
        sm = savgol_filter(a, w, po, mode='interp')
    sm = np.where(np.isfinite(sm) & (sm > 0), sm, a)             # positivity + finite (log)
    sm[0], sm[-1] = a[0], a[-1]                  # pin edges -> preserve edge peaks
    return sm


def panel_palettes(n_sources: int, color_palette: str = 'hue_shift',
                   color_kwargs: dict = {}) -> list:
    """Per-source colour palettes (one list per data source, indexed by model)."""
    if color_palette == 'hue_shift':
        color_kwargs = dict(shift=0.80 / n_sources, light_base=0.85) | color_kwargs
    return make_color_palettes(base_colors, n_sources, method=color_palette, **color_kwargs)


def plot_spectrum_series(series: list[tuple], mode: str = 'line',
                         max_ev: int = None, hist_bins: int = 256,
                         log_bins=False, input_is_bins=True,
                         smooth: int = 0, peak: int = 3):
    # smooth=0 default: sorted eigval spectra are monotone/noise-free — a SavGol window only
    # invents features at the head cliff (fake dip at idx ~10-30). Opt in for profile-like ys.
    is_hist = mode in ('hist', 'histogram')

    if is_hist and input_is_bins:
        # Already-binned data: plot counts directly against bin indices.
        # No concatenation needed — bins are fixed by the virtual hook.
        for arr, color, label in series:
            plt.bar(np.arange(len(arr)), arr, color=color, label=label, alpha=0.6)
        return

    if is_hist:
        all_vals = np.concatenate([s[0] for s in series])
        if log_bins:
            pos = all_vals[all_vals > 0]
            bin_edges = np.logspace(np.log10(pos.min()), np.log10(pos.max()), hist_bins + 1)
        else:
            bin_edges = np.linspace(all_vals.min(), all_vals.max(), hist_bins + 1)

        for arr, color, label in series:
            plt.hist(arr, bins=bin_edges, color=color, label=label, alpha=0.6)
        return

    for i, (arr, color, label) in enumerate(series):
        xs = np.arange(len(arr))
        if mode == 'line':
            plt.plot(xs, smooth_spectrum(arr, smooth, peak), color=color, lw=2, label=label)
        elif mode == 'bar':
            width = 0.8 / len(series)
            offset = (i - len(series) / 2 + 0.5) * width
            plt.bar(xs + offset, arr, width=width, color=color, label=label, alpha=0.8)

def legend_or_colorbar(legend_max: int = 12):
    """Normal legend, or — for depth-gradient groups past legend_max lines — a slim colorbar
    of the actual line colors, ticked with the first/middle/last labels."""
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(labels) <= legend_max:
        return plt.legend()
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import ScalarMappable
    colors = [h.get_color() if hasattr(h, 'get_color') else h.get_facecolor() for h in handles]
    cb = plt.colorbar(ScalarMappable(cmap=ListedColormap(colors)), ax=plt.gca(),
                      fraction=0.046, pad=0.02,
                      ticks=[0.5 / len(labels), 0.5, 1 - 0.5 / len(labels)])
    cb.ax.set_yticklabels([labels[0], labels[len(labels) // 2], labels[-1]], fontsize=9)


_grid = None

def grid_start(ncols=2, figsize=None, title=None, savefig=True, savedir=None, ylim=None, sharey=False, **kwargs):
    global _grid
    _grid = dict(ncols=ncols, figsize=figsize, title=title, savefig=savefig, savedir=savedir, ylim=ylim, sharey=sharey, extra_kwargs=kwargs, calls=[])

def grid_show():
    global _grid
    calls = _grid['calls']
    ncols = _grid['ncols']
    nrows = (len(calls) + ncols - 1) // ncols
    figsize = _grid['figsize'] or (7 * ncols, 5 * nrows)
    title = _grid['title']
    savefig = _grid['savefig']
    savedir = _grid['savedir']
    ylim = _grid['ylim']
    sharey = _grid['sharey']
    extra_kwargs = _grid['extra_kwargs']
    _grid = None

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()
    for i, (func, args, kwargs) in enumerate(calls):
        func(*args, _ax=axes[i], **(extra_kwargs | kwargs))
    used = axes[:len(calls)]
    if sharey and ylim is None:                       # share one comparable y-range across the grid
        lims = [ax.get_ylim() for ax in used]
        ylim = (min(l[0] for l in lims), max(l[1] for l in lims))
    if ylim:
        for ax in used: ax.set_ylim(*ylim)
    for ax in axes[len(calls):]:
        ax.set_visible(False)
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title)
        plt.subplots_adjust(top= 0.984 - (0.10 / nrows))
    plt.show()

    _export_transcription(title, calls, extra_kwargs)

    if savefig:
        savedir = savedir or title or 'unnamed'
        outdir = Path('analysis/figures') / savedir
        outdir.mkdir(parents=True, exist_ok=True)
        for i, (func, args, kwargs) in enumerate(calls):
            subplot_title = (extra_kwargs | kwargs).get('title')
            safe_name = f'{i}_{subplot_title}'.replace('/', '_').replace(' ', '_') if subplot_title else f'{i}_subplot'
            sf, _ = plt.subplots(figsize=(7, 5))
            func(*args, _ax=sf.axes[0], **(extra_kwargs | kwargs))
            sf.tight_layout()
            sf.savefig(outdir / f'{safe_name}.pdf', bbox_inches='tight')
            plt.close(sf)


def griddable(func: Callable):

    @wraps(func)
    def _grid_aux(*args, _ax=None, **kwargs):
        if _ax is None and _grid is not None:
            return _grid['calls'].append((_grid_aux, args, kwargs))
        if _ax is None:
            return func(*args, **kwargs)
        plt.sca(_ax)
        with disable(plt, 'show'):
            return func(*args, **kwargs)

    return _grid_aux

@griddable
def plot_group(
    yvar: str | tuple,
    data_sources: list[str],
    model_names: list[str],
    xvar: str = "tokens",
    xlog: bool = True,
    ylog: bool = False,
    title: str | None = None,
    color_palette: str = "hue_shift",
    color_kwargs: dict = {},
    disable_didx: list = [],
    disable_midx: list = [],
    ls_exceptions: dict = {},
    hold_color_for_n: int | None = None,
    aggregate: bool = False, background=None, weight=None, normalize=None,
    agg_color='tab:blue', band_color='#6e7f95', contrib_alpha=0.35, contrib_lw=1.0,
    legend_max: int = 12, lw: float = 3,
    **kwargs
):
    palettes = panel_palettes(len(data_sources), color_palette, color_kwargs)

    if hold_color_for_n is None:
        hold_color_for_n = 2 if ls_exceptions else 1
    yvarname = yvar if isinstance(yvar, str) else yvar[0]

    agg, bgr = [], None
    for midx, model_name in enumerate(model_names):
        if midx in disable_midx: continue
        norm = None                                                          # shape normalization vs the reference
        if background and normalize:
            bg_raw = np.asarray(get_ys(background[0], model_name, background[1], yvarname)[0], float)
            if normalize == 'anchor':                                    # interior peak, skipping the random-init crash
                _lo = int(np.argmin(bg_raw)); _ai = _lo + int(np.argmax(bg_raw[_lo:]))
                norm = (lambda y, i=_ai: y - y[i])
            elif normalize == 'ref':  norm = (lambda y, b=bg_raw: y - b)  # subtract ref curve
        for didx, data_source in enumerate(data_sources):
            if didx in disable_didx: continue
            if not didx % hold_color_for_n:
                color = palettes[didx][model_name_options.index(model_name)]
            r = plot_model_training(model_name, data_source,
                                xvar=xvar, yvar=yvar, color=color, transform=norm,
                                lw=contrib_lw if aggregate else lw,           # thinner contributions
                                alpha=contrib_alpha if aggregate else 1.0,    # colored but low-opacity
                                zorder=1 if aggregate else None,              # contributions at the bottom
                                label_source=not aggregate and (len(data_sources)>1 or len(model_names) <= 1),
                                label_model=not aggregate and len(model_names)>1,
                                ls_exceptions=ls_exceptions, **kwargs)
            if aggregate and r:                                              # weight = uncentered trace (energy)
                w = get_ys(data_source[0], model_name, (data_source[1][0], 'acts_uncentered'), weight)[0] if weight else None
                agg.append((*r, np.asarray(w)[-len(r[1]):] if w is not None else None))   # align to skipped step-0
        if background:                                                       # zorder 3: above contribs, just below mean
            bgr = plot_model_training(model_name, background, xvar=xvar, yvar=yvar, color='0.45', transform=norm,
                                      label_source=True, label_model=False, zorder=3, **kwargs)

    if aggregate and agg:
        xs, Y = agg[0][0], np.array([y for _, y, _ in agg])
        if weight:
            W = np.array([w for *_, w in agg]); mu = (W*Y).sum(0)/W.sum(0)
            sd = np.sqrt((W*(Y-mu)**2).sum(0)/W.sum(0))
        else:
            mu, sd = Y.mean(0), Y.std(0)
        plt.fill_between(xs, mu-sd, mu+sd, color=band_color, alpha=.3, zorder=2)   # soft blue-grey band
        plt.plot(xs, mu, color=agg_color, lw=3, zorder=4, label='weighted mean' if weight else 'mean')
        ref = [mu-sd, mu+sd] + ([np.asarray(bgr[1])] if bgr else [])          # frame y on mean+band+bg, not contribs
        lo, hi = min(a.min() for a in ref), max(a.max() for a in ref)
        plt.ylim(lo - 0.05*(hi-lo), hi + 0.05*(hi-lo))

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    if title is not None: plt.title(title)
    plt.xlabel(XVAR_LABELS[xvar], fontsize=14)
    plt.ylabel(YVAR_LABELS.get(yvarname, yvarname) + (' (rel.)' if normalize else ''), fontsize=14)
    plt.xlim(10e7, 10**12.7)
    # print([float(np.log10(lim)) for lim in plt.xlim()])
    legend_or_colorbar(legend_max)
    plt.show()


@griddable
def plot_spectrum(
    yvar: str | tuple,
    data_sources: list[str],
    model_names: list[str],
    xlog: bool = True,
    ylog: bool = True,
    mode: str = 'default',
    step: int = None,
    title: str | None = None,
    color_palette: str = 'hue_shift',
    color_kwargs: dict = {},
    disable_didx: list = [],
    disable_midx: list = [],
    legend_max: int = 12,
    **kwargs
):
    palettes = panel_palettes(len(data_sources), color_palette, color_kwargs)

    yvarname = yvar if isinstance(yvar, str) else yvar[0]
    labels = len(model_names) > 1, len(data_sources) > 1 or len(model_names) <= 1

    if mode == 'default':
        mode = 'hist' if yvarname == 'histogram' else 'line'
    if yvarname == 'histogram' and xlog:
        yvarname = 'loghistogram'

    series = []
    for midx, model_name in enumerate(model_names):
        if midx in disable_midx: continue
        for didx, data_source in enumerate(data_sources):
            if didx in disable_didx: continue
            arr, label = get_series_y(data_source, model_name, yvar, step, labels)
            if arr is None: continue
            color = palettes[didx][model_name_options.index(model_name)]
            series.append((arr, color, label))

    plot_spectrum_series(series, mode=mode, log_bins=xlog, **kwargs)

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    if title is not None: plt.title(title)
    plt.xlabel(r'$\nu$' if mode in ('hist', 'histogram') else 'Component index', fontsize=14)
    plt.ylabel(YVAR_LABELS.get(yvarname, yvarname), fontsize=14)
    legend_or_colorbar(legend_max)
    plt.show()

# 100%-stacked area: each block output's share of the residual energy (uncentered trace)
# over training, stacked bottom = first layer -> top = final layer. Light griddable that
# reuses plot_group's colour-palette logic. (Shares are among the *measured* outputs,
# matching the weighted average above.)
@griddable
def plot_layer_contribution(model, sources, xvar='tokens', yvar='trace', title=None,
                            color_palette='gradient', color_kwargs={}, key=True,
                            normalize=True, sep_lw=0.0, sep_color='white'):
    """Depth-stacked contributions over training. normalize=True: 100%-shares (nonneg yvar,
    e.g. uncentered trace = gross energy). normalize=False: raw signed values — positives
    stack up, negatives down (e.g. the ledger's delta_s, which includes overlap/cross terms)."""
    palettes = make_color_palettes(base_colors, len(sources), method=color_palette, **color_kwargs)
    W, xs = [], None
    for src in sources:
        ys, step_nums = get_ys(src[0], model, src[1], yvar)
        if ys is None: continue
        W.append(ys); xs = XVAR_FNS[xvar](model, step_nums)
    W = np.asarray(W, float)
    frac = W / W.sum(0) if normalize else W
    colors = [palettes[i][0] for i in range(len(W))]            # gradient is model-independent -> [0]
    stack = lambda Y: plt.stackplot(xs, *Y, colors=colors, linewidth=sep_lw,
                                    edgecolor=sep_color if sep_lw else 'none')
    if normalize or (frac >= 0).all():
        stack(frac)
        plt.ylim(0, 1) if normalize else None
    else:                                                       # signed: positives up, negatives down
        stack(np.clip(frac, 0, None)); stack(np.clip(frac, None, 0))
        plt.axhline(0, color='0.3', lw=0.8)
    plt.xscale('log'); plt.xlim(10e7, 10**12.7)
    if title: plt.title(title)
    plt.xlabel(XVAR_LABELS[xvar], fontsize=14)
    plt.ylabel('Share of residual energy' if normalize else YVAR_LABELS.get(yvar, yvar), fontsize=14)
    if key:                                                     # depth colour key (one bar, bottom->final)
        from matplotlib.colors import ListedColormap
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=ListedColormap(colors)); sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), pad=0.01, ticks=[]).set_label('depth (0 → final)')
    plt.show()


@griddable
def plot_heatmap(M, labels=None, labels2=None, title=None, cmap='coolwarm', vmin=-1, vmax=1,
                 hide_diag=False, dynamic=False):
    """labels = rows (y); labels2 = columns (x), defaulting to labels (square matrices)."""
    M = np.array(M, dtype=float)                         # fresh copy -> safe to NaN the diagonal
    if hide_diag and M.shape[0] == M.shape[1]: np.fill_diagonal(M, np.nan)
    if dynamic:                                          # symmetric range around 0 (gray centre)
        v = np.nanmax(np.abs(M)); vmin, vmax = -v, v
    cmap = plt.get_cmap(cmap).copy(); cmap.set_bad('white')   # NaN cells (hidden diagonal) -> white
    im = plt.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    cols = labels2 if labels2 is not None else labels
    if cols is not None: plt.xticks(range(len(cols)), cols, rotation=90, fontsize=6)
    if labels is not None: plt.yticks(range(len(labels)), labels, fontsize=6)
    plt.grid(False)                                      # seaborn's grid draws through cell centres
    if title: plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04); plt.show()


def submatrix(M, labels, rows='', cols=''):
    """Label-substring row/col selection of a labelled matrix ('' = all).
    Returns (M', row_labels, col_labels) — e.g. rows='attn', cols='mlp'."""
    ri, ci = ([i for i, l in enumerate(labels) if k in l] for k in (rows, cols))
    return np.asarray(M)[np.ix_(ri, ci)], [labels[i] for i in ri], [labels[i] for i in ci]


def block_mean_cos(model, sources, step=None):           # block×block cosine-of-means matrix at a step
    V = np.array([get_y(s[0], model, (s[1][0],), 'acts_mean_vec', step) for s in sources])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return V @ V.T


@griddable
def plot_wo_topk(source, model, yvar, k=3):
    """Final-stream yvar vs the same without the k most rank-influential writes (ranked by
    training-average |leave-one-out ΔRankMe|): the additive estimate (rankme only —
    first-order, single-write deltas ignore interactions) + the exact joint_ablation series
    when a run carries it (docs/next_run_additions.md #3)."""
    res, steps = load_results(source, model)
    last = res[steps[-1]]
    d = {n: get_ys(source, model, (n, 'ablation_contribution'), 'delta_rankme')[0]
         for n in last if re.fullmatch(r'blk\d+\.(attn|mlp)\.out', n) and 'ablation_contribution' in last[n]}
    top = sorted(d, key=lambda n: -np.mean(np.abs(d[n])))[:k]
    lbl = ', '.join(n.removeprefix('blk').removesuffix('.out') for n in top)
    r, rsteps = get_ys(source, model, ('before_final_norm', 'acts_centered'), yvar)
    xs = np.asarray(XVAR_FNS['tokens'](model, rsteps))
    plt.plot(xs[1:], np.asarray(r)[1:], lw=3, color='0.3', label=yvar)
    if yvar == 'rankme':                                 # the only yvar with a stored delta
        plt.plot(xs[1:], (np.asarray(r) - sum(np.asarray(d[n]) for n in top))[1:], lw=3,
                 color='tab:orange', ls='--', label=f'additive − top-{k} ({lbl})')
    if 'joint_ablation' in last.get('', {}):
        j, jsteps = get_ys(source, model, ('', 'joint_ablation'), yvar)
        plt.plot(np.asarray(XVAR_FNS['tokens'](model, jsteps))[1:], np.asarray(j)[1:], lw=3,
                 color='tab:red', label=f'joint − top-{k} ({lbl})')
    plt.xscale('log'); plt.xlim(10e7, 10**12.7)
    plt.xlabel(XVAR_LABELS['tokens'], fontsize=14); plt.ylabel(YVAR_LABELS.get(yvar, yvar), fontsize=14)
    plt.legend(); plt.title(f'{yvar}: normal vs without top-{k} writes')
    plt.show()



# %%
# Hook constants: short name -> (node_path, metric). Built with loops (no 2.5k-line literal).
class HookConsts(dict):
    """short hook name -> (node_path, metric); attribute access mirrors items."""
    __getattr__ = dict.__getitem__

def build_hooks(n_blocks=41):
    HK = HookConsts()
    A = {"AU": "acts_uncentered", "AC": "acts_centered"}
    G = {"GU": "grads_uncentered", "GC": "grads_centered"}
    for p, node in {"AFN": "after_final_norm", "BFN": "before_final_norm"}.items():      # residual
        for t, m in {**A, "BU": "acts_uncentered", "BC": "acts_centered", **G}.items(): HK[f"{p}_{t}"] = (node, m)
        HK[f"{p}_KF"], HK[f"{p}_GB"] = (node, "kfac"), (node, "gen")
    for i in range(n_blocks):
        for p, w in {"U": "up", "D": "down", "G": "gate"}.items():                       # mlp projections
            b = f"blk{i}.mlp.{w}"
            for t, m in A.items(): HK[f"{p}{i}_{t}"] = (f"{b}.in", m)
            for t, m in G.items(): HK[f"{p}{i}_{t}"] = (f"{b}.out", m)
            HK[f"{p}{i}_BU"], HK[f"{p}{i}_BC"] = (f"{b}.out", "acts_uncentered"), (f"{b}.out", "acts_centered")
            HK[f"{p}{i}_KF"], HK[f"{p}{i}_GB"] = (b, "kfac"), (f"{b}.out", "gen")
        up, dn = f"blk{i}.mlp.up.in", f"blk{i}.mlp.down.out"                             # whole-mlp "layer"
        for t, m in A.items(): HK[f"L{i}_{t}"] = (up, m)
        for t, m in G.items(): HK[f"L{i}_{t}"] = (dn, m)
        HK[f"L{i}_BU"], HK[f"L{i}_BC"] = (dn, "acts_uncentered"), (dn, "acts_centered")
        HK[f"L{i}_KF"], HK[f"L{i}_GB"] = (f"blk{i}.mlp", "projections_kfac"), (dn, "gen")
        for p, suf in {"AI": "attn.in", "AO": "attn.out", "ARO": "attn.raw_out",
                       "MI": "mlp.in", "MO": "mlp.out", "MRO": "mlp.raw_out"}.items():    # boundaries
            for t, m in {**A, **G, "GB": "gen"}.items(): HK[f"{p}{i}_{t}"] = (f"blk{i}.{suf}", m)
    return HK


# `import *` exports only this module's own definitions, not the re-exported imports.
__all__ = [n for n in dir() if not n.startswith('_') and n not in {
    'os', 're', 'sys', 'np', 'plt', 'Path', 'tqdm', 'seaborn', 'reduce', 'getitem',
    'wraps', 'contextmanager', 'Callable', 'make_color_palettes', 'get_token_count'}]
