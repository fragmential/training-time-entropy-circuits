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
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
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

def _drift_rate(cka, steps, model, per='dex'):
    """cka_drift corrected for unequal checkpoint spacing: (1 − CKA) per unit of training
    progressed — per='dex' (Δlog10 tokens; log-schedule-natural) or 'gtok' (per 1e9 tokens).
    The raw stored cka_drift compares consecutive checkpoints, so its level is confounded by
    the gap between them (the samples schedule is ~5x denser in log-tokens near the end)."""
    toks = np.array([get_token_count(model, s) for s in steps], float)
    gaps = np.diff(np.log10(toks)) if per == 'dex' else np.diff(toks) / 1e9
    return [np.nan] + [(1 - c) / g if g > 0 else np.nan for c, g in zip(cka[1:], gaps)]

def _rankme(evs: np.ndarray) -> float:
    p = evs[evs > 0]
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))

def _alpha(evs: np.ndarray, k0: int = 32, k1: int = 300) -> float:
    """Stringer-style weighted log-log slope over eigenvalue ranks [k0, k1) (0-based).
    k1 <= 0 and k0 < 0 count back from the end of the positive spectrum, so (-100, -11) is
    the window of the same width read from the tail rather than from the head."""
    lam = np.asarray(evs, float); lam = lam[lam > 0]
    k1 = min(k1, len(lam)) if k1 > 0 else len(lam) + k1
    k0 = k0 if k0 >= 0 else max(0, len(lam) + k0)
    r = np.arange(k0, k1) + 1.0
    x = np.stack([-np.log(r), np.ones_like(r)], 1)
    w = (1.0 / r)[:, None]
    return float(np.linalg.solve(x.T @ (x * w), (w * x).T @ np.log(lam[k0:k1]))[0])

virtual_hooks = {
    "peak_eigval": (["eigenspectrum", "trace"], (lambda evs, tr: evs[0].item()*tr)),
    "tail_rankme": (["eigenspectrum"], (lambda evs, k=32: _rankme(np.asarray(evs)[k:]))),
    "tail_matrix_entropy": (["eigenspectrum"],
                            (lambda evs, k=32: float(np.log(_rankme(np.asarray(evs)[k:]))))),
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
    "cka_drift_rate": ([{'key': lambda k: k, 'yvar': 'cka_drift'}], _drift_rate, 'series_steps'),
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
            results = [get_ys(source_file, model_name, *_operand(req, hook, model_name)) for req in required_yvars]
        except KeyError:
            print(f"hook: {hook}\nrequired vars: {required_yvars}\nhook transform: {hook_transform}")
            raise e
        operands = [r[0] for r in results]
        if None in operands:
            return None, None
        if mode and mode[0] == 'series_steps':                # transform also sees the operand's
            step_nums = results[0][1]                         # steps + model (e.g. gap-normalized drift)
            ys = hook_transform(*operands, steps=step_nums, model=model_name, **yvar_kwargs)
        else:
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
    'tail_matrix_entropy':   r'Matrix entropy of $\lambda_{>k}$ (head removed)',
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
    'cka_drift_rate':        r'$(1-\mathrm{CKA})/\Delta$ (gap-corrected drift)',
    'tr_P':                  r'$\mathrm{tr}\,P_k$ (reinforce/cancel)',
    'tr_R':                  r'$\mathrm{tr}\,R_k$',
    'R_over_r':              r'$\mathrm{tr}\,R_k/\mathrm{tr}\,\Sigma_r$',
    'R_over_ck':             r'$\mathrm{tr}\,R_k/\mathrm{tr}\,\Sigma_{c_k}$',
    'cka_cr':                r'CKA$(c_k, r)$',
    'cos_cr':                r'mean $\cos(c_i, r_i)$ (per-token)',
    'migration':             r'tail$\to$head migration',
}

PANEL_BG = '#F4F4F7'        # seaborn-style panel tint

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
    if not any(func.__name__ == 'plot_group' for func, *_ in calls):
        return
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
                source_file, hook, label, *custom = source
                y_str, y_kw = yvar_str, yvar_kwargs
                if custom:                          # per-source yvar override, as plot_model_training
                    y_str, y_kw = (custom[0], {}) if isinstance(custom[0], str) else custom[0]
                ys, step_nums = get_ys(source_file, model_name, hook, y_str, y_kw)
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
                        omit_step0: bool = False, lw: float = 3, alpha: float = 1.0, zorder=None,
                        transform=None, ls: str | None = None):
    source_file, hook, source_label, *custom_yvar = source
    if custom_yvar:
        yvar = custom_yvar[0]
    if not isinstance(yvar, str):
        yvar, yvar_kwargs = yvar

    ys, step_nums = get_ys(source_file, model_name, hook, yvar, yvar_kwargs)
    if ys is None: return
    if transform is not None: ys = transform(np.asarray(ys, float))
    xs = XVAR_FNS[xvar](model_name, step_nums)
    ls = ls or get_ls(source_label+model_name, exceptions=ls_exceptions)

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


QUAL_CMAPS = {'tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Dark2',
              'Accent', 'Paired'}

def panel_palettes(n_sources: int, color_palette: str = 'hue_shift',
                   color_kwargs: dict = {}) -> list:
    """Per-source colour palettes (one list per data source, indexed by model).

    A qualitative colormap name ('tab10', 'Set2', ...) gives one colour per SOURCE, the same in
    every model's panel. Prefer it for a handful of unordered series: base_colors is built from
    near-pure hues (turquoise, magenta, lime) that vibrate against white, where tab10 is
    deliberately mid-saturation and even in luminance. Keep the repo palettes for ordered
    series, where the light-to-dark ramp is carrying the ordering."""
    if color_palette in QUAL_CMAPS:
        cm = plt.get_cmap(color_palette)
        return [[cm(i % cm.N)] * len(base_colors) for i in range(n_sources)]
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

def depth_colorbar(colors, ticklabels=None, label=None, fs=9):
    """Slim colorbar built straight from an ordered list of line colours (a depth gradient),
    ticked with the first/middle/last of `ticklabels`."""
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import ScalarMappable
    n = len(colors)
    cb = plt.colorbar(ScalarMappable(cmap=ListedColormap(colors)), ax=plt.gca(),
                      fraction=0.046, pad=0.02,
                      ticks=[0.5 / n, 0.5, 1 - 0.5 / n] if ticklabels else [])
    if ticklabels:
        cb.ax.set_yticklabels([ticklabels[0], ticklabels[n // 2], ticklabels[-1]], fontsize=fs)
    if label:
        cb.set_label(label, fontsize=fs)
    return cb


def legend_or_colorbar(legend_max: int = 12, key_labels: set | None = None, key_fs: int = 8):
    """Normal legend, or — for depth-gradient groups past legend_max lines — a slim colorbar
    of the actual line colors, ticked with the first/middle/last labels. key_labels names
    series that sit outside the gradient (a reference curve): they keep a small legend of
    their own and are left out of the colorbar strip."""
    handles, labels = plt.gca().get_legend_handles_labels()
    key_labels = key_labels or set()
    keyed = [(h, l) for h, l in zip(handles, labels) if l in key_labels]
    rest = [(h, l) for h, l in zip(handles, labels) if l not in key_labels]
    if len(rest) <= legend_max:
        return plt.legend()
    colors = [h.get_color() if hasattr(h, 'get_color') else h.get_facecolor() for h, _ in rest]
    depth_colorbar(colors, [l for _, l in rest])
    if keyed:
        plt.legend([h for h, _ in keyed], [l for _, l in keyed], fontsize=key_fs)


_grid = None

def _safe_name(name, sep='-'):
    """One filesystem-safe path component. Slashes become `sep` rather than splitting the path
    (a title like 'RankMe / alpha' used to silently create a parent dir with a trailing space
    and a child with a leading one), and leading/trailing space is stripped. Interior spaces
    are left alone."""
    name = re.sub(r'[\\/]+', sep, str(name)).strip(' .' + sep)
    return name or 'unnamed'


def grid_start(ncols=2, figsize=None, title=None, savefig=True, savedir=None, savegroup=None, ylim=None, sharey=False, sharex=False, all_xlabels=False, pad=None, **kwargs):
    """sharex/sharey take plt.subplots' values: True (whole grid), 'row', 'col', False.
    all_xlabels=True keeps the x tick labels/label on every panel despite sharex.
    pad: extra tight_layout padding between panels, a number or (w_pad, h_pad). Applies to the
    assembled grid ONLY — the standalone per-panel PDFs lay themselves out separately.
    savedir names the output folder (default: the title); savegroup nests it under a parent, so
    a per-model sweep can share one directory without smuggling a '/' through the title."""
    global _grid
    _grid = dict(ncols=ncols, figsize=figsize, title=title, savefig=savefig, savedir=savedir, ylim=ylim, sharey=sharey, sharex=sharex, all_xlabels=all_xlabels, pad=pad, savegroup=savegroup, extra_kwargs=kwargs, calls=[])

def grid_show():
    global _grid
    calls = _grid['calls']
    ncols = _grid['ncols']
    nrows = (len(calls) + ncols - 1) // ncols
    figsize = _grid['figsize'] or (7 * ncols, 5 * nrows)
    title = _grid['title']
    savefig = _grid['savefig']
    savedir = _grid['savedir']
    savegroup = _grid['savegroup']
    ylim = _grid['ylim']
    sharey = _grid['sharey']
    sharex = _grid['sharex']
    all_xlabels = _grid['all_xlabels']
    pad = _grid['pad']
    extra_kwargs = _grid['extra_kwargs']
    _grid = None

    # native sharing also hides inner tick labels, matching plt.subplots(sharex/sharey)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    axes = np.array(axes).flatten()
    for i, (func, args, kwargs) in enumerate(calls):
        func(*args, _ax=axes[i], **(extra_kwargs | kwargs))
    used = axes[:len(calls)]
    if sharey is True and ylim is None:               # share one comparable y-range across the grid
        lims = [ax.get_ylim() for ax in used]         # ('row'/'col' sharing is matplotlib's own job)
        ylim = (min(l[0] for l in lims), max(l[1] for l in lims))
    if ylim:
        for ax in used: ax.set_ylim(*ylim)
    if all_xlabels:                                   # undo sharex's hiding of the inner tick labels
        for ax in used:
            ax.tick_params(labelbottom=True)
            ax.xaxis.get_label().set_visible(True)
    for ax in axes[len(calls):]:
        ax.set_visible(False)
    pad = (pad, pad) if np.isscalar(pad) else pad
    plt.tight_layout(**({} if pad is None else dict(w_pad=pad[0], h_pad=pad[1])))
    if title is not None:
        plt.suptitle(title, y=0.998, va='top')          # pinned to the top edge, clear of panel titles
        plt.subplots_adjust(top= 0.984 - (0.10 / nrows))
    plt.show()

    _export_transcription(title, calls, extra_kwargs)

    if savefig:
        outdir = Path('analysis/figures')
        if savegroup:
            outdir = outdir / _safe_name(savegroup)
        outdir = outdir / _safe_name(savedir or title or 'unnamed')
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / '_grid.pdf', bbox_inches='tight')   # the assembled panel, as shown
        # Standalone panels re-render at the size the panel actually occupies in the grid, not
        # a fixed 7x5. For a default grid the two are identical (figsize defaults to
        # 7*ncols x 5*nrows); they diverge only where a figure asks for its own proportions,
        # and there a fixed box squashes them — the tall band/spectrum maps were coming out
        # 4x compressed vertically against how they render on screen.
        panel = (figsize[0] / ncols, figsize[1] / nrows)
        for i, (func, args, kwargs) in enumerate(calls):
            subplot_title = (extra_kwargs | kwargs).get('title')
            safe_name = f'{i}_' + (_safe_name(subplot_title, '_').replace(' ', '_')
                                   if subplot_title else 'subplot')
            sf, _ = plt.subplots(figsize=panel)
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
    twin: int | None = None, twin_ylabel: str | None = None,
    src_colors: dict | None = None, src_ls: dict | None = None, src_lw: dict | None = None,
    key_didx: list = [],
    color_by: str = 'model', panel_bg=None, fs: float = 14,
    **kwargs
):
    """twin: index from which data sources move to a right-hand y-axis (for series whose
    scales don't share an axis). twin_ylabel names it; default is the first twinned
    source's own yvar label.

    src_colors / src_ls / src_lw: {source index: colour}, {source index: linestyle} and
    {source index: linewidth}, overriding the palette for named series — e.g. to pull one
    curve out of a gradient that is too pale on white, or to make it thick and dotted so a
    reference series reads as one.
    key_didx: source indices kept out of the depth colorbar and given a small legend instead
    (the reference curves pulled out of the gradient by the overrides above).
    color_by: 'model' gives each model its own hue (the default everywhere else); 'source'
    colours by data source only, so the same source is the same colour in every model's panel.
    panel_bg: seaborn-style tinted panel with white gridlines. True uses PANEL_BG, or pass a
    colour. Passing it to grid_start applies it to every panel in that grid."""
    palettes = panel_palettes(len(data_sources), color_palette, color_kwargs)
    src_colors, src_ls, src_lw = src_colors or {}, src_ls or {}, src_lw or {}

    if hold_color_for_n is None:
        hold_color_for_n = 2 if ls_exceptions else 1
    yvarname = yvar if isinstance(yvar, str) else yvar[0]

    ax_l = plt.gca()
    if panel_bg:
        ax_l.set_facecolor(PANEL_BG if panel_bg is True else panel_bg)
        ax_l.grid(True, which='major', color='white', lw=1.3)
        ax_l.set_axisbelow(True)
    ax_r = ax_l.twinx() if twin is not None else None
    if ax_r is not None:
        ax_r.grid(False)                                 # one gridline set, owned by the left axis

    label_source = not aggregate and (len(data_sources) > 1 or len(model_names) <= 1)
    label_model = not aggregate and len(model_names) > 1
    key_labels = set()

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
                midx_c = 0 if color_by == 'source' else model_name_options.index(model_name)
                color = palettes[didx][midx_c]
            color = src_colors.get(didx, color)
            if ax_r is not None: plt.sca(ax_r if didx >= twin else ax_l)
            if didx in key_didx:
                key_labels.add(make_label(model_name, data_source[2], label_model, label_source))
            r = plot_model_training(model_name, data_source,
                                xvar=xvar, yvar=yvar, color=color, transform=norm,
                                lw=src_lw.get(didx, contrib_lw if aggregate else lw),   # thinner contributions
                                alpha=contrib_alpha if aggregate else 1.0,    # colored but low-opacity
                                zorder=1 if aggregate else None,              # contributions at the bottom
                                label_source=label_source,
                                label_model=label_model,
                                ls_exceptions=ls_exceptions,
                                **({'ls': src_ls[didx]} if didx in src_ls else {}), **kwargs)
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

    if ax_r is not None:
        if ylog: ax_r.set_yscale('log')
        src = data_sources[twin][3] if len(data_sources[twin]) > 3 else yvar
        src = src if isinstance(src, str) else src[0]
        ax_r.set_ylabel(twin_ylabel or YVAR_LABELS.get(src, src), fontsize=fs)
        rlines = ax_r.get_lines()
        if rlines:                                       # tie the right axis to its series by color
            rc = rlines[0].get_color()
            ax_r.tick_params(axis='y', colors=rc)
            ax_r.yaxis.label.set_color(rc)
            ax_r.spines['right'].set_color(rc)
            ax_l.tick_params(axis='y', colors=ax_l.get_lines()[0].get_color())
            ax_l.yaxis.label.set_color(ax_l.get_lines()[0].get_color())
            ax_l.spines['left'].set_color(ax_l.get_lines()[0].get_color())
        plt.sca(ax_l)

    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    if title is not None: plt.title(title, fontsize=fs + 1)
    plt.xlabel(XVAR_LABELS[xvar], fontsize=fs)
    plt.ylabel(YVAR_LABELS.get(yvarname, yvarname) + (' (rel.)' if normalize else ''), fontsize=fs)
    ax_l.tick_params(labelsize=fs - 3)
    if ax_r is not None:
        ax_r.tick_params(labelsize=fs - 3)
        ax_r.yaxis.label.set_fontsize(fs)
    # x autoscales to the data: a fixed token range cut off early pythia
    # checkpoints (~2e6 tokens) and half of nanochat entirely
    if ax_r is None:
        legend_or_colorbar(legend_max, key_labels)
    else:                                                # one legend for both axes' series
        hl = [h + r for h, r in zip(ax_l.get_legend_handles_labels(), ax_r.get_legend_handles_labels())]
        ax_l.legend(*hl)
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
                            normalize=True, sep_lw=0.0, sep_color='white',
                            baseline_src=None, baseline_ls='--', baseline_delta=False,
                            total_src=None, total_lw=2.0, remainder_label=None, alpha=1.0):
    """Depth-stacked contributions over training. normalize=True: 100%-shares (nonneg yvar,
    e.g. uncentered trace = gross energy). normalize=False: raw signed values — positives
    stack up, negatives down (e.g. the ledger's delta_s, which includes overlap/cross terms).
    baseline_src (cfg, hook, label[, yvar]): gray line in baseline_ls (dashed by default,
    ':' for the dotted S(embedding stream) convention); normalized mode also adds it
    to the denominator and draws its share, signed mode draws it raw (the optional 4th
    element overrides yvar, e.g. matrix_entropy under a delta_s stack). baseline_delta:
    signed mode only — draw the baseline as its change from the first plotted step (0 at
    t0) so a large-offset curve like S_emb sits on the stack's scale. total_src (same tuple
    shape, signed mode only): black line of that series MINUS the baseline — the measured
    depth differential (S_bfn − S_emb) the signed stack should sum to, i.e. the ledger
    plot's black ΔS total read straight off the endpoints. remainder_label:
    draw 1 − Σshares as a gray dashed line (share metrics whose gap to 1 is the embedding)."""
    palettes = make_color_palettes(base_colors, len(sources), method=color_palette, **color_kwargs)
    W, xs = [], None
    for src in sources:
        ys, step_nums = get_ys(src[0], model, src[1], yvar)
        if ys is None: continue
        W.append(ys); xs = XVAR_FNS[xvar](model, step_nums)
    W, xs = np.asarray(W, float), np.asarray(xs, float)
    keep = xs > 0                                        # step 0 is off a log axis anyway
    W, xs = W[:, keep], xs[keep]
    base = None
    if baseline_src is not None:
        byvar = baseline_src[3] if len(baseline_src) > 3 else yvar
        bys, _bsteps = get_ys(baseline_src[0], model, baseline_src[1], byvar)
        base = np.asarray(bys, float)[keep] if bys is not None else None
    den = W.sum(0) + (base if base is not None else 0.0)
    frac = W / den if normalize else W
    colors = [palettes[i][0] for i in range(len(W))]            # gradient is model-independent -> [0]
    stack = lambda Y: plt.stackplot(xs, *Y, colors=colors, alpha=alpha, linewidth=sep_lw,
                                    edgecolor=sep_color if sep_lw else 'none')
    if normalize or (frac >= 0).all():
        stack(frac)
        plt.ylim(0, 1) if normalize else None
    else:                                                       # signed: positives up, negatives down
        stack(np.clip(frac, 0, None)); stack(np.clip(frac, None, 0))
        plt.axhline(0, color='0.3', lw=0.8)
    if base is not None:
        line = base / den if normalize else base - (base[0] if baseline_delta else 0.0)
        plt.plot(xs, line, color='0.25', lw=1.8, ls=baseline_ls, label=baseline_src[2])
        plt.legend(fontsize=7, loc='upper right')
    if total_src is not None and not normalize:          # measured depth differential
        tyvar = total_src[3] if len(total_src) > 3 else yvar
        tys, _tsteps = get_ys(total_src[0], model, total_src[1], tyvar)
        if tys is not None:
            tot = np.asarray(tys, float)[keep] - (base if base is not None else 0.0)
            plt.plot(xs, tot, color='k', lw=total_lw, label=total_src[2])
            plt.legend(fontsize=7, loc='upper right')
    if remainder_label is not None:
        plt.plot(xs, 1 - W.sum(0), color='0.25', lw=1.8, ls='--', label=remainder_label)
        plt.legend(fontsize=7, loc='upper right')
    plt.xscale('log')
    if title: plt.title(title)
    plt.xlabel(XVAR_LABELS[xvar], fontsize=14)
    plt.ylabel('Share of residual energy' if normalize else YVAR_LABELS.get(yvar, yvar), fontsize=14)
    if key:                                                     # depth colour key (one bar, bottom->final)
        from matplotlib.colors import ListedColormap
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=ListedColormap(colors)); sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), pad=0.01, ticks=[]).set_label('depth (0 → final)')
    plt.show()


def signed_stack(xs, series, alpha=0.55):
    """Sign-aware stack of `series` = [(label, values, color), ...]: positives stack up from
    zero, negatives down. The x-grid is refined with each series' zero crossings (interpolated
    in log-x, where the drawn segments are straight) so each fill closes vertically at zero
    instead of drawing a twisted quadrilateral across the opposite stack when a term changes
    sign. Returns (refined x, the summed series on it) for drawing a total on top."""
    lx = np.log(xs)
    cross = []
    for _, v, _ in series:
        i = np.nonzero(np.signbit(v[:-1]) != np.signbit(v[1:]))[0]
        cross.append(lx[i] + v[i] / (v[i] - v[i + 1]) * (lx[i + 1] - lx[i]))
    grid = np.unique(np.concatenate([lx, *cross]))
    gx, pos, neg = np.exp(grid), np.zeros(len(grid)), np.zeros(len(grid))
    for name, v, c in series:
        w = np.interp(grid, lx, v)
        up, dn = np.clip(w, 0, None), np.clip(w, None, 0)
        plt.fill_between(gx, pos, pos + up, label=name, color=c, alpha=alpha, lw=0)
        plt.fill_between(gx, neg, neg + dn, color=c, alpha=alpha, lw=0)
        pos, neg = pos + up, neg + dn
    return gx, pos + neg


# The showcase §3 "Ledger contributions" stack: the three ledger terms summed over blocks,
# sign-stacked over training (positives up from zero, negatives down); black = ΔS total.
@griddable
def plot_ledger_stack(model, source, n_blocks=None, blocks=None, xvar='tokens', title=None,
                      ylabel='rank entropy', legend=8, emb=True, zero_line=False,
                      total_lw=2.5, emb_lw=1.5):
    """One model's ledger stack from results dir `source`, summed over `blocks` (an explicit
    list of block indices) or the first `n_blocks`. The x-grid is refined with each
    term's sign crossings (interpolated in log-x, where the drawn segments are straight) so
    each fill closes vertically at zero instead of drawing a twisted quadrilateral across
    the opposite stack when a term changes sign. emb: dotted gray S(embedding stream), the
    base of the black depth-differential (only meaningful for a whole-depth stack). legend:
    fontsize, or None to skip (grids with a shared legend put it on one panel only). A
    missing source turns the panel off."""
    blocks = list(range(n_blocks)) if blocks is None else list(blocks)
    probe, steps = get_ys(source, model, (f'blk{blocks[0]}', 'block_ledger'), 'chi')
    if probe is None:
        plt.gca().axis('off')
        return
    terms = {y: np.sum([get_ys(source, model, (f'blk{l}', 'block_ledger'), y)[0]
                        for l in blocks], axis=0)
             for y in ('chi', 'quality', 'interference')}
    xs = np.asarray(XVAR_FNS[xvar](model, steps), float)
    keep = xs > 0                                        # step 0 is off a log axis anyway
    xs, terms = xs[keep], {k: np.asarray(v, float)[keep] for k, v in terms.items()}
    gx, tot = signed_stack(xs, [(n, terms[n], c) for n, c in
                                (('chi', 'tab:green'), ('quality', 'tab:red'),
                                 ('interference', 'tab:purple'))])
    plt.plot(gx, tot, 'k', lw=total_lw, label='ΔS total')
    if zero_line:                                        # the pivot the signed fills close on
        plt.axhline(0, color='gray', lw=0.8)
    if emb:
        es, esteps = get_ys(source, model, ('blk0.attn.in', 'acts_centered'), 'matrix_entropy')
        if es is not None:
            exs = np.asarray(XVAR_FNS[xvar](model, esteps), float)
            ek = exs > 0
            plt.plot(exs[ek], np.asarray(es, float)[ek], color='0.4', ls=':', lw=emb_lw,
                     label='S(embedding stream)')
    plt.xscale('log')
    plt.xlabel({'steps': 'step'}.get(xvar, xvar))
    if ylabel:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    if legend:
        plt.legend(fontsize=legend)
    plt.show()


# --- Attention vs MLP: the coarsest split of what the blocks write --------------------
# Energy (the uncentered write trace) exists for every model. Sub-step ΔS does NOT: it needs
# a stream BETWEEN the two writes, which only sequential families have (attn writes, then mlp
# reads the updated stream). Pythia's sub-blocks are parallel — both read the same block
# input — so its attn sub-step ΔS is identically zero and only the block total is attributable.
SUB_C = {'attn': 'tab:blue', 'mlp': 'tab:orange'}
SUB_LS = {'attn': ':', 'mlp': '-'}      # depth-coloured figures: colour is the block, ls the sub-step

def _sub_sum(source, model, n_blocks, sub, quantity, yvar):
    """Σ over blocks of one per-write series, as (tokens, values)."""
    ys = [get_ys(source, model, (f'blk{l}.{sub}.out', quantity), yvar)[0] for l in range(n_blocks)]
    steps = get_ys(source, model, (f'blk0.{sub}.out', quantity), yvar)[1]
    return np.asarray(get_xs_tokens(model, steps), float), np.sum(np.asarray(ys, float), 0)

def sub_energy(source, model, n_blocks):
    """(tokens, attn, mlp) summed uncentered write trace; step 0 and dead checkpoints dropped."""
    xs, a = _sub_sum(source, model, n_blocks, 'attn', 'acts_uncentered', 'trace')
    _,  m = _sub_sum(source, model, n_blocks, 'mlp',  'acts_uncentered', 'trace')
    keep = (xs > 0) & (a + m > 0)                        # nanochat's first ckpts write nothing
    return xs[keep], a[keep], m[keep]

def sub_ds(source, model, n_blocks):
    """(tokens, ΔS attn, ΔS mlp) summed sub-step rank entropy — sequential families only;
    the two sum to the block-summed ΔS exactly."""
    xs, a = _sub_sum(source, model, n_blocks, 'attn', 'incremental_overlap', 'sub_delta_s')
    _,  m = _sub_sum(source, model, n_blocks, 'mlp',  'incremental_overlap', 'sub_delta_s')
    keep = xs > 0
    return xs[keep], a[keep], m[keep]

def _block_ds_raw(source, model, n_blocks):
    ys = [get_ys(source, model, (f'blk{l}', 'block_ledger'), 'delta_s')[0] for l in range(n_blocks)]
    steps = get_ys(source, model, ('blk0', 'block_ledger'), 'delta_s')[1]
    return np.asarray(get_xs_tokens(model, steps), float), np.sum(np.asarray(ys, float), 0)

def block_ds(source, model, n_blocks):
    """(tokens, Σ_blocks ΔS) from the block ledger — defined for parallel sub-blocks too."""
    xs, tot = _block_ds_raw(source, model, n_blocks)
    keep = xs > 0
    return xs[keep], tot[keep]

def sub_ds_pretend(source, model, n_blocks, first='attn'):
    """(tokens, ΔS attn, ΔS mlp) under a PRETENDED write order — the counterfactual that lets
    a parallel family (Pythia) be drawn on the same axes as a sequential one. It is not a
    measurement: Pythia has no stream between its two writes, so no order is the true one.

    The write named `first` is charged its own 2-component ledger against the block input,
    χ + quality = S(mix) − S(in), which is everything attributable before the second write
    exists; its interference term is NOT available, since that would need S(in + first) and
    only sequential models ever form that tensor. The second write takes the residue, so the
    two still sum to the measured block ΔS exactly and the black total stays real.

    Where the truth exists (sequential families) this UNDERSTATES nothing and OVERSTATES the
    first write by the interference it drops — badly for OLMo-2, whose ledger is
    interference-carried. Compare against sub_ds before reading anything into it."""
    second = 'mlp' if first == 'attn' else 'attn'
    xs, chi = _sub_sum(source, model, n_blocks, first, 'incremental_overlap', 'chi')
    _,  qua = _sub_sum(source, model, n_blocks, first, 'incremental_overlap', 'sub_quality')
    xb, tot = _block_ds_raw(source, model, n_blocks)
    if len(xb) != len(xs):
        raise ValueError(f'{model}: block_ledger and incremental_overlap step lists differ')
    keep = xs > 0
    parts = {first: (chi + qua)[keep], second: (tot - chi - qua)[keep]}
    return xs[keep], parts['attn'], parts['mlp']


@griddable
def plot_sub_energy(model, source, n_blocks, share=False, title=None):
    """attn vs mlp write energy over training: share=True gives the 100% two-way stack,
    share=False the raw log-log lines (the level the share hides)."""
    xs, a, m = sub_energy(source, model, n_blocks)
    if share:
        plt.stackplot(xs, a / (a + m), m / (a + m), colors=[SUB_C['attn'], SUB_C['mlp']],
                      labels=['attn', 'mlp'], alpha=0.8)
        plt.axhline(0.5, color='0.2', lw=0.9, ls=':')
        plt.xlim(xs[0], xs[-1]); plt.ylim(0, 1)
        plt.ylabel('Share of write energy', fontsize=14)
    else:
        for lbl, v in (('attn', a), ('mlp', m)):
            plt.plot(xs, v, color=SUB_C[lbl], lw=2.5, label=lbl)
        plt.yscale('log')
        plt.ylabel(r'$\sum_{\ell=1}^{L}\mathrm{tr}\,\Sigma_\ell$', fontsize=14)
    plt.xscale('log')
    plt.xlabel(XVAR_LABELS['tokens'], fontsize=14)
    if title: plt.title(title)
    plt.legend(fontsize=8, loc='upper right' if share else 'best')
    plt.show()


@griddable
def plot_sub_ds(model, source, n_blocks, first=None, truth=False, title=None):
    """attn vs mlp sub-step ΔS, sign-stacked over training; black = their sum (= the block
    ΔS the ledger stack decomposes into χ/quality/interference). first=None: the measured
    sequential sub-steps (sequential families only). first='attn'|'mlp': sub_ds_pretend's
    counterfactual order, which every model has. truth: also draw the MEASURED attn sub-step
    dashed wherever one exists, so the counterfactual can be read against it."""
    xs, a, m = (sub_ds(source, model, n_blocks) if first is None else
                sub_ds_pretend(source, model, n_blocks, first))
    signed_stack(xs, [('attn', a, SUB_C['attn']), ('mlp', m, SUB_C['mlp'])], alpha=0.6)
    plt.plot(xs, a + m, 'k', lw=2.5, label='ΔS total')
    if truth:
        tx, ta, _ = sub_ds(source, model, n_blocks)
        if np.any(ta != 0):                              # parallel: no measured sub-step at all
            plt.plot(tx, ta, color=SUB_C['attn'], ls='--', lw=1.8, label='attn, measured')
    plt.axhline(0, color='gray', lw=0.8)
    plt.xscale('log'); plt.xlim(xs[0], xs[-1])
    plt.xlabel(XVAR_LABELS['tokens'], fontsize=14)
    plt.ylabel(r'$\sum_{\ell=1}^{L}\Delta S_\ell$', fontsize=14)
    if title: plt.title(title)
    plt.legend(fontsize=8, loc='upper right')
    plt.show()


@griddable
def plot_sub_ds_lines(models, source_fn, n_blocks, title=None,
                      colors=('tab:green', 'tab:purple', 'tab:brown')):
    """The same sub-step ΔS series as plot_sub_ds, several models overlaid as lines
    (solid = attn, dashed = mlp) so the families can be compared directly."""
    for model, c in zip(models, colors):
        xs, a, m = sub_ds(source_fn(model), model, n_blocks[model])
        plt.plot(xs, a, color=c, lw=2, label=f'{get_model_label(model)} — attn')
        plt.plot(xs, m, color=c, lw=2, ls='--', label=f'{get_model_label(model)} — mlp')
    plt.axhline(0, color='gray', lw=0.8)
    plt.xscale('log')
    plt.xlabel(XVAR_LABELS['tokens'], fontsize=14)
    plt.ylabel(r'$\sum_{\ell=1}^{L}\Delta S_\ell$', fontsize=14)
    if title: plt.title(title)
    plt.legend(fontsize=7, loc='upper left')
    plt.show()


@griddable
def plot_sub_ds_depth(model, source, blocks, subs=('attn', 'mlp'), lines=True, stack=False,
                      total=True, emb=True, ref_label=None, xvar='tokens', title=None,
                      color_palette='gradient', color_kwargs={}, lw=2.0, total_lw=2.5,
                      ref_lw=3.5, alpha=0.55, fs=14, legend=8):
    """Sub-step ΔS per individual write, one line per (block, sub) — sequential families only.

    Colour is the block (gradient, keyed by the depth colorbar) and linestyle the sub-step
    (SUB_LS: attn dotted, mlp solid), so a block's two writes share a colour. lines=False
    drops the per-write lines and leaves the aggregate. stack: sign-stacked bands instead of
    lines (positives up, negatives down), one band per write in depth order, so a block's two
    writes are adjacent in the stack and read as one coloured region.
    total: black Σ over every write drawn. emb: thick dotted S of the stream the drawn writes
    are added into — the input to the first block drawn, i.e. the embedding output when that
    is block 0 — as its change from the first plotted checkpoint, its absolute level sitting
    several nats above the ΔS scale."""
    blocks = list(blocks)
    palette = make_color_palettes(base_colors, len(blocks), method=color_palette, **color_kwargs)
    xs, tot, drawn = None, None, []
    for bi, l in enumerate(blocks):
        for sub in subs:
            ys, steps = get_ys(source, model, (f'blk{l}.{sub}.out', 'incremental_overlap'),
                               'sub_delta_s')
            if ys is None: continue
            x = np.asarray(XVAR_FNS[xvar](model, steps), float)
            keep = x > 0                                 # step 0 is off a log axis anyway
            x, y = x[keep], np.asarray(ys, float)[keep]
            xs, tot = x, y if tot is None else tot + y
            drawn.append((f'{sub} {l}', y, palette[bi][0], SUB_LS[sub]))
    if tot is None:                                      # missing source: turn the panel off
        plt.gca().axis('off')
        return
    if stack:                                            # depth order: a block's two writes adjacent
        xs, tot = signed_stack(xs, [(n, y, c) for n, y, c, _ in drawn], alpha=alpha)
    elif lines:
        for n, y, c, ls in drawn:
            plt.plot(xs, y, color=c, ls=ls, lw=lw, label=n)
    if total:
        plt.plot(xs, tot, 'k', lw=total_lw, label=r'$\sum \Delta S$')
    ref_label = ref_label or ('embedding' if blocks[0] == 0 else f'into blk{blocks[0]}')
    if emb:
        es, esteps = get_ys(source, model, (f'blk{blocks[0]}.attn.in', 'acts_centered'),
                            'matrix_entropy')
        if es is not None:
            ex = np.asarray(XVAR_FNS[xvar](model, esteps), float)
            ek = ex > 0
            ey = np.asarray(es, float)[ek]
            plt.plot(ex[ek], ey - ey[0], color='0.35', ls=':', lw=ref_lw,
                     label=f'{ref_label}, Δ vs first ckpt')
    plt.axhline(0, color='gray', lw=0.8)
    plt.xscale('log')
    if stack: plt.xlim(xs[0], xs[-1])                    # the fills own the full width
    plt.xlabel(XVAR_LABELS[xvar], fontsize=fs)
    plt.ylabel(YVAR_LABELS['sub_delta_s'], fontsize=fs)
    if title: plt.title(title, fontsize=fs + 1)
    keys = []
    if stack or lines:
        depth_colorbar([palette[i][0] for i in range(len(blocks))],
                       [str(l) for l in blocks], label='block')
        if lines and not stack and len(subs) > 1:        # no linestyles to key in a stack
            keys += [Line2D([], [], color='0.3', ls=SUB_LS[s], lw=lw, label=s) for s in subs]
    if total:
        keys.append(Line2D([], [], color='k', lw=total_lw, label=r'$\sum \Delta S$'))
    if emb:
        keys.append(Line2D([], [], color='0.35', ls=':', lw=ref_lw, label=ref_label))
    if legend and keys:
        plt.legend(handles=keys, fontsize=legend)
    plt.show()


def plot_sub_bars(models, source_fn, n_blocks, seq=(), title=None):
    """Both currencies at the final checkpoint, one row per model: the 100% energy split
    (left) and the sub-step ΔS split (right). Models outside `seq` are parallel — their ΔS
    bar is the unsplittable block total, hatched."""
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y, labels = np.arange(len(models)), [get_model_label(m) for m in models]

    E = np.array([[a[-1], m[-1]] for a, m in
                  (sub_energy(source_fn(k), k, n_blocks[k])[1:] for k in models)])
    sh = E / E.sum(1, keepdims=True)
    axes[0].barh(y, sh[:, 0], color=SUB_C['attn'])
    axes[0].barh(y, sh[:, 1], left=sh[:, 0], color=SUB_C['mlp'])
    for i, (sa, sm) in enumerate(sh):
        axes[0].text(sa / 2, i, f'{sa:.0%}', ha='center', va='center', color='w', fontsize=9)
        axes[0].text(sa + sm / 2, i, f'{sm:.0%}', ha='center', va='center', color='w', fontsize=9)
    axes[0].axvline(0.5, color='0.2', lw=0.9, ls=':')
    axes[0].set(yticks=y, yticklabels=labels, xlim=(0, 1), xlabel='Share of write energy',
                title='write energy')

    h = 0.38
    for i, model in enumerate(models):
        src = source_fn(model)
        if model in seq:
            _, a, m = sub_ds(src, model, n_blocks[model])
            axes[1].barh(i - h / 2, a[-1], height=h, color=SUB_C['attn'])
            axes[1].barh(i + h / 2, m[-1], height=h, color=SUB_C['mlp'])
        else:                                            # parallel: block total, unsplittable
            axes[1].barh(i, block_ds(src, model, n_blocks[model])[1][-1], height=2 * h,
                         color='0.75', hatch='//', edgecolor='0.45')
    axes[1].axvline(0, color='0.2', lw=0.9)
    axes[1].set(yticks=y, yticklabels=labels, title='rank entropy',
                xlabel=r'$\sum_{\ell=1}^{L}\Delta S_\ell$')

    for ax in axes: ax.invert_yaxis()
    fig.legend(handles=[Patch(color=SUB_C['attn'], label='attn'),
                        Patch(color=SUB_C['mlp'], label='mlp'),
                        Patch(facecolor='0.75', hatch='//', edgecolor='0.45',
                              label='block (parallel: no split)')],
               fontsize=9, ncol=3, loc='lower center', frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title or 'Attention vs MLP — final checkpoint')
    plt.tight_layout(); plt.show()


# --- Piecewise-α spectrum map ---------------------------------------------------------
# The log-log eigenspectrum chopped into geometric rank bands, alphaReQ fitted inside each,
# one row per checkpoint: where in the spectrum the slope steepens, and when.
AB_HOOK = ('after_final_norm', 'acts_centered')
# compute_metrics fits the headline α with stringer_get_powerlaw over arange(11, 100). Marked
# on both map families as dotted verticals: the slice of spectrum that one number actually sees.
ALPHA_FIT_WINDOW = (11, 100)

def _mark_lines(ax, values, axis, color, **kw):
    """Guide lines on one axis; limits are restored so an off-range value can't stretch it."""
    if values is None or not len(values):
        return
    lim = ax.get_xlim() if axis == 'x' else ax.get_ylim()
    for v in values:
        (ax.axvline if axis == 'x' else ax.axhline)(v, color=color, zorder=5,
                                                    **({'ls': ':', 'lw': 1.3} | kw))
    (ax.set_xlim if axis == 'x' else ax.set_ylim)(lim)

def map_tokens(model, steps, include_init=True):
    """(tokens, keep-mask) for a map's rows. A step-0 checkpoint carries real data but 0
    tokens, and 0 has no place on a log axis, so it was being dropped — 5 of our 6 models have
    one. That hurts most at OLMo-2 0425-1B, whose step 0 is the ONLY row above the
    initialisation crash (RankMe 823 against 68.9 at the next checkpoint), so dropping it made
    the map open at the bottom of a crash it never showed. include_init places it one
    geometric step below the first real checkpoint. That position is a DISPLAY CONVENTION, not
    a token count — there is no honest log-axis coordinate for zero. OLMo-2 1124-7B has no
    step 0 at all in this run, so nothing changes there."""
    xs = np.asarray(get_xs_tokens(model, steps), float)
    keep = xs > 0
    if include_init and len(keep) > 2 and not keep[0] and keep[1:].sum() >= 2:
        pos = np.nonzero(keep)[0]
        xs[~keep] = xs[pos[0]] ** 2 / xs[pos[1]]
        keep = xs > 0
    return xs, keep

def alpha_band_edges(source, model, hook=AB_HOOK, n_bands=20, k0=2, top_frac=0.75,
                     ratio=6.0, min_width=2):
    """Integer rank edges of bands running k0 → top_frac·d, with band width in LOG-RANK
    shrinking left to right by a factor of `ratio`. top_frac drops the tail, where the
    spectrum falls off a cliff and a band slope stops meaning anything.

    n_bands is the TOTAL number of bands drawn, and ratio is how many times narrower the
    rightmost band is than the leftmost. ratio=1 is a plain uniform split, so (10, 1) is
    exactly ten equal log-rank bands. The leftmost band comes out (ratio−1)/ln(ratio) times
    wider than a uniform n_bands split would give — at 20 bands and 6× that is a head band
    the width of a uniform 7-band split, and a tail band 1/6 of it.

    Why the warp: a uniform split fine enough to resolve the tail makes its head bands 1–2
    ranks wide, and a 1-rank band makes the two-parameter fit singular; shrinking toward the
    tail buys tail resolution without paying for it at the head. min_width is the backstop
    floor for the head bands that remain narrow.

    Do NOT be tempted to widen the head bands much beyond ~2 ranks to smooth them. Measured on
    the pythias, the end-of-training head steepening is a narrow feature around ranks 4–8:
    band [4,6) reads α ≈ 1.9 over pythia-1b's last ten checkpoints, and widening it to 4 ranks
    drops that to 0.74 — the feature is averaged away, not smoothed. Nor is it noise. Across
    those last ten checkpoints α moves by 0.16 (1b) and 0.05 (6.9b), while the same band
    jitters 2–7× more BEFORE the RankMe peak. The speckle a narrow head band produces is
    therefore confined to early training, where α genuinely is unstable, and widening the head
    to remove it costs a real, stable signal elsewhere in the same column."""
    evs, _ = get_ys(source, model, hook, 'eigenspectrum')
    d = len(np.asarray(evs[-1]))
    n = int(n_bands)
    x = np.arange(n + 1) / n
    f = x if ratio == 1 else np.log1p((ratio - 1) * x) / np.log(ratio)
    lo, hi = np.log(k0), np.log(top_frac * d)
    e = np.round(np.exp(lo + f * (hi - lo))).astype(int)
    for i in range(1, len(e)):
        e[i] = max(e[i], e[i - 1] + min_width)
    return e

def alpha_bands(source, model, hook=AB_HOOK, n_bands=20, k0=2, top_frac=0.75, ratio=6.0,
                include_init=True):
    """(tokens, rank edges, α) with α of shape (checkpoints, bands) — alphaReQ fitted inside
    each band per checkpoint, via the same alpha_window hook the §2 band figures use.
    Rows follow map_tokens, so row i matches spectrum_map and rankme_series row i."""
    e = alpha_band_edges(source, model, hook, n_bands, k0, top_frac, ratio)
    cols, steps = [], None
    for lo, hi in zip(e[:-1], e[1:]):
        ys, steps = get_ys(source, model, hook, 'alpha_window', {'k0': int(lo), 'k1': int(hi)})
        cols.append(np.asarray(ys, float))
    xs, keep = map_tokens(model, steps, include_init)
    return xs[keep], e, np.stack(cols, 1)[keep]

def rankme_series(source, model, hook=AB_HOOK, include_init=True):
    """(tokens, RankMe) on the same token grid the maps use (see map_tokens), so row i here is
    row i of alpha_bands / spectrum_map."""
    ys, steps = get_ys(source, model, hook, 'rankme')
    xs, keep = map_tokens(model, steps, include_init)
    return xs[keep], np.asarray(ys, float)[keep]

def rankme_phases(source, model, hook=AB_HOOK, low_frac=0.10, include_init=True):
    """(trough index, peak index). Trough = middle (geometric in tokens) of the FIRST run of
    checkpoints with RankMe below low_frac x d_model — the flat bottom of the warmup crash,
    which is more stable than its argmin. Peak = the RankMe maximum after that run ends.
    Only the first run counts: nanochat-d12's late decline re-enters the same band."""
    xs, r = rankme_series(source, model, hook, include_init)
    d = len(np.asarray(get_ys(source, model, hook, 'eigenspectrum')[0][-1]))
    low = np.nonzero(r < low_frac * d)[0]
    if not len(low):                                     # never that low: fall back to argmin
        lo = hi = int(np.argmin(r[:max(3, len(r) // 3)]))
    else:
        brk = np.nonzero(np.diff(low) > 1)[0]
        lo, hi = low[0], (low[brk[0]] if len(brk) else low[-1])
    mid = np.sqrt(xs[lo] * xs[hi])
    trough = int(np.argmin(np.abs(np.log(xs) - np.log(mid))))
    return trough, hi + int(np.argmax(r[hi:]))

def rankme_peak(source, model, hook=AB_HOOK, include_init=True):
    """Row index of the entropy-seeking peak — rankme_phases' second element. include_init must
    match the map's, or the index lands one row off."""
    return rankme_phases(source, model, hook, include_init=include_init)[1]

def map_steps(source, model, hook=AB_HOOK):
    """Checkpoint numbers on the maps' row grid (see map_tokens)."""
    steps = get_ys(source, model, hook, 'rankme')[1]
    _, keep = map_tokens(model, steps)
    return [s for s, k in zip(steps, keep) if k]

def phase_landmarks(source, model, hook=AB_HOOK, min_ratio=2.0, start=2):
    """[(label, row, step, tokens)] from the `start`-th checkpoint up to the RankMe peak,
    thinned onto a geometric ladder of ratio min_ratio. Rungs within min_ratio of the peak are
    dropped along with the peak itself, so every gap in the returned list is a real one.
    Snapping a ladder onto the real checkpoints beats greedily skipping: checkpoint sampling
    turns linear-in-steps partway through, and a greedy threshold leaves double-width gaps."""
    xs, _ = rankme_series(source, model, hook)
    steps = map_steps(source, model, hook)
    tr, pk = rankme_phases(source, model, hook)
    mid = lambda a, b: int(np.argmin(np.abs(np.log(xs) - (np.log(xs[a]) + np.log(xs[b])) / 2)))
    named = {0: 'init', 1: '2nd ckpt', 2: '3rd ckpt', mid(1, tr): 'pre-trough', tr: 'trough'}
    rows = np.arange(pk + 1)
    if min_ratio > 1:
        lx = np.log(xs[rows])
        rungs = np.arange(lx[0], lx[-1] + 1e-9, np.log(min_ratio))
        rows = np.unique([int(np.argmin(np.abs(lx - g))) for g in rungs])
    rows = [i for i in sorted(set(rows) | set(named))
            if start <= i and xs[i] * min_ratio <= xs[pk]]
    return [(named.get(i, ''), i, steps[i], xs[i]) for i in rows]

@griddable
def plot_landmark_spectra(model, source, hook=AB_HOOK, normalise=False, drop_tail=10,
                          keep=slice(None), min_ratio=2.0, cmap='viridis', highlight=None,
                          legend=True, title=None):
    """phase_landmarks spectra overlaid on one axis, coloured early→late. normalise divides by
    the trace; drop_tail cuts the cliff ranks; highlight is an index whose line keeps its
    colour while the rest go grey."""
    yvar = 'eigenspectrum' if normalise else 'eigvals'
    lm = phase_landmarks(source, model, hook, min_ratio)[keep]
    for i, ((_lbl, _row, step, tok), c) in enumerate(
            zip(lm, plt.get_cmap(cmap)(np.linspace(0, 0.92, len(lm))))):
        v = np.asarray(get_y(source, model, hook, yvar, step), float)
        v = v[:-drop_tail] if drop_tail else v
        on = highlight is None or i == highlight
        plt.plot(np.arange(1, len(v) + 1), v, color=c if on else '0.85',
                 lw=2.6 if on and highlight is not None else 2, zorder=3 if on else 1,
                 label=f'{tok:.2e} tokens' if on else '_nolegend_')
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Eigenvalue index', fontsize=14)
    plt.ylabel(YVAR_LABELS.get(yvar, yvar), fontsize=14)
    if title: plt.title(title)
    if legend: plt.legend(fontsize=8, title='checkpoint', title_fontsize=8)
    plt.show()


def phase_marks(source, model, hook=AB_HOOK):
    """Token positions of the warmup trough and the RankMe peak, for marking on a token axis."""
    xs, _ = rankme_series(source, model, hook)
    return [xs[i] for i in rankme_phases(source, model, hook)]

def _row_index(n, upto, rows, model):
    """The row positions a map keeps, as an index array — so a parallel series (the RankMe
    strip) can be sliced identically instead of re-deriving the same two slices."""
    idx = np.arange(n)
    if upto is not None:
        idx = idx[:upto + 1]
    idx = idx[rows]
    if len(idx) < 2:
        raise ValueError(f'{model}: {len(idx)} row(s) left after upto/rows — need at least 2')
    return idx

FIT_C, FIT_BG = 'tab:blue', '0.80'   # α fit-window bracket, on a full-width grey rail
STRIP_BG = '#F4F4F7'                 # seaborn-ish tint behind the RankMe strip

def _mark_fit_window(ax, window, orient, style, fs):
    """The default α fit window: 'lines' rules it across the map, 'bar' sets a bracket just
    outside the axes instead, so nothing is drawn over the data. None draws neither."""
    if not window or not style:
        return
    down = orient == 'time_down'
    if style == 'lines':
        _mark_lines(ax, window, 'x' if down else 'y', '0.55', lw=1.95 * fs / 14)
        return
    if style != 'bar':
        raise ValueError(f"fit_style must be 'lines', 'bar' or None, got {style!r}")
    from matplotlib.transforms import blended_transform_factory as blend, offset_copy
    lw = 5.0 * fs / 14
    dat, axf = ax.transData, ax.transAxes
    # offset by half the line width in POINTS, so the rail's inner edge lands flush on the
    # axes edge whatever the panel size, instead of straddling it
    if down:
        tr = offset_copy(blend(dat, axf), fig=ax.figure, y=lw / 2, units='points')
        rail, at = (ax.get_xlim(), [1, 1]), (window, [1, 1])
    else:                        # rank runs down: rail on the left, with the rank tick labels
        tr = offset_copy(blend(axf, dat), fig=ax.figure,   # pushed out past it
                         x=-lw / 2, units='points')
        rail, at = ([0, 0], ax.get_ylim()), ([0, 0], window)
        ax.tick_params(axis='y', pad=lw + 6)
    ax.plot(*rail, transform=tr, color=FIT_BG, lw=lw, solid_capstyle='butt',
            clip_on=False, zorder=5)
    ax.plot(*at, transform=tr, color=FIT_C, lw=lw, solid_capstyle='butt',
            clip_on=False, zorder=6)


PHASE_C = 'cyan'             # off-palette for magma, so the phase marks never read as data
PHASE_KW = dict(ls='--', lw=1.4, alpha=0.7)

# Orientation: 'time_down' puts rank across and time down the panel; 'time_right' transposes
# it, time across and rank down. Both read like text in their own way — the first scans a
# spectrum left-to-right per checkpoint, the second scans a training run left-to-right per rank.
ORIENTS = ('time_down', 'time_right')
# Which physical side of the panel the head (rank 1) and the tail (rank d) end up on, per
# orientation — so `side` can be named by the spectrum rather than by the screen.
SIDE_POS = {('time_down', 'head'): 'left', ('time_down', 'tail'): 'right',
            ('time_right', 'head'): 'top', ('time_right', 'tail'): 'bottom'}

def _attach_side_rankme(div, ax, source, model, hook, idx, orient, side, fs=14,
                        include_init=True):
    """Narrow RankMe-against-tokens strip glued to the head or tail side of `ax`, sharing its
    token axis so the two read across at the same position — the scalar whose shape the map
    decomposes, next to the map. `side` names a side of the SPECTRUM ('head' = the rank-1 end,
    'tail' = the rank-d end), which survives transposing; SIDE_POS turns that into a screen
    edge. Appended before the colourbar, so the colourbar lands outside it.

    The phases are carried by the curve's own dash pattern — dotted to the warmup trough,
    solid to the RankMe peak, dashed after — rather than by rules drawn across the map, which
    read as data on a heatmap."""
    pos = SIDE_POS[(orient, side)]
    down = orient == 'time_down'
    sax = div.append_axes(pos, size='20%', pad=0.07,
                          **({'sharey': ax} if down else {'sharex': ax}))
    xs, r = rankme_series(source, model, hook, include_init)   # same row grid as the map,
    tr, pk = rankme_phases(source, model, hook, include_init=include_init)   # or idx is off
    idx = np.asarray(idx)
    for sel, ls in ((idx[idx <= tr], ':'), (idx[(idx >= tr) & (idx <= pk)], '-'),
                    (idx[idx >= pk], '--')):          # segments share endpoints, so they join
        if len(sel) > 1:
            sax.plot(*((r[sel], xs[sel]) if down else (xs[sel], r[sel])), color='0.25',
                     ls=ls, lw=2.1 * fs / 14)
    # anchor the RankMe axis at 0 so the ticks always include it — the range alone can start
    # at 400+ and leave a reader guessing where the baseline is
    sax.set_xlim(left=0) if down else sax.set_ylim(bottom=0)
    # set the locator directly: locator_params(nbins=...) does not reach AutoLocator here,
    # and three labels do not fit across a strip this narrow
    (sax.xaxis if down else sax.yaxis).set_major_locator(MaxNLocator(nbins=1))
    sax.tick_params(labelsize=fs - 8)   # 4-digit RankMe values in a narrow strip
    sax.set_facecolor(STRIP_BG)
    sax.grid(True, which='major', color='white', lw=1.2)
    sax.set_axisbelow(True)
    if down:
        sax.set_xlabel('RankMe', fontsize=fs - 2)
        if pos == 'left':            # the strip reads first: hand it the token ticks/label
            sax.set_ylabel(XVAR_LABELS['tokens'], fontsize=fs)
            ax.tick_params(labelleft=False); ax.set_ylabel('')
        else:
            sax.tick_params(labelleft=False)
    else:
        # midway between the map's bottom tick label and its ylabel: the default pad puts it
        # right against the 10^3, and much more lands it ON the ylabel and reads as one line
        sax.set_ylabel('RankMe', fontsize=fs - 2, labelpad=14)
        if pos == 'bottom':          # the strip is now the bottom axis: it takes the ticks
            sax.set_xlabel(XVAR_LABELS['tokens'], fontsize=fs)
            ax.tick_params(labelbottom=False); ax.set_xlabel('')
        else:
            sax.tick_params(labelbottom=False)
    return sax

def _render_map(ax, tok_edges, rank_edges, A, orient, mesh_kw, fit_window,
                side, source, model, hook, idx, cbar, cbar_label, extend, title, fs=14,
                include_init=True, fit_style='lines'):
    """Shared body of both map figures. Lays A out in the requested orientation (A is always
    rows=checkpoints × cols=rank, and gets transposed here, not by the caller), marks the α
    fit window on whichever axis carries rank and the phases on whichever carries tokens,
    attaches the RankMe strip on the head or tail side, and hangs the colourbar outside it."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if orient not in ORIENTS:
        raise ValueError(f'orient must be one of {ORIENTS}, got {orient!r}')
    down = orient == 'time_down'
    im = (ax.pcolormesh(rank_edges, tok_edges, A, **mesh_kw) if down else
          ax.pcolormesh(tok_edges, rank_edges, A.T, **mesh_kw))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.invert_yaxis()          # time_down: time reads down. time_right: rank reads down,
    ax.grid(False)             # head on top — either way the y axis runs the way you read.
    labels = ('Eigenvalue index', XVAR_LABELS['tokens'])
    ax.set_xlabel(labels[not down], fontsize=fs)
    ax.set_ylabel(labels[bool(down)], fontsize=fs)
    ax.tick_params(labelsize=fs - 3)
    _mark_fit_window(ax, fit_window, orient, fit_style, fs)
    div = make_axes_locatable(ax)
    sax = (_attach_side_rankme(div, ax, source, model, hook, idx, orient, side, fs,
                               include_init) if side else None)
    if title:                    # a strip on top covers the map's own title, so hand it over
        (sax if sax is not None and SIDE_POS[(orient, side)] == 'top'
         else ax).set_title(title, fontsize=fs + 2,
                            pad=fs * 1.3 if fit_style == 'bar' and down else None)
    if cbar:
        cb = plt.colorbar(im, cax=div.append_axes('right', size='5%', pad=0.1), extend=extend)
        # transposed: the map is wide and short, so an upright one-character label reads better
        cb.set_label(cbar_label, fontsize=fs, **({} if down else dict(rotation=0, va='center',
                                                                     labelpad=fs)))
        cb.ax.tick_params(labelsize=fs - 3)
    return im


def _log_edges(t):
    """Cell edges at the geometric midpoints of `t`, outer two extrapolated — the log-axis
    analogue of the linear midpoints pcolormesh would otherwise assume. Exactly-tied values
    are spread apart first: a tie collapses a cell to zero size and hides that checkpoint
    outright, which OLMo-2 1124-7B needs (its registry token counts are quantised, so steps
    600/700 both report 3e9 and 850/900 both report 4e9)."""
    lt = np.log(np.asarray(t, float)).copy()
    for i in range(1, len(lt)):
        if lt[i] <= lt[i - 1]:
            j = i
            while j < len(lt) and lt[j] <= lt[i - 1]:
                j += 1
            hi = lt[j] if j < len(lt) else lt[i - 1] + 0.02
            lt[i:j] = np.linspace(lt[i - 1], hi, j - i + 2)[1:-1]
    mid = (lt[:-1] + lt[1:]) / 2
    return np.exp(np.concatenate([[2 * lt[0] - mid[0]], mid, [2 * lt[-1] - mid[-1]]]))


@griddable
def plot_alpha_bands(model, source, hook=AB_HOOK, n_bands=20, k0=2, top_frac=0.75, ratio=6.0,
                     upto=None, rows=slice(None), bands=slice(None), vlim=(0.2, 2.4),
                     pct=(2, 98), cmap='magma', fit_window=ALPHA_FIT_WINDOW,
                     orient='time_down', side='tail', include_init=True, fs=14,
                     fit_style='lines', rasterized=False, title=None, cbar=True):
    """Heatmap of band α: rank bands across, checkpoints down. BOTH axes are log and the
    cells are drawn as a pcolormesh on real edges, so a row's height is its true share of
    log-token space — late checkpoints, which are linearly spaced in steps, correctly come
    out as thin slivers rather than being spread evenly and lying about their spacing.

    upto:  last row to keep (see rankme_peak) — the entropy-seeking phase on its own.
    bands: column slice, e.g. slice(1, -1) to drop the narrowest head band (only ~2 ranks
           wide, so its α is a two-point slope) and the top band before the tail cutoff
           (steep enough to sit at the ceiling of the shared colour scale).
    rows:  row slice, applied AFTER upto. Must be contiguous, as must `bands`.
    orient: 'time_down' (rank across, time down) or 'time_right' (transposed).
    side:  which end of the SPECTRUM the RankMe strip sits on — 'head', 'tail', or None for
           no strip. Named by the spectrum so it survives transposing (see SIDE_POS).
    vlim:  COUPLED — a (vmin, vmax) pair puts every panel on one scale, so cell colours are
           comparable across models. DECOUPLED — None autoscales each panel to the `pct`
           percentiles of its OWN plotted α. Couple to compare models, decouple to read
           structure within one: a shared scale leaves the flat-α families (OLMo-2, whose
           whole range is ~0.6–1.8) in the middle greys while the steep one (nanochat-d12)
           clips to blocks of white. Percentiles, not min/max, so one outlier cell cannot
           take the scale hostage — the colourbar arrows mark what is clipped."""
    xs, e, A = alpha_bands(source, model, hook, n_bands, k0, top_frac, ratio, include_init)
    idx = _row_index(len(xs), upto, rows, model)
    xs, A = xs[idx], A[idx]
    b = range(*bands.indices(A.shape[1]))
    if b.step != 1 or not len(b):
        raise ValueError(f'{model}: bands must be a non-empty contiguous slice, got {bands}')
    A, e = A[:, b.start:b.stop], e[b.start:b.stop + 1]      # n bands need n+1 edges
    vmin, vmax = vlim if vlim is not None else np.percentile(A, pct)
    ax = plt.gca()
    # edgecolors='face' strokes each cell in its own colour: pcolormesh antialiases every
    # quad independently, so without it a PDF shows hairline seams between cells when zoomed
    # out. rasterized=True is the other cure, at the cost of a vector heatmap.
    _render_map(ax, _log_edges(xs), e, A, orient,
                dict(cmap=cmap, vmin=vmin, vmax=vmax, edgecolors='face', linewidth=0.4,
                     rasterized=rasterized),
                fit_window, side, source, model, hook, idx,
                cbar, r'$\alpha$', 'both', title, fs, include_init, fit_style)
    plt.show()


def spectrum_map(source, model, hook=AB_HOOK, include_init=True):
    """(tokens, rank edges, λ) with λ of shape (checkpoints, d) — the ABSOLUTE eigenvalues
    (eigenspectrum × trace), one column per eigenvalue, rows per map_tokens. Non-positive entries
    become NaN (pythia-6.9b has exactly one: its last eigenvalue at step 32): a zero has no
    place on a log intensity scale, and drawing it as a hole beats dropping it silently."""
    ys, steps = get_ys(source, model, hook, 'eigvals')
    xs, keep = map_tokens(model, steps, include_init)
    A = np.asarray([np.asarray(y, float) for y in ys])[keep]
    return xs[keep], _log_edges(np.arange(1.0, A.shape[1] + 1)), np.where(A > 0, A, np.nan)


@griddable
def plot_spectrum_map(model, source, hook=AB_HOOK, upto=None, rows=slice(None),
                      ranks=slice(None), vlim=None, pct=(2, 100), cmap='magma',
                      fit_window=ALPHA_FIT_WINDOW, orient='time_down', side='tail',
                      include_init=True, fs=14, fit_style='lines', title=None, cbar=True):
    """The raw spectrum as a map: one column PER EIGENVALUE (no banding at all), checkpoints
    down, colour = the eigenvalue itself on a LOG intensity scale. Same axis treatment as
    plot_alpha_bands — both axes log, cells on real edges — so the two figures stack directly.
    upto/rows/vlim/orient/side as there; `ranks` is the column slice, over individual
    eigenvalues rather than bands.

    Colour range: vlim couples panels as before; pct autoscales each panel to those
    percentiles of its own λ, and pct=None takes the full min→max. The default clips the
    BOTTOM ONLY (2, 100), which is what gives this figure its contrast. A handful of early
    checkpoints hold eigenvalues 2–3 decades below everything else and stretch the range so
    far that the bulk of the data ends up crammed into the top third of the colourmap. Cut
    those and the wave separates. Do NOT clip the top to match: the head is few eigenvalues
    by count but a large slice of a log-rank axis, so any vmax below the maximum turns it
    into one flat block. (Same reason quantile equalisation fails here — equal count per
    colour is not equal area.)

    These are ABSOLUTE eigenvalues, so a model whose trace moves bodily — OLMo-2's drops ~2
    orders of magnitude — paints that shift as a whole-row brightness change ON TOP of the
    shape change this figure is meant to show. Making it invariant to that is a separate
    question, and not a settled one: dividing by the trace, the median, or the mid-range each
    answers a different question about what "the same spectrum, moved" means."""
    xs, e, A = spectrum_map(source, model, hook, include_init)
    idx = _row_index(len(xs), upto, rows, model)
    xs, A = xs[idx], A[idx]
    r = range(*ranks.indices(A.shape[1]))
    if r.step != 1 or not len(r):
        raise ValueError(f'{model}: ranks must be a non-empty contiguous slice, got {ranks}')
    A, e = A[:, r.start:r.stop], e[r.start:r.stop + 1]
    if vlim is not None:
        (vmin, vmax), extend = vlim, 'both'
    elif pct is not None:
        (vmin, vmax) = np.nanpercentile(A, pct)
        extend = ('both' if pct[0] > 0 and pct[1] < 100 else 'min' if pct[0] > 0 else
                  'max' if pct[1] < 100 else 'neither')
    else:
        (vmin, vmax), extend = (np.nanmin(A), np.nanmax(A)), 'neither'
    cm = plt.get_cmap(cmap).copy(); cm.set_bad('0.6')          # NaN = a non-positive eigenvalue
    ax = plt.gca()
    _render_map(ax, _log_edges(xs), e, np.ma.masked_invalid(A), orient,
                dict(cmap=cm, norm=LogNorm(vmin=vmin, vmax=vmax), rasterized=True),
                fit_window, side, source, model, hook, idx,
                cbar, r'$\lambda$', extend, title, fs, include_init, fit_style)
    plt.show()


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
    plt.xscale('log')
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
