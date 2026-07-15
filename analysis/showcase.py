# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: representation-geometry (3.14.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Showcase: the compression phase, decomposed
# One-figure-per-finding companion to [docs/final_report.md](../docs/final_report.md), from the
# Li et al reproduction to the RQ2 task-geometry results. Exemplar models per figure; the full
# per-model grids live in experiments.ipynb (Exp 4.x), experiments_padded.ipynb and
# rq2_results.ipynb. Methods: docs/rank_ledger_notes.md, docs/rq2_methods.md.

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
    display(HTML(f'<div style="font-size:0.88em; opacity:0.8; max-width:56em; '   # inherit theme
                 f'margin:0.2em 0 1.2em 0.5em"><b>Figure.</b> {txt}</div>'))      # text color

HK = build_hooks()
BLOCK_SAMPLES = "block_representations_samples"
TRAINSET_PACKED, FINEWEB_PADDED = "rankme_trainset_packed_4MT", "rankme_fineweb_padded"
ALL4 = ['pythia-1b-deduped', 'pythia-6.9b-deduped', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B']
PAIR = ['pythia-1b-deduped', 'OLMo-2-1124-7B']          # one exemplar per family
n_blocks = {'pythia-1b-deduped': 16, 'pythia-6.9b-deduped': 32,
            'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32}

# %% [markdown]
# ## 1. Reproduction: the Li et al phases
# **Finding:** the warmup → entropy-seeking → compression RankMe trajectory reproduces across
# both families, both data modes (their setting: padded fineweb last-token; ours adds the
# packed training-mix). This is the curve everything below decomposes.

# %%
# Packed curve from the all-model samples run (the old rankme_trainset_packed_4MT run only
# covered a model subset, which caused skipped-file noise here before).
srcs = [(BLOCK_SAMPLES, ('after_final_norm', 'acts_centered'), 'trainset packed'),
        (FINEWEB_PADDED, HK.AFN_AC, 'fineweb padded')]
for model in ALL4:
    grid_start(ncols=2, title=f'Li et al reproduction — {get_model_label(model)}')
    plot_group('rankme', srcs, [model])
    plot_group('alpha', srcs, [model])
    grid_show()
    caption(f'RankMe (left) and power-law slope α (right) of the CENTERED final-stream covariance '
            f'(after the final norm) over training, {get_model_label(model)}, on the packed training '
            f'mix and on padded fineweb last-token data (Li et al\'s setting). The warmup → '
            f'entropy-seeking (rank rise) → compression (post-peak decline) trajectory is the '
            f'phase structure every later figure decomposes.')

# %% [markdown]
# ## 2. The phases are spectrally local: a concentration front, not a contraction
# **Finding (universal, 4/4 models):** the measured compression lives in the top ~10–30
# eigendirections; band alphas show a front that reaches ranks 32–128 late, 128–512 barely,
# and NEVER ranks 512+ (that band flattens monotonically through all of training — falling α
# = entropy-seeking never ends there). Both statistics are invariant to the head's growth.

# %%
KS, WINDOWS = (0, 8, 32, 128, 512), ((11, 100), (128, 512), (512, -20))
for model in PAIR:
    tails = [(BLOCK_SAMPLES, ('after_final_norm', 'acts_centered'), f'k={k}', ('tail_rankme', {'k': k})) for k in KS]
    alphas = [(BLOCK_SAMPLES, ('after_final_norm', 'acts_centered'), f'{a}-{b}', ('alpha_window', {'k0': a, 'k1': b})) for a, b in WINDOWS]
    grid_start(ncols=2, title=f'Head-removed RankMe / band alpha — {get_model_label(model)}')
    plot_group('tail_rankme', tails, [model], color_palette='gradient')
    plot_group('alpha_window', alphas, [model], color_palette='gradient')
    grid_show()
    caption(f'Spectral locality of the phases, {get_model_label(model)}, centered final stream '
            f'(after the final norm — Li et al\'s measurement point). Left: RankMe of the '
            f'spectrum with the top k eigendirections removed (k = 0…512, light→dark). Right: '
            f'power-law slope α fit inside fixed rank windows. '
            f'The compression decline survives k=8–32 but is gone by k=512; the deepest window '
            f'(512–end) moves in the OPPOSITE direction (α falls = flattening) throughout '
            f'training. The compression is a top-of-spectrum concentration front, not a global '
            f'contraction.')

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
    axes[1, 1].plot(xs[sl], alphas, 'k--', lw=2, label='baseline α(t), ALL bands')
    axes[1, 1].set(xscale='log', xlabel='tokens', ylabel='band α',
                   title='band slopes: a pure tilt has ONE α at every rank')
    axes[1, 1].legend(fontsize=7)
    plt.tight_layout()
    plt.show()

tilt_baseline()
caption('The tilt baseline (dashed): a power-law spectrum λ_i = i^{−α(t)} with α increasing '
        'linearly in log-tokens, endpoints calibrated so its RankMe equals the measured '
        'stream RankMe at the peak and at the RankMe MINIMUM — the point where the '
        'compression-seeking phase ends; the baseline is a model of the compression phase '
        'only, not of the late re-expansion. Top left: the headline curves (agreement at '
        'both calibration points is by construction; what matters is the smooth featureless '
        'shape in between). Top right: head-removed tail RankMes, normalized to their value '
        'at the peak — under the baseline every band falls in lockstep with the head '
        '(dashed), whereas the measured deep band (k=512, solid red) RISES through the '
        'compression phase. Bottom row, the same contrast in α: a pure tilt has a single '
        'slope at every rank, so the baseline α(t) (dashed) should describe every band at '
        'once — measured, the global α (left) steepens far less than the baseline needs, and '
        'the band slopes (right) fan out: the 512+ band FALLS (flattens) while the head '
        'bands steepen. Real compression is a head-concentration event, not a uniform tilt.')

# %% [markdown]
# ## 3. The ledger: what carries the compression, per family
# **Finding:** the exact decomposition ΔS = χ + quality + interference (sums to the log-RankMe
# trajectory, black line) splits by family: Pythia's decline is QUALITY-carried (red), OLMo-2's
# is INTERFERENCE-carried (purple) with quality recovering. χ (green) is stable everywhere —
# compression is never "writes stop exploring".

# %%
def ledger_stack(model):
    terms = {y: np.sum([_lib.get_ys(BLOCK_SAMPLES, model, (f'blk{l}', 'block_ledger'), y)[0]
                        for l in range(n_blocks[model])], axis=0)
             for y in ('chi', 'quality', 'interference')}
    xs = np.asarray(_lib.get_xs_tokens(
        model, _lib.get_ys(BLOCK_SAMPLES, model, ('blk0', 'block_ledger'), 'chi')[1]), float)
    keep = xs > 0                                        # step 0 is off a log axis anyway
    xs, terms = xs[keep], {k: np.asarray(v, float)[keep] for k, v in terms.items()}
    # Refine the grid with every term's sign crossings (interpolated in log-x, where the
    # drawn segments are straight) so each fill closes vertically at zero instead of
    # drawing a twisted quadrilateral across the opposite stack when a term changes sign.
    lx = np.log(xs)
    cross = []
    for v in terms.values():
        i = np.nonzero(np.signbit(v[:-1]) != np.signbit(v[1:]))[0]
        cross.append(lx[i] + v[i] / (v[i] - v[i + 1]) * (lx[i + 1] - lx[i]))
    grid = np.unique(np.concatenate([lx, *cross]))
    terms = {k: np.interp(grid, lx, v) for k, v in terms.items()}
    gx = np.exp(grid)
    plt.figure(figsize=(8, 4))
    pos, neg = np.zeros(len(gx)), np.zeros(len(gx))
    for name, c in (('chi', 'tab:green'), ('quality', 'tab:red'), ('interference', 'tab:purple')):
        up, dn = np.clip(terms[name], 0, None), np.clip(terms[name], None, 0)
        plt.fill_between(gx, pos, pos + up, label=name, color=c, alpha=0.55, lw=0)
        plt.fill_between(gx, neg, neg + dn, color=c, alpha=0.55, lw=0)
        pos, neg = pos + up, neg + dn
    plt.plot(gx, sum(terms.values()), 'k', lw=2.5, label='ΔS total')
    plt.xscale('log'); plt.xlabel('tokens'); plt.ylabel('rank entropy')
    plt.legend(); plt.title(f'Ledger contributions — {get_model_label(model)}'); plt.show()
    caption(f'Exact per-block rank ledger summed over blocks, {get_model_label(model)}: '
            f'ΔS = χ (green) + quality (red) + interference (purple); black = the total, i.e. '
            f'the stream log-RankMe trajectory. Positive contributions stack upward from zero, '
            f'negative downward; at a sign change a term\'s fill pinches to zero and continues '
            f'on the other side. Pythia\'s compression is quality-carried; OLMo-2\'s is '
            f'interference-carried with quality recovering; χ is stable in all four models.')

for model in ALL4:
    ledger_stack(model)

# %% [markdown]
# ## 4. Pythia's rogue write: a sink direction (window-start + first-newline)
# **Definition:** the "rogue direction" = the top eigenvector v1 of the CENTERED covariance of
# blk3.mlp's output (final ckpt) — the axis of maximal variance of that write. It earns a name
# because it alone carries 94–99% of the write's variance (that is what "rank ≈ 1 write" means).
# "Projection onto it" = (x − μ)·v1 per token row.
# **Precise claim (position-split analysis, Jul 13):** the direction is a SINK direction with
# two carriers per 511-token window: (1) POSITION 0, unconditionally — signed proj +1657 on
# arbitrary mid-text content tokens, 510/512 windows (the packed-stream form of the
# attention-sinks papers' "BOS" slot, which is positional — our mix contains no BOS at all);
# (2) the FIRST NEWLINE of the window (+2135; 478/486 spiking newlines are the window's
# first, median position 25). The other 96% of newlines are bulk-ordinary (−9.5 vs bulk −7) —
# the earlier headline "mean |proj| 98 on newlines" is that 4%/96% mixture. Variance shares:
# first-newline slots 60.4%, pos-0 slots 38.7%, all remaining 249k rows 0.9%. Correlational,
# final-checkpoint: it locates WHERE the direction activates, not what it is for. The write
# collapses to rank ≈ 1 at the RankMe peak, grows to ~22× the stream's energy, and the late
# blocks increasingly write against it (signed trace → −0.65; pos-0 output norm 70.7 < bulk
# 131 — the spike is largely cancelled by the end). OLMo-2 shows none of this (top-eigval
# share ~0.01 vs 0.99). Preimage check: the input direction that best predicts the score is
# UNALIGNED with the raw '\n' embedding (cos −0.013) — the spike is constructed by the MLP
# from processed features (sequence-position / novelty-like), not echoed from the embedding.
# **Follow-ups (Jul 13, figures below):** an explicit <|endoftext|> at position 0 does NOT
# absorb the first-newline slot (both slots are structural, not a missing-resting-token
# artifact), and the depth profile shows deposit at blk3 → unchanged ride through blk12 →
# scrubbed by the late blocks, with ordinary tokens carrying ~27%→4% of their centered norm
# along the direction mid-stack.

# %%
for model in ('pythia-1b-deduped',):
    half = n_blocks[model] // 2
    outs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'acts_centered'), f'mlp {l}')
            for l in range(half)]
    grid_start(ncols=2, title=f'Rogue write trajectory — {get_model_label(model)}')
    plot_group('rankme', outs, [model], color_palette='gradient', ylog=True, title='write RankMe (centered)')
    plot_group('trace', outs, [model], color_palette='gradient', ylog=True, title='write trace')
    grid_show()
    caption(f'The rogue write forming, {get_model_label(model)}: RankMe (left) and total '
            f'variance/trace (right) of the centered mlp-output covariance for the FIRST {half} '
            f'mlp writes (light→dark = deeper; the late half is omitted — late writes '
            f'legitimately need large magnitude to move the by-then-large stream, so their '
            f'traces would dominate the right panel without saying anything about blk3). '
            f'Among its early peers exactly one write departs from the family: blk3 collapses '
            f'to rank ≈ 1 at the stream RankMe peak while its trace keeps growing to ~22× the '
            f'stream\'s energy — one huge direction, not a quiet write. Every other early '
            f'write keeps a broad spectrum.')

# %%
# rogue_hist, step by step: (1) load the RAW per-token activations of blk3.mlp's output
# (block_rogue_id run, final ckpt; rows = the 262k packed-mix token positions, in order);
# (2) project each centered row onto the write's TOP eigendirection -> one scalar per token
# ("how strongly this token activates the rogue direction", i.e. (x − μ)·v1 with v1 the top
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
    plt.xlabel('|projection onto rogue direction|'); plt.ylabel('token count (log)')
    plt.title(f'Who carries the rogue direction — {get_model_label(model)}'); plt.legend(); plt.show()

rogue_hist('pythia-1b-deduped')
caption('Token attribution of the rogue direction (pythia-1b, final checkpoint, raw per-token '
        'activations of blk3.mlp.out): histogram of |(x − μ)·v₁| — the projection of each '
        'centered token row onto the write\'s top centered eigendirection — split three ways: '
        'window-start rows (blue, ANY token identity), newline-bearing tokens elsewhere (red), '
        'all others (gray); log count axis. Both sink slots separate from the bulk: position 0 '
        'fires unconditionally (~1657 mean), and the red tail is almost entirely each '
        'window\'s FIRST newline (~2135) — 96% of newlines sit in the bulk. Correlational and '
        'final-checkpoint only — it locates WHERE the direction activates, not what it is for.')

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
    plt.xlabel('mean |projection onto rogue direction|')
    plt.title(f'Token types ranked by rogue-direction activation — {get_model_label(model)}\n(red = newline-bearing; small n = noisy)')
    plt.show()

rogue_ranking('pythia-1b-deduped')
caption('Systematic version of the previous figure: mean |projection onto the rogue direction| '
        'per token TYPE, all types with n ≥ 20 occurrences, top 20 shown. Red bars = the '
        'decoded token contains a newline. The black tick on each bar marks the LOWER edge of '
        'the 95% CI of the mean (1.96·SD/√n, clamped at 0): a tick far left of its bar means '
        'the mean is noise-dominated — the low-count word types wash toward the bulk, while '
        'the newline variants\' huge n pins their means tightly. The wide CIs on the n≈20 '
        'types are the honest picture: only the newline rows are statistically solid. NB this '
        'per-TYPE view structurally hides the other sink slot — the 512 window-start rows '
        '(unconditional ~1657 spikes) are spread across ~500 distinct types; see the histogram '
        'above for the position-split view.')

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
#   almost fully removed by before_final_norm. Ordinary tokens DO carry rogue-direction
#   content mid-stack — ~27% of their centered row norm at blk4, decaying monotonically to
#   ~3.6% at the final stream — confirming the mid-stack route by which rogue variance can
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
    fig.suptitle('The rogue spike over depth — pythia-1b, final ckpt (midstack run)')
    for slot, c in slots:
        absv = [ms['write'][slot]] + [ms[l]['abs'][slot] for l in leaves[1:]]
        frac = [ms[l]['frac'][slot] for l in leaves[1:]]
        axes[0].plot(range(len(leaves)), absv, marker='o', lw=2, color=c, label=slot)
        axes[1].plot(range(1, len(leaves)), frac, marker='o', lw=2, color=c, label=slot)
    axes[0].set(yscale='log', ylabel='mean |projection onto rogue direction|',
                title='absolute (log)')
    axes[1].set(ylabel='mean |projection| / centered row norm', title='fraction of the row')
    for ax in axes:
        ax.set_xticks(range(len(leaves)))
        ax.set_xticklabels(leaves, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

rogue_depth_profile()
caption('Depth profile of the rogue-direction content per token class (blue/red = the two '
        'sink slots, orange = other newlines, gray = ordinary tokens): mean |(x − μ)·v₁| of '
        'the stream at each depth, absolute (left, log) and as a fraction of each row\'s '
        'centered norm (right). The blk3 write deposits the spike; the sink slots carry it '
        'at ~full row strength (fraction ≈ 0.99) until blk12, blk15 halves it, and '
        'before_final_norm has removed almost all of it (fraction ≈ 0.2 of much larger '
        'rows). Ordinary tokens are NOT clean mid-stack: a quarter of their centered norm '
        'at blk4 lies along the rogue direction, decaying to ~4% by the final stream — '
        'the late blocks scrub the direction from every token, sinks most of all. '
        'EOT-twin control (not plotted): prepending <|endoftext|> does not absorb the '
        'first-newline slot (2178 vs 2132 without it), so the second slot is not a missing '
        'resting-token artifact.')

# %%
def cancellation_headmass(model):
    L = n_blocks[model]
    mats, steps = _lib.get_ys(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), 'signed_trace')
    leaves = _lib.get_y(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), 'leaves', steps[0])
    xs, last = _lib.get_xs_tokens(model, steps), leaves.index(f'blk{L - 1}.mlp.out')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'Cancellation / head-mass / concentration — {get_model_label(model)}')
    for l in (1, 3, 5, L // 2):
        i = leaves.index(f'blk{l}.mlp.out')
        axes[0].plot(xs, [m[i, last] for m in mats], marker='o', lw=2, label=f'mlp {l} ~ mlp {L-1}')
    axes[0].set(xscale='log', title='signed trace vs last write', xlabel='tokens'); axes[0].legend(fontsize=8)
    for l in (0, L // 2, L - 1):
        cs, ss = _lib.get_ys(BLOCK_SAMPLES, model, (f'blk{l}.mlp.out', 'eigendirection_attrib'), 'contrib')
        axes[1].plot(_lib.get_xs_tokens(model, ss), [float(np.abs(c[:32]).sum() / np.abs(c).sum()) for c in cs],
                     marker='o', lw=2, label=f'mlp {l}')
    axes[1].set(xscale='log', ylim=(0, 1), title='head-mass (top-32 share)', xlabel='tokens'); axes[1].legend(fontsize=8)
    share = [_lib.get_y(BLOCK_SAMPLES, model, (f'blk{l}.mlp.out', 'acts_centered'), 'eigenspectrum', None)[0]
             for l in range(L)]
    axes[2].bar(range(L), share, color='tab:red' if 'pythia' in model else 'tab:blue')
    axes[2].set(ylim=(0, 1), title='top-eigval share per mlp write (final)', xlabel='block')
    plt.show()
    caption(f'Three views of write structure over training, {get_model_label(model)}. Left: '
            f'signed-trace coupling of early/mid mlp writes against the LAST mlp write — '
            f'negative = the final write pushes against that direction (cancellation), '
            f'positive = reinforcement. Middle: fraction of each write\'s eigendirection-'
            f'attribution mass in its top 32 directions ("head-mass" — how concentrated the '
            f'write is). Right: top-eigenvalue share of every mlp write\'s centered covariance '
            f'at the final checkpoint — the bar near 1.0 at blk3 in Pythia IS the rogue; '
            f'OLMo-2 has no bar above ~0.1.')

for model in ALL4:
    cancellation_headmass(model)

# %% [markdown]
# ## 5. OLMo-2's mechanism: distributed aligned reinforcement
# **Finding:** no rogue write anywhere; the late writes are mutually ALIGNED (positive signed
# trace between neighbors — reinforcement, the opposite of Pythia's cancellation), diffuse in
# token space (top-eigval share ~10%, no dominant token class), and carry the interference-driven compression.

# %%
# The write-ensemble structure that carries §5's claim: the mlp×mlp signed-trace coupling.
# (The mlp×attn view — a separate memory-management observation — lives in §8.)
from analysis.experiments_lib import submatrix

def coupling_grid(models, rows='mlp', cols='mlp', step=None):
    nrows = (len(models) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.5 * nrows), squeeze=False)
    fig.suptitle(f'{rows} × {cols} signed-trace coupling, final checkpoint')
    for ax, model in zip(axes.flat, models):
        bb = lambda y: _lib.get_y(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), y, step)
        M, rl, cl = submatrix(bb('signed_trace'), bb('leaves'), rows=rows, cols=cols)
        if rows == cols:
            M = M - np.diag(np.diag(M))
        v = np.abs(M).max()
        im = ax.imshow(M, cmap='coolwarm', vmin=-v, vmax=v)
        ax.set(title=get_model_label(model), xlabel=f'{cols} block', ylabel=f'{rows} block')
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()

coupling_grid(ALL4)
caption('Signed trace between every pair of mlp writes at the final checkpoint, all four '
        'models in one grid (red = aligned/reinforcing, blue = opposed/cancelling; diagonal '
        'removed; each panel has its own color scale). The family contrast is the point: '
        'both OLMo-2 panels show the late-block red square (rows/cols ~L/2 onward — every '
        'late write positively aligned with every other, the distributed reinforcement of '
        'the section title), while both Pythia panels instead show a blue band through the '
        'rogue block (the late writes anti-aligned with the rogue write — cancellation). '
        'Combined with the diffuse token attribution (§4\'s top-eigval share panel: OLMo\'s '
        'writes never exceed ~0.1) and the interference-carried ledger (§3), this is what '
        '"distributed aligned reinforcement" means operationally.')

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
#   blocks ANTI-aligned with the ensemble (the cancellation of the rogue write).

# %%
def alignment_vs_interference(model):
    L = n_blocks[model]
    mats, csteps = _lib.get_ys(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), 'signed_trace')
    leaves = _lib.get_y(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), 'leaves', csteps[0])
    cxs = _lib.get_xs_tokens(model, csteps)
    idx = [leaves.index(f'blk{k}.mlp.out') for k in range(L // 2, L)]
    late = [float(np.mean([m[i, j] for i in idx for j in idx if i != j])) for m in mats]
    intf = {k: np.asarray(_lib.get_ys(BLOCK_SAMPLES, model, (f'blk{k}', 'block_ledger'), 'interference')[0])
            for k in range(L)}
    isteps = _lib.get_ys(BLOCK_SAMPLES, model, ('blk0', 'block_ledger'), 'interference')[1]
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
    caption(f'{get_model_label(model)} — left pair (shared token axis): the late-write '
            f'alignment (red; mean signed trace over all pairs of deeper-half mlp writes — '
            f'positive = the late writes add variance along shared directions) above the '
            f'total interference ledger term (purple; the cross-block part of the stream\'s '
            f'rank-entropy change, §3 — negative = cross-terms are compressing the stream). '
            f'Right: one dot per block (label = block index, dark = deep); x = how aligned '
            f'that block\'s write is with the late-half writes, y = that block\'s own '
            f'interference contribution, both averaged over the final third of checkpoints. '
            + ('For OLMo-2 the two onsets coincide (left) and the compression-carrying '
               'negative-interference blocks are exactly the strongly-coupled late ones '
               '(dots run down-right): the aligned ensemble IS the compressor.'
               if 'OLMo' in model else
               'For Pythia there is no positive alignment to speak of (left, red stays ≤ 0) '
               'and the negative-interference blocks sit at NEGATIVE coupling (upper/lower '
               'LEFT): its interference is the rogue-write cancellation, not reinforcement.'))

for model in ALL4:
    alignment_vs_interference(model)

# %% [markdown]
# ## 6. The toy bridge (RQ3)
# **Reproduction note:** Li et al's Fig 4 requires their CONSTRUCTED init, readable off the
# gray t=0 markers in their figure (frequent-class features clustered on separated directions
# with aligned W columns; both rare classes coincident with zero weights, split only by a
# tiny jitter δ). GD preserves the rare-pair swap symmetry, so the shared path is exact and
# the split time ~ log(1/δ). Mechanism of the decline (docs/toy_fig4_addendum.md): the
# co-traveling rare pair's off-diagonal covariance CANCELS the frequent classes' tilt; the
# fork collapses that cancellation, reopening the eigengap — a kick that scales with the
# squared fork amplitude (which sets the fork→decline lag: zero when the rare pair starts at
# the exact origin, as in the paper and our canonical run; ~25 steps at a −0.25 offset). The
# kick always happens; whether it PRINTS as a compression phase is a visibility condition:
# the spectrum must have saturated before the fork (flat baseline), which the constructed
# init controls via δ. iid init at any scale fails it — large init breaks the symmetry at
# order 1 (no shared path; a stalled rare class fakes a bigger decline — our old spec), tiny
# iid init welds the fork to spectrum saturation (kick present but swallowed by the rising
# baseline). Current `clustered` spec reproduces B/C/D jointly, 4/5 jitter seeds; the toy's
# compression is a transient (RankMe back to ~2.0 by step ~3000).
#
# **Findings:** the paper's toy reproduces (phases + all four negative controls) through this
# repo's unmodified metric pipeline; the residual stream is constitutive (no-residual stack
# loses the phases entirely); the residual toy reproduces Pythia's quality-driven ledger
# signature; read-norm suppresses the toy's rogue (a toy/LLM discrepancy — real Pythia has
# read-norm and a rogue anyway); and WRITE-norm (the true OLMo-2 reordered-norm analog)
# suppresses the rogue mechanism COMPLETELY (6/6 seeds, both wirings; min write RankMe
# 5.6–7.5 vs 1.7–2.6 un-normed; 4/6 runs lose the post-peak decline entirely), while
# interference never becomes the carrier in any of the 24 grid runs. Figures: saved toy
# outputs + the architecture-knob grid.

# %% [markdown]
# # ⚠️ Multi-layer toy trajectories: RESOLVED (Jul 13) — full study in docs/toy_multilayer.md
# The MULTI-task toys (figures below) never reproduce Li et al's dip→rise→peak→decline at
# any depth or init — and the reason is now demonstrated, not conjectured: the curve needs
# (T) a compression event timed after spectrum saturation, (K) a large-enough kick, and
# (I) near-identity transmission from the feature layer to the measurement point. Deep
# stacks print the FULL sequence when those hold (residual with small-at-init blocks,
# 2/6/12 blocks; a no-residual stack with identity-initialized blocks; and a skewed
# 6-class bottlenecked task where moving ONLY the fork delay δ toggles the print). The
# MULTI task fails (T) by construction (its 26 rare-class forks smear across the
# entropy-seeking rise). Trajectory-shape wording in the older captions below (incl. "the
# residual stream is constitutive for the phases" and arch-grid "post-peak decline") is
# pre-revision; the corrected form of the residual claim is transmission (toy_multilayer
# §5). Ledger-term attributions are not implicated.
#
# **Reading note — centered vs uncentered.** All LLM figures in this notebook use CENTERED
# spectra unless labeled otherwise. The Li-facing toy figures below (Fig 4 reproduction,
# controls, multi-layer stream panels) use UNCENTERED RankMe, matching Li et al's definition
# on raw features; only the architecture-grid figure (and its rogue check) uses centered
# spectra, to match the LLM-side pipeline it is compared against — its axes say so.
# CAUTION for the multi-layer stream panels specifically: at depth the writes' accumulated
# bias means dominate the uncentered spectrum (~50% of trace), so those per-depth curves
# mostly track the mean, not the representation geometry — appendix §C plots uncentered vs
# centered side by side and resolves the apparent depth pattern.

# %%
lifig = 'reference/li et al reference tex/figures/Fig4_top.png'
if os.path.exists(lifig):
    plt.figure(figsize=(12, 5)); plt.imshow(mpimg.imread(lifig)); plt.axis('off')
    plt.title('Li et al, Fig 4 (original)'); plt.show()
    caption('The ORIGINAL Li et al Fig 4, top row, for side-by-side comparison: (A) model '
            'schematic, (B) classifier weight rows W_i in 2D over training, (C) feature '
            'vectors f(x), (D) RankMe and top singular values — phases drawn as dotted '
            '(warmup) / solid (entropy-seeking) / dashed (compression) segments.')

# %%
TOY_CAPS = {
    'fig4_single.png':
        'Reproduction of Li et al Fig 4 B–D: single-layer toy, skewed class counts (2,2,1,1), '
        'bottleneck d=2, cross-entropy full-batch GD (lr 0.25, 300 steps), THEIR constructed '
        'init (clustered frequent classes, coincident rare pair + δ=1e-3 jitter — see the '
        'reproduction note above). B: classifier rows W_i; C: features f(x); one color+marker '
        'per class (magenta ▲, orange ●, blue ■, green ◆ — blue/green are the singleton '
        'classes), line style = phase (dotted warmup, solid entropy-seeking, dashed '
        'compression). The rare pair co-travels on one path and splits late onto the frequent '
        'classes\' axes — their panel-B geometry. D: RankMe of the final features (uncentered, '
        'their definition) peaks ≈1.9996 at step 204 then genuinely declines (to 1.9931) — '
        'the fork collapses the covariance cancellation between the rare pair and the '
        'frequent classes\' tilt, reopening the λ1/λ2 gap (see the reproduction note); top-2 '
        'eigenvalues (per unique sample set, dup-corrected) on the right axis. With the rare '
        'pair starting at the exact origin the decline onset coincides with the fork (as in '
        'the paper), so the dashes begin at the visible split.',
    'fig_controls.png':
        'Negative controls, all three from the paper (uniform class counts; no bottleneck '
        'd=3 = |V|−1; MSE loss on both count patterns), one panel per variant with its own '
        'axes. First panel: the skew+bottleneck+CE reference (the canonical clustered spec '
        'run to 3000 steps; dotted vertical = the paper\'s 300-step Fig-4 window) — it alone '
        'shows the post-peak decline, and past the window the decline is a transient that '
        'recovers by ~3000 (addendum). The CE controls are monotone after their warmup dip. '
        'NB mse_skew\'s small dip-then-recovery (~steps 40–90) is NOT a rare-class fork: '
        'checked on the trajectories (toy/appendix_sweeps.py mse_skew_check, 3 seeds), the '
        'rare classes are never learned under MSE — per-class MSE stays pinned at the 0.25 '
        'no-prediction floor and the rare weight columns DECAY from their random init '
        '(norm ~0.42) to ≤0.016 — Li et al\'s gradient-starvation claim confirmed. With the '
        'rare classes inert, the wobble can only be frequent-class reorganization; the CE '
        'fork mechanism (a rare pair being learned and splitting) is not available here.',
    'fig_multi_multi_residual_nonlinear.png':
        'Residual multi-layer toy (6 blocks, nonlinear writes, d=16, 32 classes, skewed '
        'counts; final loss 0.0004 — the task is fully solved). Left: uncentered stream '
        'RankMe at every depth (viridis light→dark = deeper), blk k input, last curve = '
        'final stream. Reading warning on this panel (appendix C has the full analysis): '
        'these per-depth curves are UNCENTERED (Li\'s definition), and in the multi-layer '
        'toy the writes\' accumulated bias means hold ~50% of the uncentered trace at every '
        'depth ≥ 1 — the apparent "deep streams start collapsed and only decline" shape is '
        'that shared-mean artifact. CENTERED (the LLM-comparable object), every stream '
        'starts near-full-rank and the FINAL stream compresses hardest of all (8.9 → 2.5, '
        'small early bump to 9.1) — the end-of-model compression pattern is present, '
        'matching the LLMs. What the toy lacks at every depth is a pronounced '
        'entropy-seeking rise, because theta (the toy\'s learnable feature matrix — each '
        'sample\'s feature vector is a free parameter, Li et al\'s f(x)) initializes as '
        'random noise already at full rank — nothing to expand. '
        'The toy\'s compression evidence is the LEDGER (middle panel: ΔS = '
        'χ/quality/interference summed over blocks, quality-carried — Pythia\'s signature), '
        'computed on centered quantities throughout. Right: per-write energy shares w_k '
        '(guard: no single write dominates, so summed terms are representative).',
    'fig_multi_multi_plain.png':
        'The same multi-layer toy WITHOUT the residual stream (each block replaces rather '
        'than adds), at the fair-shot lr 0.02 that FULLY solves the task (final loss '
        '0.00003; the natural objection "maybe it just fails to learn" does not apply — '
        'lr 0.005 underfit, 0.05 diverges). Same three panels. The phase structure is '
        'still gone: every stream depth sits flat until ~10⁴ steps, then collapses to '
        'RankMe ~1–2 with a small recovery — never dip→rise→peak→decline. The residual '
        'stream is constitutive for the phase structure, not incidental.',
    'fig_arch_grid.png':
        'Architecture-knob grid: 2 writes per block × {no norm, pre-norm, write-norm, '
        'both} × {parallel, sequential wiring} × 3 seeds (one thin line per run; color = norm '
        'cell, solid/circle = parallel, dashed/square = sequential). Left: RankMe of the '
        'CENTERED FINAL stream (the stream after the last block — centered to match the LLM '
        'pipeline, unlike the uncentered Li-facing figures above). Middle: summed quality '
        'ledger term (the Pythia carrier). Right: rogue check at the final step — each run\'s '
        'minimum write RankMe vs that write\'s energy share. Write-norm (green, the OLMo-2 '
        'reordered-norm analog) abolishes the rogue in all 6 runs; no knob produces '
        'interference-carried compression.',
}
for f, cap_txt in TOY_CAPS.items():
    p = f'toy/figures/{f}'
    if os.path.exists(p):
        plt.figure(figsize=(11, 6)); plt.imshow(mpimg.imread(p)); plt.axis('off'); plt.show()
        caption(cap_txt)

# %% [markdown]
# ### The multi-layer resolution: the phase curve passes through deep stacks
# The study behind this figure — every ablation, the exact dS/dt = −Cov_p(g, log p)
# mechanism, and the three print conditions (event timing, kick size, near-identity
# transmission) — is docs/toy_multilayer.md. Shown here: the three decisive positives.

# %%
def _toy_rm(tag):
    r = np.load(f'data/results/toy_{tag}/results_toy-{tag}.npy', allow_pickle=True).item()
    steps = sorted(r)
    return steps, [r[s]['before_final_norm']['acts_uncentered']['rankme'] for s in steps]

def _toy_fork(tag):
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
                             else 'δ=1e-8 (prints)' if tag == 'skewpair_deep_d8_s0' else None))
        fk = _toy_fork(tag)
        if fk:
            ax.axvline(fk, color=ln.get_color(), lw=0.9, ls=':')
    ax.set(xscale='log', xlabel='step', title=title)
    if ax is axes[2]:
        ax.legend(fontsize=8)
axes[0].set_ylabel('RankMe (uncentered features)')
fig.suptitle('Li et al\'s phase curve through multi-layer stacks — decline onset = the rare-pair fork (dotted)')
plt.tight_layout(); plt.show()
caption('The multi-layer resolution (full study: docs/toy_multilayer.md). Left: Li et al\'s '
        'constructed init under a 6-block RESIDUAL stack with small-at-init writes — the '
        'complete dip → rise → peak → decline → transient recovery prints at the final '
        'stream, with the decline starting exactly at the rare-pair fork (dotted vertical, '
        'detected from cos(w_rare) leaving the shared path), 3/3 seeds. Middle: the same '
        'WITHOUT a residual stream, blocks initialized as identity + noise — it prints '
        'equally well (3/3): the operative condition is near-identity transmission from '
        'features to measurement point, which the residual provides by default and a plain '
        'stack must have arranged. Right: the unification case — a skewed 6-class task '
        'through a d=4 bottleneck and a depth-6 stack, where ONLY the fork delay δ is '
        'moved: at δ=1e-3 the fork lands mid-rise and is masked (no decline); at δ=1e-8 it '
        'lands after the saturation plateau and the decline prints at the fork checkpoint '
        '(2.96 → 2.58). One mechanism, three instantiations; conditions and their LLM '
        'mapping in the doc. Known open items (doc §6): the nonlinear-block variant '
        'declines BEFORE its fork (a second, unexplained mechanism), and the kick\'s '
        'growth with depth is observed, not derived.')

# %% [markdown]
# ## 7. RQ2: task geometry beyond the general population
# # ⚠️ EVERYTHING IN THIS SECTION IS UNDER HEAVY REVISION — DO NOT CITE (Jul 13)
# The documented excess-mass aggregates mixed ACTIVATION and GRADIENT quantities (acts-only
# recomputation inverts the quotes/memorized ordering and weakens math), the ref pairing of
# the stored gen_vs_ref results is unverified, and several cross-population comparisons
# straddle incompatible geometries (packed all-token vs padded last-token covariances are
# different measured objects). Verdicts and figures below are the ORIGINAL, contaminated
# versions, kept only until the redo (plan.md item 1); final_report §4 carries the full
# diagnosis.
#
# **Findings (as originally written — suspect):** maths text carries real shared structure
# (excess 2.3–2.4× the split-half null, both families), located mid-spectrum (not the
# predicted tail). The two "memorised" populations split: verbatim strings ≈ lexical
# early-layer novelty; famous quotes = the most structured population measured. Split-half
# coherence (independent whiteners, raw-space subspaces): math > quotes > memorized
# everywhere, and memorized coherence is LOWEST at the early blocks where its excess lives —
# independent per-item storage where storage happens.

# %%
RQ2_MODELS = ['pythia-1b-deduped', 'OLMo-2-0425-1B']
L16 = [f'blk{k}.attn.in' for k in range(16)] + ['before_final_norm']

def leaf_stat(cfg, model, leaf):
    res, steps = load_results(cfg, model)
    if res is None: return np.nan
    g = res[steps[-1]].get(leaf, {}).get('gen_vs_ref', {}).get('acts')
    if g is None: return np.nan
    lam = np.asarray(g['eigvals'], float)
    return float(np.log(np.clip(lam, 1e-12, None))[lam > 1].sum())

for model in RQ2_MODELS:
    pops = {'math web (packed)': 'rq2_math_web', 'null packed': 'rq2_general_packed_b',
            'quotes (padded)': 'rq2_quotes', 'memorized (padded)': 'rq2_memorized',
            'null padded': 'rq2_general_padded_b'}
    if model.startswith('OLMo'):                # pythia_memorized is a pythia-specific dataset;
        pops.pop('memorized (padded)')          # OLMo's mem set is Merullo et al's, packed
        pops['memorized OLMo (packed)'] = 'rq2_memorized_olmo'
        pops['null packed 73k (N-matched)'] = 'rq2_general_packed_n73k'
    plt.figure(figsize=(8, 4))
    for name, cfg in pops.items():
        plt.plot(range(len(L16)), [leaf_stat(cfg, model, lf) for lf in L16], marker='o', lw=2,
                 ls='--' if 'null' in name else '-', label=name)
    plt.title(f'Excess over G by depth — {get_model_label(model)}')
    plt.xlabel('block (last = before_final_norm)'); plt.ylabel('excess_mass'); plt.legend(fontsize=8); plt.show()
    caption(f'RQ2 excess structure by depth, {get_model_label(model)}: for each task population '
            f'T, the generalized eigenvalues of T\'s covariance against the general population '
            f'G at the same leaf; excess_mass = Σ log λ over λ > 1 (how much variance T has '
            f'that G cannot account for, per stream depth). Dashed = held-out general splits '
            f'(the null: what a same-size resample of G itself scores). Math sits 2.3–2.4× '
            f'above its null mid-stack; the two "memorised" populations diverge — quotes carry '
            f'the most structure, verbatim-memorized peaks early (lexical novelty). '
            + ('OLMo\'s memorized population is Merullo et al\'s 650-seq set, packed all-token '
               '(padded/last impossible at N=650 < d) — compare it against the 73k N-matched '
               'packed null, not the padded curves: ~16× its null, peaking blk8–10.'
               if model.startswith('OLMo') else
               'Pythia\'s memorized set = EleutherAI pythia-memorized-evals (padded/last).'))

# %%
SH = np.load('data/results/rq2_splithalf.npy', allow_pickle=True).item()
for model in RQ2_MODELS:
    plt.figure(figsize=(8, 4))
    for pair in ('math_web', 'quotes', 'memorized', 'null_packed', 'null_padded'):
        ys = SH.get((model, pair, 'attn.in', 8))
        if ys is None: continue
        plt.plot(range(len(ys)), ys, marker='o', lw=2, ls='--' if 'null' in pair else '-', label=pair)
    plt.title(f'Split-half excess-subspace overlap (k=8) — {get_model_label(model)}')
    plt.xlabel('block (last = before_final_norm)'); plt.ylabel('overlap'); plt.ylim(0, 1)
    plt.legend(fontsize=8); plt.show()
    caption(f'Split-half coherence, {get_model_label(model)}: each population is split in two, '
            f'each half whitened independently against its own G reference, and the top-8 '
            f'excess subspaces compared (mean squared cosine overlap; 1 = identical subspace, '
            f'dashed nulls ≈ chance). High overlap = the two halves find the SAME excess '
            f'directions — shared task geometry. Math > quotes > memorized everywhere; '
            f'memorized coherence is lowest exactly at the early blocks where its excess mass '
            f'lives — consistent with independent per-item storage rather than a shared '
            f'"memorization subspace".')

# %% [markdown]
# ## 8. Side findings
# # ⚠️ "WRITE MEANS" PANELS: FLAGGED — NO VERIFIED FINDING, DO NOT CITE (Jul 13)
# The claim previously here ("mean anomalies split by family: Pythia's rogue variance-only,
# OLMo's late writes mean-heavy 0.15–0.6") is retracted as a misinterpretation: the panels
# themselves contradict its cross-model form (OLMo-1B's late writes have LOWER mean_frac
# than pythia's ordinary writes; the 0.15–0.6 range was OLMo-7B only), and the underlying
# "mean anomaly" terminology drifted between analyses. Panels are kept for the record;
# nothing mean_frac-related should be cited until re-derived from scratch.
# **Drift** panels are raw consecutive-checkpoint CKA — see the caption caveat below.

# %%
bs_out = lambda model, family: [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', family), f'mlp {l}')
                                for l in range(0, n_blocks[model], max(1, n_blocks[model] // 8))]
for model in PAIR:
    grid_start(ncols=2, title=f'Write means / drift — {get_model_label(model)}')
    plot_group('mean_frac', bs_out(model, 'acts_centered'), [model], color_palette='gradient', ylog=True)
    if not model.endswith('7B'):
        plot_group('cka_drift', bs_out(model, 'cka_drift'), [model], color_palette='gradient')
    grid_show()
    caption(f'⚠️ FLAGGED (see section note) — kept for the record, no verified finding. '
            f'What is plotted, {get_model_label(model)}: mean_frac = ‖μ‖² / trace(Σ_centered) '
            f'per mlp write (every ~L/8th block, light→dark = deeper) — the write\'s constant '
            f'per-token offset relative to its token-dependent variance. Right panel (1B '
            f'models only): checkpoint-to-checkpoint CKA drift of the same writes (its own '
            f'caveat below).')

drift_srcs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'cka_drift'), f'mlp {l}') for l in (0, 4, 8, 12, 15)]
for model in ('pythia-1b-deduped', 'OLMo-2-0425-1B'):
    grid_start(ncols=1, title=f'Checkpoint-to-checkpoint drift — {get_model_label(model)}')
    plot_group('cka_drift', drift_srcs, [model], color_palette='gradient')
    grid_show()
    caption(f'Representation drift, {get_model_label(model)}: CKA similarity between the same '
            f'mlp write at consecutive checkpoints (1 = static geometry; dips = '
            f'reorganization), blocks 0–15 light→dark. ⚠️ Interpretation caveat: consecutive '
            f'checkpoints are not equally spaced, so this raw view conflates drift speed '
            f'with checkpoint gap — the "two reorganization events bracketing the '
            f'entropy-seeking phase" reading is NOT confirmed under the gap-corrected rate '
            f'((1−CKA)/Δ), which places pythia\'s fastest drift mid-training and shows '
            f'OLMo-1B still accelerating late. Treat as a raw time series, not a verdict '
            f'(final_report appendix; re-derivation is on the plan).')

# %%
# mlp×attn coupling (moved out of §5 — a separate observation from the write-ensemble claim)
coupling_grid(PAIR, rows='mlp', cols='attn')
caption('Signed trace between mlp writes (rows) and attention writes (cols), final '
        'checkpoint, one panel per exemplar model. The near-diagonal negatives show MLPs '
        'partially CONSUMING their neighbouring attention outputs — the memory-management '
        'pattern of "An Adversarial Example for Direct Logit Attribution: Memory '
        'Management in GELU-4L"; cf. the CKA depth structure in "The Remarkable '
        'Robustness of LLMs: Stages of Inference?". Side observation, independent of the '
        'family mechanisms.')

# %% [markdown]
# ### Compression valleys: the depth-axis view (one object, three shadows)
# Stream RankMe per block at the final checkpoint. Pythia (and nanochat — no write-norm)
# crash to rank ~2 right after blk3's write enters and recover via the late-block
# cancellation; OLMo-2 (write-norm) has no valley at all. The massive activation (feature
# axis), the compression valley (depth axis, cf. arXiv:2510.06477), and the compression
# phase (training axis) are the same rogue direction seen three ways.

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
caption('The depth-axis view, two metrics over the same stream states (blk k attn.in per '
        'block, plus the pre-final-norm stream as the last point; final checkpoint; x = depth '
        'normalized to [0, 1]). Left: centered RankMe (log). Right: the attention-sinks paper\'s '
        'metric — Shannon entropy of the UNCENTERED singular-value distribution p_i = '
        'σ_i²/‖X‖_F², in bits — for direct numeric comparison with their layer profiles '
        '(their pythia valleys drop below ~0.5 bits). Protocol caveat for that comparison: '
        'they compute H per 1024-token GSM8K example and average over 7.5k examples; ours is '
        'the pooled 262k-token pile population covariance — shapes are comparable, absolute '
        'values only approximately. Dashed = families without write-norm (Pythia, nanochat): '
        'the valley right after blk3–4; solid = OLMo-2 (write-norm): no valley. The massive '
        'activation (feature axis), the valley (depth axis) and the compression phase '
        '(training axis) are one object seen three ways.')

# %% [markdown]
# Appendix material (the dataset-swap control, the toy edge-case maps) lives in its own
# notebook: [showcase_appendix.ipynb](showcase_appendix.ipynb).
