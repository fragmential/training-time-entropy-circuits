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
PAIR = ['pythia-1b-deduped', 'OLMo-2-1124-7B']          # one exemplar per family
# §3 covers the full model set: four pythia scales, both OLMo-2 scales, nanochat-d12
LEDGER_MODELS = ['pythia-160m-deduped', 'pythia-410m-deduped', 'pythia-1b-deduped',
                 'pythia-6.9b-deduped', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'nanochat-d12']
n_blocks = {'pythia-160m-deduped': 12, 'pythia-410m-deduped': 24,
            'pythia-1b-deduped': 16, 'pythia-6.9b-deduped': 32,
            'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32, 'nanochat-d12': 12}
cfg_of = lambda m: 'nanochat_samples' if m == 'nanochat-d12' else BLOCK_SAMPLES

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
    grid_start(ncols=2, title=f'Li et al reproduction — {get_model_label(model)}')
    plot_group('rankme', srcs, [model])
    plot_group('alpha', srcs, [model])
    grid_show()
    caption(f'RankMe (left) and power-law slope α (right) of the centered final-stream '
            f'covariance over training, {get_model_label(model)}, on the packed training mix '
            f'and (where collected) padded fineweb last-token — Li et al\'s setting. The '
            f'warmup → entropy-seeking → compression trajectory is the phase structure every '
            f'later figure decomposes. The small pythias are sanity checks for §3\'s 160m '
            f'edge case.')

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
KS, WINDOWS = (0, 8, 32, 128, 512), ((11, 100), (128, 512), (512, -20))
for model in PAIR:
    tails = [(BLOCK_SAMPLES, ('after_final_norm', 'acts_centered'), f'k={k}', ('tail_rankme', {'k': k})) for k in KS]
    alphas = [(BLOCK_SAMPLES, ('after_final_norm', 'acts_centered'), f'{a}-{b}', ('alpha_window', {'k0': a, 'k1': b})) for a, b in WINDOWS]
    grid_start(ncols=2, title=f'Head-removed RankMe / band alpha — {get_model_label(model)}')
    plot_group('tail_rankme', tails, [model], color_palette='gradient')
    plot_group('alpha_window', alphas, [model], color_palette='gradient')
    grid_show()
    caption(f'Left: RankMe with the top k eigendirections removed (k = 0…512, light→dark). '
            f'Right: α fit inside fixed rank windows. The decline survives k=8–32 but is '
            f'gone by k=512, and the deepest window moves the OPPOSITE way (flattening) '
            f'through all of training: a top-of-spectrum concentration front, not a global '
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
caption('The tilt baseline (dashed): a power-law spectrum whose slope steepens over time, '
        'calibrated to match the measured RankMe at the peak and at the minimum. Under it '
        'every band falls in lockstep and one α describes every rank. Measured instead: the '
        'deep band RISES through the compression phase and the band slopes fan out — real '
        'compression is a head event, not a uniform tilt.')

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

# %%
def ledger_stack(model):
    cfg = cfg_of(model)
    terms = {y: np.sum([_lib.get_ys(cfg, model, (f'blk{l}', 'block_ledger'), y)[0]
                        for l in range(n_blocks[model])], axis=0)
             for y in ('chi', 'quality', 'interference')}
    xs = np.asarray(_lib.get_xs_tokens(
        model, _lib.get_ys(cfg, model, ('blk0', 'block_ledger'), 'chi')[1]), float)
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
    caption(f'The decomposition summed over blocks, {get_model_label(model)}: ΔS = χ (green) '
            f'+ quality (red) + interference (purple); black = the total, the '
            f'depth-differential log RankMe (stack output entropy MINUS input entropy per '
            f'checkpoint — negative means output below input, not negative entropy). '
            f'Positive parts stack up from zero, negative down. The 7-model split and its '
            f'edge cases: section header.')

for model in LEDGER_MODELS:
    ledger_stack(model)

# %% [markdown]
# ### Layer decomposition of the quality term
# The "locus" row of the ledger claim, shown rather than asserted: one line per block's own
# quality contribution over training. Numbers table: dig_findings "Per-block quality
# decomposition".

# %%
def quality_per_block(models):
    from matplotlib.cm import ScalarMappable
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, models):
        L = n_blocks[model]
        for l in range(L):
            ys, steps = _lib.get_ys(cfg_of(model), model, (f'blk{l}', 'block_ledger'), 'quality')
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
    axes[0, 0].set_ylabel('per-block quality term')
    axes[1, 0].set_ylabel('per-block quality term')
    fig.suptitle('Quality ledger term per block')
    plt.tight_layout(); plt.show()

quality_per_block(LEDGER_MODELS)
caption('One line per block\'s own quality contribution (colorbar = block index); the §3 '
        'stacks are the sums of these lines. At pythia-1b/6.9b one early block (blk3) '
        'carries essentially the whole collapse; at 410m it is blk5; at 160m and nanochat '
        'it splits between an early and the FINAL block. In OLMo-2 no block dominates. '
        'Full table: dig_findings.')

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
    grid_start(ncols=2, title=f'Sink write trajectory — {get_model_label(model)}')
    plot_group('rankme', outs, [model], color_palette='gradient', ylog=True, title='write RankMe (centered)')
    plot_group('trace', outs, [model], color_palette='gradient', ylog=True, title='write trace')
    grid_show()
    caption(f'RankMe (left) and trace (right) of the centered mlp-output covariance for the '
            f'first half of the writes, {get_model_label(model)} (light→dark = deeper; the '
            f'late half is omitted — late writes are legitimately large). Exactly one early '
            f'write departs from the family: blk3 collapses to rank ≈ 1 at the stream '
            f'RankMe peak while its trace keeps growing.')

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
    caption(f'Left: signed-trace coupling of early/mid writes against the LAST write '
            f'(negative = cancellation). Middle: share of each write\'s attribution mass in '
            f'its top 32 directions. Right: top-eigenvalue share per write at the final '
            f'checkpoint — Pythia\'s bar near 1.0 at blk3 IS the sink write; OLMo-2 has '
            f'none above ~0.1.')

for model in ALL4:
    cancellation_headmass(model)

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


# %%

coupling_grid(ALL4)
caption('Signed trace between every pair of mlp writes, final checkpoint (red = '
        'reinforcing, blue = cancelling; diagonal removed). Both OLMo-2 panels show the '
        'late-block red square — every late write aligned with every other; both Pythia '
        'panels instead show a blue band through the sink-write block. This is '
        '"distributed aligned reinforcement", operationally.')


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
    caption(f'Left pair (shared token axis), {get_model_label(model)}: late-write alignment '
            f'(red) above the total interference term (purple). Right: per block, coupling '
            f'to the late writes vs its own interference term, averaged over the final '
            f'third of checkpoints. '
            + ('The onsets coincide and the compressing blocks are exactly the '
               'most-coupled ones (down-right): the aligned ensemble IS the compressor.'
               if 'OLMo' in model else
               'No positive alignment appears, and the negative-interference blocks sit at '
               'NEGATIVE coupling: Pythia\'s interference is the sink-write cancellation, '
               'not reinforcement.'))

for model in ALL4:
    alignment_vs_interference(model)

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
# figures that used to render here are removed; the redo plan is docs/plan.md item 1, the
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
#   gap-corrected re-derivation is planned (docs/plan.md item 4).

# %%
# mlp×attn coupling (moved out of §5 — a separate observation from the write-ensemble claim)
coupling_grid(PAIR, rows='mlp', cols='attn')
caption('Signed trace between mlp writes (rows) and attention writes (cols), final '
        'checkpoint. The near-diagonal negatives show MLPs partially consuming their '
        'neighbouring attention outputs — the "memory management" pattern of the GELU-4L '
        'literature. Side observation, independent of the family mechanisms.')

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
