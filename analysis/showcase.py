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
#     display_name: representation-geometry
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
    display(HTML(f'<div style="font-size:0.88em; color:#555; max-width:56em; '
                 f'margin:0.2em 0 1.2em 0.5em"><b>Figure.</b> {txt}</div>'))

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
KS, WINDOWS = (0, 1, 8, 32, 128, 512), ((11, 100), (32, 128), (128, 512), (512, -20))
for model in PAIR:
    tails = [(BLOCK_SAMPLES, ('before_final_norm', 'acts_centered'), f'k={k}', ('tail_rankme', {'k': k})) for k in KS]
    alphas = [(BLOCK_SAMPLES, ('before_final_norm', 'acts_centered'), f'{a}-{b}', ('alpha_window', {'k0': a, 'k1': b})) for a, b in WINDOWS]
    grid_start(ncols=2, title=f'Head-removed RankMe / band alpha — {get_model_label(model)}')
    plot_group('tail_rankme', tails, [model], color_palette='gradient')
    plot_group('alpha_window', alphas, [model], color_palette='gradient')
    grid_show()
    caption(f'Spectral locality of the phases, {get_model_label(model)}, centered pre-final-norm '
            f'stream. Left: RankMe of the spectrum with the top k eigendirections removed '
            f'(k = 0…512, light→dark). Right: power-law slope α fit inside fixed rank windows. '
            f'The compression decline survives k=1–32 but is gone by k=512; the deepest window '
            f'(512–end) moves in the OPPOSITE direction (α falls = flattening) throughout '
            f'training. The compression is a top-of-spectrum concentration front, not a global '
            f'contraction.')

# %% [markdown]
# ### What a selection-bias dynamic actually does to the bands (synthetic reference)
# Li et al's mechanism (σ̇_i ∝ σ_i, dominant directions winning — their Prop. in the xent
# supplement), implemented in its most aggressive multiplicative form d log λ_i ∝ λ_i/Σλ and
# run on pythia-1b's MEASURED spectrum at its RankMe peak. What it actually produces: the
# HEAD collapses (full RankMe 555 → 1) while every deep-band statistic is essentially FROZEN
# (tail-RankMe k=512 moves 0.08%, band-α 32–128 moves 3.8% — multiplicative selection is
# scale-invariant below the head, and once the head dominates, relative growth in the tail
# vanishes; no η/T tuning changes this). Contrast with the measured bands above: the real
# deep band is NOT frozen — its α falls / band-RankMe rises through the compression phase.
# So a selection-bias dynamic reproduces the headline RankMe collapse but predicts an inert
# deep spectrum; it cannot account for the measured deep-band motion in either direction.

# %%
from analysis.experiments_lib import _rankme as _rm, _alpha as _al

def selection_bias_reference(model='pythia-1b-deduped', T=60, eta=6.0):
    ys, steps = _lib.get_ys(BLOCK_SAMPLES, model, ('before_final_norm', 'acts_centered'), 'rankme')
    peak = steps[int(np.argmax(ys[2:])) + 2]                       # skip the init transient
    lam = np.asarray(_lib.get_y(BLOCK_SAMPLES, model, ('before_final_norm', 'acts_centered'),
                                'eigenspectrum', peak), float)
    stats = {'RankMe (k=0)': lambda l: _rm(l), 'tail RankMe k=32': lambda l: _rm(l[32:]),
             'tail RankMe k=512': lambda l: _rm(l[512:]),
             'α 32-128': lambda l: _al(l, 32, 128), 'α 512-end': lambda l: _al(l, 512, -20)}
    traj = {k: [] for k in stats}
    for _ in range(T):
        for k, f in stats.items(): traj[k].append(f(lam))
        lam = lam * np.exp(eta * lam / lam.sum()); lam /= lam.sum()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Synthetic selection-bias dynamic from {get_model_label(model)} peak spectrum')
    for k in ('RankMe (k=0)', 'tail RankMe k=32', 'tail RankMe k=512'):
        axes[0].plot(traj[k], lw=2, label=k)
    axes[0].set(yscale='log', xlabel='synthetic step',
                title='full RankMe collapses; tail RankMes frozen'); axes[0].legend(fontsize=8)
    for k in ('α 32-128', 'α 512-end'):
        axes[1].plot(traj[k], lw=2, label=k)
    axes[1].set(xlabel='synthetic step',
                title='band α: frozen (measured: 32-128 rises, 512+ falls)'); axes[1].legend(fontsize=8)
    plt.show()

selection_bias_reference()
caption('Synthetic reference: the multiplicative selection-bias dynamic d log λ_i ∝ λ_i/Σλ '
        'iterated from pythia-1b\'s measured peak spectrum (2048 eigvals, 60 steps, η=6). '
        'Left (log y): full RankMe collapses to 1 as the top eigenvalue absorbs the spectrum, '
        'but the head-removed tail RankMes barely move (k=32: −1.3%; k=512: −0.08%). Right: '
        'band α is likewise near-constant (32–128: +3.8%; 512–end: +0.3%). The dynamic '
        'predicts compression as a pure head phenomenon with a frozen deep spectrum — the '
        'measured deep band (previous figure) moves instead, in the anti-compression '
        'direction. Note these curves being flat IS the result, not a plotting artifact.')

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
# ## 4. Pythia's rogue write: a newline-activated direction
# **Definition:** the "rogue direction" = the top eigenvector v1 of the CENTERED covariance of
# blk3.mlp's output (final ckpt) — the axis of maximal variance of that write. It earns a name
# because it alone carries 94–99% of the write's variance (that is what "rank ≈ 1 write" means).
# "Projection onto it" = (x − μ)·v1 per token row.
# **Precise claim (and no more):** that direction's large activations occur overwhelmingly at newline-token positions
# (38/40 top rows; mean |projection| 98 on newline rows vs 10.5 elsewhere). Final-checkpoint,
# correlational: it does NOT establish what the direction is FOR, only where it activates —
# and a heavy tail of non-newline spikes exists too. The write collapses to rank ≈ 1 at the
# RankMe peak, grows to ~22× the stream's energy, and the late blocks increasingly write
# against it (signed trace → −0.65). OLMo-2 shows none of this (top-eigval share ~0.01 vs 0.99).
# Systematic all-token check (mean |projection| per token TYPE, n>=20): newline variants top
# the ranking with by far the largest counts ('\n' 99.3 @ n=11.6k, '\n\t', '\n\n'); a noisy
# tail of low-count word types reaches 70-95 (n≈20-26, small-sample). Preimage check: the
# input direction that best predicts the score is UNALIGNED with the raw '\n' embedding
# (cos −0.013) and only weakly separates newline rows at the input — the spike is constructed
# by the MLP (nonlinearly / from processed features), not echoed from the token embedding.

# %%
for model in ('pythia-1b-deduped',):
    outs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'acts_centered'), f'mlp {l}') for l in (1, 3, 5)]
    grid_start(ncols=2, title=f'Rogue write trajectory — {get_model_label(model)}')
    plot_group('rankme', outs, [model], color_palette='gradient', ylog=True, title='write RankMe (centered)')
    plot_group('trace', outs, [model], color_palette='gradient', ylog=True, title='write trace')
    grid_show()
    caption(f'The rogue write forming, {get_model_label(model)}: RankMe (left) and total '
            f'variance/trace (right) of the centered mlp-output covariance for blocks 1, 3, 5 '
            f'(light→dark). blk3\'s write collapses to rank ≈ 1 at the stream RankMe peak '
            f'while its trace keeps growing to ~22× the stream\'s energy — one huge direction, '
            f'not a quiet write. Neighbouring writes do neither.')

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
    plt.figure(figsize=(7, 4))
    for m, lbl, c in ((~is_nl, 'other tokens', 'tab:gray'), (is_nl, 'newline tokens', 'tab:red')):
        plt.hist(score[m].abs().numpy(), bins=120, log=True, alpha=0.6, color=c, label=lbl)
    plt.xlabel('|projection onto rogue direction|'); plt.ylabel('token count (log)')
    plt.title(f'Who carries the rogue direction — {get_model_label(model)}'); plt.legend(); plt.show()

rogue_hist('pythia-1b-deduped')
caption('Token attribution of the rogue direction (pythia-1b, final checkpoint, raw per-token '
        'activations of blk3.mlp.out): histogram of |(x − μ)·v₁| — the projection of each '
        'centered token row onto the write\'s top centered eigendirection — split into '
        'newline-bearing tokens (red) vs all others (gray); log count axis. The separated red '
        'tail is the attribution evidence: large rogue activations occur overwhelmingly at '
        'newline positions. Correlational and final-checkpoint only — it locates WHERE the '
        'direction activates, not what it is for.')

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
    plt.errorbar(vals, ypos, xerr=[ci, [0] * top], fmt='none', ecolor='k', lw=1, capsize=2)
    plt.yticks(ypos, [f'{t!r}  (n={c})' for t, c in zip(decoded, ns)], fontsize=7)
    plt.xlabel('mean |projection onto rogue direction|')
    plt.title(f'Token types ranked by rogue-direction activation — {get_model_label(model)}\n(red = newline-bearing; small n = noisy)')
    plt.show()

rogue_ranking('pythia-1b-deduped')
caption('Systematic version of the previous figure: mean |projection onto the rogue direction| '
        'per token TYPE, all types with n ≥ 20 occurrences, top 20 shown. Red bars = the '
        'decoded token contains a newline; whiskers mark the lower edge of the 95% CI of the '
        'mean (1.96·SD/√n), so low-count word types whose whisker reaches far left are '
        'noise-compatible with the bulk. Newline variants top the ranking with by far the '
        'largest counts.')

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
# Two views of the signed-trace coupling matrix: mlp×mlp (the write-ensemble structure: OLMo's
# aligned late block, Pythia's blk3 anti-diagonal) and mlp×attn — the near-diagonal negatives
# show MLPs partially CONSUMING their neighboring attention outputs, cf. "An Adversarial
# Example for Direct Logit Attribution: Memory Management in GELU-4L" and the CKA structure in
# "The Remarkable Robustness of LLMs: Stages of Inference?".
from analysis.experiments_lib import submatrix

def coupling_views(model, step=None):
    bb = lambda y: _lib.get_y(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), y, step)
    st, leaves = bb('signed_trace'), bb('leaves')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Signed-trace coupling (final ckpt) — {get_model_label(model)}')
    for ax, (rows, cols, title) in zip(axes, (('mlp', 'mlp', 'mlp × mlp'), ('mlp', 'attn', 'mlp × attn'))):
        M, rl, cl = submatrix(st, leaves, rows=rows, cols=cols)
        if rows == cols: M = M - np.diag(np.diag(M))
        v = np.abs(M).max()
        im = ax.imshow(M, cmap='coolwarm', vmin=-v, vmax=v)
        ax.set(title=title, xlabel=cols + ' block', ylabel=rows + ' block')
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.show()
    caption(f'Signed-trace coupling between block outputs at the final checkpoint, '
            f'{get_model_label(model)} (red = aligned/reinforcing, blue = opposed/cancelling; '
            f'diagonal removed on the left). Left, mlp × mlp: OLMo-2\'s late writes form an '
            f'aligned (red) block — distributed reinforcement; Pythia shows the blk3 '
            f'anti-band — late writes cancel the rogue. Right, mlp × attn: the near-diagonal '
            f'negatives show MLPs partially consuming their neighbouring attention outputs '
            f'(cf. the memory-management picture in "An Adversarial Example for Direct Logit '
            f'Attribution").')

for model in ALL4:
    coupling_views(model)

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
# **Reading note — centered vs uncentered.** All LLM figures in this notebook use CENTERED
# spectra unless labeled otherwise. The Li-facing toy figures below (Fig 4 reproduction,
# controls, multi-layer stream panels) use UNCENTERED RankMe, matching Li et al's definition
# on raw features; only the architecture-grid figure (and its rogue check) uses centered
# spectra, to match the LLM-side pipeline it is compared against — its axes say so.

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
        'd=3 = |V|−1; MSE loss on both count patterns), plus "single" = the skew+bottleneck+CE '
        'reference that DOES compress. RankMe is plotted as a fraction of the feature '
        'dimension d so the d=3 no-bottleneck run is comparable to the d=2 runs. Only the '
        'reference declines after its peak; every control is monotone — each removed '
        'ingredient (skew, bottleneck, CE) is necessary for the compression phase.',
    'fig_multi_multi_residual_nonlinear.png':
        'Residual multi-layer toy (6 blocks, nonlinear writes, d=16, 32 classes, skewed '
        'counts). Left: uncentered stream RankMe at every depth (viridis light→dark = deeper), '
        'i.e. blk k input, last curve = final stream. Middle: the rank ledger summed over '
        'blocks — ΔS with its χ/quality/interference decomposition, the same object as the '
        'LLM ledger figures in §3; the decline is quality-carried, reproducing Pythia\'s '
        'signature. Right: per-write energy shares w_k (guard: no single write dominates the '
        'ledger, so the summed terms are representative).',
    'fig_multi_multi_plain.png':
        'The same multi-layer toy WITHOUT the residual stream (each block replaces rather '
        'than adds). Same three panels as above. The phase structure is gone — stream RankMe '
        'collapses monotonically at every depth — showing the residual stream is constitutive '
        'for the warmup/expansion/compression trajectory, not incidental.',
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
# ### The dataset-swap control (data vs model)
# Each 1B model re-run on the OTHER family's pretraining mix (same texts, re-tokenized) —
# if the family split followed the data, these curves would swap; they don't. Σquality
# trajectories, original (solid) vs swapped data (dashed): pythia stays quality-collapsing,
# OLMo stays quality-recovering, on either corpus. Full verdict table:
# docs/swap_run_findings.md.

# %%
SWAP = 'block_representations_samples_swap'
n_blocks_swap = {'pythia-1b-deduped': 16, 'OLMo-2-0425-1B': 16}

def swap_quality(model):
    out = {}
    for src, lbl in ((BLOCK_SAMPLES, 'original data'), (SWAP, 'swapped data')):
        q = np.sum([_lib.get_ys(src, model, (f'blk{l}', 'block_ledger'), 'quality')[0]
                    for l in range(n_blocks_swap[model])], axis=0)
        steps = _lib.get_ys(src, model, ('blk0', 'block_ledger'), 'chi')[1]
        out[lbl] = (steps, q)
    return out

plt.figure(figsize=(9, 4.5))
for model, c in (('pythia-1b-deduped', 'tab:red'), ('OLMo-2-0425-1B', 'tab:blue')):
    for lbl, (st, q) in swap_quality(model).items():
        plt.plot(st, q, color=c, lw=2, ls='--' if 'swap' in lbl else '-',
                 label=f'{get_model_label(model)}, {lbl}')
plt.xscale('log'); plt.xlabel('step'); plt.ylabel('Σ quality (over blocks)')
plt.title('Dataset-swap control: the ledger driver follows the MODEL, not the data')
plt.legend(fontsize=8); plt.show()
caption('Dataset-swap control: summed quality ledger term for each 1B model on its own '
        'pretraining mix (solid) vs re-run on the OTHER family\'s mix, same texts re-tokenized '
        '(dashed). If the family split (Pythia quality-collapsing, red, vs OLMo-2 '
        'quality-recovering, blue) followed the data, the dashed curves would swap sides; '
        'they track their solid counterparts instead — the ledger signature is a property of '
        'the model/architecture, not the corpus.')

# %% [markdown]
# ## 7. RQ2: task geometry beyond the general population
# **Findings:** maths text carries real shared structure (excess 2.3–2.4× the split-half null,
# both families), located mid-spectrum (not the predicted tail). The two "memorised"
# populations split: verbatim strings ≈ lexical early-layer novelty; famous quotes = the most
# structured population measured. Split-half coherence (independent whiteners, raw-space
# subspaces): math > quotes > memorized everywhere, and memorized coherence is LOWEST at the
# early blocks where its excess lives — independent per-item storage where storage happens.

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
    plt.figure(figsize=(8, 4))
    for name, cfg in pops.items():
        if model.startswith('OLMo') and cfg == 'rq2_memorized': continue
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
            f'the most structure, verbatim-memorized peaks early (lexical novelty).')

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
# **Mean anomalies split by family the same way** (Pythia's rogue is variance-only,
# mean_frac → 0.003; OLMo's late writes are mean-heavy, 0.15–0.6) — and **drift** shows two
# reorganization events bracketing the entropy-seeking phase (violent early: OLMo-1B CKA-drift
# bottoms at 0.12; a second dip at the RankMe peak).

# %%
bs_out = lambda model, family: [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', family), f'mlp {l}')
                                for l in range(0, n_blocks[model], max(1, n_blocks[model] // 8))]
for model in PAIR:
    grid_start(ncols=2, title=f'Write means / drift — {get_model_label(model)}')
    plot_group('mean_frac', bs_out(model, 'acts_centered'), [model], color_palette='gradient', ylog=True)
    if not model.endswith('7B'):
        plot_group('cka_drift', bs_out(model, 'cka_drift'), [model], color_palette='gradient')
    grid_show()
    caption(f'Write means, {get_model_label(model)}: mean_frac = ‖μ‖² / (‖μ‖² + trace) per mlp '
            f'write (every ~L/8th block, light→dark = deeper) — how much of the write\'s '
            f'energy is a constant offset vs variance. Pythia\'s rogue is variance-only '
            f'(mean_frac → 0.003 at blk3); OLMo-2\'s late writes are mean-heavy (0.15–0.6) — '
            f'the same family split as the ledger. Right panel (1B models only): '
            f'checkpoint-to-checkpoint CKA drift of the same writes.')

drift_srcs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'cka_drift'), f'mlp {l}') for l in (0, 4, 8, 12, 15)]
for model in ('pythia-1b-deduped', 'OLMo-2-0425-1B'):
    grid_start(ncols=1, title=f'Checkpoint-to-checkpoint drift — {get_model_label(model)}')
    plot_group('cka_drift', drift_srcs, [model], color_palette='gradient')
    grid_show()
    caption(f'Representation drift, {get_model_label(model)}: CKA similarity between the same '
            f'mlp write at consecutive checkpoints (1 = static geometry; dips = '
            f'reorganization), blocks 0–15 light→dark. Two reorganization events bracket the '
            f'entropy-seeking phase: a violent early one (OLMo-1B bottoms at CKA 0.12) and a '
            f'second dip at the RankMe peak where the compression phase begins.')

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
plt.figure(figsize=(9, 5))
for (cfg, model), L in VALLEY.items():
    res, steps = load_results(cfg, model)
    leaves = [f'blk{k}.attn.in' for k in range(L)] + ['before_final_norm']
    ys = [res[steps[-1]].get(lf, {}).get('acts_centered', {}).get('rankme', np.nan) for lf in leaves]
    plt.semilogy(np.linspace(0, 1, len(ys)), ys, marker='o', lw=2,
                 ls='-' if 'OLMo' in model else '--', label=get_model_label(model))
plt.xlabel('relative depth'); plt.ylabel('stream RankMe (centered, log)')
plt.title('Compression valleys at the final checkpoint — dashed = no write-norm')
plt.legend(fontsize=8); plt.show()
caption('The depth-axis view: RankMe of the centered residual stream at every block INPUT '
        '(blk k attn.in, plus the pre-final-norm stream as the last point), final checkpoint, '
        'x = depth normalized to [0, 1]. Dashed = families without write-norm (Pythia, '
        'nanochat): the stream crashes to rank ~2 right after blk3\'s rogue write enters and '
        'recovers via late-block cancellation — the "compression valley". Solid = OLMo-2 '
        '(write-norm): no valley. The massive activation (feature axis), the valley (depth '
        'axis) and the compression phase (training axis) are one object seen three ways.')
