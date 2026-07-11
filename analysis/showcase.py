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

# %% [markdown]
# ### What Li et al's mechanism WOULD do to the bands (synthetic reference)
# Their theorem: eigenvalue growth proportional to size (selection bias, dσ_i ∝ σ_i with the
# dominant directions winning). Simulated as d log λ_i ∝ λ_i/Σλ from pythia-1b's MEASURED
# spectrum at its RankMe peak: under that dynamic the tilt is spectrum-wide — every band's α
# rises and every band-RankMe falls together. The measured deep band does the OPPOSITE
# (α falls / band-RankMe rises through the compression phase) — that contrast, not the
# headline curve, is what confronts the proposed mechanism below rank ~100.

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
    axes[0].set(yscale='log', xlabel='synthetic step', title='RankMes: ALL fall'); axes[0].legend(fontsize=8)
    for k in ('α 32-128', 'α 512-end'):
        axes[1].plot(traj[k], lw=2, label=k)
    axes[1].set(xlabel='synthetic step', title='band α: ALL rise (measured 512+ falls)'); axes[1].legend(fontsize=8)
    plt.show()

selection_bias_reference()

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
    xs = _lib.get_xs_tokens(model, _lib.get_ys(BLOCK_SAMPLES, model, ('blk0', 'block_ledger'), 'chi')[1])
    plt.figure(figsize=(8, 4))
    pos, neg = np.zeros(len(xs)), np.zeros(len(xs))
    for name, c in (('chi', 'tab:green'), ('quality', 'tab:red'), ('interference', 'tab:purple')):
        v, base = terms[name], np.where(terms[name] >= 0, pos, neg)
        plt.fill_between(xs, base, base + v, label=name, color=c, alpha=0.55)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    plt.plot(xs, sum(terms.values()), 'k', lw=2.5, label='ΔS total')
    plt.xscale('log'); plt.xlabel('tokens'); plt.ylabel('rank entropy')
    plt.legend(); plt.title(f'Ledger contributions — {get_model_label(model)}'); plt.show()

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
    mean_abs = torch.zeros(len(uniq)).index_add_(0, inv, score.abs()) / counts
    keep = counts >= 20
    order = torch.argsort(mean_abs.masked_fill(~keep, -1), descending=True)[:top].tolist()
    names = [repr(t) for t in tok.batch_decode([[int(uniq[i])] for i in order])]
    vals = [float(mean_abs[i]) for i in order]
    ns = [int(counts[i]) for i in order]
    colors = ['tab:red' if '\n' in n else 'tab:gray' for n in names]
    plt.figure(figsize=(9, 5))
    plt.barh(range(top)[::-1], vals, color=colors)
    plt.yticks(range(top)[::-1], [f'{n}  (n={c})' for n, c in zip(names, ns)], fontsize=7)
    plt.xlabel('mean |projection onto rogue direction|')
    plt.title(f'Token types ranked by rogue-direction activation — {get_model_label(model)}\n(red = newline-bearing; small n = noisy)')
    plt.show()

rogue_ranking('pythia-1b-deduped')

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

for model in ALL4:
    coupling_views(model)

# %% [markdown]
# ## 6. The toy bridge (RQ3)
# **Findings:** the paper's toy reproduces (phases + all four negative controls) through this
# repo's unmodified metric pipeline; the residual stream is constitutive (no-residual stack
# loses the phases entirely); the residual toy reproduces Pythia's quality-driven ledger
# signature; read-norm suppresses the toy's rogue (a toy/LLM discrepancy — real Pythia has
# read-norm and a rogue anyway); and WRITE-norm (the true OLMo-2 reordered-norm analog)
# suppresses the rogue mechanism COMPLETELY (6/6 seeds, both wirings; min write RankMe
# 5.6–7.5 vs 1.7–2.6 un-normed; 4/6 runs lose the post-peak decline entirely), while
# interference never becomes the carrier in any of the 24 grid runs. Figures: saved toy
# outputs + the architecture-knob grid.

# %%
# For comparison, the ORIGINAL Li et al Fig 4 (top row): (A) their model schematic,
# (B) classifier weight-row trajectories in 2D, (C) feature trajectories, (D) RankMe and
# σ1, σ2 over training — the three phases as dotted/solid/dashed trajectory segments.
lifig = 'reference/li et al reference tex/figures/Fig4_top.png'
if os.path.exists(lifig):
    plt.figure(figsize=(12, 5)); plt.imshow(mpimg.imread(lifig)); plt.axis('off')
    plt.title('Li et al, Fig 4 (original)'); plt.show()

# %%
# Our reproduction + extensions. How to read each figure:
# - fig4_single: same panel layout as their Fig 4 B–D — left: W row trajectories (each line =
#   one class's classifier weight vector moving in 2D over training), middle: feature vectors,
#   right: RankMe + σ1/σ2 curves. Compare shapes against the original above.
# - fig_controls: RankMe curves for the four negative controls — each rises monotonically
#   (no compression), matching the paper's supplementary claim.
# - fig_multi_*: multi-layer toys — stream RankMe over training + the ledger terms; the
#   residual variant shows the Pythia-like quality-carried signature, the plain (no-residual)
#   stack collapses monotonically (phases lost).
for f, cap in (('fig4_single.png', 'Reproduction of Fig 4 B–D (single-layer toy)'),
               ('fig_controls.png', 'Negative controls: each removes compression'),
               ('fig_multi_multi_residual_nonlinear.png', 'Residual multi-layer toy: stream RankMe + ledger'),
               ('fig_multi_multi_plain.png', 'No-residual stack: phases lost entirely'),
               ('fig_arch_grid.png', 'Architecture knobs: write-norm kills the rogue/quality mechanism; no knob produces interference-carried compression')):
    p = f'toy/figures/{f}'
    if os.path.exists(p):
        plt.figure(figsize=(11, 6)); plt.imshow(mpimg.imread(p)); plt.axis('off'); plt.title(cap); plt.show()

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

drift_srcs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'cka_drift'), f'mlp {l}') for l in (0, 4, 8, 12, 15)]
for model in ('pythia-1b-deduped', 'OLMo-2-0425-1B'):
    grid_start(ncols=1, title=f'Checkpoint-to-checkpoint drift — {get_model_label(model)}')
    plot_group('cka_drift', drift_srcs, [model], color_palette='gradient')
    grid_show()

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
