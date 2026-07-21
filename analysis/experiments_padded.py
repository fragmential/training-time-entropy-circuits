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
# # Padded twin of experiments.ipynb
#
# Same experiment grids as experiments.ipynb, run on the padded/last-token samples sweep
# (`block_representations_samples_padded`): 16,384 per-document last-token rows instead of
# the packed run's 262,144 token soup. Sections that need `block_representations_all` or
# other packed-only data have no padded counterpart and are dropped (noted inline).

# %%
# Re-run this cell to pick up experiments_lib edits without restarting the kernel
# (res_cache / hook_cache survive the reload via the lib's try/except guards).
import os, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
from analysis import experiments_lib as _lib
importlib.reload(_lib)
from analysis.experiments_lib import (
    grid_start, grid_show, plot_group, plot_spectrum, plot_layer_contribution,
    plot_heatmap, build_hooks, get_model_label, submatrix, plot_wo_topk,
)

# %%
FINEWEB_PADDED = "rankme_fineweb_padded"
TRAINSET_PACKED = "rankme_trainset_packed_4MT"
BLOCK_SAMPLES = "block_representations_samples_padded"   # padded/last-token raw-sample twin
PACKED_SAMPLES = "block_representations_samples"         # the packed original, for comparison

# %%
HK = build_hooks()   # hook-name -> (node, metric); machinery lives in experiments_lib

# %%
filter_model_names = [
    'pythia-1b-deduped',
    'pythia-6.9b-deduped',
    'OLMo-2-0425-1B',
    'OLMo-2-1124-7B',
]

data_sources_1 = [
#   Data File          Hook      Label
    (TRAINSET_PACKED,  HK.AFN_AC,   'Trainset centered'),
    (TRAINSET_PACKED,  HK.AFN_AU,   'Trainset uncentered'),
    (FINEWEB_PADDED,   HK.AFN_AC,   'Fineweb  centered'),
    (FINEWEB_PADDED,   HK.AFN_AU,   'Fineweb  uncentered'),
]

# %%
# Samples-run shorthands (mirroring experiments.py's Exp 4.x setup, padded source).
n_blocks_bs = {'pythia-1b-deduped': 16, 'pythia-6.9b-deduped': 32,
               'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32}
bs_out = lambda model, family: [(BLOCK_SAMPLES, (f'blk{l}.{s}.out', family), f'{s} {l}')
                                for l in range(n_blocks_bs[model]) for s in ('attn', 'mlp')]
bs_blk = lambda model: [(BLOCK_SAMPLES, (f'blk{l}', 'block_ledger'), f'blk {l}')
                        for l in range(n_blocks_bs[model])]
_pen = lambda model: f'blk{n_blocks_bs[model] - 1}.attn.in'   # stream entering the last block

# Sub-block pairings used by the heatmap grids.
PAIRINGS = (('', ''), ('attn', 'attn'), ('mlp', 'mlp'), ('attn', 'mlp'))


# %% [markdown]
# ### Padded vs packed: final-stream RankMe

# %%
# Two estimators of the final stream's effective rank over training, per model:
# LEFT = this padded run (16,384 last-token rows, one per document), RIGHT = the packed run
# (262,144 tokens, position soup). Same hook (after_final_norm, centered), matched y-ranges,
# tick labels kept on both panels.
import numpy as np
import matplotlib.pyplot as plt

def _final_rankme_compare():
    panels = ((BLOCK_SAMPLES, 'Padded: 16,384 last-token rows'),
              (PACKED_SAMPLES, 'Packed: 262,144 tokens'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (src, title) in zip(axes, panels):
        for model in filter_model_names:
            ys, steps = _lib.get_ys(src, model, ('after_final_norm', 'acts_centered'), 'rankme')
            xs = np.asarray(_lib.get_xs_tokens(model, steps), dtype=float)
            keep = xs > 0   # step 0 has zero tokens, unplottable on log-x
            ax.plot(xs[keep], np.asarray(ys)[keep], marker='o', ms=3, lw=2,
                    label=get_model_label(model))
        ax.set(xscale='log', title=title, xlabel='tokens')
        ax.set_ylabel('RankMe (after final norm, centered)')
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].legend(fontsize=9)
    fig.suptitle('Final-stream RankMe over training: padded (last-token) vs packed (all tokens)')
    fig.tight_layout()
    plt.show()

_final_rankme_compare()


# %% [markdown]
# ### Final residual RankMe

# %%

####        #        #        #        #        #        #        ####
####     "Tracing the representation geometry.." reproduction     ####
####        #        #        #        #        #        #        ####
grid_start(ncols=2, title='RankMe reproductions')
plot_group('rankme', data_sources_1, ['OLMo-2-0425-1B'], title='OLMo-2 1B', ls_exceptions={'uncentered':'--'})
plot_group('alpha', data_sources_1, ['OLMo-2-0425-1B'], title='OLMo-2 1B', ls_exceptions={'uncentered':'--'})

plot_group('rankme', data_sources_1, ['pythia-1b-deduped'], title='Pythia 1B Deduped', ls_exceptions={'uncentered':'--'})
plot_group('alpha', data_sources_1, ['pythia-1b-deduped'], title='Pythia 1B Deduped', ls_exceptions={'uncentered':'--'})
grid_show()


# %% [markdown]
# ### Dropped sections (no padded counterpart)
#
# - Block Means: needs block_representations_all, no padded counterpart.
# - Mean-Covariance relationship: needs block_representations_all, no padded counterpart.
# - Block-vs-residual mean-covariance: needs block_representations_all, no padded counterpart.
# - Layer contribution (gross energy): needs block_representations_all, no padded counterpart.
# - Per-layer-output vs final RankMe: needs block_representations_all, no padded counterpart.
# - Block mean vs residual mean (Exp 1.2): needs block_representations_all, no padded counterpart.
# - Mean-direction drift (Exp 1.3): needs block_representations_all, no padded counterpart.
# - Block-block mean alignment (Exp 1.1): needs block_representations_all, no padded counterpart.

# %% [markdown]
# ### Layer contribution (samples run)

# %%
# R_over_r layer contribution: each write's share of the FINAL stream's energy
# (tr Cov(c_k, r) / tr Σ_r — coupling to the rest of the stream included; the gap to 1.0 is
# the embedding stream's share). Signed by definition, positive in practice: r contains c_k,
# so a write must be cancelled by more than its own energy to go negative.
MODELS_LC = filter_model_names
NB_BS = n_blocks_bs
CFG_BS = lambda m: BLOCK_SAMPLES

_rr_srcs = lambda model: [(BLOCK_SAMPLES, (f'blk{l}.{s}.out', 'block_residual_coupling'), f'{s} {l}')
                          for l in range(n_blocks_bs[model]) for s in ('attn', 'mlp')]
grid_start(ncols=2, title='Layer contribution — share of final-stream energy (R over r)')
for model in filter_model_names:
    plot_layer_contribution(model, _rr_srcs(model), yvar='R_over_r', normalize=False,
                            title=get_model_label(model), remainder_label='embedding (residual)')
grid_show()

# %%
# Share of residual energy from just the traces (previously missing in the padded twin):
# uncentered write traces normalized over writes + the embedding stream; gray dashed = the
# embedding's share.
_tr_srcs = lambda model: [(BLOCK_SAMPLES, (f'blk{l}.{s}.out', 'acts_uncentered'), f'{s} {l}')
                          for l in range(n_blocks_bs[model]) for s in ('attn', 'mlp')]
grid_start(ncols=2, title='Layer contribution — share of residual energy (trace)')
for model in filter_model_names:
    plot_layer_contribution(model, _tr_srcs(model), yvar='trace', normalize=True,
                            title=get_model_label(model),
                            baseline_src=(BLOCK_SAMPLES, ('blk0.attn.in', 'acts_uncentered'), 'embedding'))
grid_show()

# %%
# Ledger-based layer contribution: each block's signed ΔS_k (= χ + quality + interference —
# overlap and cross terms INCLUDED, unlike the gross-energy stack above). Positives stack up,
# negatives down; the net stack height is log RankMe(final) − log RankMe(embeddings).
_ledger_srcs = lambda model: [(BLOCK_SAMPLES, (f'blk{l}', 'block_ledger'), f'blk {l}')
                              for l in range(n_blocks_bs[model])]
grid_start(ncols=2, title='Layer contribution — rank-entropy ledger (signed ΔS)')
for model in filter_model_names:
    plot_layer_contribution(model, _ledger_srcs(model), yvar='delta_s', normalize=False,
                            title=get_model_label(model))
grid_show()

# %% [markdown]
# #### Ledger contribution stacks (showcase implementation — sign-crossing fills)
# Dotted gray = the embedding stream's own entropy S_emb(t). NB vs the packed twin:
# pythia's final layers' ΔS flips sign here (final_report §2.3, conditional final layers).

# %%
def ledger_stack(model):
    import numpy as np
    import matplotlib.pyplot as plt
    cfg = CFG_BS(model)
    terms = {y: np.sum([_lib.get_ys(cfg, model, (f'blk{l}', 'block_ledger'), y)[0]
                        for l in range(NB_BS[model])], axis=0)
             for y in ('chi', 'quality', 'interference')}
    xs = np.asarray(_lib.get_xs_tokens(
        model, _lib.get_ys(cfg, model, ('blk0', 'block_ledger'), 'chi')[1]), float)
    keep = xs > 0
    xs, terms = xs[keep], {k: np.asarray(v, float)[keep] for k, v in terms.items()}
    # refine the grid with sign crossings so fills pinch to zero instead of twisting
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
    es, esteps = _lib.get_ys(cfg, model, ('blk0.attn.in', 'acts_centered'), 'matrix_entropy')
    if es is not None:
        exs = np.asarray(_lib.get_xs_tokens(model, esteps), float)
        ek = exs > 0
        plt.plot(exs[ek], np.asarray(es, float)[ek], color='0.4', ls=':', lw=1.5,
                 label='S(embedding stream)')
    plt.xscale('log'); plt.xlabel('tokens'); plt.ylabel('rank entropy')
    plt.legend(fontsize=8); plt.title(f'Ledger contributions — {get_model_label(model)}')
    plt.show()

for model in MODELS_LC:
    ledger_stack(model)

# %% [markdown]
# #### Per-block ledger terms over training (showcase formatting): ΔS, quality, overlap

# %%
def term_per_block(models, term):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, models):
        L = NB_BS[model]
        for l in range(L):
            ys, steps = _lib.get_ys(CFG_BS(model), model, (f'blk{l}', 'block_ledger'), term)
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
    axes[0, 0].set_ylabel(f'per-block {term}')
    axes[1, 0].set_ylabel(f'per-block {term}')
    fig.suptitle(f'{term} ledger term per block')
    plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
for _term in ('delta_s', 'quality', 'chi'):
    term_per_block(MODELS_LC, _term)


# %% [markdown]
# ### Samples run: block↔block coupling (Exp 4.1)

# %%
# Pairwise block-output coupling at the final checkpoint: CKA (shared subspace, unsigned),
# signed trace (net reinforce/cancel, energy-weighted), mean per-token cosine (democratic).
def _coupling_heatmaps(model, rows='', cols='', step=None):
    bb = lambda y: _lib.get_y(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), y, step)
    labels = [l.removeprefix('blk').removesuffix('.out') for l in bb('leaves')]
    grid_start(ncols=3, title=f'Block↔block coupling ({rows or "all"}×{cols or "all"}, final ckpt) — {get_model_label(model)}')
    for y, kw in (('cka', dict(vmin=0, vmax=1)), ('signed_trace', dict(dynamic=True)), ('mean_cos', dict(dynamic=True))):
        M, rl, cl = submatrix(bb(y), labels, rows, cols)
        plot_heatmap(M, labels=rl, labels2=cl, title=y, hide_diag=rows == cols, **kw)
    grid_show()

for model in filter_model_names:
    for rows, cols in PAIRINGS:
        _coupling_heatmaps(model, rows, cols)

# %% [markdown]
# ### Rank ledger (Exp 4.2)

# %%
# Exact per-block ledger of the residual's rank entropy (docs/ledger.md):
# ΔS_k = χ_k (overlap) + quality_k (spectral quality of the writes) + interference_k (pure
# cross-covariance effect); Σ_k ΔS_k telescopes to log RankMe(final) − log RankMe(emb).
def _ledger_grid(model):
    blks = bs_blk(model)
    grid_start(ncols=2, title=f'Rank ledger — {get_model_label(model)}')
    for y in ('delta_s', 'chi', 'quality', 'interference'):
        plot_group(y, blks, [model], color_palette='gradient', title=y)
    grid_show()

for model in filter_model_names:
    _ledger_grid(model)

# %% [markdown]
# ### Block↔final coupling over training (Exp 4.3)

# %%
# Signed/unsigned coupling of each block output against the final residual (centered crosses):
# signed share of the final trace, reinforce/cancel at the write point, CKA, mean cosine.
def _coupling_grid(model):
    outs = bs_out(model, 'block_residual_coupling')
    grid_start(ncols=2, title=f'Block↔final coupling — {get_model_label(model)}')
    for y in ('R_over_r', 'tr_P', 'cka_cr', 'cos_cr'):
        plot_group(y, outs, [model], color_palette='gradient', title=y)
    grid_show()

for model in filter_model_names:
    _coupling_grid(model)

# %% [markdown]
# ### Ablation, overlap, drift (Exp 4.4)

# %%
# Leave-one-out RankMe effect of each write, per-step overlap fraction χ_k/H(w), and
# checkpoint-to-checkpoint drift (CKA over shared tokens; absent at the first checkpoint).
DRIFT_MODELS = {'pythia-1b-deduped', 'OLMo-2-0425-1B'}   # drift_metrics ran on the 1Bs only

def _abl_drift_grid(model):
    grid_start(ncols=2, title=f'Ablation / overlap / drift — {get_model_label(model)}')
    plot_group('delta_rankme', bs_out(model, 'ablation_contribution'), [model], color_palette='gradient')
    plot_group('chi_frac', bs_out(model, 'incremental_overlap'), [model], color_palette='gradient')
    if model in DRIFT_MODELS:
        plot_group('cka_drift', bs_out(model, 'cka_drift'), [model], color_palette='gradient')
        plot_group('rankme', bs_out(model, 'geneig_drift'), [model], color_palette='gradient', title='gen-eig drift RankMe')
    grid_show()

for model in filter_model_names:
    _abl_drift_grid(model)

# %% [markdown]
# ### Head-removed final RankMe (Exp 4.5)

# %%
# Head-removed RankMe + windowed alphaReQ (compression propagates down-spectrum with decaying
# amplitude; docs/dig_findings.md), for the final stream AND the stream entering the last
# block — is the head phenomenon written by the last block?
# Deep band (512, -20): the compression front NEVER reaches it — it flattens monotonically
# through all of training (no tail fall-off either; band metrics are invariant to head growth).
# Padded caveat: N/d ~ 4-8 here, so MP broadening contaminates the deep tail; read the k=128+
# curves and the 128-512 / deep bands as trends only.
KS = (0, 1, 2, 8, 32, 128, 512)
WINDOWS = ((11, 100), (32, 128), (128, 512), (512, -20))
_tail = lambda leaf: [(BLOCK_SAMPLES, (leaf, 'acts_centered'), f'k={k}', ('tail_rankme', {'k': k})) for k in KS]
_alpha_w = lambda leaf: [(BLOCK_SAMPLES, (leaf, 'acts_centered'), f'{k0}-{k1}', ('alpha_window', {'k0': k0, 'k1': k1}))
                         for k0, k1 in WINDOWS]

# 'Without top-3' = additive leave-one-out estimate now; the exact joint_ablation
# series joins automatically once a run carries it (docs/next_run_additions.md #3).

for model in filter_model_names:
    grid_start(ncols=2, title=f'Head-removed RankMe / bulk alpha — {get_model_label(model)}')
    plot_group('tail_rankme', _tail('before_final_norm'), [model], color_palette='gradient', title='final stream')
    plot_group('tail_rankme', _tail(_pen(model)), [model], color_palette='gradient', title='stream before last block')
    plot_group('alpha_window', _alpha_w('before_final_norm'), [model], color_palette='gradient', title='final stream')
    plot_group('alpha_window', _alpha_w(_pen(model)), [model], color_palette='gradient', title='stream before last block')
    plot_wo_topk(BLOCK_SAMPLES, model, 'rankme')
    plot_wo_topk(BLOCK_SAMPLES, model, 'alpha')
    grid_show()

# %% [markdown]
# ### Sub-block ledger (Exp 4.6, OLMo only)

# %%
# Per-residual-write ledger: the block ledger's two sequential sub-steps (attn into stream,
# then mlp), assembled notebook-side from incremental_overlap's stored entropies via the
# sub_* virtual hooks. OLMo-only — Pythia's parallel sub-blocks have no sequential sub-steps.
def _sub_ledger_grid(model):
    subs = bs_out(model, 'incremental_overlap')
    grid_start(ncols=2, title=f'Sub-block ledger — {get_model_label(model)}')
    for y in ('sub_delta_s', 'chi', 'sub_quality', 'sub_interference'):
        plot_group(y, subs, [model], color_palette='gradient', title=y)
    grid_show()

for model in filter_model_names:
    if 'OLMo' in model:
        _sub_ledger_grid(model)

# %% [markdown]
# ### Write means vs their covariance (Exp 4.7)

# %%
# mean_frac = ||mu||²/tr (bias-like-ness of a write), top_overlap = |<mû, v1_centered>|,
# migration = centered-tail ↔ uncentered-top eigvec overlap. Pythia's rogue write is
# variance-like (mean_frac → 0.003); OLMo's late writes are mean-heavy (mean_frac 0.4–0.6);
# migration ≈ 0 everywhere (docs/dig_findings.md).
def _mean_grid(model):
    grid_start(ncols=2, title=f'Write means vs covariance — {get_model_label(model)}')
    plot_group('mean_frac', bs_out(model, 'acts_centered'), [model], color_palette='gradient', ylog=True)
    plot_group('top_overlap', bs_out(model, 'acts_mean_metrics'), [model], color_palette='gradient')
    plot_group('rayleigh_normed', bs_out(model, 'acts_mean_metrics'), [model], color_palette='gradient')
    plot_group('migration', bs_out(model, 'mean_migration'), [model], color_palette='gradient')
    grid_show()

for model in filter_model_names:
    _mean_grid(model)

# %% [markdown]
# ### Ledger contribution figure (Exp 4.8)

# %%
# The three ledger terms summed over blocks, sign-stacked over training: the books balance,
# so the black ΔS line IS the log-RankMe trajectory relative to the embeddings — no extra
# weighting (R_over_r would double-count the w_i already inside each term).
def _ledger_stack(model):
    import numpy as np
    import matplotlib.pyplot as plt
    terms = {y: np.sum([_lib.get_ys(BLOCK_SAMPLES, model, (f'blk{l}', 'block_ledger'), y)[0]
                        for l in range(n_blocks_bs[model])], axis=0)
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
    plt.legend(); plt.title(f'Ledger contributions — {get_model_label(model)}')
    plt.show()

for model in filter_model_names:
    _ledger_stack(model)

# %% [markdown]
# ### Rogue write anatomy (Exp 4.9, Pythia)

# %%
# blk3.mlp's write collapses to rank ~1 at the RankMe peak and then grows in energy —
# vs its neighbor writes. (The token-level scatter from the block_rogue_id raw samples is
# dropped here: that run is packed-only, no padded counterpart.)
def _rogue_traj(model):
    outs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'acts_centered'), f'mlp {l}') for l in (1, 3, 5)]
    grid_start(ncols=2, title=f'Rogue write trajectory — {get_model_label(model)}')
    plot_group('rankme', outs, [model], color_palette='gradient', ylog=True, title='write RankMe (centered)')
    plot_group('trace', outs, [model], color_palette='gradient', ylog=True, title='write trace')
    grid_show()

for model in filter_model_names:
    if 'pythia' in model:
        _rogue_traj(model)

# %% [markdown]
# ### Cancellation, head-targeting, and write concentration (Exp 4.10)

# %%
# (a) signed trace of each early write vs the LAST write over training — Pythia's late blocks
# learn to cancel the rogue (→ -0.65); OLMo's late writes stay mutually aligned.
# (b) eigendirection head-mass of the last write (share of |contrib| in the top-32 final
# directions) — Pythia's goes 0.26→0.92 (toy-like selection bias, H1.3).
# (c) write concentration: top-eigval share of each mlp write at the final ckpt — the
# Pythia-vs-OLMo contrast (rogue flag ~95-99% vs diffuse ~10%).
def _cancellation_headmass(model):
    import numpy as np
    import matplotlib.pyplot as plt
    L = n_blocks_bs[model]
    mats, steps = _lib.get_ys(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), 'signed_trace')
    leaves = _lib.get_y(BLOCK_SAMPLES, model, ('', 'block_block_coupling'), 'leaves', steps[0])
    xs = _lib.get_xs_tokens(model, steps)
    last = leaves.index(f'blk{L - 1}.mlp.out')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f'Cancellation / head-mass / concentration — {get_model_label(model)}')
    for l in (1, 3, 5, L // 2):
        i = leaves.index(f'blk{l}.mlp.out')
        axes[0].plot(xs, [m[i, last] for m in mats], marker='o', lw=2, label=f'mlp {l} ~ mlp {L-1}')
    axes[0].set(xscale='log', title='signed trace vs last write', xlabel='tokens'); axes[0].legend(fontsize=8)
    for l in (0, L // 2, L - 1):
        cs, ss = _lib.get_ys(BLOCK_SAMPLES, model, (f'blk{l}.mlp.out', 'eigendirection_attrib'), 'contrib')
        hm = [float(np.abs(c[:32]).sum() / np.abs(c).sum()) for c in cs]
        axes[1].plot(_lib.get_xs_tokens(model, ss), hm, marker='o', lw=2, label=f'mlp {l}')
    axes[1].set(xscale='log', ylim=(0, 1), title='head-mass (top-32 share)', xlabel='tokens'); axes[1].legend(fontsize=8)
    share = [_lib.get_y(BLOCK_SAMPLES, model, (f'blk{l}.mlp.out', 'acts_centered'), 'eigenspectrum', None)[0]
             for l in range(L)]
    axes[2].bar(range(L), share, color='tab:red' if 'pythia' in model else 'tab:blue')
    axes[2].set(ylim=(0, 1), title='top-eigval share per mlp write (final)', xlabel='block')
    plt.show()

for model in filter_model_names:
    _cancellation_headmass(model)


# %% [markdown]
# ### Dropped tail sections
#
# - K-FAC up/gate, K-FAC down, up/gate/down-proj inputs and outputs, sampled gradients,
#   whole-layer K-FAC, final-residual A/G covariance, down-proj alpha: all ride the packed
#   kfac_small_shuffled run, no padded counterpart.
