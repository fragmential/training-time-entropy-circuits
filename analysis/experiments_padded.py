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
SWAP_PADDED = "block_representations_samples_swap_padded"   # padded swap (cross-tokenization)
SWAP_PACKED = "block_representations_samples_swap"          # packed swap (1B models only)
FINEWEB_PACKED = "block_representations_fineweb_packed"     # packed fineweb (bfn+afn)

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
# ### Padded vs packed: final-stream RankMe (native / swap / fineweb)

# %%
# Final-stream effective rank (RankMe) over training, padded (last-token, one row per document)
# vs packed (all tokens), across three datasets:
#   native  — each model on its own pretraining data (block_representations_samples[_padded]).
#   swap    — each 1B model on the OTHER family's data, own tokenizer (cross-tokenization control:
#             pythia<-olmo_mix, olmo<-pile; block_representations_samples_swap[_padded]). 1B only.
#   fineweb — a shared held-out web corpus (rankme_fineweb_padded [after-norm only] vs the packed
#             block_representations_fineweb_packed). The padded before-norm panel is intentionally
#             empty — that run stored after_final_norm only.
# after_final_norm = post-LayerNorm; before_final_norm = the raw pre-norm residual, centered.
# NB: Pythia's hard before-norm collapse (RankMe -> single digits) is a LAST-TOKEN effect — one
# rogue/outlier residual direction dominates the last-token pre-norm variance and LayerNorm
# divides it out; the packed (all-token) panel dilutes it to a mild dip. The two panels are thus
# NOT the same phenomenon at two sample sizes (docs/dig_findings.md). Under swap the collapse
# weakens (the rogue direction is fired by particular last tokens), which the swap row shows.
import numpy as np
import matplotlib.pyplot as plt

# (regime label, padded cfg, packed cfg, models)
RANKME_REGIMES = [
    ('native',  BLOCK_SAMPLES, PACKED_SAMPLES, filter_model_names),
    ('swap',    SWAP_PADDED,   SWAP_PACKED,    ['pythia-1b-deduped', 'OLMo-2-0425-1B']),
    ('fineweb', FINEWEB_PADDED, FINEWEB_PACKED, filter_model_names),
]

def _rankme_curve(ax, src, model, hook):
    """Plot one model's RankMe(hook, centered) vs tokens; True if data was found, else False."""
    try:
        ys, steps = _lib.get_ys(src, model, (hook, 'acts_centered'), 'rankme')
    except Exception:
        return False
    xs = np.asarray(_lib.get_xs_tokens(model, steps), dtype=float)
    keep = xs > 0   # step 0 has zero tokens, unplottable on log-x
    if not keep.any():
        return False
    ax.plot(xs[keep], np.asarray(ys)[keep], marker='o', ms=3, lw=2, label=get_model_label(model))
    return True

def _final_rankme_compare(regime, padded_cfg, packed_cfg, models, hook='after_final_norm'):
    """One 2-panel figure (padded | packed) of final-stream RankMe for a dataset regime. Panels
    share a y-range; a panel with no data for this hook (e.g. fineweb padded before-norm) is
    annotated rather than dropped, so missing coverage is visible."""
    panels = ((padded_cfg, 'padded (last-token rows)'), (packed_cfg, 'packed (all tokens)'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    drawn = False
    for ax, (src, sub) in zip(axes, panels):
        here = any(_rankme_curve(ax, src, m, hook) for m in models)
        drawn = drawn or here
        if not here:
            ax.text(0.5, 0.5, f'no {hook.replace("_", " ")} data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
        ax.set(xscale='log', title=sub, xlabel='tokens')
        ax.set_ylabel(f'RankMe ({hook.replace("_", " ")}, centered)')
    if not drawn:
        plt.close(fig)
        return
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].legend(fontsize=9)
    fig.suptitle(f'Final-stream RankMe over training — {regime}: padded vs packed '
                 f'[{hook.replace("_", " ")}]')
    fig.tight_layout()
    plt.show()

for _regime, _pad, _pack, _models in RANKME_REGIMES:
    _final_rankme_compare(_regime, _pad, _pack, _models, hook='after_final_norm')
    _final_rankme_compare(_regime, _pad, _pack, _models, hook='before_final_norm')


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
_tr_srcs = lambda model: [(BLOCK_SAMPLES, (f'blk{l}.{s}.out', 'acts_uncentered'), f'{s} {l}')
                          for l in range(n_blocks_bs[model]) for s in ('attn', 'mlp')]
_ledger_srcs = lambda model: [(BLOCK_SAMPLES, (f'blk{l}', 'block_ledger'), f'blk {l}')
                              for l in range(n_blocks_bs[model])]

# %%
grid_start(ncols=2, title='Layer contribution — share of final-stream energy (R over r)')
for model in filter_model_names:
    plot_layer_contribution(model, _rr_srcs(model), yvar='R_over_r', normalize=False,
                            title=get_model_label(model), remainder_label='embedding (residual)')
grid_show()

# %%
# Share of residual energy from just the traces (previously missing in the padded twin):
# uncentered write traces normalized over writes + the embedding stream; gray dashed = the
# embedding's share.
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
grid_start(ncols=2, title='Layer contribution — rank-entropy ledger (signed ΔS)')
for model in filter_model_names:
    plot_layer_contribution(model, _ledger_srcs(model), yvar='delta_s', normalize=False,
                            title=get_model_label(model))
grid_show()

# %%
# The same stacks per ledger term: ΔS_k split into its χ / quality / interference parts.
for term in ('chi', 'quality', 'interference'):
    grid_start(ncols=2, title=f'Layer contribution — rank-entropy ledger ({term})')
    for model in filter_model_names:
        plot_layer_contribution(model, _ledger_srcs(model), yvar=term, normalize=False,
                                title=get_model_label(model))
    grid_show()


# %% [markdown]
# #### Ledger contribution stacks (showcase implementation — sign-crossing fills)
# Dotted gray = the embedding stream's own entropy S_emb(t). NB vs the packed twin:
# pythia's final layers' ΔS flips sign here (final_report §2.3, conditional final layers).

# %%
grid_start(ncols=2, title='Ledger contribution stacks (padded)')
for model in MODELS_LC:
    _lib.plot_ledger_stack(model, CFG_BS(model), NB_BS[model],
                           title=get_model_label(model))
grid_show()

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
grid_start(ncols=2, title='Ledger contributions (Exp 4.8)')
for model in filter_model_names:
    _lib.plot_ledger_stack(model, BLOCK_SAMPLES, n_blocks_bs[model], emb=False, legend=10,
                           title=get_model_label(model))
grid_show()

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
