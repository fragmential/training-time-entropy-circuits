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
    plot_heatmap, block_mean_cos, build_hooks, get_model_label, submatrix, plot_wo_topk,
)

# %%
FINEWEB_PADDED = "rankme_fineweb_padded"
# TRAINSET_PACKED_UNC_ONLY = "rankme_trainset_packed_unc"
# TRAINSET_PACKED = "rankme_trainset_packed_40M"
TRAINSET_PACKED = "rankme_trainset_packed_4MT"
# KF_ARXIV = "kfac_small_arxiv"
KF_SMALL = "kfac_small_shuffled"
# BLOCK_REPR_PARTIAL = "block_representations"   # layer-subset run, superseded by the all-layer redo
BLOCK_REPR = "block_representations_all"
BLOCK_SAMPLES = "block_representations_samples"   # raw-sample runs (crosses computed inline)

# %%
HK = build_hooks()   # hook-name -> (node, metric); machinery lives in experiments_lib

# %%
filter_model_names = [
    # 'pythia-14m-deduped',
    # 'pythia-31m-deduped',
    # 'pythia-70m-deduped',
    # 'pythia-160m-deduped',
    # 'pythia-410m-deduped',
    'pythia-1b-deduped',
    # 'pythia-1.4b-deduped',
    # 'pythia-2.8b-deduped',
    'pythia-6.9b-deduped',
    # 'pythia-12b-deduped',
    # 'pythia-14m',
    # 'pythia-31m',
    # 'pythia-70m',
    # 'pythia-160m',
    # 'pythia-410m',
    # 'pythia-1b',
    # 'pythia-1.4b',
    # 'pythia-2.8b',
    # 'pythia-6.9b',
    # 'pythia-12b',
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

data_sources_2 = [
#   Data File          Hook      Label
    (KF_SMALL,         HK.AFN_AC,   'Trainset'),
    (FINEWEB_PADDED,   HK.AFN_AC,   'Fineweb'),
]
data_sources_3 = [
#   Data File          Hook      Label
    (KF_SMALL,         HK.BFN_AC,   'Before final norm'),
    (KF_SMALL,         HK.AFN_AC,   'After final norm'),
]
data_sources_4 = [
#   Data File          Hook      Label
    (TRAINSET_PACKED,  HK.AFN_AC,   'id head'),
    (KF_SMALL,         HK.AFN_AC,   'kfac run'),
]

data_sources_5 = [
#   Data File          Hook      Label
    (KF_SMALL,         HK.BFN_AU,   'Before final norm'),
    (KF_SMALL,         HK.AFN_AU,   'After final norm'),
]


wso = 'up', 'gate', 'down'
wsp = 'up', 'down'
ms = 'Ac', 'Au', 'Bc', 'Bu', 'Gc', 'Gu'

def mlp_(src):
    def mlp_src(ws, blks, ms):
        return [(src, HK[f'{w[0].upper()}{l}_{m.upper()[:2]}'],
                 f'{m} {l} {w}')
                 for l in blks for w in ws for m in ms]
    return mlp_src

def res_(src):
    def res_src(pos, ms):
        return [(src, HK[f'{p[0].upper()}FN_{m.upper()[:2]}'],
                 f'{m} {p}') for p in pos for m in ms]
    return res_src


mlp_ts = mlp_(KF_SMALL)
res_ts = res_(KF_SMALL)

measured = {
    'pythia-1b-deduped': [0,5,10, 15],
    'pythia-6.9b-deduped': [0,10,21,31],
    'OLMo-2-0425-1B': [0,5,10, 15],
    'OLMo-2-1124-7B': [0,10,21,31],
}


mlp_ac = lambda model: mlp_ts(wso, measured[model], ['Ac'])
mlp_bc = lambda model: mlp_ts(wso, measured[model], ['Bc'])
mlp_gc = lambda model: mlp_ts(wso, measured[model], ['Gc'])
mlp_au = lambda model: mlp_ts(wso, measured[model], ['Au'])
mlp_bu = lambda model: mlp_ts(wso, measured[model], ['Bu'])
mlp_gu = lambda model: mlp_ts(wso, measured[model], ['Gu'])
mlp_kf = lambda model: mlp_ts(['up', 'gate'], measured[model], ['kfac'])
mlp_kfu = lambda model: mlp_ts(['up'], measured[model], ['kfac'])
mlp_kfd = lambda model: mlp_ts(['down'], measured[model], ['kfac'])
mlp_kf0 = lambda model: mlp_ts(['up', 'gate'], [measured[model][0]], ['kfac'])
mlp_kfd0 = lambda model: mlp_ts(['down'], [measured[model][0]], ['kfac'])
mlp_kf15 = lambda model: mlp_ts(['up', 'gate'], [measured[model][-1]], ['kfac'])
mlp_kfd15 = lambda model: mlp_ts(['down'], [measured[model][-1]], ['kfac'])

mlp_lkf = lambda model: mlp_ts(['layer'], measured[model], ['kfac'])

mlp_magu_u = lambda model: mlp_ts(['up'], measured[model], ['Au', 'Bu', 'Gu'])
mlp_magu_c = lambda model: mlp_ts(['up'], measured[model], ['Ac', 'Bc', 'Gc'])
mlp_magd_u = lambda model: mlp_ts(['down'], measured[model], ['Au', 'Bu', 'Gu'])

res_bfn_u = res_ts(['before norm'], ['Au', 'Gu'])
res_bfn_c = res_ts(['before norm'], ['Ac', 'Gc'])
res_afn_u = res_ts(['after norm'], ['Au', 'Gu'])
res_afn_c = res_ts(['after norm'], ['Ac', 'Gc'])
res_fn_u = res_ts(['before norm', 'after norm'], ['Au', 'Gu'])
res_fn_c = res_ts(['before norm', 'after norm'], ['Ac', 'Gc'])

res_bfnk = res_ts(['before norm'], ['kfac'])
res_afnk = res_ts(['after norm'], ['kfac'])

mlp_gb = lambda model: mlp_ts(['up', 'gate'], measured[model], ['GB'])
mlp_gb15 = lambda model: mlp_ts(['up', 'gate'], [measured[model][-1]], ['GB'])
mlp_gb0 = lambda model: mlp_ts(['up', 'gate'], [measured[model][0]], ['GB'])
mlp_gbd = lambda model: mlp_ts(['down'], measured[model], ['GB'])
mlp_gbd15 = lambda model: mlp_ts(['down'], [measured[model][-1]], ['GB'])
mlp_gbd0 = lambda model: mlp_ts(['down'], [measured[model][0]], ['GB'])


# --- block_representations_all runs: every layer + sub-block boundary shorthands ---
measured_br = {
    'pythia-1b-deduped':   list(range(16)),
    'pythia-6.9b-deduped': list(range(32)),
    'OLMo-2-0425-1B':      list(range(16)),
    'OLMo-2-1124-7B':      list(range(32)),
}

def bnd_(src):
    def bnd_src(prefix, blks, ms):
        return [(src, HK[f'{prefix}{l}_{m.upper()[:2]}'], f'{m} {l} {prefix}')
                for l in blks for m in ms]
    return bnd_src

bnd_br   = bnd_(BLOCK_REPR)
attn_in  = lambda model, ms=['Au']: bnd_br('AI', measured_br[model], ms)
attn_out = lambda model, ms=['Au']: bnd_br('AO', measured_br[model], ms)
mlp_in   = lambda model, ms=['Au']: bnd_br('MI', measured_br[model], ms)
mlp_out  = lambda model, ms=['Au']: bnd_br('MO', measured_br[model], ms)


# Per-OV-head shorthand. which='slice' (pre-W_o, d_head) or 'contrib' (post-W_o, d_model,
# derived). tail ∈ {Au,Ac,Gu,Gc,GB}; blk/h/tail accept an int/str or a list; blk=None ->
# all measured_br[model] layers. (block_representations_all has heads for the 1Bs only —
# the 7Bs ran the boundary-only _all_7b variant; use BLOCK_REPR_PARTIAL's source for 7B heads.)
_HEAD_METRIC = {'AU': 'acts_uncentered', 'AC': 'acts_centered',
                'GU': 'grads_uncentered', 'GC': 'grads_centered', 'GB': 'gen'}

def head(model, blk, h, tail=['Au'], which='slice', src=BLOCK_REPR):
    blks = measured_br[model] if blk is None else ([blk] if isinstance(blk, int) else blk)
    hs   = [h] if isinstance(h, int) else h
    tls  = [tail] if isinstance(tail, str) else tail
    return [(src, (f'blk{l}.attn.head{hh}.{which}', _HEAD_METRIC[t.upper()[:2]]),
             f'{t} b{l}h{hh} {which}')
            for l in blks for hh in hs for t in tls]



# %% [markdown]
# ### Final residual RankMe, Trace, and Mean

# %%

####        #        #        #        #        #        #        ####
####     "Tracing the representation geometry.." reproduction     ####
####        #        #        #        #        #        #        ####
grid_start(ncols=2, title='RankMe reproductions')
plot_group('rankme', data_sources_1, ['OLMo-2-0425-1B'], title='OLMo-2 1B', ls_exceptions={'uncentered':'--'})
plot_group('alpha', data_sources_1, ['OLMo-2-0425-1B'], title='OLMo-2 1B', ls_exceptions={'uncentered':'--'})

plot_group('rankme', data_sources_1, ['pythia-1b-deduped'], title='Pythia 1B Deduped', ls_exceptions={'uncentered':'--'})
plot_group('alpha', data_sources_1, ['pythia-1b-deduped'], title='Pythia 1B Deduped', ls_exceptions={'uncentered':'--'})

plot_group('rankme', data_sources_2, filter_model_names, ls_exceptions={'Fineweb':'--'})
plot_group('alpha', data_sources_2, filter_model_names, ls_exceptions={'Fineweb':'--'})

# plot_group('rankme', data_sources_4, ['OLMo-2-0425-1B'])
# plot_group('alpha', data_sources_4, ['OLMo-2-0425-1B'])

# plot_group('rankme', data_sources_3, ['OLMo-2-0425-1B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)
# plot_group('true_rankme', data_sources_3, ['OLMo-2-0425-1B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)
# plot_group('matrix_entropy', data_sources_3, ['OLMo-2-0425-1B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)
# plot_group('sv_entropy', data_sources_3, ['OLMo-2-0425-1B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)

plot_group('rankme', data_sources_3, ['OLMo-2-1124-7B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)
plot_group('true_rankme', data_sources_3, ['OLMo-2-1124-7B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)
plot_group('matrix_entropy', data_sources_3, ['OLMo-2-1124-7B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)
plot_group('sv_entropy', data_sources_3, ['OLMo-2-1124-7B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)

# plot_group('alpha', data_sources_3, ['OLMo-2-0425-1B'], ls_exceptions={'before':'--'}, hold_color_for_n=1)
plot_group('rankme_center_diff', data_sources_2[:1], filter_model_names)
plot_group('rankme_center_prop', data_sources_2[:1], filter_model_names)
grid_show()

# grid_start(ncols=2, title='Final residual Trace and Mean interaction', omit_step0=False)
# plot_group('trace', data_sources_5[:1], filter_model_names, title='Before final norm, uncentered')
# plot_group('trace', data_sources_5[1:], filter_model_names, title='After final norm, uncentered')

# plot_group('trace', data_sources_3[:1], filter_model_names, title='Before final norm, centered')
# plot_group('trace', data_sources_3[1:], filter_model_names, title='After final norm, centered')

# plot_group('mean_norm', data_sources_5[:1], filter_model_names, title='Before final norm')
# plot_group('mean_norm', data_sources_5[1:], filter_model_names, title='After final norm')

# plot_group('mean_frac', data_sources_5[:1], filter_model_names, title='Before final norm')
# plot_group('mean_frac', data_sources_5[1:], filter_model_names, title='After final norm')

# # plot_group('mean_ratio', data_sources_5[:1], filter_model_names, title='Before final norm', disable_midx=[1])
# # plot_group('mean_ratio', data_sources_5[1:], filter_model_names, title='After final norm', disable_midx=[1])

# grid_show()

# %% [markdown]
# ### Block Means

# %%

# Sub-block boundary means / spectra — block_representations runs.
# Tail letters = quantity + centering: 'Au' = acts uncentered, 'Ac' = acts centered
# (boundaries have no derived "B" output, so it's A, not the old Bu/Bc). _boundary_grid
# plots the uncentered set (trace/log_det/means/rankme) and the centered set
# (peak/spectrum/rankme/center-diff/prop), mirroring the down-proj cell.
def _boundary_grid(model, title, bnd):
    u, c = bnd(model, ['Au']), bnd(model, ['Ac'])
    grid_start(ncols=2, title=f'{title} — {model}')
    plot_group('trace', u, [model], ylog=True)
    plot_group('log_det', u, [model])
    plot_group('peak_eigval', c, [model], ylog=True)
    plot_spectrum('eigvals', c, [model], title="Final Checkpoint Eigvals")
    plot_group('mean_norm', u, [model])
    plot_group('mean_frac', u, [model])
    plot_group('rankme', u, [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', c, [model], title="Centered Covariance Rankme")
    plot_group('rankme_center_diff', c, [model])
    plot_group('rankme_center_prop', c, [model])
    grid_show()

for model in filter_model_names:
    _boundary_grid(model, 'MLP out', mlp_out)

for model in filter_model_names:
    _boundary_grid(model, 'Attn out', attn_out)

# mlp.in is a distinct boundary only in OLMo-2 (post-norm sequential); Pythia's parallel
# residual has no separate MLP input (it is equal to attn.in), so plot it for OLMo only.
for model in [m for m in filter_model_names if 'olmo' in m.lower()]:
    _boundary_grid(model, 'MLP in', mlp_in)

for model in filter_model_names:
    _boundary_grid(model, 'Attn in', attn_in)


# %% [markdown]
# ### Mean-Covariance relationship

# %%
# --- Mean-covariance relationship -------------------------------------------
# How the mean direction μ sits inside the *centered* covariance eigenbasis.
# Metrics come from mean_metrics() in compute_metrics.py, stored per leaf under
# `<quantity>_mean_metrics`. No virtual hooks needed: we just point each boundary
# hook at the `acts_mean_metrics` sub-dict, so every metric name resolves through
# the plain get_ys() reduce path as an ordinary yvar. Same boundaries / models /
# layers as the Block Means grid above. (Labels live in YVAR_LABELS.)
def _meancov_grid(model, title, bnd):
    u = [(s, (h[0], 'acts_mean_metrics'), lbl) for s, h, lbl in bnd(model, ['Au'])]
    grid_start(ncols=2, title=f'{title} mean-cov -- {model}')
    plot_group('mean_norm', u, [model], ylog=True)
    plot_group('rayleigh', u, [model], ylog=True)
    plot_group('rayleigh_normed', u, [model])
    plot_group('mahalanobis', u, [model], ylog=True)
    plot_group('max_overlap', u, [model])
    plot_group('max_overlap_idx', u, [model])
    plot_group('pr', u, [model], ylog=True)
    plot_group('pr_weighted', u, [model], ylog=True)
    plot_group('centroid_idx', u, [model])
    plot_group('centroid_idx_weighted', u, [model])
    plot_spectrum('profile', u, [model], xlog=False, ylog=True, smooth=22, peak=3, title='Final-ckpt profile $p_j$')
    plot_spectrum('profile_weighted', u, [model], xlog=False, ylog=True, smooth=22, peak=3, title='Final-ckpt weighted profile')
    grid_show()

for model in filter_model_names:
    _meancov_grid(model, 'MLP out', mlp_out)

for model in filter_model_names:
    _meancov_grid(model, 'Attn out', attn_out)

# mlp.in is a distinct boundary only in OLMo-2 (see Block Means cell).
for model in [m for m in filter_model_names if 'olmo' in m.lower()]:
    _meancov_grid(model, 'MLP in', mlp_in)

for model in filter_model_names:
    _meancov_grid(model, 'Attn in', attn_in)


# %% [markdown]
# ### Block-vs-residual mean-covariance relationship

# %%
# --- Block-output mean vs the residual it joins -----------------------------
# Same as the Mean-Covariance grid above, but the metric is `mean_metrics_blk_vs_res`
# (METRICS in compute_metrics.py), stored at the sub-block NODE (blkN.mlp / blkN.attn):
# the sub-block output mean measured against the *input* residual's centered covariance
# — the residual it adds into. We just strip `.out` -> node and point at the new key.
# (mlp.in / attn.in would resolve to the same nodes, so we only need mlp_out / attn_out.)
def _blkres_grid(model, title, bnd):
    u = [(s, (h[0].rsplit('.', 1)[0], 'mean_metrics_blk_vs_res'), lbl) for s, h, lbl in bnd(model, ['Au'])]
    grid_start(ncols=2, title=f'{title} blk-vs-res mean -- {model}')
    plot_group('mean_norm', u, [model], ylog=True)
    plot_group('rayleigh', u, [model], ylog=True)
    plot_group('rayleigh_normed', u, [model])
    plot_group('mahalanobis', u, [model], ylog=True)
    plot_group('max_overlap', u, [model])
    plot_group('top_overlap', u, [model])
    plot_group('max_overlap_idx', u, [model])
    plot_group('pr', u, [model], ylog=True)
    plot_group('pr_weighted', u, [model], ylog=True)
    plot_group('centroid_idx', u, [model])
    plot_group('centroid_idx_weighted', u, [model])
    plot_spectrum('profile', u, [model], xlog=False, ylog=True, smooth=22, peak=3, title='Final-ckpt profile $p_j$')
    plot_spectrum('profile_weighted', u, [model], xlog=False, ylog=True, smooth=22, peak=3, title='Final-ckpt weighted profile')
    grid_show()

for model in filter_model_names:
    _blkres_grid(model, 'MLP out', mlp_out)

for model in filter_model_names:
    _blkres_grid(model, 'Attn out', attn_out)

# %% [markdown]
# ### Layer contribution

# %%

# block outputs ordered by depth (attn then mlp within a layer): bottom = layer 0 -> top = final
layer_outs = lambda model: [(BLOCK_REPR, HK[f'{p}{l}_AU'], f'{sub} {l}')
                            for l in measured_br[model] for p, sub in [('AO', 'attn'), ('MO', 'mlp')]]

grid_start(ncols=2, title='Layer contribution')
for model in ['pythia-1b-deduped', 'pythia-6.9b-deduped', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B']:
    plot_layer_contribution(model, layer_outs(model), title=get_model_label(model))
grid_show()

# %%
# R_over_r layer contribution: each write's share of the FINAL stream's energy
# (tr Cov(c_k, r) / tr Σ_r — coupling to the rest of the stream included; the gap to 1.0 is
# the embedding stream's share). Signed by definition, positive in practice: r contains c_k,
# so a write must be cancelled by more than its own energy to go negative.
# All samples-run models (nanochat lives in its own config)
MODELS_LC = ['pythia-160m-deduped', 'pythia-410m-deduped', 'pythia-1b-deduped',
             'pythia-6.9b-deduped', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'nanochat-d12']
NB_BS = {'pythia-160m-deduped': 12, 'pythia-410m-deduped': 24, 'pythia-1b-deduped': 16,
         'pythia-6.9b-deduped': 32, 'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32,
         'nanochat-d12': 12}
CFG_BS = lambda m: 'nanochat_samples' if m == 'nanochat-d12' else BLOCK_SAMPLES

_rr_srcs = lambda model: [(CFG_BS(model), (f'blk{l}.{s}.out', 'block_residual_coupling'), f'{s} {l}')
                          for l in range(NB_BS[model]) for s in ('attn', 'mlp')]
_tr_srcs = lambda model: [(CFG_BS(model), (f'blk{l}.{s}.out', 'acts_uncentered'), f'{s} {l}')
                          for l in range(NB_BS[model]) for s in ('attn', 'mlp')]
_ledger_srcs = lambda model: [(CFG_BS(model), (f'blk{l}', 'block_ledger'), f'blk {l}')
                              for l in range(NB_BS[model])]



# %%

grid_start(ncols=2, title='Layer contribution — share of final-stream energy (R over r)')
for model in MODELS_LC:
    plot_layer_contribution(model, _rr_srcs(model), yvar='R_over_r', normalize=False,
                            title=get_model_label(model), remainder_label='embedding (residual)')
grid_show()

# %%
# Share of residual energy from just the traces: uncentered write traces normalized over
# writes + the embedding stream; gray dashed = the embedding's share.
grid_start(ncols=2, title='Layer contribution — share of residual energy (trace)')
for model in MODELS_LC:
    plot_layer_contribution(model, _tr_srcs(model), yvar='trace', normalize=True,
                            title=get_model_label(model),
                            baseline_src=(CFG_BS(model), ('blk0.attn.in', 'acts_uncentered'), 'embedding'))
grid_show()

# %%
# Ledger-based layer contribution: each block's signed ΔS_k (= χ + quality + interference —
# overlap and cross terms INCLUDED, unlike the gross-energy stack above). Positives stack up,
# negatives down; the net stack height is log RankMe(final) − log RankMe(embeddings).
# Dotted gray = S(embedding stream) as its change from the first step (0 at t0), so it sits
# on the stack's scale rather than at its own absolute offset. Black = the measured depth
# differential S(before_final_norm) − S(embedding), which the signed stack should sum to.
grid_start(ncols=2, title='Layer contribution — rank-entropy ledger (signed ΔS)')
for model in MODELS_LC:
    plot_layer_contribution(model, _ledger_srcs(model), yvar='delta_s', normalize=False,
                            title=get_model_label(model), baseline_ls=':', baseline_delta=True,
                            baseline_src=(CFG_BS(model), ('blk0.attn.in', 'acts_centered'),
                                          'ΔS(embedding stream)', 'matrix_entropy'),
                            total_src=(CFG_BS(model), ('before_final_norm', 'acts_centered'),
                                       'S(bfn) − Sf(emb)', 'matrix_entropy'))
grid_show()

# %%
# The same stacks per ledger term: ΔS_k split into its χ / quality / interference parts.
for term in ('chi', 'quality', 'interference'):
    grid_start(ncols=2, title=f'Layer contribution — rank-entropy ledger ({term})')
    for model in MODELS_LC:
        plot_layer_contribution(model, _ledger_srcs(model), yvar=term, normalize=False,
                                title=get_model_label(model))
    grid_show()

# %% [markdown]
# #### Ledger contribution stacks (Exp 4.8, showcase implementation — sign-crossing fills)
# Dotted gray = the embedding stream's own entropy S_emb(t), the base the black
# depth-differential is measured against.

# %%
grid_start(ncols=2, title='Ledger contribution stacks')
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
# ### Per-layer-output vs final RankMe

# %%
# --- Exp 2: average per-layer-output RankMe vs the final residual ------------
# The final residual is the *sum* of all block outputs, yet its effective rank
# (RankMe) != the mean of the per-output RankMes. Each panel shows the per-output
# spread (grey) + their mean±band (aggregate=True), with the final residual
# (before_final_norm = the literal sum) overlaid via `background`. Uncentered and
# centered, for all / attn-only / mlp-only outputs.
def _out_rankme_grid(model, yvar='rankme', normalize=None):
    bg = (BLOCK_REPR, HK.BFN_AC, 'final')   # reference = final *centered* RankMe, same for every panel
    groups = {'all':  lambda t: mlp_out(model, [t]) + attn_out(model, [t]),
              'attn': lambda t: attn_out(model, [t]),
              'mlp':  lambda t: mlp_out(model, [t])}
    grid_start(ncols=2, title=f'Per-output vs final RankMe{" (shape)" if normalize else ""} — {model}', sharey=True)
    for grp, sel in groups.items():
        for t, lbl in [('Au', 'uncentered'), ('Ac', 'centered')]:
            plot_group(yvar, sel(t), [model], aggregate=True, weight='trace',
                       background=bg, normalize=normalize, title=f'{grp} outputs, {lbl} - {model}')
    grid_show()

for model in filter_model_names:
    _out_rankme_grid(model)                       # absolute

# %%
for model in filter_model_names:
    _out_rankme_grid(model, normalize='anchor')   # shape (each line matched at the reference's peak)


# %%
# for model in filter_model_names:
#     _out_rankme_grid(model, yvar='true_rankme')#, normalize='anchor')
# for model in filter_model_names:
#     _out_rankme_grid(model, yvar='matrix_entropy')#, normalize='anchor')

# %% [markdown]
# ### Block mean ↔ residual mean (Exp 1.2)

# %%
# Each block output's mean vs the local residual it adds into (its sub-block input,
# blkN.{mlp,attn}.in) — cosine of mean directions + magnitude ratio ‖μ_out‖/‖μ_in‖.
# Same "residual it joins" convention as mean_metrics_blk_vs_res. Coloured by depth.
def _blkres_mean_grid(model):
    outs = layer_outs(model)
    grid_start(ncols=2, title=f'Block mean vs the residual it joins — {model}')
    plot_group('cos_to_res', outs, [model], color_palette='gradient')
    plot_group('magratio_res', outs, [model], color_palette='gradient', ylog=True)
    grid_show()

for model in filter_model_names:
    _blkres_mean_grid(model)

# %% [markdown]
# ### Mean-direction drift (Exp 1.3)

# %%
# How each block-output mean *direction* moves over training: cosine with the previous
# checkpoint (consecutive) and with an EMA of its own history (denoised). Series-mode hooks.
def _drift_grid(model):
    outs = layer_outs(model)
    grid_start(ncols=2, title=f'Mean-direction drift — {model}')
    plot_group('cos_drift', outs, [model], color_palette='gradient')
    plot_group('cos_drift_ema', outs, [model], color_palette='gradient')
    grid_show()

for model in filter_model_names:
    _drift_grid(model)

# %% [markdown]
# ### Block↔block mean alignment (Exp 1.1)

# %%
# Pairwise cosine between block-output means at the final checkpoint (block×block),
# ordered by depth (attn, mlp per layer). Diagonal hidden; symmetric dynamic colour
# range. Off-diagonal structure shows which layers' mean directions co-align.
# One grid per sub-block pairing: all×all, attn×attn, mlp×mlp, attn×mlp.
PAIRINGS = (('', ''), ('attn', 'attn'), ('mlp', 'mlp'), ('attn', 'mlp'))

for rows, cols in PAIRINGS:
    grid_start(ncols=2, title=f'Block-mean cosine ({rows or "all"}×{cols or "all"}, final ckpt)', savefig=False)
    for model in filter_model_names:
        outs = layer_outs(model)
        M, rl, cl = submatrix(block_mean_cos(model, outs), [s[2] for s in outs], rows, cols)
        plot_heatmap(M, labels=rl, labels2=cl, title=get_model_label(model),
                     hide_diag=rows == cols, dynamic=True)
    grid_show()


# %% [markdown]
# ### Samples run: block↔block coupling (Exp 4.1)

# %%
# block_representations_samples (BLOCK_SAMPLES, defined with the constants up top).
n_blocks_bs = {'pythia-1b-deduped': 16, 'pythia-6.9b-deduped': 32,
               'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32}
bs_out = lambda model, family: [(BLOCK_SAMPLES, (f'blk{l}.{s}.out', family), f'{s} {l}')
                                for l in range(n_blocks_bs[model]) for s in ('attn', 'mlp')]
bs_blk = lambda model: [(BLOCK_SAMPLES, (f'blk{l}', 'block_ledger'), f'blk {l}')
                        for l in range(n_blocks_bs[model])]

# %%
# Pairwise block-output coupling at the final checkpoint: CKA (shared subspace, unsigned),
# signed trace (net reinforce/cancel, energy-weighted), mean per-token cosine (democratic).
# Covariance-level counterpart of the block-mean cosine heatmaps above.
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
KS = (0, 1, 2, 8, 32, 128, 512)
WINDOWS = ((11, 100), (32, 128), (128, 512), (512, -20))
_pen = lambda model: f'blk{n_blocks_bs[model] - 1}.attn.in'   # penultimate stream: entering the last block
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
# ### Rogue write anatomy (Exp 4.9, Pythia)

# %%
# (a) blk3.mlp's write collapses to rank ~1 at the RankMe peak and then grows in energy —
# vs its neighbor writes. (b) Token-level evidence from the block_rogue_id raw samples
# (final ckpt): projections onto the write's top centered direction, newline tokens vs rest
# (38/40 top spikes are \n variants; docs/dig_findings.md).
def _rogue_traj(model):
    outs = [(BLOCK_SAMPLES, (f'blk{l}.mlp.out', 'acts_centered'), f'mlp {l}') for l in (1, 3, 5)]
    grid_start(ncols=2, title=f'Rogue write trajectory — {get_model_label(model)}')
    plot_group('rankme', outs, [model], color_palette='gradient', ylog=True, title='write RankMe (centered)')
    plot_group('trace', outs, [model], color_palette='gradient', ylog=True, title='write trace')
    grid_show()

for model in filter_model_names:
    if 'pythia' in model:
        _rogue_traj(model)

# %%
# Heavier cell: loads raw samples + the packed mix + tokenizer.
import torch as _t
from utils.accessor import DataAccessor as _DA

def _rogue_scatter(model, step=143000):
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    acc = _DA(f'data/inferences/block_rogue_id/{model}/step{step}.pt')
    v = acc['blk3.mlp.out'].acts
    score = (v.samples.float() - v.mean.float()) @ v.eigvecs_centered[:, 0].float()
    ids = _t.load('data/mixes/pile_30M_512.pt')[:, :511].reshape(-1)[:len(score)]
    tok = AutoTokenizer.from_pretrained(f'EleutherAI/{model}')
    uniq = _t.unique(ids)
    nl_ids = {int(t) for t in uniq if '\n' in tok.decode([int(t)])}
    is_nl = _t.tensor([int(t) in nl_ids for t in ids])
    plt.figure(figsize=(7, 4))
    for m, lbl, c in ((~is_nl, 'other tokens', 'tab:gray'), (is_nl, 'newline tokens', 'tab:red')):
        plt.hist(score[m].abs().numpy(), bins=120, log=True, alpha=0.6, color=c, label=lbl)
    plt.xlabel('|projection onto rogue direction|'); plt.ylabel('token count (log)')
    plt.title(f'Who carries the rogue direction — {get_model_label(model)}, final ckpt')
    plt.legend(); plt.show()

for model in filter_model_names:
    if 'pythia' in model:
        _rogue_scatter(model)

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
