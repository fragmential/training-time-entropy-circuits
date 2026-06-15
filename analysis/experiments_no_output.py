# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: representation-geometry
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
    build_hooks, get_model_label,
)

# %%
FINEWEB_PADDED = "rankme_fineweb_padded"
# TRAINSET_PACKED_UNC_ONLY = "rankme_trainset_packed_unc"
# TRAINSET_PACKED = "rankme_trainset_packed_40M"
TRAINSET_PACKED = "rankme_trainset_packed_4MT"
# KF_ARXIV = "kfac_small_arxiv"
KF_SMALL = "kfac_small_shuffled"
BLOCK_REPR = "block_representations"

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


# --- block_representations runs: measured layers + sub-block boundary shorthands ---
measured_br = {
    'pythia-1b-deduped':   [0, 1, 3, 5, 8, 10, 13, 15],
    'pythia-6.9b-deduped': [0, 1, 5, 10, 15, 21, 26, 31],
    'OLMo-2-0425-1B':      [0, 1, 3, 5, 8, 10, 13, 15],
    'OLMo-2-1124-7B':      [0, 1, 5, 10, 15, 21, 26, 31],
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
# all measured_br[model] layers. (Heads live in the block_representations runs.)
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

grid_start(ncols=2, title='Final residual Trace and Mean interaction', omit_step0=False)
plot_group('trace', data_sources_5[:1], filter_model_names, title='Before final norm, uncentered')
plot_group('trace', data_sources_5[1:], filter_model_names, title='After final norm, uncentered')

plot_group('trace', data_sources_3[:1], filter_model_names, title='Before final norm, centered')
plot_group('trace', data_sources_3[1:], filter_model_names, title='After final norm, centered')

plot_group('mean_norm', data_sources_5[:1], filter_model_names, title='Before final norm')
plot_group('mean_norm', data_sources_5[1:], filter_model_names, title='After final norm')

plot_group('mean_frac', data_sources_5[:1], filter_model_names, title='Before final norm')
plot_group('mean_frac', data_sources_5[1:], filter_model_names, title='After final norm')

# plot_group('mean_ratio', data_sources_5[:1], filter_model_names, title='Before final norm', disable_midx=[1])
# plot_group('mean_ratio', data_sources_5[1:], filter_model_names, title='After final norm', disable_midx=[1])

grid_show()


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
    plot_spectrum('profile', u, [model], xlog=False, ylog=True, title='Final-ckpt profile $p_j$')
    plot_spectrum('profile_weighted', u, [model], xlog=False, ylog=True, title='Final-ckpt weighted profile')
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
    plot_spectrum('profile', u, [model], xlog=False, ylog=True, title='Final-ckpt profile $p_j$')
    plot_spectrum('profile_weighted', u, [model], xlog=False, ylog=True, title='Final-ckpt weighted profile')
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
for model in filter_model_names:
    _out_rankme_grid(model, yvar='true_rankme')#, normalize='anchor')
for model in filter_model_names:
    _out_rankme_grid(model, yvar='matrix_entropy')#, normalize='anchor')

# %% [markdown]
# ### K-FAC up/gate

# %%

#############################################################################

for model in filter_model_names:
    grid_start(ncols=2, title=f'K-FAC up — {model}') #, disable_didx=[1,3,5,7] if 'pythia' in model else [])
    plot_group('trace', mlp_kf(model)[::2], [model], ylog=True)
    plot_group('log_det', mlp_kf(model)[::2], [model])
    plot_group('rankme', mlp_kf(model)[::2], [model])
    plot_group('peak_eigval', mlp_kf(model)[::2], [model], ylog=True)
    grid_show()


# %% [markdown]
# ### K-FAC down

# %%

for model in filter_model_names:
    grid_start(ncols=2, title=f'K-FAC down — {model}')
    plot_group('trace', mlp_kfd(model), [model], ylog=True)
    plot_group('log_det', mlp_kfd(model), [model])
    plot_group('rankme', mlp_kfd(model), [model])
    plot_group('peak_eigval', mlp_kfd(model), [model], ylog=True)
    grid_show()

# %% [markdown]
# ### Up-proj and Gate-proj inputs and outputs

# %%

for model in filter_model_names:
    grid_start(ncols=2, title=f'Up-proj inputs — {model}')
    plot_group('trace', mlp_au(model)[::3], [model], ylog=True)
    plot_group('log_det', mlp_au(model)[::3], [model])
    plot_group('peak_eigval', mlp_ac(model)[::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_ac(model)[::3], [model], title="Final Checkpoint Eigvals")
    plot_group('mean_norm', mlp_au(model)[::3], [model])
    plot_group('mean_frac', mlp_au(model)[::3], [model])
    plot_group('rankme', mlp_au(model)[::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_ac(model)[::3], [model], title="Centered Covariance Rankme")
    plot_group('rankme_center_diff', mlp_ac(model)[::3], [model])
    plot_group('rankme_center_prop', mlp_ac(model)[::3], [model])
    # plot_spectrum(('histogram', dict(bins=128)), mlp_ac(model)[::3], [model], title="Final Checkpoint Histogram", ylog=False)
    grid_show()

for model in filter_model_names:
    grid_start(ncols=2, title=f'Up-proj outputs — {model}')
    plot_group('trace', mlp_bu(model)[::3], [model], ylog=True)
    plot_group('log_det', mlp_bu(model)[::3], [model])
    plot_group('peak_eigval', mlp_bc(model)[::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_bc(model)[::3], [model], title="Final Checkpoint Eigvals")
    plot_group('mean_norm', mlp_bu(model)[::3], [model])
    plot_group('mean_frac', mlp_bu(model)[::3], [model])
    plot_group('rankme', mlp_bu(model)[::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_bc(model)[::3], [model], title="Centered Covariance Rankme")
    plot_group('rankme_center_diff', mlp_bc(model)[::3], [model])
    plot_group('rankme_center_prop', mlp_bc(model)[::3], [model])
    grid_show()

for model in filter_model_names[2:]:
    grid_start(ncols=2, title=f'Gate-proj outputs — {model}')
    plot_group('trace', mlp_bu(model)[1::3], [model], ylog=True)
    plot_group('log_det', mlp_bu(model)[1::3], [model])
    plot_group('peak_eigval', mlp_bc(model)[1::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_bc(model)[1::3], [model], title="Final Checkpoint Eigvals")
    plot_group('mean_norm', mlp_bu(model)[1::3], [model])
    plot_group('mean_frac', mlp_bu(model)[1::3], [model])
    plot_group('rankme', mlp_bu(model)[1::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_bc(model)[1::3], [model], title="Centered Covariance Rankme")
    plot_group('rankme_center_diff', mlp_bc(model)[1::3], [model])
    plot_group('rankme_center_prop', mlp_bc(model)[1::3], [model])
    grid_show()


# %% [markdown]
# ### Down-proj inputs and outputs

# %%

for model in filter_model_names:
    grid_start(ncols=2, title=f'Down-proj inputs — {model}')
    plot_group('trace', mlp_au(model)[2::3], [model], ylog=True)
    plot_group('log_det', mlp_au(model)[2::3], [model])
    plot_group('peak_eigval', mlp_ac(model)[2::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_ac(model)[2::3], [model], title="Final Checkpoint Eigvals")
    plot_group('mean_norm', mlp_au(model)[2::3], [model])
    plot_group('mean_frac', mlp_au(model)[2::3], [model])
    plot_group('rankme', mlp_au(model)[2::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_ac(model)[2::3], [model], title="Centered Covariance Rankme")
    plot_group('rankme_center_diff', mlp_ac(model)[2::3], [model])
    plot_group('rankme_center_prop', mlp_ac(model)[2::3], [model])
    grid_show()

for model in filter_model_names:
    grid_start(ncols=2, title=f'Down-proj outputs — {model}')
    plot_group('trace', mlp_bu(model)[2::3], [model], ylog=True)
    plot_group('log_det', mlp_bu(model)[2::3], [model])
    plot_group('peak_eigval', mlp_bc(model)[2::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_bc(model)[2::3], [model], title="Final Checkpoint Eigvals")
    plot_group('mean_norm', mlp_bu(model)[2::3], [model])
    plot_group('mean_frac', mlp_bu(model)[2::3], [model])
    plot_group('rankme', mlp_bu(model)[2::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_bc(model)[2::3], [model], title="Centered Covariance Rankme")
    plot_group('rankme_center_diff', mlp_bc(model)[2::3], [model])
    plot_group('rankme_center_prop', mlp_bc(model)[2::3], [model])
    grid_show()


# %% [markdown]
# ### Sampled-gradients

# %%

for model in filter_model_names:
    grid_start(ncols=2, title=f'Up-proj sampled gradients (loss wrt model\'s own samples) — {model}',
            disable_didx=[]) # omit l0 bc it's too noisy
    plot_group('trace', mlp_gu(model)[::3], [model], ylog=True)
    plot_group('log_det', mlp_gu(model)[::3], [model])
    plot_group('peak_eigval', mlp_gc(model)[::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_gc(model)[::3], [model], title="Final Checkpoint Eigvals")
    # plot_group('mean_norm', mlp_gu(model)[::3], [model])
    # plot_group('mean_frac', mlp_gu(model)[::3], [model])
    # plot_group('rankme', mlp_gu(model)[::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_gc(model)[::3], [model], title="Centered Covariance Rankme")
    # plot_group('rankme_center_diff', mlp_gc(model)[::3], [model])
    # plot_group('rankme_center_prop', mlp_gc(model)[::3], [model])
    grid_show()

for model in filter_model_names[2:]:
    grid_start(ncols=2, title=f'Gate-proj sampled gradients (loss wrt model\'s own samples) — {model}',
            disable_didx=[]) # omit l0 bc it's too noisy
    plot_group('trace', mlp_gu(model)[1::3], [model], ylog=True)
    plot_group('log_det', mlp_gu(model)[1::3], [model])
    plot_group('peak_eigval', mlp_gc(model)[1::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_gc(model)[1::3], [model], title="Final Checkpoint Eigvals")
    # plot_group('mean_norm', mlp_gu(model)[1::3], [model])
    # plot_group('mean_frac', mlp_gu(model)[1::3], [model])
    # plot_group('rankme', mlp_gu(model)[1::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_gc(model)[1::3], [model], title="Centered Covariance Rankme")
    # plot_group('rankme_center_diff', mlp_gc(model)[1::3], [model])
    # plot_group('rankme_center_prop', mlp_gc(model)[1::3], [model])
    grid_show()

for model in filter_model_names:
    grid_start(ncols=2, title=f'Down-proj sampled gradients (loss wrt model\'s own samples) — {model}',
            disable_didx=[]) # omit l0 bc it's too noisy
    plot_group('trace', mlp_gu(model)[2::3], [model], ylog=True)
    plot_group('log_det', mlp_gu(model)[2::3], [model])
    plot_group('peak_eigval', mlp_gc(model)[2::3], [model], ylog=True)
    plot_spectrum('eigvals', mlp_gc(model)[2::3], [model], title="Final Checkpoint Eigvals")
    # plot_group('mean_norm', mlp_gu(model)[2::3], [model])
    # plot_group('mean_frac', mlp_gu(model)[2::3], [model])
    # plot_group('rankme', mlp_gu(model)[2::3], [model], title="Uncentered Covariance RankMe")
    plot_group('rankme', mlp_gc(model)[2::3], [model], title="Centered Covariance Rankme")
    # plot_group('rankme_center_diff', mlp_gc(model)[2::3], [model])
    # plot_group('rankme_center_prop', mlp_gc(model)[2::3], [model])
    grid_show()

for model in filter_model_names:
    grid_start(ncols=2, title=f'"Whole layer K-FAC" up A x down G — {model}')
    plot_group('trace', mlp_lkf(model), [model], ylog=True)
    plot_group('log_det', mlp_lkf(model), [model])
    plot_group('rankme', mlp_lkf(model), [model])
    plot_group('alpha', mlp_lkf(model), [model])
    grid_show()

#############################################################################


# %% [markdown]
# ### Final Residuals

# %%

#############################################################################



# ####        #        #        #        #        #        #        ####
# ####                          Magnitudes                          ####
# ####        #        #        #        #        #        #        ####

# grid_start(ncols=2, title='Up-proj A, G, B magnitudes — OL1B')
# plot_group('trace', mlp_magu_u[::3], ['OLMo-2-0425-1B'], title='trace up A', ylog=True)
# plot_group('log_det', mlp_magu_u[::3], ['OLMo-2-0425-1B'], title='log-det up A')
# plot_group('trace', mlp_magu_u[2::3], ['OLMo-2-0425-1B'], title='trace up G', ylog=True)
# plot_group('log_det', mlp_magu_u[2::3], ['OLMo-2-0425-1B'], title='log-det up G')
# plot_group('trace', mlp_magu_u[1::3], ['OLMo-2-0425-1B'], title='trace up B', ylog=True)
# plot_group('log_det', mlp_magu_u[1::3], ['OLMo-2-0425-1B'], title='log-det up B')
# grid_show()

# grid_start(ncols=2, title='Down-proj A, G, B magnitudes — OL1B')
# plot_group('trace', mlp_magd_u[::3], ['OLMo-2-0425-1B'], title='trace down A', ylog=True)
# plot_group('log_det', mlp_magd_u[::3], ['OLMo-2-0425-1B'], title='log-det down A')
# plot_group('trace', mlp_magd_u[2::3], ['OLMo-2-0425-1B'], title='trace down G', ylog=True)
# plot_group('log_det', mlp_magd_u[2::3], ['OLMo-2-0425-1B'], title='log-det down G')
# plot_group('trace', mlp_magd_u[1::3], ['OLMo-2-0425-1B'], title='trace down B', ylog=True)
# plot_group('log_det', mlp_magd_u[1::3], ['OLMo-2-0425-1B'], title='log-det down B')
# grid_show()

for model in filter_model_names:
    grid_start(ncols=3, title=f'Residual A, G Covariance (uncentered) — {model}')
    plot_group('trace', res_fn_u[::2], [model], ylog=True)
    plot_group('log_det', res_fn_u[::2], [model])
    plot_group('peak_eigval', res_fn_u[::2], [model], ylog=True)
    plot_group('trace', res_fn_u[1::2], [model], ylog=True, disable_didx=[])
    plot_group('log_det', res_fn_u[1::2], [model], disable_didx=[])
    plot_group('peak_eigval', res_fn_u[1::2], [model], ylog=True, disable_didx=[])
    grid_show()

for model in filter_model_names:
    grid_start(ncols=3, title=f'Residual A, G Covariance (centered) — {model}')
    plot_group('trace', res_fn_c[::2], [model], ylog=True)
    plot_group('log_det', res_fn_c[::2], [model])
    plot_group('peak_eigval', res_fn_c[::2], [model], ylog=True)
    plot_group('trace', res_fn_c[1::2], [model], ylog=True, disable_didx=[])
    plot_group('log_det', res_fn_c[1::2], [model], disable_didx=[])
    plot_group('peak_eigval', res_fn_c[1::2], [model], ylog=True, disable_didx=[])
    grid_show()

for model in filter_model_names:
    grid_start(ncols=3, title=f'Residual A RankMe centering difference — {model}')
    plot_group('rankme', res_fn_c[::2], [model])
    plot_group('rankme_center_diff', res_fn_c[::2], [model])
    plot_group('rankme_center_prop', res_fn_c[::2], [model])

    # G is naturally centered
    # plot_group('rankme', res_fn_c[1::2], [model])
    # plot_group('rankme_center_diff', res_fn_c[1::2], [model])
    # plot_group('rankme_center_prop', res_fn_c[1::2], [model])
    grid_show()


# grid_start(ncols=2, title='Residual A x G Covariance — OL1B')
# plot_group('trace', res_afnk, ['OLMo-2-0425-1B'], ylog=True)
# plot_group('log_det', res_afnk, ['OLMo-2-0425-1B'])
# plot_group('trace', res_bfnk, ['OLMo-2-0425-1B'], ylog=True)
# plot_group('log_det', res_bfnk, ['OLMo-2-0425-1B'])
# grid_show()


# %%

# Down-proj output (B) alpha — measured layers, one subplot per model
grid_start(ncols=2, title='Down-proj output (B) alpha')
for model in filter_model_names:
    plot_group('alpha', mlp_bc(model)[2::3], [model], title=get_model_label(model))
grid_show()


# %% [markdown]
#

# %%
