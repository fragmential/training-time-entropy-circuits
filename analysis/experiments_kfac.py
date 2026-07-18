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
# # K-FAC experiments (kfac_small_shuffled)
#
# The K-FAC material split out of experiments_padded, kept runnable: everything built on
# the `kfac_small_shuffled` config (K-FAC factors, up/gate/down-proj input/output covariances,
# sampled gradients, whole-layer K-FAC, final-residual A/G covariances, down-proj alpha).

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
    grid_start, grid_show, plot_group, plot_spectrum, build_hooks, get_model_label,
)

# %%
FINEWEB_PADDED = "rankme_fineweb_padded"
TRAINSET_PACKED = "rankme_trainset_packed_4MT"
# KF_ARXIV = "kfac_small_arxiv"
KF_SMALL = "kfac_small_shuffled"

# %%
HK = build_hooks()   # hook-name -> (node, metric); machinery lives in experiments_lib

# %%
filter_model_names = [
    'pythia-1b-deduped',
    'pythia-6.9b-deduped',
    'OLMo-2-0425-1B',
    'OLMo-2-1124-7B',
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


# %% [markdown]
# ### Final residual RankMe, Trace, and Mean

# %%
grid_start(ncols=2, title='Final residual RankMe (kfac run)')
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
