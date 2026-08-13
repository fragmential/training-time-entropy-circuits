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
# # Ledger contributions per layer group, baseline vs ablation
#
# The showcase §3 stacked decomposition (ΔS = χ + quality + interference), but summed
# over a chosen layer group only, side by side: left the unablated packed sweep, right
# the run with a block's writes zeroed (data/results/ablate_*).

# %%
import os, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from analysis import experiments_lib as _lib
importlib.reload(_lib)
from analysis.experiments_lib import (plot_layer_contribution, plot_ledger_stack,
                                      grid_start, grid_show)

NB = {'pythia-160m-deduped': 12, 'pythia-410m-deduped': 24, 'pythia-1b-deduped': 16,
      'pythia-6.9b-deduped': 32, 'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32,
      'nanochat-d12': 12}
CFG = lambda m: 'nanochat_samples' if m == 'nanochat-d12' else 'block_representations_samples'
QUARTERS = {12: ['blk0-2', 'blk3-5', 'blk6-8', 'blk9-11', 'blk3-8'],
            16: ['blk0-3', 'blk4-7', 'blk8-11', 'blk12-15', 'blk4-11'],
            24: ['blk0-5', 'blk6-11', 'blk12-17', 'blk18-23', 'blk6-17'],
            32: ['blk0-7', 'blk8-15', 'blk16-23', 'blk24-31', 'blk8-23']}
CARRIER = {'pythia-1b-deduped': 'blk3', 'pythia-6.9b-deduped': 'blk4-5'}

def ledger_grid(model, rows, cols, title, savedir, figsize=None):
    """Grid of experiments_lib.plot_ledger_stack panels: rows = (label, block list),
    cols = (label, results dir). sharey='row' keeps each row's panels comparable without
    flattening single-layer rows against whole-quarter ones; all_xlabels keeps the token
    axis on every row. emb=False: the embedding line is only meaningful full-depth."""
    grid_start(ncols=len(cols), figsize=figsize or (6.5 * len(cols), 4 * len(rows)),
               sharex=True, sharey='row', all_xlabels=True, emb=False, zero_line=True,
               title=title, savedir=savedir)
    for i, (rlabel, layers) in enumerate(rows):
        for j, (clabel, cfg) in enumerate(cols):
            plot_ledger_stack(model, cfg, blocks=layers, title=f'{clabel} — {rlabel}',
                              ylabel='rank entropy' if j == 0 else None,
                              legend=7 if i == 0 and j == 0 else None)
    grid_show()

def ledger_fig(model, ablate):
    """3x2: rows = penultimate / final layer / last quarter, cols = baseline / ablated."""
    L = NB[model]
    rows = [(f'penultimate layer (blk{L - 2})', [L - 2]),
            (f'final layer (blk{L - 1})', [L - 1]),
            (f'last quarter (blk{3 * L // 4}-{L - 1})', list(range(3 * L // 4, L)))]
    ledger_grid(model, rows, [('baseline', CFG(model)), (f'− {ablate}', f'ablate_{ablate}')],
                f'Ledger contributions per layer group — {model}, {ablate} writes zeroed',
                f'ledger_ablation_layers/{model}_{ablate}')

# %%
# pythia-1b (16 blocks), blk3 (the carrier) ablated. Rows: the penultimate layer's
# contributions (blk14), the final layer's (blk15), the last quarter's (blk12-15).
# Left baseline, right ablated.
ledger_fig('pythia-1b-deduped', 'blk3')
# FIGURE B2?.?
# Not sure where. but this needs to be in. Also the pythia 6.9b version but that's blk3-4 which we currently don't have yet.

# %%
# The other models: the carrier where one exists (pythia-6.9b: blk4-5), otherwise the
# first-quarter ablation — the early-blocks analog of blk3 (no other single-block runs).
ABLATIONS = {'pythia-160m-deduped': 'blk0-2', 'pythia-410m-deduped': 'blk0-5',
             'pythia-6.9b-deduped': 'blk4-5', 'OLMo-2-0425-1B': 'blk0-3',
             'OLMo-2-1124-7B': 'blk0-7', 'nanochat-d12': 'blk0-2'}
for model, ablate in ABLATIONS.items():
    ledger_fig(model, ablate)

# %%
# nanochat-d12: the late layers 9/10/11 separately and together, baseline vs the two
# early-block ablations.
model = 'nanochat-d12'
ledger_grid(model,
            [('blk9', [9]), ('blk10', [10]), ('blk11', [11]), ('blk9-11', [9, 10, 11])],
            [('baseline', CFG(model)), ('− blk0-2', 'ablate_blk0-2'), ('− blk3-5', 'ablate_blk3-5')],
            f'Ledger contributions of late layers — {model}, baseline vs early-block ablations',
            f'ledger_ablation_layers/{model}_late_layers', figsize=(16, 14))

# %%
# The per-block depth stacks (the experiments.ipynb ledger stacks), baseline vs ablated,
# per ledger term. The embedding entropy line only makes sense on the ΔS stack.
TERM_LABEL = {'delta_s': 'signed ΔS', 'quality': 'quality', 'interference': 'interference'}

def ledger_term_stacks(mdl, ablate, terms=('delta_s', 'quality', 'interference')):
    srcs = lambda cfg: [(cfg, (f'blk{l}', 'block_ledger'), f'blk {l}') for l in range(NB[mdl])]
    emb = lambda cfg: (cfg, ('blk0.attn.in', 'acts_centered'), 'embedding', 'matrix_entropy')
    for term in terms:
        grid_start(ncols=2, sharey=True,
                   title=f'{mdl} — rank-entropy ledger ({TERM_LABEL[term]}), baseline vs − {ablate}')
        for ttl, cfg in (('baseline', CFG(mdl)), (f'− {ablate}', f'ablate_{ablate}')):
            plot_layer_contribution(mdl, srcs(cfg), yvar=term, normalize=False, title=ttl,
                                    alpha=0.85,
                                    **({'baseline_src': emb(cfg)} if term == 'delta_s' else {}))
        grid_show()

ledger_term_stacks('nanochat-d12', 'blk3-5')

# %%
# pythia-1b: the same three depth stacks, baseline vs the blk3 (carrier) ablation.
ledger_term_stacks('pythia-1b-deduped', 'blk3')

# %%
for model, nb in NB.items():
    ledger_term_stacks(model, QUARTERS[nb][0])

# %% [markdown]
# ## OLMo-2: single mid-late layer + third quarter, baseline vs second-quarter ablation

# %%
def ledger_pair(model, single):
    """2x2 ledger: rows = single layer / third quarter, cols = baseline / − second quarter."""
    L = NB[model]
    q2 = f'blk{L // 4}-{L // 2 - 1}'
    q3 = list(range(L // 2, 3 * L // 4))
    rows = [(f'blk{single}', [single]), (f'third quarter (blk{q3[0]}-{q3[-1]})', q3)]
    ledger_grid(model, rows, [('baseline', CFG(model)), (f'− {q2}', f'ablate_{q2}')],
                f'Ledger contributions — {model}, baseline vs {q2} writes zeroed',
                f'ledger_ablation_layers/{model}_q2_ablation_single_q3', figsize=(13, 8))

# %%
ledger_pair('OLMo-2-0425-1B', 10)

# %%
ledger_pair('OLMo-2-1124-7B', 16)

# %% [markdown]
# ## OLMo-2: the middle four layers, no ablation
#
# Both models in one figure, each panel the baseline ledger stack summed over that model's
# four central blocks (1B: blk6-9, 7B: blk14-17).

# %%
OLMO = ['OLMo-2-0425-1B', 'OLMo-2-1124-7B']
mid4 = lambda L: list(range(L // 2 - 2, L // 2 + 2))     # the four blocks straddling mid-depth

grid_start(ncols=len(OLMO), figsize=(13, 4.5), sharex=True, emb=False, zero_line=True,
           title='Ledger contributions of the middle four layers — OLMo-2, no ablation',
           savedir='ledger_ablation_layers/olmo_middle4')
for j, model in enumerate(OLMO):
    layers = mid4(NB[model])
    plot_ledger_stack(model, CFG(model), blocks=layers,
                      title=f'{model} — blk{layers[0]}-{layers[-1]}',
                      ylabel='rank entropy' if j == 0 else None, legend=7 if j == 0 else None)
grid_show()
