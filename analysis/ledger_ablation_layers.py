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
#     display_name: Python 3
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
from analysis.experiments_lib import plot_layer_contribution, grid_start, grid_show

NB = {'pythia-160m-deduped': 12, 'pythia-410m-deduped': 24, 'pythia-1b-deduped': 16,
      'pythia-6.9b-deduped': 32, 'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32,
      'nanochat-d12': 12}
CFG = lambda m: 'nanochat_samples' if m == 'nanochat-d12' else 'block_representations_samples'

def ledger_ax(ax, cfg, model, layers, ttl):
    """Stacked χ/quality/interference fills over training, summed over `layers` only."""
    r = np.load(f'data/results/{cfg}/results_{model}.npy', allow_pickle=True).item()
    steps = [s for s in sorted(r) if s > 0]
    terms = {y: np.array([sum(float(r[s][f'blk{l}']['block_ledger'][y]) for l in layers)
                          for s in steps]) for y in ('chi', 'quality', 'interference')}
    # refine the grid with sign crossings (interpolated in log-x) so fills close at zero
    lx = np.log(np.array(steps, float))
    cross = []
    for v in terms.values():
        i = np.nonzero(np.signbit(v[:-1]) != np.signbit(v[1:]))[0]
        cross.append(lx[i] + v[i] / (v[i] - v[i + 1]) * (lx[i + 1] - lx[i]))
    grid = np.unique(np.concatenate([lx, *cross]))
    terms = {k: np.interp(grid, lx, v) for k, v in terms.items()}
    gx = np.exp(grid)
    pos, neg = np.zeros(len(gx)), np.zeros(len(gx))
    for name, c in (('chi', 'tab:green'), ('quality', 'tab:red'), ('interference', 'tab:purple')):
        up, dn = np.clip(terms[name], 0, None), np.clip(terms[name], None, 0)
        ax.fill_between(gx, pos, pos + up, label=name, color=c, alpha=0.55, lw=0)
        ax.fill_between(gx, neg, neg + dn, color=c, alpha=0.55, lw=0)
        pos, neg = pos + up, neg + dn
    ax.plot(gx, sum(terms.values()), 'k', lw=2, label='ΔS (group)')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set(xscale='log', xlabel='step', title=ttl)

def ledger_fig(model, ablate):
    """2x2: rows = final layer / last quarter contributions, cols = baseline / ablated."""
    L = NB[model]
    rows = [(f'final layer (blk{L - 1})', [L - 1]),
            (f'last quarter (blk{3 * L // 4}-{L - 1})', list(range(3 * L // 4, L)))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for (label, layers), (axl, axr) in zip(rows, axes):
        ledger_ax(axl, CFG(model), model, layers, f'baseline — {label}')
        ledger_ax(axr, f'ablate_{ablate}', model, layers, f'− {ablate} — {label}')
        axl.set_ylabel('rank entropy')
        lo = min(axl.get_ylim()[0], axr.get_ylim()[0])
        hi = max(axl.get_ylim()[1], axr.get_ylim()[1])
        axl.set_ylim(lo, hi); axr.set_ylim(lo, hi)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'Ledger contributions per layer group — {model}, {ablate} writes zeroed')
    plt.tight_layout()
    outdir = Path('analysis/figures/ledger_ablation_layers')
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f'{model}_{ablate}.pdf', bbox_inches='tight')
    plt.show()

# %%
# pythia-1b (16 blocks), blk3 (the carrier) ablated. Row 1: the final layer's
# contributions (blk15); row 2: the last quarter's (blk12-15). Left baseline, right ablated.
ledger_fig('pythia-1b-deduped', 'blk3')

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
rows = [('blk9', [9]), ('blk10', [10]), ('blk11', [11]), ('blk9-11', [9, 10, 11])]
cols = [('baseline', CFG(model)), ('− blk0-2', 'ablate_blk0-2'), ('− blk3-5', 'ablate_blk3-5')]
fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharex=True)
for (rlabel, layers), axrow in zip(rows, axes):
    for (clabel, cfg), ax in zip(cols, axrow):
        ledger_ax(ax, cfg, model, layers, f'{clabel} — {rlabel}')
    lo = min(a.get_ylim()[0] for a in axrow)
    hi = max(a.get_ylim()[1] for a in axrow)
    for a in axrow:
        a.set_ylim(lo, hi)
    axrow[0].set_ylabel('rank entropy')
axes[0, 0].legend(fontsize=7)
fig.suptitle(f'Ledger contributions of late layers — {model}, baseline vs early-block ablations')
plt.tight_layout()
outdir = Path('analysis/figures/ledger_ablation_layers')
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / f'{model}_late_layers.pdf', bbox_inches='tight')
plt.show()

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
