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

# %%
# RQ2 results: depth-resolved excess structure, tail locality, and acts<->grads coupling.
# Single-checkpoint data -> depth profiles (x = block), not training curves.
# Methods: docs/rq2.md; numbers: docs/final_report.md §4.
import os, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from analysis import experiments_lib as _lib
importlib.reload(_lib)
from analysis.experiments_lib import load_results, get_model_label

# %%
MODELS = ['pythia-1b-deduped', 'OLMo-2-0425-1B']
PACKED = {'math web': 'rq2_math_web', 'null (G′ vs G)': 'rq2_general_packed_b'}
PADDED = {'math answers': 'rq2_math_answers', 'quotes': 'rq2_quotes',
          'memorized': 'rq2_memorized', 'null (G′ vs G)': 'rq2_general_padded_b'}
L = 16
LEAVES = {'attn.in': [f'blk{k}.attn.in' for k in range(L)] + ['before_final_norm'],
          'mlp.up.in': [f'blk{k}.mlp.up.in' for k in range(L)]}


def _leaf_stat(cfg, model, leaf, fn):
    res, steps = load_results(cfg, model)
    if res is None: return np.nan
    d = res[steps[-1]].get(leaf, {})
    return fn(d) if d else np.nan


def _excess_mass(d):
    g = d.get('gen_vs_ref', {}).get('acts')
    if g is None: return np.nan
    lam = np.asarray(g['eigvals'], float)
    return float(np.log(np.clip(lam, 1e-12, None))[lam > 1].sum())


def _tail_centroid(d):
    g = d.get('gen_vs_ref', {}).get('acts', {})
    return float(g.get('tail_centroid', np.nan))


def _gen_rankme(d):
    return float(d.get('gen', {}).get('rankme', np.nan))


def depth_profile(model, pops, fn, title, ylabel):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{title} — {get_model_label(model)}')
    for ax, (kind, leaves) in zip(axes, LEAVES.items()):
        for name, cfg in pops.items():
            ys = [_leaf_stat(cfg, model, lf, fn) for lf in leaves]
            ax.plot(range(len(leaves)), ys, marker='o', lw=2, alpha=0.8,
                    ls='--' if 'null' in name else '-', label=name)
        ax.set_title(kind); ax.set_ylabel(ylabel); ax.legend(fontsize=8)
        ax.set_xlabel('block (last = before_final_norm)' if 'attn' in kind else 'block')
    plt.show()

# %% [markdown]
# ### Excess structure over G, by depth (H2.2)
# excess_mass = Σ log λ over geneig λ>1 vs the geometry-matched G. The dashed null is what
# pure sampling noise produces (G′ vs G) — "structure" means clearing that line, not zero.
# Memorized's early-block spike = lexical novelty of high-entropy strings; math-web clears
# the null at all depths (pythia: increasingly with depth; OLMo: strongest early).

# %%
for model in MODELS:
    depth_profile(model, PACKED, _excess_mass, 'Excess over G (packed)', 'excess_mass')

# %%
for model in MODELS:
    pops = {k: v for k, v in PADDED.items() if not (model.startswith('OLMo') and v == 'rq2_memorized')}
    depth_profile(model, pops, _excess_mass, 'Excess over G (padded/last)', 'excess_mass')

# %% [markdown]
# ### Where the excess lives in G's spectrum (tail locality)
# tail_centroid = rank-centroid of the top-8 excess directions' energy over G's EIGENRANK
# (0 = G's top direction, 2048 = G's most minor). H2.2 predicted the tail; measured centroids
# track the null (mid-spectrum) — the excess is built from G's middle-variance directions.

# %%
for model in MODELS:
    depth_profile(model, PACKED, _tail_centroid, 'Tail centroid (packed)', "centroid in G's eigenrank")

# %% [markdown]
# ### Acts↔grads coupling by depth (H2.1)
# RankMe of the geneig spectrum of grads w.r.t. acts at each leaf: flat spectrum (high
# RankMe) = gradient variance proportional to activation variance = coupled. NOTE: raw
# within-population statistic (no G-whitening yet); compare populations within one geometry.

# %%
for model in MODELS:
    pops = {'general': 'rq2_general_padded', **{k: v for k, v in PADDED.items()
            if 'null' not in k and not (model.startswith('OLMo') and v == 'rq2_memorized')}}
    depth_profile(model, pops, _gen_rankme, 'acts↔grads coupling (padded)', 'gen RankMe')

# %% [markdown]
# ### The T-eigenbasis variant of the excess ratio (proposed check)
# r_i = λ_T,i / (v_T,iᵀ Σ_G v_T,i): the T-vs-G variance ratio evaluated at T's OWN principal
# directions (the geneig instead optimizes the ratio over all directions). Computed from the
# stored cov_svd .pt files (heavier cell: loads covariance data, not results).

# %%
import torch
from utils.accessor import DataAccessor
from glob import glob


def t_basis_ratio(t_cfg, g_cfg, model, leaf='blk8.attn.in'):
    at = DataAccessor(sorted(glob(f'data/inferences/{t_cfg}/{model}/step*.pt'))[-1])
    ag = DataAccessor(sorted(glob(f'data/inferences/{g_cfg}/{model}/step*.pt'))[-1])
    lt, vt = at[leaf].acts.eigvals_centered, at[leaf].acts.eigvecs_centered
    Sg = ag[leaf].acts.cov_centered
    denom = ((vt.T.double() @ Sg.double()) * vt.T.double()).sum(1)
    return (lt.double() / denom.clamp(min=1e-12)).numpy()


for model in MODELS:
    plt.figure(figsize=(7, 4))
    for name, cfg in (('math web', 'rq2_math_web'), ('null G′', 'rq2_general_packed_b')):
        r = np.sort(t_basis_ratio(cfg, 'rq2_general_packed', model))[::-1]
        plt.loglog(np.arange(1, len(r) + 1), r, lw=2, label=name)
    plt.axhline(1, color='k', lw=0.8); plt.xlabel('rank'); plt.ylabel('λ_T / var_G along v_T')
    plt.title(f'T-eigenbasis ratio spectra, blk8.attn.in — {get_model_label(model)}')
    plt.legend(); plt.show()

# %% [markdown]
# ### Split-half coherence by depth (the sharp H2.2 test)
# overlap(k) between the two halves' top-k excess-over-G subspaces, RAW space, each half
# whitened by an INDEPENDENT G sample (v1's shared whitener inflated the null to 0.4-0.6;
# see docs/rq2.md §6). Dashed = null (two further independent G pairs).
# Precomputed by the rq2_splithalf_save script -> data/results/rq2_splithalf.npy.

# %%
SH = np.load('data/results/rq2_splithalf.npy', allow_pickle=True).item()
SH_PAIRS = {'packed': ['math_web', 'null_packed'], 'padded': ['quotes', 'memorized', 'null_padded']}

def splithalf_grid(model, k=8):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Split-half excess-subspace overlap (k={k}) — {get_model_label(model)}')
    for ax, (kind, n) in zip(axes, (('attn.in', 17), ('mlp.up.in', 16))):
        for geom, pairs in SH_PAIRS.items():
            for pair in pairs:
                ys = SH.get((model, pair, kind, k))
                if ys is None: continue
                ax.plot(range(len(ys)), ys, marker='o', lw=2, alpha=0.8,
                        ls='--' if 'null' in pair else '-', label=f'{pair} ({geom})')
        ax.set_title(kind); ax.set_ylabel('subspace overlap'); ax.set_ylim(0, 1)
        ax.set_xlabel('block (last = before_final_norm)' if 'attn' in kind else 'block')
        ax.legend(fontsize=8)
    plt.show()

for model in MODELS:
    for k in (8, 32):
        splithalf_grid(model, k)

# %% [markdown]
# ### G-whitened acts↔grads coupling (H2.1, whitened variant)
# Per leaf: task acts whitened by G-acts, task grads by G-grads, then the gen spectrum's
# RankMe. The G′ baseline is the no-real-excess ceiling (both whitened matrices ≈ identity →
# trivially proportional). Every real task falls below it; MATH falls furthest — the scalar
# conflates excess amount with acts/grads mismatch, so read it as gradient-side corroboration
# of H2.2's ordering, not a coupling verdict (docs/final_report.md §4).

# %%
H21W = np.load('data/results/rq2_h21_whitened.npy', allow_pickle=True).item()

def h21w_grid(model):
    plt.figure(figsize=(8, 4))
    for (m, name), ys in H21W.items():
        if m != model: continue
        plt.plot(range(len(ys)), ys, marker='o', lw=2, alpha=0.85,
                 ls='--' if 'baseline' in name else '-', label=name)
    plt.title(f'Whitened acts↔grads gen RankMe — {get_model_label(model)}')
    plt.xlabel('leaf (every 2nd attn.in + before_final_norm)'); plt.ylabel('gen RankMe')
    plt.yscale('log'); plt.legend(fontsize=8); plt.show()

for model in MODELS:
    h21w_grid(model)
