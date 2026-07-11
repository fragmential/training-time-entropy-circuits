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
# # Vocabulary-space entropy lens
# The entropy-lens metric from the nanochat session: mean output-vocabulary entropy of the
# logit-lens readout at every layer, per checkpoint (`entropy_lens.per_layer`). Covers
# nanochat-d12 (config `nanochat`) AND the four main models (config `vocab_entropy`) — the
# cross-architecture comparison this exists for. Staging ground before joining the main story.

# %%
import os, sys
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from analysis.experiments_lib import load_results

RUNS = {'nanochat-d12': 'nanochat', 'pythia-1b-deduped': 'vocab_entropy',
        'pythia-6.9b-deduped': 'vocab_entropy', 'OLMo-2-0425-1B': 'vocab_entropy',
        'OLMo-2-1124-7B': 'vocab_entropy'}

def entropy_of(model):
    res, steps = load_results(RUNS[model], model)
    E = np.array([res[s]['vocab_entropy']['entropy_lens']['per_layer'] for s in steps])
    return E, res[steps[0]]['vocab_entropy']['entropy_lens']['layers'], steps, res

E, layers, steps, res = entropy_of('nanochat-d12')

# %% [markdown]
# ### Per-layer entropy over training
# Each line = one layer's logit-lens vocabulary entropy across checkpoints (x = step). The
# final layer's curve is the model's actual predictive entropy; earlier layers show how far
# down the stack the eventual prediction sharpens.

# %%
plt.figure(figsize=(9, 5))
for i, name in enumerate(layers):
    plt.plot(steps, E[:, i], lw=2, alpha=0.85,
             color=plt.cm.viridis(i / (len(layers) - 1)), label=name)
plt.xscale('log'); plt.xlabel('step'); plt.ylabel('vocab entropy (nats)')
plt.title('Entropy lens per layer — nanochat-d12'); plt.legend(fontsize=7, ncol=2); plt.show()

# %%
plt.figure(figsize=(9, 4))
plt.imshow(E.T, aspect='auto', cmap='magma', origin='lower',
           extent=(0, len(steps), -0.5, len(layers) - 0.5))
plt.yticks(range(len(layers)), layers, fontsize=7)
plt.xlabel('checkpoint index'); plt.colorbar(label='vocab entropy (nats)')
plt.title('Entropy lens, layers × training — nanochat-d12'); plt.show()

# %% [markdown]
# ### Per-layer PROFILE view (the old repo's presentation)
# The previous project plotted entropy vs LAYER (one line per model/checkpoint), not vs time —
# that axis difference is most of why the training-view above "looks nothing like" the old
# plots. Absolute levels also differ: the old CSVs were computed on that project's own sample
# set; these use this repo's eval batch.

# %%
picks = [0, len(steps) // 4, len(steps) // 2, -1]
plt.figure(figsize=(8, 4.5))
for j in picks:
    plt.plot(range(len(layers)), E[j], marker='o', lw=2, label=f'step {steps[j]}')
plt.xticks(range(len(layers)), layers, fontsize=7)
plt.xlabel('layer'); plt.ylabel('vocab entropy (nats)')
plt.title('Entropy-lens layer profile at selected checkpoints — nanochat-d12')
plt.legend(fontsize=8); plt.show()

# %%
# Linear-x version of the training view (the log-x above compresses late training).
plt.figure(figsize=(9, 5))
for i, name in enumerate(layers):
    plt.plot(steps, E[:, i], lw=2, alpha=0.85, color=plt.cm.viridis(i / (len(layers) - 1)), label=name)
plt.xlabel('step (linear)'); plt.ylabel('vocab entropy (nats)')
plt.title('Entropy lens per layer, linear steps — nanochat-d12'); plt.legend(fontsize=7, ncol=2); plt.show()

# %% [markdown]
# ### Final-layer entropy vs the RankMe phases
# Overlay of the last layer's predictive entropy with the final-stream centered RankMe from the
# same run — do the RankMe phases have a visible signature in predictive entropy?

# %%
rk = [res[s]['after_final_norm']['acts_centered']['rankme'] for s in steps]
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(steps, E[:, -1], 'tab:red', lw=2, label='final-layer vocab entropy')
ax1.set(xscale='log', xlabel='step', ylabel='vocab entropy (nats)')
ax2 = ax1.twinx()
ax2.plot(steps, rk, 'tab:blue', lw=2, label='RankMe (after_final_norm, centered)')
ax2.set_ylabel('RankMe')
fig.legend(loc='upper right', fontsize=8)
plt.title('Predictive entropy vs RankMe — nanochat-d12'); plt.show()


# %% [markdown]
# ### Cross-architecture comparison: layer profiles at matched relative depth
# Final-checkpoint entropy-lens profiles for all five models, x = relative depth so different
# layer counts overlay. The nanochat-vs-pythia-410m token-matched comparison belongs here once
# 410m carries the metric.

# %%
plt.figure(figsize=(9, 5))
for model in RUNS:
    Em, lay, st, _ = entropy_of(model)
    x = np.linspace(0, 1, len(lay))
    plt.plot(x, Em[-1], marker='o', lw=2, alpha=0.85, label=f'{model} (final)')
plt.xlabel('relative depth (emb → last block)'); plt.ylabel('vocab entropy (nats)')
plt.title('Entropy-lens profiles, final checkpoints — all architectures')
plt.legend(fontsize=8); plt.show()

# %%
# Per-model training views (log steps), one panel each.
fig, axes = plt.subplots(1, len(RUNS), figsize=(4 * len(RUNS), 3.6), sharey=False)
for ax, model in zip(axes, RUNS):
    Em, lay, st, _ = entropy_of(model)
    for i in range(len(lay)):
        ax.plot(st, Em[:, i], lw=1.5, alpha=0.8, color=plt.cm.viridis(i / (len(lay) - 1)))
    ax.set(xscale='log', title=model, xlabel='step')
axes[0].set_ylabel('vocab entropy (nats)')
fig.suptitle('Entropy lens per layer over training (viridis: emb → last)')
plt.tight_layout(); plt.show()
