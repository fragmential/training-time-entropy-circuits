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
# Mean output-vocabulary entropy of the logit-lens readout at every layer, per checkpoint.
# Two protocols:
# - **Own-generations (DEFAULT, nanochat only for now):** the entropy-lens paper's protocol —
#   64 BOS-seeded generations at temperature 0.7, 32 tokens each, entropy at the generated
#   positions, softcap applied (`oneoff_scripts/entropy_own_gen.py`,
#   `data/results/vocab_entropy_own_gen`). Data-free: the model's inherent profile.
# - **Teacher-forced (secondary):** rides the collection forward passes over the eval batch
#   (`vocab_entropy: true`); covers nanochat-d12 (config `nanochat`) AND the four main models
#   (config `vocab_entropy`) — the cross-architecture comparison lives on this one.
#
# **Comparison to the old attention-experiments report (Project_AI_report):** the old plot's
# "final layer roughly constant through pretraining" is a SOFTCAP-OMISSION ARTIFACT and must
# not be used as a baseline. Its lens called `lm_head(norm(h))` without nanochat's 15·tanh
# logit softcap, measuring a sharpened distribution the model never emits (entropy ~1.8 nats,
# far below the ≥3.0 val-CE floor — impossible for the true predictive distribution, and
# trend-flat by accident). This lens applies the cap; the final-layer downtrend is the real
# signal and shadows val CE almost exactly (3.46→3.13 vs CE 3.49→3.05 over steps 1050→7080).
# Verified by a controlled sweep: cap on/off flips the trend under EITHER data regime; data
# (own-generations vs teacher-forced fineweb, 32 vs 512 ctx) moves levels ~0.3–0.5 nats, never
# the trend. Caveat: intermediate-layer levels are cap-inflated too (a modeling choice — those
# residuals never pass the head), consistent across checkpoints; Pythia/OLMo have no cap.

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
# ## Own-generations profile (default view) — nanochat-d12
# The old report's exact spec (BOS seed, temp 0.7, 32×64) with the softcap applied. One line
# per layer over training, plus the layer profile at selected checkpoints.

# %%
og_res, og_steps = load_results('vocab_entropy_own_gen', 'nanochat-d12')
OG = np.array([og_res[s]['vocab_entropy']['entropy_lens_own_gen']['per_layer'] for s in og_steps])
og_names = ['emb'] + [f'blk{i}' for i in range(OG.shape[1] - 1)]

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for i, name in enumerate(og_names):
    axes[0].plot(og_steps, OG[:, i], lw=1.8, alpha=0.85,
                 color=plt.cm.viridis(i / (len(og_names) - 1)), label=name)
axes[0].set(xscale='log', xlabel='step', ylabel='vocab entropy (nats)',
            title='per layer over training (viridis: emb → last)')
axes[0].legend(fontsize=6, ncol=2)
for j in [0, len(og_steps) // 4, len(og_steps) // 2, -1]:
    axes[1].plot(range(len(og_names)), OG[j], marker='o', lw=2, label=f'step {og_steps[j]}')
axes[1].set(xlabel='layer', ylabel='vocab entropy (nats)', title='layer profile at selected checkpoints')
axes[1].set_xticks(range(len(og_names))); axes[1].set_xticklabels(og_names, fontsize=7)
axes[1].legend(fontsize=8)
fig.suptitle('Own-generations entropy lens (softcap applied) — nanochat-d12')
plt.tight_layout(); plt.show()

# %% [markdown]
# ### Protocol comparison: own-generations vs teacher-forced, final layer
# The final layer is the model's actual predictive entropy under each protocol. Expectation
# from the softcap sweep: levels differ ~0.3–0.5 nats (own-gen text is lower-entropy than
# fineweb under teacher forcing), the downtrend agrees.

# %%
plt.figure(figsize=(8, 4))
plt.plot(og_steps, OG[:, -1], lw=2, label='own-generations (default)')
plt.plot(steps, E[:, -1], lw=2, label='teacher-forced (eval batch)')
plt.xscale('log'); plt.xlabel('step'); plt.ylabel('final-layer vocab entropy (nats)')
plt.title('Predictive entropy, protocol comparison — nanochat-d12'); plt.legend(); plt.show()

# %% [markdown]
# ## Teacher-forced lens (secondary view)
#
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
