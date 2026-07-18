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
# # Showcase appendix
# Appendix-grade material split out of showcase.ipynb: (A) the dataset-swap control,
# (B) the toy edge cases behind [docs/toy_fig4_addendum.md](../docs/toy_fig4_addendum.md).
# Toy data: `data/results/toy/appendix_sweeps.pt`, regenerated in ~2 min by
# `uv run python -m toy.appendix_sweeps` (single-layer dynamics, canonical clustered init,
# lr 0.25).

# %%
import os, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import matplotlib.pyplot as plt
from analysis import experiments_lib as _lib
importlib.reload(_lib)
from analysis.experiments_lib import get_model_label

BLOCK_SAMPLES = 'block_representations_samples'
SWAP = 'block_representations_samples_swap'
PADDED = 'block_representations_samples_padded'

def caption(txt):
    """Paper-style figure caption, rendered below the figure it follows."""
    from IPython.display import display, HTML
    display(HTML(f'<div style="font-size:0.97em; opacity:0.95; max-width:56em; '
                 f'margin:0.2em 0 1.2em 0.5em"><b>Figure.</b> {txt}</div>'))

# %%
# Shared data + helpers (no plots in this cell): toy sweeps, fork/peak detectors.
SW = torch.load('data/results/toy/appendix_sweeps.pt', weights_only=True)

def fork_step(r):                      # first step the rare pair leaves cos > 0.97
    ok = (r['cos_w'] > 0.97).numpy()
    i = int(np.argmax(ok))
    j = int(np.argmax(~ok[i:])) + i
    return j if not ok[j:].all() else None

def peak_step(r, skip=4):
    rm = r['rm'].numpy()
    return int(np.argmax(rm[skip:])) + skip

# %% [markdown]
# ## A. The dataset-swap control (data vs model)
# Each 1B model re-run on the OTHER family's pretraining mix (same texts, re-tokenized).
# Shown: Σquality — the family-discriminating ledger term. The full multi-metric verdict
# table (all ledger terms, rogue diagnostics, coupling) is in docs/swap_run_findings.md.

# %%
n_blocks_swap = {'pythia-1b-deduped': 16, 'OLMo-2-0425-1B': 16}

def swap_quality(model):
    out = {}
    for src, lbl in ((BLOCK_SAMPLES, 'original data'), (SWAP, 'swapped data')):
        q = np.sum([_lib.get_ys(src, model, (f'blk{l}', 'block_ledger'), 'quality')[0]
                    for l in range(n_blocks_swap[model])], axis=0)
        steps = _lib.get_ys(src, model, ('blk0', 'block_ledger'), 'chi')[1]
        out[lbl] = (steps, q)
    return out

plt.figure(figsize=(9, 4.5))
for model, c in (('pythia-1b-deduped', 'tab:red'), ('OLMo-2-0425-1B', 'tab:blue')):
    for lbl, (st, q) in swap_quality(model).items():
        plt.plot(st, q, color=c, lw=2, ls='--' if 'swap' in lbl else '-',
                 label=f'{get_model_label(model)}, {lbl}')
plt.xscale('log'); plt.xlabel('step'); plt.ylabel('Σ quality (over blocks)')
plt.title('Dataset-swap control: the ledger driver follows the MODEL, not the data')
plt.legend(fontsize=8); plt.show()
caption('Dataset-swap control: summed quality ledger term for each 1B model on its own '
        'pretraining mix (solid) vs re-run on the OTHER family\'s mix, same texts re-tokenized '
        '(dashed). If the family split (Pythia quality-collapsing, red, vs OLMo-2 '
        'quality-recovering, blue) followed the data, the dashed curves would swap sides; '
        'they track their solid counterparts instead. Σquality is shown because it is the '
        'discriminating term; the swap verdict holds across ALL tracked metrics — full table '
        'in docs/swap_run_findings.md.')

# %% [markdown]
# ## B. Toy edge cases of the Fig-4 compression transient
#
# Vocabulary used throughout (defined once): the two rare classes start at the same point
# and travel together; the **fork** is the moment their weight vectors separate. The fork
# changes the covariance structure of the features, which widens the gap between the two
# eigenvalues — we call that one-time push the **kick**. RankMe falls when the eigenvalue
# gap widens relative to the spectrum, so the kick is what can produce a visible decline.
# Each subsection below answers one question about when that decline actually appears.

# %% [markdown]
# ### B1. Does the TIMING of the fork decide whether the decline appears? (δ sweep)
# **Question.** The jitter δ is the tiny random offset that breaks the rare classes'
# symmetry; the fork arrives later the smaller δ is (fork time ≈ log(1/δ) / growth rate).
# If everything else is fixed and only δ moves, does the decline's size depend only on WHEN
# the fork lands?
# **Why it matters.** If yes, "compression" in this toy is not a property of the learning
# problem but of an event's timing relative to the observation window — which is the core
# of the fourth-condition claim in the addendum.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
deltas = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
cmap = plt.get_cmap('viridis')
rows = []
for i, d in enumerate(deltas):
    for s in (0, 1, 2):
        r = SW['delta'][f'{d:g}/s{s}']
        rm = r['rm'].numpy()
        axes[0].plot(rm, color=cmap(i / (len(deltas) - 1)), lw=1.2, alpha=0.8,
                     label=f'δ={d:g}' if s == 0 else None)
        pk = peak_step(r)
        rows.append((d, s, fork_step(r), pk, rm[pk] - rm[pk:].min()))
axes[0].set(xlabel='step', ylabel='RankMe (uncentered features)', title='RankMe by δ (3 seeds each)')
axes[0].legend(fontsize=8)
for d in deltas:
    grp = [r for r in rows if r[0] == d and r[2] is not None]
    if grp:
        axes[1].scatter([np.mean([g[2] for g in grp])], [np.mean([g[4] for g in grp])], s=70)
        axes[1].annotate(f'δ={d:g}', (np.mean([g[2] for g in grp]), np.mean([g[4] for g in grp])),
                         fontsize=8, xytext=(4, 4), textcoords='offset points')
axes[1].set(xlabel='fork-onset step (mean over seeds)', ylabel='RankMe decline (peak − post-peak min)',
            title='decline vs fork timing: the window')
plt.show()
caption('Answer: yes — decline size is a function of fork timing alone here. Left: RankMe '
        'trajectories; smaller δ (darker) forks later. Right: decline size against fork '
        'time, an inverted U. Early forks (δ=1e-1) land while RankMe is still rising, so '
        'the kick is absorbed by the rise; late forks (δ ≤ 1e-4) land at or beyond the '
        '300-step window edge, so the decline is cut off; only the middle (δ ∈ [1e-3, '
        '1e-2]) shows the full decline. Logical scope: this proves timing is DECISIVE when '
        'only δ varies; it does not by itself rule out other routes to a decline — section '
        'B3 covers whether the kick exists in the runs where no decline is visible.')

# %% [markdown]
# ### B2. Why can the decline start LATER than the visible fork? (rare-pair offset rx)
# **Question.** In some runs the trajectories visibly fork while RankMe is still flat, and
# the decline only starts tens of steps later. What controls that delay?
# **Mechanism in words.** The kick's size grows with the SQUARE of how far the two rare
# classes have separated (it is a covariance, a product of two coordinates). If the pair
# starts away from the origin (offset rx), the early separation is small relative to the
# pair's position, the squared quantity stays negligible, and RankMe reacts only once the
# separation is large — a delay. If the pair starts AT the origin, any separation is
# immediately large relative to position, and the decline starts with the fork.
# **Why it matters.** The paper's figure shows the dashed (compression) styling beginning
# exactly at the fork; that is reproduced here only with the origin start — which is the
# evidence that their rare pair initializes at the origin.

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for rx, c in zip(('0', '-0.1', '-0.25', '-0.5'), plt.get_cmap('plasma')(np.linspace(0.1, 0.8, 4))):
    r = SW['rx'][rx]
    rm, fk, pk = r['rm'].numpy(), fork_step(r), peak_step(r)
    ax.plot(rm, color=c, lw=1.6, label=f'rx={rx}: fork {fk}, peak {pk}, lag {pk - fk if fk else "—"}')
    if fk: ax.axvline(fk, color=c, lw=0.8, ls=':')
    ax.axvline(pk, color=c, lw=0.8, ls='--')
ax.set(xlabel='step', ylabel='RankMe', title='fork onset (dotted) vs decline onset (dashed) by rx')
ax.legend(fontsize=8, loc='upper left')
axz = ax.inset_axes([0.55, 0.07, 0.43, 0.5])            # zoom: the peak/decline region
for rx, c in zip(('0', '-0.1', '-0.25', '-0.5'), plt.get_cmap('plasma')(np.linspace(0.1, 0.8, 4))):
    r = SW['rx'][rx]
    rm, fk, pk = r['rm'].numpy(), fork_step(r), peak_step(r)
    axz.plot(rm, color=c, lw=1.4)
    if fk: axz.axvline(fk, color=c, lw=0.8, ls=':')
    axz.axvline(pk, color=c, lw=0.8, ls='--')
axz.set(xlim=(150, len(rm) - 1), ylim=(1.99, 2.005))
axz.tick_params(labelsize=7)
ax.indicate_inset_zoom(axz, edgecolor='0.4')
plt.show()
caption('Answer: the rare pair\'s starting distance from the origin controls the delay. '
        'Dotted vertical = fork onset (the weights\' cosine drops below 0.97); dashed '
        'vertical = RankMe peak, where the decline starts. Moving the pair from the origin '
        'to (−0.5, 0) stretches the delay from ~5 to ~41 steps and shrinks the decline '
        '(0.0067 → 0.0029), exactly as the squared-separation account predicts. At the '
        'origin the two verticals coincide, matching the paper\'s dashes-at-the-fork '
        'rendering. Logical scope: a monotone four-point trend consistent with the '
        'mechanism; the quantitative threshold model (~1/3 of final separation) is checked '
        'in the addendum, not here.')

# %% [markdown]
# #### The same four runs in Li et al's Fig-4 style
# Classifier rows $W_i$ and features $f_\theta(x)$ as 2-D trajectories, phase-styled as in
# the paper (dotted warmup, solid expansion, dashed compression; gray dot = start), plus
# RankMe with the top two eigenvalues. One figure per rare-pair offset rx. The fork
# (dotted vertical line) and the decline onset (dashed vertical line) drift apart as the
# pair starts further from the origin.

# %%
CLASS_COLORS4 = ('magenta', 'orange', 'royalblue', 'seagreen')
CLASS_MARKERS4 = ('^', 'o', 's', 'D')
PHASES4 = (':', '-', '--')                               # warmup / expansion / compression
Y4 = (0, 0, 1, 1, 2, 3)                                  # class of each feature row

def _bounds4(rm):
    pk = int(np.argmax(rm))
    return int(np.argmin(rm[:pk + 1])), pk

def _phased4(ax, path, bounds, color, marker):
    for lo, hi, style in zip((0, *bounds), (*bounds, len(path) - 1), PHASES4):
        ax.plot(path[lo:hi + 1, 0], path[lo:hi + 1, 1], style, color=color, lw=1.5)
    ax.scatter(*path[0], color='0.6', s=18, zorder=2)
    ax.scatter(*path[-1], color=color, s=60, marker=marker, zorder=3)

for rx in ('0', '-0.1', '-0.25', '-0.5'):
    r = SW['rx'][rx]
    rm, lam = r['rm'].numpy(), r['lam'].numpy()
    bounds, fk = _bounds4(rm), fork_step(r)
    fig, (bx, cx, dx) = plt.subplots(1, 3, figsize=(15, 4.2))
    for i in range(4):
        _phased4(bx, r['W_path'][:, :, i].numpy(), bounds, CLASS_COLORS4[i], CLASS_MARKERS4[i])
    for j, c in enumerate(Y4):
        _phased4(cx, r['theta_path'][:, j, :].numpy(), bounds, CLASS_COLORS4[c], CLASS_MARKERS4[c])
    for ax in (bx, cx):
        m = 1.1 * max(abs(v) for v in (*ax.get_xlim(), *ax.get_ylim()))
        ax.set(xlabel='dim 1', ylabel='dim 2', aspect='equal', xlim=(-m, m), ylim=(-m, m))
    bx.set_title('$W_i$')
    cx.set_title(r'$f_\theta(x)$')
    steps = np.arange(len(rm))
    for lo, hi, style in zip((0, *bounds), (*bounds, len(rm) - 1), PHASES4):
        dx.plot(steps[lo:hi + 1], rm[lo:hi + 1], style, color='navy', lw=2)
    ex = dx.twinx()
    ex.plot(steps, lam[:, 0], color='violet', lw=1.5, label=r'$\lambda_1$')
    ex.plot(steps, lam[:, 1], color='yellowgreen', lw=1.5, label=r'$\lambda_2$')
    ex.set_ylabel(r'eigenvalues $\lambda_i$')
    ex.legend(fontsize=8, loc='center right')
    if fk is not None:
        dx.axvline(fk, color='gray', lw=0.9, ls=':')
    dx.axvline(bounds[1], color='gray', lw=0.9, ls='--')
    dx.set(title='RankMe & eigenvalues', xlabel='step', ylabel='RankMe')
    fig.suptitle(f'rare-pair offset rx = {rx}  (fork: dotted vertical; decline onset: dashed vertical)')
    plt.tight_layout()
    plt.show()
caption('The four rx runs in the paper\'s presentation. As the rare pair starts further '
        'from the origin (top to bottom figure), the shared path the blue/green singleton '
        'classes travel gets longer before the fork, and the gap between the fork (dotted '
        'vertical) and the decline onset (dashed vertical) widens from zero to tens of '
        'steps, while the printed decline shrinks — both as the squared-separation account '
        'predicts. The dashed compression styling in the trajectory panels begins at the '
        'RankMe peak, so at rx = 0 it starts at the visible split (the paper\'s rendering) '
        'and at larger offsets it starts after it.')

# %% [markdown]
# ### B3. When no decline is visible, is the kick absent — or hidden? (init blend)
# **Question.** Runs started from small random (iid) init show no decline at all. Does the
# fork simply not produce a kick there, or does the kick happen and get hidden under the
# still-rising RankMe curve?
# **How we test it.** Blend the constructed init toward iid-tiny in steps (mix 0 → 1) and,
# instead of looking at RankMe (which confounds the kick with its baseline), look directly
# at the DERIVATIVE of the eigenvalue-gap: if the fork pushes the gap open, that derivative
# must show a positive bump at fork time in every run, visible decline or not.
# **Why it matters.** This separates two different claims: "the mechanism requires the
# constructed init" (false) vs "SEEING the mechanism requires the fork to land on a flat
# baseline" (true). The addendum's earlier phrasing conflated them.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for a, c in zip(('0', '0.25', '0.5', '1'), plt.get_cmap('cividis')(np.linspace(0.1, 0.85, 4))):
    r = SW['homotopy'][a]
    rm, lam = r['rm'].numpy(), r['lam'].numpy()
    axes[0].plot(rm, color=c, lw=1.6, label=f'iid mix {a}')
    ratio = np.log(lam[:, 0] / np.clip(lam[:, 1], 1e-12, None))
    axes[1].plot(np.convolve(np.diff(ratio), np.ones(9) / 9, 'same'), color=c, lw=1.4)
axes[0].set(xlabel='step', ylabel='RankMe', title='RankMe along the homotopy')
axes[0].legend(fontsize=8)
axes[1].axhline(0, color='gray', lw=0.8)
axes[1].set(xlabel='step', ylabel=r'd log($\lambda_1/\lambda_2$)/dt (smoothed)',
            title='the kick is present in every run — a positive bump at the fork')
plt.show()
caption('Answer: hidden, not absent. Left: RankMe — the visible decline dies as the init '
        'blends toward iid (mix 0 → 1). Right: the eigenvalue-gap derivative — EVERY run, '
        'including the ones with no visible decline, shows the positive bump at its fork. '
        'The bump simply sits on a baseline that is still rising (the spectrum is still '
        'filling out), so it never appears as a downturn in RankMe. Conclusion: the fork '
        'always delivers the kick; a flat (saturated) baseline at fork time is what makes '
        'it VISIBLE. Logical scope: four blend levels, one seed each — the direction of the '
        'result is unambiguous (bump present in all), but the bump-size trend along the '
        'blend is not established here.')

# %% [markdown]
# ### B4. Is the compression permanent, or does it wash out? (long horizon)
# **Question.** The paper's window ends at step 300, right after the decline begins. If
# training simply continues, does the compressed state persist?
# **Mechanism in words.** The kick is a one-time, fixed-size change to the covariance;
# meanwhile cross-entropy keeps growing every class's weights (roughly like log t) without
# bound. A fixed-size effect divided by an ever-growing total shrinks toward zero, so the
# spectrum should drift back to balance.

# %%
r = SW['long']['canonical']
rm = r['rm'].numpy()
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(rm, lw=1.6)
ax.axvline(300, color='gray', lw=0.8, ls=':')
ax.set(xscale='log', xlabel='step (log)', ylabel='RankMe',
       title='canonical run to 3000 steps — the decline is a transient')
plt.show()
caption('Answer: it washes out. The canonical run extended to 3000 steps (dotted vertical '
        '= the paper\'s 300-step window): the decline bottoms out shortly after the window '
        'ends and RankMe recovers to ~2.0, exactly as the fixed-kick-vs-unbounded-growth '
        'argument predicts (proof sketch: addendum appendix). Consequence: the paper\'s '
        'Fig-4 "compression phase" is a window onto a transient, not an endpoint. Logical '
        'scope: shown for the canonical configuration; the addendum reports the same '
        'recovery out to 6000 steps and across lr settings.')

# %% [markdown]
# ### B5. Is mse_skew's small dip a rare-class fork, or does MSE starve the rare classes?
# **Question.** The controls figure (main showcase §6) shows a small RankMe dip-then-recovery
# (~steps 40–90) in the mse_skew control. The CE compression mechanism is a rare-class fork —
# the two rare classes are learned along a shared path and then split. Could the MSE dip be
# the same event, which would make the control less "negative" than claimed?
# **Mechanism in words.** Under MSE with skewed counts, Li et al's supplementary claims the
# rare classes are simply never learned (gradient starvation). If so, no fork is possible —
# an unlearned class has nothing to split from — and the dip must come from the frequent
# classes reorganizing. Data: `mse_skew` in appendix_sweeps.pt (exact mse_skew spec:
# dup=3, lr=0.3, init=0.3, 3 seeds).

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for s in (0, 1, 2):
    r = SW['mse_skew'][f's{s}']
    axes[0].plot(r['rm'].numpy(), lw=1.4, label=f'seed {s}')
    cm = r['cls_mse'].numpy()
    for c, (ls, lbl) in enumerate((('-', 'frequent 0'), ('--', 'frequent 1'),
                                   ('-', 'rare 2'), ('--', 'rare 3'))):
        if s == 0:
            axes[1].plot(cm[:, c], ls, lw=1.4, color='tab:blue' if c < 2 else 'tab:red',
                         label=f'class {c} ({"n=2" if c < 2 else "n=1"})')
    axes[2].plot(r['rare_w_norm'].numpy(), lw=1.2, alpha=0.8,
                 label=[f'‖w₂‖ seed {s}', f'‖w₃‖ seed {s}'] if s == 0 else [None, None])
axes[0].set(xlabel='step', ylabel='RankMe (uncentered features)', title='the dip, 3 seeds')
axes[0].legend(fontsize=8)
axes[1].axhline(0.25, color='gray', lw=0.8, ls=':')
axes[1].set(xlabel='step', ylabel='per-class MSE', title='per-class loss (seed 0): rare classes pinned at 0.25')
axes[1].legend(fontsize=8)
axes[2].set(xlabel='step', ylabel='‖rare weight column‖', title='rare weight columns decay to ~0 (all seeds)')
axes[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
caption('Answer: starvation, not a fork — the negative control stands. Left: the RankMe '
        'dip-then-recovery, all 3 seeds. Middle: per-class MSE — both frequent classes reach '
        '0 while both rare classes stay pinned at exactly 0.25, the loss of predicting all '
        'zeros for a one-hot target (the no-prediction floor): the rare classes are NEVER '
        'learned, confirming Li et al\'s gradient-starvation claim. Right: the rare weight '
        'columns decay from their random init (norm ≈0.42) to ≤0.016 — under MSE the optimal '
        'response to a starved class is to zero its weights. With the rare classes inert, '
        'the CE fork mechanism (a rare pair being learned, then splitting) is structurally '
        'unavailable; the dip can only be frequent-class reorganization. Logical scope: the '
        'exact mse_skew spec, 3 seeds; says nothing about MSE with balanced counts '
        '(mse_uniform is monotone anyway).')

# %% [markdown]
# ## C. The multi-layer toys: where over depth do the phases live, and what centering hides
# **Question.** In the LLMs (and Li et al) the phase trajectory is measured at the END of the
# model. The multi-layer toy's per-depth stream panels (main showcase §6, uncentered — Li's
# definition) seem to show the opposite: a full dip→rise→peak→decline only right after the
# first block, and deep streams that start already-collapsed and only decline. Does the toy
# really contradict the end-of-model pattern?
# **Mechanism in words.** Each nonlinear write carries bias terms, which add the SAME offset
# to every sample's row. Offsets accumulate with depth, and a large shared mean is one
# dominant singular direction — it crushes UNCENTERED RankMe regardless of what the
# per-sample geometry does. Centering removes the mean and measures the actual
# representation geometry (this is what all LLM figures in this project use).

# %%
res_mr = np.load('data/results/toy_multi_residual_nonlinear/results_toy-multi_residual_nonlinear.npy',
                 allow_pickle=True).item()
steps_mr = sorted(res_mr)
nodes = [f'blk{k}.attn.in' for k in range(6)] + ['before_final_norm']
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, kind in zip(axes, ('acts_uncentered', 'acts_centered')):
    for i, node in enumerate(nodes):
        rm = [res_mr[s][node][kind]['rankme'] for s in steps_mr]
        ax.plot(steps_mr, rm, lw=1.6, color=plt.get_cmap('viridis')(i / (len(nodes) - 1)),
                label=node if kind == 'acts_centered' else None)
    ax.set(xscale='log', xlabel='step', title=kind.replace('acts_', ''))
axes[0].set_ylabel('stream RankMe')
axes[1].legend(fontsize=7)
fig.suptitle('multi_residual_nonlinear: per-depth stream RankMe, uncentered vs centered')
plt.tight_layout(); plt.show()
caption('Answer: no contradiction — the "deep streams start collapsed and only decline" '
        'picture is a mean artifact of the uncentered measurement. Left (uncentered): deep '
        'streams appear near-collapsed from init (2.4–5.9) and the dip→rise→peak→decline '
        'appears only after blk0. Right (centered, the LLM-comparable object): every stream '
        'starts near-full-rank (init 14.4 → 8.9 with depth — the residual path preserves '
        'the spectrum), shows at most a small early bump (e.g. final stream 8.91 → 9.14), '
        'and then compresses hard — and the FINAL stream compresses the most (→ 2.49), '
        'exactly the end-of-model pattern the LLMs show. The uncentered/centered gap is the '
        'accumulated write bias: the shared mean holds ~50% of the uncentered trace at every '
        'depth ≥ 1 by the end. What the toy genuinely lacks at ALL depths is a pronounced '
        'entropy-seeking rise — theta (the toy\'s learnable feature matrix: each sample\'s '
        'feature vector is a free parameter, Li et al\'s f(x)) initializes as random noise '
        'already at full rank 15.5, so there is nothing to expand; Li\'s single-layer rise '
        'starts from his near-degenerate constructed init instead. Logical scope: one '
        'variant (nonlinear residual, canonical '
        'spec); the plain stack has no such mean artifact (bias share ~9%) and its missing '
        'phases are real.')

# %% [markdown]
# ## D. Compression valleys, packed vs padded final-token (pythia)
# **Question.** The depth-axis valley (main showcase §8) is measured on packed all-token
# data. Does it survive padded final-token geometry — the geometry Li et al measure in —
# and does the stream recover by the output there too?

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
for ax, cfg, title in ((0, BLOCK_SAMPLES, 'packed all-token'),
                       (1, PADDED, 'padded final-token')):
    ax = axes[ax]
    for model, L in (('pythia-1b-deduped', 16), ('pythia-6.9b-deduped', 32),
                     ('OLMo-2-0425-1B', 16), ('OLMo-2-1124-7B', 32)):
        res, steps = _lib.load_results(cfg, model)
        leaves = [f'blk{k}.attn.in' for k in range(L)] + ['before_final_norm']
        fin = res[steps[-1]]
        rank = [fin.get(lf, {}).get('acts_centered', {}).get('rankme', np.nan) for lf in leaves]
        ax.semilogy(np.linspace(0, 1, len(leaves)), rank, marker='o', lw=2,
                    ls='--' if 'pythia' in model else '-', label=get_model_label(model))
    ax.set(xlabel='relative depth (last point = before_final_norm)', title=title)
axes[0].set_ylabel('stream RankMe (centered, log)')
axes[0].legend(fontsize=8, loc='lower left')
ylims = [ax.get_ylim() for ax in axes]                  # same range, ticks kept on both
for ax in axes:
    ax.set_ylim(min(l[0] for l in ylims), max(l[1] for l in ylims))
fig.suptitle('Compression valley at the final checkpoint: packed vs padded final-token (dashed = pythia)')
plt.tight_layout(); plt.show()
caption('Answer: the valley survives the geometry change; the output recovery does not — '
        'and both effects are pythia-specific. Pythia (dashed), packed: the familiar crash '
        'right after blk3\'s write enters (RankMe ~848 → 1.7 at 1b) with a monotone late '
        'recovery to ~194/260 at the pre-final-norm stream. Pythia, padded final-token: the '
        'same cliff at the same depth (468 → 11 at 1b; 854 → 260 → 68 at 6.9b), a shallower '
        'valley floor (tens, not ~2), a partial mid-stack recovery — and then the '
        'pre-final-norm stream itself drops to SINGLE DIGITS (8.7 at 1b, 7.6 at 6.9b): in '
        'last-token geometry the output stream is head-crushed in a way the packed view '
        'never shows. OLMo-2 (solid): no valley in EITHER geometry, and no output crush '
        'either — the padded profile is just the packed profile at ~70–80% level '
        '(before_final_norm 366/662 padded vs 447/657 packed). Two caveats: the padded runs '
        'have N/d ≈ 8 (16,384 rows over d=2048), so absolute levels carry finite-sample '
        'distortion in the tail (sample-count study, §E), and packed vs padded are '
        'different measured objects (all-token vs last-token covariance) — shapes are '
        'comparable, levels are not.')

# %% [markdown]
# ## E. How much of the spectrum is sample count? (packed vs padded, output leaves)
# **Question.** Padded runs estimate a d=2048 covariance from 16,384 rows (N/d ≈ 8); packed
# runs use 262k tokens. How much of the measured spectrum — especially its tail, and RankMe —
# depends on the sample count rather than the model? Data: fresh oversized final-checkpoint
# collections (2x the standard budgets; configs/sample_count_*.yaml), eigenspectra on nested
# random subsets bracketing the standard budgets (oneoff_scripts/sample_count_spectra.py).

# %%
# Data + constants for §E (no plots here).
SC = torch.load('data/results/sample_count_spectra.pt', weights_only=True)
NS_SC = {'packed': (16_384, 32_768, 65_536, 131_072, 262_144),
         'padded': (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)}
STD_N = {'packed': 262_144, 'padded': 16_384}

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for i, geom in enumerate(('packed', 'padded')):
    for j, leaf in enumerate(('before_final_norm', 'after_final_norm')):
        ax = axes[i, j]
        ns = NS_SC[geom]
        for k, n in enumerate(ns):
            lam = SC[f'{geom}.{leaf}.{n}'].numpy()
            lam = lam[lam > 0]
            p = lam / lam.sum()
            rm = np.exp(-(p * np.log(p)).sum())
            ax.loglog(np.arange(1, len(lam) + 1), lam,
                      lw=2.2 if n == STD_N[geom] else 1.2,
                      color=plt.get_cmap('plasma')(k / (len(ns) - 1)),
                      label=f'n={n:,} (RankMe {rm:.1f})' + (' ← standard' if n == STD_N[geom] else ''))
        ax.set(title=f'{geom} · {leaf}', xlabel='eigenvalue rank', ylabel='eigenvalue')
        ax.legend(fontsize=7, loc='lower left')
for row in axes:                                        # same range per row, ticks kept on both
    ylims = [ax.get_ylim() for ax in row]
    for ax in row:
        ax.set_ylim(min(l[0] for l in ylims), max(l[1] for l in ylims))
fig.suptitle('Centered eigenspectra vs sample count — pythia-1b, final checkpoint (thick = standard budget)')
plt.tight_layout(); plt.show()
caption('Answer: geometry- and leaf-dependent. Packed (top row): the spectrum and RankMe '
        'are essentially sample-count-invariant from 16k rows up (before_final_norm RankMe '
        '194→196 across 16k→262k; after_final_norm 262→269) — the packed budgets are '
        'comfortably in the asymptotic regime, and only the extreme tail lifts with n. '
        'Padded before_final_norm (bottom left): the head is so dominant that RankMe is '
        'stable and single-digit at EVERY n (7.1@1k → 8.6@32k) — the last-token head-crush '
        'is a property of the representation, not of the sample count. Padded '
        'after_final_norm (bottom right): here sample count matters — RankMe climbs 63 → '
        '101 from n=1k to the standard 16,384 and gains only ~2 more by 32k, with the '
        'growth coming from the tail filling in. Practical reading: our standard padded '
        'budget sits at the start of the plateau, so trends at fixed N are safe, RankMe '
        'levels at the padded POST-norm leaf are mildly conservative (a few percent below '
        'asymptotic), and any comparison of padded tail-band LEVELS across different '
        'sample counts is invalid. One count is missing by construction: the packed '
        'collection keeps 511 positions per 512-token window, so the 524k subset was '
        'skipped (523,264 rows collected).')
