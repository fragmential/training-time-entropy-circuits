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
# # Showcase appendix
#
# Appendix-grade material split out of showcase.ipynb: (A) the dataset-swap control,
# (B) the toy edge cases behind [docs/toy.md](../docs/toy.md) §3, (C)–(F) supporting digs,
# (G) leave-one-out and (H) layer-ablation digs for RQ1c (BLIND — descriptive captions
# only, pending cross-check).
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

CLASS_COLORS4 = ('magenta', 'orange', 'royalblue', 'seagreen')
CLASS_MARKERS4 = ('^', 'o', 's', 'D')
PHASES4 = (':', '-', '--', '-')                          # warmup / expansion / compression / recovery
Y4 = (0, 0, 1, 1, 2, 3)                                  # class of each feature row

def _bounds4(rm, eps=5e-4):
    """(dip, peak, trough). The compression segment = the largest post-warmup DRAWDOWN
    (running max minus curve, computed after the warmup dip): peak = its start, trough =
    its bottom. Horizon-independent, unlike the global argmax, which lands on the window
    edge whenever the transient recovers. Drawdown < eps -> no compression segment."""
    g = int(np.argmax(rm))
    dip = int(np.argmin(rm[:g + 1]))
    seg = rm[dip:]
    dd = np.maximum.accumulate(seg) - seg
    if dd.max() < eps:
        return dip, len(rm) - 1, len(rm) - 1
    trough = dip + int(np.argmax(dd))
    peak = dip + int(np.argmax(rm[dip:trough + 1]))
    return dip, peak, trough

def _phased4(ax, path, bounds, color, marker):
    for lo, hi, style in zip((0, *bounds), (*bounds, len(path) - 1), PHASES4):
        ax.plot(path[lo:hi + 1, 0], path[lo:hi + 1, 1], style, color=color, lw=1.5)
    ax.scatter(*path[0], color='0.6', s=18, zorder=2)
    ax.scatter(*path[-1], color=color, s=60, marker=marker, zorder=3)

def fig4_style(r, title, until=None):
    """One run in Li et al's Fig-4 presentation (1×4): W rows, features, RankMe + top
    eigenvalues, and per-class + total loss. `until` clips the plotted steps."""
    import torch.nn.functional as F_
    if until is not None:
        r = {k: (v[:until + 1] if v.dim() else v) for k, v in r.items()}
    rm, lam = r['rm'].numpy(), r['lam'].numpy()
    bounds, fk = _bounds4(rm), fork_step(r)
    fig, (bx, cx, dx, lx) = plt.subplots(1, 4, figsize=(21, 4.6))
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
    dx.set(title='RankMe & eigenvalues', xlabel='step', ylabel='RankMe')
    logits = torch.matmul(r['theta_path'], r['W_path'])              # (T, 6, 4)
    y = torch.tensor(Y4)
    ce = F_.cross_entropy(logits.reshape(-1, 4), y.repeat(len(logits)),
                          reduction='none').reshape(len(logits), 6)
    for c in range(4):
        rows = [j for j, yc in enumerate(Y4) if yc == c]
        lx.plot(np.maximum(ce[:, rows].mean(1).numpy(), 1e-8), lw=1.4,
                color=CLASS_COLORS4[c], label=f'class {c} (n={"2" if c < 2 else "1"})')
    lx.plot(np.maximum(ce.mean(1).numpy(), 1e-8), lw=2.2, color='0.5', label='total')
    lx.set(yscale='log', xlabel='step', ylabel='loss (log)', title='per-class + total loss')
    lx.legend(fontsize=8)
    for ax in (dx, lx):
        if fk is not None:
            ax.axvline(fk, color='gray', lw=0.9, ls=':')
        ax.axvline(bounds[1], color='gray', lw=0.9, ls='--')
    fig.suptitle(title + '  (fork: dotted; decline onset: dashed)')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## A. The dataset-swap control (data vs model)
#
# Each 1B model re-run on the OTHER family's pretraining mix (same texts, re-tokenized).
# Shown: Σquality — the family-discriminating ledger term. The full multi-metric verdict
# table (all ledger terms, rogue diagnostics, coupling) is in docs/dig_findings.md (dataset-swap controls section).

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
caption('Σquality for each 1B model on its own mix (solid) vs the other family\'s mix '
        '(dashed). The dashed curves track their solid counterparts instead of swapping '
        'sides: the family split follows the model, not the data. Full multi-metric table: '
        'dig_findings, dataset-swap controls.')

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
# ### B1. Does the RankMe decline's existence depend on δ?
#
# The jitter δ is a variable determining distance between the rare classes at
# initialisation. It is a random offset that breaks the rare classes' symmetry. The fork
# arrives later the smaller δ is (fork time ≈ log(1/δ) / growth rate).
#
# **Question.** Does the RankMe decline's existence depend on δ?

# %%
DELTAS = (0.316, 0.1, 0.0316, 1e-2, 0.00316, 1e-3)
MARKERS = ('o', 's', '^', 'D', 'v', 'P')
SEEDS = range(10)
PLOT_UNTIL = 400
cmap = plt.get_cmap('viridis')

def fork_or_start(r):
    """Fork step; 0 for a pair separated from init (never co-traveled)."""
    fk = fork_step(r)
    return fk if fk is not None else (0 if float(r['cos_w'][0]) < 0.97 else None)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
rows = []
for i, d in enumerate(DELTAS):
    c = cmap(i / (len(DELTAS) - 1))
    rms = []
    for s in SEEDS:
        r = SW['delta'][f'{d:g}/s{s}']
        rm = r['rm'].numpy()
        rms.append(rm)
        _, pk, tr = _bounds4(rm)
        rows.append((d, s, fork_or_start(r), rm[pk] - rm[tr]))
    axes[0].plot(np.mean(rms, axis=0)[:276], color=c, lw=1.5, label=f'δ={d:g}',
                 marker=MARKERS[i], ms=7, markevery=25, mec='k', mew=0.4)
axes[0].set(xlabel='step', ylabel='RankMe (uncentered features)', xlim=(50, 275),
            ylim=(1.980, 2.0), xticks=np.arange(50, 276, 25),
            title=f'mean RankMe by δ ({len(SEEDS)} seeds)')
axes[0].legend(fontsize=8)
for i, d in enumerate(DELTAS):
    grp = [g for g in rows if g[0] == d and g[2] is not None]
    if grp:
        c = cmap(i / (len(DELTAS) - 1))
        axes[1].scatter([g[2] for g in grp], [g[3] for g in grp], s=25, color=c,
                        alpha=0.35, marker=MARKERS[i], edgecolors='k', linewidths=0.3)
        mx, my = np.mean([g[2] for g in grp]), np.mean([g[3] for g in grp])
        axes[1].scatter([mx], [my], s=80, color=c, edgecolors='k', lw=0.5, marker=MARKERS[i])
        axes[1].annotate(f'δ={d:g}', (mx, my), fontsize=8, xytext=(4, 4),
                         textcoords='offset points')
axes[1].set(xlabel='fork-onset step (0 = separated from init)',
            ylabel='decline (largest post-warmup drawdown)', title='decline vs fork timing')
plt.tight_layout(); plt.show()
caption('Left: mean RankMe per δ over 10 seeds (darker = smaller δ = later fork; δ=0.316 '
        'starts with the pair already separated). Right: decline vs fork step (faint = '
        'seeds, solid = means; declines measured on the full 1000-step runs).')

# %% [markdown]
# #### The δ-sweep runs in Li et al's Fig-4 style

# %%
for d in DELTAS[:3]:
    fig4_style(SW['delta'][f'{d:g}/s0'], f'δ = {d:g} (seed 0, first 400 steps)', until=PLOT_UNTIL)
caption('The first three δ (seed 0 each) in the paper\'s presentation, for orientation.')

# %% [markdown]
# ### B2. Why can the decline start LATER than the visible fork? (rare-pair offset rx)
#
# **Question.** In some runs the trajectories visibly fork while RankMe is still flat, and
# the decline only starts tens of steps later. What controls that delay?
#
# **Mechanism in words.** The kick's size grows with the SQUARE of how far the two rare
# classes have separated (it is a covariance, a product of two coordinates). If the pair
# starts away from the origin (offset rx), the early separation is small relative to the
# pair's position, the squared quantity stays negligible, and RankMe reacts only once the
# separation is large — a delay. If the pair starts AT the origin, any separation is
# immediately large relative to position, and the decline starts with the fork.
#
# **Why it matters.** The paper's figure shows the dashed (compression) styling beginning
# exactly at the fork; that is reproduced here only with the origin start — which is the
# evidence that their rare pair initializes at the origin.

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for rx, c in zip(('0', '-0.1', '-0.25', '-0.5'), plt.get_cmap('plasma')(np.linspace(0.1, 0.8, 4))):
    r = SW['rx'][rx]
    rm, fk = r['rm'].numpy(), fork_step(r)
    pk = _bounds4(rm)[1]
    ax.plot(rm, color=c, lw=1.6, label=f'rx={rx}: fork {fk}, peak {pk}, lag {pk - fk if fk else "—"}')
    if fk: ax.axvline(fk, color=c, lw=0.8, ls=':')
    ax.axvline(pk, color=c, lw=0.8, ls='--')
ax.set(xlabel='step', ylabel='RankMe', title='fork onset (dotted) vs decline onset (dashed) by rx')
ax.legend(fontsize=8, loc='upper left')
axz = ax.inset_axes([0.55, 0.07, 0.43, 0.5])            # zoom: the peak/decline region
for rx, c in zip(('0', '-0.1', '-0.25', '-0.5'), plt.get_cmap('plasma')(np.linspace(0.1, 0.8, 4))):
    r = SW['rx'][rx]
    rm, fk = r['rm'].numpy(), fork_step(r)
    pk = _bounds4(rm)[1]
    axz.plot(rm, color=c, lw=1.4)
    if fk: axz.axvline(fk, color=c, lw=0.8, ls=':')
    axz.axvline(pk, color=c, lw=0.8, ls='--')
axz.set(xlim=(170, 260), ylim=(1.996, 2.0))
axz.tick_params(labelsize=7)
ax.indicate_inset_zoom(axz, edgecolor='0.4')
plt.show()
caption('Dotted = fork onset, dashed = decline onset (the RankMe peak). Starting the rare '
        'pair further from the origin stretches the fork→decline lag (~5 → ~41 steps) and '
        'shrinks the decline, as the squared-separation account predicts. At the origin '
        'the two coincide — the paper\'s dashes-at-the-fork rendering.')

# %% [markdown]
# #### The same four runs in Li et al's Fig-4 style

# %%
for rx in ('0', '-0.1', '-0.25', '-0.5'):
    fig4_style(SW['rx'][rx], f'rare-pair offset rx = {rx}')
caption('The B2 runs in the paper\'s presentation, for orientation.')

# %% [markdown]
# ### B3. Is the compression permanent, or does it wash out? (long horizon)
#
# **Question.** The paper's window ends at step 300, right after the decline begins. If
# training simply continues, does the compressed state persist?
#
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
caption('The canonical run to 3000 steps (dotted = the paper\'s 300-step window): the '
        'decline bottoms out and RankMe recovers to ~2.0. The compression is a transient; '
        'the paper\'s window ends right after it begins. Proof sketch: toy.md appendix.')

# %% [markdown]
# ### B4. Is mse_skew's small dip a rare-class fork — or is the task unsolvable under MSE?
#
# **Question.** The controls figure (main showcase §6) shows a small RankMe dip-then-recovery
# (~steps 40–90) in the mse_skew control. The CE compression mechanism is a rare-class fork —
# the two rare classes are learned along a shared path and then split. Could the MSE dip be
# the same event, which would make the control less "negative" than claimed?
#
# **Mechanism in words.** With logits = FW at d = 2, MSE's global optimum is the best
# rank-2 approximation of the one-hot targets: it keeps the two frequent classes and maps
# every rare row to ZERO (per-rare-row MSE 0.25, rare weight columns → 0). Abandoning the
# rare classes is the optimum, not a failed optimization — no fork is possible, so the dip
# must come from the frequent classes' geometry. Li et al's only statement is a
# supplementary caption sentence: "only information about the most frequently occurring
# classes are learned." Data: `mse_skew` in appendix_sweeps.pt (exact mse_skew spec:
# dup=3, lr=0.3, init=0.3, 3 seeds).
#
# (note from dante: I have lowkey zero clue what the above slop text is intending to say, but I will make it clear that based on the visual evidence, there does still seem to be compression, it just doesn't stay because the optimal solution here is actually to only fit the two common classes, but yeah seems like there's still a kick.)

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
    if s == 0:
        axes[1].plot(cm @ np.array([2, 2, 1, 1]) / 6, lw=2.2, color='0.5', label='total')
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

# The ACTUAL showcase §6 control run (variant `mse_skew`), same three panels.
r_ctl = np.load('data/results/toy_mse_skew/results_toy-mse_skew.npy', allow_pickle=True).item()
ctl_steps = sorted(r_ctl)
ctl_rm = [r_ctl[s]['before_final_norm']['acts_uncentered']['rankme'] for s in ctl_steps]
t_ctl = torch.load('data/results/toy/trajectories_mse_skew.pt')
tsteps, F_ctl, W_ctl = t_ctl['steps'].tolist(), t_ctl['F'], t_ctl['W']
y_ctl = torch.repeat_interleave(torch.arange(4), 3 * torch.tensor([2, 2, 1, 1]))
onehot = torch.nn.functional.one_hot(y_ctl, 4).float()
row_mse = torch.stack([((F_ctl[i] @ W_ctl[i] - onehot) ** 2).mean(1) for i in range(len(tsteps))])
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ctl_steps, ctl_rm, lw=1.4)
for c, (ls, _) in enumerate((('-', ''), ('--', ''), ('-', ''), ('--', ''))):
    axes[1].plot(tsteps, [float(row_mse[i][y_ctl == c].mean()) for i in range(len(tsteps))],
                 ls, lw=1.4, color='tab:blue' if c < 2 else 'tab:red',
                 label=f'class {c} ({"n=2" if c < 2 else "n=1"})')
axes[1].plot(tsteps, [float(row_mse[i].mean()) for i in range(len(tsteps))],
             lw=2.2, color='0.5', label='total')
for col, lbl in ((-2, '‖w₂‖'), (-1, '‖w₃‖')):
    axes[2].plot(tsteps, [float(W_ctl[i][:, col].norm()) for i in range(len(tsteps))],
                 lw=1.2, label=lbl)
axes[0].set(xlabel='step', ylabel='RankMe (uncentered features)', title='the showcase §6 control run')
axes[1].axhline(0.25, color='gray', lw=0.8, ls=':')
axes[1].set(xlabel='step', ylabel='per-class MSE', title='per-class loss')
axes[1].legend(fontsize=8)
axes[2].set(xlabel='step', ylabel='‖rare weight column‖', title='rare weight columns')
axes[2].legend(fontsize=8)
fig.suptitle('The actual showcase §6 mse_skew control (variant run: dup=3, lr=0.3, init std=0.3)')
plt.tight_layout(); plt.show()
caption('The global optimum, reached — sweep seeds and control run alike converge to the '
        'rank-2 truncation of the targets: rare classes pinned at the 0.25 floor (middle), '
        'rare weight columns → 0 (right). No fork is possible, hence no compression. The '
        'dip is not interpreted, though observationally the RankMe drop aligns with the '
        'rare weight columns\' peak (the onset of their decay).')

# %% [markdown]
# ## C. Per-depth stream RankMe: what centering hides
#
# **Question.** Does the phase pattern live only at the end of the stack — and does the
# uncentered (Li-style) measurement tell the same story as the centered one used for all
# LLM figures?
#
# **Answer, previewed.**
#
# - In the runs that show the pattern (top two rows), it shows at EVERY depth, and the two
#   columns agree — these layers are linear with no bias, so no mean builds up.
#
# - In the 32-class tanh run (bottom row) the columns disagree completely: the uncentered
#   depth structure is an artifact of accumulated bias means. That artifact is why the old
#   per-depth figures were removed from the main showcase.

# %%
import torch.nn.functional as _Fn

def _fork_of(tag):
    t = torch.load(f'data/results/toy/trajectories_{tag}.pt')
    ts, Wt = t['steps'].tolist(), t['W']
    cos = [float(_Fn.cosine_similarity(Wt[i][:, -2], Wt[i][:, -1], dim=0)) for i in range(len(ts))]
    shared = next((i for i in range(len(ts)) if cos[i] > 0.97), 0)
    return next((ts[j] for j in range(shared, len(ts)) if cos[j] < 0.9), None)

C_ROWS = (('residual', 'single_d6_bs_s0', True),
          ('plain (identity init)', 'single_d6_id_plain_s0', True),
          ('32-class tanh residual', 'multi_residual_nonlinear', False))
fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharey='row', sharex='row')
for (label, tag, has_fork), (axu, axc) in zip(C_ROWS, axes):
    r = np.load(f'data/results/toy_{tag}/results_toy-{tag}.npy', allow_pickle=True).item()
    steps = sorted(r)
    nodes = [f'blk{k}.attn.in' for k in range(6)] + ['before_final_norm']
    fork = _fork_of(tag) if has_fork else None
    for ax, kind in ((axu, 'acts_uncentered'), (axc, 'acts_centered')):
        for i, node in enumerate(nodes):
            ax.plot(steps, [r[s][node][kind]['rankme'] for s in steps], lw=1.5,
                    color=plt.get_cmap('viridis')(i / (len(nodes) - 1)),
                    label=(f'depth {i}' if node != 'before_final_norm' else 'final stream')
                          if ax is axes[0, 1] else None)
        if fork is not None:
            ax.axvline(fork, color='crimson', lw=1.0, ls=':')
        ax.set_xscale('log')
    axu.set_ylabel(f'{label}\nstream RankMe')
axes[0, 1].legend(fontsize=7, loc='lower right')
axes[0, 0].set_title("uncentered (Li's definition)")
axes[0, 1].set_title('centered (the LLM-side object)')
for ax in axes[-1]:
    ax.set_xlabel('step')
fig.suptitle('Per-depth stream RankMe, uncentered vs centered — dotted crimson = the rare-pair fork')
plt.tight_layout(); plt.show()
caption('Top two rows (pattern-showing runs): the pattern appears at every depth, the '
        'decline starts at the fork (crimson), and the columns agree — linear bias-free '
        'layers build no mean. Bottom row (32-class tanh): the uncentered depth collapse '
        'is accumulated bias means, not geometry; centered, streams start near full rank '
        'and the final stream compresses most. No phase pattern in either column.')

# %% [markdown]
# ## D. Compression valleys, packed vs padded final-token (pythia)
#
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
caption('Pythia (dashed): the valley appears at the same depth in both geometries, but only '
        'the padded last-token view ends with the output stream itself crushed to single '
        'digits — the packed view recovers. OLMo-2 (solid): no valley and no crush in '
        'either geometry. Caveats: padded N/d ≈ 8 distorts tail levels (§E), and packed vs '
        'padded are different measured objects — compare shapes, not levels.')

# %% [markdown]
# ## E. How much of the spectrum is sample count? (packed vs padded, output leaves)
#
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
caption('Packed (top): spectrum and RankMe essentially invariant from 16k rows up. Padded '
        'pre-norm (bottom left): single-digit RankMe at EVERY n — the last-token '
        'head-crush is a representation property, not sample count. Padded post-norm '
        '(bottom right): RankMe is still climbing at the standard budget, so trends at '
        'fixed N are safe but tail-band levels across different n are not comparable.')

# %% [markdown]
# ## F. One candidate route for the padded last-token puzzle, ruled out
#
# **Question.** Why is the padded LAST-TOKEN measurement head-crushed (§E)? One candidate:
# attention deposits each window's slot spike into downstream tokens' residuals along the
# sink direction v₁, and the last token inherits it. Does such a window-modulated deposit
# survive to the final stream?
#
# **Scope note.** That sink content reaches other tokens AT ALL is already established and
# not at issue: heads read the spike (H1.5) and every ordinary token carries v₁-content
# mid-stack (showcase §4). What the padded collapse would need is spike content that
# survives TO THE MEASUREMENT POINT, window-modulated — that specific route is tested
# here. What actually carries the last-token collapse remains open (plan.md item 3).
# Visual form of the deposit test recorded in sink_literature §(d).

# %%
from utils.accessor import DataAccessor
from transformers import AutoTokenizer

def deposit_panels(model='pythia-1b-deduped', step=143000):
    acc = DataAccessor(f'data/inferences/block_rogue_id/{model}/step{step}.pt')
    v1 = acc['blk3.mlp.out'].acts.eigvecs_centered[:, 0].float()
    n_rows = len(acc['blk3.mlp.out'].acts.samples)
    W = n_rows // 511                                       # full 511-token windows
    ids = torch.load('data/mixes/pile_30M_512.pt')[:W, :511]
    tok = AutoTokenizer.from_pretrained(f'EleutherAI/{model}')
    uniq = torch.unique(ids)
    toks = tok.convert_ids_to_tokens(uniq.tolist())
    nl_of_uniq = torch.tensor(['Ċ' in t or '\n' in t for t in toks])
    is_nl = nl_of_uniq[torch.searchsorted(uniq, ids)]       # (W, 511)
    slot = torch.zeros_like(is_nl)
    has = is_nl.any(1)
    slot[torch.arange(W)[has], is_nl.float().argmax(1)[has]] = True   # first newline
    slot[:, 0] = True                                                 # + position 0
    proj, share = {}, {}
    for leaf in ('blk3.attn.in', 'blk3.mlp.out', 'before_final_norm'):
        a = acc[leaf].acts
        x = a.samples[:W * 511].float() - a.mean.float()
        p = (x @ v1).reshape(W, 511)
        proj[leaf] = p
        share[leaf] = float(p[~slot].var() / x[~slot.reshape(-1)].square().sum(1).mean())
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.4))
    leaves = list(proj)
    xpos = np.arange(3)
    for off, (mask, lbl, c) in enumerate(((slot, 'slot rows (pos 0 + first \\n)', 'tab:red'),
                                          (~slot, 'bulk rows', 'tab:gray'))):
        rms = [float(proj[l][mask].square().mean().sqrt()) for l in leaves]
        ax0.bar(xpos + 0.35 * off, rms, width=0.32, color=c, label=lbl, log=True)
    ax0.set_xticks(xpos + 0.17)
    ax0.set_xticklabels(['blk3.attn.in\n(entering)', 'blk3.mlp.out\n(the write)',
                         'before_final_norm\n(end of stack)'], fontsize=8)
    ax0.set_ylabel('RMS v₁-component (log)')
    ax0.set_title('slots are crushed; bulk stays flat\n'
                  f'bulk variance SHARE along v₁: {share["blk3.attn.in"]:.4f} → '
                  f'{share["before_final_norm"]:.4f}', fontsize=9)
    ax0.legend(fontsize=8)
    spike = torch.stack([proj['blk3.mlp.out'][w][slot[w]].abs().mean() for w in range(W)])
    for leaf, c, lbl in (('before_final_norm', 'tab:blue', 'final stream'),
                         ('blk3.attn.in', 'tab:gray', 'control: entering blk3')):
        bulkm = torch.stack([proj[leaf][w][~slot[w]].mean() for w in range(W)])
        r = float(np.corrcoef(spike.numpy(), bulkm.numpy())[0, 1])
        ax1.scatter(spike.numpy(), bulkm.numpy(), s=10, alpha=0.5, color=c,
                    label=f'{lbl} (r = {r:+.2f})')
    ax1.set(xlabel='window slot-spike size (mean |proj| at the write)',
            ylabel='window mean bulk v₁-component')
    ax1.set_title('no window-specific deposit', fontsize=9)
    ax1.legend(fontsize=8)
    fig.suptitle(f'Modulated sink-direction deposit test — {model}, final checkpoint')
    plt.tight_layout(); plt.show()

deposit_panels()
caption('Answer: no. Left: the slot rows\' enormous write is almost entirely gone by the '
        'end of the stack, while the bulk rows\' component barely moves; the bulk variance '
        'share along v₁ (title) falls rather than rises. Right: windows with bigger slot '
        'spikes do not leave their bulk rows with more v₁-content at the final stream — '
        'the correlation is as small as the entering-blk3 control. Scope: projections are '
        'centered, so a constant deposit is invisible by construction — this tests '
        'variance transport only, at two depths (mid-stack profile: showcase §4).')


# %% [markdown]
# ## G. Leave-one-out: final-stream entropy without a group of writes (RQ1c.1) — BLIND
#
# The `loo` metric: the centered final-stream spectrum recomputed with a group of writes
# subtracted from the samples, S(final − Σ writes in group). Groups per model: every
# block singly (attn+mlp jointly), the four L/4 chunks, the middle half. Data:
# data/results/loo_samples (all 7 models). Shown: delta_rankme = RankMe(final) −
# RankMe(final − group).
#
# Captions are descriptive only — no interpretation until cross-checked.

# %%
LOO_MODELS = ['pythia-160m-deduped', 'pythia-410m-deduped', 'pythia-1b-deduped',
              'pythia-6.9b-deduped', 'OLMo-2-0425-1B', 'OLMo-2-1124-7B', 'nanochat-d12']
NB = {'pythia-160m-deduped': 12, 'pythia-410m-deduped': 24, 'pythia-1b-deduped': 16,
      'pythia-6.9b-deduped': 32, 'OLMo-2-0425-1B': 16, 'OLMo-2-1124-7B': 32,
      'nanochat-d12': 12}

def loo_res(model, key='delta_rankme'):
    r = np.load(f'data/results/loo_samples/results_{model}.npy', allow_pickle=True).item()
    steps = sorted(r)
    return steps, {g: [r[s]['']['loo'][g][key] for s in steps]
                   for g in r[steps[0]]['']['loo']}

def loo_base(model):
    r = np.load(f'data/results/loo_samples/results_{model}.npy', allow_pickle=True).item()
    steps = sorted(r)
    return steps, [r[s]['before_final_norm']['acts_centered']['rankme'] for s in steps]


# %%
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, model in zip(axes.flat, LOO_MODELS):
    steps, loo = loo_res(model)
    L = NB[model]
    for g, ys in loo.items():
        if '-' in g:
            continue
        k = int(g[3:])
        ax.plot(steps, ys, lw=1.3, color=plt.get_cmap('viridis')(k / (L - 1)))
    ax.axhline(0, color='gray', lw=0.8)
    ax.set(xscale='log', xlabel='step', title=model)
for ax in axes.flat[len(LOO_MODELS):]:
    ax.axis('off')
axes[0, 0].set_ylabel('delta RankMe (single blocks)')
axes[1, 0].set_ylabel('delta RankMe (single blocks)')
fig.suptitle('Leave-one-out, single blocks: RankMe(final) − RankMe(final − block), light → dark = deeper')
plt.tight_layout(); plt.show()
caption('delta_rankme per single-block group over training, per model (colorbar = block '
        'depth). Descriptive only.')

# %%
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, model in zip(axes.flat, LOO_MODELS):
    st, base = loo_base(model)
    ax.plot(st, base, 'k', lw=2.2, label='baseline')
    steps, loo = loo_res(model, key='rankme')
    for g in [g for g in loo if '-' in g]:
        ax.plot(steps, loo[g], lw=1.4, label=f'− {g}')
    ax.set(xscale='log', xlabel='step', title=model)
    ax.legend(fontsize=6)
for ax in axes.flat[len(LOO_MODELS):]:
    ax.axis('off')
axes[0, 0].set_ylabel('RankMe (final − group)')
axes[1, 0].set_ylabel('RankMe (final − group)')
fig.suptitle('Leave-one-out, block groups (absolute): baseline (black) vs RankMe(final − group)')
plt.tight_layout(); plt.show()
caption('Absolute RankMe of the group-subtracted final stream over training, per model: '
        'black = the unsubtracted final stream, colors = the four quarter-chunks and the '
        'middle half. Descriptive only.')

fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, model in zip(axes.flat, LOO_MODELS):
    steps, loo = loo_res(model, key='delta_entropy')
    chunks = [g for g in loo if '-' in g]
    for g in chunks:
        ax.plot(steps, loo[g], lw=1.6, label=g)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set(xscale='log', xlabel='step', title=model)
    ax.legend(fontsize=6)
for ax in axes.flat[len(LOO_MODELS):]:
    ax.axis('off')
axes[0, 0].set_ylabel('delta S (block groups)')
axes[1, 0].set_ylabel('delta S (block groups)')
fig.suptitle('Leave-one-out, block groups: the four L/4 chunks and the middle half (rank entropy S)')
plt.tight_layout(); plt.show()
caption('delta_entropy (rank entropy S, unablated − ablated) for the contiguous groups '
        '(four quarter-chunks + the middle half) over training, per model. Descriptive only.')

# %%
# Single-WRITE leave-one-out (`ablation_contribution`, covariance-space) from the MAIN
# packed sweeps — per attn/mlp write separately.
CFG = lambda m: 'nanochat_samples' if m == 'nanochat-d12' else 'block_representations_samples'
for sub in ('mlp', 'attn'):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, LOO_MODELS):
        r = np.load(f'data/results/{CFG(model)}/results_{model}.npy', allow_pickle=True).item()
        steps = sorted(r)
        L = NB[model]
        for l in range(L):
            ys = [r[s].get(f'blk{l}.{sub}.out', {}).get('ablation_contribution', {}).get('delta_rankme')
                  for s in steps]
            if ys[0] is None:
                continue
            ax.plot(steps, ys, lw=1.2, color=plt.get_cmap('viridis')(l / (L - 1)))
        ax.axhline(0, color='gray', lw=0.8)
        ax.set(xscale='log', xlabel='step', title=model)
    for ax in axes.flat[len(LOO_MODELS):]:
        ax.axis('off')
    axes[0, 0].set_ylabel(f'delta RankMe ({sub} writes)')
    axes[1, 0].set_ylabel(f'delta RankMe ({sub} writes)')
    fig.suptitle(f'Single-write leave-one-out ({sub}.out), main packed sweeps — light → dark = deeper')
    plt.tight_layout(); plt.show()
    caption(f'ablation_contribution delta_rankme per {sub} write over training, per model '
            f'(colorbar = block depth). Descriptive only.')

# %% [markdown]
# ## H. Layer ablation: full inference with block writes zeroed (RQ1c.2) — BLIND
#
# The ablated runs (data/results/ablate_*): per model the four L/4 quarter-chunks, the
# middle half, and the pythia carriers (blk3 at 1b; blk4-5 at 6.9b), each a full
# collection with all metrics on ~38 checkpoints. Baseline = the unablated packed sweep.
#
# Captions are descriptive only — no interpretation until cross-checked.

# %%
QUARTERS = {12: ['blk0-2', 'blk3-5', 'blk6-8', 'blk9-11', 'blk3-8'],
            16: ['blk0-3', 'blk4-7', 'blk8-11', 'blk12-15', 'blk4-11'],
            24: ['blk0-5', 'blk6-11', 'blk12-17', 'blk18-23', 'blk6-17'],
            32: ['blk0-7', 'blk8-15', 'blk16-23', 'blk24-31', 'blk8-23']}
CARRIER = {'pythia-1b-deduped': 'blk3', 'pythia-6.9b-deduped': 'blk4-5'}

def interventions(model):
    return QUARTERS[NB[model]] + ([CARRIER[model]] if model in CARRIER else [])

def series(cfg, model, node, fam, key):
    r = np.load(f'data/results/{cfg}/results_{model}.npy', allow_pickle=True).item()
    steps = sorted(r)
    return steps, [r[s][node][fam][key] for s in steps]

# %%
# Ledger contributions (the showcase §3 stacked panel) per ablated run, next to the
# baseline — plot_ledger_stack in the grid environment, styled as before (steps on x,
# legend on the baseline panel only, ylabel on the left column, no figure export).
for model in LOO_MODELS:
    _lib.grid_start(ncols=3, figsize=(16, 7), sharex=True, sharey=True, savefig=False,
                    title=f'Ledger contributions under ablation — {model}',
                    xvar='steps', total_lw=2, emb_lw=1.2)
    _lib.plot_ledger_stack(model, CFG(model), NB[model], title='baseline',
                           legend=7, ylabel='rank entropy')
    for i, tag in enumerate(QUARTERS[NB[model]]):
        _lib.plot_ledger_stack(model, f'ablate_{tag}', NB[model], title=f'− {tag}',
                               legend=None, ylabel='rank entropy' if i == 2 else None)
    _lib.grid_show()
    caption(f'The decomposition ΔS = χ (green) + quality (red) + interference (purple), '
            f'summed over blocks, for {model}: top-left the unablated packed sweep, the '
            f'other five panels the ablated runs (− tag = those blocks\' writes zeroed). '
            f'Black = ΔS total; dotted gray = the embedding stream\'s own entropy. '
            f'Descriptive only.')

for model, tag in CARRIER.items():
    _lib.grid_start(ncols=2, figsize=(13, 4.5), sharex=True, sharey=True, savefig=False,
                    title=f'Ledger contributions, {model}: baseline vs carrier ablation',
                    xvar='steps', total_lw=2, emb_lw=1.2)
    _lib.plot_ledger_stack(model, CFG(model), NB[model], title='baseline',
                           legend=7, ylabel='rank entropy')
    _lib.plot_ledger_stack(model, f'ablate_{tag}', NB[model], title=f'− {tag}',
                           legend=None, ylabel=None)
    _lib.grid_show()
    caption(f'The same stacked decomposition for {model}: left the unablated sweep, right '
            f'the run with {tag}\'s writes zeroed. Descriptive only.')

# %%
for leaf in ('after_final_norm', 'before_final_norm'):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, LOO_MODELS):
        st, base = series(CFG(model), model, leaf, 'acts_centered', 'rankme')
        ax.plot(st, base, 'k', lw=2.2, label='baseline')
        for i, tag in enumerate(interventions(model)):
            try:
                st2, ys = series(f'ablate_{tag}', model, leaf, 'acts_centered', 'rankme')
            except FileNotFoundError:
                continue
            ax.plot(st2, ys, lw=1.4, color=plt.get_cmap('tab10')(i), label=f'− {tag}')
        ax.set(xscale='log', xlabel='step', title=model)
        ax.legend(fontsize=6)
    for ax in axes.flat[len(LOO_MODELS):]:
        ax.axis('off')
    axes[0, 0].set_ylabel(f'{leaf} RankMe (centered)')
    axes[1, 0].set_ylabel(f'{leaf} RankMe (centered)')
    fig.suptitle(f'{leaf} RankMe under ablation: baseline (black) vs each intervention')
    plt.tight_layout(); plt.show()
    caption(f'{leaf} centered RankMe over training: black = the unablated packed '
            f'sweep, colors = each ablated run (− tag = those blocks\' writes zeroed). '
            f'Descriptive only.')

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, LOO_MODELS):
        st, base = series(CFG(model), model, leaf, 'acts_centered', 'matrix_entropy')
        bmap = dict(zip(st, base))
        for i, tag in enumerate(interventions(model)):
            try:
                st2, ys = series(f'ablate_{tag}', model, leaf, 'acts_centered', 'matrix_entropy')
            except FileNotFoundError:
                continue
            pts = [(s, bmap[s] - y) for s, y in zip(st2, ys) if s in bmap]
            ax.plot(*zip(*pts), lw=1.4, color=plt.get_cmap('tab10')(i), label=f'− {tag}')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set(xscale='log', xlabel='step', title=model)
        ax.legend(fontsize=6)
    for ax in axes.flat[len(LOO_MODELS):]:
        ax.axis('off')
    axes[0, 0].set_ylabel('delta S (baseline − ablated)')
    axes[1, 0].set_ylabel('delta S (baseline − ablated)')
    fig.suptitle(f'{leaf} rank entropy S under ablation, as deltas: baseline − each ablated run')
    plt.tight_layout(); plt.show()
    caption(f'The same runs as deltas at {leaf}, in rank entropy S: baseline S minus ablated S '
            f'at matching checkpoints, one line per intervention. Positive = the ablated run '
            f'sits below the baseline. Descriptive only.')

# %%
def summed_term(cfg, model, term):
    r = np.load(f'data/results/{cfg}/results_{model}.npy', allow_pickle=True).item()
    steps = sorted(r)
    return steps, [sum(float(r[s][f'blk{l}']['block_ledger'][term]) for l in range(NB[model]))
                   for s in steps]

for term in ('quality', 'interference', 'chi'):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, model in zip(axes.flat, LOO_MODELS):
        st, base = summed_term(CFG(model), model, term)
        ax.plot(st, base, 'k', lw=2.2, label='baseline')
        for i, tag in enumerate(interventions(model)):
            try:
                st2, ys = summed_term(f'ablate_{tag}', model, term)
            except FileNotFoundError:
                continue
            ax.plot(st2, ys, lw=1.4, color=plt.get_cmap('tab10')(i), label=f'− {tag}')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set(xscale='log', xlabel='step', title=model)
        ax.legend(fontsize=6)
    for ax in axes.flat[len(LOO_MODELS):]:
        ax.axis('off')
    axes[0, 0].set_ylabel(f'Σ {term}')
    axes[1, 0].set_ylabel(f'Σ {term}')
    fig.suptitle(f'Σ {term} (over blocks) under ablation: baseline vs each intervention')
    plt.tight_layout(); plt.show()
    caption(f'The summed {term} decomposition term over training: black = baseline, '
            f'colors = each ablated run. Descriptive only.')

# %%
# Per-block quality and total delta_s under the carrier ablations, next to the baseline.
for term in ('quality', 'delta_s'):
    for model, tag in CARRIER.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, cfg, ttl in ((axes[0], CFG(model), 'baseline'),
                             (axes[1], f'ablate_{tag}', f'− {tag}')):
            r = np.load(f'data/results/{cfg}/results_{model}.npy', allow_pickle=True).item()
            steps = sorted(r)
            L = NB[model]
            for l in range(L):
                ys = [float(r[s][f'blk{l}']['block_ledger'][term]) for s in steps]
                ax.plot(steps, ys, lw=1.3, color=plt.get_cmap('viridis')(l / (L - 1)))
            ax.axhline(0, color='gray', lw=0.8)
            ax.set(xscale='log', xlabel='step', title=ttl)
        axes[0].set_ylabel(f'per-block {term}')
        fig.suptitle(f'Per-block {term}, {model}: baseline vs carrier ablation')
        plt.tight_layout(); plt.show()
        caption(f'Per-block {term} term over training for {model} (colorbar = block depth): '
                f'left the unablated sweep, right the run with {tag}\'s writes zeroed. '
                f'Descriptive only.')
