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
#     display_name: representation-geometry (3.14.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Multi-layer toy classifiers: when does the phase pattern survive depth?
#
# This notebook shows every experiment behind [docs/toy.md](../docs/toy.md) §4.
# The question it answers: Li et al's single-layer toy classifier shows the three-phase RankMe
# trajectory, and so do real multi-layer LLMs, but our first multi-layer toys did not. What
# exactly decides whether a deep toy shows the pattern?
#
# ### Terminology
#
# - **Expansion phase / compression phase.** Our names for Li et al's "entropy-seeking" and
#   "compression-seeking" phases: the long RankMe rise, and the decline after the peak.
#
# - **The phase pattern.** The full trajectory shape from Li et al's Fig 4: a brief early dip,
#   then expansion, a peak, compression, and (in the toys) a slow partial recovery afterwards.
#   Where a run is said to "show the pattern", it means this shape is visible in its RankMe
#   curve.
#
# - **Saturated.** A run is saturated once its RankMe has stopped rising, so the curve is flat
#   at or near its peak. The expansion phase is over.
#
# - **The rare pair and its separation.** In the tasks below, the two classes with the fewest
#   samples start with identical weight vectors (call them $w_a$ and $w_b$). Early in training
#   they move together, so $\cos(w_a, w_b) \approx 1$; later they separate onto their own
#   directions. We call the moment of separation the **fork**. It is measured, not guessed:
#   every figure that talks about the fork has a bottom strip plotting $\cos(w_a, w_b)$ over
#   training, and the **fork step** is defined as the first checkpoint where the cosine drops
#   below 0.9 after having been above 0.97. Vertical lines in those figures mark the fork
#   step in the same color as the run.
#
# - **Learnable features.** These models have no input data at all: each training sample's
#   feature vector is itself a trained parameter (a row of the matrix $\theta$). Every
#   feature is learnable; the phrase is a reminder of this unusual setup, not a contrast
#   with some non-learnable features.
#
# - **RankMe centering.** For the 4-class and 6-class tasks the plots use uncentered RankMe
#   (Li et al's definition on raw features). For the 32-class task they use centered RankMe,
#   because there the layers' bias terms build up a large shared mean that dominates the
#   uncentered spectrum (shown in showcase_appendix §C).

# %% [markdown]
# ### Model architectures and training setup
#
# Everything is trained by full-batch gradient descent (plain SGD, no momentum) at a constant
# learning rate. There is no learning-rate schedule anywhere, including in the single-layer
# baseline. There are no normalization layers and no attention in any variant in this
# notebook; a "layer" here is NOT a transformer block.
#
# **B0 — the single-layer baseline (Li et al's model).**
# $\text{logits} = W\theta$ (features as column vectors; the code stores them as rows), where $\theta \in \mathbb{R}^{N \times d}$ are the learnable
# features and $W \in \mathbb{R}^{d \times C}$ the classifier. No layers in between.
#
# **The multi-layer variants.** $f_0 = \theta$, then $L$ layers are applied and the head reads
# the last representation: $\text{logits} = W f_L$. RankMe is measured on $f_L$ (the analog of
# measuring an LLM's final residual stream). One layer $g_k$ is:
#
# - **A1 — residual, linear layers:** $f_{l+1} = f_l + W_l f_l$ with $W_l \in \mathbb{R}^{d \times d}$,
#   no bias.
#
# - **A2 — residual, linear-tanh-linear layers:**
#   $f_{l+1} = f_l + W^{(2)}_l \tanh(W^{(1)}_l f_l + b^{(1)}_l) + b^{(2)}_l$, hidden width $4d$.
#   The tanh was inherited from the repo's first toy variants and is not otherwise
#   motivated; its weak rationale is bounded, zero-symmetric layer outputs (a write can
#   cancel as easily as reinforce). No other activation has been tried (§7).
#
# - **A3 — plain (no skip connections), linear layers:** $f_{l+1} = W_l f_l$. "Plain" follows
#   He et al's plain-vs-residual naming for networks without skip connections.
#
# **Layer weight initialisations** (PyTorch's default for a linear layer is Kaiming-uniform:
# each entry drawn from $U(-1/\sqrt{d_{in}}, +1/\sqrt{d_{in}})$):
#
# - **LI-default:** the PyTorch default above.
#
# - **LI-small:** the PyTorch default, then all layer parameters multiplied by 0.05.
#
# - **LI-identity** (A3 only, linear): $W_l = I + 0.05 \cdot \mathcal{N}(0, 1)$ per entry.
#
# **Feature/classifier initialisations:**
#
# - **FI-iid:** every entry of $\theta$ and $W$ independently $\mathcal{N}(0, 0.1^2)$.
#
# - **FI-constructed** (Li et al's, read off their Fig 4 markers; 4-class task): each frequent
#   class's feature rows clustered on its own direction (scale 0.6 to 0.75, spread 0.05), $W$
#   columns pre-aligned with those directions; both rare classes' rows exactly at the origin
#   with zero $W$ columns; then a global jitter $\delta \cdot \mathcal{N}(0,1)$ on everything.
#   $\delta$ controls how long the rare pair stays together, hence the fork step.
#
# - **FI-constructed-general** (6-class task): the same geometry for any dimension and class
#   count: frequent classes on orthonormal random directions, the rare pair at the origin,
#   same $\delta$ jitter.
#
# - **FI-flat:** every row of $\theta$ is a random multiple of ONE shared unit vector $u$,
#   plus $0.01 \cdot \mathcal{N}(0,1)$ noise. All samples start on a single line through the
#   origin, so the representation starts at rank $\approx$ 1.
#
# **Tasks:**
#
# | task | classes | samples per class | feature dim $d$ | lr | steps |
# |---|---|---|---|---|---|
# | T-skew | 4 | 2, 2, 1, 1 (each row ×3) | 2 | 0.25 | 1000 |
# | T-multi | 32 | 128, 64, 32, ..., 4, then 2×26 | 16 | 0.02 | 30,000 |
# | T-pair | 6 | 32, 16, 8, 4, 2, 2 | 4 | 0.25 | 3000 |
#
# T-skew is Li et al's task. T-multi is the repo's original multi-layer task. T-pair is
# T-multi's structure shrunk to where a single late fork can be arranged: heavy skew, a real
# bottleneck ($d < C$), and exactly one rare pair as the last two classes.
#
# A detail for exactness: where a task lists "each row ×3", the feature rows are duplicated
# and $\theta$'s learning rate is multiplied by the duplication factor, which makes the
# duplicated problem exactly equivalent to the original (it only rescales the covariance).

# %%
import os, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F_
import matplotlib.pyplot as plt

def caption(txt):
    """Figure caption, rendered below the figure it follows."""
    from IPython.display import display, HTML
    display(HTML(f'<div style="font-size:0.97em; opacity:0.95; max-width:56em; '
                 f'margin:0.2em 0 1.2em 0.5em"><b>Figure.</b> {txt}</div>'))

# %% [markdown]
# ### Diagram 1: Architectures

# %%
import matplotlib.patches as mpatches

def _dbox(ax, x, y, w, h, label, fc='#dbe9ff'):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.03',
                                         fc=fc, ec='k', lw=1))
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=9)

def _darr(ax, x0, y0, x1, y1):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='->', lw=1.2))

def _dline(ax, x0, y0, x1, y1):
    ax.plot([x0, x1], [y0, y1], color='k', lw=1.2, solid_capstyle='butt')

def _dplus(ax, x, y, r=0.14):
    ax.add_patch(mpatches.Circle((x, y), r, fc='white', ec='k', lw=1, zorder=3))
    ax.text(x, y, '+', ha='center', va='center', fontsize=10, zorder=4)

def _dgroup(ax, x0, y0, x1, y1, label):
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fc='none', ec='0.45',
                                    lw=0.9, ls=(0, (4, 2))))
    ax.text(x1 - 0.08, y0 + 0.08, label, fontsize=10, ha='right', va='bottom', color='0.3')

YS, YB = 1.95, 0.85           # stream height; branch height
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
for ax, (name, kind) in zip(axes, [('B0: single layer (Li et al)', 'b0'),
                                   ('A1: residual, linear layers', 'res'),
                                   ('A2: residual, linear-tanh-linear layers', 'res_mlp'),
                                   ('A3: plain (no skip connections), linear layers', 'plain')]):
    ax.set(xlim=(0, 12.6), ylim=(0, 2.9))
    ax.axis('off')
    ax.text(0.1, 2.62, name, fontsize=10, fontweight='bold')
    ax.text(0.35, YS, r'$x$', fontsize=12, ha='center', va='center')
    _darr(ax, 0.55, YS, 1.0, YS)
    _dbox(ax, 1.0, YS - 0.35, 1.3, 0.7, r'$f_\theta(\cdot)$', fc='#ffe7c2')
    ax.text(2.45, YS + 0.24, r'$f_\theta(x) \in \mathbb{R}^d$', fontsize=8)
    if kind == 'b0':
        _darr(ax, 2.3, YS, 8.9, YS)
    elif kind == 'res':
        _darr(ax, 2.3, YS, 8.9, YS)
        _dline(ax, 3.9, YS, 3.9, YB)                    # read: down, then right into the layer
        _darr(ax, 3.9, YB, 4.5, YB)
        _dbox(ax, 4.5, YB - 0.35, 1.4, 0.7, 'linear' + '\n' + r'$W_l$')
        _dline(ax, 5.9, YB, 6.9, YB)                    # write: right, then up into the sum
        _darr(ax, 6.9, YB, 6.9, YS - 0.14)
        _dplus(ax, 6.9, YS)
        _dgroup(ax, 3.55, 0.28, 7.35, 2.35, r'$\times\, L$')
    elif kind == 'res_mlp':
        _darr(ax, 2.3, YS, 8.9, YS)
        _dline(ax, 2.95, YS, 2.95, YB)                  # read: down, then right into the layer
        _darr(ax, 2.95, YB, 3.35, YB)
        _dbox(ax, 3.35, YB - 0.35, 1.25, 0.7, 'linear' + '\n' + r'$W_l^{(1)}$')
        _darr(ax, 4.6, YB, 4.9, YB)
        _dbox(ax, 4.9, YB - 0.35, 0.9, 0.7, 'tanh', fc='#d8f0d8')
        _darr(ax, 5.8, YB, 6.1, YB)
        _dbox(ax, 6.1, YB - 0.35, 1.3, 0.7, 'linear' + '\n' + r'$W_l^{(2)}$')
        _dline(ax, 7.4, YB, 7.95, YB)                   # write: right, then up into the sum
        _darr(ax, 7.95, YB, 7.95, YS - 0.14)
        _dplus(ax, 7.95, YS)
        _dgroup(ax, 2.6, 0.28, 8.35, 2.35, r'$\times\, L$')
    else:                                                # plain: the layer sits in series
        _darr(ax, 2.3, YS, 4.9, YS)
        _dbox(ax, 4.9, YS - 0.35, 1.4, 0.7, 'linear' + '\n' + r'$W_l$')
        _darr(ax, 6.3, YS, 8.9, YS)
        _dgroup(ax, 4.7, 1.32, 6.5, 2.58, r'$\times\, L$')
        ax.text(5.6, 1.12, '(in series)', fontsize=8, ha='center', color='0.3')
    ax.text(8.55, YS + 0.24, r'$f_L$', fontsize=11, ha='center')
    _dbox(ax, 8.9, YS - 0.35, 1.9, 0.7, 'linear classifier' + '\n' + r'$W_{\mathrm{cls}}$',
          fc='#e4d5f5')
    _darr(ax, 10.8, YS, 11.9, YS)
    ax.text(11.35, YS + 0.24, r'$W_{\mathrm{cls}} f_L$', fontsize=10, ha='center')
    ax.text(11.95, YS, 'logits', fontsize=9, va='center')
fig.suptitle('No attention and no normalization in any variant; trained by full-batch '
             'gradient descent at a constant learning rate')
plt.tight_layout()
plt.show()
caption('The four architectures. Following Li et al\'s Fig 4A, the input x (a one-hot '
        'sample id) passes through the feature map f_θ(·) into d dimensions; because the '
        'inputs are one-hot, f_θ(x) is simply that sample\'s row of the parameter matrix '
        'θ, which is why the features are learnable. A1 and A2 then apply L layers that '
        'read the running representation and add their output back at the ⊕ junction (the dashed box is the repeated unit, including the skip path and the sum); '
        'A2\'s layer is linear, tanh, linear (hidden width 4d). A3\'s layers sit in series '
        'with no skip connection ("plain" in He et al\'s plain-vs-residual sense). The '
        'classifier reads the final representation: logits = W_cls f_L. RankMe is '
        'measured on f_L.')

# %% [markdown]
# ### Diagram 2: Feature initialisations

# %%
from toy.train import Spec, SKEW, _clustered_init

torch.manual_seed(0)
theta_con, _ = _clustered_init(Spec(SKEW, d=2, lr=0.25, steps=300, clustered=True, dup=3))
theta_iid = 0.1 * torch.randn(24, 2)
u = torch.nn.functional.normalize(torch.randn(2), dim=0)
theta_flat = 0.1 * torch.randn(24, 1) * u + 0.01 * torch.randn(24, 2)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
cls_colors = ['magenta', 'orange', 'royalblue', 'seagreen']
labels = ['class 0 (n=2)', 'class 1 (n=2)', 'rare a (n=1)', 'rare b (n=1)']
row_cls = [0, 0, 1, 1, 2, 3]
for i, row in enumerate(theta_con):
    c = row_cls[i]
    axes[0].scatter(*row, s=70, color=cls_colors[c],
                    label=labels[c] if row_cls.index(c) == i else None)
axes[0].annotate('rare pair: both at the origin\n(split only by the jitter δ)',
                 (0, 0), xytext=(0.12, -0.35), fontsize=8,
                 arrowprops=dict(arrowstyle='->', lw=0.8))
axes[0].set_title('FI-constructed (T-skew): frequent classes pre-clustered')
axes[0].legend(fontsize=7, loc='upper left')
axes[1].scatter(*theta_iid.T, s=40, color='gray')
axes[1].set_title(r'FI-iid: every entry $\sim \mathcal{N}(0,\, 0.1^2)$')
axes[2].scatter(*theta_flat.T, s=40, color='gray')
axes[2].plot(*(torch.stack([-u, u]) * 0.35).T, color='tab:red', lw=1, ls='--')
axes[2].set_title('FI-flat: all rows on one shared direction (rank ≈ 1)')
for ax, pts in zip(axes, (theta_con, theta_iid, theta_flat)):
    m = 1.15 * float(pts.abs().max())
    ax.set(xlabel='dim 1', ylabel='dim 2', aspect='equal', xlim=(-m, m), ylim=(-m, m))
plt.tight_layout()
plt.show()
caption('The three feature initialisations, instantiated from the actual code (2-D for '
        'display; FI-iid and FI-flat generalize to any dimension). Left: Li et al\'s '
        'constructed geometry — each frequent class clustered on its own direction with '
        'the classifier pre-aligned, both rare classes exactly at the origin so they '
        'co-travel until the δ-jitter splits them. Middle: independent Gaussian entries. '
        'Right: every sample a random multiple of one shared unit vector plus small '
        'noise, so the representation starts at rank about 1.')

# %%
# Shared data loaders and the fork detector (no plots in this cell).
def toy_rm(tag, kind='acts_uncentered'):
    r = np.load(f'data/results/toy_{tag}/results_toy-{tag}.npy', allow_pickle=True).item()
    steps = sorted(r)
    return steps, [r[s]['before_final_norm'][kind]['rankme'] for s in steps]

def traj(tag):
    return torch.load(f'data/results/toy/trajectories_{tag}.pt')

def cos_rare(tag):
    """cos(w_a, w_b) of the two rare classes' weight vectors, per checkpoint."""
    t = traj(tag)
    ts, W = t['steps'].tolist(), t['W']
    return ts, [float(F_.cosine_similarity(W[i][:, -2], W[i][:, -1], dim=0)) for i in range(len(ts))]

def fork_step(tag):
    """First checkpoint where cos(w_a, w_b) < 0.9 after having been > 0.97 (definition above)."""
    ts, cos = cos_rare(tag)
    i = next((i for i in range(len(ts)) if cos[i] > 0.97), None)
    return None if i is None else next((ts[j] for j in range(i, len(ts)) if cos[j] < 0.9), None)

def panels(specs, suptitle, ylabel='RankMe (uncentered)'):
    """One column per spec: RankMe on top, loss strip below, and (when the spec has a rare
    pair) a cos(w_a, w_b) strip at the bottom. Vertical lines mark each run's fork step."""
    ncols = len(specs)
    any_cos = any(sp.get('cos', True) for sp in specs)
    nrows = 3 if any_cos else 2
    fig = plt.figure(figsize=(5.3 * ncols, 5.6 if any_cos else 4.6))
    gs = fig.add_gridspec(nrows, ncols, height_ratios=(3, 1, 1)[:nrows], hspace=0.14)
    for c, sp in enumerate(specs):
        tags = [t for t in sp['tags'] if os.path.exists(f'data/results/toy_{t}')]
        axm = fig.add_subplot(gs[0, c])
        axl = fig.add_subplot(gs[1, c], sharex=axm)
        axc = fig.add_subplot(gs[2, c], sharex=axm) if sp.get('cos', True) else None
        for i, tag in enumerate(tags):
            st, rm = toy_rm(tag, sp.get('kind', 'acts_uncentered'))
            ln, = axm.plot(st, rm, lw=1.6, label=sp['labels'][i])
            t = traj(tag)
            axl.plot(t['steps'].tolist(), np.maximum(t['loss'].numpy(), 1e-6),
                     lw=1.2, color=ln.get_color())
            if axc is not None:
                ts, cos = cos_rare(tag)
                axc.plot(ts, cos, lw=1.2, color=ln.get_color())
                if (fk := fork_step(tag)) is not None:
                    for ax in (axm, axl, axc):
                        ax.axvline(fk, color=ln.get_color(), lw=0.9, ls=':')
        for ax in filter(None, (axm, axl, axc)):
            ax.set_xscale('log')
        axl.set_yscale('log')
        axm.set_title(sp['title'], fontsize=10)
        axm.legend(fontsize=8, loc=sp.get('legend_loc', 'lower right'))
        (axc if axc is not None else axl).set_xlabel('training step')
        plt.setp(axm.get_xticklabels(), visible=False)
        if axc is not None:
            axc.axhline(0.9, color='gray', lw=0.7, ls='--')
            axc.set_ylim(-1.1, 1.1)
            plt.setp(axl.get_xticklabels(), visible=False)
        if c == 0:
            axm.set_ylabel(sp.get('ylabel', ylabel))
            axl.set_ylabel('loss (log)', fontsize=8)
            if axc is not None:
                axc.set_ylabel('cos(w_a, w_b)', fontsize=8)
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 1. Reference pattern
#
# **Point.**
# In the single-layer baseline, the compression phase starts exactly when the rare pair
# separates.
#
# This figure also shows how to read every later figure: RankMe on top, training loss in
# the middle, the rare pair's cosine at the bottom, and a vertical line at the fork step.

# %%
panels([{'title': 'B0 on T-skew, FI-constructed (run to 3000 steps)',
         'tags': ['single_long'], 'labels': ['single-layer baseline']}],
       'The single-layer baseline (B0): pattern and fork')
caption('The single-layer baseline on Li et al\'s task. Top: the phase pattern (dip, '
        'expansion, peak, compression, slow recovery). Bottom: the rare pair co-travels at '
        'cos ≈ 1 and separates at the vertical line; the compression phase begins there. '
        'The recovery afterwards is the transience result from toy.md §3.')

# %% [markdown]
# ## 2. The 32-class task shows no phase pattern at any depth
#
# **Point.**
# On T-multi, no depth from 0 to 6 produces the phase pattern, so depth is not what
# decides it.
#
# Depth 0 is the true single-layer version of this task, added because "is depth 1 like
# the single layer?" needs a measured answer. It is not:
#
# - with no layers, the spectrum barely moves (flat at 15.5, drifting to 11.4 late) and
#   the fit stays incomplete (final loss 0.11);
#
# - with even one layer, the trajectory becomes collapse-then-partial-recovery and the
#   task is solved.
#
# Why one layer changes the optimization path this much is an open item (§7).

# %%
panels([{'title': 'A2 on T-multi, FI-iid: depth 0 to 6',
         'tags': ['multi_d0', 'multi_residual_nonlinear_d1', 'multi_residual_nonlinear_d2',
                  'multi_residual_nonlinear_d3', 'multi_residual_nonlinear'],
         'labels': ['depth 0 (no layers)', 'depth 1', 'depth 2', 'depth 3', 'depth 6'],
         'kind': 'acts_centered', 'cos': False, 'legend_loc': 'lower left',
         'ylabel': 'RankMe (centered)'},
        {'title': 'A2 on T-multi, FI-flat (rank-1 start), depth 6',
         'tags': ['multi_residual_nonlinear_flat_s0', 'multi_residual_nonlinear_flat_s1'],
         'labels': ['seed 0', 'seed 1'],
         'kind': 'acts_centered', 'cos': False, 'legend_loc': 'lower left',
         'ylabel': 'RankMe (centered)'}],
       'T-multi never shows the pattern: not at any depth (left), not from a rank-1 start (right)')
caption('Left: the 32-class task at depths 0 to 6. No curve shows the phase pattern. Depth '
        '0 stays near full rank and never fully fits; every depth from 1 up collapses at '
        'learning onset and partially recovers to about 2.5. Right: starting the features '
        'at rank 1 (FI-flat) does not create an expansion phase either; the curve wanders '
        'inside the narrow window between its starting rank and the task\'s final rank. '
        'There is no cosine strip here because this task has 26 rare classes, not one pair.')

# %% [markdown]
# **Point.**
# T-multi cannot show the pattern because classes are learned strictly in frequency order,
# so its many rare-class separations happen while the spectrum is still rising.
#
# The figure shows the class-frequency groups' losses against the RankMe trajectory: the
# collapse happens while the frequent classes are being fit, the recovery while the mid
# and rare classes are being fit.
#
# By the visibility argument (toy.md §3.2; appendix §B1), a separation that happens before
# the curve is saturated leaves no visible mark.

# %%
t = traj('multi_residual_nonlinear')
steps_t, F_t, W_t = t['steps'].tolist(), t['F'], t['W']
counts = [max(2, 128 >> i) for i in range(32)]
y = torch.repeat_interleave(torch.arange(32), torch.tensor(counts))

def rm_centered(X):
    lam = torch.linalg.svdvals(X - X.mean(0)).square()
    p = lam / lam.sum(); p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))

ce = [F_.cross_entropy(F_t[i] @ W_t[i], y, reduction='none') for i in range(len(steps_t))]
groups = {'frequent classes (0-1)': y < 2, 'mid classes (2-5)': (y >= 2) & (y < 6),
          'rare classes (6-31)': y >= 6}
fig, ax1 = plt.subplots(figsize=(9.5, 4.3))
ax1.plot(steps_t, [rm_centered(F_t[i]) for i in range(len(steps_t))], 'k', lw=2.2,
         label='RankMe of $f_L$ (centered)')
ax1.set(xscale='log', xlabel='training step', ylabel='RankMe')
ax2 = ax1.twinx()
for (lbl, m), c in zip(groups.items(), ('tab:blue', 'tab:orange', 'tab:red')):
    ax2.plot(steps_t, [float(e[m].mean()) for e in ce], lw=1.6, color=c, label=lbl)
ax2.set_ylabel('mean loss of the group')
fig.legend(loc='upper right', fontsize=8)
plt.title('T-multi (A2, depth 6): the trajectory tracks class-frequency learning order')
plt.show()
caption('Black: the stream RankMe. Colors: mean loss per class-frequency group. The collapse '
        'coincides with the frequent group being fit (blue falling); the recovery coincides '
        'with the mid and rare groups being fit. The rare-class separations are spread over '
        'the whole recovery, each landing on a still-rising curve.')

# %% [markdown]
# ## 3. The same conditions that make the baseline work make deep residual networks work
#
# **Point.**
# A 6-layer residual network (A1) shows the complete phase pattern when both of the
# following hold, and only then:
#
# - (i) the fork lands after the curve is saturated (FI-constructed arranges this);
#
# - (ii) the layers start small enough not to scramble the constructed geometry
#   (LI-small arranges this).
#
# The compression phase starts at the fork in all three seeds; the cosine strips show
# this directly.
#
# Removing either condition removes the pattern: middle panel, default layer
# initialisation; right panel, the layer-scale threshold.

# %%
panels([{'title': 'A1 on T-skew, FI-constructed, LI-small (x0.05): 3 seeds',
         'tags': [f'single_d6_bs_s{s}' for s in (0, 1, 2)],
         'labels': [f'seed {s}' for s in (0, 1, 2)]},
        {'title': 'same, LI-default: the pattern is gone',
         'tags': [f'single_d6_s{s}' for s in (0, 1, 2)],
         'labels': [f'seed {s}' for s in (0, 1, 2)]},
        {'title': 'layer-scale sweep (seed 0)',
         'tags': ['single_d6_bs0p01_s0', 'single_d6_bs_s0', 'single_d6_bs0p1_s0', 'single_d6_bs0p3_s0'],
         'labels': ['x0.01', 'x0.05', 'x0.1', 'x0.3']}],
       '6-layer residual network (A1): the pattern appears under the two conditions, and only then')
caption('Left: with both conditions met, every seed shows dip, expansion, peak, compression '
        'at the fork (vertical lines, cosine strips below), and recovery. Middle: identical '
        'runs with default layer init show no pattern; the initial layer outputs are as '
        'large as the constructed feature geometry and destroy its timing (the cosine '
        'strips show forks landing early, on unsaturated curves). Right: the smallness '
        'condition is a threshold, not a trend: multiplying the default init by 0.01, 0.05 '
        'or 0.1 gives identical patterns; 0.3 already destroys it.')

# %%
panels([{'title': 'A1 on T-skew, FI-constructed, LI-small: depth 2 / 6 / 12',
         'tags': ['single_d2_bs_s0', 'single_d6_bs_s0', 'single_d12_bs_s0'],
         'labels': ['depth 2', 'depth 6', 'depth 12']}],
       'The pattern survives depth 2 to 12')
caption('Depth does not remove the pattern. Two regularities are visible but unexplained '
        '(§7): the compression step is larger at higher depth, and the fork arrives '
        'earlier.')

# %% [markdown]
# ## 4. Identity-initialised plain networks also show the pattern
#
# **Point.**
# A plain network (A3, no skip connections) shows the full pattern when its
# layers start near the identity map.
#
# The necessary condition is therefore that the map from the features to the measured
# representation starts near the identity, not the residual connection itself. A residual
# layer provides this automatically (identity path plus a small added term); a plain
# network has to be initialised there.
#
# "Near the identity" is only a statement about the initialisation. Nothing constrains
# the layers during training, there is no regularisation toward the identity anywhere,
# and the layers move freely from step 0.

# %%
panels([{'title': 'A3 (plain) on T-skew, FI-constructed, LI-identity: 3 seeds',
         'tags': [f'single_d6_id_plain_s{s}' for s in (0, 1, 2)],
         'labels': [f'seed {s}' for s in (0, 1, 2)]},
        {'title': 'A3 failure modes (no pattern)',
         'tags': ['single_d6_plain', 'single_d6_bs_plain_s0'],
         'labels': ['LI-default (fit incomplete, loss 0.44)', 'LI-small (never learns, loss = ln 4)'],
         'legend_loc': 'center right'}],
       'Plain networks (A3): identity-initialised layers show the pattern; generic inits fail')
caption('Left: six plain layers initialised at identity-plus-noise transmit the '
        'baseline dynamics to the output; the pattern and its fork timing match the '
        'residual version. Right: the two generic initialisations fail for different '
        'reasons. Default init composes six random maps, which is badly conditioned, and '
        'the representation pins near rank 1. Small init (the x0.05 recipe that works for '
        'residual layers) multiplies the signal by 0.05 six times, so no gradient reaches '
        'anything and the loss never moves from ln 4: "small layers" is only meaningful '
        'RELATIVE to an identity path, which a plain network does not have.')

# %% [markdown]
# ## 5. Open anomaly: tanh-MLP layers compress before the fork
#
# **Point.**
# With tanh-MLP layers (A2) the compression phase robustly begins well before the fork,
# so something other than the rare-pair separation is compressing these runs, and we do
# not know what.
#
# The cosine strips make the mismatch visible: the decline starts around step 30 while
# the forks land at steps 108 to 137, in all three seeds.
#
# Do not cite the nonlinear variant as explained by the fork mechanism.

# %%
panels([{'title': 'A2 on T-skew, FI-constructed, LI-small: 3 seeds',
         'tags': ['single_d6_bs_nl', 'single_d6_bs_nl_s1', 'single_d6_bs_nl_s2'],
         'labels': [f'seed {s}' for s in (0, 1, 2)]}],
       'tanh-MLP layers: the decline precedes the fork (unexplained)')
caption('The decline (top) starts near step 30 in every seed; the fork (vertical lines, '
        'strips below) arrives only near step 120. The compression here is also several '
        'times larger than in the linear version. Open item in §7.')

# %% [markdown]
# ## 6. The fork delay alone controls the pattern (T-pair)
#
# **Point.**
# On T-pair, delaying the single rare-pair fork past saturation switches the pattern on,
# with nothing else changed.
#
# The jitter scale δ of the constructed initialisation sets how long the rare pair stays
# together. At δ = 1e-3 the fork lands mid-expansion and leaves no mark; at δ = 1e-8 the
# same task, network, and initialisation geometry fork after the curve has flattened, and
# the compression phase appears at the fork step in all three seeds.

# %%
panels([{'title': 'T-pair, δ = 1e-3: fork lands mid-expansion',
         'tags': [f'skewpair_deep_s{s}' for s in (0, 1, 2)],
         'labels': [f'seed {s}' for s in (0, 1, 2)]},
        {'title': 'T-pair, δ = 1e-8: fork lands after saturation',
         'tags': [f'skewpair_deep_d8_s{s}' for s in (0, 1, 2)],
         'labels': [f'seed {s}' for s in (0, 1, 2)]}],
       'T-pair (A1, depth 6, LI-small): the fork delay alone decides the pattern')
caption('Same task, same network, same initialisation geometry; only δ differs. Left: with '
        'the fork at steps 130 to 170 (vertical lines), inside the expansion, no '
        'compression phase is visible. Right: with the fork delayed to about step 340, '
        'after the curve has flattened, RankMe drops at the fork step in all three seeds '
        '(seed 0: 2.96 to 2.58 between adjacent checkpoints). One honest wart: the '
        'constructed frequent-class geometry is not an equilibrium here, so every run '
        'starts with a collapse before its expansion; the pattern needs saturation before '
        'the fork, not a quiet start.')

# %% [markdown]
# ## 7. Known holes and open items
#
# - The fork step definition (cos < 0.9 after co-travel above 0.97) is a convention; no
#   sensitivity analysis of the thresholds has been run, and timing resolution is limited to
#   the checkpoint grid.
#
# - Several ablations are single-seed: the layer-scale sweep points (x0.01, x0.1, x0.3), the
#   depth-2 and depth-12 runs, and the new depth-0 run on T-multi.
#
# - Why a single layer changes T-multi's trajectory so drastically (flat near full rank at
#   depth 0 vs collapse-and-recover at depth 1, with the depth-0 fit staying incomplete at
#   loss 0.11) is unexplained. Expressivity is not the answer: the logits have rank at most
#   d in both cases.
#
# - The tanh-MLP pre-fork compression (§5) is unexplained; no candidate mechanism has been
#   tested.
#
# - The compression step growing with depth and the fork arriving earlier at higher depth
#   (§3) are observed, not modeled.
#
# - The layer-scale threshold is only located between x0.1 and x0.3.
#
# - One learning rate per task; no learning-rate sweeps for the deep variants.
#
# - T-pair's constructed frequent-class geometry is not an equilibrium (early collapse in
#   every run); it was not tuned further once the δ contrast worked.
#
# - LI-identity exists only for linear layers; a near-identity linear-tanh-linear layer
#   was not constructed.
#
# - The tanh activation was never motivated or ablated; no ReLU-family variant has been
#   run.
#
# - No normalization layers appear anywhere in this notebook; the normalization experiments
#   live in the architecture-knob grid (toy.md §6), which predates the conditions
#   established here and is marked for a rerun.
