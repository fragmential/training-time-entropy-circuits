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
# # The unembedding and the final stream (RQ1c.3)
#
# Verification-first: Li et al only ever show the features folding onto the classifier
# visually (Fig 4 B/C look alike). Before running the alignment statistic on LLMs, §1–§2
# test both halves on the toy model, where everything is inspectable — including a
# negative control. §3–§4 then run the IDENTICAL statistic on the LLMs.
#
# **BLIND dig** — captions are descriptive only (what is plotted, which runs), pending
# cross-check.
#
# The statistic (both scales): share_k = tr(P_k Σ) / tr(Σ), where P_k projects onto the
# top-k stream-space singular directions of the classifier/unembedding (left singular
# vectors of the toy's W (d × C); right singular vectors of the LLM head (V × d)), and Σ
# is the representation covariance (centered and uncentered both shown). Chance level for
# a random k-subspace is k/d.

# %%
import os, sys, importlib
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F_
import matplotlib.pyplot as plt
from analysis import experiments_lib as _lib

def caption(txt):
    """Figure caption, rendered below the figure it follows."""
    from IPython.display import display, HTML
    display(HTML(f'<div style="font-size:0.97em; opacity:0.95; max-width:56em; '
                 f'margin:0.2em 0 1.2em 0.5em"><b>Figure.</b> {txt}</div>'))

def traj(tag):
    return torch.load(f'data/results/toy/trajectories_{tag}.pt', weights_only=True)

PHASES = (':', '-', '--', '-')                           # warmup / expansion / compression / recovery

def phase_bounds(rm, eps=5e-4):
    """(dip, peak, trough): compression = largest post-warmup drawdown (appendix rule)."""
    g = int(np.argmax(rm))
    dip = int(np.argmin(rm[:g + 1]))
    dd = np.maximum.accumulate(rm[dip:]) - rm[dip:]
    if dd.max() < eps:
        return dip, len(rm) - 1, len(rm) - 1
    trough = dip + int(np.argmax(dd))
    return dip, dip + int(np.argmax(rm[dip:trough + 1])), trough

def rankme_uncentered(Fm):
    lam = torch.linalg.svdvals(Fm.float()).square()
    pr = (lam / lam.sum()).clamp(min=1e-12)
    return float(torch.exp(-(pr * pr.log()).sum()))

def fork_of(W_traj, steps):
    cos = [float(F_.cosine_similarity(W_traj[i][:, 2], W_traj[i][:, 3], dim=0))
           for i in range(len(steps))]
    shared = next((i for i in range(len(steps)) if cos[i] > 0.97), 0)
    j = next((j for j in range(shared, len(steps)) if cos[j] < 0.9), None)
    return None if j is None else int(steps[j])

def share_k(F, W, ks, centered=True, rotate_seed=None):
    """share_k = tr(P_k Σ_F)/tr(Σ_F); P_k from W's top-k stream-space singular vectors.
    rotate_seed: apply a fixed random rotation to W's stream side — the negative control
    (same spectrum, scrambled directions; expectation = k/d)."""
    if rotate_seed is not None:
        g = torch.Generator().manual_seed(rotate_seed)
        Q, _ = torch.linalg.qr(torch.randn(W.shape[0], W.shape[0], generator=g))
        W = Q @ W
    U = torch.linalg.svd(W.float(), full_matrices=False)[0]        # (d, min(d, C))
    X = F.float() - (F.float().mean(0) if centered else 0)
    tot = float((X ** 2).sum())
    proj = ((X @ U) ** 2).sum(0).cumsum(0)
    return [float(proj[k - 1] / tot) for k in ks]

# %% [markdown]
# ## 1. Toy, weights side: the classifier's singular values over training
#
# What 1a measures at LLM scale, run on the toy first: σ_i(W) per checkpoint, for the
# canonical run and every control variant.

# %%
TOY_VARIANTS = ('single_long', 'uniform', 'nobottleneck', 'mse_skew', 'mse_uniform')
CLS_COLORS = ('magenta', 'orange', 'royalblue', 'seagreen')
PLOT_UNTIL = 400

# %%
fig, axes = plt.subplots(3, 5, figsize=(21, 10.8))
for col, tag in enumerate(TOY_VARIANTS):
    t_ = traj(tag)
    keep = t_['steps'].numpy() <= PLOT_UNTIL
    steps = t_['steps'].numpy()[keep]
    Fs, Ws = t_['F'][keep], t_['W'][keep]
    rm = np.array([rankme_uncentered(Fs[i]) for i in range(len(steps))])
    sv = torch.stack([torch.linalg.svdvals(Ws[i].float()) for i in range(len(steps))]).numpy()
    bounds = phase_bounds(rm)
    ax = axes[0, col]
    ex = ax.twinx()
    for lo, hi, style in zip((0, *bounds), (*bounds, len(steps) - 1), PHASES):
        ax.plot(steps[lo:hi + 1], rm[lo:hi + 1], style, color='navy', lw=2)
        ex.plot(steps[lo:hi + 1], sv[lo:hi + 1, 0], style, color='violet', lw=1.5)
        ex.plot(steps[lo:hi + 1], sv[lo:hi + 1, 1], style, color='yellowgreen', lw=1.5)
    fk = fork_of(Ws, steps)
    if fk is not None:
        ax.axvline(fk, color='gray', lw=0.9, ls=':')
    ax.axvline(steps[bounds[1]], color='gray', lw=0.9, ls='--')
    ax.set(title=tag, ylabel='RankMe' if col == 0 else None)
    ex.set_ylabel(r'$\sigma_i(W)$' if col == 4 else None)
    if col == 0:
        from matplotlib.lines import Line2D
        ax.legend(fontsize=7, handles=[
            Line2D([], [], color='navy', lw=2, label='RankMe (phase-styled)'),
            Line2D([], [], color='violet', lw=1.5, label=r'$\sigma_1(W)$'),
            Line2D([], [], color='yellowgreen', lw=1.5, label=r'$\sigma_2(W)$')])
    ax = axes[1, col]
    logits = torch.matmul(Fs.float(), Ws.float())          # (T, N, C)
    for c in range(logits.shape[2]):
        ax.plot(steps, logits[:, :, c].square().mean(1).sqrt().numpy(), lw=1.4,
                color=CLS_COLORS[c], label=f'class {c}' if col == 0 else None)
    ax.plot(steps, logits.square().sum(2).mean(1).sqrt().numpy(), lw=2.2, color='0.5',
            label='total' if col == 0 else None)
    ax.set(ylabel=r'RMS of $W f_\theta(x)$' if col == 0 else None)
    if col == 0:
        ax.legend(fontsize=7)
    ax = axes[2, col]
    for c in range(Ws.shape[2]):
        ax.plot(steps, Ws[:, :, c].square().sum(1).sqrt().numpy(), lw=1.4, color=CLS_COLORS[c])
    ax.plot(steps, Ws.square().sum((1, 2)).sqrt().numpy(), lw=2.2, color='0.5')
    ax.set(xlabel='step', ylabel=r'$\|w_c\|$ (class weight norms)' if col == 0 else None)
fig.suptitle('Top: RankMe & classifier singular values (fig-4 D-panel style; fork dotted, decline onset dashed). '
             'Middle: logit magnitudes. Bottom: class weight-vector norms')
plt.tight_layout(); plt.show()
caption('Per variant, first 400 steps. Top row: uncentered feature RankMe (navy, '
        'phase-styled) with the classifier\'s singular values σ₁, σ₂ on the right axis '
        '(violet, yellowgreen). Middle row: RMS logit magnitude per class (class colors) '
        'and the RMS norm of the full logit vector (gray). Bottom row: the norm of each '
        'class\'s weight vector w_c (same colors), gray = ‖W‖_F. Descriptive only.')

# %%
# single_long only: W-vs-representation RankMe, per-class representation norms, and the
# per-class logit RMS (verbatim the panel above), side by side.
t_ = traj('single_long')
keep = t_['steps'].numpy() <= PLOT_UNTIL
steps = t_['steps'].numpy()[keep]
Fs, Ws = t_['F'][keep], t_['W'][keep]
y_cls = np.repeat(np.arange(4), 3 * np.array([2, 2, 1, 1]))     # class-blocked sample order
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.2))
ax0.plot(steps, [rankme_uncentered(Fs[i]) for i in range(len(steps))], lw=2, color='navy',
         label=r'RankMe $f_\theta(x)$')
ax0.plot(steps, [rankme_uncentered(Ws[i]) for i in range(len(steps))], lw=2, color='violet',
         label='RankMe $W$')
ax0.set(xlabel='step', ylabel='RankMe'); ax0.legend(fontsize=8)
for c in range(4):
    ax1.plot(steps, Fs[:, y_cls == c, :].square().sum(-1).mean(1).sqrt().numpy(), lw=1.4,
             color=CLS_COLORS[c], label=f'class {c}')
ax1.plot(steps, Fs.square().sum(-1).mean(1).sqrt().numpy(), lw=2.2, color='0.5', label='total')
ax1.set(xlabel='step', ylabel=r'RMS of $f_\theta(x)$'); ax1.legend(fontsize=7)
logits = torch.matmul(Fs.float(), Ws.float())
for c in range(logits.shape[2]):
    ax2.plot(steps, logits[:, :, c].square().mean(1).sqrt().numpy(), lw=1.4,
             color=CLS_COLORS[c], label=f'class {c}')
ax2.plot(steps, logits.square().sum(2).mean(1).sqrt().numpy(), lw=2.2, color='0.5', label='total')
ax2.set(xlabel='step', ylabel=r'RMS of $W f_\theta(x)$'); ax2.legend(fontsize=7)
fig.suptitle('single_long: W vs representation RankMe; per-class representation and logit magnitudes')
plt.tight_layout(); plt.show()
caption('single_long, first 400 steps. Left: uncentered RankMe of the representations '
        '(navy) and of the classifier W (violet), one axis. Middle: RMS norm of the '
        'representations per sample class (class colors; gray = all samples). Right: RMS '
        'logit magnitude per class, identical to the middle-row panel of the figure '
        'above. Descriptive only.')

# %% [markdown]
# ## 2. Toy, alignment: feature variance in the classifier's top-k subspace
#
# The 1b statistic on the toy. Single-layer runs only: there W reads the features F
# directly, so F ↔ W is exactly the stream-feeding-the-head pair the LLM statistic
# measures (in a deep toy, W reads the final stream, not F — comparing F to W would be
# the wrong pair). Shown per variant: centered and uncentered share, the
# chance line k/d, and the negative control — the same W with its stream side randomly
# rotated (same spectrum, scrambled directions).

# %%
K1 = (1,)
fig, axes = plt.subplots(1, 5, figsize=(19, 3.6), sharey=True)
for ax, tag in zip(axes, TOY_VARIANTS):
    t = traj(tag)
    steps = t['steps'].numpy()
    for centered, ls, lbl in ((True, '-', 'centered'), (False, '--', 'uncentered')):
        ys = [share_k(t['F'][i], t['W'][i], K1, centered=centered)[0] for i in range(len(steps))]
        ax.plot(steps, ys, ls, lw=1.6, color='tab:blue', label=f'share (k=1, {lbl})')
    rot = [share_k(t['F'][i], t['W'][i], K1, rotate_seed=0)[0] for i in range(len(steps))]
    ax.plot(steps, rot, lw=1.4, color='tab:red', label='rotated-W control')
    d = t['F'].shape[2]
    ax.axhline(1 / d, color='gray', lw=0.9, ls=':', label=f'chance k/d = {1 / d:g}')
    ax.set(xscale='log', xlabel='step', title=tag, ylim=(0, 1.02))
    if ax is axes[0]:
        ax.set_ylabel('share of feature variance')
        ax.legend(fontsize=7)
fig.suptitle('Feature variance in the classifier\'s top-1 subspace (d=2), per variant')
plt.tight_layout(); plt.show()
caption('share_1 over training for the canonical run and the controls: solid = centered '
        'feature covariance, dashed = uncentered, red = the rotated-W negative control, '
        'dotted gray = chance (k/d = 0.5). Descriptive only.')

# %%
t = traj('multi_d0')
steps = t['steps'].numpy()
KS16 = (1, 4, 8)
fig, ax = plt.subplots(figsize=(8, 4.2))
for k, c in zip(KS16, ('tab:blue', 'tab:orange', 'tab:green')):
    ys = [share_k(t['F'][i], t['W'][i], (k,))[0] for i in range(len(steps))]
    rot = [share_k(t['F'][i], t['W'][i], (k,), rotate_seed=0)[0] for i in range(len(steps))]
    ax.plot(steps, ys, lw=1.6, color=c, label=f'k={k}')
    ax.plot(steps, rot, lw=1.1, color=c, alpha=0.45)
    ax.axhline(k / 16, color=c, lw=0.8, ls=':')
ax.set(xscale='log', xlabel='step', ylabel='share of feature variance (centered)',
       title='32-class task at depth 0 (d=16): share_k, faded = rotated-W control, dotted = chance k/16')
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
caption('The same statistic on the 32-class task at depth 0, k = 1, 4, 8: solid = '
        'measured share, faded = rotated-W negative control, dotted = chance. '
        'Descriptive only.')

# %% [markdown]
# ### CKA alignment, with and without the frequency/mean corrections
#
# The k-free alignment statistic: linear CKA ⟨Σ_F, W̃W̃ᵀ⟩ / (‖Σ_F‖·‖W̃W̃ᵀ‖), Σ_F the centered
# feature covariance, W̃ the classifier under the same corrections as the LLM head variants
# (§3): standard; freq_centered = subtract the class-frequency-weighted mean class vector
# (the softmax-invariant part); mean_deflated = project the feature-mean direction off W's
# stream side. freq_mean (both at once) is hidden here: at d=2 the pair of rank-1 removals
# strips a 2-dim stream side to ~nothing (nobottleneck's d=3 survives it) — degenerate in
# the toy regime, meaningful only at LLM widths. Red = rotated-W negative control. Second
# figure: RankMe of the same W̃ per variant. Third/fourth figures: the same two statistics
# with the control panels replaced by the appendix init homotopy (α = 0 clustered →
# 1 iid-tiny init; data: appendix_sweeps.pt).

# %%
TOY_COUNTS = {'single_long': (2, 2, 1, 1), 'uniform': (2, 2, 2, 2), 'nobottleneck': (2, 2, 1, 1),
              'mse_skew': (2, 2, 1, 1), 'mse_uniform': (2, 2, 2, 2)}
VKEYS = ('standard', 'freq_centered')
VCOLORS = ('tab:blue', 'tab:orange')

def w_variants(W, F, counts):
    f = torch.tensor(counts, dtype=torch.float32); f = f / f.sum()
    m = W.float() @ f                                    # frequency-weighted mean class vector
    mu = F.float().mean(0)
    muh = mu / mu.norm().clamp(min=1e-12)
    P = torch.eye(len(mu)) - torch.outer(muh, muh)
    Wf = W.float() - m[:, None]
    return {'standard': W.float(), 'freq_centered': Wf,
            'mean_deflated': P @ W.float(), 'freq_mean': P @ Wf}

def cka_fw(F, W, rotate_seed=None):
    if rotate_seed is not None:
        g = torch.Generator().manual_seed(rotate_seed)
        Q, _ = torch.linalg.qr(torch.randn(W.shape[0], W.shape[0], generator=g))
        W = Q @ W.float()
    X = F.float() - F.float().mean(0)
    S = X.T @ X / X.shape[0]
    G = W.float() @ W.float().T
    return float((S * G).sum() / (S.norm() * G.norm()).clamp(min=1e-12))

# The appendix init homotopy: α = 0 clustered → 1 iid-tiny init (all SKEW counts, dense
# per-step trajectories); panels below reuse it.
SW = torch.load('data/results/toy/appendix_sweeps.pt', weights_only=True)
HOM_ALPHAS, HOM_SEEDS = ('0', '0.5', '0.75', '0.875', '1'), (0, 1, 2)

# %%
fig, axes = plt.subplots(1, 5, figsize=(19, 3.6), sharey=True)
for ax, tag in zip(axes, TOY_VARIANTS):
    t = traj(tag)
    steps = t['steps'].numpy()
    variants = [w_variants(t['W'][i], t['F'][i], TOY_COUNTS[tag]) for i in range(len(steps))]
    for v, c in zip(VKEYS, VCOLORS):
        ax.plot(steps, [cka_fw(t['F'][i], variants[i][v]) for i in range(len(steps))],
                lw=1.6, color=c, label=v)
    ax.plot(steps, [cka_fw(t['F'][i], t['W'][i], rotate_seed=0) for i in range(len(steps))],
            lw=1.4, color='tab:red', label='rotated-W control')
    ax.set(xscale='log', xlabel='step', title=tag, ylim=(-0.05, 1.02))
    if ax is axes[0]:
        ax.set_ylabel(r'CKA$(\Sigma_F, \tilde{W}\tilde{W}^\top)$')
        ax.legend(fontsize=6)
fig.suptitle('CKA alignment of features and classifier, per W correction')
plt.tight_layout(); plt.show()
caption('Linear CKA between the centered feature covariance and W̃W̃ᵀ over training, per '
        'variant: the four W corrections (colors) and the rotated-W negative control '
        '(red). Descriptive only.')

fig, axes = plt.subplots(1, 5, figsize=(19, 3.6))
for ax, tag in zip(axes, TOY_VARIANTS):
    t = traj(tag)
    steps = t['steps'].numpy()
    variants = [w_variants(t['W'][i], t['F'][i], TOY_COUNTS[tag]) for i in range(len(steps))]
    for v, c in zip(VKEYS, VCOLORS):
        ax.plot(steps, [rankme_uncentered(variants[i][v]) for i in range(len(steps))],
                lw=1.6, color=c, label=v)
    ax.plot(steps, [rankme_uncentered(t['F'][i]) for i in range(len(steps))],
            '--', lw=1.4, color='0.5', label=r'RankMe $f_\theta(x)$')
    ax.set(xscale='log', xlabel='step', title=tag)
    if ax is axes[0]:
        ax.set_ylabel('RankMe of W̃')
        ax.legend(fontsize=6)
fig.suptitle('RankMe of the classifier under each correction')
plt.tight_layout(); plt.show()
caption('RankMe of W̃ over training for the three corrections (colors), with the '
        'representations\' uncentered RankMe (dashed gray) for reference. Descriptive '
        'only.')

# %%
# The same two statistics across the init homotopy (SW): columns = α blends of the
# clustered init toward iid-tiny, rows = seeds.
for stat in ('cka', 'rankme'):
    fig, axes = plt.subplots(3, 5, figsize=(19, 10))
    for row, s in enumerate(HOM_SEEDS):
        for col, a in enumerate(HOM_ALPHAS):
            ax = axes[row, col]
            e = SW['homotopy'][f'{a}/s{s}']
            steps, Fs, Ws = np.arange(len(e['rm'])), e['theta_path'], e['W_path']
            ok = steps > 0
            variants = [w_variants(Ws[i], Fs[i], (2, 2, 1, 1)) for i in range(len(steps))]
            for v, c in zip(VKEYS, VCOLORS):
                ys = np.asarray([cka_fw(Fs[i], variants[i][v]) if stat == 'cka'
                                 else rankme_uncentered(variants[i][v]) for i in range(len(steps))])
                ax.plot(steps[ok], ys[ok], lw=1.6, color=c, label=v)
            if stat == 'cka':
                rot = np.asarray([cka_fw(Fs[i], Ws[i], rotate_seed=0) for i in range(len(steps))])
                ax.plot(steps[ok], rot[ok], lw=1.4, color='tab:red', label='rotated-W control')
            else:
                ax.plot(steps[ok], e['rm'].numpy()[ok], '--', lw=1.4, color='0.5',
                        label=r'RankMe $f_\theta(x)$')
            ax.set(xscale='log', title=f'α={a}, s{s}' if row == 0 else f's{s}',
                   xlabel='step' if row == 2 else None)
            if row == 0 and col == 0:
                ax.legend(fontsize=6)
            if col == 0:
                ax.set_ylabel(r'CKA$(\Sigma_F, \tilde{W}\tilde{W}^\top)$' if stat == 'cka'
                              else 'RankMe of W̃')
    fig.suptitle(('CKA alignment' if stat == 'cka' else 'Classifier RankMe')
                 + ' across the init homotopy (clustered → iid), one row per seed')
    plt.tight_layout(); plt.show()
    caption(('Linear CKA between the centered feature covariance and W̃W̃ᵀ'
             if stat == 'cka' else 'RankMe of W̃ (dashed gray = feature RankMe)')
            + ' over training for the init-homotopy runs: columns α ∈ {0, 0.5, 0.75, '
              '0.875, 1} (clustered → iid-tiny init), rows = seeds 0–2. Same '
              'corrections/colors as above. Descriptive only.')

# %% [markdown]
# ## 3. LLMs, weights side: unembedding singular spectra over training
#
# From data/results/unembedding_spectra.pt (weights-only; all 7 models).

# %%
# Weights-side sources and spectrum statistics, shared by every §3 figure (and §4's d).
# US: raw unembedding singular values per model/step. VS: the corrected variants
# (W − 1·(Wᵀf), W(I−μ̂μ̂ᵀ), both), from oneoff_scripts/unembedding_variant_spectra.py —
# absent until that job lands, in which case those figures are skipped.
US = torch.load('data/results/unembedding_spectra.pt', weights_only=False)
VS_PATH = 'data/results/unembedding_variant_spectra.pt'
VS = torch.load(VS_PATH, weights_only=False) if os.path.exists(VS_PATH) else None
HEAD_VARIANTS = ('standard', 'drop_top1', 'freq_centered', 'mean_deflated', 'freq_mean')
WINDOWS = ((11, 100), (128, 512))                 # alphaReQ rank windows

def _rm(sv):
    lam = sv.numpy().astype(float) ** 2
    p = lam / lam.sum(); p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

def _al(sv, k0, k1):
    """alphaReQ: weighted log-log slope of the sv² decay over ranks [k0, k1)."""
    try:
        return _lib._alpha(sv.numpy().astype(float) ** 2, k0=k0, k1=k1)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan

def _sv_of(src, per_step, step, var):
    """The spectrum to score: raw for standard, sv[1:] for drop_top1, else the variant."""
    sv = per_step[step] if src is US else per_step[step].get(var)
    return None if sv is None else (sv[1:] if var == 'drop_top1' else sv)

# %%
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, (model, per_step) in zip(axes.flat, US.items()):
    steps = sorted(per_step)
    show = steps[:: max(1, len(steps) // 6)]
    for i, s in enumerate(show):
        sv = per_step[s].numpy()
        ax.loglog(np.arange(1, len(sv) + 1), sv, lw=1.2,
                  color=plt.get_cmap('viridis')(i / max(len(show) - 1, 1)), label=f'step {s}')
    ax.set(title=model, xlabel='rank', ylabel='σ_i')
    ax.legend(fontsize=6)
for ax in axes.flat[len(US):]:
    ax.axis('off')
fig.suptitle('Unembedding singular spectra at ~6 checkpoints per model (light → dark = later)')
plt.tight_layout(); plt.show()
caption('Singular-value spectra of each model\'s unembedding at ~6 checkpoints across '
        'training. Descriptive only.')

# %%
# One fig per spectrum: standard = raw sv; drop_top1 = sv[1:]; freq_centered = W − 1·(Wᵀf)
# (empirical unigram f from the collection mix — softmax-invariant); mean_deflated =
# W(I−μ̂μ̂ᵀ) with μ the after_final_norm stream mean at that step; freq_mean = both.
for var in HEAD_VARIANTS:
    src = US if var in ('standard', 'drop_top1') else VS
    if src is None:
        print(f'{var}: unembedding_variant_spectra.pt not present yet — job pending')
        continue
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, (model, per_step) in zip(axes.flat, src.items()):
        steps = sorted(per_step)
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        keep = xs > 0
        ys = np.asarray([_rm(sv) if (sv := _sv_of(src, per_step, s, var)) is not None
                         else np.nan for s in steps])
        ok = keep & np.isfinite(ys)                       # join across missing checkpoints
        ax.plot(xs[ok], ys[ok], lw=1.8)
        ax.set(xscale='log', xlabel='tokens', title=model)
    for ax in axes.flat[len(src):]:
        ax.axis('off')
    axes[0, 0].set_ylabel('RankMe of σ²(unembedding)')
    axes[1, 0].set_ylabel('RankMe of σ²(unembedding)')
    fig.suptitle(f'Effective rank of the unembedding over training — {var}')
    plt.tight_layout(); plt.show()
    caption(f'RankMe of the unembedding\'s {var} spectrum over training, one panel per '
            f'model, tokens on the x-axis. Descriptive only.')

# %%
# The same figures with alphaReQ (weighted log-log slope of the sv² decay) instead of
# RankMe; two rank windows per panel: 11–100 (the standard bulk window) and 128–512.
for var in HEAD_VARIANTS:
    src = US if var in ('standard', 'drop_top1') else VS
    if src is None:
        print(f'{var}: unembedding_variant_spectra.pt not present yet — job pending')
        continue
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, (model, per_step) in zip(axes.flat, src.items()):
        steps = sorted(per_step)
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        keep = xs > 0
        ys0 = None
        for k0, k1 in WINDOWS:
            ys = np.asarray([_al(sv, k0, k1) if (sv := _sv_of(src, per_step, s, var)) is not None
                             else np.nan for s in steps])
            ok = keep & np.isfinite(ys)                   # join across missing checkpoints
            ax.plot(xs[ok], ys[ok], lw=1.8, label=f'{k0}–{k1}')
            ys0 = ys[ok] if ys0 is None else ys0
        if ys0 is not None and len(ys0):                  # y-range framed on 11–100 only
            pad = 0.05 * (ys0.max() - ys0.min()) or 0.05
            ax.set_ylim(ys0.min() - pad, ys0.max() + pad)
        ax.set(xscale='log', xlabel='tokens', title=model)
        ax.legend(fontsize=6)
    for ax in axes.flat[len(src):]:
        ax.axis('off')
    axes[0, 0].set_ylabel('alphaReQ of σ²(unembedding)')
    axes[1, 0].set_ylabel('alphaReQ of σ²(unembedding)')
    fig.suptitle(f'alphaReQ of the unembedding over training — {var}')
    plt.tight_layout(); plt.show()
    caption(f'alphaReQ (log-log slope of the sv² spectrum) of the unembedding\'s {var} '
            f'spectrum over training, rank windows 11–100 and 128–512, one panel per '
            f'model. Descriptive only.')

# %% [markdown]
# ## 4. LLMs, alignment: final-stream variance in the unembedding's top-k subspace
#
# The identical statistic to §2, from data/results/unembedding_alignment.pt (produced by
# oneoff_scripts/unembedding_alignment.py; matrix products over the stored cov_svd files
# and head singular vectors). Both final leaves; chance = k/d.

# %%
# Alignment sources for §4 (each None until its job lands; the figures below skip on None).
LEAVES = ('before_final_norm', 'after_final_norm')
CKEYS = ('standard', 'freq_centered', 'mean_deflated', 'freq_mean')
CCOLORS = ('tab:blue', 'tab:orange', 'tab:green', 'tab:purple')
ROT_SEEDS = (0, 1, 2)

def _load(path):
    return torch.load(path, weights_only=False) if os.path.exists(path) else None

AL = _load('data/results/unembedding_alignment.pt')       # share_k
CK = _load('data/results/unembedding_cka.pt')             # CKA, packed stream
CKP = _load('data/results/unembedding_cka_padded.pt')     # CKA, padded last-token stream

# %%
if AL is None:
    print('unembedding_alignment.pt not present yet — job pending')
else:
    for leaf in LEAVES:
        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        for ax, (model, res) in zip(axes.flat, AL.items()):
            r = res[leaf]
            steps, ks = r['steps'], r['ks']
            share = np.asarray(r['share_centered'])
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            keep = xs > 0
            d = len(next(iter(US[model].values())))       # min(d, V) = d for every model here
            for j, k in enumerate(ks):
                c = plt.get_cmap('viridis')(j / (len(ks) - 1))
                ax.plot(xs[keep], share[keep, j], lw=1.5, color=c, label=f'k={k}')
                ax.axhline(min(k / d, 1), color=c, lw=0.7, ls=':')
            ax.set(xscale='log', title=model, xlabel='tokens', ylabel='share (centered)', ylim=(0, 1.02))
            ax.legend(fontsize=6)
        for ax in axes.flat[len(AL):]:
            ax.axis('off')
        fig.suptitle(f'Final-stream variance in the unembedding\'s top-k subspace — {leaf}')
        plt.tight_layout(); plt.show()
        caption(f'share_k per checkpoint at {leaf}, one panel per model, k = 1…512 '
                f'(light → dark); dotted = chance k/d. Descriptive only.')

# %% [markdown]
# ### CKA alignment: stream covariance vs corrected head Grams
#
# From data/results/unembedding_cka.pt (oneoff_scripts/unembedding_cka.py): the k-free
# statistic s = tr(Σc·G) / (‖Σc‖·‖G‖) per checkpoint, leaf, and head correction
# (docs/cka_alignment.md). Σc is the centered stream covariance from the final_stream_svd
# files; G = W̃ᵀW̃. Shaded bands = the rotated-W controls (min–max over 3 seeds) for
# standard and freq_centered — the spectrum-matched chance floor; the quantity to read is
# the gap to the matching solid line, not the absolute level.

# %%
if CK is None:
    print('unembedding_cka.pt not present yet — job pending')
else:
    for leaf in LEAVES:
        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        for ax, (model, by) in zip(axes.flat, CK.items()):
            steps = sorted(by)
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            keep = xs > 0
            for v, c in zip(CKEYS, CCOLORS):
                ys = np.asarray([by[s][leaf][v] for s in steps])
                ax.plot(xs[keep], ys[keep], lw=1.6, color=c, label=v)
            for v, c in (('standard', 'tab:blue'), ('freq_centered', 'tab:orange')):
                rot = np.asarray([[by[s][leaf][f'{v}_rot{r}'] for r in ROT_SEEDS] for s in steps])
                ax.fill_between(xs[keep], rot[keep].min(1), rot[keep].max(1), color=c,
                                alpha=0.25, lw=0, label=f'{v} rotated')
            ax.set(xscale='log', xlabel='tokens', title=model)
            ax.legend(fontsize=6)
        for ax in axes.flat[len(CK):]:
            ax.axis('off')
        axes[0, 0].set_ylabel(r'CKA$(\Sigma_c, \tilde{W}^\top\tilde{W})$')
        axes[1, 0].set_ylabel(r'CKA$(\Sigma_c, \tilde{W}^\top\tilde{W})$')
        fig.suptitle(f'Stream ↔ unembedding CKA over training — {leaf}')
        plt.tight_layout(); plt.show()
        caption(f'CKA between the centered stream covariance at {leaf} and the head Gram '
                f'under each correction (solid lines), with the rotated-W controls as '
                f'shaded min–max bands (3 seeds) for standard and freq_centered. One '
                f'panel per model, tokens on the x-axis. Descriptive only.')

# %%
if CK is None:
    print('unembedding_cka.pt not present yet — job pending')
else:
    for leaf in LEAVES:
        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        for ax, (model, by) in zip(axes.flat, CK.items()):
            steps = sorted(by)
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            keep = xs > 0
            rot_base = np.asarray([[by[s][leaf][f'standard_rot{r}'] for r in ROT_SEEDS] for s in steps])[keep].mean(1)
            for v, c in zip(CKEYS, CCOLORS):
                ys = np.asarray([by[s][leaf][v] for s in steps])
                ax.plot(xs[keep], ys[keep]/rot_base, lw=1.6, color=c, label=v)
            for v, c in (('standard', 'tab:blue'), ('freq_centered', 'tab:orange')):
                rot = np.asarray([[by[s][leaf][f'{v}_rot{r}'] for r in ROT_SEEDS] for s in steps])
                ax.fill_between(xs[keep], rot[keep].min(1)/rot_base, rot[keep].max(1)/rot_base, color=c,
                                alpha=0.25, lw=0, label=f'{v} rotated')
            ax.set(xscale='log', xlabel='tokens', title=model)
            ax.legend(fontsize=6)
        for ax in axes.flat[len(CK):]:
            ax.axis('off')
        axes[0, 0].set_ylabel(r'CKA$(\Sigma_c, \tilde{W}^\top\tilde{W})$')
        axes[1, 0].set_ylabel(r'CKA$(\Sigma_c, \tilde{W}^\top\tilde{W})$')
        fig.suptitle(f'Stream ↔ unembedding CKA over training — {leaf}')
        plt.tight_layout(); plt.show()
        caption(f'CKA between the centered stream covariance at {leaf} and the head Gram '
                f'under each correction (solid lines), with the rotated-W controls as '
                f'shaded min–max bands (3 seeds) for standard and freq_centered. One '
                f'panel per model, tokens on the x-axis. Descriptive only.')

# %%
# The identical figures on the padded (last-token) stream set — Σc from
# final_stream_svd_padded via unembedding_cka.py --stream_dir; no nanochat (no padded data).
if CKP is None:
    print('unembedding_cka_padded.pt not present yet — job pending')
else:
    for leaf in LEAVES:
        fig, axes = plt.subplots(2, 4, figsize=(19, 8))
        for ax, (model, by) in zip(axes.flat, CKP.items()):
            steps = sorted(by)
            xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
            keep = xs > 0
            for v, c in zip(CKEYS, CCOLORS):
                ys = np.asarray([by[s][leaf][v] for s in steps])
                ax.plot(xs[keep], ys[keep], lw=1.6, color=c, label=v)
            for v, c in (('standard', 'tab:blue'), ('freq_centered', 'tab:orange')):
                rot = np.asarray([[by[s][leaf][f'{v}_rot{r}'] for r in ROT_SEEDS] for s in steps])
                ax.fill_between(xs[keep], rot[keep].min(1), rot[keep].max(1), color=c,
                                alpha=0.25, lw=0, label=f'{v} rotated')
            ax.set(xscale='log', xlabel='tokens', title=model)
            ax.legend(fontsize=6)
        for ax in axes.flat[len(CKP):]:
            ax.axis('off')
        axes[0, 0].set_ylabel(r'CKA$(\Sigma_c, \tilde{W}^\top\tilde{W})$')
        axes[1, 0].set_ylabel(r'CKA$(\Sigma_c, \tilde{W}^\top\tilde{W})$')
        fig.suptitle(f'Stream ↔ unembedding CKA over training — {leaf} (padded, last token)')
        plt.tight_layout(); plt.show()
        caption(f'The identical statistic on the padded last-token stream set at {leaf}: '
                f'CKA between the centered stream covariance and the head Gram per '
                f'correction, rotated-W controls shaded. No nanochat (no padded data). '
                f'Descriptive only.')

# %% [markdown]
# ### Head alphaReQ vs stream alphaReQ, same window
#
# The windowed alphaReQ of the standard unembedding spectrum (§3) overlaid on the SAME
# statistic computed on the model's own after_final_norm centered stream covariance
# (block_representations_samples; nanochat from nanochat_samples). Separate y-axes — the
# two live on different scales; only the shapes are comparable. Shown for the bulk window
# 11–100 and for the deeper 32–512.

# %%
BLOCK_SAMPLES = 'block_representations_samples'
AFN_HOOK = ('after_final_norm', 'acts_centered')

def cfg_of(model):
    return 'nanochat_samples' if model == 'nanochat-d12' else BLOCK_SAMPLES

X_MIN = 10 ** 8.5                                  # shared x starts here (non-nanochat panels)

def _span(vs, log=False):
    """Common axis range over a list of value arrays, padded 5% (multiplicatively if log)."""
    lo, hi = min(v.min() for v in vs), max(v.max() for v in vs)
    if log:
        f = (hi / lo) ** 0.02
        return lo / f, hi * f
    pad = 0.05 * (hi - lo) or 0.05
    return lo - pad, hi + pad

def alpha_overlay(k0, k1, rankme=False):
    """Per model: unembedding alphaReQ[k0,k1) (blue, left) vs the same statistic on its own
    after_final_norm stream (red, twin right), and with rankme=True the stream's RankMe as a
    third series (green, second right spine). Every color and the token axis share one range
    across panels — set explicitly, not via sharex/sharey, so every panel keeps its own
    drawn-and-labelled axes — with nanochat excluded from all of them (its run is ~2 dex
    shorter and its levels sit elsewhere)."""
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    rights, shared, x_vals = [], [], []
    vals = {'blue': [], 'red': [], 'green': []}                  # non-nanochat, on-screen only
    for ax, (model, per_step) in zip(axes.flat, US.items()):
        steps = sorted(per_step)
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        ys = np.asarray([_al(per_step[s], k0, k1) for s in steps])
        ok = (xs > 0) & np.isfinite(ys)
        lines = [ax.plot(xs[ok], ys[ok], lw=1.8, color='tab:blue',
                         label='unembedding (standard)')[0]]
        ax.tick_params(axis='y', labelcolor='tab:blue')
        share = 'nanochat' not in model                          # nanochat: own x and y ranges
        if share:
            x_vals.append(xs[ok])
            if (vis := ok & (xs >= X_MIN)).any():
                vals['blue'].append(ys[vis])

        exs = [ax.twinx()]
        if rankme:
            exs.append(ax.twinx())
            exs[1].spines['right'].set_position(('outward', 42))
        rights.append(exs)
        series = [('red', 'alpha_window', {'k0': k0, 'k1': k1}, 'stream alphaReQ')]
        if rankme:
            series.append(('green', 'rankme', {}, 'stream RankMe'))
        for ex, (col, yvar, kw, lbl) in zip(exs, series):
            sy, ssteps = _lib.get_ys(cfg_of(model), model, AFN_HOOK, yvar, kw)
            ex.tick_params(axis='y', labelcolor=f'tab:{col}')
            if sy is None:
                continue
            sxs = np.asarray(_lib.get_xs_tokens(model, ssteps), float)
            sy = np.asarray(sy, float)
            sok = (sxs > 0) & np.isfinite(sy)
            lines.append(ex.plot(sxs[sok], sy[sok], lw=1.8, color=f'tab:{col}', label=lbl)[0])
            if share:
                x_vals.append(sxs[sok])
                if (svis := sok & (sxs >= X_MIN)).any():
                    vals[col].append(sy[svis])
        if share:
            shared.append((ax, exs))
        ax.set(xscale='log', xlabel='tokens', title=model)
        ax.legend(handles=lines, fontsize=6)
    for ax in axes.flat[len(US):]:
        ax.axis('off')

    if x_vals:                                                   # left edge pinned at X_MIN
        _, xhi = _span(x_vals, log=True)
        for ax, _ in shared:
            ax.set_xlim(X_MIN, xhi)
    if vals['blue']:
        ylim = _span(vals['blue'])
        for ax, _ in shared:
            ax.set_ylim(*ylim)
    for j, col in enumerate(('red', 'green')):
        if vals[col] and j < len(rights[0]):
            ylim = _span(vals[col])
            for _, exs in shared:
                exs[j].set_ylim(*ylim)
    for r in (0, 1):
        axes[r, 0].set_ylabel(f'alphaReQ [{k0}–{k1}] of the unembedding', color='tab:blue')
    for j in (3, len(rights) - 1):                               # labels on the right edge
        rights[j][0].set_ylabel(f'alphaReQ [{k0}–{k1}] of the stream', color='tab:red')
        if rankme:
            rights[j][1].set_ylabel('RankMe of the stream', color='tab:green')
    fig.suptitle(f'alphaReQ (ranks {k0}–{k1}) — unembedding (left axis) vs the model\'s own '
                 f'after_final_norm stream (right axis)'
                 + (', with the stream RankMe (outer right axis)' if rankme else ''))
    plt.tight_layout(); plt.show()

# (metric name, head scorer on a singular-value tensor, stream yvar full / top-1 dropped)
OVERLAY_METRICS = {
    'RankMe': (lambda sv: _rm(sv), 'rankme', 'tail_rankme'),
    'matrix entropy': (lambda sv: float(np.log(_rm(sv))), 'matrix_entropy',
                       'tail_matrix_entropy'),
}

def rankme_overlay(metric='RankMe', top=1):
    """Per model: `metric` of the unembedding spectrum (blue, left) vs the same statistic on
    the after_final_norm stream covariance (red, twin right), each solid = full spectrum,
    dashed = with the top eigendirection dropped. x is shared as in alpha_overlay (nanochat
    excepted); y is NOT shared — every panel autoscales both of its axes."""
    head_fn, full_var, tail_var = OVERLAY_METRICS[metric]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    twins, shared, x_vals = [], [], []
    for ax, (model, per_step) in zip(axes.flat, US.items()):
        steps = sorted(per_step)
        xs = np.asarray(_lib.get_xs_tokens(model, steps), float)
        share = 'nanochat' not in model                          # nanochat: own x and y ranges
        ex = ax.twinx()
        twins.append(ex)
        lines = []
        for drop1, ls, lbl in ((False, '-', 'full spectrum'),
                               (True, '--', f'top-{top} dropped')):
            hv = np.asarray([head_fn(per_step[s][1:] if drop1 else per_step[s]) for s in steps])
            ok = (xs > 0) & np.isfinite(hv)
            lines.append(ax.plot(xs[ok], hv[ok], ls, lw=1.8, color='tab:blue',
                                 label=f'unembedding, {lbl}')[0])
            sv, ssteps = _lib.get_ys(cfg_of(model), model, AFN_HOOK,
                                     tail_var if drop1 else full_var,
                                     {'k': top} if drop1 else {})
            if share:
                x_vals.append(xs[ok])
            if sv is None:
                continue
            sxs = np.asarray(_lib.get_xs_tokens(model, ssteps), float)
            sv = np.asarray(sv, float)
            sok = (sxs > 0) & np.isfinite(sv)
            lines.append(ex.plot(sxs[sok], sv[sok], ls, lw=1.8, color='tab:red',
                                 label=f'stream, {lbl}')[0])
            if share:
                x_vals.append(sxs[sok])
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ex.tick_params(axis='y', labelcolor='tab:red')
        if share:
            shared.append((ax, ex))
        ax.set(xscale='log', xlabel='tokens', title=model)
        ax.legend(handles=lines, fontsize=6)
    for ax in axes.flat[len(US):]:
        ax.axis('off')
    if x_vals:                                                   # left edge pinned at X_MIN
        _, xhi = _span(x_vals, log=True)
        for ax, _ in shared:
            ax.set_xlim(X_MIN, xhi)
    for r in (0, 1):
        axes[r, 0].set_ylabel(f'{metric} of the unembedding', color='tab:blue')
    for j in (3, len(twins) - 1):
        twins[j].set_ylabel(f'{metric} of the stream', color='tab:red')
    fig.suptitle(f'{metric} — unembedding (left axis) vs the after_final_norm stream '
                 f'(right axis); solid = full spectrum, dashed = top eigendirection dropped; '
                 f'y autoscaled per panel')
    plt.tight_layout(); plt.show()

def overlay_caption(k0, k1, rankme=False):
    return (f'Blue (left axis): alphaReQ of the standard unembedding singular-value-squared '
            f'spectrum over ranks {k0}–{k1}, as in §3. Red (right axis): the same windowed '
            f'alphaReQ on the model\'s own centered after_final_norm stream covariance '
            f'(packed samples run). '
            + ('Green (outer right axis): RankMe of that same stream covariance — the '
               'headline curve, window-free. ' if rankme else '')
            + f'The series keep separate scales, but each scale is shared across panels — '
            f'blue comparable to blue, red to red{", green to green" if rankme else ""} — as '
            f'is the token axis, which starts at 10^8.5 tokens; nanochat-d12 is excluded from '
            f'all of them and autoscales on its own (much shorter run). Every panel draws its '
            f'own labelled axes. Checkpoint grids differ between the unembedding and stream '
            f'series. Descriptive only.')

# %%
alpha_overlay(*WINDOWS[0])                                       # the standard bulk window, 11–100
caption(overlay_caption(*WINDOWS[0]))
#FIGURE D1

# %%
alpha_overlay(11, 256)                                           # deeper band, both series
caption(overlay_caption(11, 256))

# %%
alpha_overlay(*WINDOWS[0], rankme=True)                          # same, with the stream RankMe
caption(overlay_caption(*WINDOWS[0], rankme=True))


# %%
rankme_overlay()
caption('Blue (left axis): RankMe of the unembedding singular-value-squared spectrum. Red '
        '(right axis): RankMe of the centered after_final_norm stream covariance (packed '
        'samples run). Solid = full spectrum; dashed = the top eigendirection dropped '
        '(sv[1:] for the head, tail_rankme k=1 for the stream) — the control for the two '
        'objects\' rank-1 spikes, the head\'s unigram direction and the stream\'s dominant '
        'direction. Every panel autoscales its own y axes (levels differ by model, so only '
        'the shapes are comparable); the token axis is shared from 10^8.5, nanochat-d12 '
        'excepted. Descriptive only.')

# %%
rankme_overlay('matrix entropy', top=10)
caption('The same figure in matrix entropy (Shannon entropy of the normalized eigenvalue '
        'spectrum) instead of RankMe: blue (left axis) the unembedding sv² spectrum, red '
        '(right axis) the centered after_final_norm stream covariance, solid = full spectrum, '
        'dashed = top eigendirection dropped. Since RankMe = exp(matrix entropy), this is a '
        'log rescaling of the panel above — the curves are monotonically equivalent and only '
        'the relative sizes of the swings change (multiplicative in RankMe, additive here). '
        'Descriptive only.')
