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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fork → decline lag: what actually crosses what
#
# Two animations over one single-layer run (pick `RX` below), plus the identity they
# visualize.
#
# The spectrum's rate is bilinear in position × velocity: for the uncentered feature Gram
# G = Σᵢ fᵢfᵢᵀ, first-order perturbation gives, per eigendirection k,
#
# $$\dot\lambda_k = 2\sum_i \langle f_i, u_k\rangle\,\langle v_i, u_k\rangle$$
#
# — each sample feeds direction k in proportion to (its position component) × (its
# velocity component) along $u_k$. RankMe declines iff
# $R = \dot\lambda_1/\lambda_1 - \dot\lambda_2/\lambda_2 > 0$, and the decline ONSET is
# the zero crossing of R. Sample contributions to R sum exactly, so "who causes the
# decline" is a well-posed per-sample question — animation 2 colors it.

# %%
import os, sys
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

SW = torch.load('data/results/toy/appendix_sweeps.pt', weights_only=True)

RX = '-0.5'                                    # one of '0', '-0.1', '-0.25', '-0.5'
r = SW['rx'][RX]
F = r['theta_path'].numpy()                    # (T, 6, 2) feature rows
W = r['W_path'].numpy()                        # (T, 2, 4) classifier columns
T = len(F)
rm, cosw = r['rm'].numpy(), r['cos_w'].numpy()
COLS = ('magenta', 'orange', 'royalblue', 'seagreen')
Y = (0, 0, 1, 1, 2, 3)

lams = np.zeros((T, 2))
us = np.zeros((T, 2, 2))
for t in range(T):
    w_, V = np.linalg.eigh(F[t].T @ F[t])
    lams[t] = w_[::-1]
    us[t] = V[:, ::-1]
for t in range(1, T):                          # fix eigvec sign flips between frames
    for k in (0, 1):
        if us[t, :, k] @ us[t - 1, :, k] < 0:
            us[t, :, k] *= -1
Vel = np.gradient(F, axis=0)                   # per-step velocity of each sample

contrib = np.zeros((T, 6))                     # per-sample contribution to R
for t in range(T):
    for k, s in ((0, 1), (1, -1)):
        u = us[t, :, k]
        contrib[t] += s * 2 * (F[t] @ u) * (Vel[t] @ u) / lams[t, k]
R_tot = contrib.sum(1)
fork = int(np.argmax((cosw < 0.97) & (np.arange(T) > 10)))
peak = int(np.argmax(rm))
print(f'rx={RX}: fork@{fork}, RankMe peak (= R zero crossing / decline onset) @ {peak}')

# %% [markdown]
# ## Animation 1 — positions + the eigenframe
#
# Dotted lines: the two eigendirections of the feature Gram. Solid segments on them: the
# per-sample RMS extent along each direction, $\sqrt{\lambda_k / N}$ — direction 1 violet,
# direction 2 yellowgreen, matching the Fig-4-style RankMe/eigenvalue panel. Thin trails:
# the last 40 steps of each point.

# %%
STRIDE = 2
lim = 1.2 * np.abs(F).max()
wlim = 1.2 * np.abs(W).max()

fig1, (ax_w, ax_f) = plt.subplots(1, 2, figsize=(11, 5.4))

def _panel(ax, title, L):
    ax.clear()
    ax.set(xlim=(-L, L), ylim=(-L, L), aspect='equal', title=title,
           xlabel='dim 1', ylabel='dim 2')
    ax.axhline(0, color='0.9', lw=0.6)
    ax.axvline(0, color='0.9', lw=0.6)

def _eigaxes(ax, t, L):
    for k, c in ((0, 'violet'), (1, 'yellowgreen')):
        u = us[t, :, k]
        s = np.sqrt(lams[t, k] / 6)            # RMS per-sample extent along u_k
        ax.plot([-L * u[0], L * u[0]], [-L * u[1], L * u[1]], ':', color=c, lw=1)
        ax.plot([0, s * u[0]], [0, s * u[1]], '-', color=c, lw=3.5)

def _draw1(t):
    _panel(ax_w, '$W_i$', wlim)
    _panel(ax_f, r'$f_\theta(x)$', lim)
    t0 = max(0, t - 40)
    for i in range(4):
        ax_w.plot(W[t0:t + 1, 0, i], W[t0:t + 1, 1, i], '-', color=COLS[i], lw=0.7, alpha=0.5)
        ax_w.scatter(*W[t, :, i], s=70, color=COLS[i])
    for j, c in enumerate(Y):
        ax_f.plot(F[t0:t + 1, j, 0], F[t0:t + 1, j, 1], '-', color=COLS[c], lw=0.7, alpha=0.5)
        ax_f.scatter(*F[t, j], s=50, color=COLS[c])
    _eigaxes(ax_f, t, lim)
    fig1.suptitle(f'rx = {RX}   step {t}   RankMe {rm[t]:.4f}'
                  f'   {"(fork passed)" if t >= fork else ""}'
                  f'   {"(declining)" if t >= peak else ""}')

anim1 = animation.FuncAnimation(fig1, _draw1, frames=range(0, T, STRIDE), interval=60)
plt.close(fig1)
HTML(anim1.to_jshtml())

# %% [markdown]
# ## Animation 2 — velocities, colored by their effect on RankMe
#
# Arrows: each sample's per-step velocity (×25 for visibility). Color: that sample's
# instantaneous contribution to $R = \dot\lambda_1/\lambda_1 - \dot\lambda_2/\lambda_2$ —
# crimson = pushes RankMe DOWN (feeds the gap), steelblue = pushes RankMe up. The same
# velocity changes color as position accumulates: the boundary lives on the
# position×velocity product, not on velocity. Bottom strip: R over time (black), zero
# line, fork (dotted) and decline onset (dashed), with the frame cursor.

# %%
VSCALE = 25

fig2 = plt.figure(figsize=(10.5, 7.6))
gs = fig2.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.28)
axv = fig2.add_subplot(gs[0])
axr = fig2.add_subplot(gs[1])

def _draw2(t):
    _panel(axv, r'$f_\theta(x)$ + velocity (×%d)' % VSCALE, lim)
    _eigaxes(axv, t, lim)
    for j, c in enumerate(Y):
        axv.scatter(*F[t, j], s=45, color=COLS[c], zorder=3)
        col = 'crimson' if contrib[t, j] > 0 else 'steelblue'
        axv.annotate('', F[t, j] + VSCALE * Vel[t, j], F[t, j],
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.8))
    axr.clear()
    axr.plot(R_tot, color='k', lw=1.4)
    axr.axhline(0, color='gray', lw=0.8)
    axr.axvline(fork, color='gray', lw=0.9, ls=':')
    axr.axvline(peak, color='gray', lw=0.9, ls='--')
    axr.axvline(t, color='crimson', lw=1.2, alpha=0.7)
    axr.set(xlabel='step', ylabel=r'$R = \dot\lambda_1/\lambda_1 - \dot\lambda_2/\lambda_2$')
    fig2.suptitle(f'rx = {RX}   step {t}   R = {R_tot[t]:+.5f}'
                  f'   ({"declining" if R_tot[t] > 0 else "expanding"})')

anim2 = animation.FuncAnimation(fig2, _draw2, frames=range(0, T, STRIDE), interval=60)
plt.close(fig2)
HTML(anim2.to_jshtml())

# %% [markdown]
# ## The channel decomposition (static)
#
# The pair's Gram contribution splits exactly: $f_3f_3^T + f_4f_4^T = 2(mm^T + aa^T)$ with
# $m$ the common mode and $a$ the half-difference. R therefore splits into three exact
# channels: frequent rows, pair-common, pair-difference. The fork's ONLY lever is the
# difference channel, $\propto r\,a(t)^2$ — exponentially growing, and initially
# rank-INCREASING (a starts ⊥ the shared path, feeding λ₂) before it flips sign. Decline
# onset = where the sum of the three crosses zero.

# %%
m = (F[:, 4] + F[:, 5]) / 2
a = (F[:, 4] - F[:, 5]) / 2
vm = np.gradient(m, axis=0)
va = np.gradient(a, axis=0)

def _chan(pos, vel):
    out = np.zeros(T)
    for t in range(T):
        for k, s in ((0, 1), (1, -1)):
            u = us[t, :, k]
            out[t] += s * 4 * (pos[t] @ u) * (vel[t] @ u) / lams[t, k]
    return out

R_m, R_a = _chan(m, vm), _chan(a, va)
R_freq = R_tot - R_m - R_a

fig, ax = plt.subplots(figsize=(10, 4.2))
for arr, lbl, c in ((R_freq, 'frequent rows', 'tab:orange'),
                    (R_m, 'pair common mode', 'tab:blue'),
                    (R_a, 'pair difference (the fork)', 'tab:red'),
                    (R_tot, 'total R', 'k')):
    ax.plot(arr, color=c, lw=2 if lbl == 'total R' else 1.4, label=lbl)
ax.axhline(0, color='gray', lw=0.8)
ax.axvline(fork, color='gray', lw=0.9, ls=':')
ax.axvline(peak, color='gray', lw=0.9, ls='--')
ax.set(xlabel='step', ylabel='contribution to R',
       title=f'rx = {RX}: decline onset (dashed) = where the red channel drags the total across zero')
ax.legend(fontsize=8)
plt.show()
