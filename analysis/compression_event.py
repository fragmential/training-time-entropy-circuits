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
# # A genuinely large compression event in the single-layer toy
#
# **Problem.** The canonical toy (`single`: d=2, 4 classes) compresses by 0.0065 RankMe
# units, transiently, and ends almost isotropic — so end-state alignment is near-trivially
# satisfied. The dip cannot be big there **by construction**: CE must separate 4 classes
# in 2 dims, so the final solution needs every direction the transient ever explores;
# RankMe's floor sits just under its ceiling.
#
# **The levers.** A large compression needs the transient to occupy MORE directions than
# the final solution keeps:
#
# 1. **Headroom** — ambient d ≫ C−1. The CE end state is the class simplex: exactly C−1
#    surviving directions (the final spectrum below is [1,1,1,0,...,0] up to skew).
# 2. **A low-rank start** so an expansion phase exists at all (iid init starts at
#    RankMe ≈ min(N, d) and can only decline).
# 3. **Expansion overshoot for free**: the theta–W coupling is bilinear, so early CE
#    dynamics grow the features through the RANDOM init head frame (plus amplified
#    jitter) — more directions than the C−1 the solution keeps. The peak grows with
#    headroom: ~3.9 / 5.9 / 12.0 at d = 8 / 16 / 32 (C=4). Once logits saturate, only the
#    simplex is defended → the strays decay (wd) or freeze and dilute (no wd).
# 4. **Weight decay (optional)** — which Pythia/OLMo have and the toy lacked — makes the
#    collapse exact and permanent, but the dip does NOT need it (drawdown 2.76 without wd,
#    stable through 10k steps — no recovery to isotropy, unlike the canonical toy's
#    rebound to 1.9993).
#
# Two runs below, sharing one train loop:
# **(A) expansion → compression** (flat init, C=4 skewed, d=16): drawdown ~3 RankMe units,
# ~500× the canonical dip. **(B) warmup drop → plateau → late expansion** (the bimodal
# construction: a few super-common classes + coincident n=1 singletons, NO in-between
# rarities — in-between frequencies smooth the phases out). Why not all three phases in
# ONE run — see the last section: in a single layer the warmup contraction and the
# exploration overshoot share a clock, and that is itself informative about what the LLM
# phases require.

# %%
import os, sys, itertools
if os.path.basename(os.getcwd()) == "analysis":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm

PHI = (1 + 5 ** 0.5) / 2  # golden ratio, for the icosa/dodeca/4d vertex coordinates

def _rot(X, d, seed):
    """Unit-normalise rows and apply a random rotation of R^d (seed-controlled)."""
    g = torch.Generator().manual_seed(seed)
    R = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    return F.normalize(X, dim=1) @ R

def _cyc3(a, b):
    """(0,±a,±b) and its two cyclic rotations, all sign combos — 12 rows in R^3."""
    return [p for va in (-a, a) for vb in (-b, b)
            for p in ([0., va, vb], [vb, 0., va], [va, vb, 0.])]

def _min_energy(k, d, seed, iters=1500, restarts=6):
    """Thomson / Riesz(s=1) energy minimiser: k points maximally spread on S^{d-1}.
    Multi-restart projected Adam; returns the lowest-energy configuration found."""
    best, best_e, off = None, float("inf"), ~torch.eye(k, dtype=bool)
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 1000 + r)
        X = F.normalize(torch.randn(k, d, generator=g), dim=1).clone().requires_grad_()
        opt = torch.optim.Adam([X], lr=0.05)
        for _ in range(iters):
            opt.zero_grad()
            Xn = F.normalize(X, dim=1)
            D = torch.cdist(Xn, Xn)[off].clamp(min=1e-6)
            (1.0 / D).sum().backward()
            opt.step()
        with torch.no_grad():
            Xn = F.normalize(X, dim=1)
            e = (1.0 / torch.cdist(Xn, Xn)[off].clamp(min=1e-6)).sum().item()
            if e < best_e:
                best_e, best = e, Xn.clone()
    return best

def even_directions(k, d, seed=0):
    """k unit vectors spread as evenly as possible on S^{d-1}. Exact convex regular
    polytopes wherever k matches one in dimension d — regular simplex (k<=d+1, equidistant),
    regular k-gon (d=2), cross-polytope (k=2d), hypercube (k=2**d), icosahedron (12,d=3),
    dodecahedron (20,d=3), 24-cell (24,d=4) — otherwise a Thomson/Riesz energy minimiser
    (which itself recovers the energy-minimising polytopes, e.g. octahedron, icosahedron).
    Randomly rotated by `seed`. The 4d 120-/600-cells are not special-cased (fall through
    to the numeric minimiser)."""
    if k <= d + 1:                                   # regular simplex (optimal, equidistant)
        M = torch.eye(k) - torch.ones(k, k) / k
        c = F.normalize(torch.linalg.eigh(M)[1][:, -(k - 1):], dim=1)
        g = torch.Generator().manual_seed(seed)
        Q = torch.linalg.qr(torch.randn(d, k - 1, generator=g))[0]
        return c @ Q.T                               # isometric embed + rotate
    if d == 2:                                       # regular k-gon
        a = 2 * torch.pi * torch.arange(k, dtype=torch.float) / k
        X = torch.stack([torch.cos(a), torch.sin(a)], 1)
    elif k == 2 * d:                                 # cross-polytope (orthoplex)
        X = torch.cat([torch.eye(d), -torch.eye(d)])
    elif k == 2 ** d:                                # hypercube
        X = torch.tensor(list(itertools.product((-1., 1.), repeat=d)))
    elif (k, d) == (12, 3):                          # icosahedron
        X = torch.tensor(_cyc3(1.0, PHI))
    elif (k, d) == (20, 3):                          # dodecahedron: cube (8) + icosa-dual (12)
        X = torch.tensor(list(itertools.product((-1., 1.), repeat=3)) + _cyc3(1 / PHI, PHI))
    elif (k, d) == (24, 4):                          # 24-cell: perms of (±1,±1,0,0)
        X = torch.tensor([[si if p == i else sj if p == j else 0. for p in range(4)]
                          for i, j in itertools.combinations(range(4), 2)
                          for si in (-1., 1.) for sj in (-1., 1.)])
    else:                                            # no matching regular polytope
        return _rot(_min_energy(k, d, seed), d, seed)
    return _rot(X, d, seed)

def run(counts, d, steps=3000, lr=0.25, wd=0.0, init="flat", init_std=0.1,
        flat_jitter=0.02, delta=1e-3, seed=0, com_threshold=2):
    torch.manual_seed(seed)
    N, C = sum(counts), len(counts)
    y = torch.repeat_interleave(torch.arange(C), torch.tensor(counts))
    if init == "flat":
        u = F.normalize(torch.randn(d), dim=0)
        theta = init_std * torch.randn(N, 1) * u + flat_jitter * torch.randn(N, d)
        W = init_std * torch.randn(d, C)
    elif init == "iid":
        theta = init_std * torch.randn(N, d)
        W = init_std * torch.randn(d, C)
    elif init in ("clustered", "simplex", "even"):
        n_com = sum(n >= com_threshold for n in counts)
        if init == "clustered":
            # orthogonal cluster centers (one axis per common): needs d >= n_com
            dirs = torch.linalg.qr(torch.randn(d, n_com))[0].T
        elif init == "simplex":
            # equidistant (simplex-ETF) cluster centers: needs only d >= n_com - 1
            M = torch.eye(n_com) - torch.ones(n_com, n_com) / n_com
            coords = F.normalize(torch.linalg.eigh(M)[1][:, -(n_com - 1):], dim=1)  # (n_com, n_com-1) unit ETF
            Q = torch.linalg.qr(torch.randn(d, n_com - 1))[0]         # (d, n_com-1) orthonormal cols
            dirs = coords @ Q.T                                       # (n_com, d) isometric embed, angles preserved
        else:
            # maximally-even spherical code (regular polytope where one exists)
            dirs = even_directions(n_com, d, seed)
        rows = [0.6 * dirs[c] + 0.05 * torch.randn(d)
                for c in range(n_com) for _ in range(counts[c])]
        rows += [torch.zeros(d)] * (N - len(rows))
        theta = torch.stack(rows) + delta * torch.randn(N, d)
        W = torch.zeros(d, C)
        W[:, :n_com] = 0.74 * dirs.T
        W = W + delta * torch.randn(d, C)
    theta.requires_grad_()
    W.requires_grad_()
    rm, lams, loss_c, loss_tot = [], [], [], []
    for _ in range(steps + 1):
        li = F.cross_entropy(theta @ W, y, reduction="none")
        loss = li.mean()
        with torch.no_grad():
            lam = torch.linalg.svdvals(theta).square()
            p = lam / lam.sum()
            rm.append(float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())))
            lams.append(lam)
            loss_c.append([float(li[y == c].mean()) for c in range(C)])
            loss_tot.append(float(loss))
        g_t, g_w = torch.autograd.grad(loss, (theta, W))
        with torch.no_grad():
            theta -= lr * (g_t + wd * theta)
            W -= lr * (g_w + wd * W)
    return dict(rm=np.array(rm), lams=torch.stack(lams).numpy(), loss_c=np.array(loss_c),
                loss=np.array(loss_tot), theta=theta.detach().numpy(), y=y.numpy())

def loss_panel(ax, r, counts, cols=None, log=True):
    C = len(counts)
    cols = cols or [cm.viridis(1 - i / max(C - 1, 1)) for i in range(C)]
    for c in range(C):
        ax.plot(r['loss_c'][:, c], color=cols[c], lw=0.9,
                label=f'class {c} (n={counts[c]})' if C <= 6 else None)
    ax.plot(r['loss'], color='k', lw=1.8, label='total')
    ax.set(xlabel='step', ylabel='CE loss', xscale='log', yscale='log' if log else 'linear')
    ax.legend(fontsize=7)
    return cols

# %% [markdown]
# ## Run A — expansion → compression (the big dip)
#
# Flat init, counts (32,16,8,8), d=16, wd=0.01. RankMe rises 1 → ~5.2, then compresses to
# 2.8 (the skew prints INSIDE the simplex: final < C−1=3, so the end state is anisotropic
# even in its surviving subspace). Right: per-class + total loss — the compression onset
# sits where the classes' losses saturate: once every class is fit, nothing defends the
# extra directions.

# %%
COUNTS_A, D_A = (32, 16, 8, 8), 16
ra = run(COUNTS_A, D_A, wd=0.01)
peak_a = int(ra['rm'].argmax())
print(f"A: peak {ra['rm'].max():.2f} @ {peak_a}, final {ra['rm'][-1]:.2f}, "
      f"drawdown {ra['rm'].max() - ra['rm'][peak_a:].min():.2f}  (canonical toy: 0.0065)")

COLS4 = ('magenta', 'orange', 'royalblue', 'seagreen')
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
a0.plot(ra['rm'], color='crimson', lw=1.5)
a0.axhline(len(COUNTS_A) - 1, color='gray', lw=0.8, ls='--')
a0.text(1.2, len(COUNTS_A) - 0.8, 'C−1 (simplex floor)', color='gray', fontsize=8)
a0.axvline(peak_a, color='gray', lw=0.8, ls=':')
a0.set(title=f'RankMe: drawdown {ra["rm"].max() - ra["rm"][peak_a:].min():.2f}',
       xlabel='step', ylabel='RankMe', xscale='log', ylim=(0, D_A))
loss_panel(a1, ra, COUNTS_A, cols=COLS4)
a1.axvline(peak_a, color='gray', lw=0.8, ls=':')
a1.set_title('per-class + total loss')
plt.show()

# %% [markdown]
# ### Mechanism: the full spectrum
#
# Expansion = early growth through the random head frame + amplified jitter (many
# eigenvalues rise in parallel). Compression = after saturation only the simplex is
# defended: three survivors (crimson) keep growing, everything else decays under wd.

# %%
fig, ax = plt.subplots(figsize=(9, 4.2))
for k in range(D_A):
    ax.plot(np.maximum(ra['lams'][:, k], 1e-12), lw=1.2 if k < 3 else 0.7,
            color='crimson' if k < 3 else 'gray', alpha=1.0 if k < 3 else 0.55)
ax.axvline(peak_a, color='gray', lw=0.8, ls=':')
ax.set(xlabel='step', ylabel=r'$\lambda_k$', yscale='log', xscale='log', ylim=(1e-7, None),
       title='run A eigenvalues: 3 survivors vs the given-back directions')
plt.show()

# %%
lam_f = ra['lams'][-1] / ra['lams'][-1].max()
V = np.linalg.eigh(ra['theta'].T @ ra['theta'])[1][:, ::-1]
proj = ra['theta'] @ V[:, :2]

fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 4))
a0.bar(range(D_A), lam_f, color=['crimson'] * 3 + ['gray'] * (D_A - 3))
a0.set(xlabel='eigenvalue index', ylabel=r'$\lambda_k/\lambda_1$',
       title='final spectrum: 3 of 16 directions, skew visible — NOT isotropic')
for c in range(len(COUNTS_A)):
    m = ra['y'] == c
    a1.scatter(proj[m, 0], proj[m, 1], s=18, color=COLS4[c], label=f'class {c} (n={COUNTS_A[c]})')
a1.set(xlabel='survivor 1', ylabel='survivor 2', aspect='equal',
       title='features on the top-2 survivors (neural-collapse geometry)')
a1.legend(fontsize=8)
plt.show()

# %% [markdown]
# ## Run B — warmup drop → plateau → late expansion (the bimodal construction)
#
# A BIMODAL frequency split: 4 super-common classes (n=32) clustered on random orthogonal
# directions with aligned head columns, plus 12 super-rare singletons (n=1) coincident at
# the origin with zero head columns, δ jitter only — deliberately no in-between rarities,
# which smooth the phase boundaries out. d=24, no wd (wd ≥ 0.01 kills the singleton forks
# outright: their n=1 CE pull never beats the decay — measured). The phases are sharp:
#
# - **drop** (6.8 → 4.75 by ~1500): the commons' coherent growth swamps the within-cluster
#   init jitter — the warmup, and it fully completes;
# - **plateau**: the singletons are still riding their shared path (exact swap-symmetric
#   blob at the origin, split time δ-controlled, ~e^{rt} against δ);
# - **late expansion** (→ ~9.3): the singletons fork into their own random directions, one
#   per class — visible on the right as the singleton losses (thin light curves) breaking
#   away and saturating late, long after the commons (dark) are done.
#
# This is the LLM ordering — unigram fix first, rare structure much later — produced by
# frequency alone, no clock tricks. What it does NOT produce is a third phase: with free
# dims available each singleton keeps its direction (equilibrium approached from below),
# and with no free dims (d = #commons) the spectrum is pinned at the ceiling and only the
# canonical ~0.01-scale kick remains (measured, next section).

# %%
COUNTS_B, D_B = (20,) * 4 + (1,) * 80, 4
rb = run(COUNTS_B, D_B, steps=40000, lr=0.5, init="clustered", com_threshold=2, flat_jitter=0.03, delta=0.005)
tr = int(rb['rm'][:4000].argmin())
print(f"B: start {rb['rm'][0]:.2f} -> trough {rb['rm'][tr]:.2f} @ {tr} -> final {rb['rm'][-1]:.2f}")

# %%
fig, (b0, b1) = plt.subplots(1, 2, figsize=(11, 4.2))
b0.plot(rb['rm'], color='steelblue', lw=1.5)
b0.axvline(tr, color='gray', lw=0.8, ls=':')
b0.set(title='RankMe: drop (warmup) → plateau → late singleton forks', xlabel='step',
       ylabel='RankMe', xscale='log')
loss_panel(b1, rb, COUNTS_B, log=False)
b1.axvline(tr, color='gray', lw=0.8, ls=':')
b1.set_title('per-class loss (dark = common, light = singleton) + total')
plt.show()

# %%
COUNTS_B, D_B = (2,) * 2 + (1,) * 2, 2
rb = run(COUNTS_B, D_B, steps=4000, lr=0.5, init="clustered", com_threshold=3, flat_jitter=0.03, delta=0.005)
tr = int(rb['rm'][:4000].argmin())
print(f"B: start {rb['rm'][0]:.2f} -> trough {rb['rm'][tr]:.2f} @ {tr} -> final {rb['rm'][-1]:.2f}")

# %%
fig, (b0, b1) = plt.subplots(1, 2, figsize=(11, 4.2))
b0.plot(rb['rm'], color='steelblue', lw=1.5)
b0.axvline(tr, color='gray', lw=0.8, ls=':')
b0.set(title='RankMe: drop (warmup) → plateau → late singleton forks', xlabel='step',
       ylabel='RankMe', xscale='log')
loss_panel(b1, rb, COUNTS_B, log=False)
b1.axvline(tr, color='gray', lw=0.8, ls=':')
b1.set_title('per-class loss (dark = common, light = singleton) + total')
plt.show()

# %%
COUNTS_B, D_B = (3,) * 2 + (1,) * 2, 2
rb = run(COUNTS_B, D_B, steps=10000, lr=0.5, init="clustered", com_threshold=3, flat_jitter=0.03, delta=0.005)
tr = int(rb['rm'][:4000].argmin())
print(f"B: start {rb['rm'][0]:.2f} -> trough {rb['rm'][tr]:.2f} @ {tr} -> final {rb['rm'][-1]:.2f}")

# %%
fig, (b0, b1) = plt.subplots(1, 2, figsize=(11, 4.2))
b0.plot(rb['rm'], color='steelblue', lw=1.5)
b0.axvline(tr, color='gray', lw=0.8, ls=':')
b0.set(title='RankMe: drop (warmup) → plateau → late singleton forks', xlabel='step',
       ylabel='RankMe', xscale='log')
loss_panel(b1, rb, COUNTS_B, log=False)
b1.axvline(tr, color='gray', lw=0.8, ls=':')
b1.set_title('per-class loss (dark = common, light = singleton) + total')
plt.show()

# %% [markdown]
# ## Why not all three phases in one single-layer run: because claude is STUPID that's why. I did it easily
#
# Tried and failed, systematically (2–3 seeds each): borderline init scales between
# "drops first" and "rises first"; low-rank subspace inits (k = 3…8, d = 16/32); junk
# spread orthogonal to the head's span killed by fast wd while a tiny in-span seed
# amplifies late (wd 0.01–0.06 × jitter 1e-4…1e-6); Zipf classes at every wd; a
# shared-prior drift boosted by extreme skew; and, for a compression AFTER run B's late
# forks: partial bottlenecks (2 free dims), zero free dims (d = #commons), and wd 1e-3.
# Every time, the drop swallows the expansion overshoot, the expansion ends in a plateau,
# or the ceiling pins the spectrum.
#
# Two structural reasons:
#
# 1. **Shared clock.** The warmup contraction and the exploration overshoot are both paced
#    by logit saturation, so the overshoot peak always lands inside the drop whenever the
#    start rank is above it. Weight decay is an independent clock, but any wd fast enough
#    to clear the init spread first also strangles the overshoot (and wd ≥ 0.01 prevents
#    n=1 forks entirely).
# 2. **Late expansions end at their ceiling.** Fork-driven buildout (run B) approaches the
#    max-margin/wd equilibrium from below — nothing overshoots, so nothing is given back.
#    Forcing the forks to reuse the common axes (no free dims) reduces the compression to
#    the canonical kick, which is ~0.01-scale by the floor-vs-ceiling argument: at
#    d = #commons the measured trajectories sit at RankMe ≈ d ± 0.03 throughout.
#
# **Implication for the LLM phases:** the LLM sequence drop → expansion → compression
# requires the expansion to run on a SLOWER clock than the warmup fix. In the real model
# that separation is natural: the unigram/frequency fix is shallow (one effective layer,
# fast), while the expanding features are compositional circuits built over many layers
# (slow). The single-layer toy collapses those clocks into one — so reproducing all three
# phases at once should need the multi-layer toy (depth as the clock separator), not more
# init engineering. That is a testable next step.
#
# ## Knobs (measured, 3 seeds each)
#
# | knob | effect |
# |---|---|
# | ambient d (run A, C=4) | peak 3.9 / 5.9 / 12.0 at d = 8 / 16 / 32 → drawdown 0.9 / 2.9 / 9.0 |
# | weight decay | optional for A: 2.76 drawdown at wd=0 (stable at 10k steps); wd makes the collapse exact (final = C−1). Fatal for B: wd 0.01 kills the n=1 forks (monotone decline, no expansion); 1e-3 is safe |
# | iid init in A's setting | kills the expansion phase: starts at RankMe ≈ min(N, d), monotone decline |
# | more classes (C=8, d=16) | floor rises to C−1=7 → dip shrinks to 1.2: headroom d/C is the lever |
# | skew | survives everywhere and prints in the final spectrum (A ends at 2.79 < 3) |
#
# The canonical toy's dip is a timing effect (rare-pair fork briefly outpacing λ₂) with no
# rank to give back — tiny and transient. Run A's compression is *consolidation*:
# directions explored during expansion are abandoned because the solution doesn't need
# them. If the LLM decline is of this kind, post-decline spectra should stay anisotropic —
# which they do, and which also makes end-state alignment analyses meaningful again.
