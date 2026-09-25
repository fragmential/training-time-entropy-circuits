"""
Toy double descent (Henighan et al. 2023) + RankMe + neural-collapse measures.

Setup follows the paper: n=10,000 features, S=0.999, x_i ~ U[0,1] when active,
columns rescaled to ||x||=1, ReLU-output model h = Wx / x' = ReLU(W^T h + b),
Xavier init, AdamW wd=1e-2, 50,000 FULL-BATCH updates, 2,500-step linear warmup
to lr=1e-3 then cosine decay to zero.

Outputs are numpy/JSON only -- no torch pickles -- so they can be re-opened and
re-plotted anywhere:
    results.json   sweep metrics, human-readable, small enough to paste inline
    results.npz    same metrics + scatter coordinates, numpy-loadable

Run:
    python toy_dd_sweep.py --out results
    python toy_dd_sweep.py --T-grid 6 20000 --steps 200   # quick timing probe
"""

import argparse, json, math, time
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# model                                                                        #
# --------------------------------------------------------------------------- #

def make_data(n, T, S, device, gen):
    x = torch.rand(n, T, generator=gen, device=device)
    x = x * (torch.rand(n, T, generator=gen, device=device) >= S)
    return x / x.norm(dim=0, keepdim=True).clamp_min(1e-12)


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * step / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train_one(n, m, T, S, steps, seed, device, chunk, warmup, peak_lr, wd):
    gen = torch.Generator(device=device).manual_seed(seed)
    X = make_data(n, T, S, device, gen)

    torch.manual_seed(seed)
    W = torch.empty(m, n, device=device)
    torch.nn.init.xavier_normal_(W)
    W.requires_grad_(True)
    b = torch.zeros(n, 1, device=device, requires_grad=True)

    opt = torch.optim.AdamW([W, b], lr=peak_lr, weight_decay=wd)
    cs = chunk or T
    train = float("nan")
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak_lr, warmup)
        opt.zero_grad(set_to_none=True)
        train = 0.0
        for s in range(0, T, cs):
            xb = X[:, s:s + cs]
            loss = ((torch.relu(W.T @ (W @ xb) + b) - xb) ** 2).sum(0).sum() / T
            loss.backward()
            train += loss.item()
        opt.step()
    return W.detach(), b.detach(), X, train


# --------------------------------------------------------------------------- #
# metrics                                                                      #
# --------------------------------------------------------------------------- #

def rankme(M, eps=1e-7):
    sv = torch.linalg.svdvals(M.float())
    p = sv / sv.sum().clamp_min(1e-12) + eps
    p = p / p.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def cov(v):
    """Coefficient of variation, sigma/mu (Wu & Papyan sec. 3.3). Signed, as specified."""
    v = v.float()
    return float(v.std(unbiased=False) / v.mean())


def collapse_stats(mu, var, Wc, tag, k=2):
    """
    Wu & Papyan (2024), eqs. 3-8. All inputs UNCENTRED; centring happens here.
      mu  (C, m)  class means
      var (C,)    within-class variance  sigma_c^2 = (1/N_c) sum_i ||h_i - mu_c||^2
      Wc  (C, m)  classifier per class, or None to skip (U)NC3
    """
    o, C = {}, mu.shape[0]
    if C < 2:
        return o
    off = ~torch.eye(C, dtype=torch.bool, device=mu.device)

    # global mean is the UNWEIGHTED average of class means (eq. 2), not the
    # sample mean -- these differ whenever classes are imbalanced
    M = mu - mu.mean(0)
    nrm = M.norm(dim=1).clamp_min(1e-12)
    U = M / nrm.unsqueeze(1)
    G = U @ U.T

    # NC1 -- class-distance normalised variance, eq. 3, with their extra
    # ||mu_c - mu_c'||^k factor (k=2), so the denominator is 2*||.||^(k+2)
    D = torch.cdist(mu, mu).clamp_min(1e-12)
    cdnv = (var.unsqueeze(0) + var.unsqueeze(1)) / (2 * D.pow(k + 2))
    o[f"{tag}_NC1_cdnv"] = float(cdnv[off].mean())
    o[f"{tag}_NC1_cdnv_cov"] = cov(cdnv[off])

    # (G)NC2 equinormness, eq. 5 -- CoV of the LOG norms, not of the norms.
    # CoV blows up when mean log-norm ~ 0 (i.e. mean norm ~ 1), which is exactly
    # where this toy model lives, so carry the std as a stable companion.
    ln = nrm.log()
    o[f"{tag}_NC2_lognorm_cov"] = cov(ln)
    o[f"{tag}_NC2_lognorm_std"] = float(ln.std(unbiased=False))

    # NC2 interference / equiangularity, eq. 6 -- target -1/(C-1).
    # Signed CoV flips sign with the mean and diverges as mean -> 0; same reason.
    inter = G[off]
    o[f"{tag}_NC2_interference"] = float(inter.mean())
    o[f"{tag}_NC2_interference_cov"] = cov(inter)
    o[f"{tag}_NC2_interference_std"] = float(inter.std(unbiased=False))
    o[f"{tag}_NC2_etf_target"] = -1.0 / (C - 1)

    # GNC2 hyperspherical uniformity, eq. 7 -- log inverse-distance kernel
    d = (2 - 2 * G).clamp_min(1e-12).sqrt()
    o[f"{tag}_GNC2_logkern"] = float(-d[off].log().mean())

    # (U)NC3 self-duality, eq. 8, and the CoV of those similarities
    if Wc is not None:
        Wn = Wc / Wc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim = (Wn * U).sum(1)
        o[f"{tag}_NC3_align"] = float(sim.mean())
        o[f"{tag}_UNC3_cov"] = cov(sim)
    return o


@torch.no_grad()
def analyse(W, b, X, n, m, T, S, device, max_scatter, min_count):
    out, H = {}, W @ X
    hn, wn = H.norm(dim=0), W.norm(dim=0)
    learned = (wn > 0.5 * wn.max()).nonzero().flatten()

    out["n_learned"] = int(learned.numel())
    out["rankme_H"] = rankme(H)
    out["rankme_W"] = rankme(W)
    out["W_fro"] = float(W.norm())
    out["b_l2"] = float(b.norm())
    out["h_norm_median"] = float(hn.median())

    # paper's dimensionality metrics
    Hu = H / hn.clamp_min(1e-12)
    out["D_X_median"] = float(((hn ** 2) / ((Hu.T @ H) ** 2).sum(1).clamp_min(1e-12)).median())
    out["D_X_expected"] = m / T
    Wu = W / wn.clamp_min(1e-12)
    Df = (wn ** 2) / ((Wu.T @ W) ** 2).sum(1).clamp_min(1e-12)
    out["D_f_max"] = float(Df.max())
    out["D_f_top5_mean"] = float(Df.topk(min(5, n)).values.mean())

    if learned.numel():
        Hl = W[:, learned] @ X[learned]
        out["var_share_learned"] = float(Hl.pow(2).sum() / H.pow(2).sum())
    if m == 2:
        a = torch.atan2(H[1], H[0]).sort().values
        gaps = torch.cat([a[1:] - a[:-1], (a[0] + 2 * math.pi - a[-1]).view(1)])
        out["gap_mean"], out["gap_std"] = float(gaps.mean()), float(gaps.std())
        out["gap_ideal"] = 2 * math.pi / T
        if learned.numel() > 1:
            fa = torch.atan2(W[1, learned], W[0, learned]).sort().values
            fg = torch.cat([fa[1:] - fa[:-1], (fa[0] + 2 * math.pi - fa[-1]).view(1)])
            out["feature_gaps_deg"] = (fg * 180 / math.pi).tolist()

    # ---- neural collapse, view A: FEATURES are the classes -----------------
    active = X > 0
    counts = active.sum(1)
    keep = (counts >= min_count).nonzero().flatten()
    if keep.numel() >= 2:
        keep = keep[Df[keep].topk(min(64, keep.numel())).indices]   # cap cost
        cnt = active[keep].float().sum(1)
        mu = (H @ active[keep].float().T).T / cnt.unsqueeze(1)
        var = torch.stack([((H[:, active[k]] - mu[i].unsqueeze(1)) ** 2).sum()
                           for i, k in enumerate(keep)]) / cnt.clamp_min(1)
        out.update(collapse_stats(mu, var, W[:, keep].T, "feat"))
        out["feat_n_classes"] = int(keep.numel())

        # NC4 -- MAP argmax (eq. 1) vs nearest-class-centre (eq. 9), on held-out
        gen4 = torch.Generator(device=device).manual_seed(1234)
        Xv = make_data(n, min(2000, 4 * T), S, device, gen4)
        Hv = W @ Xv
        mapc = (W[:, keep].T @ Hv + b[keep]).argmax(0)
        ncc = torch.cdist(Hv.T, mu).argmin(1)
        out["feat_NC4_agree"] = float((mapc == ncc).float().mean())

    # ---- neural collapse, view B: DATAPOINTS are the classes ---------------
    # one sample per class, so sigma_c^2 = 0 and NC1 vanishes by construction;
    # NC2/GNC2 are then measuring the T-gon. No per-datapoint classifier exists,
    # so (U)NC3 is skipped rather than reported as a trivial 1.
    if T <= 4096:
        out.update(collapse_stats(H.T, torch.zeros(T, device=device), None, "point"))
        out["point_n_classes"] = int(T)

    gen = torch.Generator(device=device).manual_seed(999)
    Xt = make_data(n, 2000, S, device, gen)
    out["test"] = float(((torch.relu(W.T @ (W @ Xt) + b) - Xt) ** 2).sum(0).mean())

    idx = torch.randperm(T, device=device)[:max_scatter]
    return out, W.cpu().numpy(), H[:, idx].cpu().numpy(), Df.cpu().numpy()


# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--S", type=float, default=0.999)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--warmup", type=int, default=2500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--seeds", type=int, default=1, help="keep lowest train loss")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--T-grid", type=int, nargs="+",
                   default=[6, 20, 60, 200, 600, 2000, 6000, 20000, 30000])
    p.add_argument("--scatter-at", type=int, nargs="+", default=[6, 30000],
                   help="T values whose W and H get saved for the scatter panels")
    p.add_argument("--max-scatter", type=int, default=4000)
    p.add_argument("--min-count", type=int, default=8, help="min samples for a feature-class")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="results")
    a = p.parse_args()

    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    dev = torch.device(a.device)
    print(f"device={dev} n={a.n} m={a.m} S={a.S} steps={a.steps} seeds={a.seeds}", flush=True)

    rows, arrays = [], {}
    for T in a.T_grid:
        t0, best = time.time(), None
        for seed in range(a.seeds):
            W, b, X, tr = train_one(a.n, a.m, T, a.S, a.steps, seed, dev,
                                    a.chunk, a.warmup, a.lr, a.wd)
            if best is None or tr < best[0]:
                best = (tr, W, b, X)
        tr, W, b, X = best
        st, Wnp, Hnp, Dfnp = analyse(W, b, X, a.n, a.m, T, a.S, dev,
                                     a.max_scatter, a.min_count)
        st["T"], st["train"], st["secs"] = T, tr, round(time.time() - t0, 1)
        rows.append(st)
        if T in a.scatter_at:
            arrays[f"W_T{T}"], arrays[f"H_T{T}"], arrays[f"Df_T{T}"] = Wnp, Hnp, Dfnp
        print(f"T={T:<6d} {st['secs']:6.1f}s  train={tr:.5f} test={st['test']:.4f} "
              f"learned={st['n_learned']:3d} rkH={st['rankme_H']:.3f} "
              f"rkW={st['rankme_W']:.3f} D_X={st['D_X_median']:.2e}", flush=True)

    meta = {"config": vars(a), "rows": rows}
    with open(f"{a.out}.json", "w") as f:
        json.dump(meta, f, indent=1)
    np.savez_compressed(f"{a.out}.npz", meta=json.dumps(meta), **arrays)
    print(f"\nwrote {a.out}.json and {a.out}.npz")


if __name__ == "__main__":
    main()