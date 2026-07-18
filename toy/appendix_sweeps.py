"""Edge-case sweeps behind docs/toy_fig4_addendum.md, regenerable in ~2 min CPU:
delta (fork timing window), rx (rare-pair offset -> decline-onset lag), init homotopy
(masked vs printed kick), long horizon (transience). Single-layer dynamics only (depth 0),
run with a local dup-free GD loop identical to toy/train.py's; saves one dict to
data/results/toy/appendix_sweeps.pt for analysis/showcase_toy_appendix.py."""

import os

import torch
import torch.nn.functional as F

from toy.train import SKEW, Spec, _clustered_init

LR, STEPS = 0.25, 300
Y = torch.repeat_interleave(torch.arange(4), torch.tensor(SKEW))


def run(theta0: torch.Tensor, W0: torch.Tensor, steps: int = STEPS, lr: float = LR) -> dict:
    """Full-batch GD on CE for the single-layer toy; per-step spectral/fork trajectories
    plus the full weight/feature paths (for Fig-4-style trajectory panels)."""
    theta, W = theta0.clone().requires_grad_(), W0.clone().requires_grad_()
    out: dict = {"rm": [], "cos_w": [], "lam": []}
    paths: dict = {"theta_path": [], "W_path": []}
    for _ in range(steps + 1):
        loss = F.cross_entropy(theta @ W, Y)
        with torch.no_grad():
            paths["theta_path"].append(theta.detach().clone())
            paths["W_path"].append(W.detach().clone())
            lam = torch.linalg.svdvals(theta).square()
            p = lam / lam.sum()
            out["rm"].append(float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())))
            out["cos_w"].append(float(F.cosine_similarity(W[:, 2], W[:, 3], dim=0)))
            out["lam"].append(lam.tolist())
        g_theta, g_w = torch.autograd.grad(loss, (theta, W))
        with torch.no_grad():
            theta -= lr * g_theta
            W -= lr * g_w
    out["loss"] = float(loss)
    return ({k: torch.tensor(v) for k, v in out.items()}
            | {k: torch.stack(v) for k, v in paths.items()})


def _init(delta: float = 1e-3, rx: float = 0.0, iid_mix: float = 0.0, seed: int = 0):
    """Clustered init with a movable rare-pair offset, optionally blended toward iid-tiny."""
    torch.manual_seed(seed)
    theta0, W0 = _clustered_init(Spec(SKEW, d=2, lr=LR, steps=STEPS, clustered=True, delta=delta))
    theta0[4:] += torch.tensor([rx, 0.0])
    if iid_mix:
        torch.manual_seed(seed + 1000)
        theta0 = (1 - iid_mix) * theta0 + iid_mix * 0.01 * torch.randn_like(theta0)
        W0 = (1 - iid_mix) * W0 + iid_mix * 0.01 * torch.randn_like(W0)
    return theta0, W0


def mse_skew_check(steps: int = 1000, lr: float = 0.3, init: float = 0.3,
                   seed: int = 0) -> dict:
    """Is mse_skew's small RankMe dip a rare-class fork (the CE mechanism) or does MSE starve
    the rare classes (Li et al's supplementary claim)? Tracks per-class MSE, the rare weight
    columns, and cos(w2, w3) along the exact mse_skew spec's trajectory."""
    torch.manual_seed(seed)
    dup = 3                                              # the mse_skew spec's replication
    theta = (init * torch.randn(sum(SKEW), 2)).repeat_interleave(dup, 0).requires_grad_()
    W = (init * torch.randn(2, 4)).requires_grad_()
    Ym = torch.repeat_interleave(Y, dup)
    tgt = F.one_hot(Ym, 4).float()
    out: dict = {"rm": [], "cos_w": [], "rare_w_norm": [], "cls_mse": []}
    for _ in range(steps + 1):
        loss = F.mse_loss(theta @ W, tgt)
        with torch.no_grad():
            lam = torch.linalg.svdvals(theta).square()
            p = lam / lam.sum()
            out["rm"].append(float(torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())))
            out["cos_w"].append(float(F.cosine_similarity(W[:, 2], W[:, 3], dim=0)))
            out["rare_w_norm"].append([float(W[:, 2].norm()), float(W[:, 3].norm())])
            err = (theta @ W - tgt).square().mean(1)
            out["cls_mse"].append([float(err[Ym == c].mean()) for c in range(4)])
        g_theta, g_w = torch.autograd.grad(loss, (theta, W))
        with torch.no_grad():
            theta -= lr * g_theta
            W -= lr * g_w
    out["loss"] = float(loss)
    return {k: torch.tensor(v) for k, v in out.items()}


def main(out_path: str = "data/results/toy/appendix_sweeps.pt") -> None:
    sweeps: dict = {
        "delta": {f"{d:g}/s{s}": run(*_init(delta=d, seed=s))
                  for d in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5) for s in (0, 1, 2)},
        "rx": {f"{rx:g}": run(*_init(rx=rx)) for rx in (0.0, -0.1, -0.25, -0.5)},
        "homotopy": {f"{a:g}": run(*_init(iid_mix=a)) for a in (0.0, 0.25, 0.5, 1.0)},
        "long": {"canonical": run(*_init(), steps=3000)},
        "mse_skew": {f"s{s}": mse_skew_check(seed=s) for s in (0, 1, 2)},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(sweeps, out_path)
    print(f"saved {out_path}: " + ", ".join(f"{k}({len(v)})" for k, v in sweeps.items()))


if __name__ == "__main__":
    main()
