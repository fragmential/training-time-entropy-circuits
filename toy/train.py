"""Li et al. toy models (RQ3): linear / residual-stream classifiers, full-batch GD,
dumping per-checkpoint activations in DataAccessor `acts` format so
scripts/compute_metrics.py reads them exactly like collected model data.

Leaves: blk{k}.attn.in = stream f_k, blk{k}.mlp.out = write g_k(f_k),
before/after_final_norm = final features (no norm, so both are f_L; storing both keeps
needs_model_weights() False). All dumps go through one fixed isometric embedding into
AMBIENT dims plus a fixed per-leaf noise floor: spectral_metrics' power-law fit needs
>= 13 positive eigenvalues, which d=2 features (and N=8 batches) cannot supply raw.
"""

import math
import os
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

AMBIENT = 16
NOISE = 3e-3   # relative to feature RMS: negligible spectrally, but keeps every embedded
               # eigenvalue above fp32 eigh's error floor so spectral_metrics sees full rank


@dataclass(frozen=True)
class Spec:
    counts: tuple[int, ...]      # samples per class; len = |V|
    d: int                       # feature dim (bottleneck iff d < |V|)
    lr: float
    steps: int
    depth: int = 0               # 0 = the paper's single-layer model
    residual: bool = True
    nonlinear: bool = False
    loss: str = "ce"             # "ce" | "mse"
    init: float = 0.1            # theta / W init std
    balanced: bool = False       # init with f^T f = W W^T (the paper's alignment assumption)
    dup: int = 1                 # exact sample replication: identical init -> identical dynamics,
    seed: int = 0                # RankMe-invariant, lifts N above spectral_metrics' 13-eigval floor


SKEW, UNIFORM = (2, 2, 1, 1), (2, 2, 2, 2)   # the paper's Fig 4 / control class counts
MULTI = Spec(tuple(max(2, 128 >> i) for i in range(32)), d=16, depth=6, lr=0.02, steps=30_000)

VARIANTS: dict[str, Spec] = {
    "single":                   Spec(SKEW, d=2, lr=0.3, steps=200, init=0.3, dup=3, seed=3),
    "uniform":                  Spec(UNIFORM, d=2, lr=0.3, steps=1000, init=0.3, dup=2),
    "nobottleneck":             Spec(SKEW, d=3, lr=0.3, steps=1000, init=0.3, dup=3),
    "mse_uniform":              Spec(UNIFORM, d=2, lr=0.3, steps=1000, init=0.3, dup=2, loss="mse"),
    "mse_skew":                 Spec(SKEW, d=2, lr=0.3, steps=1000, init=0.3, dup=3, loss="mse"),
    "multi_residual":           MULTI,
    "multi_residual_nonlinear": replace(MULTI, nonlinear=True),
    "multi_plain":              replace(MULTI, residual=False, lr=0.005, steps=100_000),
}


class Toy(torch.nn.Module):
    """f_0 = theta rows (inputs S = I, so features are free parameters, as in the paper);
    f_{k+1} = f_k [+] g_k(f_k); logits = f_L @ W."""

    def __init__(self, s: Spec) -> None:
        super().__init__()
        self.s = s
        self.theta = torch.nn.Parameter(s.init * torch.randn(sum(s.counts), s.d).repeat_interleave(s.dup, 0))
        _, sv, Vt = torch.linalg.svd(self.theta.detach(), full_matrices=False)
        Q = torch.linalg.qr(torch.randn(len(s.counts), s.d))[0].T
        self.W = torch.nn.Parameter(Vt.T @ torch.diag(sv) @ Q if s.balanced
                                    else s.init * torch.randn(s.d, len(s.counts)))
        make = ((lambda: torch.nn.Sequential(torch.nn.Linear(s.d, 4 * s.d), torch.nn.Tanh(),
                                             torch.nn.Linear(4 * s.d, s.d)))
                if s.nonlinear else (lambda: torch.nn.Linear(s.d, s.d, bias=False)))
        self.blocks = torch.nn.ModuleList(make() for _ in range(s.depth))

    def forward(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        streams, writes = [self.theta + 0], []
        for g in self.blocks:
            writes.append(g(streams[-1]))
            streams.append(streams[-1] + writes[-1] if self.s.residual else writes[-1])
        return streams, writes


def _loss(s: Spec, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (F.cross_entropy(logits, y) if s.loss == "ce"
            else F.mse_loss(logits, F.one_hot(y, len(s.counts)).float()))


def _leaves(streams: list[torch.Tensor], writes: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {"before_final_norm": streams[-1], "after_final_norm": streams[-1],
            **{f"blk{k}.attn.in": f for k, f in enumerate(streams[:-1])},
            **{f"blk{k}.mlp.out": w for k, w in enumerate(writes)}}


class Dumper:
    """Fixed isometric embedding d -> AMBIENT plus a fixed per-leaf noise floor."""

    def __init__(self, s: Spec, out_dir: str, gen: torch.Generator) -> None:
        self.Q = torch.linalg.qr(torch.randn(AMBIENT, s.d, generator=gen))[0]
        self.noise: dict[str, torch.Tensor] = {}
        self.gen, self.out_dir = gen, out_dir

    def _embed(self, leaf: str, x: torch.Tensor) -> torch.Tensor:
        z = self.noise.setdefault(leaf, torch.randn(x.shape[0], AMBIENT, generator=self.gen))
        e = x.detach().float() @ self.Q.T
        return e + NOISE * e.square().mean().sqrt() * z

    def write(self, leaves: dict[str, torch.Tensor], step: int) -> str:
        entries = {l: {"acts_samples": (e := self._embed(l, x)), "acts_n": e.shape[0]}
                   for l, x in leaves.items()}
        path = os.path.join(self.out_dir, f"step{step}.pt")
        torch.save(dict(entries, __format__="acts"), path)
        return path


def _log_steps(total: int, k: int) -> set[int]:
    return {0, total, *(round(10 ** (i * math.log10(total) / (k - 1))) for i in range(k))}


def train(variant: str, spec: Spec | None = None, output_root: str = "data/inferences",
          results_root: str = "data/results", checkpoints: int = 60,
          seed: int | None = None) -> dict[str, torch.Tensor]:
    """Train one variant, dumping acts checkpoints + returning (and saving) raw trajectories."""
    s = spec or VARIANTS[variant]
    seed = s.seed if seed is None else seed
    torch.manual_seed(seed)
    y = torch.repeat_interleave(torch.arange(len(s.counts)), s.dup * torch.tensor(s.counts))
    model = Toy(s)
    # theta at lr*dup: the batch mean divides every duplicate row's gradient by dup, while W
    # (and blocks) sum over all rows — this rescaling makes dup>1 EXACTLY the dup=1 dynamics.
    opt = torch.optim.SGD([{"params": [model.theta], "lr": s.lr * s.dup},
                           {"params": [model.W, *model.blocks.parameters()], "lr": s.lr}])
    out_dir = os.path.join(output_root, f"toy_{variant}", f"toy-{variant}")
    os.makedirs(out_dir, exist_ok=True)
    dumper = Dumper(s, out_dir, torch.Generator().manual_seed(seed))
    logged = _log_steps(s.steps, checkpoints)
    traj: dict[str, list] = {"steps": [], "loss": [], "W": [], "F": [], "sigma": []}
    for step in range(s.steps + 1):
        streams, writes = model()
        loss = _loss(s, streams[-1] @ model.W, y)
        if step in logged:
            dumper.write(_leaves(streams, writes), step)
            for key, val in [("steps", step), ("loss", float(loss.detach())), ("W", model.W.detach().clone()),
                             ("F", streams[-1].detach().clone()),
                             ("sigma", torch.stack([torch.linalg.svdvals(f.detach()) for f in streams]))]:
                traj[key].append(val)
        opt.zero_grad()
        loss.backward()
        opt.step()
    packed = {k: torch.tensor(v) if k in ("steps", "loss") else torch.stack(v) for k, v in traj.items()}
    path = os.path.join(results_root, "toy", f"trajectories_{variant}.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(packed, path)
    return packed


def main(variant: str = "all", checkpoints: int = 60, seed: int | None = None) -> None:
    for name in list(VARIANTS) if variant == "all" else [variant]:
        t = train(name, checkpoints=checkpoints, seed=seed)
        print(f"{name}: {len(t['steps'])} checkpoints, final loss {t['loss'][-1]:.4f}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
