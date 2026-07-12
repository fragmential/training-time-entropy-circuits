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
    clustered: bool = False      # Li et al Fig-4 init (read off their gray t=0 markers):
                                 #   frequent-class features clustered on separated directions with
                                 #   W columns aligned; ALL rare classes coincident with zero weights.
                                 #   GD preserves the rare-pair swap symmetry, so the shared path is
                                 #   exact and `delta` sets the split time (~ log 1/delta).
    delta: float = 1e-3          # clustered: global jitter std breaking the rare-class symmetry
    dup: int = 1                 # exact sample replication: identical init -> identical dynamics,
    seed: int = 0                # RankMe-invariant, lifts N above spectral_metrics' 13-eigval floor
    norm: str = "none"           # "prenorm": each write reads rms(stream) | "writenorm": each write's
                                 #   output is rms-normalized before the residual add (OLMo-2's
                                 #   reordered norm) | "bothnorm": both; the stream itself stays raw
    writes: int = 1              # writes per block; 2 dumps them as blk{k}.{attn,mlp}.out
    parallel: bool = True        # writes>1: all read the block input | each reads input + prior writes


SKEW, UNIFORM = (2, 2, 1, 1), (2, 2, 2, 2)   # the paper's Fig 4 / control class counts
MULTI = Spec(tuple(max(2, 128 >> i) for i in range(32)), d=16, depth=6, lr=0.02, steps=30_000)

VARIANTS: dict[str, Spec] = {
    "single":                   Spec(SKEW, d=2, lr=0.25, steps=300, clustered=True, dup=3),
    "uniform":                  Spec(UNIFORM, d=2, lr=0.3, steps=1000, init=0.3, dup=2),
    "nobottleneck":             Spec(SKEW, d=3, lr=0.3, steps=1000, init=0.3, dup=3),
    "mse_uniform":              Spec(UNIFORM, d=2, lr=0.3, steps=1000, init=0.3, dup=2, loss="mse"),
    "mse_skew":                 Spec(SKEW, d=2, lr=0.3, steps=1000, init=0.3, dup=3, loss="mse"),
    "multi_residual":           MULTI,
    "multi_residual_nonlinear": replace(MULTI, nonlinear=True),
    "multi_plain":              replace(MULTI, residual=False, lr=0.005, steps=100_000),
    # architecture-knob grid on the multi_residual_nonlinear base (two writes per block,
    # identical parameter count across parallel/sequential): norm {none, prenorm} x wiring
    "arch_par":     (MULTI2 := replace(MULTI, nonlinear=True, writes=2)),
    "arch_seq":     replace(MULTI2, parallel=False),
    "arch_pre_par": replace(MULTI2, norm="prenorm"),
    "arch_pre_seq": replace(MULTI2, norm="prenorm", parallel=False),
    "arch_wn_par":  replace(MULTI2, norm="writenorm"),
    "arch_wn_seq":  replace(MULTI2, norm="writenorm", parallel=False),
    "arch_bn_par":  replace(MULTI2, norm="bothnorm"),
    "arch_bn_seq":  replace(MULTI2, norm="bothnorm", parallel=False),
}


def _clustered_init(s: Spec) -> tuple[torch.Tensor, torch.Tensor]:
    """The paper's Fig-4 t=0 geometry (two frequent classes on separated directions, all
    rare classes coincident): features per frequent class at scale·dir ± eps, rare samples
    all at (-0.25, 0); W columns near the class directions, rare columns zero. A global
    delta-scale jitter breaks the exact rare-class symmetry and sets the split time."""
    assert s.d == 2 and len(s.counts) == 4 and s.counts[2:] == (1, 1), \
        "clustered init implements the paper's Fig-4 geometry (4 classes, d=2, rare pair last)"
    a = math.radians(100)
    dirs = [torch.tensor([math.cos(a), math.sin(a)]), torch.tensor([1.0, 0.03])]
    perp = lambda u: torch.tensor([-float(u[1]), float(u[0])])
    rare = torch.zeros(2)   # exact origin: matches the paper's zero fork→decline lag (addendum)
    theta0 = torch.stack([0.6 * dirs[0] + 0.05 * perp(dirs[0]), 0.6 * dirs[0] - 0.05 * perp(dirs[0]),
                          0.75 * dirs[1] + 0.05 * perp(dirs[1]), 0.75 * dirs[1] - 0.05 * perp(dirs[1]),
                          rare, rare])
    W0 = torch.stack([torch.tensor([0.0, 0.6]), 0.74 * dirs[1],
                      torch.zeros(2), torch.zeros(2)], dim=1)
    return theta0 + s.delta * torch.randn_like(theta0), W0 + s.delta * torch.randn_like(W0)


class Toy(torch.nn.Module):
    """f_0 = theta rows (inputs S = I, so features are free parameters, as in the paper);
    f_{k+1} = f_k [+] g_k(f_k); logits = f_L @ W."""

    def __init__(self, s: Spec) -> None:
        super().__init__()
        self.s = s
        if s.clustered:
            theta0, W0 = _clustered_init(s)
            self.theta = torch.nn.Parameter(theta0.repeat_interleave(s.dup, 0))
            self.W = torch.nn.Parameter(W0)
        else:
            self.theta = torch.nn.Parameter(s.init * torch.randn(sum(s.counts), s.d).repeat_interleave(s.dup, 0))
            _, sv, Vt = torch.linalg.svd(self.theta.detach(), full_matrices=False)
            Q = torch.linalg.qr(torch.randn(len(s.counts), s.d))[0].T
            self.W = torch.nn.Parameter(Vt.T @ torch.diag(sv) @ Q if s.balanced
                                        else s.init * torch.randn(s.d, len(s.counts)))
        make = ((lambda: torch.nn.Sequential(torch.nn.Linear(s.d, 4 * s.d), torch.nn.Tanh(),
                                             torch.nn.Linear(4 * s.d, s.d)))
                if s.nonlinear else (lambda: torch.nn.Linear(s.d, s.d, bias=False)))
        self.blocks = torch.nn.ModuleList(make() for _ in range(s.depth * s.writes))

    def forward(self) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        read = _rms if self.s.norm in ("prenorm", "bothnorm") else (lambda x: x)
        emit = _rms if self.s.norm in ("writenorm", "bothnorm") else (lambda x: x)
        streams, writes = [self.theta + 0], []
        for k in range(self.s.depth):
            f, ws = streams[-1], []
            for g in self.blocks[k * self.s.writes:(k + 1) * self.s.writes]:
                ws.append(emit(g(read(f if self.s.parallel else sum(ws, f)))))
            writes.append(ws)
            streams.append(sum(ws, f if self.s.residual else torch.zeros_like(f)))
        return streams, writes


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x * x.square().mean(-1, keepdim=True).add(1e-8).rsqrt()


def _loss(s: Spec, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (F.cross_entropy(logits, y) if s.loss == "ce"
            else F.mse_loss(logits, F.one_hot(y, len(s.counts)).float()))


def _leaves(streams: list[torch.Tensor], writes: list[list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"before_final_norm": streams[-1], "after_final_norm": streams[-1],
            **{f"blk{k}.attn.in": f for k, f in enumerate(streams[:-1])},
            **{f"blk{k}.{n}": w for k, ws in enumerate(writes)
               for n, w in zip(("attn.out", "mlp.out")[-len(ws):], ws)}}


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


def main(variant: str = "all", checkpoints: int = 60, seed: int | None = None,
         seeds: tuple[int, ...] = ()) -> None:
    """`seeds` sweeps a variant, suffixing dump/results names with _s{seed}."""
    for name in list(VARIANTS) if variant == "all" else [variant]:
        for tag, sd in [(f"{name}_s{s}", s) for s in seeds] or [(name, seed)]:
            t = train(tag, spec=VARIANTS[name], checkpoints=checkpoints, seed=sd)
            print(f"{tag}: {len(t['steps'])} checkpoints, final loss {t['loss'][-1]:.4f}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
