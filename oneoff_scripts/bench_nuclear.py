"""Benchmark the nuclear-norm decomposition at compute_metrics.py:847.

For each model scale (d, n from configs/block_representations_samples.yaml), time:
  - the pre-existing per-pair cost  (Xj^T Xk matmul + the mean-cos reduction)
  - svdvals default driver         (what is in the code now)
  - svdvals driver='gesvdj'        (Jacobi)
  - svdvals driver='gesvda'
  - eigvalsh(C^T C).sqrt().sum()   (the alternative route)
and check the nuclear-norm relative error of each against an fp64 svdvals reference.
"""
import time, torch

print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
print(f"matmul.allow_tf32 = {torch.backends.cuda.matmul.allow_tf32}")
try:
    print(f"matmul.fp32_precision = {torch.backends.cuda.matmul.fp32_precision}")
except AttributeError:
    pass
print()

# (d, n) per configs/block_representations_samples.yaml max_tokens
CASES = [(768, 262_144, "pythia-160m"), (1024, 262_144, "pythia-410m"),
         (2048, 262_144, "pythia-1b / OLMo-2-1B"), (4096, 81_920, "pythia-6.9b / OLMo-2-7B")]
REPS = 5


def timed(fn, reps=REPS, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def make_C(d, n, seed=0):
    """A realistic centered cross-covariance: two correlated activation-like populations
    with a decaying spectrum (Jacobi convergence depends on the spectrum, so this matters)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = torch.logspace(0, -2.5, d, device="cuda")            # decaying per-direction std
    Z = torch.randn(n, d, generator=g, device="cuda") * scale
    Xj = (Z + 0.3 * torch.randn(n, d, generator=g, device="cuda") * scale).bfloat16()
    Xk = (Z + 0.3 * torch.randn(n, d, generator=g, device="cuda") * scale).bfloat16()
    return Xj, Xk


for d, n, label in CASES:
    print(f"=== d={d}  n={n}   ({label})")
    Xj, Xk = make_C(d, n)
    mj, mk = Xj.float().mean(0), Xk.float().mean(0)

    def pair_baseline():
        C = Xj.float().T @ Xk.float() / Xj.shape[0] - torch.outer(mj, mk)
        C.trace(), C.square().sum()
        ((Xj.float() * Xk.float()).sum(1)
         / (Xj.float().norm(dim=1) * Xk.float().norm(dim=1))).mean()

    t_base = timed(pair_baseline, reps=3)
    C = (Xj.float().T @ Xk.float() / Xj.shape[0] - torch.outer(mj, mk)).contiguous()
    del Xj, Xk
    torch.cuda.empty_cache()

    ref = float(torch.linalg.svdvals(C.double()).sum())

    variants = {
        "svdvals (default)": lambda: torch.linalg.svdvals(C),
        "svdvals gesvdj":    lambda: torch.linalg.svdvals(C, driver="gesvdj"),
        "svdvals gesvda":    lambda: torch.linalg.svdvals(C, driver="gesvda"),
        "eigvalsh(C^T C)":   lambda: torch.linalg.eigvalsh(C.T @ C).clamp_min(0).sqrt(),
    }
    print(f"  {'pre-existing pair work':<22} {t_base*1e3:8.1f} ms")
    for name, fn in variants.items():
        try:
            t = timed(fn)
            err = abs(float(fn().sum()) - ref) / ref
            print(f"  {name:<22} {t*1e3:8.1f} ms   {t/t_base:5.2f}x pair   "
                  f"nuc rel.err {err:.2e}")
        except Exception as e:
            print(f"  {name:<22} FAILED: {type(e).__name__}: {e}")
    del C
    torch.cuda.empty_cache()
    print()
