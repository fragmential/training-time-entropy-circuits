"""GPU-execution policy, shared by the accessor and compute_metrics so it isn't an
accessor concern: a process-wide lock that serializes GPU work across Pool workers, the
prefer/require-GPU wrappers every GPU op funnels through, and a timing profiler."""
import os
import queue
import threading
import time
from contextlib import nullcontext
from functools import wraps
from typing import Callable

import torch

_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_GPU_EXPECTED = bool(os.environ.get("CUDA_VISIBLE_DEVICES")) or torch.cuda.is_available()

_gpu_lock = nullcontext()   # replaced per-worker by set_gpu_lock (a Pool initializer)


def set_gpu_lock(lock) -> None:
    """Install the cross-worker GPU lock; called in each Pool worker's initializer."""
    global _gpu_lock
    _gpu_lock = lock if lock is not None else nullcontext()


class _DecompProfiler:
    """Times the decompositions it decorates, when enabled (off by default → no overhead)."""
    def __init__(self):
        self.enabled = False
        self.calls: list[tuple[str, tuple, float]] = []

    def enable(self): self.enabled, self.calls = True, []
    def disable(self): self.enabled = False

    def __call__(self, fn: Callable) -> Callable:
        @wraps(fn)
        def timed(M):
            if not self.enabled:
                return fn(M)
            t = time.time()
            out = fn(M)
            self.calls.append((fn.__name__, tuple(M.shape), time.time() - t))
            return out
        return timed

    def summary(self) -> str:
        total = sum(d for *_, d in self.calls)
        return "\n".join([f"{len(self.calls)} decompositions, {total:.2f}s total:",
                          *(f"  {n} {s}: {d:.3f}s" for n, s, d in self.calls)])

decomp_profiler = _DecompProfiler()


def _to_cpu(r):
    return r.cpu() if isinstance(r, torch.Tensor) else tuple(t.cpu() for t in r) if isinstance(r, tuple) else r


def prefer_gpu(fn: Callable) -> Callable:
    """Run fn on the GPU (no-op on CPU), under the GPU lock, result back on CPU — the single
    point all GPU work funnels through, so the lock serializes it across workers."""
    @wraps(fn)
    def wrapped(*args):
        with _gpu_lock:
            return _to_cpu(fn(*(a.to(_DEV) if isinstance(a, torch.Tensor) else a for a in args)))
    return wrapped


def require_gpu(fn: Callable) -> Callable:
    """prefer_gpu, but refuse a large CPU run when a GPU is expected (the forked-worker regression)."""
    run = prefer_gpu(fn)
    @wraps(fn)
    def wrapped(M: torch.Tensor):
        if _GPU_EXPECTED and not torch.cuda.is_available() and M.shape[-1] >= 1024:
            raise RuntimeError(f"eigendecomp {tuple(M.shape)} on CPU while a GPU is present — refusing")
        return run(M)
    return wrapped


def run_pipeline(items, work: Callable, *, workers: int,
                 prefetch: Callable | None = None, max_inflight: int | None = None) -> None:
    """Process `items` in parallel — the one shared parallel-over-files mechanism for
    compute_metrics and the accessor CLI. A producer thread runs `prefetch(item)` (e.g.
    pre-load / pre-download the next checkpoint) and feeds a bounded queue; `workers` consumer
    THREADS run `work(payload)`. Threads share one CUDA context; GPU work serializes through
    the RLock funnel (reentrant — the funnel nests); prefetch I/O overlaps compute. `max_inflight`
    (default workers+2) caps queued payloads so RAM/disk stay bounded. workers<=1 runs inline."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        for it in items:
            work(prefetch(it) if prefetch else it)
        return
    set_gpu_lock(threading.RLock())
    sem, q, done = threading.Semaphore(max_inflight or workers + 2), queue.Queue(), object()

    def producer():
        for it in items:
            sem.acquire()
            q.put(prefetch(it) if prefetch else it)
        for _ in range(workers):
            q.put(done)

    def consumer():
        while (p := q.get()) is not done:
            try:
                work(p)
            finally:
                sem.release()

    ts = [threading.Thread(target=producer)] + [threading.Thread(target=consumer) for _ in range(workers)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
