# utils/hessian_trace.py
"""Hutchinson trace estimator for per-layer Hessian diagnostics.

Efficient: 1 forward + 1 first-order backward + n_draws Hv products per call.
Uses true next-token labels (teacher-forced cross-entropy, mean reduction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def hutchinson_trace(
    model: nn.Module,
    input_ids: torch.Tensor,
    params: list,
    n_draws: int = 6,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Estimate tr(H) for a parameter block via Rademacher Hutchinson.

    Args:
        model: In train() mode, only target params should have requires_grad=True.
        input_ids: (B, T) packed token IDs — labels are the shifted input_ids.
        params: Parameters whose Hessian block to estimate (e.g. one Linear layer).
        n_draws: Number of Rademacher probe vectors.
        generator: Seeded torch.Generator on the same device as input_ids.
    Returns:
        Scalar trace estimate (float).
    """
    device = input_ids.device
    d = sum(p.numel() for p in params)

    # Single forward pass
    logits = model(input_ids).logits
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
    )

    # Single first-order backward (retain graph for Hv products)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.reshape(-1) for g in grads])

    # N Hv products through the same graph
    total = 0.0
    for i in range(n_draws):
        v = torch.randint(0, 2, (d,), generator=generator, device=device).float() * 2 - 1
        Hv = torch.autograd.grad(
            (grad_flat * v).sum(), params,
            retain_graph=(i < n_draws - 1),
        )
        Hv_flat = torch.cat([h.reshape(-1).detach() for h in Hv])
        total += (v * Hv_flat).sum().item()

    return total / n_draws
