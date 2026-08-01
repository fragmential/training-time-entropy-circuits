"""Null-space composition of MLP write directions (Stolfo et al., arXiv:2406.16254).

A neuron's write direction w_out either reaches the logits or does not. The part that does
not lies in the *effective null space* of the unembedding: the span V0 of its bottom-k right
singular vectors. rho is the share of the write direction inside that span,

    rho_i = ||V0^T w_out^(i)|| / ||w_out^(i)||,     k ~ 0.01 d_model,

which the paper uses (with a high weight norm and a low LogitVar) to identify entropy
neurons. One definition here, shared by the offline sweep (oneoff_scripts/entropy_neurons.py)
and the ablation runs that select neurons by rho during collection.
"""

import torch


def head_subspace(head_weight: torch.Tensor, gamma: "torch.Tensor | None", k: int,
                  center: bool = True) -> torch.Tensor:
    """The bottom-k right singular vectors of the unembedding, as (k, d) orthonormal rows.

    gamma folds the final norm's gain in, as the paper does. Centring over the vocabulary
    (not in the paper) removes the all-ones direction, which softmax ignores; it lands in
    the top singular vectors and moves the bottom-k span by <0.01 in rho.
    """
    w = head_weight.float()
    if center:
        w = w - w.mean(0, keepdim=True)
    if gamma is not None:
        w = w * gamma.float()
    return torch.linalg.svd(w, full_matrices=False).Vh[-k:]


def write_rho(v0: torch.Tensor, w_out: torch.Tensor,
              post_gain: "torch.Tensor | None" = None) -> torch.Tensor:
    """rho for every column of w_out (d, d_ff), given V0 as (k, d).

    post_gain is a norm applied to the block's MLP output after the down-projection
    (OLMo-2's post-feedforward norm): its normalisation is a scalar, so folding the
    per-coordinate gain in keeps the write direction well defined up to scale.
    """
    w = w_out.float()
    if post_gain is not None:
        w = w * post_gain.float()[:, None]
    return (v0 @ (w / w.norm(dim=0).clamp_min(1e-12))).norm(dim=0)
