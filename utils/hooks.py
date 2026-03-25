"""Hook-based collectors for activation covariance, gradient covariance, and residual streams.

CovarianceCollector: attaches to a nn.Linear layer to collect any combination of
    A = E[xxT]  (input covariance)
    B = E[bbT]  (output covariance, where b = layer(x))
    G = E[ggT]  (gradient covariance)

ResidualCapture: captures residual stream activations at configurable hook points.
"""

import torch
import torch.nn as nn
from typing import Optional


class CovarianceCollector:
    """Collect running outer-product covariance for a single nn.Linear layer.

    Supports any combination of A (input), B (output), G (gradient) collection.
    Uses an externally-set token mask to select which positions to include,
    replacing the hardcoded [:, :-1] from the original KFAC class.
    """

    def __init__(
        self,
        layer: nn.Linear,
        collect_A: bool = True,
        collect_G: bool = True,
        collect_B: bool = False,
    ):
        d_out, d_in = layer.weight.shape
        dev = layer.weight.device

        self.collect_A = collect_A
        self.collect_G = collect_G
        self.collect_B = collect_B

        if collect_A:
            self.A = torch.zeros(d_in, d_in, dtype=torch.float32, device=dev)
            self.n_A = 0
        if collect_B:
            self.B = torch.zeros(d_out, d_out, dtype=torch.float32, device=dev)
            self.n_B = 0
        if collect_G:
            self.G = torch.zeros(d_out, d_out, dtype=torch.float32, device=dev)
            self.n_G = 0

        self._token_mask: Optional[torch.BoolTensor] = None
        self._buf_x: Optional[torch.Tensor] = None
        self.active = True

        # Register hooks
        self._handles = []
        if collect_A or collect_B:
            self._handles.append(layer.register_forward_pre_hook(self._fwd_pre))
        if collect_B:
            self._handles.append(layer.register_forward_hook(self._fwd_post))
        if collect_G:
            self._handles.append(layer.register_full_backward_hook(self._bwd))

    def set_token_mask(self, mask: Optional[torch.BoolTensor]):
        """Set per-batch token mask. Shape: (batch, seq_len). True = include."""
        self._token_mask = mask

    def _apply_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply current token mask to a (batch, seq, dim) tensor → (N, dim)."""
        if tensor.dim() == 2:
            # Already flat (batch*seq, dim) — shouldn't happen in normal use
            return tensor.float()
        if self._token_mask is not None:
            return tensor[self._token_mask].float()
        # Fallback: all except last position (backward compat with original KFAC)
        return tensor[:, :-1].reshape(-1, tensor.size(-1)).float()

    def _fwd_pre(self, module, inp):
        if not self.active:
            return
        x = inp[0].detach()
        x_flat = self._apply_mask(x)
        self._buf_x = x_flat  # buffer for B hook if needed
        if self.collect_A:
            self.A.add_(x_flat.T @ x_flat)
            self.n_A += x_flat.size(0)

    def _fwd_post(self, module, inp, output):
        if not self.active:
            return
        b = output.detach()
        b_flat = self._apply_mask(b)
        if self.collect_B:
            self.B.add_(b_flat.T @ b_flat)
            self.n_B += b_flat.size(0)

    def _bwd(self, module, grad_input, grad_output):
        if not self.active:
            return
        go = grad_output[0]
        if go is None:
            return
        g = go.detach()
        g_flat = self._apply_mask(g)
        if self.collect_G:
            self.G.add_(g_flat.T @ g_flat)
            self.n_G += g_flat.size(0)
        self._buf_x = None

    def factors(self) -> dict:
        """Return collected factors as CPU tensors.

        Returns dict with keys "A", "B", "G" (whichever were collected) and "n".
        Matrices are the raw unnormalized sums (Σ xxT). Divide by n to get E[xxT].
        """
        result = {}
        if self.collect_A:
            result["A"] = self.A.cpu()
            result["n_A"] = self.n_A
        if self.collect_B:
            result["B"] = self.B.cpu()
            result["n_B"] = self.n_B
        if self.collect_G:
            result["G"] = self.G.cpu()
            result["n_G"] = self.n_G
        # Convenience: "n" is the A count if available, else G, else B
        result["n"] = self.n_A if self.collect_A else (self.n_G if self.collect_G else self.n_B)
        return result

    def reset(self):
        """Reset accumulators to zero (for reuse across checkpoints)."""
        if self.collect_A:
            self.A.zero_()
            self.n_A = 0
        if self.collect_B:
            self.B.zero_()
            self.n_B = 0
        if self.collect_G:
            self.G.zero_()
            self.n_G = 0
        self._buf_x = None

    def close(self):
        """Remove all hooks and free buffers."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._buf_x = None
        self._token_mask = None


class ResidualCapture:
    """Capture residual stream activations at configurable hook points.

    Hook points:
        "identity_head"      — replace lm_head/embed_out with nn.Identity() (fast path, after final norm)
        "after_final_norm"   — forward_hook on final layernorm output
        "before_final_norm"  — forward_pre_hook on final layernorm input (raw residual)
        "post_attn_N"        — forward_pre_hook on block N's post-attention layernorm
        "pre_block_N"        — forward_pre_hook on block N's input layernorm

    The identity_head method replaces the output head entirely. The hook methods
    capture activations non-destructively.
    """

    def __init__(self, model, config, hook_point: str = "before_final_norm"):
        from utils.model_registry import get_final_layernorm, get_block_layernorms

        self.hook_point = hook_point
        self._handle = None
        self._original_head = None
        self._head_attr = None
        self.activations = None

        if hook_point == "identity_head":
            self._setup_identity_head(model)
        elif hook_point == "after_final_norm":
            target = get_final_layernorm(model, config)
            self._handle = target.register_forward_hook(self._capture_output)
        elif hook_point == "before_final_norm":
            target = get_final_layernorm(model, config)
            self._handle = target.register_forward_pre_hook(self._capture_input)
        elif hook_point.startswith("post_attn_"):
            block_idx = int(hook_point.split("_")[-1])
            _, post_attn_ln = get_block_layernorms(model, config, block_idx)
            self._handle = post_attn_ln.register_forward_pre_hook(self._capture_input)
        elif hook_point.startswith("pre_block_"):
            block_idx = int(hook_point.split("_")[-1])
            input_ln, _ = get_block_layernorms(model, config, block_idx)
            self._handle = input_ln.register_forward_pre_hook(self._capture_input)
        else:
            raise ValueError(f"Unknown hook_point: {hook_point}. "
                             f"Expected: identity_head, after_final_norm, before_final_norm, post_attn_N, pre_block_N")

    def _setup_identity_head(self, model):
        """Replace lm_head/embed_out with nn.Identity()."""
        for attr in ("lm_head", "embed_out"):
            if hasattr(model, attr):
                self._original_head = getattr(model, attr)
                self._head_attr = attr
                setattr(model, attr, nn.Identity())
                return
        if hasattr(model, "set_output_embeddings"):
            self._original_head = model.get_output_embeddings()
            self._head_attr = "__output_embeddings__"
            model.set_output_embeddings(nn.Identity())
            return
        raise ValueError("Could not locate output head for identity_head method")

    def _capture_input(self, module, inp):
        """Forward pre-hook: capture input to the module."""
        self.activations = inp[0].detach()

    def _capture_output(self, module, inp, output):
        """Forward hook: capture output of the module."""
        self.activations = output.detach()

    def get_activations(self, input_ids=None, attention_mask=None, token_selection="last"):
        """Extract activations based on token_selection.

        For identity_head: activations are in model output logits.
        For hook methods: activations are captured by the hook.

        Args:
            input_ids: only needed for identity_head to determine output shape
            attention_mask: for selecting last-token positions
            token_selection: "last" or "all"

        Returns:
            Tensor of shape (batch, dim) for "last" or (N, dim) for "all"
        """
        acts = self.activations
        if acts is None:
            raise RuntimeError("No activations captured. Run a forward pass first.")

        if token_selection == "last":
            if attention_mask is not None:
                last_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(acts.size(0), device=acts.device)
                return acts[batch_indices, last_indices]
            # Packed: last position of sequence
            return acts[:, -1]

        if token_selection == "all":
            if attention_mask is not None:
                # Return only non-pad positions
                return acts[attention_mask.bool()]
            return acts.reshape(-1, acts.size(-1))

        raise ValueError(f"Unknown token_selection: {token_selection}")

    def restore(self, model):
        """Remove hooks and restore original head if replaced."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._original_head is not None and self._head_attr is not None:
            if self._head_attr == "__output_embeddings__":
                model.set_output_embeddings(self._original_head)
            else:
                setattr(model, self._head_attr, self._original_head)
            self._original_head = None
        self.activations = None

    def close(self):
        """Alias for restore (without model arg — hooks only)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.activations = None
