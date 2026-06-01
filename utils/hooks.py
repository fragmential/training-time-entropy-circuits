"""Hook-based collector for activations and covariance at arbitrary model hook points.

HookCollector: unified class that attaches to any nn.Module to collect:
    - Forward activations (input or output) as raw tensors or accumulated covariance
    - Gradient covariance (always accumulated, never raw)

Replaces the previous CovarianceCollector + ResidualCapture split.
"""

import torch
import torch.nn as nn
from typing import Optional

_DTYPE_MAP = {
    "float32": torch.float32, "fp32": torch.float32,
    "float64": torch.float64, "fp64": torch.float64,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
}


def _hook_input(args, kwargs):
    """Primary input tensor to a hooked module, whether passed positionally or by keyword.

    Modules like OLMo-2's self_attn are invoked as `self_attn(hidden_states=...)`,
    so a forward-pre-hook's positional `args` is empty; the tensor lives in kwargs.
    """
    if args:
        return args[0]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    return next(iter(kwargs.values()))


class HookCollector:
    """Collect activations or covariance at a single hook point.

    Modes:
        "cov"  — accumulates Σ xxT and n. Optionally means. Default.
        "acts" — stores raw (masked) activation tensors in a list.

    For gradient collection (collect_grad=True), gradient covariance is always
    accumulated regardless of mode (storing raw gradients is impractical).

    Args:
        module: Module to hook. If None, call accumulate() manually (e.g. identity_head).
        capture: "input" (forward_pre_hook) or "output" (forward_hook).
        mode: "cov" or "acts".
        collect_grad: Also collect gradient covariance via backward hook.
        collect_means: Store running mean of activations (for +m modifier).
        accumulation_dtype: Dtype for covariance accumulators A and G. Default fp64
                            for numerical stability of log-det computation.
        activation_dtype: Dtype activations/gradients are cast to before outer products.
                          Default fp32.
    """

    def __init__(
        self,
        module: nn.Module = None,
        capture: str = "input",
        mode: str = "cov",
        collect_grad: bool = False,
        collect_means: bool = False,
        accumulation_dtype: str = "fp64",
        activation_dtype: str = "fp32",
        grad_capture: str = None,
        token_selection: str = None,
        acts_key: str = "in.acts",
        grads_key: str = "out.grads",
    ):
        assert mode in ("cov", "acts"), f"Unknown mode: {mode}"
        assert capture in ("input", "output"), f"Unknown capture: {capture}"

        self.mode = mode
        self.capture = capture
        self.token_selection = token_selection
        self.grad_capture = grad_capture if grad_capture is not None else capture
        self.collect_grad = collect_grad
        self.collect_means = collect_means
        self.acts_key = acts_key      # signal key for the forward (acts) cov
        self.grads_key = grads_key    # signal key for the backward (grads) cov
        self.active = True
        self._acc_dtype = _DTYPE_MAP.get(accumulation_dtype, torch.float64)
        self._act_dtype = _DTYPE_MAP.get(activation_dtype, torch.float32)

        # Token mask — set externally per batch
        self._token_mask: Optional[torch.BoolTensor] = None

        # Forward (acts) storage (lazy-initialized on first data)
        self._acts_init = False
        self._acts_cov = None    # (d, d) covariance or None
        self._n_acts = 0
        self._acts_sum = None    # (d,) running sum for means
        self._acts_list = [] if mode == "acts" else None

        # Backward (grads) storage (lazy-initialized)
        self._grads_init = False
        self._grads_cov = None
        self._n_grads = 0
        self._grads_sum = None   # (d,) running sum for gradient means

        # Register hooks
        self._handles = []
        if module is not None:
            if capture == "input":
                self._handles.append(module.register_forward_pre_hook(self._fwd_pre, with_kwargs=True))
            else:
                self._handles.append(module.register_forward_hook(self._fwd_post))
            if collect_grad:
                self._handles.append(module.register_full_backward_hook(self._bwd))

    # ------------------------------------------------------------------
    # Token masking
    # ------------------------------------------------------------------

    def set_token_mask(self, mask: Optional[torch.BoolTensor]):
        """Set per-batch token mask. Shape: (batch, seq_len). True = include."""
        self._token_mask = mask

    def _apply_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply current token mask to a (batch, seq, dim) tensor → (N, dim)."""
        if tensor.dim() == 2:
            return tensor.to(dtype=self._act_dtype)
        if self._token_mask is not None:
            return tensor[self._token_mask].to(dtype=self._act_dtype)
        # Fallback: all except last position
        return tensor[:, :-1].reshape(-1, tensor.size(-1)).to(dtype=self._act_dtype)

    # ------------------------------------------------------------------
    # Accumulation (public — also used for identity_head manual feeding)
    # ------------------------------------------------------------------

    def accumulate(self, x_flat: torch.Tensor):
        """Accumulate a (N, d) activation tensor into storage.

        For cov mode: updates Σ xxT and n.
        For acts mode: appends to raw tensor list.
        """
        if not self.active:
            return
        x_f = x_flat.to(dtype=self._act_dtype)
        n = x_f.size(0)

        if self.mode == "cov":
            d = x_f.size(1)
            if not self._acts_init:
                self._acts_cov = torch.zeros(d, d, dtype=self._acc_dtype, device=x_f.device)
                if self.collect_means:
                    self._acts_sum = torch.zeros(d, dtype=torch.float32, device=x_f.device)
                self._acts_init = True
            self._acts_cov.add_((x_f.T @ x_f).to(dtype=self._acc_dtype))
            self._n_acts += n
            if self.collect_means:
                self._acts_sum.add_(x_f.float().sum(dim=0))
        else:
            self._acts_list.append(x_f.cpu())
            self._n_acts += n

    def accumulate_grad(self, g_flat: torch.Tensor):
        """Accumulate a (N, d) gradient tensor into the grads covariance."""
        if not self.active:
            return
        g_f = g_flat.to(dtype=self._act_dtype)
        d = g_f.size(1)
        if not self._grads_init:
            self._grads_cov = torch.zeros(d, d, dtype=self._acc_dtype, device=g_f.device)
            if self.collect_means:
                self._grads_sum = torch.zeros(d, dtype=torch.float32, device=g_f.device)
            self._grads_init = True
        self._grads_cov.add_((g_f.T @ g_f).to(dtype=self._acc_dtype))
        self._n_grads += g_f.size(0)
        if self.collect_means:
            self._grads_sum.add_(g_f.float().sum(dim=0))

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _fwd_pre(self, module, args, kwargs):
        if not self.active:
            return
        x = _hook_input(args, kwargs).detach()
        self.accumulate(self._apply_mask(x))

    def _fwd_post(self, module, inp, output):
        if not self.active:
            return
        # Attention modules return (attn_output, attn_weights[, ...]); unwrap to the first tensor.
        if isinstance(output, tuple):
            output = output[0]
        out = output.detach()
        self.accumulate(self._apply_mask(out))

    def _bwd(self, module, grad_input, grad_output):
        if not self.active:
            return
        go = grad_input[0] if self.grad_capture == "input" else grad_output[0]
        if go is None:
            return
        g = go.detach()
        self.accumulate_grad(self._apply_mask(g))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def factors(self) -> dict:
        """Return collected data as CPU tensors, keyed by signal.

        With acts_key="in.acts", grads_key="out.grads" the dict holds:
            "in.acts": covariance (d,d) in cov mode, or raw samples (N,d) in acts mode
            "n_in.acts": token count
            "in.acts_mean": mean vector (d,) if collect_means and cov mode
            "out.grads": gradient covariance (d,d) if collect_grad
            "n_out.grads": gradient token count
            "out.grads_mean": mean gradient vector (d,) if collect_means and collect_grad
            "n": convenience alias for n_acts
        """
        a, g = self.acts_key, self.grads_key
        result = {}

        if self.mode == "cov":
            if self._acts_cov is not None:
                result[a] = self._acts_cov.cpu()
            if self.collect_means and self._acts_sum is not None:
                result[f"{a}_mean"] = (self._acts_sum / self._n_acts).cpu()
        else:
            if self._acts_list:
                result[a] = torch.cat(self._acts_list, dim=0)

        result[f"n_{a}"] = self._n_acts

        if self.collect_grad and self._grads_cov is not None:
            result[g] = self._grads_cov.cpu()
            result[f"n_{g}"] = self._n_grads
            if self.collect_means and self._grads_sum is not None:
                result[f"{g}_mean"] = (self._grads_sum / self._n_grads).cpu()

        result["n"] = self._n_acts
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset accumulators to zero (for reuse across checkpoints)."""
        if self.mode == "cov":
            if self._acts_cov is not None:
                self._acts_cov.zero_()
            if self._acts_sum is not None:
                self._acts_sum.zero_()
        else:
            self._acts_list = []
        self._n_acts = 0
        if self._grads_cov is not None:
            self._grads_cov.zero_()
        if self._grads_sum is not None:
            self._grads_sum.zero_()
        self._n_grads = 0

    def close(self):
        """Remove all hooks and free buffers."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._token_mask = None
        self._acts_list = None


# ---------------------------------------------------------------------------
# Identity head setup (used by collect.py for fast post-norm residual capture)
# ---------------------------------------------------------------------------

def setup_identity_head(model) -> tuple:
    """Replace lm_head/embed_out with nn.Identity().

    Returns (original_module, attr_name) for later restoration via restore_head().
    """
    for attr in ("lm_head", "embed_out"):
        if hasattr(model, attr):
            original = getattr(model, attr)
            setattr(model, attr, nn.Identity())
            return original, attr
    if hasattr(model, "set_output_embeddings"):
        original = model.get_output_embeddings()
        model.set_output_embeddings(nn.Identity())
        return original, "__output_embeddings__"
    raise ValueError("Could not locate output head for identity_head setup")


def restore_head(model, original, attr_name):
    """Restore the original output head after identity_head usage."""
    if attr_name == "__output_embeddings__":
        model.set_output_embeddings(original)
    else:
        setattr(model, attr_name, original)


# ---------------------------------------------------------------------------
# Per-OV-head dispatcher
# ---------------------------------------------------------------------------

class MultiHeadOVDispatcher:
    """Collect per-OV-head pre-W_o slices from an attention output projection.

    The attention output projection (o_proj / dense) is always called on a
    tensor of shape (B, T, num_heads * head_dim) regardless of the attention
    backend (eager / SDPA / flash). Per head h, the pre-W_o slice is:
        attended_h = o_proj_input[..., h*d_head:(h+1)*d_head]  # (B, T, d_head)
    and the residual contribution `contrib_h = attended_h @ W_h.T` can be
    recovered losslessly from this slice plus the corresponding column block
    of o_proj.weight (handled at metric time by DataAccessor's .O derivation).
    Storing the pre-W_o slice is `H²×` smaller than storing the (d_model, d_model)
    covariance of contrib_h.

    Owns one forward pre-hook on o_proj (plus an optional backward hook) and
    dispatches per-head slices into per-head d_head-sized HookCollectors keyed by
    f"blk{block_idx}.attn.head{h}" for each h in `selected_heads`.

    Note on G: the gradient pulled back through W_h is
        grad_input_h = grad_output @ W_h ∈ R^d_head
    per head. The backward hook on o_proj receives a single d_model-sized
    grad_output and projects it through the corresponding W_h slice for each
    selected head.
    """

    def __init__(
        self,
        o_proj_module: nn.Module,
        num_heads: int,
        head_dim: int,
        block_idx: int,
        selected_heads=None,
        mode: str = "cov",
        collect_means: bool = False,
        collect_grad: bool = False,
        accumulation_dtype: str = "fp64",
        activation_dtype: str = "fp32",
        token_selection: str = None,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_idx = block_idx
        self.collect_grad = collect_grad
        if selected_heads is None:
            selected_heads = list(range(num_heads))
        self.selected_heads = list(selected_heads)
        self._weight_ref = o_proj_module.weight  # (d_model, num_heads * head_dim)
        self.collectors = {
            h: HookCollector(
                module=None, mode=mode,
                collect_means=collect_means,
                collect_grad=collect_grad,
                accumulation_dtype=accumulation_dtype,
                activation_dtype=activation_dtype,
                token_selection=token_selection,
                acts_key="slice.acts", grads_key="slice.grads",
            )
            for h in self.selected_heads
        }
        self._handles = [o_proj_module.register_forward_pre_hook(self._fwd_pre, with_kwargs=True)]
        if collect_grad:
            self._handles.append(o_proj_module.register_full_backward_hook(self._bwd))

    def set_token_mask(self, mask):
        for c in self.collectors.values():
            c.set_token_mask(mask)

    def _fwd_pre(self, module, args, kwargs):
        x = _hook_input(args, kwargs).detach()  # (B, T, num_heads * head_dim)
        d = self.head_dim
        for h, collector in self.collectors.items():
            attended_h = x[..., h * d : (h + 1) * d]  # (B, T, d_head)
            collector.accumulate(collector._apply_mask(attended_h))

    def _bwd(self, module, grad_input, grad_output):
        go = grad_output[0] if grad_output else None
        if go is None:
            return
        W = self._weight_ref.detach()  # (d_model, num_heads * d_head)
        g = go.detach()  # (B, T, d_model)
        d = self.head_dim
        for h, collector in self.collectors.items():
            W_h = W[:, h * d : (h + 1) * d]  # (d_model, d_head)
            g_h = g @ W_h  # (B, T, d_head)
            collector.accumulate_grad(collector._apply_mask(g_h))

    def factors(self) -> dict:
        return {
            f"blk{self.block_idx}.attn.head{h}": c.factors()
            for h, c in self.collectors.items()
        }

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for c in self.collectors.values():
            c.close()
