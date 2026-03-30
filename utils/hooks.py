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
    ):
        assert mode in ("cov", "acts"), f"Unknown mode: {mode}"
        assert capture in ("input", "output"), f"Unknown capture: {capture}"

        self.mode = mode
        self.capture = capture
        self.collect_grad = collect_grad
        self.collect_means = collect_means
        self.active = True
        self._acc_dtype = _DTYPE_MAP.get(accumulation_dtype, torch.float64)
        self._act_dtype = _DTYPE_MAP.get(activation_dtype, torch.float32)

        # Token mask — set externally per batch
        self._token_mask: Optional[torch.BoolTensor] = None

        # Forward signal storage (lazy-initialized on first data)
        self._A_initialized = False
        self.A = None       # (d, d) covariance or None
        self.n_A = 0
        self.A_sum = None   # (d,) running sum for means
        self._acts_list = [] if mode == "acts" else None
        self._mask_list = [] if mode == "acts" else None

        # Gradient storage (lazy-initialized)
        self._G_initialized = False
        self.G = None
        self.n_G = 0

        # Register hooks
        self._handles = []
        if module is not None:
            if capture == "input":
                self._handles.append(module.register_forward_pre_hook(self._fwd_pre))
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

    def _accumulate_acts(self, x_3d: torch.Tensor):
        """Acts mode: save ALL tokens (no mask applied). Track mask separately."""
        x_flat = x_3d.reshape(-1, x_3d.shape[-1]).to(dtype=self._act_dtype)
        self._acts_list.append(x_flat.cpu())
        if self._token_mask is not None:
            self._mask_list.append(self._token_mask.reshape(-1).cpu())
        self.n_A += x_flat.shape[0]

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
            if not self._A_initialized:
                self.A = torch.zeros(d, d, dtype=self._acc_dtype, device=x_f.device)
                if self.collect_means:
                    self.A_sum = torch.zeros(d, dtype=torch.float32, device=x_f.device)
                self._A_initialized = True
            self.A.add_((x_f.T @ x_f).to(dtype=self._acc_dtype))
            self.n_A += n
            if self.collect_means:
                self.A_sum.add_(x_f.float().sum(dim=0))
        else:
            self._acts_list.append(x_f.cpu())
            self.n_A += n

    def accumulate_grad(self, g_flat: torch.Tensor):
        """Accumulate a (N, d) gradient tensor into G covariance."""
        if not self.active:
            return
        g_f = g_flat.to(dtype=self._act_dtype)
        d = g_f.size(1)
        if not self._G_initialized:
            self.G = torch.zeros(d, d, dtype=self._acc_dtype, device=g_f.device)
            self._G_initialized = True
        self.G.add_((g_f.T @ g_f).to(dtype=self._acc_dtype))
        self.n_G += g_f.size(0)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _fwd_pre(self, module, inp):
        if not self.active:
            return
        x = inp[0].detach()
        if self.mode == "acts" and x.dim() == 3:
            self._accumulate_acts(x)
        else:
            self.accumulate(self._apply_mask(x))

    def _fwd_post(self, module, inp, output):
        if not self.active:
            return
        out = output.detach()
        if self.mode == "acts" and out.dim() == 3:
            self._accumulate_acts(out)
        else:
            self.accumulate(self._apply_mask(out))

    def _bwd(self, module, grad_input, grad_output):
        if not self.active:
            return
        go = grad_output[0]
        if go is None:
            return
        g = go.detach()
        self.accumulate_grad(self._apply_mask(g))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def factors(self) -> dict:
        """Return collected data as CPU tensors.

        Returns dict with:
            "A": covariance matrix (d,d) in cov mode, or raw activations (N,d) in acts mode
            "n_A": token count
            "A_mean": mean vector (d,) if collect_means and cov mode
            "G": gradient covariance (d,d) if collect_grad
            "n_G": gradient token count
            "n": convenience alias for n_A
        """
        result = {}

        if self.mode == "cov":
            if self.A is not None:
                result["A"] = self.A.cpu()
            if self.collect_means and self.A_sum is not None:
                result["A_mean"] = (self.A_sum / self.n_A).cpu()
        else:
            if self._acts_list:
                result["A"] = torch.cat(self._acts_list, dim=0)
                if self._mask_list:
                    result["A_mask"] = torch.cat(self._mask_list, dim=0)

        result["n_A"] = self.n_A

        if self.collect_grad and self.G is not None:
            result["G"] = self.G.cpu()
            result["n_G"] = self.n_G

        result["n"] = self.n_A
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Reset accumulators to zero (for reuse across checkpoints)."""
        if self.mode == "cov":
            if self.A is not None:
                self.A.zero_()
            if self.A_sum is not None:
                self.A_sum.zero_()
        else:
            self._acts_list = []
            if self._mask_list is not None:
                self._mask_list = []
        self.n_A = 0
        if self.G is not None:
            self.G.zero_()
        self.n_G = 0

    def close(self):
        """Remove all hooks and free buffers."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._token_mask = None
        self._acts_list = None
        self._mask_list = None


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
