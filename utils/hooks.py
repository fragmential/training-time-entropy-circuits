"""Hook-based collector for activations and gradients at one leaf capture-point.

A HookCollector attaches to one nn.Module side (the leaf) and collects the
requested quantities there: forward acts (raw samples or accumulated covariance)
and/or backward grad covariance. `quantities` (subset of {"acts","grads"})
decides which hooks register; everything lands under the single `leaf` name.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
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


class Capturer(ABC):
    """A collection endpoint collect.py drives uniformly: set its token mask, read what it
    captured ({leaf: {...}}), then close it. HookCollector covers one leaf; MultiHeadOVDispatcher
    fans a block's OV heads out to per-head HookCollectors."""
    @abstractmethod
    def set_token_mask(self, mask) -> None: ...
    @abstractmethod
    def captured(self) -> dict: ...
    @abstractmethod
    def close(self) -> None: ...


class HookCollector(Capturer):
    """Collect acts and/or grads at a single leaf capture-point.

    Modes (for acts only — grads are always accumulated as covariance):
        "cov"  — accumulates Σ xxT and n. Optionally means. Default.
        "acts" — stores raw (masked) activation tensors in a list.

    Args:
        module: Module to hook. If None, feed data manually via feed()/feed_grad().
        capture: "input" (pre-hook / grad_input) or "output" (post-hook / grad_output).
        quantities: subset of {"acts","grads"} — decides which hooks register.
        mode: "cov" or "acts" (acts storage only).
        collect_means: Store running mean of activations (for +m modifier).
        accumulation_dtype: covariance accumulator dtype (default fp64 for log-det stability).
        activation_dtype: dtype acts/grads cast to before outer products (default fp32).
        leaf: storage leaf name everything is written under.
    """

    def __init__(
        self,
        module: Optional[nn.Module] = None,
        capture: str = "input",
        quantities=("acts",),
        mode: str = "cov",
        collect_means: bool = False,
        accumulation_dtype: str = "fp64",
        activation_dtype: str = "fp32",
        token_selection: str = None,
        leaf: str = "acts",
    ):
        assert mode in ("cov", "acts"), f"Unknown mode: {mode}"
        assert capture in ("input", "output"), f"Unknown capture: {capture}"

        self.mode = mode
        self.capture = capture
        self.quantities = set(quantities)
        self.token_selection = token_selection
        self.collect_means = collect_means
        self.leaf = leaf
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

        # Register hooks: forward for acts (side decides pre/post), backward for grads.
        self._handles = []
        if module is not None:
            if "acts" in self.quantities:
                if capture == "input":
                    self._handles.append(module.register_forward_pre_hook(self._fwd_pre, with_kwargs=True))
                else:
                    self._handles.append(module.register_forward_hook(self._fwd_post))
            if "grads" in self.quantities:
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
    # Accumulation (public — also used for fast_final_norm manual feeding)
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

    def feed(self, raw: torch.Tensor):
        """Mask a raw (B,T,d)/(N,d) activation tensor, then accumulate."""
        self.accumulate(self._apply_mask(raw))

    def feed_grad(self, raw: torch.Tensor):
        """Mask a raw (B,T,d)/(N,d) gradient tensor, then accumulate."""
        self.accumulate_grad(self._apply_mask(raw))

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _fwd_pre(self, module, args, kwargs):
        if not self.active:
            return
        x = _hook_input(args, kwargs).detach()
        self.feed(x)

    def _fwd_post(self, module, inp, output):
        if not self.active:
            return
        # Attention modules return (attn_output, attn_weights[, ...]); unwrap to the first tensor.
        if isinstance(output, tuple):
            output = output[0]
        out = output.detach()
        self.feed(out)

    def _bwd(self, module, grad_input, grad_output):
        if not self.active:
            return
        go = grad_input[0] if self.capture == "input" else grad_output[0]
        if go is None:
            return
        self.feed_grad(go.detach())

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def captured(self) -> dict:
        """Collected data as {leaf: {q-keyed tensors}} under this collector's single
        leaf. Keys are uniform `{q}_{component}` — `acts_gram` (raw Σxxᵀ) or
        `acts_samples` (N,d), `acts_n`, `acts_mean`; same for grads. Emits whatever
        was accumulated."""
        e: dict = {}
        if self.mode == "cov":
            if self._acts_cov is not None:
                e["acts_gram"] = self._acts_cov.cpu()
                e["acts_n"] = self._n_acts
                if self.collect_means and self._acts_sum is not None:
                    e["acts_mean"] = (self._acts_sum / self._n_acts).cpu()
        elif self._acts_list:
            e["acts_samples"] = torch.cat(self._acts_list, dim=0)
            e["acts_n"] = self._n_acts

        if self._grads_cov is not None:
            e["grads_gram"] = self._grads_cov.cpu()
            e["grads_n"] = self._n_grads
            if self.collect_means and self._grads_sum is not None:
                e["grads_mean"] = (self._grads_sum / self._n_grads).cpu()

        return {self.leaf: e} if e else {}

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

class MultiHeadOVDispatcher(Capturer):
    """Collect per-OV-head pre-W_o slices at one block's o_proj.

    o_proj is always called on (B, T, num_heads*head_dim), so head h's pre-W_o
    slice is o_proj_input[..., h*d_head:(h+1)*d_head]; its post-W_o contribution
    (the `…head{h}.contrib` leaf) is derived later from o_proj's column block.
    Storing the slice is H²× smaller than the contribution covariance. One
    forward pre-hook (+ optional backward, pulling grads through W_h) feeds a
    per-head HookCollector writing the `blk{i}.attn.head{h}.slice` leaf.
    """

    def __init__(
        self,
        o_proj_module: nn.Module,
        num_heads: int,
        head_dim: int,
        block_idx: int,
        selected_heads=None,
        quantities=("acts",),
        mode: str = "cov",
        collect_means: bool = False,
        accumulation_dtype: str = "fp64",
        activation_dtype: str = "fp32",
        token_selection: str = None,
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_idx = block_idx
        self.quantities = set(quantities)
        if selected_heads is None:
            selected_heads = list(range(num_heads))
        self.selected_heads = list(selected_heads)
        self._weight_ref = o_proj_module.weight  # (d_model, num_heads * head_dim)
        self.collectors = {
            h: HookCollector(
                module=None, mode=mode,
                collect_means=collect_means,
                accumulation_dtype=accumulation_dtype,
                activation_dtype=activation_dtype,
                token_selection=token_selection,
                leaf=f"blk{block_idx}.attn.head{h}.slice",
            )
            for h in self.selected_heads
        }
        self._handles = []
        if "acts" in self.quantities:
            self._handles.append(o_proj_module.register_forward_pre_hook(self._fwd_pre, with_kwargs=True))
        if "grads" in self.quantities:
            self._handles.append(o_proj_module.register_full_backward_hook(self._bwd))

    def set_token_mask(self, mask):
        for c in self.collectors.values():
            c.set_token_mask(mask)

    def _fwd_pre(self, module, args, kwargs):
        x = _hook_input(args, kwargs).detach()  # (B, T, num_heads * head_dim)
        d = self.head_dim
        for h, collector in self.collectors.items():
            attended_h = x[..., h * d : (h + 1) * d]  # (B, T, d_head)
            collector.feed(attended_h)

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
            collector.feed_grad(g_h)

    def captured(self) -> dict:
        out: dict = {}
        for c in self.collectors.values():
            out.update(c.captured())   # each per-head collector yields its own .slice leaf
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for c in self.collectors.values():
            c.close()
