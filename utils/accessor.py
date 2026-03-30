"""Format-agnostic interface for reading collected data.

DataAccessor provides a chainable property-based API for accessing stored
activation/covariance data regardless of storage format:

    acc = DataAccessor(data)

    acc["blk3.up"].A.eigvals           # 1D float Tensor, descending
    acc["blk3.up"].B.cov               # (d_out, d_out) Tensor, derived if model given
    acc["blk3.up"].G.eigh              # (eigvals Tensor, eigvecs Tensor)
    acc["blk3.up"].A.mean              # stored mean (+m modifier)
    acc["blk3.up"].A.svd               # (S, V) from eigdecomp, or (U, S, V) from acts
    acc.blocks[3].up.A.eigvals         # same as above, block-indexed

    acc.after_final_norm.A.eigvals     # final residual stream, post-norm
    acc.before_final_norm.A.eigvals    # final residual stream, pre-norm
    acc["blk3.up"].input.eigvals       # alias: input = A
    acc["blk3.up"].output.cov          # alias: output = B
    acc["blk3.up"].grad.eigvals        # alias: grad = G

Derivation priority (transparent, in order):
    eigvals:  stored eigvals → eigendecompose cov → PCA on acts → derive B from A+W
    eigvecs:  stored eigvecs → eigendecompose cov → eigendecompose acts
    cov:      stored cov     → reconstruct V diag(λ) V^T → compute from acts
    acts:     stored acts    → reconstruct from acts_svd (U S V^T)
    svd:      full SVD if acts available, else (S, V) from eigh
    B:        stored → derived from A + W + bias + mean (needs model + model_config)

Note:
    DataAccessor will be merged with storage.py in Phase 7.
"""

import re
import torch
import numpy as np
from typing import Optional


class DataAccessor:
    """Format-agnostic reader for collected activation / covariance data.

    Args:
        data: Loaded .pt dict or path to a .pt file.
        model: Optional nn.Module for model-dependent derivations (B).
        model_config: Optional ModelConfig for resolving hook names to layers.
    """

    def __init__(self, data, model=None, model_config=None):
        if isinstance(data, str):
            data = torch.load(data, map_location="cpu", weights_only=False)
        self.data = data
        self.model = model
        self.model_config = model_config
        self._layer_cache: dict = {}

    # ------------------------------------------------------------------
    # Public access points
    # ------------------------------------------------------------------

    def __getitem__(self, hook_name: str) -> "HookView":
        return HookView(hook_name, self)

    @property
    def blocks(self) -> "BlocksView":
        return BlocksView(self)

    @property
    def after_final_norm(self) -> "HookView":
        return self["after_final_norm"]

    @property
    def before_final_norm(self) -> "HookView":
        return self["before_final_norm"]

    def hook_names(self) -> list:
        """All hook point names stored in this file (excludes metadata keys)."""
        return [k for k in self.data if not k.startswith("__")]

    # ------------------------------------------------------------------
    # Internal: entry access
    # ------------------------------------------------------------------

    def _entry(self, hook_name: str) -> dict:
        return self.data.get(hook_name, {})

    # ------------------------------------------------------------------
    # Internal: layer resolution for B derivation
    # ------------------------------------------------------------------

    def _get_layer(self, hook_name: str):
        if hook_name in self._layer_cache:
            return self._layer_cache[hook_name]
        if self.model is None or self.model_config is None:
            return None
        m = re.match(r"blk(\d+)\.(up|down|gate)", hook_name)
        if not m:
            return None
        from utils.model_registry import get_mlp_projections
        block_idx = int(m.group(1))
        for name, layer in get_mlp_projections(self.model, self.model_config, block_idx):
            self._layer_cache[name] = layer
        return self._layer_cache.get(hook_name)

    # ------------------------------------------------------------------
    # Internal: computation methods (used by FactorView properties)
    # ------------------------------------------------------------------

    def _eigenvalues(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            return self._B_eigenvalues(hook_name)

        entry = self._entry(hook_name)

        if f"{factor}_eigvals" in entry:
            return entry[f"{factor}_eigvals"].float()

        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor):
                if t.dim() == 2 and t.shape[0] == t.shape[1]:
                    n = entry.get(f"n_{factor}", entry.get("n", 1))
                    return torch.linalg.eigvalsh(t.float() / n).flip(0)
                elif t.dim() == 2:
                    from utils.powerlaw import get_eigenspectrum
                    return torch.from_numpy(get_eigenspectrum(t.numpy())).float()

        return None

    def _eigenvectors(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            return self._B_eigenvectors(hook_name)

        entry = self._entry(hook_name)

        if f"{factor}_eigvecs" in entry:
            return entry[f"{factor}_eigvecs"]

        cov = self._covariance(hook_name, factor)
        if cov is not None:
            eigvals, eigvecs = torch.linalg.eigh(cov.float())
            return eigvecs[:, eigvals.argsort(descending=True)]

        return None

    def _covariance(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        if factor == "B":
            return self._B_covariance(hook_name)

        entry = self._entry(hook_name)

        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get(f"n_{factor}", entry.get("n", 1))
                return t.float() / n

        vk, sk = f"{factor}_eigvecs", f"{factor}_eigvals"
        if vk in entry and sk in entry:
            V = entry[vk].float()
            S = entry[sk].float()
            return V @ torch.diag(S) @ V.T

        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2:
                X = t.float()
                mask = entry.get(f"{factor}_mask")
                if mask is not None:
                    X = X[mask.bool()]
                return (X.T @ X) / X.shape[0]

        return None

    def _activations(self, hook_name: str, factor: str, apply_mask: bool = True) -> Optional[torch.Tensor]:
        entry = self._entry(hook_name)

        raw = None
        if factor in entry:
            t = entry[factor]
            if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
                raw = t

        if raw is None:
            uk = f"{factor}_U"
            if uk in entry:
                U = entry[uk].float()
                S = entry[f"{factor}_S"].float()
                V = entry[f"{factor}_V"].float()
                raw = (U * S.unsqueeze(0)) @ V.T

        if raw is None:
            return None

        if apply_mask:
            mask = entry.get(f"{factor}_mask")
            if mask is not None:
                return raw[mask.bool()]
        return raw

    def _mean(self, hook_name: str, factor: str) -> Optional[torch.Tensor]:
        entry = self._entry(hook_name)
        return entry.get(f"{factor}_mean")

    def _svd(self, hook_name: str, factor: str):
        """(U, S, V) from acts if available, else (S, V) from eigdecomp."""
        acts = self._activations(hook_name, factor)
        if acts is not None:
            U, S, Vt = torch.linalg.svd(acts.float(), full_matrices=False)
            return U, S, Vt.T

        eigvals = self._eigenvalues(hook_name, factor)
        eigvecs = self._eigenvectors(hook_name, factor)
        if eigvals is not None and eigvecs is not None:
            entry = self._entry(hook_name)
            n = entry.get(f"n_{factor}", entry.get("n", 1))
            S = torch.from_numpy(np.sqrt(np.maximum(eigvals, 0) * n)).float()
            return S, eigvecs

        return None

    # ------------------------------------------------------------------
    # Internal: B derivation
    # ------------------------------------------------------------------

    def _B_covariance(self, hook_name: str) -> Optional[torch.Tensor]:
        entry = self._entry(hook_name)

        if "B" in entry:
            t = entry["B"]
            if t.dim() == 2 and t.shape[0] == t.shape[1]:
                n = entry.get("n_B", entry.get("n", 1))
                return t.float() / n

        if "B_eigvecs" in entry and "B_eigvals" in entry:
            V = entry["B_eigvecs"].float()
            S = entry["B_eigvals"].float()
            return V @ torch.diag(S) @ V.T

        A_cov = self._covariance(hook_name, "A")
        if A_cov is None:
            return None
        layer = self._get_layer(hook_name)
        if layer is None:
            return None

        W = layer.weight.detach().float()
        b = layer.bias.detach().float() if layer.bias is not None else None

        B_cov = W @ A_cov @ W.T
        if b is not None:
            mu = self._mean(hook_name, "A")
            if mu is None:
                raise ValueError(
                    f"Cannot derive B for '{hook_name}': layer has a bias but no A_mean is stored. "
                    f"Re-collect with a storage format that includes the '+m' modifier (e.g. cov_svd+m)."
                )
            Wmu = W @ mu.float()
            B_cov = B_cov + torch.outer(Wmu, b) + torch.outer(b, Wmu) + torch.outer(b, b)

        return B_cov

    def _B_eigenvalues(self, hook_name: str) -> Optional[torch.Tensor]:
        entry = self._entry(hook_name)
        if "B_eigvals" in entry:
            return entry["B_eigvals"].float()
        B_cov = self._B_covariance(hook_name)
        if B_cov is None:
            return None
        return torch.linalg.eigvalsh(B_cov).flip(0).clamp(min=0)

    def _B_eigenvectors(self, hook_name: str) -> Optional[torch.Tensor]:
        entry = self._entry(hook_name)
        if "B_eigvecs" in entry:
            return entry["B_eigvecs"]
        B_cov = self._B_covariance(hook_name)
        if B_cov is None:
            return None
        eigvals, eigvecs = torch.linalg.eigh(B_cov)
        return eigvecs[:, eigvals.argsort(descending=True)]


# ---------------------------------------------------------------------------
# View classes
# ---------------------------------------------------------------------------

class BlocksView:
    """acc.blocks[N] → BlockView."""

    def __init__(self, acc: DataAccessor):
        self._acc = acc

    def __getitem__(self, block_idx: int) -> "BlockView":
        return BlockView(block_idx, self._acc)


class BlockView:
    """acc.blocks[N].up / .down / .gate → HookView."""

    def __init__(self, block_idx: int, acc: DataAccessor):
        self._idx = block_idx
        self._acc = acc

    @property
    def up(self) -> "HookView":
        return self._acc[f"blk{self._idx}.up"]

    @property
    def down(self) -> "HookView":
        return self._acc[f"blk{self._idx}.down"]

    @property
    def gate(self) -> "HookView":
        return self._acc[f"blk{self._idx}.gate"]


# Residual-stream hook names: no weight matrix, so A = B (forward acts = output acts).
_RESIDUAL_HOOK_PREFIXES = ("after_final_norm", "before_final_norm", "post_attn_", "pre_block_")


def _is_residual_hook(hook_name: str) -> bool:
    return any(hook_name.startswith(p) for p in _RESIDUAL_HOOK_PREFIXES)


class HookView:
    """acc[hook_name] → HookView. Access factors via .A / .B / .G."""

    def __init__(self, hook_name: str, acc: DataAccessor):
        self._hook = hook_name
        self._acc = acc

    def _factor(self, key: str) -> "FactorView":
        return FactorView(self._hook, key, self._acc)

    @property
    def A(self) -> "FactorView":
        return self._factor("A")

    @property
    def B(self) -> "FactorView":
        # Residual hooks have no weight matrix — A and B are the same activations.
        if _is_residual_hook(self._hook):
            return self._factor("A")
        return self._factor("B")

    @property
    def G(self) -> "FactorView":
        return self._factor("G")

    # Aliases
    @property
    def input(self) -> "FactorView":
        return self.A

    @property
    def output(self) -> "FactorView":
        return self.B

    @property
    def grad(self) -> "FactorView":
        return self.G

    def __repr__(self):
        return f"HookView({self._hook!r})"


class FactorView:
    """acc[hook].A → FactorView. All properties are lazily computed."""

    def __init__(self, hook_name: str, factor: str, acc: DataAccessor):
        self._hook = hook_name
        self._factor = factor
        self._acc = acc

    @property
    def eigvals(self) -> Optional[torch.Tensor]:
        """Eigenvalues as 1D float tensor, descending order."""
        return self._acc._eigenvalues(self._hook, self._factor)

    @property
    def eigvecs(self) -> Optional[torch.Tensor]:
        """Eigenvectors as columns (d, k), descending order."""
        return self._acc._eigenvectors(self._hook, self._factor)

    @property
    def cov(self) -> Optional[torch.Tensor]:
        """Normalized covariance E[xx^T] as (d, d) float tensor."""
        return self._acc._covariance(self._hook, self._factor)

    @property
    def acts(self) -> Optional[torch.Tensor]:
        """Selected activations (N, d). Applies stored mask if present."""
        return self._acc._activations(self._hook, self._factor)

    @property
    def acts_raw(self) -> Optional[torch.Tensor]:
        """All activations (N_total, d) without applying stored mask."""
        return self._acc._activations(self._hook, self._factor, apply_mask=False)

    @property
    def mean(self) -> Optional[torch.Tensor]:
        """Mean activation vector (d,) if stored via +m modifier."""
        return self._acc._mean(self._hook, self._factor)

    @property
    def eigh(self) -> Optional[tuple]:
        """(eigvals: Tensor, eigvecs: Tensor) — sorted descending."""
        ev = self.eigvals
        V = self.eigvecs
        if ev is None or V is None:
            return None
        return ev, V

    @property
    def svd(self):
        """(U, S, V) from acts if available, else (S, V) from eigdecomp."""
        return self._acc._svd(self._hook, self._factor)

    @property
    def n(self) -> Optional[int]:
        """Number of tokens/samples this factor was accumulated over.

        For covariance formats: reads stored n_{factor} or n count.
        For raw acts (N, d): reads from tensor shape.
        """
        entry = self._acc._entry(self._hook)
        v = entry.get(f"n_{self._factor}") or entry.get("n")
        if v is not None:
            return int(v)
        # Fallback: raw acts tensor shape
        t = entry.get(self._factor)
        if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[0] != t.shape[1]:
            return int(t.shape[0])
        return None

    def __repr__(self):
        return f"FactorView({self._hook!r}, {self._factor!r})"
