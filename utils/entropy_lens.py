"""Vocabulary-entropy lens: per-layer Shannon entropy of the next-token distribution.

Port of the attention-experiments entropy experiments: at every depth, push the
residual stream through the model's own final norm + unembedding and measure the
entropy of softmax(logits) over the vocabulary, averaged over tokens. Unlike the
original (which sampled generations from BOS), this runs teacher-forced on the
collection batches, so it piggybacks on collect.py's forward passes for free.

Layers: "emb" (the residual entering block 0) then "blk0".."blk{L-1}" (the residual
after each block). Streaming token-mean; logits are computed in fp32 in token
chunks so the (tokens x vocab) intermediate stays small.

Alongside the entropy, the same distribution's alphaReQ: the next-token probabilities
SORTED descending are already a spectrum over vocabulary ranks (no covariance, no
unembedding SVD — the token identities are irrelevant), so the weighted log-log slope
over ranks [k0, k1) is defined per token and averaged the same streaming way. Only the
top k1 probabilities are needed, so this is a topk, not a full sort.
"""

import numpy as np
import torch

from utils.model_registry import get_output_head, get_num_layers, _fam, _getattr_path


def _alpha_req(log_p, k0=11, k1=100):
    """Per-row alphaReQ of the next-token distribution: the 1/r-weighted log-log slope of the
    SORTED probabilities over ranks [k0, k1) (0-based), the same fit as
    analysis.experiments_lib._alpha but on probabilities instead of eigenvalues, batched over
    tokens. log_p is (tokens, vocab) log-probabilities; returns (tokens,). The window is
    clipped to the vocabulary, and a vocab too small to fit a 2-point fit gives NaN."""
    k1 = min(k1, log_p.shape[-1])
    if k1 - k0 < 2:
        return log_p.new_full((log_p.shape[0],), float("nan"))
    y = log_p.topk(k1, dim=-1).values[:, k0:]                       # log p, descending
    r = torch.arange(k0 + 1, k1 + 1, device=log_p.device, dtype=y.dtype)
    u, w = -r.log(), 1.0 / r                                        # regressor, 1/rank weights
    sw, su, suu = w.sum(), (w * u).sum(), (w * u * u).sum()
    sy = (y * w).sum(-1)
    suy = (y * w * u).sum(-1)
    return (sw * suy - su * sy) / (sw * suu - su * su)


class EntropyLens:
    """Attaches on construction; call close() to detach and get the result dict."""

    def __init__(self, model, config, chunk_tokens=2048, alpha_window=(11, 100)):
        self.head = get_output_head(model, config)
        self.chunk_tokens = chunk_tokens
        self.alpha_window = tuple(alpha_window)
        n_layers = get_num_layers(model, config)
        self.layers = ["emb"] + [f"blk{i}" for i in range(n_layers)]
        self.sums = {name: 0.0 for name in self.layers}
        self.alpha_sums = {name: 0.0 for name in self.layers}
        self.counts = {name: 0 for name in self.layers}
        self.handles = []

        blocks = _getattr_path(model, _fam(config).blocks)
        self.handles.append(blocks[0].register_forward_pre_hook(
            self._pre_hook("emb"), with_kwargs=True))
        for i, block in enumerate(blocks):
            self.handles.append(block.register_forward_hook(self._post_hook(f"blk{i}")))

    def _pre_hook(self, name):
        def hook(module, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            self._accumulate(name, h)
        return hook

    def _post_hook(self, name):
        def hook(module, inputs, output):
            self._accumulate(name, output[0] if isinstance(output, tuple) else output)
        return hook

    @torch.no_grad()
    def _accumulate(self, name, h):
        if h is None:
            return
        flat = h.detach().reshape(-1, h.shape[-1])
        for start in range(0, flat.shape[0], self.chunk_tokens):
            chunk = flat[start:start + self.chunk_tokens]
            log_p = torch.log_softmax(self.head(chunk).float(), dim=-1)
            entropy = -(log_p.exp() * log_p).sum(-1)
            self.sums[name] += entropy.sum().item()
            self.alpha_sums[name] += _alpha_req(log_p, *self.alpha_window).sum().item()
            self.counts[name] += entropy.shape[0]

    def close(self):
        """Detach hooks; return {"layers", "per_layer", "n_tokens"} (numpy, results-file ready)."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        per_layer = np.array([self.sums[n] / max(self.counts[n], 1) for n in self.layers])
        alpha_per_layer = np.array([self.alpha_sums[n] / max(self.counts[n], 1)
                                    for n in self.layers])
        return {
            "layers": list(self.layers),
            "per_layer": per_layer,
            "alpha_per_layer": alpha_per_layer,
            "alpha_window": list(self.alpha_window),
            "n_tokens": int(self.counts[self.layers[-1]]),
        }
