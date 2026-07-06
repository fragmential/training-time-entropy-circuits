"""Vendored nanochat GPT (Karpathy nanochat fork, ~/personal_project/attention-experiments).

Collection-only port: KV cache, optimizers, and generation are stripped; the
functional RMSNorms become param-free modules (ln1/ln2/final_norm) so the hook
machinery has capture points. State-dict keys are unchanged (the norms carry no
params), so training checkpoints load with strict=True.

Architecture: pre-norm sequential blocks, RoPE, QK-norm, GQA-capable attention,
gateless ReLU^2 MLP, untied embeddings, logit softcap, param-free RMSNorm.

Checkpoints live under $NANOCHAT_DIR (default /projects/prjs1815/nanochat) as
written by nanochat's checkpoint_manager: base_checkpoints/<tag>/model_{step:06d}.pt
+ meta_{step:06d}.json. Model names are "nanochat-<tag>", revisions "step<N>".
"""

import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

NANOCHAT_DIR = os.environ.get("NANOCHAT_DIR", "/projects/prjs1815/nanochat")


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6      # query heads
    n_kv_head: int = 6   # key/value heads (GQA when < n_head)
    n_embd: int = 768
    use_cache: bool = False  # HF-interface shim; nanochat has no KV cache here

    # HF-style aliases so family-agnostic code (hook_specs, collect) can read them
    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def head_dim(self):
        return self.n_embd // self.n_head

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_hidden_layers(self):
        return self.n_layer


class RMSNorm(nn.Module):
    """Param-free RMSNorm as a module, purely so hooks can attach to it."""
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_head != self.n_kv_head)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm()   # pre-attn norm; its input is the residual into the block
        self.ln2 = RMSNorm()   # pre-mlp norm; its input is the mid-block residual
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(self.ln1(x), cos_sin)
        x = x + self.mlp(self.ln2(x))
        return x


class _Trunk(nn.Module):
    """The `transformer.{wte,h}` subtree — a typed stand-in for nanochat's ModuleDict
    (same state-dict keys, but attribute access that survives static analysis)."""
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])


class GPT(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = _Trunk(config)
        self.final_norm = RMSNorm()
        self.lm_head: nn.Module = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # swapped for Identity by fast_final_norm
        cos, sin = self._precompute_rotary(config.sequence_len * 10, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, input_ids, attention_mask=None, **kwargs):
        assert attention_mask is None or bool(attention_mask.all()), (
            "nanochat attention has no padding-mask support; collect with packing=packed"
        )
        T = input_ids.size(1)
        assert T <= self.cos.size(1), f"sequence length {T} exceeds rotary cache {self.cos.size(1)}"
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if not isinstance(self.lm_head, nn.Identity):   # fast_final_norm: skip softcap
            if logits.requires_grad:
                logits = 15 * torch.tanh(logits / 15)
            else:
                # in-place: the (B,T,65536) logits tensor is huge; out-of-place triples peak VRAM
                logits = torch.tanh_(logits.div_(15)).mul_(15)
        return SimpleNamespace(logits=logits)

    # --- HF-interface shims used by collect.py's grad path ---
    def gradient_checkpointing_enable(self, **kwargs):
        pass  # 185M params; never worth the recompute

    def enable_input_require_grads(self):
        def hook(module, inputs, output):
            output.requires_grad_(True)
        self.transformer.wte.register_forward_hook(hook)


# ---------------------------------------------------------------------------
# Tokenizer: nanochat's rustbpe-trained tokenizer.pkl is a pickled tiktoken
# Encoding — only tiktoken is needed to load it. This adapter exposes the small
# HF-tokenizer surface that data_utils/collect actually call.
# ---------------------------------------------------------------------------

class NanochatTokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token("<|bos|>")
        # <|bos|> is nanochat's document delimiter — it plays eos's boundary role here
        self.eos_token_id = self.bos_token_id
        self.pad_token_id = self.bos_token_id
        self.eos_token = self.pad_token = "<|bos|>"
        self.vocab_size = enc.n_vocab

    @classmethod
    def from_base_dir(cls, base_dir):
        with open(os.path.join(base_dir, "tokenizer", "tokenizer.pkl"), "rb") as f:
            return cls(pickle.load(f))

    def __call__(self, text, add_special_tokens=True, padding=None,
                 return_tensors=None, max_length=None, truncation=False):
        texts = [text] if isinstance(text, str) else list(text)
        batch = self.enc.encode_ordinary_batch(texts)
        if add_special_tokens:
            batch = [[self.bos_token_id] + ids for ids in batch]
        if truncation and max_length:
            batch = [ids[:max_length] for ids in batch]
        if return_tensors == "pt":
            longest = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), longest), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), longest), dtype=torch.long)
            for i, ids in enumerate(batch):
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :len(ids)] = 1
            return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)
        ids = batch[0] if isinstance(text, str) else batch
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids):
        return self.enc.decode(list(ids))


# ---------------------------------------------------------------------------
# Checkpoint discovery + loading (local dirs, no HF hub)
# ---------------------------------------------------------------------------

_MODEL_FILE = re.compile(r"model_(\d{6})\.pt$")


def _ckpt_dir(config):
    tag = config.hf_repo.split("/")[-1].removeprefix("nanochat-")
    return os.path.join(NANOCHAT_DIR, "base_checkpoints", tag)


def checkpoint_steps(config):
    """Sorted step numbers with a model_{step:06d}.pt in the checkpoint dir."""
    d = _ckpt_dir(config)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"No nanochat checkpoint dir at {d} (set NANOCHAT_DIR?)")
    steps = sorted(int(m.group(1)) for f in os.listdir(d) if (m := _MODEL_FILE.match(f)))
    if not steps:
        raise ValueError(f"No model_*.pt checkpoints in {d}")
    return steps


def _load_state_dict(config, revision):
    step = int(revision.removeprefix("step"))
    path = os.path.join(_ckpt_dir(config), f"model_{step:06d}.pt")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()}


def load_nanochat_model(config, revision):
    step = int(revision.removeprefix("step"))
    meta_path = os.path.join(_ckpt_dir(config), f"meta_{step:06d}.json")
    with open(meta_path) as f:
        meta = json.load(f)
    model = GPT(GPTConfig(**meta["model_config"]))
    model.load_state_dict(_load_state_dict(config, revision), strict=True)
    return model.to(getattr(torch, config.dtype))


def load_nanochat_tokenizer(config):
    return NanochatTokenizer.from_base_dir(NANOCHAT_DIR)


def load_sd_tensors(config, revision, keys):
    """Selective-tensor lookup against a local checkpoint (weight-cache miss path)."""
    sd = _load_state_dict(config, revision)
    return {k: sd[k] for k in keys if k in sd}
