"""Model-specific logic for checkpoint discovery, model loading, tokenizer setup, and HF cache management."""
from __future__ import annotations
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Protocol
import numpy as np
import torch

WeightBias = tuple[torch.Tensor, torch.Tensor | None] | None   # (weight, bias) or None
NormFn = Callable[[torch.Tensor], torch.Tensor] | None          # the final-norm callable

# Repo root, so revisions_file paths resolve regardless of the caller's cwd.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ModelConfig:
    family: str
    hf_repo: str          # actual HF repo to download from
    model_class: str      # "GPTNeoXForCausalLM" or "AutoModelForCausalLM"
    dtype: str            # "float16" or "bfloat16"
    trust_remote_code: bool
    pad_token_from_eos: bool
    training_dataset: str | None = None      # dataset name for "native" resolution
    revisions_file: str | None = None        # path to file mapping step→revision hash
    early_training_model: str | None = None  # separate HF repo for early checkpoints


# Per-model overrides for fields that differ within a family.
_OLMO_OVERRIDES = {
    "allenai/OLMo-2-0425-1B": {
        "revisions_file": "utils/revisions/1b_revisions.txt",
        "early_training_model": "allenai/OLMo-2-0425-1B-early-training",
    },
    "allenai/OLMo-2-1124-7B": {
        "revisions_file": "utils/revisions/7b_revisions.txt",
    },
}


def get_model_config(model_name: str) -> ModelConfig:
    name = model_name.lower()
    if "pythia" in name:
        return ModelConfig(
            family="pythia",
            hf_repo=model_name,
            model_class="GPTNeoXForCausalLM",
            dtype="float16",
            trust_remote_code=False,
            pad_token_from_eos=True,
            training_dataset="pile_deduped_eleutherai",
        )
    if "olmo" in name:
        # match by basename so the short name (e.g. "OLMo-2-0425-1B", as used in
        # result filenames) resolves the same overrides as the full repo path
        short = model_name.split("/")[-1]
        overrides = next((v for k, v in _OLMO_OVERRIDES.items() if k.split("/")[-1] == short), {})
        return ModelConfig(
            family="olmo",
            hf_repo=model_name,
            model_class="AutoModelForCausalLM",
            dtype="bfloat16",
            trust_remote_code=True,
            pad_token_from_eos=False,
            training_dataset="olmo_mix",
            **overrides,
        )
    if "nanochat" in name:
        return ModelConfig(
            family="nanochat",
            hf_repo=model_name,          # local tag (e.g. "nanochat-d12"); nothing is downloaded
            model_class="NanochatGPT",
            dtype="bfloat16",
            trust_remote_code=False,
            pad_token_from_eos=False,
            training_dataset="fineweb_edu_100b",
        )
    raise ValueError(f"Unknown model family for '{model_name}'. Expected 'pythia', 'olmo', or 'nanochat' in the name.")


# --- Checkpoint discovery ---

def _subsample(items, max_n, spacing="linear"):
    if not max_n or len(items) <= max_n:
        return items
    n = len(items)
    if spacing == "log":
        t = np.geomspace(1, n, max_n) - 1
    elif spacing == "sqrt":
        t = np.linspace(0, np.sqrt(n - 1), max_n) ** 2
    else:
        t = np.linspace(0, n - 1, max_n)
    indices = np.unique(np.round(t).astype(int))
    # If rounding lost slots, fill gaps with linear interpolation
    while len(indices) < max_n and len(indices) < n:
        gaps = np.diff(indices)
        biggest = np.argmax(gaps)
        mid = (indices[biggest] + indices[biggest + 1]) // 2
        indices = np.sort(np.append(indices, mid))
    return [items[i] for i in indices]


def _pythia_checkpoints(config, max_checkpoints, spacing="linear"):
    early = [0, 8, 16, 32, 64, 128, 256, 512]
    later = list(np.arange(1000, 143_001, 1000))
    return [(s, f"step{s}", config.hf_repo) for s in early + _subsample(later, max_checkpoints, spacing)]


def _olmo_checkpoints(config, max_checkpoints, spacing="linear"):
    if config.revisions_file is None:
        raise ValueError(f"OLMo model '{config.hf_repo}' has no revisions_file configured in model_registry.")
    checkpoint_map = _read_revisions_file(config.revisions_file)
    all_steps = sorted(checkpoint_map.keys())
    early = sorted(s for s in all_steps if s <= 10_000)
    later = sorted(s for s in all_steps if s > 10_000 and s % 10_000 == 0)
    later = _subsample(later, max_checkpoints, spacing)
    schedule = []
    for s in early + later:
        rev = checkpoint_map[s]
        m = config.hf_repo
        if config.early_training_model and 1000 <= s <= 10_000:
            m = config.early_training_model
        schedule.append((s, rev, m))
    return schedule


def _nanochat_checkpoints(config, max_checkpoints, spacing="linear"):
    from utils.nanochat_gpt import checkpoint_steps
    steps = checkpoint_steps(config)
    early = [s for s in steps if s <= 512]
    later = _subsample([s for s in steps if s > 512], max_checkpoints, spacing)
    return [(s, f"step{s}", config.hf_repo) for s in early + later]


def _read_revisions_file(filepath: str) -> dict:
    """{step: revision_str} from a revisions file. Path resolves relative to the
    repo root, so cwd doesn't matter; a missing/empty file raises (no silent {})."""
    path = filepath if os.path.isabs(filepath) else os.path.join(_REPO_ROOT, filepath)
    checkpoint_map = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "step" in line and "-tokens" in line:
                checkpoint_map[int(line.split("-tokens")[0].split("step")[-1])] = line
    if not checkpoint_map:
        raise ValueError(f"No checkpoints parsed from {path}")
    return checkpoint_map


@dataclass
class _Family:
    """Every architecture fact that varies by model family, in one place. `blocks` /
    `final_norm` / `oproj` double as live-module dotted paths AND state-dict prefixes."""
    schedule: Callable      # (config, max_checkpoints, spacing) -> [(step, rev, repo), ...]
    blocks: str             # dotted path to the block list (e.g. "gpt_neox.layers")
    final_norm: str         # dotted path / sd-prefix of the final norm
    norm_kind: str          # "layernorm" | "rmsnorm" | "rmsnorm_bare" (no learnable params)
    mlp_projs: dict         # {leaf-name: submodule-name under block.mlp}
    mlp_bias: bool          # MLP projections carry a bias
    oproj: str              # attention output-proj path under a block
    boundaries: tuple       # ((leaf-suffix, submodule-path-under-block, capture), ...)
    head: str = "lm_head"   # dotted path to the unembedding (Pythia: "embed_out")
    logit_softcap: "float | None" = None   # tanh softcap applied after the head (nanochat: 15)


_FAMILY = {
    "pythia": _Family(
        schedule=_pythia_checkpoints,
        blocks="gpt_neox.layers",
        final_norm="gpt_neox.final_layer_norm",
        norm_kind="layernorm",
        mlp_projs={"up": "dense_h_to_4h", "down": "dense_4h_to_h"},
        mlp_bias=True,
        oproj="attention.dense",
        head="embed_out",
        boundaries=(
            ("attn.in",  "input_layernorm", "input"),
            ("attn.out", "attention",       "output"),
            ("mlp.out",  "mlp",             "output"),
        ),
    ),
    "olmo": _Family(
        schedule=_olmo_checkpoints,
        blocks="model.layers",
        final_norm="model.norm",
        norm_kind="rmsnorm",
        mlp_projs={"gate": "gate_proj", "up": "up_proj", "down": "down_proj"},
        mlp_bias=False,
        oproj="self_attn.o_proj",
        boundaries=(
            ("attn.in",      "self_attn",                  "input"),
            ("attn.raw_out", "self_attn",                  "output"),
            ("attn.out",     "post_attention_layernorm",   "output"),
            ("mlp.in",       "mlp",                        "input"),
            ("mlp.raw_out",  "mlp",                        "output"),
            ("mlp.out",      "post_feedforward_layernorm", "output"),
        ),
    ),
    "nanochat": _Family(
        schedule=_nanochat_checkpoints,
        blocks="transformer.h",
        final_norm="final_norm",
        norm_kind="rmsnorm_bare",   # param-free RMSNorm: no weight in the state dict
        mlp_projs={"up": "c_fc", "down": "c_proj"},
        mlp_bias=False,
        oproj="attn.c_proj",
        logit_softcap=15.0,
        boundaries=(
            # pre-norm sequential: x = x + attn(ln1(x)); x = x + mlp(ln2(x))
            ("attn.in",  "ln1",  "input"),
            ("attn.out", "attn", "output"),
            ("mlp.in",   "ln2",  "input"),
            ("mlp.out",  "mlp",  "output"),
        ),
    ),
}


def _fam(config) -> "_Family":
    return _FAMILY[config.family]


def _getattr_path(obj, dotted):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def get_checkpoint_schedule(config, max_checkpoints: int | None = 50, checkpoint_spacing="linear"):
    """Return list of (step_num, revision_str, hf_model_name) tuples."""
    return _fam(config).schedule(config, max_checkpoints, checkpoint_spacing)


# --- Model loading ---

def load_model(config, hf_repo, revision):
    """Load model on CPU with appropriate class and dtype for the model family."""
    if config.model_class == "NanochatGPT":
        from utils.nanochat_gpt import load_nanochat_model
        return load_nanochat_model(config, revision)
    if config.model_class == "GPTNeoXForCausalLM":
        from transformers import GPTNeoXForCausalLM
        return GPTNeoXForCausalLM.from_pretrained(hf_repo, revision=revision, torch_dtype=config.dtype)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        hf_repo, revision=revision, torch_dtype=config.dtype, trust_remote_code=config.trust_remote_code
    )


def load_tokenizer(config, revision=None):
    """Load tokenizer with appropriate settings for the model family."""
    if config.model_class == "NanochatGPT":
        from utils.nanochat_gpt import load_nanochat_tokenizer
        return load_nanochat_tokenizer(config)
    from transformers import AutoTokenizer
    kwargs = {}
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if revision:
        kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(config.hf_repo, **kwargs)
    if config.pad_token_from_eos and tok.eos_token is None and revision:
        tok = AutoTokenizer.from_pretrained(config.hf_repo, **{k: v for k, v in kwargs.items() if k != "revision"})
    if config.pad_token_from_eos and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --- Architecture helpers ---

def get_mlp_projections(model, config, block_idx):
    """Return [(name, nn.Linear), ...] for MLP projections in the given block."""
    fam = _fam(config)
    block = _getattr_path(model, fam.blocks)[block_idx]
    return [(f"blk{block_idx}.mlp.{leaf}", getattr(block.mlp, sub))
            for leaf, sub in fam.mlp_projs.items()]


def get_num_layers(model, config):
    """Return number of transformer blocks."""
    return len(_getattr_path(model, _fam(config).blocks))


def get_final_layernorm(model, config):
    """Return the final layernorm module (before the lm_head).

    Pythia: model.gpt_neox.final_layer_norm (nn.LayerNorm)
    OLMo:   model.model.norm (RMSNorm)
    """
    return _getattr_path(model, _fam(config).final_norm)


def get_block_boundary_hooks(model, config, block_idx):
    """Return [(name, module, capture), ...] for block-boundary residual-stream hooks.

    Captures the residual flowing into each sub-block (.in) and the contribution
    being added back (.out). For OLMo-2, also exposes .raw_out — the raw attention
    output before the post-norm — which has no Pythia analogue (Pythia has no
    post-norm on attention output).

    Pythia (GPTNeoX, parallel-residual pre-norm):
        attn_out = attention(input_layernorm(x))
        mlp_out  = mlp(post_attention_layernorm(x))       # same input as attention branch
        x_next   = x + attn_out + mlp_out
      → only 3 distinct tensors: x (= attn.in == mlp.in), attn_out, mlp_out

    OLMo-2 (post-norm sequential, no input_layernorm):
        attn_raw  = self_attn(x)
        attn_out  = post_attention_layernorm(attn_raw)
        x_mid     = x + attn_out
        mlp_raw   = mlp(x_mid)                             # mlp.raw_out (= down-proj output)
        mlp_out   = post_feedforward_layernorm(mlp_raw)
        x_next    = x_mid + mlp_out
      → 6 distinct tensors (raw_out captures the pre-post-norm side of each sub-block).
    """
    fam = _fam(config)
    block = _getattr_path(model, fam.blocks)[block_idx]
    return [(f"blk{block_idx}.{suf}", _getattr_path(block, sub), cap)
            for suf, sub, cap in fam.boundaries]


def get_output_head(model, config):
    """Return a callable h -> next-token logits through the model's own final norm +
    unembedding (+ logit softcap where the family applies one). Binds the current
    module references, so it stays correct even if fast_final_norm later swaps the
    head attribute for Identity."""
    fam = _fam(config)
    norm = _getattr_path(model, fam.final_norm)
    head = _getattr_path(model, fam.head)
    cap = fam.logit_softcap
    if cap is None:
        return lambda h: head(norm(h))
    return lambda h: cap * torch.tanh(head(norm(h)) / cap)


def get_attention_output_proj(model, config, block_idx):
    """Return (name, o_proj_module, num_heads, head_dim) for the attention output projection.

    Used by per-OV-head decomposition: o_proj is always invoked on a
    (B, T, num_attention_heads * head_dim) tensor regardless of the attention
    backend (eager / SDPA / flash), so per-head contributions can be recovered
    by slicing the input + the corresponding columns of o_proj.weight.

    For OLMo-2 with GQA, num_heads is the query-head count — V is already
    repeated via repeat_kv before reaching o_proj.
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    fam = _fam(config)
    block = _getattr_path(model, fam.blocks)[block_idx]
    return fam.oproj.split(".")[-1], _getattr_path(block, fam.oproj), n_heads, head_dim


# Derivation graph: a derived (leaf, quantity) <- a source (leaf, quantity) via a
# Transform the accessor applies. The only place that knows derivation edges + weight
# state-dict locations.

MLP_OUT = re.compile(r"(.*)\.mlp\.(up|down|gate)\.out$")
HEAD_CONTRIB = re.compile(r"(.*)\.attn\.head(\d+)\.contrib$")


def _proj_sd_prefix(config, idx, proj):
    fam = _fam(config)
    return f"{fam.blocks}.{idx}.mlp.{fam.mlp_projs[proj]}"


def _oproj_sd_prefix(config, idx):
    fam = _fam(config)
    return f"{fam.blocks}.{idx}.{fam.oproj}"


def _blk_idx(prefix):
    m = re.search(r"blk(\d+)", prefix)
    return m.group(1) if m else None


Recipes = dict[str, list[tuple[tuple[str, ...], Callable]]]   # component -> [(source components, fn)]


class WeightProvider(Protocol):
    """Minimal surface a Transform derives through: weights by sd-prefix, the final norm."""
    def weight(self, sd_prefix: str) -> WeightBias: ...
    def norm(self) -> NormFn: ...


def _cov_proj(W: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    return W @ C @ W.T

def _cov_proj_bias(W: torch.Tensor, b: torch.Tensor, C: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    Wm = W @ mu
    return W @ C @ W.T + Wm[:, None] * b[None, :] + b[:, None] * Wm[None, :] + b[:, None] * b[None, :]


class Transform(ABC):
    """One derivation step. `ingredients(weights)` binds the weight values it needs (None if
    unavailable); `recipes()` gives, per output component, its source components + build fn."""
    kind: str

    @abstractmethod
    def ingredients(self, wp: WeightProvider) -> tuple | None: ...

    @abstractmethod
    def recipes(self) -> Recipes: ...


class Linear(Transform):
    """Weight matrix (optionally a head column-slice, optionally +bias). `recipes()` gives,
    per output component, its source components + the function that builds it."""
    kind = "linear"

    def __init__(self, sd_prefix: str, head: int | None = None, bias: bool = False) -> None:
        self.sd_prefix, self.head, self.bias = sd_prefix, head, bias

    def ingredients(self, wp: WeightProvider) -> WeightBias:
        return wp.weight(self.sd_prefix)

    def recipes(self) -> Recipes:
        h = self.head
        sl = (lambda W, d: W[:, h * d:(h + 1) * d]) if h is not None else (lambda W, d: W)
        if self.bias:
            return {
                "samples": [(("samples",),    lambda W, b, X:  X.float() @ sl(W, X.shape[-1]).T + b)],
                "mean":    [(("mean",),       lambda W, b, mu: sl(W, mu.shape[0]) @ mu + b)],
                "cov":     [(("cov", "mean"), lambda W, b, C, mu: _cov_proj_bias(sl(W, C.shape[0]), b, C, mu))],
                "n":       [(("n",),          lambda W, b, n:  n)],
            }
        return {
            "samples": [(("samples",), lambda W, b, X:  X.float() @ sl(W, X.shape[-1]).T)],
            "mean":    [(("mean",),    lambda W, b, mu: sl(W, mu.shape[0]) @ mu)],
            "cov":     [(("cov",),     lambda W, b, C:  _cov_proj(sl(W, C.shape[0]), C))],
            "n":       [(("n",),       lambda W, b, n:  n)],
        }


class Norm(Transform):
    """Nonlinear norm module: produces only `samples` (cov/eigvals reached via conversion)."""
    kind = "norm"

    def __init__(self, sd_prefix: str) -> None:
        self.sd_prefix = sd_prefix

    def ingredients(self, wp: WeightProvider) -> tuple[Callable] | None:
        f = wp.norm()
        return (f,) if f is not None else None

    def recipes(self) -> Recipes:
        return {"samples": [(("samples",), lambda f, X: f(X.float()))],
                "n":       [(("n",),       lambda f, n: n)]}


# Each rule: (leaf_regex, build_source_leaf(match)->str, src_quantity, build_Transform(config,match))
_DERIVATIONS = [
    (MLP_OUT,
     lambda m: f"{m.group(1)}.mlp.{m.group(2)}.in",
     "acts",
     lambda cfg, m: Linear(_proj_sd_prefix(cfg, _blk_idx(m.group(1)), m.group(2)),
                           bias=_fam(cfg).mlp_bias)),
    (HEAD_CONTRIB,
     lambda m: f"{m.group(1)}.attn.head{m.group(2)}.slice",
     "acts",
     lambda cfg, m: Linear(_oproj_sd_prefix(cfg, _blk_idx(m.group(1))), head=int(m.group(2)))),
    (re.compile(r"after_final_norm$"),
     lambda m: "before_final_norm",
     "acts",
     lambda cfg, m: Norm(_norm_sd_prefix(cfg))),
]


def derivation(config: ModelConfig, leaf: str, quantity: str) -> tuple[str, str, Transform] | None:
    """(src_leaf, src_quantity, Transform) for a derivable (leaf, quantity), else None."""
    for pat, src_of, src_q, make_T in _DERIVATIONS:
        if quantity == src_q and (m := pat.match(leaf)):
            return src_of(m), src_q, make_T(config, m)
    return None


# Forward of _DERIVATIONS, per (leaf, quantity): a stored (src_leaf, src_q) slot -> the
# (leaf, q) slot it feeds. (Duplicates _DERIVATIONS' edges; "option 3" in the plan collapses them.)
_PRODUCES = [
    (re.compile(r"(.*)\.attn\.head(\d+)\.slice$"), "acts", lambda m: (f"{m[1]}.attn.head{m[2]}.contrib", "acts")),
    (re.compile(r"(.*)\.mlp\.(up|down|gate)\.in$"), "acts", lambda m: (f"{m[1]}.mlp.{m[2]}.out", "acts")),
    (re.compile(r"^before_final_norm$"),            "acts", lambda m: ("after_final_norm", "acts")),
]


def derivable_slots(stored: set[tuple[str, str]]) -> set[tuple[str, str]]:
    """(leaf, q) slots derivable in one step from a stored (src_leaf, src_q), not themselves stored."""
    return {prod(m) for (leaf, q) in stored
            for pat, src_q, prod in _PRODUCES if q == src_q and (m := pat.match(leaf))} - stored


def is_derivable(leaf: str, q: str) -> bool:
    """A derivation rule exists for (leaf, q) — structural, config-free."""
    return any(q == src_q and pat.match(leaf) for pat, _, src_q, _ in _DERIVATIONS)


def _norm_sd_prefix(config):
    """State dict key prefix for the final layer norm."""
    return _fam(config).final_norm


def _load_selective_tensors(snap_dir, keys):
    """Load specific tensor keys from model files in a snapshot directory.

    Tries safetensors first, falls back to pytorch .bin format.
    """
    import json

    # --- Try safetensors ---
    index_path = os.path.join(snap_dir, "model.safetensors.index.json")
    single_st = os.path.join(snap_dir, "model.safetensors")
    if os.path.exists(index_path) or os.path.exists(single_st):
        from safetensors import safe_open
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            shards = {}
            for k in keys:
                if k in weight_map:
                    shards.setdefault(weight_map[k], []).append(k)
        else:
            shards = {"model.safetensors": list(keys)}

        result = {}
        for shard_name, shard_keys in shards.items():
            with safe_open(os.path.join(snap_dir, shard_name), framework="pt") as f:
                avail = set(f.keys())
                for k in shard_keys:
                    if k in avail:
                        result[k] = f.get_tensor(k)
        return result

    # --- Fallback: pytorch .bin ---
    import torch
    bin_index = os.path.join(snap_dir, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index):
        with open(bin_index) as f:
            weight_map = json.load(f)["weight_map"]
        shards = {}
        for k in keys:
            if k in weight_map:
                shards.setdefault(weight_map[k], []).append(k)
    else:
        shards = {"pytorch_model.bin": list(keys)}

    result = {}
    for shard_name, shard_keys in shards.items():
        path = os.path.join(snap_dir, shard_name)
        if not os.path.exists(path):
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except RuntimeError:
            state = torch.load(path, map_location="cpu", weights_only=False)
        for k in shard_keys:
            if k in state:
                result[k] = state[k]
    return result


class _RMSNorm:
    """Minimal callable RMSNorm (no nn.Module overhead). weight=None → param-free."""
    def __init__(self, weight, eps=1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        y = x * (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()
        return y if self.weight is None else y * self.weight.float()

    def float(self):
        return _RMSNorm(None if self.weight is None else self.weight.float(), self.eps)


# --- Weight cache (single-tensor on-disk cache for derived factors) ---

def _weight_cache_root():
    return os.environ.get("WEIGHT_CACHE_DIR", os.path.join("data", "weight_cache"))


def _model_short_name(hf_repo):
    return hf_repo.split("/", 1)[-1]


def _cache_path(family, hf_repo, revision, sd_key):
    return os.path.join(
        _weight_cache_root(), family, _model_short_name(hf_repo), str(revision), f"{sd_key}.pt"
    )


def _cache_lookup(family, hf_repo, revision, sd_key):
    import torch
    path = _cache_path(family, hf_repo, revision, sd_key)
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None  # corrupt → treat as miss; will be rewritten


def _cache_store(family, hf_repo, revision, sd_key, tensor):
    import torch
    path = _cache_path(family, hf_repo, revision, sd_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    try:
        torch.save(tensor, tmp)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _get_cached_snap_dir(hf_repo, revision):
    """Return snapshot directory path if already cached locally, else None."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", f"models--{hf_repo.replace('/', '--')}")
    # revision might already be a commit hash
    snap_dir = os.path.join(model_dir, "snapshots", revision)
    if os.path.isdir(snap_dir):
        return snap_dir
    # revision is a branch/tag — resolve via refs
    ref_path = os.path.join(model_dir, "refs", revision)
    if os.path.isfile(ref_path):
        with open(ref_path) as f:
            commit_hash = f.read().strip()
        snap_dir = os.path.join(model_dir, "snapshots", commit_hash)
        if os.path.isdir(snap_dir):
            return snap_dir
    return None


def load_selective_weights(config, hf_repo, revision, sd_prefixes, need_norm=False):
    """Load only the weight tensors at the given state-dict prefixes.

    Returns dict keyed BY STATE-DICT PREFIX:
        {sd_prefix: SimpleNamespace(weight=Tensor, bias=Tensor|None), ...}
        If need_norm: "__norm__" key with a callable norm (supports .float()).
    The accessor's _weight(sd_prefix) looks up by prefix; derivation() supplies the
    prefixes (see derivation_sd_prefixes).
    """
    all_keys = set()
    for prefix in sd_prefixes:
        all_keys.add(f"{prefix}.weight")
        all_keys.add(f"{prefix}.bias")

    norm_prefix = None
    if need_norm and _fam(config).norm_kind != "rmsnorm_bare":  # bare norms have no weights to fetch
        norm_prefix = _norm_sd_prefix(config)
        all_keys.add(f"{norm_prefix}.weight")
        all_keys.add(f"{norm_prefix}.bias")

    tensors = {}
    missing = set()
    for k in all_keys:
        t = _cache_lookup(config.family, hf_repo, revision, k)
        if t is not None:
            tensors[k] = t
        else:
            missing.add(k)

    if missing:
        if config.family == "nanochat":
            from utils.nanochat_gpt import load_sd_tensors
            loaded = load_sd_tensors(config, revision, missing)
        else:
            from huggingface_hub import snapshot_download
            snap_dir = snapshot_download(hf_repo, revision=revision)
            loaded = _load_selective_tensors(snap_dir, missing)
        for k, t in loaded.items():
            _cache_store(config.family, hf_repo, revision, k, t)
        tensors.update(loaded)

    result = {}
    for prefix in sd_prefixes:
        w = tensors.get(f"{prefix}.weight")
        if w is not None:
            result[prefix] = SimpleNamespace(weight=w, bias=tensors.get(f"{prefix}.bias"))

    if need_norm and _fam(config).norm_kind == "rmsnorm_bare":
        result["__norm__"] = _RMSNorm(None)
    if norm_prefix:
        w = tensors.get(f"{norm_prefix}.weight")
        if w is not None:
            b = tensors.get(f"{norm_prefix}.bias")
            if _fam(config).norm_kind == "layernorm":
                import torch.nn as nn
                norm = nn.LayerNorm(w.shape[0])
                norm.weight.data.copy_(w)
                if b is not None:
                    norm.bias.data.copy_(b)
            else:
                norm = _RMSNorm(w)
            result["__norm__"] = norm

    return result


# --- WeightProvider implementations (inherit the Protocol => weight()/norm() enforced) ---

class LazyWeights(WeightProvider):
    """WeightProvider that loads each sd-prefix (and the final norm) from the HF
    snapshot / on-disk cache on first request, then memoizes."""
    def __init__(self, config: ModelConfig, hf_repo: str, revision: str) -> None:
        self._config, self._hf_repo, self._revision = config, hf_repo, revision
        self._weights: dict[str, SimpleNamespace | None] = {}   # sd_prefix -> namespace(.weight/.bias)
        self._norm: NormFn | None = None
        self._norm_loaded = False

    def weight(self, sd_prefix: str) -> WeightBias:
        if sd_prefix not in self._weights:
            self._weights[sd_prefix] = load_selective_weights(
                self._config, self._hf_repo, self._revision, {sd_prefix}).get(sd_prefix)
        ns = self._weights[sd_prefix]
        if ns is None:
            return None
        return ns.weight.detach().float(), (ns.bias.detach().float() if ns.bias is not None else None)

    def norm(self) -> NormFn | None:
        if not self._norm_loaded:
            self._norm = load_selective_weights(
                self._config, self._hf_repo, self._revision, set(), need_norm=True).get("__norm__")
            self._norm_loaded = True
        return self._norm


class ModelWeights(WeightProvider):
    """WeightProvider backed by a live model (collection-time derivation)."""
    def __init__(self, model: "torch.nn.Module", config: ModelConfig) -> None:
        self._model, self._config = model, config

    def weight(self, sd_prefix: str) -> WeightBias:
        try:
            m = self._model.get_submodule(sd_prefix)
        except AttributeError:
            return None
        b = getattr(m, "bias", None)
        assert isinstance(m.weight, torch.Tensor)
        return m.weight.detach().float(), (b.detach().float() if b is not None else None)

    def norm(self) -> NormFn:
        return get_final_layernorm(self._model, self._config).float()


def load_inference(path: str, derive: bool = True) -> "tuple[dict, ModelConfig | None, WeightProvider | None]":
    """Load an inference .pt + (config, LazyWeights) from its recorded identity; derive=False -> no weights."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    hf, rev = data.get("__hf_model__"), data.get("__revision__")
    config = get_model_config(hf) if hf else None
    return data, config, LazyWeights(config, hf, rev) if (derive and config) else None


# --- Token counting ---

PYTHIA_TOKENS_PER_STEP = 2_097_152

_token_count_cache = {}

NANOCHAT_TOKENS_PER_STEP = 524_288   # total_batch_size of the d12 training run


def get_token_count(model_name: str, step_num: int) -> int:
    """Return the number of pretraining tokens at a given step for a model."""
    if 'pythia' in model_name.lower():
        return step_num * PYTHIA_TOKENS_PER_STEP
    if 'nanochat' in model_name.lower():
        return step_num * NANOCHAT_TOKENS_PER_STEP

    config = get_model_config(model_name)
    if config.revisions_file is None:
        raise ValueError(f"No revisions_file for '{model_name}'")

    if config.revisions_file not in _token_count_cache:
        _token_count_cache[config.revisions_file] = _read_revisions_file(config.revisions_file)

    rev_string = _token_count_cache[config.revisions_file].get(step_num)
    if rev_string is None:
        raise ValueError(f"Step {step_num} not found in {config.revisions_file}")

    tokens_str = rev_string.split('-tokens')[-1].rstrip('B')
    return int(tokens_str) * 10**9


# --- HF cache management ---

def prefetch_checkpoint(model_name, revision):
    """Download model weights in background so they're cached for next iteration."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_name, revision=revision)
    except Exception:
        pass


def delete_cached_revision(model_name, revision):
    """Remove a specific revision from the HF cache to free disk space.
    Uses inodes to track references (works with both symlinks and hardlinks)."""
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_cache = os.path.join(hf_home, "hub", f"models--{model_name.replace('/', '--')}")
    if not os.path.isdir(model_cache):
        return

    ref_path = os.path.join(model_cache, "refs", revision)
    if not os.path.exists(ref_path):
        return

    try:
        with open(ref_path) as f:
            commit_hash = f.read().strip()

        snapshot_dir = os.path.join(model_cache, "snapshots", commit_hash)
        if os.path.isdir(snapshot_dir):
            shutil.rmtree(snapshot_dir)

        os.remove(ref_path)

        # Collect inodes still referenced by remaining snapshots
        snapshots_base = os.path.join(model_cache, "snapshots")
        referenced_inodes = set()
        if os.path.isdir(snapshots_base):
            for snap in os.listdir(snapshots_base):
                snap_path = os.path.join(snapshots_base, snap)
                if os.path.isdir(snap_path):
                    for root, _, files in os.walk(snap_path):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            try:
                                referenced_inodes.add(os.stat(fpath).st_ino)
                            except OSError:
                                pass

        # Delete orphaned blobs
        blobs_dir = os.path.join(model_cache, "blobs")
        if os.path.isdir(blobs_dir):
            for fname in os.listdir(blobs_dir):
                blob_path = os.path.join(blobs_dir, fname)
                try:
                    if os.stat(blob_path).st_ino not in referenced_inodes:
                        os.remove(blob_path)
                except OSError:
                    pass
    except Exception as e:
        print(f"Warning: could not clean cache for {revision}: {e}")
