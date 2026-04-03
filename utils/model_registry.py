"""Model-specific logic for checkpoint discovery, model loading, tokenizer setup, and HF cache management."""
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np


@dataclass
class ModelConfig:
    family: str
    hf_repo: str          # actual HF repo to download from
    model_class: str      # "GPTNeoXForCausalLM" or "AutoModelForCausalLM"
    dtype: str            # "float16" or "bfloat16"
    trust_remote_code: bool
    pad_token_from_eos: bool
    training_dataset: str = None      # dataset name for "native" resolution
    revisions_file: str = None        # path to file mapping step→revision hash
    early_training_model: str = None  # separate HF repo for early checkpoints


# Per-model overrides for fields that differ within a family.
_OLMO_OVERRIDES = {
    "allenai/OLMo-2-0425-1B": {
        "revisions_file": "1b_revisions.txt",
        "early_training_model": "allenai/OLMo-2-0425-1B-early-training",
    },
    "allenai/OLMo-2-1124-7B": {
        "revisions_file": "7b_revisions.txt",
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
        overrides = _OLMO_OVERRIDES.get(model_name, {})
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
    raise ValueError(f"Unknown model family for '{model_name}'. Expected 'pythia' or 'olmo' in the name.")


# --- Checkpoint discovery ---

def _pythia_checkpoints(config, max_checkpoints):
    early = [0, 8, 16, 32, 64, 128, 256, 512]
    later = list(np.arange(1000, 143_001, 1000))
    if max_checkpoints and len(later) > max_checkpoints:
        indices = np.linspace(0, len(later) - 1, max_checkpoints, dtype=int)
        later = [later[i] for i in indices]
    return [(s, f"step{s}", config.hf_repo) for s in early + later]


def _olmo_checkpoints(config, max_checkpoints):
    if config.revisions_file is None:
        raise ValueError(f"OLMo model '{config.hf_repo}' has no revisions_file configured in model_registry.")
    checkpoint_map = _read_revisions_file(config.revisions_file)
    all_steps = sorted(checkpoint_map.keys())
    early = sorted(s for s in all_steps if s <= 10_000)
    later = sorted(s for s in all_steps if s > 10_000 and s % 10_000 == 0)
    if max_checkpoints and len(later) > max_checkpoints:
        indices = np.linspace(0, len(later) - 1, max_checkpoints, dtype=int)
        later = [later[i] for i in indices]
    schedule = []
    for s in early + later:
        rev = checkpoint_map[s]
        m = config.hf_repo
        if config.early_training_model and 1000 <= s <= 10_000:
            m = config.early_training_model
        schedule.append((s, rev, m))
    return schedule


def _read_revisions_file(filepath: str) -> dict:
    print(f"Opening checkpoint file: {filepath}")
    checkpoint_map = {}
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if "step" in line and "-tokens" in line:
                    try:
                        step = int(line.split("-tokens")[0].split("step")[-1])
                        checkpoint_map[step] = line
                    except ValueError:
                        print(f"Skipping line (invalid step): {line}")
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
    except Exception as e:
        print(f"Unexpected error: {e}")
    print(f"Found {len(checkpoint_map)} checkpoints")
    return checkpoint_map


def get_checkpoint_schedule(config, max_checkpoints=50):
    """Return list of (step_num, revision_str, hf_model_name) tuples."""
    if config.family == "pythia":
        return _pythia_checkpoints(config, max_checkpoints)
    return _olmo_checkpoints(config, max_checkpoints)


# --- Model loading ---

def load_model(config, hf_repo, revision):
    """Load model on CPU with appropriate class and dtype for the model family."""
    if config.model_class == "GPTNeoXForCausalLM":
        from transformers import GPTNeoXForCausalLM
        return GPTNeoXForCausalLM.from_pretrained(hf_repo, revision=revision, torch_dtype=config.dtype)
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        hf_repo, revision=revision, torch_dtype=config.dtype, trust_remote_code=config.trust_remote_code
    )


def load_tokenizer(config, revision=None):
    """Load tokenizer with appropriate settings for the model family."""
    from transformers import AutoTokenizer
    kwargs = {}
    if config.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if revision:
        kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(config.hf_repo, **kwargs)
    if config.pad_token_from_eos and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --- Architecture helpers ---

def get_mlp_projections(model, config, block_idx):
    """Return [(name, nn.Linear), ...] for MLP projections in the given block."""
    if config.family == "pythia":
        block = model.gpt_neox.layers[block_idx]
        return [
            (f"blk{block_idx}.up",   block.mlp.dense_h_to_4h),
            (f"blk{block_idx}.down", block.mlp.dense_4h_to_h),
        ]
    block = model.model.layers[block_idx]
    return [
        (f"blk{block_idx}.gate", block.mlp.gate_proj),
        (f"blk{block_idx}.up",   block.mlp.up_proj),
        (f"blk{block_idx}.down", block.mlp.down_proj),
    ]


def get_num_layers(model, config):
    """Return number of transformer blocks."""
    if config.family == "pythia":
        return len(model.gpt_neox.layers)
    return len(model.model.layers)


def get_final_layernorm(model, config):
    """Return the final layernorm module (before the lm_head).

    Pythia: model.gpt_neox.final_layer_norm (nn.LayerNorm)
    OLMo:   model.model.norm (RMSNorm)
    """
    if config.family == "pythia":
        return model.gpt_neox.final_layer_norm
    return model.model.norm


def get_block_layernorms(model, config, block_idx):
    """Return (input_layernorm, post_attention_layernorm) for a block.

    These allow hooking:
    - Before attention: input_layernorm (pre-hook = block input residual)
    - Between attention and MLP: post_attention_layernorm (pre-hook = post-attention residual)
    """
    if config.family == "pythia":
        block = model.gpt_neox.layers[block_idx]
        return block.input_layernorm, block.post_attention_layernorm
    block = model.model.layers[block_idx]
    return block.input_layernorm, block.post_attention_layernorm


# --- Selective weight loading ---

# Hook name → state dict key mapping (architecture-dependent)
_PYTHIA_PROJ_MAP = {"up": "dense_h_to_4h", "down": "dense_4h_to_h"}
_OLMO_PROJ_MAP = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}


def _hook_to_sd_keys(config, hook_name):
    """Map hook name (e.g. 'blk14.up') to state dict key prefix."""
    m = re.match(r"blk(\d+)\.(up|down|gate)", hook_name)
    if not m:
        return None
    idx, proj = m.group(1), m.group(2)
    if config.family == "pythia":
        return f"gpt_neox.layers.{idx}.mlp.{_PYTHIA_PROJ_MAP[proj]}"
    return f"model.layers.{idx}.mlp.{_OLMO_PROJ_MAP[proj]}"


def _norm_sd_prefix(config):
    """State dict key prefix for the final layer norm."""
    return "gpt_neox.final_layer_norm" if config.family == "pythia" else "model.norm"


def _load_from_safetensors(snap_dir, keys):
    """Load specific tensor keys from safetensors files in a snapshot directory."""
    import json
    from safetensors import safe_open

    index_path = os.path.join(snap_dir, "model.safetensors.index.json")
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


class _RMSNorm:
    """Minimal callable RMSNorm (no nn.Module overhead)."""
    def __init__(self, weight, eps=1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        return x * (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight.float()

    def float(self):
        return _RMSNorm(self.weight.float(), self.eps)


def load_selective_weights(config, hf_repo, revision, hook_names, need_norm=False):
    """Load only the weight tensors needed for specific hooks from safetensors.

    Only downloads the safetensors shards that contain the needed keys (not the full model).

    Returns dict:
        {hook_name: SimpleNamespace(weight=Tensor, bias=Tensor|None), ...}
        If need_norm: "__norm__" key with a callable norm (supports .float()).
    """
    from huggingface_hub import snapshot_download

    # Collect all state dict keys we need
    all_keys = set()
    hook_prefixes = {}  # hook_name -> sd_prefix
    for h in hook_names:
        prefix = _hook_to_sd_keys(config, h)
        if prefix:
            hook_prefixes[h] = prefix
            all_keys.add(f"{prefix}.weight")
            all_keys.add(f"{prefix}.bias")

    norm_prefix = None
    if need_norm:
        norm_prefix = _norm_sd_prefix(config)
        all_keys.add(f"{norm_prefix}.weight")
        all_keys.add(f"{norm_prefix}.bias")

    snap_dir = snapshot_download(hf_repo, revision=revision)
    tensors = _load_from_safetensors(snap_dir, all_keys)

    result = {}
    for h, prefix in hook_prefixes.items():
        w = tensors.get(f"{prefix}.weight")
        if w is not None:
            result[h] = SimpleNamespace(weight=w, bias=tensors.get(f"{prefix}.bias"))

    if norm_prefix:
        w = tensors.get(f"{norm_prefix}.weight")
        if w is not None:
            b = tensors.get(f"{norm_prefix}.bias")
            if config.family == "pythia":
                import torch.nn as nn
                norm = nn.LayerNorm(w.shape[0])
                norm.weight.data.copy_(w)
                if b is not None:
                    norm.bias.data.copy_(b)
            else:
                norm = _RMSNorm(w)
            result["__norm__"] = norm

    return result


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
