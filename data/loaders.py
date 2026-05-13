from datasets import load_dataset

# (hf_repo, default_kwargs) — all loaders follow load_dataset(repo, **kwargs, split=split, streaming=streaming)
_REGISTRY = {
    "fineweb":                ("HuggingFaceFW/fineweb", {"name": "sample-10BT"}),
    "wikitext":               ("wikitext", {"name": "wikitext-103-raw-v1"}),
    "sciq":                   ("sciq", {}),
    "pile":                   ("EleutherAI/the_pile_deduplicated", {}),
    "pile_deduped_eleutherai":("EleutherAI/the_pile_deduplicated", {}),
    "pile_deduped_pietrolesci":("pietrolesci/pile-deduped", {}),
    "olmo_mix":               ("allenai/olmo-mix-1124", {"name": "dclm"}),
    "dolmino":                ("allenai/dolmino-mix-1124", {"name": "dclm"}),
    "tulu_sft":               ("allenai/tulu-3-sft-mixture", {}),
    "lam":                    ("lama", {"name": "trex"}),
}

def get_dataset(name, split="train", streaming=True, **overrides):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {sorted(_REGISTRY)}")
    repo, defaults = _REGISTRY[name]
    kwargs = {**defaults, **overrides}
    return load_dataset(repo, split=split, streaming=streaming, **kwargs)
