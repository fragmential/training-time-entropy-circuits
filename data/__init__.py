"""Data loaders for various datasets used in representation geometry analysis.

Centralized DATASET_LOADERS registry — all scripts import from here.
"""

from data import fineweb_loader, wikitext_loader, lam_loader, sciq_loader

DATASET_LOADERS = {
    "fineweb": fineweb_loader.get_dataset,
    "wikitext": wikitext_loader.get_dataset,
    "lam": lam_loader.get_dataset,
    "sciq": sciq_loader.get_dataset,
}

# Lazy-loaded optional loaders (to avoid import errors if not installed)
_OPTIONAL_LOADERS = {
    "pile": ("data.pile_loader", "get_dataset", {}),
    "pile_deduped_eleutherai": ("data.pile_loader", "get_dataset", {"variant": "eleutherai-dedup"}),
    "pile_deduped_pietrolesci": ("data.pile_loader", "get_dataset", {"variant": "pietrolesci-dedup"}),
    "olmo_mix": ("data.olmomix_loader", "get_dataset", {}),
    "dolmino": ("data.dolmino_loader", "get_dataset", {}),
    "tulu_sft": ("data.tulu_sft_loader", "get_dataset", {}),
}


def get_loader(name: str):
    """Get a dataset loader function by name. Supports lazy loading for optional datasets."""
    if name in DATASET_LOADERS:
        return DATASET_LOADERS[name]

    if name in _OPTIONAL_LOADERS:
        module_path, fn_name, default_kwargs = _OPTIONAL_LOADERS[name]
        import importlib
        mod = importlib.import_module(module_path)
        fn = getattr(mod, fn_name)
        if default_kwargs:
            import functools
            return functools.partial(fn, **default_kwargs)
        return fn

    raise ValueError(
        f"Unknown dataset: {name}. "
        f"Available: {sorted(list(DATASET_LOADERS.keys()) + list(_OPTIONAL_LOADERS.keys()))}"
    )
