from functools import partial
from data.loaders import get_dataset, _REGISTRY

AVAILABLE_DATASETS = sorted(_REGISTRY)

def get_loader(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {AVAILABLE_DATASETS}")
    return partial(get_dataset, name)
