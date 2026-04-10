"""Loader for allenai/dolmino-mix-1124 (OLMo-2 second-stage pretraining data).

Configs: dclm, flan, math, pes2o, stackexchange, wiki.
"""

from datasets import load_dataset


def get_dataset(config: str = "dclm", split: str = "train", streaming: bool = True):
    """Load Dolmino-mix-1124 dataset from HuggingFace.

    Args:
        config: Dataset configuration/subset. Default "dclm" (DCLM, ~47-89% of
                training mix depending on stage). Use "default" for all sources
                (not shuffled across sources).
        split: Dataset split (default: "train").
        streaming: Whether to use streaming mode (default: True).

    Returns:
        HuggingFace Dataset with 'text' field.
    """
    return load_dataset("allenai/dolmino-mix-1124", name=config, split=split, streaming=streaming)
