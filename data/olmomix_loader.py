"""Loader for allenai/olmo-mix-1124 (OLMo-2 pretraining data).

Configs: default, dclm, algebraic-stack, arxiv, open-web-math, pes2o, starcoder, wiki.
"""

from datasets import load_dataset


def get_dataset(config: str = "dclm", split: str = "train", streaming: bool = True):
    """Load OLMo-mix-1124 dataset from HuggingFace.

    Args:
        config: Dataset configuration/subset. Default "dclm" (DCLM-Baseline, ~95% of
                training mix). Use "default" for all sources (not shuffled across sources).
        split: Dataset split (default: "train").
        streaming: Whether to use streaming mode (default: True).

    Returns:
        HuggingFace Dataset with 'text' field.
    """
    return load_dataset("allenai/olmo-mix-1124", name=config, split=split, streaming=streaming)
