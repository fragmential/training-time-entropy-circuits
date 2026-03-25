"""Loader for The Pile dataset and deduplicated variants.

Variants:
    "eleutherai-dedup" (default)  — EleutherAI/the_pile_deduplicated
    "uncopyrighted"               — monology/pile-uncopyrighted (didn't seem to reach HF last time)
    "pietrolesci-dedup"           — pietrolesci/pile-deduped (pre-tokenized, no raw text)
"""

from datasets import load_dataset


_REPOS = {
    "uncopyrighted": "monology/pile-uncopyrighted",
    "eleutherai-dedup": "EleutherAI/the_pile_deduplicated",
    "pietrolesci-dedup": "pietrolesci/pile-deduped",
}


def get_dataset(variant: str = "eleutherai-dedup", split: str = "train", streaming: bool = True):
    """Load a Pile variant from HuggingFace.

    Args:
        variant: Which Pile variant to load.
        split: Dataset split (default: "train").
        streaming: Whether to use streaming mode (default: True).

    Returns:
        HuggingFace Dataset with 'text' field (except pietrolesci which has 'input_ids').
    """
    if variant not in _REPOS:
        raise ValueError(f"Unknown Pile variant: {variant}. Available: {list(_REPOS.keys())}")
    return load_dataset(_REPOS[variant], split=split, streaming=streaming)
