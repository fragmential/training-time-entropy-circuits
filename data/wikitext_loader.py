"""Loader for WikiText dataset."""

from datasets import load_dataset


def get_dataset(split: str = "train", version: str = "wikitext-103-raw-v1"):
    """
    Load WikiText dataset from HuggingFace.

    Args:
        split: Dataset split to load (default: "train")
        version: WikiText version to use (default: "wikitext-103-raw-v1")
                Options: "wikitext-2-raw-v1", "wikitext-103-raw-v1"

    Returns:
        HuggingFace Dataset with 'text' field
    """
    dataset = load_dataset("wikitext", version, split=split)
    return dataset
