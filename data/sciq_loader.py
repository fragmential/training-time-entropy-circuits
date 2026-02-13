"""Loader for SciQ dataset."""

from datasets import load_dataset


def get_dataset(split: str = "train"):
    """
    Load SciQ (Science Question Answering) dataset from HuggingFace.

    Args:
        split: Dataset split to load (default: "train")
               Options: "train", "validation", "test"

    Returns:
        HuggingFace Dataset with question/answer fields
    """
    dataset = load_dataset("sciq", split=split)
    return dataset
