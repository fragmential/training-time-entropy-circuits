"""Loader for FineWeb dataset."""

from datasets import load_dataset


def get_dataset(split: str = "train", streaming: bool = True):
    """
    Load FineWeb dataset from HuggingFace.

    Args:
        split: Dataset split to load (default: "train")
        streaming: Whether to use streaming mode (default: True)

    Returns:
        HuggingFace Dataset with 'text' field
    """
    return load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split=split,
        streaming=streaming,
    )
