"""Loader for FineWeb dataset."""

from datasets import load_dataset


def get_dataset(split: str = "train", num_samples: int = 10000, streaming: bool = False):
    """
    Load FineWeb dataset from HuggingFace.

    Args:
        split: Dataset split to load (default: "train")
        num_samples: Number of samples to load (default: 10000)
        streaming: Whether to use streaming mode (default: False)

    Returns:
        HuggingFace Dataset with 'text' field
    """
    if streaming:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split=split,
            streaming=True
        )
        # Take first num_samples if streaming
        dataset = dataset.take(num_samples)
    else:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split=f"{split}[:{num_samples}]"
        )

    return dataset
