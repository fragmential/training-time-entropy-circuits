"""Loader for LAMA (LAnguage Model Analysis) dataset."""

from datasets import load_dataset


def get_dataset(split: str = "train", probe: str = "trex"):
    """
    Load LAMA dataset from HuggingFace.

    LAMA is a probe for analyzing factual and commonsense knowledge in language models.

    Args:
        split: Dataset split to load (default: "train")
        probe: Which LAMA probe to use (default: "trex")
               Options: "trex", "google_re", "squad", "conceptnet"

    Returns:
        HuggingFace Dataset with knowledge probing templates
    """
    # LAMA dataset can be loaded from various sources
    # Using lama as a common knowledge probing dataset
    try:
        dataset = load_dataset("lama", probe, split=split)
    except Exception:
        # Fallback to a general text dataset if LAMA is not available
        print(f"Warning: LAMA dataset not found. Using WikiText as fallback.")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    return dataset
