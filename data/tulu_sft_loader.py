"""Loader for allenai/tulu-3-sft-mixture (Tulu 3 instruction tuning data).

Each example has a 'messages' field: [{"role": "user"|"assistant", "content": "..."}].
"""

from datasets import load_dataset


def get_dataset(split: str = "train", streaming: bool = True):
    """Load Tulu 3 SFT mixture from HuggingFace.

    Args:
        split: Dataset split (default: "train").
        streaming: Whether to use streaming mode (default: True).

    Returns:
        HuggingFace Dataset with 'messages' field (list of role/content dicts).
    """
    return load_dataset("allenai/tulu-3-sft-mixture", split=split, streaming=streaming)
