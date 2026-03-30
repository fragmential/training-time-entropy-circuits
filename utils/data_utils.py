"""Data loading, packing/padding, and token mask utilities for the unified collection pipeline."""

import json
import os
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Text loading and caching
# ---------------------------------------------------------------------------

def load_and_cache_texts(
    dataset_loader_fn,
    num_samples: int,
    min_length: int,
    tokenizer,
    dataset_name: str,
    content_key: str = "text",
    max_bytes: Optional[int] = None,
) -> list:
    """Load dataset texts, filtering by minimum token length. Caches to disk.

    Args:
        num_samples: Maximum number of documents to collect (used when max_bytes is None).
        max_bytes: If set, stop collecting once total UTF-8 byte count of accepted texts
                   exceeds this value. Overrides num_samples as the stopping criterion.
                   Example: 80_000_000 ≈ 20M tokens, matching the reference K-FAC repo.
    """
    os.makedirs("data/cache", exist_ok=True)
    if max_bytes is not None:
        mb_hundredths = int(max_bytes / 10_000)   # truncate to 2 decimal places in MB
        mb_str = f"{mb_hundredths / 100:.2f}".rstrip("0").rstrip(".")
        cache_key = f"{dataset_name}_{mb_str}MB"
    else:
        cache_key = f"{dataset_name}_{num_samples}"
    cache_path = os.path.join("data", "cache", f"filtered_texts_{cache_key}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            texts = json.load(f)
        print(f"Loaded {len(texts)} cached sequences from {cache_path}")
        return texts

    from tqdm import tqdm
    dataset = dataset_loader_fn()
    texts = []
    total_bytes = 0
    for seq in tqdm(dataset, desc="Filtering"):
        text = seq[content_key]
        if tokenizer(text, return_tensors="pt").input_ids.shape[-1] > min_length:
            if max_bytes is not None:
                doc_bytes = len(text.encode("utf-8"))
                # Skip docs that alone would overshoot budget by >2x
                if total_bytes == 0 and doc_bytes > max_bytes * 2:
                    continue
                texts.append(text)
                total_bytes += doc_bytes
                if total_bytes >= max_bytes:
                    break
            else:
                texts.append(text)
                if len(texts) >= num_samples:
                    break
    with open(cache_path, "w") as f:
        json.dump(texts, f)
    if max_bytes is not None:
        print(f"Filtered and cached {len(texts)} sequences ({total_bytes/1e6:.1f} MB) to {cache_path}")
    else:
        print(f"Filtered and cached {len(texts)} sequences to {cache_path}")
    return texts


# ---------------------------------------------------------------------------
# Sequence preparation
# ---------------------------------------------------------------------------

def pack_sequences(texts: list, tokenizer, seq_len: int) -> torch.Tensor:
    """Tokenize texts and pack into fixed-length chunks (no padding).

    Returns: Tensor of shape (n_chunks, seq_len) with dtype=long.
    """
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
    n_chunks = len(all_ids) // seq_len
    if n_chunks == 0:
        raise ValueError(f"Not enough tokens ({len(all_ids)}) for even one chunk of {seq_len}")
    all_ids = all_ids[: n_chunks * seq_len]
    return torch.tensor(all_ids, dtype=torch.long).view(n_chunks, seq_len)


def pad_and_tokenize(texts: list, tokenizer, max_length: int) -> dict:
    """Tokenize texts with padding to longest in batch.

    Returns dict with 'input_ids' and 'attention_mask' tensors.
    """
    tokenized = tokenizer(
        texts,
        padding="longest",
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    )
    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
    }


# ---------------------------------------------------------------------------
# Token mask computation
# ---------------------------------------------------------------------------

def compute_token_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_selection: str = "all",
    skip_positions: int = 0,
    boundary_token_ids: Optional[list] = None,
    answer_start_positions: Optional[torch.Tensor] = None,
) -> torch.BoolTensor:
    """Compute a boolean mask selecting which token positions to include.

    Args:
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) or None for packed data
        token_selection: "all" or "last"
        skip_positions: skip first k tokens after each document boundary (packed only)
        boundary_token_ids: list of token IDs marking document boundaries (e.g., EOS)
        answer_start_positions: (batch,) for answer-only mode

    Returns:
        BoolTensor of shape (batch, seq_len). True = include this position.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if token_selection == "all":
        if attention_mask is not None:
            # Padded: include all non-pad tokens except last real position
            # (last position has no next-token target for gradient computation)
            mask = attention_mask.bool().clone()
            # Exclude the very last real token per sequence
            last_indices = attention_mask.sum(dim=1) - 1
            mask[torch.arange(batch_size, device=device), last_indices] = False
        else:
            # Packed: include all except last position (no target)
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
            mask[:, -1] = False

    elif token_selection == "last":
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        if attention_mask is not None:
            # Padded: last real token per sequence
            last_indices = attention_mask.sum(dim=1) - 1
            mask[torch.arange(batch_size, device=device), last_indices] = True
        elif boundary_token_ids is not None:
            # Packed: last token before each document boundary
            for eos_id in boundary_token_ids:
                is_boundary = (input_ids == eos_id)
                # The token just before EOS is the "last token" of the document
                # Shift right: position i-1 is last token if position i is EOS
                shifted = torch.zeros_like(is_boundary)
                shifted[:, 1:] = is_boundary[:, :-1]
                # But the EOS itself could also be the "last" token — use position before it
                mask |= shifted
            # If no boundaries found, fall back to last position
            if not mask.any():
                mask[:, -1] = True
        else:
            # Packed without boundaries: just the last position
            mask[:, -1] = True
    else:
        raise ValueError(f"Unknown token_selection: {token_selection}")

    # Apply skip_positions after boundaries
    if skip_positions > 0 and boundary_token_ids is not None:
        for eos_id in boundary_token_ids:
            is_boundary = (input_ids == eos_id)
            for k in range(1, skip_positions + 1):
                # Mask out k positions after each boundary
                skip_mask = torch.zeros_like(is_boundary)
                if k < seq_len:
                    skip_mask[:, k:] = is_boundary[:, :-k]
                mask &= ~skip_mask

    # Answer-only mode: mask out everything before answer_start
    if answer_start_positions is not None:
        pos_range = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        starts = answer_start_positions.unsqueeze(1)  # (batch, 1)
        answer_mask = pos_range >= starts
        mask &= answer_mask

    return mask


def compute_labels(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    answer_start_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute next-token prediction labels from input_ids.

    Args:
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) or None for packed
        answer_start_positions: (batch,) — mask prompt tokens with -100

    Returns:
        labels: (batch, seq_len) with -100 for ignored positions
    """
    labels = input_ids.clone()
    # Shift: label at position i is token at position i+1
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # no target for last position

    # Mask padding
    if attention_mask is not None:
        labels[~attention_mask.bool()] = -100

    # Mask prompt tokens for answer-only gradient collection
    if answer_start_positions is not None:
        batch_size, seq_len = input_ids.shape
        pos_range = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        starts = answer_start_positions.unsqueeze(1)
        labels[pos_range < starts] = -100

    return labels
