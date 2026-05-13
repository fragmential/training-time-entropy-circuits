import torch
from utils.data_utils import compute_token_mask, compute_labels


# --- compute_token_mask ---

def test_mask_all_no_padding():
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = compute_token_mask(ids, token_selection="all")
    assert mask.tolist() == [[True, True, True, True, False]]


def test_mask_all_with_padding():
    ids = torch.tensor([[1, 2, 3, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 0, 0]])
    mask = compute_token_mask(ids, attention_mask=attn, token_selection="all")
    # all non-pad except last real token (pos 2)
    assert mask.tolist() == [[True, True, False, False, False]]


def test_mask_last_padded():
    ids = torch.tensor([[1, 2, 3, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 0, 0]])
    mask = compute_token_mask(ids, attention_mask=attn, token_selection="last")
    assert mask.tolist() == [[False, False, True, False, False]]


def test_mask_last_packed_with_boundary():
    # EOS at positions 2 and 4. The code shifts right: if boundary at pos i,
    # mark pos i+1 (the token whose "last" input was the boundary doc).
    ids = torch.tensor([[10, 20, 99, 30, 99]])
    mask = compute_token_mask(ids, token_selection="last", boundary_token_ids=[99])
    # Verify at least some positions are selected
    assert mask.any()
    # Non-boundary, non-selected positions should be False
    assert mask[0, 0].item() is False


def test_mask_last_packed_no_boundary():
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = compute_token_mask(ids, token_selection="last")
    # no boundary → last position only
    assert mask.tolist() == [[False, False, False, False, True]]


def test_mask_skip_positions():
    # boundary at position 1
    ids = torch.tensor([[10, 99, 20, 30, 40]])
    mask = compute_token_mask(
        ids, token_selection="all", skip_positions=1, boundary_token_ids=[99],
    )
    # position 2 (1 after boundary at 1) should be skipped
    assert mask[0, 2].item() is False


def test_mask_answer_start():
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    answer_start = torch.tensor([3])
    mask = compute_token_mask(ids, token_selection="all", answer_start_positions=answer_start)
    # positions 0,1,2 should be masked out; 3 included, 4 excluded (last)
    assert mask[0, :3].tolist() == [False, False, False]
    assert mask[0, 3].item() is True


# --- compute_labels ---

def test_labels_shift():
    ids = torch.tensor([[10, 20, 30, 40, 50]])
    labels = compute_labels(ids)
    assert labels[0, 0].item() == 20
    assert labels[0, 3].item() == 50
    assert labels[0, 4].item() == -100  # last position


def test_labels_padding_masked():
    ids = torch.tensor([[10, 20, 30, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 0, 0]])
    labels = compute_labels(ids, attention_mask=attn)
    assert labels[0, 3].item() == -100
    assert labels[0, 4].item() == -100


def test_labels_answer_start():
    ids = torch.tensor([[10, 20, 30, 40, 50]])
    answer_start = torch.tensor([2])
    labels = compute_labels(ids, answer_start_positions=answer_start)
    assert labels[0, 0].item() == -100
    assert labels[0, 1].item() == -100
    # labels shifted first (label[i]=ids[i+1]), then mask applied: label[2] = ids[3] = 40
    assert labels[0, 2].item() == 40
