import torch
import pytest

from utils.accessor import DataAccessor


def _cov(d, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(64, d, generator=g)
    return X.T @ X / 64, X.mean(0)


class _Layer:
    def __init__(self, W, b=None):
        self.weight = W
        self.bias = b


def _mlp_acc():
    d_in, d_out = 8, 6
    A_cov, A_mean = _cov(d_in, 1)
    G_cov, _ = _cov(d_out, 2)
    data = {"blk0.up": {"A": A_cov, "n_A": 64, "A_mean": A_mean, "G": G_cov, "n_G": 64}}
    acc = DataAccessor(data)
    acc._layer_cache["blk0.up"] = _Layer(
        torch.randn(d_out, d_in, generator=torch.Generator().manual_seed(3)))
    return acc


def _residual_acc():
    d = 6
    A_cov, A_mean = _cov(d, 4)
    G_cov, _ = _cov(d, 5)
    data = {
        "after_final_norm": {"A": A_cov, "n_A": 64, "A_mean": A_mean, "G": G_cov, "n_G": 64},
        "blk0.attn.out": {"A": A_cov.clone(), "n_A": 64},
    }
    return DataAccessor(data)


def _ov_acc():
    d_model, d_head = 12, 4
    A_cov, A_mean = _cov(d_head, 6)
    G_cov, _ = _cov(d_head, 7)
    data = {"blk0.attn.head1": {"A": A_cov, "n_A": 64, "A_mean": A_mean, "G": G_cov, "n_G": 64}}
    acc = DataAccessor(data)
    acc._layer_cache["blk0.attn.head1"] = _Layer(
        torch.randn(d_model, d_model, generator=torch.Generator().manual_seed(8)))
    return acc


def test_mlp_nodes_match_factors():
    acc = _mlp_acc()
    h = "blk0.up"
    assert torch.allclose(acc[h].in_.acts.eigvals, acc[h].A.eigvals)
    assert torch.allclose(acc[h].out.grads.eigvals, acc[h].G.eigvals)
    assert torch.allclose(acc[h].out.acts.cov, acc[h].B.cov)
    assert acc[h].in_.grads.eigvals is None


def test_residual_nodes_match_factors():
    acc = _residual_acc()
    h = "after_final_norm"
    assert torch.allclose(acc[h].value.acts.eigvals, acc[h].A.eigvals)
    assert torch.allclose(acc[h].value.grads.eigvals, acc[h].G.eigvals)
    b = "blk0.attn.out"
    assert torch.allclose(acc[b].value.acts.eigvals, acc[b].A.eigvals)
    assert acc[b].value.grads.eigvals is None  # boundary fixture has no G


def test_ov_nodes_match_factors():
    acc = _ov_acc()
    h = "blk0.attn.head1"
    assert torch.allclose(acc[h].slice.acts.eigvals, acc[h].A.eigvals)
    assert torch.allclose(acc[h].slice.grads.eigvals, acc[h].G.eigvals)
    assert torch.allclose(acc[h].contrib.acts.cov, acc[h].O.cov)
    assert acc[h].contrib.grads.eigvals is None


def test_wrong_kind_role_raises():
    acc = _mlp_acc()
    with pytest.raises(AttributeError):
        acc["blk0.up"].value
    res = _residual_acc()
    with pytest.raises(AttributeError):
        res["after_final_norm"].out
    with pytest.raises(AttributeError):
        res["after_final_norm"].slice
