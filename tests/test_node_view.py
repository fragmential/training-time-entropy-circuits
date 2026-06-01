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
    in_cov, in_mean = _cov(d_in, 1)
    grad_cov, _ = _cov(d_out, 2)
    data = {"blk0.up": {"in.acts": in_cov, "n_in.acts": 64, "in.acts_mean": in_mean,
                        "out.grads": grad_cov, "n_out.grads": 64}}
    acc = DataAccessor(data)
    acc._layer_cache["blk0.up"] = _Layer(
        torch.randn(d_out, d_in, generator=torch.Generator().manual_seed(3)))
    return acc


def _residual_acc():
    d = 6
    val_cov, val_mean = _cov(d, 4)
    grad_cov, _ = _cov(d, 5)
    data = {
        "after_final_norm": {"value.acts": val_cov, "n_value.acts": 64,
                             "value.acts_mean": val_mean, "value.grads": grad_cov,
                             "n_value.grads": 64},
        "blk0.attn.out": {"value.acts": val_cov.clone(), "n_value.acts": 64},
    }
    return DataAccessor(data)


def _ov_acc():
    d_model, d_head = 12, 4
    slice_cov, slice_mean = _cov(d_head, 6)
    grad_cov, _ = _cov(d_head, 7)
    data = {"blk0.attn.head1": {"slice.acts": slice_cov, "n_slice.acts": 64,
                                "slice.acts_mean": slice_mean, "slice.grads": grad_cov,
                                "n_slice.grads": 64}}
    acc = DataAccessor(data)
    acc._layer_cache["blk0.attn.head1"] = _Layer(
        torch.randn(d_model, d_model, generator=torch.Generator().manual_seed(8)))
    return acc


def test_mlp_nodes_match_factors():
    acc = _mlp_acc()
    h = "blk0.up"
    assert torch.allclose(acc[h].in_.acts.eigvals, acc[h]._factor("in.acts").eigvals)
    assert torch.allclose(acc[h].out.grads.eigvals, acc[h]._factor("out.grads").eigvals)
    assert torch.allclose(acc[h].out.acts.cov, acc[h]._factor("out.acts").cov)
    assert acc[h].in_.grads.eigvals is None


def test_residual_nodes_match_factors():
    acc = _residual_acc()
    h = "after_final_norm"
    assert torch.allclose(acc[h].value.acts.eigvals, acc[h]._factor("value.acts").eigvals)
    assert torch.allclose(acc[h].value.grads.eigvals, acc[h]._factor("value.grads").eigvals)
    b = "blk0.attn.out"
    assert torch.allclose(acc[b].value.acts.eigvals, acc[b]._factor("value.acts").eigvals)
    assert acc[b].value.grads.eigvals is None  # boundary fixture has no grads


def test_ov_nodes_match_factors():
    acc = _ov_acc()
    h = "blk0.attn.head1"
    assert torch.allclose(acc[h].slice.acts.eigvals, acc[h]._factor("slice.acts").eigvals)
    assert torch.allclose(acc[h].slice.grads.eigvals, acc[h]._factor("slice.grads").eigvals)
    assert torch.allclose(acc[h].contrib.acts.cov, acc[h]._factor("contrib.acts").cov)
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
