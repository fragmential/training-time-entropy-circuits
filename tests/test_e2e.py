"""End-to-end tests using a single pythia-14m checkpoint.

These require model downloads and should be run on a compute node:
    srun --partition=staging --cpus-per-task=4 --time=00:30:00 \
        uv run pytest tests/test_e2e.py -v

Mark with pytest.mark.e2e so they can be skipped in fast runs:
    pytest tests/ -v -m "not e2e"
"""
import os
import gc
import pytest
import tempfile
import torch
import numpy as np

pytestmark = pytest.mark.e2e

MODEL = "EleutherAI/pythia-14m-deduped"
STEP = "step143000"
NUM_SAMPLES = 30
SEQ_LEN = 256
ATOL = 1e-4


@pytest.fixture(scope="module")
def model_and_config():
    from utils.model_registry import get_model_config, load_model
    config = get_model_config(MODEL)
    model = load_model(config, config.hf_repo, STEP)
    yield model, config
    del model
    gc.collect()


@pytest.fixture(scope="module")
def tokenizer():
    from utils.model_registry import get_model_config, load_tokenizer
    config = get_model_config(MODEL)
    return load_tokenizer(config, revision=STEP)


@pytest.fixture(scope="module")
def texts_and_packed(tokenizer):
    from utils.data_utils import get_loader, load_and_cache_texts, pack_sequences
    loader_fn = get_loader("fineweb")
    texts = load_and_cache_texts(loader_fn, NUM_SAMPLES, 32, tokenizer, "fineweb")
    packed = pack_sequences(texts, tokenizer, SEQ_LEN)
    return texts, packed


def test_identity_vs_hook(model_and_config, tokenizer, texts_and_packed):
    """identity_head and after_final_norm hook produce identical activations."""
    from utils.hooks import HookCollector, setup_identity_head, restore_head
    from utils.model_registry import get_final_layernorm

    model, config = model_and_config
    texts, _ = texts_and_packed
    batch = texts[:4]
    tok = tokenizer(batch, padding="longest", return_tensors="pt", max_length=SEQ_LEN, truncation=True)
    input_ids, attention_mask = tok.input_ids, tok.attention_mask

    # identity_head
    original, attr = setup_identity_head(model)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(input_ids.size(0))
    acts_identity = out.logits[batch_idx, last_idx].float()
    restore_head(model, original, attr)

    # hook
    ln = get_final_layernorm(model, config)
    last_mask = torch.zeros(input_ids.shape, dtype=torch.bool)
    last_mask[batch_idx, last_idx] = True
    collector = HookCollector(ln, capture="output", mode="acts")
    collector.set_token_mask(last_mask)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    acts_hook = torch.cat(collector._acts_list, dim=0)
    collector.close()

    assert torch.allclose(acts_identity, acts_hook, atol=ATOL)


def test_covariance_shapes(model_and_config, texts_and_packed):
    """HookCollector produces correct shapes for MLP A and G."""
    from utils.model_registry import get_mlp_projections
    from utils.hooks import HookCollector
    from utils.data_utils import compute_token_mask, compute_labels

    model, config = model_and_config
    _, packed = texts_and_packed
    x = packed[:2]
    projections = get_mlp_projections(model, config, 0)
    name, layer = projections[0]
    d_out, d_in = layer.weight.shape

    collector = HookCollector(layer, capture="input", mode="cov", collect_grad=True)
    mask = compute_token_mask(x, token_selection="all")
    collector.set_token_mask(mask)

    for p in model.parameters():
        p.requires_grad_(False)
    layer.weight.requires_grad_(True)
    model.enable_input_require_grads()

    model.zero_grad(set_to_none=True)
    logits = model(x).logits[:, :-1].float()
    labels = compute_labels(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels[:, :-1].reshape(-1), ignore_index=-100,
    )
    loss.backward()

    factors = collector.factors()
    collector.close()

    assert factors["A"].shape == (d_in, d_in)
    # G captured on same side as A (capture="input"), so also d_in x d_in
    assert factors["G"].shape == (d_in, d_in)
    assert factors["n"] > 0


def test_storage_roundtrip_real_data(model_and_config, texts_and_packed):
    """Full pipeline: collect → save → convert → accessor reads correctly."""
    from utils.model_registry import get_final_layernorm
    from utils.hooks import HookCollector, setup_identity_head, restore_head
    from utils.accessor import DataAccessor
    from utils.data_utils import compute_token_mask

    model, config = model_and_config
    _, packed = texts_and_packed
    x = packed[:2]

    # Collect activations
    ln = get_final_layernorm(model, config)
    mask = compute_token_mask(x, token_selection="all")
    collector = HookCollector(ln, capture="output", mode="cov", collect_means=True)
    collector.set_token_mask(mask)
    with torch.no_grad():
        model(x)
    factors = {"residual": collector.factors()}
    collector.close()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as cov+m → convert to cov_svd
        cov_path = os.path.join(tmpdir, "cov.pt")
        DataAccessor(factors).save(cov_path, format="cov+m")

        svd_path = os.path.join(tmpdir, "svd.pt")
        DataAccessor(cov_path).save(svd_path, format="cov_svd+m")

        # Read via accessor
        acc = DataAccessor(svd_path)
        eigvals = acc["residual"].A.eigvals
        assert eigvals is not None
        assert len(eigvals) > 0
        assert (eigvals[:-1] >= eigvals[1:]).all()

        # Verify mean is stored
        mean = acc["residual"].A.mean
        assert mean is not None


def test_before_norm_recovers_after_norm(model_and_config, texts_and_packed, tokenizer):
    """Applying final layernorm to before_final_norm recovers after_final_norm."""
    from utils.model_registry import get_final_layernorm

    model, config = model_and_config
    texts, _ = texts_and_packed
    batch = texts[:2]
    tok = tokenizer(batch, padding="longest", return_tensors="pt", max_length=SEQ_LEN, truncation=True)
    input_ids, attention_mask = tok.input_ids, tok.attention_mask

    ln = get_final_layernorm(model, config)
    before_buf, after_buf = [], []
    h1 = ln.register_forward_pre_hook(lambda m, inp: before_buf.append(inp[0].detach()))
    h2 = ln.register_forward_hook(lambda m, inp, out: after_buf.append(out.detach()))

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    h1.remove()
    h2.remove()

    acts_before = before_buf[0]
    acts_after = after_buf[0]

    with torch.no_grad():
        acts_normed = ln(acts_before)

    assert torch.allclose(acts_normed.float(), acts_after.float(), atol=ATOL)
    # Sanity: before and after should actually differ
    assert (acts_before[0, 0].float() - acts_after[0, 0].float()).abs().max() > 0.01


def test_B_equals_WAWt(model_and_config, texts_and_packed):
    """B is recoverable from A via W A W^T + bias correction terms."""
    from utils.model_registry import get_mlp_projections, get_num_layers
    from utils.hooks import HookCollector
    from utils.data_utils import compute_token_mask

    model, config = model_and_config
    _, packed = texts_and_packed
    x = packed[:2]
    mask = compute_token_mask(x, token_selection="all")

    n_layers = get_num_layers(model, config)
    projections = get_mlp_projections(model, config, n_layers - 1)

    for name, layer in projections:
        ic = HookCollector(layer, capture="input", mode="cov", collect_means=True)
        ic.set_token_mask(mask)
        oc = HookCollector(layer, capture="output", mode="cov")
        oc.set_token_mask(mask)

        with torch.no_grad():
            model(x)

        a_factors = ic.factors()
        b_factors = oc.factors()
        ic.close()
        oc.close()

        n = a_factors["n"]
        A_cov = a_factors["A"].float() / n
        B_cov = b_factors["A"].float() / n  # output collector's "A" is B
        W = layer.weight.detach().float()
        b = layer.bias.detach().float() if layer.bias is not None else None
        mu = a_factors["A_mean"]

        B_derived = W @ A_cov @ W.T
        if b is not None:
            Wmu = W @ mu
            B_derived = B_derived + torch.outer(Wmu, b) + torch.outer(b, Wmu) + torch.outer(b, b)

        rel_err = (B_cov - B_derived).norm().item() / B_cov.norm().item() * 100
        assert rel_err < 0.01, f"{name}: B recovery error {rel_err:.4f}% > 0.01%"
