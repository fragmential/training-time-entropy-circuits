"""Nanochat family: vendored GPT surface, hook resolution, residual identities, tokenizer."""
import pytest
import torch
import torch.nn.functional as F

from utils.hook_specs import candidate_leaves, resolve
from utils.model_registry import (
    ModelWeights,
    derivation,
    get_block_boundary_hooks,
    get_model_config,
    get_num_layers,
    get_token_count,
    NANOCHAT_TOKENS_PER_STEP,
)
from utils.nanochat_gpt import GPT, GPTConfig, NanochatTokenizer, Block


def _tiny_model(n_layer=2, n_head=2, n_embd=16, vocab_size=64):
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=32, vocab_size=vocab_size, n_layer=n_layer,
                    n_head=n_head, n_kv_head=n_head, n_embd=n_embd)
    model = GPT(cfg)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return model.eval()


_CONFIG = get_model_config("nanochat-d12")


# ---------------------------------------------------------------------------
# ModelConfig + schedule facts
# ---------------------------------------------------------------------------

def test_model_config_fields():
    assert _CONFIG.family == "nanochat"
    assert _CONFIG.model_class == "NanochatGPT"
    assert _CONFIG.dtype == "bfloat16"
    assert _CONFIG.training_dataset == "fineweb_edu_100b"


def test_token_count():
    assert get_token_count("nanochat-d12", 50) == 50 * NANOCHAT_TOKENS_PER_STEP


def test_hf_config_aliases():
    m = _tiny_model()
    assert m.config.num_attention_heads == 2
    assert m.config.head_dim == 8
    assert m.config.hidden_size == 16
    m.config.use_cache = False  # collect.py's grad path sets this


# ---------------------------------------------------------------------------
# State dict: keys match the training checkpoints (norms carry no params)
# ---------------------------------------------------------------------------

def test_state_dict_has_no_norm_keys_and_roundtrips():
    m = _tiny_model()
    keys = set(m.state_dict())
    assert "transformer.wte.weight" in keys and "lm_head.weight" in keys
    assert "transformer.h.0.attn.c_q.weight" in keys
    assert "transformer.h.0.mlp.c_fc.weight" in keys
    assert not any("ln1" in k or "ln2" in k or "final_norm" in k for k in keys)
    assert not any("cos" in k or "sin" in k for k in keys)  # rotary buffers not persisted
    m2 = _tiny_model()
    m2.load_state_dict(m.state_dict(), strict=True)


# ---------------------------------------------------------------------------
# Hook resolution: the nanochat leaf universe
# ---------------------------------------------------------------------------

def test_candidate_leaves():
    m = _tiny_model()
    cand = candidate_leaves(m, _CONFIG, [0, 1], num_heads_per_block=2)
    for k in ["blk0.mlp.up.in", "blk0.mlp.up.out", "blk0.mlp.down.in", "blk0.mlp.down.out",
              "blk0.attn.in", "blk0.attn.out", "blk0.mlp.in", "blk0.mlp.out",
              "blk0.attn.head0.slice", "blk0.attn.head1.slice",
              "before_final_norm", "after_final_norm"]:
        assert k in cand
    # no gate (gateless MLP), no raw_out (no post-norms)
    assert not any(".gate." in c for c in cand)
    assert not any("raw_out" in c for c in cand)


def test_resolve_full_suite_patterns():
    m = _tiny_model()
    singles, ovs, _ = resolve(
        m, _CONFIG, [0, 1],
        ["blk*.attn.in", "blk*.attn.out", "blk*.mlp.in", "blk*.mlp.out",
         "blk*.attn.head*.slice", "before_final_norm", "after_final_norm",
         "blk*.attn.raw_out", "blk*.mlp.gate.in"],   # silent no-ops on nanochat
        default_token_selection="all",
    )
    leaves = {s.leaf for s in singles}
    assert {"blk0.attn.in", "blk0.attn.out", "blk0.mlp.in", "blk0.mlp.out"} <= leaves
    assert not any("raw_out" in l or ".gate." in l for l in leaves)
    assert len(ovs) == 2 and ovs[0].selected_heads == [0, 1]


# ---------------------------------------------------------------------------
# Forward + residual-stream identities at the boundary leaves
# ---------------------------------------------------------------------------

def _capture_boundaries(model, ids):
    captured = {}
    handles = []
    for i in range(get_num_layers(model, _CONFIG)):
        for name, mod, cap in get_block_boundary_hooks(model, _CONFIG, i):
            def mk(name, cap):
                def hook(module, inputs, output):
                    captured[name] = (inputs[0] if cap == "input" else output).detach()
                return hook
            handles.append(mod.register_forward_hook(mk(name, cap)))
    out = model(input_ids=ids)
    for h in handles:
        h.remove()
    return captured, out


def test_boundary_identities():
    m = _tiny_model()
    ids = torch.randint(0, 64, (2, 8))
    cap, _ = _capture_boundaries(m, ids)
    # pre-norm sequential: attn.in + attn.out == mlp.in; mlp.in + mlp.out == next attn.in
    torch.testing.assert_close(cap["blk0.attn.in"] + cap["blk0.attn.out"], cap["blk0.mlp.in"])
    torch.testing.assert_close(cap["blk0.mlp.in"] + cap["blk0.mlp.out"], cap["blk1.attn.in"])


def test_final_norm_leaves_and_logits():
    m = _tiny_model()
    ids = torch.randint(0, 64, (1, 8))
    grabbed = {}
    fn = m.final_norm
    h = fn.register_forward_hook(
        lambda mod, inp, out: grabbed.update(before=inp[0].detach(), after=out.detach()))
    out = m(input_ids=ids)
    h.remove()
    torch.testing.assert_close(grabbed["after"], F.rms_norm(grabbed["before"], (16,)))
    assert out.logits.shape == (1, 8, 64)
    assert out.logits.abs().max() <= 15.0   # softcap


def test_identity_head_skips_softcap():
    m = _tiny_model()
    ids = torch.randint(0, 64, (1, 8))
    m.lm_head = torch.nn.Identity()
    out = m(input_ids=ids)
    assert out.logits.shape == (1, 8, 16)   # post-final-norm residual


def test_padding_mask_rejected():
    m = _tiny_model()
    ids = torch.randint(0, 64, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    m(input_ids=ids, attention_mask=mask)   # all-ones is fine
    mask[0, -1] = 0
    with pytest.raises(AssertionError, match="packed"):
        m(input_ids=ids, attention_mask=mask)


# ---------------------------------------------------------------------------
# Grad path shims
# ---------------------------------------------------------------------------

def test_input_require_grads_lets_boundary_grads_flow():
    m = _tiny_model()
    m.enable_input_require_grads()
    m.train()
    grads = {}
    blk0 = m.transformer.h[0]
    assert isinstance(blk0, Block)
    h = blk0.ln1.register_full_backward_hook(
        lambda mod, gin, gout: grads.update(attn_in=gin[0]))
    ids = torch.randint(0, 64, (1, 8))
    m(input_ids=ids).logits.sum().backward()
    h.remove()
    assert grads["attn_in"] is not None and grads["attn_in"].shape == (1, 8, 16)


# ---------------------------------------------------------------------------
# Derivation: mlp.up.out from mlp.up.in via the live-model WeightProvider
# ---------------------------------------------------------------------------

def test_mlp_out_derivation_matches_capture():
    m = _tiny_model()
    ids = torch.randint(0, 64, (2, 8))
    blk0 = m.transformer.h[0]
    assert isinstance(blk0, Block)
    seen = {}
    h = blk0.mlp.c_fc.register_forward_hook(
        lambda mod, inp, out: seen.update(x=inp[0].detach(), y=out.detach()))
    m(input_ids=ids)
    h.remove()

    deriv = derivation(_CONFIG, "blk0.mlp.up.out", "acts")
    assert deriv is not None
    src_leaf, src_q, T = deriv
    assert (src_leaf, src_q) == ("blk0.mlp.up.in", "acts")
    ing = T.ingredients(ModelWeights(m, _CONFIG))
    assert ing is not None
    W, b = ing
    assert b is None   # no biases anywhere
    x = seen["x"].reshape(-1, 16)
    y = seen["y"].reshape(-1, 64)
    cov_in = x.T.float() @ x.float()
    fn = dict(T.recipes())["cov"][0][1]
    torch.testing.assert_close(fn(W, b, cov_in), y.T.float() @ y.float(), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Vocabulary-entropy lens
# ---------------------------------------------------------------------------

def test_entropy_lens_matches_manual_computation():
    import math
    from utils.entropy_lens import EntropyLens

    m = _tiny_model()
    ids = torch.randint(0, 64, (2, 8))
    lens = EntropyLens(m, _CONFIG, chunk_tokens=5)   # non-divisor chunk exercises chunking
    m(input_ids=ids)
    res = lens.close()

    assert res["layers"] == ["emb", "blk0", "blk1"]
    assert res["n_tokens"] == 16
    assert all(0.0 <= e <= math.log(64) + 1e-6 for e in res["per_layer"])

    # manual: residual entering blk0 -> final_norm -> lm_head -> softcap -> entropy
    with torch.no_grad():
        h = F.rms_norm(m.transformer.wte(ids), (16,))
        logits = 15 * torch.tanh(m.lm_head(m.final_norm(h)) / 15)
        log_p = torch.log_softmax(logits.float(), dim=-1)
        expected = -(log_p.exp() * log_p).sum(-1).mean().item()
    assert abs(res["per_layer"][0] - expected) < 1e-5


def test_entropy_lens_survives_repeated_passes():
    from utils.entropy_lens import EntropyLens

    m = _tiny_model()
    ids = torch.randint(0, 64, (1, 8))
    lens = EntropyLens(m, _CONFIG)
    m(input_ids=ids)
    once = [lens.sums[n] / lens.counts[n] for n in lens.layers]
    m(input_ids=ids)   # second pass over the same batch (max_layers_per_pass > 0)
    res = lens.close()
    assert res["n_tokens"] == 16
    for a, b in zip(once, res["per_layer"]):
        assert abs(a - b) < 1e-9   # mean is pass-count invariant


# ---------------------------------------------------------------------------
# Tokenizer adapter (offline tiny tiktoken encoding)
# ---------------------------------------------------------------------------

def _tiny_tokenizer():
    import tiktoken
    enc = tiktoken.Encoding(
        name="tiny",
        pat_str=r"\S+|\s+",
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={"<|bos|>": 256},
    )
    return NanochatTokenizer(enc)


def test_tokenizer_surface():
    tok = _tiny_tokenizer()
    assert tok.eos_token_id == tok.bos_token_id == 256

    ids = tok("ab", add_special_tokens=False).input_ids
    assert ids == [ord("a"), ord("b")]
    ids = tok("ab").input_ids
    assert ids[0] == tok.bos_token_id

    batch = tok(["ab", "abcd"], padding="longest", return_tensors="pt",
                max_length=8, truncation=True)
    assert batch.input_ids.shape == batch.attention_mask.shape == (2, 5)
    assert batch.attention_mask[0].tolist() == [1, 1, 1, 0, 0]
    assert batch.input_ids[0, 3] == tok.pad_token_id

    single = tok("abc", return_tensors="pt")
    assert single.input_ids.shape == (1, 4)
    assert tok.decode([ord("a"), ord("b")]) == "ab"
