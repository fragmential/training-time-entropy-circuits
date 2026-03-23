#!/usr/bin/env python
"""Verify that the unified collection pipeline reproduces existing outputs.

To verify on server:
    1. Ensure pythia-14m activations exist: activations/fineweb/pythia-14m/step0.npy
    2. Ensure pythia-14m kfac factors exist: kfac_factors/fineweb/pythia-14m/step0.pt
    3. Run: python scripts/verify_collect.py

To verify locally (no existing data needed, CPU only):
    python scripts/verify_collect.py --mode self_consistency

Tests run on CPU, use <100MB disk, take ~2-5 minutes.

For Claude on server:
    cd ~/Tracing-representation-geometry-reproduction
    python scripts/verify_collect.py --mode all
    # Expects exit code 0 if all checks pass.
"""

import argparse
import gc
import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_registry import (
    get_model_config, load_model, load_tokenizer, get_mlp_projections,
    get_num_layers, get_final_layernorm,
)
from utils.hooks import CovarianceCollector, ResidualCapture
from utils.data_utils import load_and_cache_texts, pack_sequences, compute_token_mask, compute_labels
from data import get_loader

MODEL = "EleutherAI/pythia-14m"
NUM_SAMPLES = 50
SEQ_LEN = 512
BATCH_SIZE = 16
ATOL = 1e-4  # tolerance for float32 comparisons


def _load_model_cpu():
    config = get_model_config(MODEL)
    model = load_model(config, config.hf_repo, "step0")
    return model, config


def _get_texts_and_tokenizer():
    config = get_model_config(MODEL)
    tokenizer = load_tokenizer(config, revision="step0")
    loader_fn = get_loader("fineweb")
    texts = load_and_cache_texts(loader_fn, NUM_SAMPLES, 32, tokenizer, "fineweb")
    return texts, tokenizer


# ---------------------------------------------------------------------------
# Test 1: Identity head vs hook post_norm
# ---------------------------------------------------------------------------

def test_identity_vs_hook():
    """Verify identity_head and post_norm hook produce identical post-norm activations."""
    print("\n=== Test 1: identity_head vs post_norm hook ===")
    model, config = _load_model_cpu()
    texts, tokenizer = _get_texts_and_tokenizer()

    batch = texts[:8]
    tokenized = tokenizer(batch, padding="longest", return_tensors="pt", max_length=SEQ_LEN, truncation=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    # Method 1: identity_head
    capture_id = ResidualCapture(model, config, hook_point="identity_head")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.size(0))
    acts_identity = out.logits[batch_indices, last_indices].float().numpy()
    capture_id.restore(model)

    # Method 2: post_norm hook
    capture_hook = ResidualCapture(model, config, hook_point="post_norm")
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    acts_hook = capture_hook.activations[batch_indices, last_indices].float().numpy()
    capture_hook.restore(model)

    max_diff = np.max(np.abs(acts_identity - acts_hook))
    print(f"  Max diff (identity vs hook post_norm): {max_diff:.2e}")
    assert max_diff < ATOL, f"FAIL: max_diff={max_diff} > {ATOL}"
    print("  PASS")

    del model
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Test 2: Pre-norm hook captures raw residual (not normed)
# ---------------------------------------------------------------------------

def test_pre_norm_hook():
    """Verify pre_norm hook captures the raw residual (different from post_norm)."""
    print("\n=== Test 2: pre_norm hook captures raw residual ===")
    model, config = _load_model_cpu()
    texts, tokenizer = _get_texts_and_tokenizer()

    batch = texts[:4]
    tokenized = tokenizer(batch, padding="longest", return_tensors="pt", max_length=SEQ_LEN, truncation=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    capture_pre = ResidualCapture(model, config, hook_point="pre_norm")
    capture_post = ResidualCapture(model, config, hook_point="post_norm")

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    acts_pre = capture_pre.activations[0, 0].float().numpy()
    acts_post = capture_post.activations[0, 0].float().numpy()

    capture_pre.close()
    capture_post.close()

    # They should be different (layernorm changes the values)
    diff = np.max(np.abs(acts_pre - acts_post))
    print(f"  Max diff (pre_norm vs post_norm): {diff:.2e}")
    assert diff > 0.01, f"FAIL: pre and post norm should differ (diff={diff})"
    print("  PASS (pre and post norm differ as expected)")

    del model
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Test 3: CovarianceCollector produces correct shapes
# ---------------------------------------------------------------------------

def test_covariance_collector_shapes():
    """Verify CovarianceCollector produces correct tensor shapes."""
    print("\n=== Test 3: CovarianceCollector shapes ===")
    model, config = _load_model_cpu()
    texts, tokenizer = _get_texts_and_tokenizer()

    packed = pack_sequences(texts, tokenizer, SEQ_LEN)
    x = packed[:4]  # 4 chunks

    # Get MLP projections for block 0
    projections = get_mlp_projections(model, config, 0)
    name0, layer0 = projections[0]
    d_out, d_in = layer0.weight.shape

    collector = CovarianceCollector(layer0, collect_A=True, collect_G=True, collect_B=True)
    mask = compute_token_mask(x, token_selection="all")
    collector.set_token_mask(mask)

    # Need backward for G
    for p in model.parameters():
        p.requires_grad_(False)
    layer0.weight.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(x).logits[:, :-1].float()
    labels = compute_labels(x)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels[:, :-1].reshape(-1),
        ignore_index=-100,
    )
    loss.backward()

    factors = collector.factors()
    collector.close()

    print(f"  {name0}: d_in={d_in}, d_out={d_out}")
    print(f"  A shape: {factors['A'].shape} (expected ({d_in}, {d_in}))")
    print(f"  G shape: {factors['G'].shape} (expected ({d_out}, {d_out}))")
    print(f"  B shape: {factors['B'].shape} (expected ({d_out}, {d_out}))")
    print(f"  n = {factors['n']}")

    assert factors["A"].shape == (d_in, d_in), f"FAIL: A shape {factors['A'].shape}"
    assert factors["G"].shape == (d_out, d_out), f"FAIL: G shape {factors['G'].shape}"
    assert factors["B"].shape == (d_out, d_out), f"FAIL: B shape {factors['B'].shape}"
    assert factors["n"] > 0, "FAIL: n should be > 0"
    print("  PASS")

    del model
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Test 4: Token mask modes
# ---------------------------------------------------------------------------

def test_token_masks():
    """Verify token mask computation for different modes."""
    print("\n=== Test 4: Token mask modes ===")

    # Test all-tokens packed
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    mask = compute_token_mask(ids, token_selection="all")
    assert mask.shape == (1, 5)
    assert mask.tolist() == [[True, True, True, True, False]], f"FAIL: all-tokens packed: {mask.tolist()}"

    # Test last-token padded
    ids = torch.tensor([[1, 2, 3, 0, 0]])
    attn = torch.tensor([[1, 1, 1, 0, 0]])
    mask = compute_token_mask(ids, attention_mask=attn, token_selection="last")
    assert mask.tolist() == [[False, False, True, False, False]], f"FAIL: last-token padded: {mask.tolist()}"

    # Test all-tokens padded (excludes last real + all padding)
    mask = compute_token_mask(ids, attention_mask=attn, token_selection="all")
    assert mask.tolist() == [[True, True, False, False, False]], f"FAIL: all-tokens padded: {mask.tolist()}"

    # Test skip_positions
    ids = torch.tensor([[1, 50256, 2, 3, 4]])  # 50256 = boundary
    mask = compute_token_mask(ids, token_selection="all", skip_positions=1, boundary_token_ids=[50256])
    # Position 2 (after boundary at 1) should be skipped
    assert mask[0, 2].item() == False, f"FAIL: skip_positions should mask position after boundary"

    print("  PASS (all token mask modes correct)")
    return True


# ---------------------------------------------------------------------------
# Test 5: Storage round-trip
# ---------------------------------------------------------------------------

def test_storage_roundtrip():
    """Verify cov → cov_svd → eigenvalues conversion preserves eigenvalues."""
    print("\n=== Test 5: Storage round-trip ===")
    from utils.storage import save_factors, convert, info

    # Create fake factors
    d = 32
    n = 100
    X = torch.randn(n, d)
    A = X.T @ X  # unnormalized covariance
    factors = {"blk0.up": {"A": A, "n": n, "n_A": n}}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as cov
        cov_path = os.path.join(tmpdir, "test_cov.pt")
        save_factors(factors, cov_path, storage_format="cov")

        # Convert to cov_svd
        svd_path = os.path.join(tmpdir, "test_svd.pt")
        convert(cov_path, "cov_svd", svd_path)

        # Convert to eigenvalues
        eig_path = os.path.join(tmpdir, "test_eig.pt")
        convert(svd_path, "eigenvalues", eig_path)

        # Load and compare
        svd_data = torch.load(svd_path, map_location="cpu", weights_only=False)
        eig_data = torch.load(eig_path, map_location="cpu", weights_only=False)

        svd_eigvals = svd_data["blk0.up"]["A_eigvals"]
        eig_eigvals = eig_data["blk0.up"]["A_eigvals"]

        max_diff = (svd_eigvals - eig_eigvals).abs().max().item()
        print(f"  Max diff (cov_svd vs eigenvalues): {max_diff:.2e}")
        assert max_diff < 1e-5, f"FAIL: eigenvalues differ after conversion: {max_diff}"

        # Verify we can reconstruct covariance from svd
        V = svd_data["blk0.up"]["A_eigvecs"]
        S = svd_data["blk0.up"]["A_eigvals"]
        A_reconstructed = V @ torch.diag(S) @ V.T
        A_original = A.float() / n

        recon_diff = (A_reconstructed - A_original).abs().max().item()
        print(f"  Max reconstruction diff (V diag(λ) V^T vs A/n): {recon_diff:.2e}")
        assert recon_diff < 1e-4, f"FAIL: reconstruction error: {recon_diff}"

        # Print info
        print(info(svd_path))

    print("  PASS")
    return True


# ---------------------------------------------------------------------------
# Test 6: Residual activations match extract_activations.py output (server only)
# ---------------------------------------------------------------------------

def test_residual_vs_extract_activations():
    """Re-run identity head on pythia-14m step0, compare against
    activations/fineweb/pythia-14m/step0.npy produced by
    rankme_alpha_scripts/extract_activations.py.
    """
    print("\n=== Test 6: Residual vs rankme_alpha_scripts/extract_activations.py ===")
    ref_path = "activations/fineweb/pythia-14m/step0.npy"
    if not os.path.exists(ref_path):
        print(f"  SKIP: {ref_path} not found (run on server with existing data)")
        return None

    ref_acts = np.load(ref_path)
    n_ref = ref_acts.shape[0]
    print(f"  Reference: {ref_path} shape={ref_acts.shape}")

    # Need the exact same text cache that extract_activations.py used
    import json
    cache_path = os.path.join("data", "cache", f"filtered_texts_fineweb_{n_ref}.json")
    if not os.path.exists(cache_path):
        print(f"  SKIP: text cache {cache_path} not found (need same data as original run)")
        return None
    with open(cache_path) as f:
        texts = json.load(f)

    model, config = _load_model_cpu()
    tokenizer = load_tokenizer(config, revision="step0")

    capture = ResidualCapture(model, config, hook_point="identity_head")
    all_acts = []

    with torch.no_grad():
        for bidx in range(0, len(texts), BATCH_SIZE):
            batch = texts[bidx : bidx + BATCH_SIZE]
            tokenized = tokenizer(
                batch, padding="longest", return_tensors="pt",
                max_length=512, truncation=True,
            )
            out = model(input_ids=tokenized.input_ids, attention_mask=tokenized.attention_mask)
            last_idx = tokenized.attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(out.logits.size(0))
            all_acts.append(out.logits[batch_idx, last_idx].float().numpy())

    capture.restore(model)
    new_acts = np.vstack(all_acts)[:n_ref]

    max_diff = np.max(np.abs(ref_acts[:len(new_acts)] - new_acts))
    print(f"  New shape: {new_acts.shape}")
    print(f"  Max diff: {max_diff:.2e}")

    ok = max_diff < ATOL
    print(f"  {'PASS' if ok else 'FAIL'}")

    del model
    gc.collect()
    return ok


# ---------------------------------------------------------------------------
# Test 7: Input covariance A matches collect_kfac.py output (server only)
# ---------------------------------------------------------------------------

def test_input_cov_vs_collect_kfac():
    """Re-collect input covariance A on pythia-14m step0, compare against
    kfac_factors/fineweb/pythia-14m/step0.pt produced by
    kfac_scripts/collect_kfac.py.

    A comes from the forward hook (input activations) and does not depend on
    the loss or labels, so it is fully deterministic and comparable regardless
    of how G was collected in the original run.
    """
    print("\n=== Test 7: Input cov (A) vs kfac_scripts/collect_kfac.py ===")
    ref_path = "kfac_factors/fineweb/pythia-14m/step0.pt"
    if not os.path.exists(ref_path):
        print(f"  SKIP: {ref_path} not found (run on server with existing data)")
        return None

    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
    first_proj = next(iter(ref_data))
    ref_A = ref_data[first_proj]["A"].float()
    ref_n = ref_data[first_proj]["n"]
    print(f"  Reference: {ref_path} proj={first_proj} A={ref_A.shape} n={ref_n}")

    model, config = _load_model_cpu()
    tokenizer = load_tokenizer(config, revision="step0")

    # Load same cached texts
    import json
    cache_5000 = os.path.join("data", "cache", "filtered_texts_fineweb_5000.json")
    cache_50 = os.path.join("data", "cache", "filtered_texts_fineweb_50.json")
    for cache_path in (cache_5000, cache_50):
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                texts = json.load(f)
            break
    else:
        print("  SKIP: no matching text cache found")
        del model; gc.collect()
        return None

    packed = pack_sequences(texts, tokenizer, SEQ_LEN)
    print(f"  Packed: {packed.shape[0]} chunks of {SEQ_LEN}")

    # Collect A only (forward-only, no backward needed)
    block_idx = int(first_proj.split(".")[0].replace("blk", ""))
    projections = get_mlp_projections(model, config, block_idx)
    target_name, target_layer = next((n, l) for n, l in projections if n == first_proj)

    collector = CovarianceCollector(target_layer, collect_A=True, collect_G=False, collect_B=False)

    with torch.no_grad():
        for i in range(0, min(packed.size(0), 64), BATCH_SIZE):  # cap at 64 chunks
            x = packed[i : i + BATCH_SIZE]
            mask = compute_token_mask(x, token_selection="all")
            collector.set_token_mask(mask)
            model(x)

    factors = collector.factors()
    collector.close()

    new_A = factors["A"].float()
    new_n = factors["n"]

    # Compare normalized covariance E[xxT] = A/n
    ref_norm = ref_A / ref_n
    new_norm = new_A / new_n

    max_diff = (ref_norm - new_norm).abs().max().item()
    print(f"  New: A={new_A.shape} n={new_n}")
    print(f"  Max diff (E[xxT]): {max_diff:.2e}")

    # n will differ if we process fewer chunks than the original run,
    # but E[xxT] should converge to the same value on the same data prefix.
    ok = max_diff < 0.1
    if ok:
        print("  PASS")
    else:
        print(f"  WARN: max_diff={max_diff:.2e} (may differ if text cache or chunk count differs)")

    del model
    gc.collect()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify unified collection pipeline")
    parser.add_argument("--mode", default="self_consistency",
                        choices=["self_consistency", "server", "all"],
                        help="self_consistency: no existing data needed. "
                             "server: compare against existing data. "
                             "all: both.")
    args = parser.parse_args()

    results = {}

    if args.mode in ("self_consistency", "all"):
        results["identity_vs_hook"] = test_identity_vs_hook()
        results["pre_norm_hook"] = test_pre_norm_hook()
        results["covariance_shapes"] = test_covariance_collector_shapes()
        results["token_masks"] = test_token_masks()
        results["storage_roundtrip"] = test_storage_roundtrip()

    if args.mode in ("server", "all"):
        results["residual_vs_extract_activations"] = test_residual_vs_extract_activations()
        results["input_cov_vs_collect_kfac"] = test_input_cov_vs_collect_kfac()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result else ("SKIP" if result is None else "FAIL")
        print(f"  {name}: {status}")
        if result is False:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
