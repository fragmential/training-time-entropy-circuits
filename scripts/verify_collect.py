#!/usr/bin/env python
"""Verify that the unified collection pipeline reproduces existing outputs.

To verify on server:
    1. Ensure pythia-14m-deduped activations exist: activations/fineweb/pythia-14m-deduped/step0.npy
    2. Run: python scripts/verify_collect.py --mode all

To verify locally (no existing data needed, CPU only):
    python scripts/verify_collect.py --mode self_consistency

Self-consistency tests run on CPU. Server tests use GPU if available.

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
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_registry import (
    get_model_config, load_model, load_tokenizer, get_mlp_projections,
    get_final_layernorm,
)
from utils.hooks import CovarianceCollector, ResidualCapture
from utils.data_utils import load_and_cache_texts, pack_sequences, compute_token_mask, compute_labels
from data import get_loader

MODEL = "EleutherAI/pythia-14m-deduped"
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
# Test 1: Identity head vs after_final_norm hook
# ---------------------------------------------------------------------------

def test_identity_vs_hook():
    """Verify identity_head and after_final_norm hook produce identical activations."""
    print("\n=== Test 1: identity_head vs after_final_norm hook ===")
    t0 = time.time()
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

    # Method 2: after_final_norm hook
    capture_hook = ResidualCapture(model, config, hook_point="after_final_norm")
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    acts_hook = capture_hook.activations[batch_indices, last_indices].float().numpy()
    capture_hook.restore(model)

    max_diff = np.max(np.abs(acts_identity - acts_hook))
    print(f"  Max diff (identity vs after_final_norm hook): {max_diff:.2e}")
    assert max_diff < ATOL, f"FAIL: max_diff={max_diff} > {ATOL}"
    print(f"  PASS ({time.time() - t0:.1f}s)")

    del model
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Test 2: before_final_norm + layernorm = after_final_norm
# ---------------------------------------------------------------------------

def test_before_norm_recovers_after_norm():
    """Verify that applying the final layernorm to before_final_norm output
    exactly recovers the after_final_norm output."""
    print("\n=== Test 2: before_final_norm + layernorm = after_final_norm ===")
    t0 = time.time()
    model, config = _load_model_cpu()
    texts, tokenizer = _get_texts_and_tokenizer()

    batch = texts[:4]
    tokenized = tokenizer(batch, padding="longest", return_tensors="pt", max_length=SEQ_LEN, truncation=True)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    capture_before = ResidualCapture(model, config, hook_point="before_final_norm")
    capture_after = ResidualCapture(model, config, hook_point="after_final_norm")

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)

    acts_before = capture_before.activations
    acts_after = capture_after.activations[0, 0].float().numpy()

    # Apply the final layernorm to before_final_norm activations (keep in model dtype)
    final_ln = get_final_layernorm(model, config)
    with torch.no_grad():
        acts_normed = final_ln(acts_before)
    acts_normed = acts_normed[0, 0].float().numpy()

    capture_before.close()
    capture_after.close()

    max_diff = np.max(np.abs(acts_normed - acts_after))
    print(f"  Max diff (layernorm(before) vs after): {max_diff:.2e}")
    assert max_diff < ATOL, f"FAIL: layernorm recovery failed, max_diff={max_diff} > {ATOL}"

    # Sanity: before and after should actually differ (norm changes values)
    raw_diff = np.max(np.abs(acts_before[0, 0].float().numpy() - acts_after))
    print(f"  Raw diff (before vs after, unnormed): {raw_diff:.2e}")
    assert raw_diff > 0.01, f"FAIL: before and after final norm should differ (diff={raw_diff})"
    print(f"  PASS ({time.time() - t0:.1f}s)")

    del model
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Test 3: CovarianceCollector produces correct shapes
# ---------------------------------------------------------------------------

def test_covariance_collector_shapes():
    """Verify CovarianceCollector produces correct tensor shapes."""
    print("\n=== Test 3: CovarianceCollector shapes ===")
    t0 = time.time()
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
    print(f"  PASS ({time.time() - t0:.1f}s)")

    del model
    gc.collect()
    return True


# ---------------------------------------------------------------------------
# Test 4: Token mask modes
# ---------------------------------------------------------------------------

def test_token_masks():
    """Verify token mask computation for different modes."""
    print("\n=== Test 4: Token mask modes ===")
    t0 = time.time()

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

    print(f"  PASS (all token mask modes correct, {time.time() - t0:.1f}s)")
    return True


# ---------------------------------------------------------------------------
# Test 5: Storage round-trip
# ---------------------------------------------------------------------------

def test_storage_roundtrip():
    """Verify cov → cov_svd → eigenvalues conversion preserves eigenvalues."""
    print("\n=== Test 5: Storage round-trip ===")
    t0 = time.time()
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

    print(f"  PASS ({time.time() - t0:.1f}s)")
    return True


# ---------------------------------------------------------------------------
# Test 6: collect.py vs extract_activations.py (server only)
# ---------------------------------------------------------------------------

def test_collect_vs_extract_activations():
    """Run both collect.py and extract_activations.py on pythia-14m-deduped step0,
    verify they produce identical results and time both. Also compare against
    stored activations at activations/fineweb/pythia-14m-deduped/step0.npy.
    """
    from scripts.collect import _collect_residual_for_checkpoint
    from rankme_alpha_scripts.extract_activations import (
        extract_activations, replace_output_head_with_identity,
    )

    print("\n=== Test 6: collect.py vs extract_activations.py ===")
    t0 = time.time()

    # Load text cache (required to guarantee exact reproducibility)
    import json
    for n in (5000, 2000, 50):
        cache_path = os.path.join("data", "cache", f"filtered_texts_fineweb_{n}.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                texts = json.load(f)
            break
    else:
        print("  SKIP: no text cache found")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(MODEL)
    tokenizer = load_tokenizer(config, revision="step0")
    batch_size = 128
    print(f"  Device: {device}, texts: {len(texts)}, batch_size: {batch_size}")

    # --- Old pipeline (extract_activations.py) ---
    model = load_model(config, config.hf_repo, "step0")
    replace_output_head_with_identity(model)
    model.to(device)

    t_old = time.time()
    old_acts = extract_activations(model, texts, tokenizer, "identity",
                                   max_length=512, batch_size=batch_size)
    t_old = time.time() - t_old

    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    gc.collect()

    # --- New pipeline (collect.py) ---
    model = load_model(config, config.hf_repo, "step0")
    model.to(device)

    t_new = time.time()
    new_acts = _collect_residual_for_checkpoint(
        model, config, texts, tokenizer,
        residual_hook_point="identity_head",
        token_selection="last",
        max_length=512,
        batch_size=batch_size,
        packing="padded",
        packed_ids=None,
        boundary_token_ids=None,
    )
    t_new = time.time() - t_new

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # --- Compare live results ---
    live_diff = np.max(np.abs(old_acts - new_acts))
    print(f"  Live comparison:")
    print(f"    Old shape: {old_acts.shape}, New shape: {new_acts.shape}")
    print(f"    Max diff: {live_diff:.2e}")
    print(f"    extract_activations.py: {t_old:.2f}s")
    print(f"    collect.py:             {t_new:.2f}s")

    ok = live_diff < ATOL
    print(f"    {'PASS' if ok else 'FAIL'}")

    # --- Compare against stored activations ---
    stored_path = "activations/fineweb/pythia-14m-deduped/step0.npy"
    if os.path.exists(stored_path):
        stored_acts = np.load(stored_path)
        diff_old = np.max(np.abs(old_acts - stored_acts))
        diff_new = np.max(np.abs(new_acts - stored_acts))
        print(f"  vs stored ({stored_path}):")
        print(f"    Stored shape: {stored_acts.shape}")
        print(f"    extract_activations diff: {diff_old:.2e}")
        print(f"    collect.py diff:          {diff_new:.2e}")
        stored_ok = diff_old < ATOL and diff_new < ATOL
        print(f"    {'PASS' if stored_ok else 'FAIL'}")
        ok = ok and stored_ok
    else:
        print(f"  SKIP stored comparison: {stored_path} not found")

    print(f"  Overall: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f}s)")
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
        results["before_norm_recovers_after_norm"] = test_before_norm_recovers_after_norm()
        results["covariance_shapes"] = test_covariance_collector_shapes()
        results["token_masks"] = test_token_masks()
        results["storage_roundtrip"] = test_storage_roundtrip()

    if args.mode in ("server", "all"):
        results["collect_vs_extract_activations"] = test_collect_vs_extract_activations()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_pass = True
    for name, result in results.items():
        status = "PASS" if result else ("SKIP" if result is None else "FAIL")
        print(f"  {name}: {status}")
        if result == False and result is not None:
            all_pass = False

    if all_pass:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
