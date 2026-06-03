"""End-to-end coverage of EVERY hook kind on the smallest model of each family.

Drives the REAL collection pipeline (resolve_hook_specs -> HookCollector /
MultiHeadOVDispatcher -> forward + backward -> factors) via
scripts.collect._collect_for_checkpoint, then asserts that each produced leaf
carries covariance factors of the right shape — including the per-OV-head and
the kwargs-called `attn.in` boundary paths that only a real model exercises
(and that the unit suite, which feeds tensors straight to the accumulator,
never touches).

`_collect_for_checkpoint` returns {leaf: {quantity-keyed dict}} where leaves are
full nested paths (e.g. blk0.mlp.up.in, blk0.attn.in, blk0.attn.head0.slice), each
entry keyed by uniform `{q}_{fmt}` — acts_cov / acts_n / acts_mean / grads_cov / ...

Uses each model's final checkpoint so weights are cached once under $HF_HOME
and reused. Prefetch with:
    uv run python -m tests.prefetch_e2e_models

RUN ON A GPU NODE: OLMo-2-1B is bf16 and unusably slow on CPU. Skipped in fast
runs (mark: e2e). Run all e2e tests via srun:
    export HF_HOME="/projects/prjs1815/hf_cache"
    srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:40:00 \
        uv run pytest tests/ -m e2e
"""
import gc

import pytest
import torch

from tests.e2e_models import HOOKS_BY_MODEL, PYTHIA, OLMO, final_checkpoint
from scripts.collect import CollectConfig, _collect_for_checkpoint
from utils.model_registry import get_model_config, load_model, load_tokenizer
from utils import hook_names as hn

pytestmark = pytest.mark.e2e

# Two short blocks are enough to exercise every hook kind without paying for all
# layers of the 1B OLMo model.
TARGET_LAYERS = [0, 1]

TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank. " * 6,
    "In a hole in the ground there lived a hobbit who loved second breakfast. " * 6,
    "To be or not to be, that is the question that haunts every restless mind. " * 6,
    "Machine learning models are trained on large and diverse text corpora. " * 6,
]


def _run_collection(model_name, hooks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(model_name)
    hf_model, revision, _step = final_checkpoint(config)
    tokenizer = load_tokenizer(config, revision=revision)
    model = load_model(config, hf_model, revision).to(device).eval()
    cfg = CollectConfig(
        model_name=model_name,
        hooks=hooks,
        packing="padded",
        token_selection="all",
        batch_size=2,
        max_length=128,
        max_layers_per_pass=0,        # all target layers in one pass
        storage_format="cov_svd+m+o",  # means + O so collect_means is on
        sample_labels=False,           # deterministic label loss for the backward pass
    )
    try:
        factors = _collect_for_checkpoint(
            model, config, cfg, TEXTS, tokenizer, None,
            TARGET_LAYERS, device, None,
        )
    finally:
        del model
        gc.collect()
    return factors


def _leaf_kind(leaf):
    """Classify a storage leaf into one of the broad hook kinds, else None."""
    if leaf in hn.RESIDUAL_NAMES:
        return "residual"
    if leaf.endswith(".slice") or leaf.endswith(".contrib"):
        return "ov_head"
    if hn.mlp_proj_leaf(leaf):   # blk{i}.mlp.{up,down,gate}.{in,out}
        return "mlp"
    if hn.boundary(leaf):        # blk{i}.{attn,mlp}.{in,out,raw_out}
        return "boundary"
    return None


def _n(entry):
    """Token count for whichever quantity this leaf carries."""
    return entry.get("acts_n") or entry.get("grads_n") or 0


def _assert_square_cov(leaf, entry, quantity):
    key = f"{quantity}_cov"
    assert key in entry, f"{leaf}: missing {key!r}"
    cov = entry[key]
    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1], \
        f"{leaf}: {key} not square: {tuple(cov.shape)}"
    assert torch.isfinite(cov).all(), f"{leaf}: {key} has non-finite values"


@pytest.mark.parametrize("model_name", [PYTHIA, OLMO])
def test_all_hooks_collect(model_name):
    """Every requested leaf produces a finite square cov on a real forward/backward."""
    factors = _run_collection(model_name, HOOKS_BY_MODEL[model_name])

    assert factors, "no factors collected"
    seen_kinds = set()
    for leaf, entry in factors.items():
        kind = _leaf_kind(leaf)
        assert kind is not None, f"unclassifiable leaf: {leaf!r}"
        seen_kinds.add(kind)

        # Whatever quantity this leaf carries must be a finite square cov. MLP
        # K-FAC splits acts (.in) and grads (.out) across sibling leaves.
        if "acts_cov" in entry:
            _assert_square_cov(leaf, entry, "acts")
        if "grads_cov" in entry:
            _assert_square_cov(leaf, entry, "grads")
        assert "acts_cov" in entry or "grads_cov" in entry, f"{leaf}: no quantity stored"
        assert _n(entry) > 0, f"{leaf}: zero tokens accumulated"

    # Every kind the family exposes must have been collected.
    assert {"mlp", "ov_head", "boundary", "residual"} <= seen_kinds, f"missing kinds: {seen_kinds}"


def test_olmo_family_specific_hooks():
    """OLMo-2 exposes gate + raw_out + mlp.in, which Pythia silently lacks."""
    factors = _run_collection(OLMO, HOOKS_BY_MODEL[OLMO])
    names = set(factors)
    assert any(".mlp.gate." in n or n.endswith(".mlp.gate") for n in names), "OLMo gate proj not collected"
    assert any(n.endswith(".attn.raw_out") for n in names), "OLMo attn.raw_out not collected"
    assert any(n.endswith(".mlp.in") for n in names), "OLMo mlp.in not collected"


def test_pythia_skips_olmo_only_hooks():
    """Pythia has no gate / raw_out / mlp.in — patterns for them are silent no-ops."""
    # Pass the OLMo (superset) patterns to Pythia; the extras must just not appear.
    factors = _run_collection(PYTHIA, HOOKS_BY_MODEL[OLMO])
    names = set(factors)
    assert not any(".mlp.gate" in n for n in names)
    assert not any(n.endswith(".attn.raw_out") for n in names)
    assert not any(n.endswith(".mlp.in") for n in names)
    # but the shared kinds are still there
    assert any(_leaf_kind(n) == "ov_head" for n in names)
