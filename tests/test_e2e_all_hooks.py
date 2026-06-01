"""End-to-end coverage of EVERY hook kind on the smallest model of each family.

Drives the REAL collection pipeline (resolve_hook_specs -> HookCollector /
MultiHeadOVDispatcher -> forward + backward -> factors) via
scripts.collect._collect_for_checkpoint, then asserts that each requested hook
produced covariance factors of the right shape — including the per-OV-head and
the kwargs-called `attn.in` boundary paths that only a real model exercises
(and that the unit suite, which feeds tensors straight to the accumulator,
never touches).

Uses each model's final checkpoint so weights are cached once under $HF_HOME
and reused. Prefetch with:
    uv run python -m tests.prefetch_e2e_models

RUN ON A GPU NODE: OLMo-2-1B is bf16 and unusably slow on CPU. Skipped in fast
runs (mark: e2e). Submit with ./slurm/e2e.sh, or:
    srun --partition=gpu_a100 --gpus=1 --cpus-per-task=18 --time=00:20:00 \
        uv run pytest tests/test_e2e_all_hooks.py -v -m e2e
"""
import gc

import pytest
import torch

from tests.e2e_models import HOOKS_BY_MODEL, PYTHIA, OLMO, final_checkpoint
from collect import CollectConfig, _collect_for_checkpoint
from utils.model_registry import get_model_config, load_model, load_tokenizer
from utils.hook_names import classify, captured_signals

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


def _signal(name, quantity):
    """The captured 'name.acts' or 'name.grads' signal key for this hook, by kind."""
    return next(s for s in captured_signals(name) if s.endswith(f".{quantity}"))


def _assert_square_cov(name, fac, quantity):
    key = _signal(name, quantity)
    assert key in fac, f"{name}: missing {key!r}"
    cov = fac[key]
    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1], f"{name}: {key} not square: {tuple(cov.shape)}"
    assert torch.isfinite(cov).all(), f"{name}: {key} has non-finite values"


@pytest.mark.parametrize("model_name", [PYTHIA, OLMO])
def test_all_hooks_collect(model_name):
    """Every requested hook produces finite A (and G where +G) on a real forward/backward."""
    hooks = HOOKS_BY_MODEL[model_name]
    grad_patterns = {h[:-2] for h in hooks if h.endswith("+G")}
    factors = _run_collection(model_name, hooks)

    assert factors, "no factors collected"
    seen_kinds = set()
    for name, fac in factors.items():
        kind = classify(name)
        assert kind is not None, f"unclassifiable hook name: {name!r}"
        seen_kinds.add(kind)

        _assert_square_cov(name, fac, "acts")       # acts covariance: always present
        assert fac.get("n", 0) > 0, f"{name}: zero tokens accumulated"

        # grads present exactly when the matching pattern carried +G. (acts and
        # grads can differ in size — MLP K-FAC has d_in acts vs d_out grads.)
        wants_grad = _wants_grad(name, grad_patterns)
        if wants_grad:
            _assert_square_cov(name, fac, "grads")

    # Every kind the family exposes must have been collected.
    assert {"mlp", "ov_head", "boundary", "residual"} <= seen_kinds, f"missing kinds: {seen_kinds}"


def test_olmo_family_specific_hooks():
    """OLMo-2 exposes gate + raw_out + mlp.in, which Pythia silently lacks."""
    factors = _run_collection(OLMO, HOOKS_BY_MODEL[OLMO])
    names = set(factors)
    assert any(n.endswith(".gate") for n in names), "OLMo gate proj not collected"
    assert any(n.endswith(".attn.raw_out") for n in names), "OLMo attn.raw_out not collected"
    assert any(n.endswith(".mlp.in") for n in names), "OLMo mlp.in not collected"


def test_pythia_skips_olmo_only_hooks():
    """Pythia has no gate / raw_out / mlp.in — patterns for them are silent no-ops."""
    # Pass the OLMo (superset) patterns to Pythia; the extras must just not appear.
    factors = _run_collection(PYTHIA, HOOKS_BY_MODEL[OLMO])
    names = set(factors)
    assert not any(n.endswith(".gate") for n in names)
    assert not any(n.endswith(".attn.raw_out") for n in names)
    assert not any(n.endswith(".mlp.in") for n in names)
    # but the shared kinds are still there
    assert any(classify(n) == "ov_head" for n in names)


def _wants_grad(name, grad_patterns):
    import fnmatch
    return any(fnmatch.fnmatchcase(name, p) for p in grad_patterns)
