"""Whole-process integration (GPU): one REAL collection of every hook kind on
pythia-14m, then driven through the full save-flag matrix and compute_metrics.

Asserts two things, both decided up front:
  1. the EXACT stored-key sets and node->metric structure for every input variant
     (all kinds / boundary-grads on+off / subset / each storage flag), and
  2. the actual metric VALUES against a committed snapshot (numeric regression).

The single collection (forward+backward) feeds every case — boundary-grads-off and
subsets are derived by filtering the real factors dict, so it stays one GPU pass.

RUN ON A GPU NODE (mark: e2e). First run / --snapshot-update writes the value
snapshot to tests/snapshots/pipeline_values.pt; commit it.
"""
import gc
import os

import pytest
import torch

from tests.e2e_models import PYTHIA, final_checkpoint
from scripts.collect import CollectConfig, _collect_for_checkpoint
from utils.model_registry import get_model_config, load_model, load_tokenizer
from utils.accessor import DataAccessor
from scripts.compute_metrics import compute_metrics_for_checkpoint

pytestmark = pytest.mark.e2e

TARGET_LAYERS = [0, 1]
TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank. " * 6,
    "In a hole in the ground there lived a hobbit who loved second breakfast. " * 6,
    "To be or not to be, that is the question that haunts every restless mind. " * 6,
    "Machine learning models are trained on large and diverse text corpora. " * 6,
]
# Every hook kind, with grads on ALL of them — incl. input-side boundaries
# (attn.in), which capture grads correctly once the grad path is set up.
HOOKS = [
    "preset:kfac",                # blk*.mlp.{up,down}.in:acts + .out:grads (pythia: no gate)
    "blk*.attn.head*.slice:both",
    "blk*.attn.in:both", "blk*.attn.out:both", "blk*.mlp.out:both",
    "before_final_norm:both", "after_final_norm:both",
]


@pytest.fixture(scope="module")
def collected():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(PYTHIA)
    hf, rev, _ = final_checkpoint(config)
    tokenizer = load_tokenizer(config, revision=rev)
    model = load_model(config, hf, rev).to(device).eval()
    cfg = CollectConfig(
        model_name=PYTHIA, hooks=HOOKS, packing="padded", token_selection="all",
        batch_size=2, max_length=128, max_layers_per_pass=0,
        storage_format="cov_svd+m+o", sample_labels=False, seed=42,
    )
    try:
        factors = _collect_for_checkpoint(model, config, cfg, TEXTS, tokenizer, None,
                                          TARGET_LAYERS, device, None)
    finally:
        pass
    yield factors, model, config
    del model
    gc.collect()


# --- helpers over the real factors dict ---

def _acc(collected):
    factors, model, config = collected
    return DataAccessor(factors, model=model, model_config=config)

def _leaves(factors):
    return [k for k in factors if not k.startswith("__")]

def _captured(factors):
    return {(leaf, q) for leaf in _leaves(factors)
            for q in ("acts", "grads") if f"{q}_cov" in factors[leaf]}

def _copy(factors, keep=None, strip_boundary_grads=False):
    out = {"__format__": "cov", "__hf_model__": factors.get("__hf_model__")}
    for leaf in _leaves(factors):
        if keep is not None and not keep(leaf):
            continue
        e = dict(factors[leaf])
        if strip_boundary_grads and (".attn.in" in leaf or ".attn.out" in leaf or leaf.endswith(".mlp.out")):
            e = {k: v for k, v in e.items() if not k.startswith("grads")}
        out[leaf] = e
    return out

def _save(collected, factors, fmt, tmp_path):
    _, model, config = collected
    out = str(tmp_path / (fmt.replace("+", "p").replace("-", "m") + ".pt"))
    DataAccessor(factors, model=model, model_config=config).save(out, format=fmt)
    return torch.load(out, map_location="cpu", weights_only=False)

def _structure(res):
    return {n: set(m) for n, m in res.items() if not str(n).startswith("__")}


# ===========================================================================
# Keys decided up front
# ===========================================================================

_PRIMARY = {"cov": "{q}_cov", "cov_svd": "{q}_eigvals", "eigenvalues": "{q}_eigvals"}

@pytest.mark.parametrize("fmt", ["cov", "cov_svd", "eigenvalues"])
def test_lossless_preserves_every_captured_quantity(collected, fmt, tmp_path):
    factors = collected[0]
    data = _save(collected, factors, fmt, tmp_path)
    for leaf, q in _captured(factors):
        key = _PRIMARY[fmt].format(q=q)
        assert leaf in data and key in data[leaf], f"{fmt}: captured {leaf}.{q} lost"


def _svd_keys(q):
    return {f"{q}_eigvals", f"{q}_eigvecs", f"{q}_n", f"{q}_eigvals_centered", f"{q}_mean"}

def test_exact_keys_cov_svd(collected, tmp_path):
    d = _save(collected, collected[0], "cov_svd", tmp_path)
    assert set(d["blk0.attn.head0.slice"]) == _svd_keys("acts") | _svd_keys("grads")
    assert set(d["blk0.mlp.up.in"]) == _svd_keys("acts")    # captured acts only
    assert set(d["blk0.mlp.up.out"]) == _svd_keys("grads")  # grads captured; acts derived, unstored
    assert "blk0.attn.head0.contrib" not in d               # derived, no +o


def test_flag_matrix(collected, tmp_path):
    f = collected[0]
    assert "acts_eigvals" not in _save(collected, f, "cov_svd", tmp_path)["blk0.mlp.up.out"]
    assert "acts_eigvals" in _save(collected, f, "cov_svd+b", tmp_path)["blk0.mlp.up.out"]
    assert "blk0.attn.head0.contrib" not in _save(collected, f, "cov_svd", tmp_path)
    assert "acts_eigvals" in _save(collected, f, "cov_svd+o", tmp_path)["blk0.attn.head0.contrib"]
    ev = _save(collected, f, "eigenvalues", tmp_path)
    assert "acts_eigvals" in ev["blk0.mlp.up.out"] and "blk0.attn.head0.contrib" in ev  # +b/+o implied
    assert "acts_eigvals" not in _save(collected, f, "eigenvalues-b", tmp_path)["blk0.mlp.up.out"]
    assert "blk0.attn.head0.contrib" not in _save(collected, f, "eigenvalues-o", tmp_path)
    nm = _save(collected, f, "cov_svd-m", tmp_path)
    assert not any(k.endswith("_mean") for leaf, e in nm.items()
                   if not leaf.startswith("__") for k in e)


# ===========================================================================
# compute_metrics node -> metric structure (decided up front)
# ===========================================================================

SPEC_A = {"acts_uncentered", "acts_centered", "acts_mean_metrics", "acts_mean_vec"}
SPEC_G = {"grads_uncentered", "grads_centered", "grads_mean_metrics", "grads_mean_vec"}
BOTH = SPEC_A | SPEC_G | {"gen"}

def test_structure_boundary_grads_on(collected):
    s = _structure(compute_metrics_for_checkpoint(_acc(collected)))
    assert s["blk0.mlp.up"] == {"kfac"} and s["blk0.mlp.down"] == {"kfac"}
    # pythia mlp.in aliases to attn.in (shared parallel-residual input), so the
    # cross-boundary kfac (mlp.in.acts x mlp.out.grads) fires alongside projections_kfac.
    assert s["blk0.mlp"] == {"kfac", "projections_kfac", "mean_metrics_blk_vs_res"}
    assert s["blk0.attn"] == {"kfac", "mean_metrics_blk_vs_res"}  # cross kfac + acts mean in-vs-out
    assert s["blk0.mlp.up.in"] == SPEC_A
    assert s["blk0.mlp.up.out"] == BOTH            # derived acts + captured grads
    assert s["blk0.attn.head0.slice"] == BOTH
    assert s["blk0.attn.head0.contrib"] == SPEC_A  # derived acts only
    # every :both boundary/residual point captures grads -> full acts+grads+gen
    for leaf in ("blk0.attn.in", "blk0.attn.out", "blk0.mlp.out",
                 "before_final_norm", "after_final_norm"):
        assert s[leaf] == BOTH, leaf


def test_structure_boundary_grads_off(collected):
    factors, model, config = collected
    stripped = _copy(factors, strip_boundary_grads=True)
    s = _structure(compute_metrics_for_checkpoint(
        DataAccessor(stripped, model=model, model_config=config)))
    assert s["blk0.attn"] == {"mean_metrics_blk_vs_res"}  # lost grads -> no cross kfac; acts mean-metric stays
    assert s["blk0.mlp"] == {"projections_kfac", "mean_metrics_blk_vs_res"}
    assert s["blk0.attn.in"] == SPEC_A             # boundary now acts-only
    assert s["blk0.mlp.up"] == {"kfac"}            # mlp K-FAC unaffected


def test_structure_subset_mlp_up_only(collected):
    factors, model, config = collected
    sub = _copy(factors, keep=lambda l: l.startswith("blk0.mlp.up."))
    s = _structure(compute_metrics_for_checkpoint(
        DataAccessor(sub, model=model, model_config=config)))
    assert s["blk0.mlp.up"] == {"kfac"}
    assert "projections_kfac" not in s.get("blk0.mlp", set())  # no down -> none
    assert s["blk0.mlp.up.in"] == SPEC_A


def test_every_grad_leaf_actually_has_grads(collected):
    # Guard against silent grad drops: every leaf asked for grads must carry them
    # (this is what caught input-side boundaries silently collecting none).
    factors = collected[0]
    for leaf in ("blk0.mlp.up.out", "blk0.mlp.down.out",
                 "blk0.attn.head0.slice", "blk0.attn.head1.slice",
                 "blk0.attn.in", "blk0.attn.out", "blk0.mlp.out",
                 "before_final_norm", "after_final_norm"):
        assert "grads_cov" in factors[leaf], f"{leaf}: grad hook captured no gradients"


def test_fast_final_norm_errors_when_grads_requested(collected):
    # fast_final_norm replaces the output head with Identity -> no backward pass.
    # Requesting any grads alongside it must hard-error (not silently drop grads).
    _, model, config = collected
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = CollectConfig(
        model_name=PYTHIA, fast_final_norm=True,
        hooks=["after_final_norm:acts", "blk*.mlp.up.out:grads"],
        packing="padded", max_layers_per_pass=0, sample_labels=False,
    )
    with pytest.raises(ValueError, match="fast_final_norm"):
        _collect_for_checkpoint(model, config, cfg, TEXTS, None, None,
                                TARGET_LAYERS, device, None)


def test_empty_collection_no_metrics():
    assert _structure(compute_metrics_for_checkpoint(DataAccessor({"__format__": "cov"}))) == {}


# ===========================================================================
# Metric VALUES vs committed snapshot (numeric regression)
# ===========================================================================

def _snapshot(name, current, request, atol=1e-4, rtol=2e-3):
    path = os.path.join(os.path.dirname(__file__), "snapshots", f"{name}.pt")
    if request.config.getoption("--snapshot-update", default=False) or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(current, path)
        pytest.skip("snapshot created/updated")
    ref = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in current.items():
        r = ref[k]
        if isinstance(v, torch.Tensor):
            assert torch.allclose(v.double().cpu(), torch.as_tensor(r).double(), atol=atol, rtol=rtol), \
                f"{k}: tensor mismatch"
        else:
            assert abs(v - r) <= atol + rtol * abs(r), f"{k}: {v} vs {r}"


def test_metric_values_snapshot(collected, request):
    s = compute_metrics_for_checkpoint(_acc(collected))
    cur = {}
    def grab(node, metric, *fields):
        for f in fields:
            cur[f"{node}|{metric}|{f}"] = s[node][metric][f]
    grab("blk0.mlp.up.in", "acts_uncentered", "rankme", "alpha", "log_det", "trace")
    grab("blk0.mlp.up.out", "grads_uncentered", "rankme", "log_det")
    grab("blk0.mlp.up.out", "gen", "trace", "rankme")
    grab("blk0.mlp.up", "kfac", "log_det", "trace")
    grab("blk0.mlp", "projections_kfac", "log_det", "trace")
    grab("blk0.attn.head0.slice", "acts_uncentered", "rankme")
    grab("blk0.attn.head0.contrib", "acts_uncentered", "rankme", "d")
    grab("blk0.attn.in", "acts_uncentered", "rankme")
    grab("after_final_norm", "acts_uncentered", "rankme", "log_det")
    grab("after_final_norm", "gen", "trace")
    cur["blk0.mlp.up.in|acts_uncentered|spectrum10"] = \
        torch.as_tensor(s["blk0.mlp.up.in"]["acts_uncentered"]["eigenspectrum"][:10]).double().cpu()
    _snapshot("pipeline_values", cur, request)
