"""Whole-process integration (GPU): one REAL collection of every hook kind on
pythia-14m, then driven through the full save-flag matrix and compute_metrics.

Asserts two things, both decided up front:
  1. the EXACT stored-key sets and node->metric structure for every input variant
     (all kinds / boundary-grads on+off / subset / each storage flag), and
  2. metric VALUES on a FULL-RANK corpus against a reference minted by `main`
     (the trusted pre-rebuild code) — see test_alpha_fullrank_reference.

The single collection (forward+backward) feeds every case — boundary-grads-off and
subsets are derived by filtering the real factors dict, so it stays one GPU pass.

RUN ON A GPU NODE (mark: e2e). The value reference (tests/snapshots/pipeline_fullrank.pt)
is generated on `main`, not via --snapshot-update here.
"""
import gc
import os

import pytest
import torch

from tests.e2e_models import PYTHIA, final_checkpoint
from scripts.collect import CollectConfig, _collect_for_checkpoint, _parse_storage_format
from utils.model_registry import get_model_config, load_model, load_tokenizer, ModelWeights
from utils.accessor import DataAccessor
from scripts.compute_metrics import compute_metrics_for_checkpoint

pytestmark = pytest.mark.e2e
_ID = (PYTHIA, "final", "test")   # save asserts a complete identity

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

# Diverse, non-repetitive corpus -> full-rank up.in spectrum (npos=128) -> the alpha fit
# window pins to arange(11,100) -> alpha is reproducible across implementations/hardware.
# MUST stay identical to scripts/gen_fullrank_ref.py (used to mint the `main` reference).
FULLRANK_TEXTS = [
    "The mitochondrion is the powerhouse of the cell, generating adenosine triphosphate through oxidative phosphorylation across the inner membrane, where electron transport chains pump protons to establish an electrochemical gradient that drives synthesis.",
    "In fourteen fifty-three the fall of Constantinople marked the end of the Byzantine Empire, as Ottoman cannons breached the ancient Theodosian walls after a siege that had lasted nearly two relentless months.",
    "She wandered through the abandoned orchard at dusk, fingertips brushing the rough bark of forgotten apple trees, while distant thunder rolled across the violet hills and the first cold raindrops began to fall.",
    "Quicksort partitions an array around a chosen pivot element, recursively sorting the subarrays on either side; its average case runs in n log n time, though an unlucky pivot degrades it toward quadratic behaviour.",
    "The monsoon winds reverse direction with the seasons, drawing moist air inland from the warm Indian Ocean through summer and then pushing dry continental air back out to sea during the cooler winter months.",
    "Jazz improvisation rewards both discipline and abandon: a soloist internalises the chord changes and scales so thoroughly that the conscious mind can step aside, letting fresh melody emerge above the steady rhythm section.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy captured by chlorophyll, releasing oxygen as a byproduct and forming the base of nearly every terrestrial and aquatic food web on earth.",
    "The cartographer unrolled a brittle parchment across the table, tracing faded coastlines with a trembling finger, certain that the uncharted strait he had glimpsed would finally connect the two distant trading empires.",
    "The printing press, refined by Gutenberg around fourteen forty, dramatically lowered the cost of reproducing books, accelerating the spread of literacy, scientific exchange, and dissenting religious ideas across a fractious Europe.",
    "Tectonic plates drift atop the ductile asthenosphere, colliding to raise mountain ranges, pulling apart to open ocean basins, and grinding past one another along faults where stress accumulates until it releases as earthquakes.",
    "He debugged the flaky test for hours before realising the race condition lived not in his own code but in a shared cache that two worker processes kept mutating without ever acquiring the advisory lock.",
    "The chef reduced the stock over low heat until it coated the back of a wooden spoon, then whisked in cold butter piece by piece, emulsifying a glossy sauce brightened at the last moment with lemon.",
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
    return DataAccessor(factors, config=config, weights=ModelWeights(model, config), identity=_ID)

def _leaves(factors):
    return [k for k in factors if not k.startswith("__")]

def _captured(factors):
    return {(leaf, q) for leaf in _leaves(factors)
            for q in ("acts", "grads") if f"{q}_gram" in factors[leaf]}

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
    base, overrides = _parse_storage_format(fmt)
    DataAccessor(factors, config=config, weights=ModelWeights(model, config),
                 identity=_ID).save(out, format=base, overrides=overrides)
    return torch.load(out, map_location="cpu", weights_only=False)

def _structure(res):
    return {n: set(m) for n, m in res.items() if not str(n).startswith("__")}


# ===========================================================================
# Keys decided up front
# ===========================================================================

_PRIMARY = {"cov": "{q}_gram", "cov_svd": "{q}_eigvals", "eigenvalues": "{q}_eigvals"}

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
    # (means are in every format now — no -m strip path; that capability was dropped)


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
        DataAccessor(stripped, config=config, weights=ModelWeights(model, config))))
    assert s["blk0.attn"] == {"mean_metrics_blk_vs_res"}  # lost grads -> no cross kfac; acts mean-metric stays
    assert s["blk0.mlp"] == {"projections_kfac", "mean_metrics_blk_vs_res"}
    assert s["blk0.attn.in"] == SPEC_A             # boundary now acts-only
    assert s["blk0.mlp.up"] == {"kfac"}            # mlp K-FAC unaffected


def test_structure_subset_mlp_up_only(collected):
    factors, model, config = collected
    sub = _copy(factors, keep=lambda l: l.startswith("blk0.mlp.up."))
    s = _structure(compute_metrics_for_checkpoint(
        DataAccessor(sub, config=config, weights=ModelWeights(model, config))))
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
        assert "grads_gram" in factors[leaf], f"{leaf}: grad hook captured no gradients"


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
# Metric VALUES: full-rank cross-implementation reference (vs `main`, trusted old code)
# ===========================================================================
# alpha is ill-posed on the rank-deficient TEXTS above (its fit window arange(11,min(100,npos))
# straddles the rank floor, npos ~73-92, so it swings 7.98<->9.80 across impl/hardware with no
# true value). On full-rank input (npos=128) the window pins to arange(11,100) and the fit is
# clean (r2~0.84) -> alpha is reproducible. tests/snapshots/pipeline_fullrank.pt was minted on
# `main` (the trusted pre-rebuild code) from FULLRANK_TEXTS; the rebuild must reproduce it.

def test_alpha_fullrank_reference(collected):
    _, model, config = collected
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, rev, _ = final_checkpoint(config)
    tok = load_tokenizer(config, revision=rev)
    cfg = CollectConfig(model_name=PYTHIA, hooks=["blk*.mlp.up.in:acts"], packing="padded",
                        token_selection="all", batch_size=2, max_length=128,
                        max_layers_per_pass=0, sample_labels=False, seed=42)
    factors = _collect_for_checkpoint(model, config, cfg, FULLRANK_TEXTS, tok, None,
                                      TARGET_LAYERS, device, None)
    m = compute_metrics_for_checkpoint(
        DataAccessor(factors, config=config, weights=ModelWeights(model, config), identity=_ID),
    )["blk0.mlp.up.in"]["acts_uncentered"]
    npos = int((torch.as_tensor(m["eigenspectrum"]) > 0).sum())
    assert npos >= 110, f"corpus not full-rank (npos={npos}) — alpha would be ill-posed"
    ref = torch.load(os.path.join(os.path.dirname(__file__), "snapshots", "pipeline_fullrank.pt"),
                     map_location="cpu", weights_only=False)["blk0.mlp.up.in|acts_uncentered"]
    tol = {"alpha": (2e-2, 2e-2), "rankme": (0.0, 5e-3), "true_rankme": (0.0, 5e-3), "r2": (2e-2, 0.0)}
    for k, (atol, rtol) in tol.items():
        assert abs(float(m[k]) - ref[k]) <= atol + rtol * abs(ref[k]), \
            f"{k}: rebuild {float(m[k]):.5f} vs main-reference {ref[k]:.5f}"
