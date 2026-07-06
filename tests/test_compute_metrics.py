import numpy as np
import torch
from scripts.compute_metrics import (
    spectral_metrics, mean_metrics, kfac_metrics,
    generalized_eigenvalues, _top_k_outer_products,
    compute_metrics_for_checkpoint, _merge_step, _to_numpy,
)
from utils.accessor import DataAccessor


def test_spectral_metrics_keys():
    eigvals = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1] * 20)  # 120 values
    out = spectral_metrics(eigvals)
    for key in ("rankme", "alpha", "r2", "trace", "log_det", "d", "eigenspectrum"):
        assert key in out


def test_spectral_metrics_alpha(make_powerlaw_eigvals):
    eigvals = make_powerlaw_eigvals(d=500, alpha=1.5)
    out = spectral_metrics(eigvals)
    assert abs(out["alpha"] - 1.5) < 0.2


# Real collection produces fp64 eigenvalues (fp64 cov -> eigvalsh). The metric
# functions must accept them; the fp32 fixtures above never exercised this.
def test_metrics_accept_fp64_eigvals():
    ev = torch.tensor([10.0 / (i + 1) for i in range(200)], dtype=torch.float64)
    o32 = spectral_metrics(ev.float())
    o64 = spectral_metrics(ev)  # crashed in stringer (float vs double) before the fix
    assert abs(o32["alpha"] - o64["alpha"]) < 1e-3
    assert kfac_metrics(ev, ev * 2.0)["trace"] > 0
    d = 200  # >100 so the alpha-fit window (trange 11..100) is valid
    vec = torch.eye(d, dtype=torch.float64)
    gen = generalized_eigenvalues(ev, vec, ev, vec)  # fp64 -> spectral_metrics
    assert spectral_metrics(gen)["d"] == d


# Per-OV-head slices are d_head-sized (32 < 100); the alpha-fit window must not
# index past the spectrum.
def test_spectral_metrics_small_dim():
    out = spectral_metrics(torch.tensor([10.0 / (i + 1) for i in range(32)], dtype=torch.float64))
    assert out["d"] == 32 and out["rankme"] > 0  # crashed (index OOB) before the fix


# Degenerate spectra occur in the wild: nanochat zero-inits c_proj, so attn.out /
# mlp.out are exactly (or numerically) zero at the earliest checkpoints.
def test_spectral_metrics_zero_and_low_rank_spectra():
    out = spectral_metrics(torch.zeros(128, dtype=torch.float64))
    assert out["trace"] == 0.0 and np.isnan(out["alpha"]) and np.isnan(out["rankme"])

    low_rank = torch.zeros(128, dtype=torch.float64)
    low_rank[:5] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])  # < 12 positive: no fit window
    out = spectral_metrics(low_rank)
    assert np.isnan(out["alpha"]) and out["rankme"] > 0


def test_mixture_with_zero_trace_component():
    from scripts.compute_metrics import _mixture, _spectral_entropy
    d = 16
    lam = torch.linspace(4.0, 0.5, d).double()
    cov = torch.diag(lam)
    zero = torch.zeros(d, dtype=torch.float64)
    out = _mixture([lam, zero], [cov, torch.zeros(d, d, dtype=torch.float64)])
    assert out is not None
    w, s, s_mix = out
    assert w.tolist() == [1.0, 0.0] and s[1] == 0.0
    assert abs(s_mix - _spectral_entropy(lam)) < 1e-9   # mix collapses to the live component
    assert _mixture([zero, zero], [cov * 0, cov * 0]) is None
    assert _spectral_entropy(zero) == 0.0


def test_spectral_metrics_trace():
    # Need >= 100 eigenvalues for stringer_get_powerlaw (uses trange 11-100)
    eigvals = torch.tensor([10.0 / (i + 1) for i in range(200)])
    out = spectral_metrics(eigvals)
    assert abs(out["trace"] - float(eigvals.sum())) < 1e-6


def test_spectral_metrics_rankme_range():
    eigvals = torch.ones(200)  # need >= 100 for stringer_get_powerlaw
    out = spectral_metrics(eigvals)
    assert 1 <= out["rankme"] <= 200 + 1  # small float overshoot OK


# --- mean_metrics ---

def test_mean_metrics_zero_mean():
    d = 16
    out = mean_metrics(torch.eye(d), torch.ones(d), torch.zeros(d))
    assert out["mean_norm"] == 0.0
    assert out["max_overlap"] == 0.0


def test_mean_metrics_aligned_top_eigvec():
    d = 16
    eigvecs = torch.eye(d)
    eigvals = torch.tensor([100.0] + [1.0] * (d - 1))
    mean = eigvecs[:, 0]  # aligned with top eigenvector
    out = mean_metrics(eigvecs, eigvals, mean)
    assert out["max_overlap"] > 0.99
    assert out["max_overlap_idx"] == 0
    # aligned with v_0 -> top_overlap == max_overlap
    assert out["top_overlap"] > 0.99
    assert abs(out["top_overlap"] - out["max_overlap"]) < 1e-6


def test_mean_metrics_top_overlap_distinct_from_max():
    d = 16
    eigvecs = torch.eye(d)
    eigvals = torch.tensor([100.0] + [1.0] * (d - 1))
    out = mean_metrics(eigvecs, eigvals, eigvecs[:, 3])  # aligned with a non-top eigvec
    assert out["max_overlap_idx"] == 3
    assert out["max_overlap"] > 0.99
    assert out["top_overlap"] < 0.01            # ~no overlap with v_0
    out0 = mean_metrics(eigvecs, eigvals, torch.zeros(d))
    assert out0["top_overlap"] == 0.0           # degenerate branch carries the key


# --- kfac_metrics ---

def test_kfac_trace_product():
    e_acts = torch.tensor([10.0, 5.0, 2.0])
    e_grads = torch.tensor([4.0, 3.0])
    out = kfac_metrics(e_acts, e_grads, top_k=10, sample_k=6)
    expected_trace = sum(e_acts) * sum(e_grads)
    assert abs(out["trace"] - expected_trace) < 1e-6


def test_top_k_outer_products_largest():
    a = torch.tensor([10.0, 5.0, 2.0])  # _top_k_outer_products takes + returns torch
    b = torch.tensor([4.0, 3.0])
    top = _top_k_outer_products(a, b, 3)
    assert abs(top[0] - 40.0) < 1e-6  # 10 * 4


def test_top_k_outer_products_count():
    a = torch.tensor([10.0, 5.0, 2.0])
    b = torch.tensor([4.0, 3.0])
    top = _top_k_outer_products(a, b, 4)
    assert len(top) == 4


# --- generalized eigenvalues ---

def test_generalized_eigvals_identity():
    d = 16
    eigvals = torch.tensor([10.0 / (i + 1) for i in range(d)])
    eigvecs = torch.eye(d)
    # grads == acts → generalized eigenvalues should all be 1
    gen = generalized_eigenvalues(eigvals, eigvecs, eigvals, eigvecs)
    assert torch.allclose(gen, torch.ones_like(gen), atol=0.05)


def test_generalized_eigvals_scaled():
    d = 16
    eigvals_acts = torch.tensor([10.0 / (i + 1) for i in range(d)])
    eigvals_grads = eigvals_acts * 3.0
    eigvecs = torch.eye(d)
    gen = generalized_eigenvalues(eigvals_grads, eigvecs, eigvals_acts, eigvecs)
    assert torch.allclose(gen, torch.full_like(gen, 3.0), atol=0.1)


# ---------------------------------------------------------------------------
# Metric walk over the hook tree: which node gets which metric.
# ---------------------------------------------------------------------------

def _cov(d, n=200):
    X = torch.randn(n, d, dtype=torch.float64)
    return X.T @ X


def _walk_data():
    d = 16
    return {
        "blk0.attn.in":  {"acts_gram": _cov(d), "acts_n": 200, "acts_mean": torch.randn(d).float(),
                          "grads_gram": _cov(d), "grads_n": 200},
        "blk0.attn.out": {"acts_gram": _cov(d), "acts_n": 200, "grads_gram": _cov(d), "grads_n": 200},
        "blk0.mlp.up.in":   {"acts_gram": _cov(d), "acts_n": 200},
        "blk0.mlp.up.out":  {"grads_gram": _cov(d), "grads_n": 200},
        "blk0.mlp.down.in": {"acts_gram": _cov(d), "acts_n": 200},
        "blk0.mlp.down.out": {"grads_gram": _cov(d), "grads_n": 200},
        "__format__": "cov",
    }


def test_walk_kfac_at_projection_and_subblock_nodes():
    res = compute_metrics_for_checkpoint(DataAccessor(_walk_data()))
    # Real weight K-FAC at each projection node...
    assert "kfac" in res["blk0.mlp.up"] and "kfac" in res["blk0.mlp.down"]
    # ...and the cross-boundary K-FAC at the sub-block nodes (in/out boundary leaves).
    assert "kfac" in res["blk0.attn"]


def test_walk_projections_kfac_at_mlp_node():
    res = compute_metrics_for_checkpoint(DataAccessor(_walk_data()))
    assert "projections_kfac" in res["blk0.mlp"]   # up.in.acts x down.out.grads
    assert "kfac" not in res["blk0.mlp"]            # blk0.mlp has no in/out boundary leaves here


def test_walk_leaf_spectral_family_and_gen():
    res = compute_metrics_for_checkpoint(DataAccessor(_walk_data()))
    leaf = res["blk0.attn.in"]
    assert {"acts_uncentered", "grads_uncentered", "gen"} <= set(leaf)
    assert "acts_centered" in leaf and "acts_mean_metrics" in leaf  # mean present
    # .in leaf of a projection: acts only -> no gen, no grads
    assert "gen" not in res["blk0.mlp.up.in"] and "grads_uncentered" not in res["blk0.mlp.up.in"]


def test_walk_derived_out_acts_needs_no_metric_without_grads_pairing():
    # The .out leaf carries grads; its acts is derived only with a model. Without one,
    # only grads metrics appear (no gen, no derived acts spectral).
    res = compute_metrics_for_checkpoint(DataAccessor(_walk_data()))
    assert set(res["blk0.mlp.up.out"]) == {"grads_uncentered"}


# ---------------------------------------------------------------------------
# Key-level merge of step results (lossless re-runs)
# ---------------------------------------------------------------------------

def test_merge_step_unions_and_new_wins():
    old = {"blk0.mlp.out": {"acts_uncentered": {"rankme": 1.0}}}
    new = {"blk0.mlp.out": {"acts_uncentered": {"rankme": 2.0},      # recomputed -> new wins
                            "acts_mean_in_residual": {"x": 9}}}       # new group -> added
    out = _merge_step(old, new)
    assert out["blk0.mlp.out"]["acts_uncentered"]["rankme"] == 2.0
    assert out["blk0.mlp.out"]["acts_mean_in_residual"] == {"x": 9}


def test_merge_step_preserves_leaf_absent_in_new():
    # a --derive false run omits weight-derived contrib leaves; the merge must keep them
    old = {"blk0.attn.head0.contrib": {"acts_uncentered": {"rankme": 3.0}}}
    new = {"blk0.attn.out": {"acts_uncentered": {"rankme": 1.0}}}
    out = _merge_step(old, new)
    assert out["blk0.attn.head0.contrib"]["acts_uncentered"]["rankme"] == 3.0
    assert out["blk0.attn.out"]["acts_uncentered"]["rankme"] == 1.0


def test_to_numpy_recursive():
    out = _to_numpy({"l": {"g": {"rankme": 1.0, "eig": torch.tensor([1.0, 2.0])}}})
    assert isinstance(out["l"]["g"]["eig"], np.ndarray)
    assert out["l"]["g"]["rankme"] == 1.0   # non-tensors pass through


def test_merge_step_does_not_mutate_inputs():
    old = {"l": {"acts_uncentered": {"a": 1}}}
    new = {"l": {"acts_centered": {"b": 2}}}
    out = _merge_step(old, new)
    assert old == {"l": {"acts_uncentered": {"a": 1}}}            # untouched
    assert set(out["l"]) == {"acts_uncentered", "acts_centered"}


# ---------------------------------------------------------------------------
# Stored mean vector + residual-basis mean metrics
# ---------------------------------------------------------------------------

def _resbasis_data(d=16):
    torch.manual_seed(0)
    leaf = lambda: {"acts_gram": _cov(d), "acts_n": 200, "acts_mean": torch.randn(d).float()}
    return {
        "blk0.attn.out": leaf(),
        "blk0.mlp.out": leaf(),
        "before_final_norm": leaf(),
        "after_final_norm": leaf(),
        "__format__": "cov",
    }


def test_quantity_metrics_stores_mean_vec():
    res = compute_metrics_for_checkpoint(DataAccessor(_resbasis_data()))
    mv = res["blk0.mlp.out"]["acts_mean_vec"]
    assert mv.shape == (16,)


def _blkres_data(d=16, d2=32):
    leaf = lambda dim: {"acts_gram": _cov(dim), "acts_n": 200, "acts_mean": torch.randn(dim).float()}
    return {
        "blk0.mlp.in":     leaf(d),    # residual into the mlp sub-block
        "blk0.mlp.out":    leaf(d),     # mlp's contribution (same space)
        "blk0.mlp.up.in":  leaf(d),     # projection boundary: mismatched in/out dims
        "blk0.mlp.up.out": leaf(d2),    # -> would crash mean_metrics; must be skipped
        "__format__": "cov",
    }


def test_blk_mean_metrics_at_subblock_node():
    res = compute_metrics_for_checkpoint(DataAccessor(_blkres_data()))
    m = res["blk0.mlp"]["mean_metrics_blk_vs_res"]   # mlp.out mean vs mlp.in centered basis
    assert {"top_overlap", "rayleigh", "max_overlap"} <= set(m)


def test_blk_mean_metrics_skips_projection_node():
    # node_re excludes blk0.mlp.up; without it the d-vs-d2 mismatch crashes the whole walk
    res = compute_metrics_for_checkpoint(DataAccessor(_blkres_data()))
    assert "mean_metrics_blk_vs_res" not in res.get("blk0.mlp.up", {})


def test_blk_mean_metrics_matches_direct():
    data = _blkres_data()
    res = compute_metrics_for_checkpoint(DataAccessor(data))
    acc = DataAccessor(data)
    v_in = acc["blk0.mlp.in"].acts
    evals, evecs = v_in.eigvals_centered, v_in.eigvecs_centered
    mean = acc["blk0.mlp.out"].acts.mean
    ref = mean_metrics(evecs, evals, mean)
    got = res["blk0.mlp"]["mean_metrics_blk_vs_res"]
    assert abs(got["rayleigh"] - ref["rayleigh"]) < 1e-9


def test_main_parallel_path_does_not_deadlock(tmp_path):
    """main() runs its workers as THREADS sharing one GPU lock. The GPU funnel is re-entrant,
    so that lock must be an RLock — a plain Lock self-deadlocks. Run the real threaded main()
    (CPU, synthetic data) under a join-timeout: a regression to a non-reentrant lock then
    surfaces as a failure, not a frozen suite. (The funnel runs on CPU when no GPU is present,
    so this reproduces the deadlock without one.)"""
    import threading
    from scripts.compute_metrics import main
    from utils.gpu import set_gpu_lock

    d, N = 32, 64
    mdir = tmp_path / "cfg" / "pythia-14m"
    mdir.mkdir(parents=True)
    for step in (0, 1):
        acts = torch.randn(N, d, dtype=torch.float64)
        torch.save({"after_final_norm": {"acts_gram": acts.T @ acts, "acts_n": N,
                                          "acts_mean": acts.float().mean(0)}, "__format__": "cov"},
                   mdir / f"step{step}.pt")

    t = threading.Thread(target=main, kwargs=dict(
        config_directory="cfg", model_name="pythia-14m", num_workers=2, derive=False,
        data_root=str(tmp_path), output_root=str(tmp_path / "res")), daemon=True)
    try:
        t.start()
        t.join(timeout=30)
        assert not t.is_alive(), "main() deadlocked — the GPU lock must be reentrant (RLock)"
        res = np.load(str(tmp_path / "res" / "cfg" / "results_pythia-14m.npy"), allow_pickle=True).item()
        assert set(res) == {0, 1}
    finally:
        set_gpu_lock(None)


# --- node-operand metrics ({"node": ""}) + block-composition stubs ---

def _samples_walk_data(d=32, N=64):
    X = torch.randn(N, d)
    return {"blk0.attn.out": {"acts_samples": X, "acts_n": N},
            "before_final_norm": {"acts_samples": X * 2, "acts_n": N}, "__format__": "acts"}


def test_node_operand_metric_receives_node():
    from scripts.compute_metrics import METRICS, _Metric, get_metrics
    acc = DataAccessor(_samples_walk_data())
    probe = _Metric("probe", {"node": ""}, lambda node: {"path": node.path}, node_re=r"blk0\.attn\.out$")
    METRICS.append(probe)
    try:
        res = get_metrics(acc.v)
    finally:
        METRICS.remove(probe)
    assert res["blk0.attn.out"]["probe"]["path"] == "blk0.attn.out"


def test_stub_metrics_skip_cleanly():
    res = compute_metrics_for_checkpoint(DataAccessor(_samples_walk_data()))
    assert "block_residual_coupling" not in res.get("blk0.attn.out", {})
    assert "overlap_chi" not in res.get("", {})


def _composition_acc(d=32, N=256):
    g = torch.Generator().manual_seed(11)
    I, A, B = (torch.randn(N, d, generator=g) for _ in range(3))
    return DataAccessor({"blk0.attn.in": {"acts_samples": I, "acts_n": N},
                         "blk0.attn.out": {"acts_samples": A, "acts_n": N},
                         "blk0.mlp.out": {"acts_samples": B, "acts_n": N},
                         "before_final_norm": {"acts_samples": I + A + B, "acts_n": N},
                         "__format__": "acts"})


def test_block_composition_metrics_fire():
    from scripts.compute_metrics import get_metrics
    res = get_metrics(_composition_acc().v)
    out = res["blk0.attn.out"]
    for key in ("block_residual_coupling", "ablation_contribution", "eigendirection_attrib",
                "gen_block_vs_residual", "mean_migration"):
        assert key in out, key
    assert 0.0 <= out["block_residual_coupling"]["cka_cr"] <= 1.0
    bb, chi = res[""]["block_block_coupling"], res[""]["overlap_chi"]
    assert bb["leaves"] == ["blk0.attn.out", "blk0.mlp.out"] and bb["cka"].shape == (2, 2)
    assert torch.allclose(bb["cka"].diagonal(), torch.ones(2), atol=1e-4)
    assert -1e-6 <= chi["chi"] <= chi["h_w"] + 1e-6


def test_ablation_closes_algebraically():
    # ablating attn.out from r = in + attn + mlp must recover rankme of cov(in + mlp)
    from scripts.compute_metrics import get_metrics, rankme_metrics
    from utils.accessor import eigvalsh_descending
    acc = _composition_acc()
    res = get_metrics(acc.v)
    rest = acc["blk0.attn.in"].acts.samples + acc["blk0.mlp.out"].acts.samples
    mu = rest.float().mean(0)
    lam = eigvalsh_descending(rest.float().T @ rest.float() / rest.shape[0] - torch.outer(mu, mu))
    expected = rankme_metrics(lam)["rankme"]
    assert abs(res["blk0.attn.out"]["ablation_contribution"]["rankme_ablated"] - expected) < 1e-2


def test_incremental_overlap_orthogonal_supports():
    # r_in confined to the first d/2 coords, c to the last d/2 -> chi == H(w) exactly
    from scripts.compute_metrics import get_metrics
    d, N = 32, 256
    g = torch.Generator().manual_seed(13)
    I, A = torch.zeros(N, d), torch.zeros(N, d)
    I[:, :d // 2] = torch.randn(N, d // 2, generator=g)
    A[:, d // 2:] = torch.randn(N, d // 2, generator=g)
    acc = DataAccessor({"blk0.attn.in": {"acts_samples": I, "acts_n": N},
                        "blk0.attn.out": {"acts_samples": A, "acts_n": N},
                        "before_final_norm": {"acts_samples": I + A, "acts_n": N},
                        "__format__": "acts"})
    chi = get_metrics(acc.v)["blk0.attn.out"]["incremental_overlap"]
    assert chi["chi_frac"] > 0.98 and abs(chi["chi"] - chi["h_w"]) < 0.02


def test_drift_metrics_with_ctx():
    from scripts.compute_metrics import get_metrics
    acc, prev = _composition_acc(), _composition_acc()   # same seed -> identical checkpoints
    res = get_metrics(acc.v, ctx={"prev": prev.v})
    out = res["blk0.attn.out"]
    assert abs(out["cka_drift"]["cka_drift"] - 1.0) < 1e-4
    gen = out["geneig_drift"]["eigvals"]
    assert torch.allclose(gen, torch.ones_like(gen), atol=0.05)


def test_gen_vs_ref_with_ctx():
    from scripts.compute_metrics import get_metrics
    acc, ref = _composition_acc(), _composition_acc()   # same seed -> identical populations
    res = get_metrics(acc.v, ctx={"ref": ref.v})
    gen = res["blk0.attn.out"]["gen_vs_ref"]["acts"]["eigvals"]
    assert torch.allclose(gen, torch.ones_like(gen), atol=0.05)
    assert 0 <= res["blk0.attn.out"]["gen_vs_ref"]["acts"]["tail_centroid"] <= len(gen)
    assert "gen_vs_ref" not in get_metrics(_composition_acc().v)["blk0.attn.out"]


def test_drift_metrics_skip_without_ctx():
    from scripts.compute_metrics import get_metrics
    res = get_metrics(_composition_acc().v)
    assert "cka_drift" not in res["blk0.attn.out"] and "geneig_drift" not in res["blk0.attn.out"]


def test_block_ledger_identity_closes():
    from scripts.compute_metrics import get_metrics
    res = get_metrics(_composition_acc().v)
    led = res["blk0"]["block_ledger"]
    recon = led["chi"] + led["quality"] + led["interference"]
    assert abs(led["delta_s"] - recon) < 1e-9          # exact by construction
    # fixture: before_final_norm = in + attn + mlp, all independent full-rank -> chi ~ 0
    assert abs(led["chi"]) < 0.05 and len(led["w"]) == 3


def test_mean_cos_in_couplings():
    from scripts.compute_metrics import get_metrics
    res = get_metrics(_composition_acc().v)
    # r = in + attn + mlp contains c itself -> E[cos(c, r)] ~ 1/sqrt(3)
    assert abs(res["blk0.attn.out"]["block_residual_coupling"]["cos_cr"] - 3 ** -0.5) < 0.05
    mc = res[""]["block_block_coupling"]["mean_cos"]
    assert torch.allclose(mc.diagonal(), torch.ones(2)) and abs(mc[0, 1]) < 0.2
