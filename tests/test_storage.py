"""Save / convert through the rebuilt accessor: a format is the set of components it
writes (FORMATS), `gram` is the raw stored covariance, fp32 throughout, and metadata
(identity, token filter) is stamped — never inferred from the path."""
import torch

from utils.accessor import DataAccessor, decomp_profiler

HOOK = "after_final_norm"
ACTS = "acts"
GRADS = "grads"
ID = ("EleutherAI/pythia-14m", "step0", "test")   # save asserts a complete identity


def test_save_load_cov(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=True, with_means=False)
    path = str(tmp_path / "cov.pt")
    DataAccessor(factors, identity=ID).save(path, format="cov")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert HOOK in data
    assert f"{ACTS}_gram" in data[HOOK]            # cov format stores the raw gram (+ n)
    assert data["__format__"] == "cov"


def test_save_load_cov_svd(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    path = str(tmp_path / "cov_svd.pt")
    DataAccessor(factors, identity=ID).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert f"{ACTS}_eigvals" in entry
    assert f"{ACTS}_eigvecs" in entry
    assert entry[f"{ACTS}_eigvals"].shape[0] == 32


def test_save_load_eigenvalues(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    path = str(tmp_path / "eigvals.pt")
    DataAccessor(factors, identity=ID).save(path, format="eigenvalues")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert f"{ACTS}_eigvals" in entry
    assert f"{ACTS}_eigvecs" not in entry


def test_save_load_acts(tmp_path):
    N, d = 100, 32
    acts = torch.randn(N, d)
    factors = {HOOK: {f"{ACTS}_samples": acts, f"{ACTS}_n": N}}
    path = str(tmp_path / "acts.pt")
    DataAccessor(factors, identity=ID).save(path, format="acts")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.allclose(data[HOOK][f"{ACTS}_samples"], acts.float(), atol=1e-6)


def test_format_metadata(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    for fmt in ("cov", "cov_svd", "eigenvalues"):
        path = str(tmp_path / f"{fmt}.pt")
        DataAccessor(factors, identity=ID).save(path, format=fmt)
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["__format__"] == fmt


def test_means_included_by_default(tmp_path, make_factors_dict):
    # means are part of the cov / cov_svd component sets — no modifier needed.
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    for fmt in ("cov", "cov_svd"):
        path = str(tmp_path / f"{fmt}.pt")
        DataAccessor(factors, identity=ID).save(path, format=fmt)
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert f"{ACTS}_mean" in data[HOOK]


def test_roundtrip_cov_to_cov_svd_to_eigenvalues(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    cov_path = str(tmp_path / "cov.pt")
    DataAccessor(factors, identity=ID).save(cov_path, format="cov")
    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(cov_path).save(svd_path, format="cov_svd")           # identity carried in metadata
    eig_path = str(tmp_path / "eig.pt")
    DataAccessor(svd_path).save(eig_path, format="eigenvalues")
    svd_data = torch.load(svd_path, map_location="cpu", weights_only=False)
    eig_data = torch.load(eig_path, map_location="cpu", weights_only=False)
    assert torch.allclose(svd_data[HOOK][f"{ACTS}_eigvals"], eig_data[HOOK][f"{ACTS}_eigvals"], atol=1e-5)


def test_cov_svd_reconstruction(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    n = factors[HOOK][f"{ACTS}_n"]
    original_cov = factors[HOOK][f"{ACTS}_gram"].float() / n
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors, identity=ID).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    V, S = entry[f"{ACTS}_eigvecs"], entry[f"{ACTS}_eigvals"]
    assert torch.allclose(V @ torch.diag(S.to(V.dtype)) @ V.T, original_cov, atol=1e-4)


def test_token_filter_metadata(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "test.pt")
    tf = {"token_selection": "last", "skip_positions": 0}
    DataAccessor(factors, identity=ID).stamp(token_filter=tf).save(path, format="cov")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["__token_filter__"] == tf


def test_save_preserve_updates_token_filter_without_materializing(tmp_path, make_factors_dict):
    # save(format=None) preserves stored data verbatim and only re-stamps metadata —
    # the path the `set-filter` CLI now takes.
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "cov_svd.pt")
    DataAccessor(factors, identity=ID).save(path, format="cov_svd")
    before = torch.load(path, map_location="cpu", weights_only=False)

    DataAccessor(path).stamp(token_filter={"token_selection": "last"}).save(path)  # format=None preserve
    after = torch.load(path, map_location="cpu", weights_only=False)
    assert after["__token_filter__"] == {"token_selection": "last"}
    assert after["__format__"] == "cov_svd"  # preserved, not re-materialized
    assert torch.equal(after[HOOK][f"{ACTS}_eigvals"], before[HOOK][f"{ACTS}_eigvals"])


def test_info_returns_string(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "test.pt")
    DataAccessor(factors, identity=ID).save(path, format="cov_svd")
    result = DataAccessor(path).info()
    assert isinstance(result, str) and len(result) > 0


def _count_decomps(fn):
    """Run fn under the decomp profiler; return (#_eigh, #_svd)."""
    decomp_profiler.enable()
    try:
        fn()
        calls = list(decomp_profiler.calls)
    finally:
        decomp_profiler.disable()
    return sum(c[0] == "_eigh" for c in calls), sum(c[0] == "_svd" for c in calls)


def test_save_cov_svd_single_decomposition(tmp_path, make_factors_dict):
    # cov_svd from a raw gram is exactly one full eigendecomposition (eigvals + eigvecs share it).
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "svd.pt")
    n_eigh, n_svd = _count_decomps(lambda: DataAccessor(factors, identity=ID).save(path, format="cov_svd"))
    assert n_eigh == 1 and n_svd == 0


def test_save_cov_svd_means_one_extra_eigh(tmp_path, make_factors_dict):
    # With a mean, the centered spectrum is a second eigendecomposition: two eigh, no svd.
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    n_eigh, n_svd = _count_decomps(lambda: DataAccessor(factors, identity=ID).save(path, format="cov_svd"))
    assert n_eigh == 2 and n_svd == 0


def test_self_projection_invariant(make_factors_dict):
    # Projecting a covariance into its own eigenbasis returns its own spectrum.
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    view = DataAccessor(factors, identity=ID)[HOOK].acts
    projected = view.in_basis(view).eigvals
    assert torch.allclose(projected.double(), view.eigvals.double(), atol=1e-4)


def test_convert_preserves_mean_and_centered(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    cov_path = str(tmp_path / "cov.pt")
    DataAccessor(factors, identity=ID).save(cov_path, format="cov")
    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(cov_path).save(svd_path, format="cov_svd")
    entry = torch.load(svd_path, map_location="cpu", weights_only=False)[HOOK]
    assert f"{ACTS}_mean" in entry and f"{ACTS}_eigvals_centered" in entry


def test_acts_to_cov_svd_derives_mean_by_default(tmp_path):
    acts = torch.randn(20, 8)
    path = str(tmp_path / "svd.pt")
    DataAccessor({HOOK: {f"{ACTS}_samples": acts, f"{ACTS}_n": len(acts)}},
                 identity=ID).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert torch.allclose(entry[f"{ACTS}_mean"], acts.mean(0), atol=1e-6)


def _mlp_proj_data():
    # MLP up-projection with stored acts at both the .in and .out leaves (.out matches the
    # `b` override family).
    return {
        "blk0.mlp.up.in": {"acts_n": 5, "acts_gram": torch.eye(4) * 5, "acts_mean": torch.ones(4)},
        "blk0.mlp.up.out": {"acts_n": 5, "acts_gram": torch.eye(4) * 10, "acts_mean": torch.arange(4).float()},
        "__format__": "cov",
    }


def test_minus_b_override_drops_mlp_out(tmp_path):
    # cov_svd keeps the .out leaf; cov_svd with -b removes the MLP_OUT family.
    DataAccessor(_mlp_proj_data(), identity=ID).save(str(tmp_path / "svd.pt"), format="cov_svd")
    entry = torch.load(str(tmp_path / "svd.pt"), map_location="cpu", weights_only=False)["blk0.mlp.up.out"]
    assert {"acts_eigvals", "acts_eigvecs", "acts_mean"} <= set(entry)

    DataAccessor(_mlp_proj_data(), identity=ID).save(str(tmp_path / "drop.pt"), format="cov_svd",
                                                     overrides=("-b",))
    reloaded = torch.load(str(tmp_path / "drop.pt"), map_location="cpu", weights_only=False)
    assert "blk0.mlp.up.out" not in reloaded


def test_eigenvalues_preserves_means_no_eigvecs(tmp_path):
    DataAccessor(_mlp_proj_data(), identity=ID).save(str(tmp_path / "eig.pt"), format="eigenvalues")
    reloaded = torch.load(str(tmp_path / "eig.pt"), map_location="cpu", weights_only=False)
    for leaf in ("blk0.mlp.up.in", "blk0.mlp.up.out"):
        e = reloaded[leaf]
        assert {"acts_mean", "acts_eigvals"} <= set(e)
        assert "acts_eigvecs" not in e


def test_captured_after_final_norm_acts_not_dropped_as_derived(tmp_path):
    # after_final_norm.acts is a derivable slot AND directly captured — a captured value
    # must be stored, not dropped because a derivation rule exists for that slot.
    d = 16
    X = torch.randn(200, d, dtype=torch.float64)
    G = torch.randn(200, d, dtype=torch.float64)
    factors = {"after_final_norm": {"acts_gram": X.T @ X, "acts_n": 200, "acts_mean": X.float().mean(0),
                                    "grads_gram": G.T @ G, "grads_n": 200}, "__format__": "cov"}
    out = str(tmp_path / "afn.pt")
    DataAccessor(factors, identity=ID).save(out, format="cov_svd")
    entry = torch.load(out, map_location="cpu", weights_only=False)["after_final_norm"]
    assert "acts_eigvals" in entry and "grads_eigvals" in entry
