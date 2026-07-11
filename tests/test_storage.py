import os
import torch
from utils.accessor import DataAccessor, decomp_profiler

# Fixture entries live at leaf "after_final_norm" with quantities acts / grads.
HOOK = "after_final_norm"
ACTS = "acts"
GRADS = "grads"


def test_save_load_cov(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=True, with_means=False)
    path = str(tmp_path / "cov.pt")
    DataAccessor(factors).save(path, format="cov")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert HOOK in data
    assert f"{ACTS}_cov" in data[HOOK]
    assert data["__format__"] == "cov"


def test_save_load_cov_svd(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    path = str(tmp_path / "cov_svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert f"{ACTS}_eigvals" in entry
    assert f"{ACTS}_eigvecs" in entry
    assert entry[f"{ACTS}_eigvals"].shape[0] == 32


def test_save_load_eigenvalues(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    path = str(tmp_path / "eigvals.pt")
    DataAccessor(factors).save(path, format="eigenvalues")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert f"{ACTS}_eigvals" in entry
    assert f"{ACTS}_eigvecs" not in entry


def test_save_load_acts(tmp_path):
    N, d = 100, 32
    acts = torch.randn(N, d)
    factors = {HOOK: {f"{ACTS}_samples": acts, f"{ACTS}_n": N}}
    path = str(tmp_path / "acts.pt")
    DataAccessor(factors).save(path, format="acts")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.allclose(data[HOOK][f"{ACTS}_samples"], acts.float(), atol=1e-6)


def test_format_metadata(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    for fmt in ("cov", "cov_svd", "eigenvalues"):
        path = str(tmp_path / f"{fmt}.pt")
        DataAccessor(factors).save(path, format=fmt)
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["__format__"] == fmt


def test_means_modifier(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    for fmt in ("cov+m", "cov_svd+m"):
        path = str(tmp_path / f"{fmt.replace('+', '_')}.pt")
        DataAccessor(factors).save(path, format=fmt)
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert f"{ACTS}_mean" in data[HOOK]


def test_roundtrip_cov_to_cov_svd_to_eigenvalues(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    cov_path = str(tmp_path / "cov.pt")
    DataAccessor(factors).save(cov_path, format="cov")

    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(cov_path).save(svd_path, format="cov_svd")

    eig_path = str(tmp_path / "eig.pt")
    DataAccessor(svd_path).save(eig_path, format="eigenvalues")

    svd_data = torch.load(svd_path, map_location="cpu", weights_only=False)
    eig_data = torch.load(eig_path, map_location="cpu", weights_only=False)
    assert torch.allclose(svd_data[HOOK][f"{ACTS}_eigvals"], eig_data[HOOK][f"{ACTS}_eigvals"], atol=1e-5)


def test_cov_svd_reconstruction(tmp_path, make_factors_dict):
    d = 32
    factors = make_factors_dict(d=d, with_grad=False, with_means=False)
    n = factors[HOOK][f"{ACTS}_n"]
    original_cov = factors[HOOK][f"{ACTS}_cov"].float() / n

    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    V = entry[f"{ACTS}_eigvecs"]
    S = entry[f"{ACTS}_eigvals"]
    reconstructed = V @ torch.diag(S.to(V.dtype)) @ V.T
    assert torch.allclose(reconstructed, original_cov, atol=1e-4)


def test_token_filter_metadata(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "test.pt")
    token_filter = {"token_selection": "last", "skip_positions": 0}
    DataAccessor(factors).save(path, format="cov", token_filter=token_filter)
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["__token_filter__"] == token_filter


def test_save_preserve_updates_token_filter_without_materializing(tmp_path, make_factors_dict):
    # save(format=None) preserves stored data verbatim and only updates metadata —
    # the path the `set-filter` CLI now takes (replacing the old set_token_filter).
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "cov_svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    before = torch.load(path, map_location="cpu", weights_only=False)

    DataAccessor(path).save(path, token_filter={"token_selection": "last"})
    after = torch.load(path, map_location="cpu", weights_only=False)
    assert after["__token_filter__"] == {"token_selection": "last"}
    assert after["__format__"] == "cov_svd"  # preserved, not re-materialized to a default
    assert torch.equal(after[HOOK][f"{ACTS}_eigvals"], before[HOOK][f"{ACTS}_eigvals"])


def test_info_returns_string(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "test.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    result = DataAccessor(path).info()
    assert isinstance(result, str)
    assert len(result) > 0


def _count_decomps(fn):
    """Run fn under the decomp profiler; return (n_eigh, n_eigvalsh)."""
    decomp_profiler.enable()
    try:
        fn()
        calls = list(decomp_profiler.calls)
    finally:
        decomp_profiler.disable()
    return (sum(c[0].startswith("eigh[") for c in calls),
            sum(c[0].startswith("eigvalsh[") for c in calls))


def test_save_cov_svd_single_decomposition(tmp_path, make_factors_dict):
    # cov_svd from a raw cov must do exactly one full eigendecomposition, no extra eigvalsh.
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "svd.pt")
    n_eigh, n_eigvalsh = _count_decomps(lambda: DataAccessor(factors).save(path, format="cov_svd"))
    assert n_eigh == 1
    assert n_eigvalsh == 0


def test_save_cov_svd_means_one_extra_eigvalsh(tmp_path, make_factors_dict):
    # With +m the centered spectrum is a second full eigendecomposition (the
    # uncentered eigh plus the centered eigh), so exactly two eigh and no eigvalsh.
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    n_eigh, n_eigvalsh = _count_decomps(lambda: DataAccessor(factors).save(path, format="cov_svd+m"))
    assert n_eigh == 2
    assert n_eigvalsh == 0


def test_storage_dtype_defaults(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd+m")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert entry[f"{ACTS}_eigvals"].dtype == torch.float64
    assert entry[f"{ACTS}_eigvecs"].dtype == torch.float32
    assert entry[f"{ACTS}_mean"].dtype == torch.float32


def test_storage_dtype_global_override(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd+m", storage_dtype="bf16")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    for key in (f"{ACTS}_eigvals", f"{ACTS}_eigvecs", f"{ACTS}_mean"):
        assert entry[key].dtype == torch.bfloat16


def test_storage_dtype_per_item_override(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd+m",
                               storage_dtype={"eigvals": "fp32", "eigvecs": "fp16", "mean": "bf16"})
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert entry[f"{ACTS}_eigvals"].dtype == torch.float32
    assert entry[f"{ACTS}_eigvecs"].dtype == torch.float16
    assert entry[f"{ACTS}_mean"].dtype == torch.bfloat16


def test_cross_checkpoint_self_projection_invariant(tmp_path, make_factors_dict):
    # Projecting a covariance onto its own eigenbasis yields its own spectrum.
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "step0.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    DataAccessor(path).save(path, cross_basis_refs=[path])

    stem = os.path.splitext(os.path.basename(path))[0]
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert torch.allclose(entry[f"{ACTS}_cross_eigvals_{stem}"].double(),
                          entry[f"{ACTS}_eigvals"].double(), atol=1e-4)


def test_convert_preserves_mean_by_default_and_minus_m_drops_it(tmp_path, make_factors_dict):
    # Bare cov_svd preserves stored means; -m explicitly drops them.
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    cov_path = str(tmp_path / "cov.pt")
    DataAccessor(factors).save(cov_path, format="cov+m")

    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(cov_path).save(svd_path, format="cov_svd")
    entry = torch.load(svd_path, map_location="cpu", weights_only=False)[HOOK]
    assert f"{ACTS}_mean" in entry
    assert f"{ACTS}_eigvals_centered" in entry

    svd_drop_path = str(tmp_path / "svd_drop.pt")
    DataAccessor(cov_path).save(svd_drop_path, format="cov_svd-m")
    entry_drop = torch.load(svd_drop_path, map_location="cpu", weights_only=False)[HOOK]
    assert f"{ACTS}_mean" not in entry_drop
    assert f"{ACTS}_eigvals_centered" in entry_drop


def test_acts_to_cov_svd_derives_mean_by_default(tmp_path):
    acts = torch.randn(20, 8)
    path = str(tmp_path / "svd.pt")
    DataAccessor({HOOK: {f"{ACTS}_samples": acts, f"{ACTS}_n": len(acts)}}).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert torch.allclose(entry[f"{ACTS}_mean"], acts.mean(0), atol=1e-6)


def _mlp_proj_data():
    # MLP up-projection with stored acts at both the .in leaf (captured) and the
    # .out leaf (a materialized derived quantity, gated by the `b` flag).
    return {
        "blk0.mlp.up.in": {
            "acts_n": 5,
            "acts_cov": torch.eye(4) * 5,
            "acts_mean": torch.ones(4),
        },
        "blk0.mlp.up.out": {
            "acts_n": 5,
            "acts_cov": torch.eye(4) * 10,
            "acts_mean": torch.arange(4).float(),
        },
        "__format__": "cov",
    }


def test_derived_preserve_and_drop_overrides(tmp_path):
    # MLP .out leaf with a stored derived acts: cov_svd keeps it, cov_svd-b drops it.
    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(_mlp_proj_data(), model_name="EleutherAI/pythia-14m").save(svd_path, format="cov_svd")
    entry = torch.load(svd_path, map_location="cpu", weights_only=False)["blk0.mlp.up.out"]
    assert "acts_eigvals" in entry
    assert "acts_eigvecs" in entry
    assert "acts_mean" in entry

    drop_path = str(tmp_path / "drop.pt")
    DataAccessor(_mlp_proj_data(), model_name="EleutherAI/pythia-14m").save(drop_path, format="cov_svd-b")
    reloaded = torch.load(drop_path, map_location="cpu", weights_only=False)
    # The .out leaf's acts are dropped under -b (it has no other quantity → leaf gone).
    assert "blk0.mlp.up.out" not in reloaded


def test_eigenvalues_preserves_means_and_derived_by_default(tmp_path):
    path = str(tmp_path / "eig.pt")
    DataAccessor(_mlp_proj_data(), model_name="EleutherAI/pythia-14m").save(path, format="eigvals")
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    in_e = reloaded["blk0.mlp.up.in"]
    out_e = reloaded["blk0.mlp.up.out"]
    assert {"acts_mean", "acts_eigvals"} <= set(in_e)
    assert {"acts_mean", "acts_eigvals"} <= set(out_e)
    assert "acts_eigvecs" not in in_e
    assert "acts_eigvecs" not in out_e


def test_captured_after_final_norm_acts_not_dropped_as_derived(tmp_path):
    # after_final_norm.acts is a derived slot (norm @ before_final_norm) AND can be
    # captured directly by the after_final_norm hook. A captured value must be stored,
    # not dropped just because a derivation rule exists for that slot.
    d = 16
    X = torch.randn(200, d, dtype=torch.float64)
    G = torch.randn(200, d, dtype=torch.float64)
    factors = {"after_final_norm": {"acts_cov": X.T @ X, "acts_n": 200, "acts_mean": X.float().mean(0),
                                    "grads_cov": G.T @ G, "grads_n": 200},
               "__format__": "cov", "__hf_model__": "EleutherAI/pythia-14m"}
    out = str(tmp_path / "afn.pt")
    DataAccessor(factors, model_name="EleutherAI/pythia-14m").save(out, format="cov_svd+m")
    entry = torch.load(out, map_location="cpu", weights_only=False)["after_final_norm"]
    assert "acts_eigvals" in entry and "grads_eigvals" in entry  # captured acts kept


def test_no_eager_metadata_inference_on_construct(tmp_path, make_factors_dict):
    # Constructing over a model-dir/stepN.pt path must not infer metadata (no path/dir I/O).
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    model_dir = tmp_path / "pythia-14m"
    model_dir.mkdir()
    path = str(model_dir / "step0.pt")
    DataAccessor(factors).save(path, format="cov")

    data = torch.load(path, map_location="cpu", weights_only=False)
    data.pop("__hf_model__", None)
    data.pop("__revision__", None)
    torch.save(data, path)

    acc = DataAccessor(path)
    assert acc._model_name is None


def test_existing_metadata_not_overwritten(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    model_dir = tmp_path / "pythia-14m"
    model_dir.mkdir()
    path = str(model_dir / "step0.pt")
    DataAccessor(factors).save(path, format="cov")

    data = torch.load(path, map_location="cpu", weights_only=False)
    # A family-recognizable but non-inferred repo (stamping would otherwise fill in
    # EleutherAI/pythia-14m + step0 from the path).
    data["__hf_model__"] = "EleutherAI/pythia-custom-x"
    data["__revision__"] = "rev99"
    torch.save(data, path)

    DataAccessor(path).save(path, format="eigenvalues")
    stamped = torch.load(path, map_location="cpu", weights_only=False)
    assert stamped["__hf_model__"] == "EleutherAI/pythia-custom-x"
    assert stamped["__revision__"] == "rev99"


def test_save_stamps_missing_checkpoint_metadata(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    model_dir = tmp_path / "pythia-14m"
    model_dir.mkdir()
    path = str(model_dir / "step0.pt")
    DataAccessor(factors).save(path, format="cov")

    data = torch.load(path, map_location="cpu", weights_only=False)
    data.pop("__hf_model__", None)
    data.pop("__revision__", None)
    torch.save(data, path)

    DataAccessor(path).save(path, format="eigenvalues")
    stamped = torch.load(path, map_location="cpu", weights_only=False)
    assert stamped["__hf_model__"] == "EleutherAI/pythia-14m"
    assert stamped["__revision__"] == "step0"
