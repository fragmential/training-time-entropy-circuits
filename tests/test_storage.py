import os
import torch
from utils.accessor import DataAccessor, decomp_profiler, eigh_descending


def test_save_load_cov(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=True, with_means=False)
    path = str(tmp_path / "cov.pt")
    DataAccessor(factors).save(path, format="cov")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert "hook0" in data
    assert "A" in data["hook0"]
    assert data["__format__"] == "cov"


def test_save_load_cov_svd(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    path = str(tmp_path / "cov_svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    data = torch.load(path, map_location="cpu", weights_only=False)
    entry = data["hook0"]
    assert "A_eigvals" in entry
    assert "A_eigvecs" in entry
    assert entry["A_eigvals"].shape[0] == 32


def test_save_load_eigenvalues(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=False, with_means=False)
    path = str(tmp_path / "eigvals.pt")
    DataAccessor(factors).save(path, format="eigenvalues")
    data = torch.load(path, map_location="cpu", weights_only=False)
    entry = data["hook0"]
    assert "A_eigvals" in entry
    assert "A_eigvecs" not in entry


def test_save_load_acts(tmp_path):
    N, d = 100, 32
    acts = torch.randn(N, d)
    factors = {"hook0": {"A": acts, "n_A": N, "n": N}}
    path = str(tmp_path / "acts.pt")
    DataAccessor(factors).save(path, format="acts")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.allclose(data["hook0"]["A"], acts.float(), atol=1e-6)


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
        assert "A_mean" in data["hook0"]


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
    assert torch.allclose(svd_data["hook0"]["A_eigvals"], eig_data["hook0"]["A_eigvals"], atol=1e-5)


def test_cov_svd_reconstruction(tmp_path, make_factors_dict):
    d = 32
    factors = make_factors_dict(d=d, with_grad=False, with_means=False)
    n = factors["hook0"]["n"]
    original_cov = factors["hook0"]["A"].float() / n

    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    data = torch.load(path, map_location="cpu", weights_only=False)
    V = data["hook0"]["A_eigvecs"]
    S = data["hook0"]["A_eigvals"]
    reconstructed = V @ torch.diag(S.to(V.dtype)) @ V.T
    assert torch.allclose(reconstructed, original_cov, atol=1e-4)


def test_same_layer_cross_basis(tmp_path, make_factors_dict):
    # Residual hook: forward factor resolves to A (no model needed for B).
    factors = {"after_final_norm": make_factors_dict(d=16, with_grad=True, with_means=False)["hook0"]}
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    acc = DataAccessor(path)
    acc.project_same_layer(onto="both", output_path=path)
    cross_keys = [k for k in acc.data["after_final_norm"] if "cross_eigvals" in k]
    assert len(cross_keys) > 0


def test_token_filter_metadata(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "test.pt")
    token_filter = {"token_selection": "last", "skip_positions": 0}
    DataAccessor(factors).save(path, format="cov", token_filter=token_filter)
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert data["__token_filter__"] == token_filter


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
    # With +m, the centered eigenvalues add exactly one eigvalsh (eigvecs reused from the eigh).
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    n_eigh, n_eigvalsh = _count_decomps(lambda: DataAccessor(factors).save(path, format="cov_svd+m"))
    assert n_eigh == 1
    assert n_eigvalsh == 1


def test_storage_dtype_defaults(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd+m")
    entry = torch.load(path, map_location="cpu", weights_only=False)["hook0"]
    assert entry["A_eigvals"].dtype == torch.float64
    assert entry["A_eigvecs"].dtype == torch.float32
    assert entry["A_mean"].dtype == torch.float32


def test_storage_dtype_global_override(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd+m", storage_dtype="bf16")
    entry = torch.load(path, map_location="cpu", weights_only=False)["hook0"]
    for key in ("A_eigvals", "A_eigvecs", "A_mean"):
        assert entry[key].dtype == torch.bfloat16


def test_storage_dtype_per_item_override(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd+m",
                               storage_dtype={"eigvals": "fp32", "eigvecs": "fp16", "mean": "bf16"})
    entry = torch.load(path, map_location="cpu", weights_only=False)["hook0"]
    assert entry["A_eigvals"].dtype == torch.float32
    assert entry["A_eigvecs"].dtype == torch.float16
    assert entry["A_mean"].dtype == torch.bfloat16


def test_cross_checkpoint_self_projection_invariant(tmp_path, make_factors_dict):
    # Projecting a covariance onto its own eigenbasis yields its own spectrum.
    factors = make_factors_dict(d=16, with_grad=False, with_means=False)
    path = str(tmp_path / "step0.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    DataAccessor(path).project_onto_basis(path, output_path=path)

    stem = os.path.splitext(os.path.basename(path))[0]
    entry = torch.load(path, map_location="cpu", weights_only=False)["hook0"]
    assert torch.allclose(entry[f"A_cross_eigvals_{stem}"].double(),
                          entry["A_eigvals"].double(), atol=1e-4)


def test_same_layer_mlp_selects_B(tmp_path):
    # An MLP hook with stored B (and G) must project against B, not silently fall back to A.
    d = 16

    def spd():
        M = torch.randn(d, d, dtype=torch.float64)
        return M @ M.T + 0.1 * torch.eye(d, dtype=torch.float64)

    bvals, bvecs = eigh_descending(spd())
    gvals, gvecs = eigh_descending(spd())
    data = {
        "blk0.up": {
            "B_eigvals": bvals, "B_eigvecs": bvecs.float(), "n_B": 200,
            "G_eigvals": gvals, "G_eigvecs": gvecs.float(), "n_G": 200,
        },
        "__format__": "cov_svd",
    }
    path = str(tmp_path / "svd.pt")
    DataAccessor(data).project_same_layer(onto="both", output_path=path)
    entry = torch.load(path, map_location="cpu", weights_only=False)["blk0.up"]
    assert "G_cross_eigvals_B" in entry
    assert "B_cross_eigvals_G" in entry
    assert "G_cross_eigvals_A" not in entry


def test_convert_drops_mean_keeps_centered(tmp_path, make_factors_dict):
    # Raw mean is only persisted with +m, but centered eigvals are derived whenever a mean exists.
    factors = make_factors_dict(d=16, with_grad=False, with_means=True)
    cov_path = str(tmp_path / "cov.pt")
    DataAccessor(factors).save(cov_path, format="cov+m")

    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(cov_path).save(svd_path, format="cov_svd")
    entry = torch.load(svd_path, map_location="cpu", weights_only=False)["hook0"]
    assert "A_mean" not in entry
    assert "A_eigvals_centered" in entry

    svdm_path = str(tmp_path / "svd_m.pt")
    DataAccessor(cov_path).save(svdm_path, format="cov_svd+m")
    entry_m = torch.load(svdm_path, map_location="cpu", weights_only=False)["hook0"]
    assert "A_mean" in entry_m


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
    data["__hf_model__"] = "custom/x"
    data["__revision__"] = "rev99"
    torch.save(data, path)

    DataAccessor(path).save(path, format="eigenvalues")
    stamped = torch.load(path, map_location="cpu", weights_only=False)
    assert stamped["__hf_model__"] == "custom/x"
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
