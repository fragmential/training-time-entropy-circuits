import os
import torch
from utils.accessor import DataAccessor, decomp_profiler, eigh_descending

# Fixture entries live at hook "after_final_norm" with signals value.acts / value.grads.
HOOK = "after_final_norm"
ACTS = "value.acts"
GRADS = "value.grads"


def test_save_load_cov(tmp_path, make_factors_dict):
    factors = make_factors_dict(d=32, with_grad=True, with_means=False)
    path = str(tmp_path / "cov.pt")
    DataAccessor(factors).save(path, format="cov")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert HOOK in data
    assert ACTS in data[HOOK]
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
    factors = {HOOK: {ACTS: acts, f"n_{ACTS}": N, "n": N}}
    path = str(tmp_path / "acts.pt")
    DataAccessor(factors).save(path, format="acts")
    data = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.allclose(data[HOOK][ACTS], acts.float(), atol=1e-6)


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
    n = factors[HOOK]["n"]
    original_cov = factors[HOOK][ACTS].float() / n

    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    V = entry[f"{ACTS}_eigvecs"]
    S = entry[f"{ACTS}_eigvals"]
    reconstructed = V @ torch.diag(S.to(V.dtype)) @ V.T
    assert torch.allclose(reconstructed, original_cov, atol=1e-4)


def test_same_layer_cross_basis(tmp_path, make_factors_dict):
    # Residual hook: forward-acts signal is the captured value.acts (no model needed).
    factors = make_factors_dict(d=16, with_grad=True, with_means=False)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    acc = DataAccessor(path)
    acc.project_same_layer(onto="both", output_path=path)
    cross_keys = [k for k in acc.data[HOOK] if "cross_eigvals" in k]
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
    DataAccessor(path).project_onto_basis(path, output_path=path)

    stem = os.path.splitext(os.path.basename(path))[0]
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert torch.allclose(entry[f"{ACTS}_cross_eigvals_{stem}"].double(),
                          entry[f"{ACTS}_eigvals"].double(), atol=1e-4)


def test_same_layer_mlp_selects_out_acts(tmp_path):
    # An MLP hook with stored out.acts (and out.grads) must project against out.acts,
    # not silently fall back to the input acts.
    d = 16

    def spd():
        M = torch.randn(d, d, dtype=torch.float64)
        return M @ M.T + 0.1 * torch.eye(d, dtype=torch.float64)

    ovals, ovecs = eigh_descending(spd())
    gvals, gvecs = eigh_descending(spd())
    data = {
        "blk0.up": {
            "out.acts_eigvals": ovals, "out.acts_eigvecs": ovecs.float(), "n_out.acts": 200,
            "out.grads_eigvals": gvals, "out.grads_eigvecs": gvecs.float(), "n_out.grads": 200,
        },
        "__format__": "cov_svd",
    }
    path = str(tmp_path / "svd.pt")
    DataAccessor(data).project_same_layer(onto="both", output_path=path)
    entry = torch.load(path, map_location="cpu", weights_only=False)["blk0.up"]
    assert "out.grads_cross_eigvals_out.acts" in entry
    assert "out.acts_cross_eigvals_out.grads" in entry
    assert "out.grads_cross_eigvals_in.acts" not in entry


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
    DataAccessor({HOOK: {ACTS: acts, f"n_{ACTS}": len(acts), "n": len(acts)}}).save(path, format="cov_svd")
    entry = torch.load(path, map_location="cpu", weights_only=False)[HOOK]
    assert torch.allclose(entry[f"{ACTS}_mean"], acts.mean(0), atol=1e-6)


def test_derived_preserve_and_drop_overrides(tmp_path):
    # MLP hook with a stored derived out.acts: cov_svd keeps it, cov_svd-b drops it.
    data = {
        "blk0.up": {
            "n": 5,
            "in.acts": torch.eye(4) * 5,
            "in.acts_mean": torch.ones(4),
            "out.acts": torch.eye(4) * 10,
            "out.acts_mean": torch.arange(4).float(),
        }
    }

    svd_path = str(tmp_path / "svd.pt")
    DataAccessor(data).save(svd_path, format="cov_svd")
    entry = torch.load(svd_path, map_location="cpu", weights_only=False)["blk0.up"]
    assert "out.acts_eigvals" in entry
    assert "out.acts_eigvecs" in entry
    assert "out.acts_mean" in entry

    drop_path = str(tmp_path / "drop.pt")
    DataAccessor(data).save(drop_path, format="cov_svd-b")
    entry_drop = torch.load(drop_path, map_location="cpu", weights_only=False)["blk0.up"]
    assert not any(k.startswith("out.acts") for k in entry_drop)


def test_eigenvalues_preserves_means_and_derived_by_default(tmp_path):
    data = {
        "blk0.up": {
            "n": 5,
            "in.acts": torch.eye(4) * 5,
            "in.acts_mean": torch.ones(4),
            "out.acts": torch.eye(4) * 10,
            "out.acts_mean": torch.arange(4).float(),
        }
    }
    path = str(tmp_path / "eig.pt")
    DataAccessor(data).save(path, format="eigvals")
    entry = torch.load(path, map_location="cpu", weights_only=False)["blk0.up"]
    assert {"in.acts_mean", "out.acts_mean", "in.acts_eigvals", "out.acts_eigvals"} <= set(entry)
    assert "in.acts_eigvecs" not in entry
    assert "out.acts_eigvecs" not in entry


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
