import torch
from utils.accessor import DataAccessor


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
    factors = make_factors_dict(d=16, with_grad=True, with_means=False)
    path = str(tmp_path / "svd.pt")
    DataAccessor(factors).save(path, format="cov_svd")
    acc = DataAccessor(path)
    acc.add_cross_basis(onto="both")
    entry = acc.data["hook0"]
    cross_keys = [k for k in entry if "cross_eigvals" in k]
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
