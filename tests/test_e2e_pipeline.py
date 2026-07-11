"""Full-pipeline e2e: collect.py main() over MULTIPLE checkpoints (schedule → collect →
save → inline metrics), in both cov and samples (collect_format=acts) modes, then
compute_metrics.py main() over the produced files. Compute node required (downloads):
    srun ... uv run pytest tests/test_e2e_pipeline.py -v
"""
import os
import pytest
import numpy as np
import torch

pytestmark = pytest.mark.e2e

MODEL = "EleutherAI/pythia-14m-deduped"
STEPS = [0, 512]


def _cfg(tmp_path, **overrides):
    from scripts.collect import CollectConfig
    base = dict(model_name=MODEL, checkpoints=STEPS, num_samples=30, seq_len=128,
                max_length=128, batch_size=8, packing="packed", token_selection="all",
                hooks=["blk*.attn.out", "blk*.mlp.out", "before_final_norm"],
                max_layers_per_pass=0, keep_cached=True, output_dir=str(tmp_path),
                hf_progress_bars=False)
    return CollectConfig(**base | overrides)


def _saved(tmp_path):
    mdir = os.path.join(str(tmp_path), MODEL.split("/")[-1])
    return {s: torch.load(os.path.join(mdir, f"step{s}.pt"), map_location="cpu", weights_only=False)
            for s in STEPS}


def test_collect_main_multicheckpoint_cov(tmp_path):
    from scripts.collect import main
    main(_cfg(tmp_path, storage_format="cov_svd"))
    for s, data in _saved(tmp_path).items():
        assert data["__format__"] == "cov_svd"
        assert data["before_final_norm"]["acts_eigvals"] is not None
        assert "blk0.attn.out" in data and "blk0.mlp.out" in data


def test_collect_main_multicheckpoint_samples_mode(tmp_path):
    from scripts.collect import main
    from scripts.compute_metrics import main as metrics_main
    main(_cfg(tmp_path, collect_format="acts", storage_format="eigenvalues",
              activation_dtype="bf16", compute_metrics=True))

    for s, data in _saved(tmp_path).items():
        assert data["__format__"] == "eigenvalues"
        for leaf, entry in data.items():
            if not leaf.startswith("__"):
                assert "acts_eigvals" in entry and "acts_eigvals_centered" in entry
                assert "acts_samples" not in entry            # samples never persisted

    # inline metrics wrote both steps with the leaf spectral family
    res_path = os.path.join(str(tmp_path), f"results_{MODEL.split('/')[-1]}.npy")
    res = np.load(res_path, allow_pickle=True).item()
    assert set(res) == set(STEPS)
    assert "rankme" in res[STEPS[0]]["before_final_norm"]["acts_uncentered"]

    # compute_metrics.py main() over the same files (separate output) agrees on the keys
    metrics_main(config_directory=str(tmp_path), model_name=MODEL.split("/")[-1],
                 num_workers=2, derive=False, output_root=str(tmp_path / "res2"))
    res2 = np.load(os.path.join(str(tmp_path), "res2", os.path.basename(str(tmp_path)),
                                f"results_{MODEL.split('/')[-1]}.npy"), allow_pickle=True).item()
    assert set(res2) == set(STEPS)
    # inline run had live model weights (derived leaves like after_final_norm); derive=False can't
    assert set(res2[STEPS[0]]) <= set(res[STEPS[0]]) and "before_final_norm" in res2[STEPS[0]]
