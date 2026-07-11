"""Toy dumps load through DataAccessor / compute_metrics exactly like collected model data."""

from dataclasses import replace

import pytest

from scripts.compute_metrics import get_metrics
from toy.train import VARIANTS, train
from utils.accessor import DataAccessor


@pytest.fixture(scope="module")
def micro(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("toy")
    specs = {"single": replace(VARIANTS["single"], steps=5),
             "multi": replace(VARIANTS["multi_residual"], steps=5, depth=2)}
    for name, spec in specs.items():
        train(name, spec=spec, output_root=str(root / "inf"), results_root=str(root / "res"),
              checkpoints=2)
    return {name: str(root / "inf" / f"toy_{name}" / f"toy-{name}" / "step5.pt") for name in specs}


def test_single_dump_reads_like_model_data(micro: dict[str, str]) -> None:
    acc = DataAccessor(micro["single"])
    assert not acc.needs_model_weights()
    metrics = get_metrics(acc.v)
    assert metrics["before_final_norm"]["acts_uncentered"]["rankme"] > 1
    assert "acts_centered" in metrics["after_final_norm"]


def test_multi_dump_yields_block_ledger(micro: dict[str, str]) -> None:
    acc = DataAccessor(micro["multi"])
    assert not acc.needs_model_weights()
    metrics = get_metrics(acc.v)
    ledger = metrics["blk0"]["block_ledger"]
    assert {"delta_s", "chi", "quality", "interference"} <= set(ledger)
    assert "w" in metrics[""]["overlap_chi"]
    assert metrics["blk1.attn.in"]["acts_uncentered"]["rankme"] > 1
