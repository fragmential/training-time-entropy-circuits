"""Token attribution for the blk3.mlp sink direction (pythia-1b, final checkpoint).

Reduces the three non-portable inputs showcase.py §4 used to open at plot time -- the raw acts
(data/inferences/block_rogue_id, 6 GB), the packed mix (data/mixes/pile_30M_512.pt, 229 MB) and
the pythia tokenizer -- to the scores the two figures actually draw: each collected row's signed
projection onto the write's top centered eigendirection, plus the same scores aggregated per
token type.

Both of the notebook's newline rules are preserved: the histogram groups rows by 'C-with-a-dot'
or '\\n' in the RAW token string, the ranking colours bars by '\\n' in the DECODED string. The
n >= 20 cut and the top-k stay notebook-side, so the full per-type table is saved.

Saves data/results/rogue_id_attribution.pt. Run on a GPU node: the run stored raw samples, so
the eigendirection is derived here (261k x 2048 covariance + eigh).
"""
from typing import Any

import torch
from transformers import AutoTokenizer

from utils.accessor import DataAccessor

MODEL, STEP, LEAF = "pythia-1b-deduped", 143000, "blk3.mlp.out"
MIX, K = "data/mixes/pile_30M_512.pt", 511   # seq_len 512, token_selection=all -> chunk's last dropped
OUT = "data/results/rogue_id_attribution.pt"


def projections(model: str) -> torch.Tensor:
    """Each row's signed projection onto the write's top centered eigendirection."""
    v: Any = DataAccessor(f"data/inferences/block_rogue_id/{model}/step{STEP}.pt")[LEAF].acts
    return (v.samples.float() - v.mean.float()) @ v.eigvecs_centered[:, 0].float()


def token_ids(n: int) -> torch.Tensor:
    """The token id behind each collected row -- chunks in file order, last position dropped."""
    return torch.load(MIX, weights_only=True)[:, :K].reshape(-1)[:n]


def newline_ids(tok: Any, uniq: torch.Tensor) -> set[int]:
    toks = tok.convert_ids_to_tokens(uniq.tolist())        # one vectorized call, not 50k decodes
    return {int(i) for i, t in zip(uniq, toks) if "Ċ" in t or "\n" in t}   # 'Ċ' = byte-level \n


def per_type(uniq, inv, counts, score, tok: Any) -> dict:
    """Mean and (population) variance of |score| per token TYPE, decoded for the bar labels."""
    a = score.abs()
    mean = torch.zeros(len(uniq)).index_add_(0, inv, a) / counts
    return {"token": tok.batch_decode([[int(i)] for i in uniq]), "mean_abs": mean,
            "var_abs": torch.zeros(len(uniq)).index_add_(0, inv, a.square()) / counts - mean.square(),
            "count": counts}


def analyse(model: str) -> dict:
    score = projections(model)
    ids = token_ids(len(score))
    tok = AutoTokenizer.from_pretrained(f"EleutherAI/{model}")
    uniq, inv, counts = torch.unique(ids, return_inverse=True, return_counts=True)
    nl = newline_ids(tok, uniq)
    return {"step": STEP, "leaf": LEAF, "K": K, "score": score,
            "is_newline": torch.tensor([int(t) in nl for t in ids]),
            "is_pos0": (torch.arange(len(score)) % K) == 0,
            "types": per_type(uniq, inv, counts, score, tok)}


def main() -> None:
    res = {MODEL: analyse(MODEL)}
    for m, r in res.items():
        print(f"{m}: {len(r['score'])} rows, {int(r['is_newline'].sum())} newline-bearing, "
              f"{int(r['is_pos0'].sum())} window-start, {len(r['types']['count'])} token types")
    torch.save(res, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
