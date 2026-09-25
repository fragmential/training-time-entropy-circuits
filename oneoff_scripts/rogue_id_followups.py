"""Follow-up analyses on the blk3.mlp rogue direction (pythia-1b, final checkpoint).

A) EOT-twin (data/inferences/block_rogue_id_eot): windows are the same pile mix but with an
   explicit <|endoftext|> prepended at position 0. Question: does an explicit resting token
   absorb the first-newline sink slot?
B) Mid-stack (data/inferences/block_rogue_id_midstack): stream inputs at several depths.
   Question: do ordinary tokens carry rogue-direction content in their own stream mid-stack
   (content the late blocks then remove before before_final_norm)?

Saves data/results/rogue_id_followups.pt and prints the summary tables.
Run on a compute node (loads 6 + 14 GB of raw acts).
"""
from typing import Any

import torch
from transformers import AutoTokenizer

from utils.accessor import DataAccessor

MODEL, STEP = "pythia-1b-deduped", 143000


def _newline_ids(tok: Any, ids: torch.Tensor) -> set[int]:
    uniq = torch.unique(ids)
    toks = tok.convert_ids_to_tokens(uniq.tolist())
    return {int(i) for i, t in zip(uniq, toks) if "Ċ" in t or "\n" in t}


def _slots(ids: torch.Tensor, nl_ids: set[int]) -> dict[str, torch.Tensor]:
    """Boolean masks over (windows, K) rows: position 0, first newline per window,
    other newlines, everything else."""
    is_nl = torch.tensor([[int(t) in nl_ids for t in row] for row in ids.tolist()])
    pos0 = torch.zeros_like(is_nl)
    pos0[:, 0] = True
    first_nl = torch.zeros_like(is_nl)
    for w in range(is_nl.shape[0]):
        idx = is_nl[w].nonzero()
        if len(idx):
            first_nl[w, idx[0, 0]] = True
    first_nl &= ~pos0
    return {"pos0": pos0, "first_nl": first_nl,
            "other_nl": is_nl & ~first_nl & ~pos0,
            "bulk": ~is_nl & ~pos0}


def _proj(acc: DataAccessor, leaf: str, v1: torch.Tensor) -> torch.Tensor:
    v: Any = acc[leaf].acts
    return (v.samples.float() - v.mean.float()) @ v1


def _stats(score: torch.Tensor, masks: dict[str, torch.Tensor]) -> dict[str, float]:
    a = score.abs().reshape(-1)
    return {k: float(a[m.reshape(-1)].mean()) for k, m in masks.items()}


def analyse(run: str, mix_path: str, stream_leaves: list[str]) -> dict:
    acc = DataAccessor(f"data/inferences/{run}/{MODEL}/step{STEP}.pt")
    w: Any = acc["blk3.mlp.out"].acts
    v1 = w.eigvecs_centered[:, 0].float()
    score = _proj(acc, "blk3.mlp.out", v1)
    mix = torch.load(mix_path)
    K = mix.shape[1] - 1                      # collection keeps seq_len - 1 positions per window
    ids = mix[:score.numel() // K, :K]        # the run consumed the head of the mix
    tok = AutoTokenizer.from_pretrained(f"EleutherAI/{MODEL}")
    masks = _slots(ids, _newline_ids(tok, ids))
    out: dict[str, Any] = {"v1": v1, "write": _stats(score.reshape(ids.shape), masks)}
    for leaf in stream_leaves:
        v: Any = acc[leaf].acts
        proj = _proj(acc, leaf, v1).reshape(ids.shape)
        norms = (v.samples.float() - v.mean.float()).norm(dim=1).reshape(ids.shape)
        out[leaf] = {"abs": _stats(proj, masks),
                     "frac": {k: float((proj.abs() / norms)[m].mean()) for k, m in masks.items()}}
    return out


def main() -> None:
    res: dict[str, Any] = {
        "eot": analyse("block_rogue_id_eot", "data/mixes/pile_30M_512_eot.pt",
                       ["blk3.attn.in", "before_final_norm"]),
        "midstack": analyse("block_rogue_id_midstack", "data/mixes/pile_30M_512.pt",
                            ["blk4.attn.in", "blk6.attn.in", "blk9.attn.in",
                             "blk12.attn.in", "blk15.attn.in", "before_final_norm"]),
    }
    # original run, for the side-by-side baseline (same mix as midstack)
    res["orig"] = analyse("block_rogue_id", "data/mixes/pile_30M_512.pt", ["before_final_norm"])
    for a, b in (("orig", "midstack"), ("orig", "eot")):
        res[f"v1_cos_{a}_{b}"] = float((res[a]["v1"] @ res[b]["v1"]).abs())
    for name, r in [(k, v) for k, v in res.items() if isinstance(v, dict)]:
        print(f"\n=== {name} ===")
        for leaf, s in [(k, v) for k, v in r.items() if k != "v1"]:
            print(f"  {leaf}: {s}")
    print({k: v for k, v in res.items() if k.startswith("v1_cos")})
    for r in res.values():
        if isinstance(r, dict):
            r.pop("v1")
    torch.save(res, "data/results/rogue_id_followups.pt")


if __name__ == "__main__":
    main()
