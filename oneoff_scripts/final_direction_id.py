"""Token identity of the dominant direction at the END of the model (pythia-1b + 6.9b, final).

All existing attribution (type ranking, position split, substitution tests) was computed on
the blk3-defined rogue direction v1. The mid-stack follow-up then tracked THAT direction down
the stack. What was never asked: take the top eigendirection OF before_final_norm — call it
v_f — and ask which tokens carry it. Is it the same object as v1 (cos ~ 1, so the sink
survives in weakened form), a rotated relative, or something else that the late blocks build?

No new collection needed: configs/block_rogue_id.yaml already persists raw before_final_norm
samples at the final checkpoint for both models.

Saves data/results/final_direction_id.pt and prints the tables.
Run on a compute node (loads the raw acts: ~6 GB for 1b, ~14 GB for 6.9b).
"""
from typing import Any

import torch
from transformers import AutoTokenizer

from oneoff_scripts.rogue_id_followups import _newline_ids, _proj, _slots, _stats
from scripts.compute_metrics import rankme_metrics
from utils.accessor import DataAccessor

RUN = "data/inferences/block_rogue_id"
MODELS = {"pythia-1b-deduped": 143000, "pythia-6.9b-deduped": 143000}
MIX = "data/mixes/pile_30M_512.pt"
LEAF = "before_final_norm"
MIN_COUNT = 20        # token types rarer than this are not ranked (same bar as the v1 pass)
TOP_TYPES = 40


def _spectrum(view: Any) -> dict[str, float]:
    """How top-heavy is before_final_norm, and how much of that is the single direction."""
    ev = view.eigvals_centered.float().clamp(min=0)
    return {"top1_share": float(ev[0] / ev.sum()),
            "rankme": rankme_metrics(ev)["rankme"],
            "rankme_top1_removed": rankme_metrics(ev[1:])["rankme"]}


def _by_type(score: torch.Tensor, ids: torch.Tensor, tok: Any) -> list[tuple[str, float, int]]:
    """Mean |projection| per token type, descending; types with >= MIN_COUNT occurrences."""
    a, flat = score.abs().reshape(-1), ids.reshape(-1)
    out = []
    for t in torch.unique(flat).tolist():
        m = flat == t
        n = int(m.sum())
        if n >= MIN_COUNT:
            out.append((tok.convert_ids_to_tokens([t])[0], float(a[m].mean()), n))
    return sorted(out, key=lambda r: -r[1])


def analyse(model: str, step: int) -> dict:
    acc = DataAccessor(f"{RUN}/{model}/step{step}.pt")
    final: Any = acc[LEAF].acts
    v_f = final.eigvecs_centered[:, 0].float()
    write: Any = acc["blk3.mlp.out"].acts
    v_1 = write.eigvecs_centered[:, 0].float()

    score = _proj(acc, LEAF, v_f)
    mix = torch.load(MIX)
    K = mix.shape[1] - 1                      # collection keeps seq_len - 1 positions per window
    ids = mix[:score.numel() // K, :K]
    score = score.reshape(ids.shape)
    tok = AutoTokenizer.from_pretrained(f"EleutherAI/{model}")
    masks = _slots(ids, _newline_ids(tok, ids))

    var = score.reshape(-1) ** 2
    return {"cos_with_v1": float((v_f @ v_1).abs()),
            "spectrum": _spectrum(final),
            "abs_by_class": _stats(score, masks),
            "var_share_by_class": {k: float(var[m.reshape(-1)].sum() / var.sum())
                                   for k, m in masks.items()},
            "types": _by_type(score, ids, tok)[:TOP_TYPES],
            # the v1 score at the same leaf, for the side-by-side: same rows, other direction
            "v1_abs_by_class": _stats(_proj(acc, LEAF, v_1).reshape(ids.shape), masks)}


def main() -> None:
    res = {m: analyse(m, s) for m, s in MODELS.items()}
    for model, r in res.items():
        print(f"\n=== {model} @ {LEAF} ===")
        print(f"  |cos(v_f, v1_blk3)| = {r['cos_with_v1']:.4f}")
        print(f"  spectrum: {r['spectrum']}")
        print(f"  mean |proj on v_f| by class: {r['abs_by_class']}")
        print(f"  variance share by class:     {r['var_share_by_class']}")
        print(f"  mean |proj on v1|  by class: {r['v1_abs_by_class']}")
        print(f"  top {TOP_TYPES} types (type, mean |proj|, n):")
        for t, v, n in r["types"]:
            print(f"    {t!r:20s} {v:10.2f} {n:7d}")
    torch.save(res, "data/results/final_direction_id.pt")


if __name__ == "__main__":
    main()
