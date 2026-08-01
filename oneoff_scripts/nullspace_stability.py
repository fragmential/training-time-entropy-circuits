"""Is the unembedding's effective null space a stable target, or does it move?

rho (oneoff_scripts/entropy_neurons.py) is a relation between two things that both change:
a neuron's write direction and the bottom-k right-singular subspace V0 of W_U. A neuron
cannot align with V0 before V0 stops being an arbitrary k-plane, so the flat pre-transition
plateau has two candidate explanations -- the neurons hadn't moved, or there was nothing to
move towards. This measures the second.

    overlap(A, B) = ||A B^T||_F^2 / k   in [0, 1],  = k/d for two independent random k-planes

reported between consecutive checkpoints (is V0 holding still?) and against the final
checkpoint (has it locked onto where it ends up?), alongside the tail of the singular
spectrum (is the bottom of W_U separating from the bulk at all?).

Reads the right singular vectors already saved by oneoff_scripts/unembedding_spectra.py.
Note those are the RAW head weight, without the centring and final-norm folding used for
rho -- the question here is stability, which those corrections do not affect.

Writes data/results/nullspace_stability.pt.
"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import get_token_count

VEC_DIR = "data/results/unembedding_svd"
SPECTRA = "data/results/unembedding_spectra.pt"


def _overlap(a, b):
    """Squared Frobenius overlap of two orthonormal row-bases, normalised to [0, 1]."""
    return float((a @ b.T).pow(2).sum() / a.shape[0])


def main(out_path: str = "data/results/nullspace_stability.pt", frac: float = 0.01) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spectra = torch.load(SPECTRA, weights_only=False) if os.path.exists(SPECTRA) else {}
    out = {}
    for model in sorted(os.listdir(VEC_DIR)):
        matches = [re.search(r"step(\d+)", f) for f in glob.glob(f"{VEC_DIR}/{model}/step*.pt")]
        steps = sorted(int(m.group(1)) for m in matches if m)
        if len(steps) < 2:
            continue
        load = lambda s: torch.load(f"{VEC_DIR}/{model}/step{s}.pt", map_location=dev,
                                    weights_only=True).float()
        vh_final = load(steps[-1])
        d = vh_final.shape[1]
        k = max(1, round(frac * d))
        res = {"steps": steps, "k": k, "d": d, "chance": k / d,
               "tokens": [get_token_count(model, s) for s in steps],
               "to_next": [], "to_final": [], "tail_ratio": []}
        prev = None
        for s in steps:
            vh = load(s)
            bottom = vh[-k:]
            res["to_final"].append(_overlap(bottom, vh_final[-k:]))
            if prev is not None:
                res["to_next"].append(_overlap(prev, bottom))
            prev = bottom
            sv = spectra.get(model, {}).get(s)
            res["tail_ratio"].append(float(sv[-k:].mean() / sv.median()) if sv is not None
                                     else float("nan"))
        out[model] = res
        print(f"{model}: k={k} d={d} chance={k/d:.4f}", flush=True)
        for i, s in enumerate(steps):
            nxt = res["to_next"][i] if i < len(res["to_next"]) else float("nan")
            print(f"  step{s:>7d} {res['tokens'][i]:.2e}  to_next={nxt:.3f} "
                  f"to_final={res['to_final'][i]:.3f}  tail/med={res['tail_ratio'][i]:.3f}",
                  flush=True)
    torch.save(out, out_path)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
