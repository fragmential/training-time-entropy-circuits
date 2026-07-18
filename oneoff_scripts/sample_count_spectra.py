"""Finite-sample dependence of the output-leaf spectra (pythia-1b, final checkpoint).

For each geometry (packed all-token / padded last-token) and each leaf (before/after final
norm), computes the centered eigenspectrum on nested random row subsets bracketing the
standard collection budgets (padded: 16,384 rows; packed: 262,144 tokens). Feeds
showcase_appendix §E — insight into how much of tail structure is sample-count artifact
(Marchenko–Pastur-style broadening at small N/d).

Inputs: data/inferences/sample_count_{packed,padded}/pythia-1b-deduped/step143000.pt
(configs/sample_count_*.yaml). Output: data/results/sample_count_spectra.pt.
Run on a GPU node (a handful of 2048x2048 eigendecompositions).
"""
from typing import Any

import torch

from utils.accessor import DataAccessor

NS = {"packed": (16_384, 32_768, 65_536, 131_072, 262_144, 524_288),
      "padded": (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)}
LEAVES = ("before_final_norm", "after_final_norm")


def main() -> None:
    torch.manual_seed(0)
    out: dict[str, Any] = {}
    for geom, ns in NS.items():
        acc = DataAccessor(f"data/inferences/sample_count_{geom}/pythia-1b-deduped/step143000.pt")
        for leaf in LEAVES:
            v: Any = acc[leaf].acts
            X = v.samples.float()
            perm = torch.randperm(X.shape[0])
            for n in ns:
                if n > X.shape[0]:
                    continue
                S = X[perm[:n]]
                S = S - S.mean(0)
                lam = torch.linalg.eigvalsh(S.T @ S / n).flip(0).clamp(min=0)
                out[f"{geom}.{leaf}.{n}"] = lam.cpu()
                print(f"{geom} {leaf} n={n}: top {lam[0]:.1f}, "
                      f"rankme {float(torch.exp(-(p := lam / lam.sum())[p > 0].log().mul(p[p > 0]).sum())):.1f}",
                      flush=True)
    torch.save(out, "data/results/sample_count_spectra.pt")
    print("saved data/results/sample_count_spectra.pt")


if __name__ == "__main__":
    main()
