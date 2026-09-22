"""The spike-direction depth table for the compression-valley section, as LaTeX.

Columns are the residual read points from the spike layer onward, plus after the final norm.
Two reference directions are tracked across depth: the leading direction of the blk3 MLP write
(the spike direction) and the leading direction of the final residual. Per column the table
gives the cosine between each reference and that depth's own leading direction, so the handover
from one to the other is visible. Earlier depths are omitted: the stream carries too little
variance before the spike write for a leading direction to mean anything there.

Everything is measured on the CENTERED covariance, which the store does not hold directly:
it keeps the uncentered eigendecomposition and the mean, so Sigma_c = V diag(lam) V^T - mu mu^T
as in oneoff_scripts/alpha_conservation.py.

Reads data/inferences/spike_direction_depth (configs/spike_direction_depth.yaml).
Writes the table body to stdout. CPU, a few seconds.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

SRC = "data/inferences/spike_direction_depth/pythia-1b-deduped/step143000.pt"
L = 16
START = 4          # the spike write lands here; earlier depths carry no meaningful direction
LABEL = "tab:spike_ablation"


def centered(store: dict, leaf: str) -> torch.Tensor:
    e = store[leaf]
    lam = e["acts_eigvals"].double().clamp_min(0)
    V = e["acts_eigvecs"].double()
    mu = e["acts_mean"].double()
    return (V * lam) @ V.T - torch.outer(mu, mu)


def leading(S: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigh(S)[1][:, -1]


def measure() -> tuple[list, dict]:
    d = torch.load(SRC, weights_only=False)
    cols = [(f"blk{i}.attn.in", str(i)) for i in range(START, L)] + \
           [("before_final_norm", str(L)), ("after_final_norm", "afn")]
    spike = leading(centered(d, "blk3.mlp.out"))
    ref = leading(centered(d, "before_final_norm"))
    out = {}
    for leaf, name in cols:
        S = centered(d, leaf)
        top = leading(S)
        out[name] = dict(cs=float(abs(spike @ top)), cr=float(abs(ref @ top)))
    return [n for _, n in cols], out


def emit(names: list, R: dict) -> str:
    def cosine(key, other):
        """Bold the larger of the two references per column, which marks the handover."""
        cells = []
        for n in names:
            txt = f"{R[n][key]:.2f}"
            cells.append(f"\\textbf{{{txt}}}" if R[n][key] > R[n][other] else txt)
        return cells
    # v, not u: u_ell is the compound update in the notation section. v is unused there, and
    # is the conventional eigenvector symbol. Subscript indexes depth throughout, so v_L is
    # just v_ell at the final residual.
    rows = [(r"$\cos(\mathbf v_\text{s}, \mathbf v_\ell)$", cosine("cs", "cr")),
            (r"$\cos(\mathbf v_L, \mathbf v_\ell)$", cosine("cr", "cs"))]
    body = "\n".join(f"{lbl} & " + " & ".join(cells) + r" \\" for lbl, cells in rows)
    return (
        "\\begin{table}[ht]\n\\centering\n\\setlength{\\tabcolsep}{2pt}\n"
        "\\renewcommand{\\arraystretch}{1.25}\n"
        "\\caption{Leading direction of the residual at each depth, Pythia~1B at its final "
        "checkpoint. $\\mathbf v_\\ell$ is the leading eigenvector of "
        "$\\bm\\Sigma_{\\mathbf r_\\ell}$, and of $\\bm\\Sigma_{\\hat{\\mathbf r}_L}$ for afn; "
        "$\\mathbf v_\\text{s}$ is that of the Layer~4 MLP write, the spike direction. Bold "
        "marks the larger cosine per column.}\n"
        f"\\label{{{LABEL}}}\n"
        "\\begin{tabular}{@{}l" + "r" * len(names) + "@{}}\n\\toprule\n"
        f"Residual point & {' & '.join(names)} \\\\\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    names, R = measure()
    print(emit(names, R))
