"""Plots for collapse_geometry.py: the frequency handover, the cone, and neural collapse.

Row per model, columns:
  1  frequency  share of the final stream and of the last MLP's write lying along d_freq
                (the stream direction whose logits are proportional to log unigram
                frequency), with r2 of that fit -- how well W_U can express it at all.
  2  cone       anisotropy (mean pairwise cosine) and mean-share of the second moment, for
                the post-final-norm stream and for the unembedding rows. Uncentred, as the
                representation-degeneration literature defines them.
  3+ NC1..NC4   the metrics as Wu & Papyan (arXiv:2405.17767) define them for language
                models: CDNV for NC1, CoV of log class-mean norms and CoV of the
                interference for NC2, their GNC2 hyperspherical uniformity, NC3 self-duality
                with their UNC3 uniform-duality variant, and NC4 nearest-class-centre
                agreement.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib.pyplot as plt
import numpy as np
import torch

PANELS = (
    ("frequency", (("afn_freq_share", "stream along d_freq"), ("mlp_freq_share", "last MLP write"),
                   ("freq_r2", "r2 of W_U d ~ log f"))),
    ("cone", (("afn_anisotropy", "stream anisotropy"), ("afn_cone", "stream mean-share"),
              ("head_anisotropy", "unembedding anisotropy"), ("head_cone", "unembedding mean-share"))),
    ("NC1 (Wu & Papyan CDNV)",
     (("afn_nc1_log_cdnv", "NC1: mean LOG class-distance normalised variance"),)),
    ("NC2 geometry", (("afn_nc2_equinorm_cv_log", "NC2 equinormness: CoV of log class-mean norms"),
                      ("afn_nc2_equiangle_cv", "NC2 equiangularity: CoV of interference"),
                      ("afn_nc2_cos_mean", "mean interference"),
                      ("afn_nc2_cos_target", "its simplex-ETF target, -1/(C-1)"))),
    ("NC3 / NC4", (("afn_nc3_dual", "NC3 self-duality: mean cos(unembedding row, class mean)"),
                   ("afn_unc3_dual_cv", "UNC3 uniform duality: CoV of those cosines"),
                   ("afn_nc4_agree", "NC4: nearest-class-centre agreement"))),
    ("GNC2", (("afn_gnc2_uniformity", "GNC2 hyperspherical uniformity"),)),
    ("frequency in W_U", (("freq_corr_head_norm", "corr(||W_U row||, log f)"),
                          ("d_freq_null_share", "d_freq share in the null space"))),
    ("equinorm / equiangular", (("afn_nc2_cv_class_norms", "CV class-mean norms"),
                                ("afn_nc2_cv_head_norms", "CV head-row norms"),
                                ("afn_nc2_cos_mean", "mean pairwise cos"),
                                ("afn_nc2_cos_std", "std pairwise cos"))),
)


def main(in_path: str = "data/results/collapse_geometry.pt",
         out_path: str = "analysis/figures/entropy_neurons/collapse_geometry.png") -> None:
    data = torch.load(in_path, weights_only=False)
    models = [m for m in data if data[m]]
    fig, axes = plt.subplots(len(models), len(PANELS), figsize=(4.2 * len(PANELS), 3 * len(models)),
                             squeeze=False)
    for row, model in enumerate(models):
        steps = sorted(data[model])
        xs = np.array([data[model][s]["tokens"] for s in steps], dtype=float)
        for col, (title, keys) in enumerate(PANELS):
            ax = axes[row, col]
            for key, label in keys:
                ys = np.array([data[model][s].get(key, np.nan) for s in steps], dtype=float)
                if np.isfinite(ys).any():
                    ax.plot(xs, ys, marker=".", ms=3, lw=1.3, label=label)
            ax.set(xscale="log", xlabel="tokens", title=title if row == 0 else "")
            # the log-CDNV is already logarithmic; raw CDNV spans 1e24 and is unreadable
            ax.legend(fontsize=6)
            if col == 0:
                ax.set_ylabel(model)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
