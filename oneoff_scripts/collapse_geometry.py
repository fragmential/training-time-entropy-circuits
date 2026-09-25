"""Does the unembedding hand its frequency-biasing role to the final layer, and does the
final stream head for neural collapse while that happens?

Three literatures, one forward pass per checkpoint.

FREQUENCY (Stolfo et al. define v_freq = log p_freq - mean(log p_freq) from corpus counts
and never ask what W_U does with it). The unembedding CAN implement frequency bias: the
stream direction that produces it is the least-squares d_freq with W_U d_freq ~ v_freq. So
measure how well W_U can express it at all (r2), how much of the final stream lies along
d_freq, and how much the last MLP writes along it. A handover shows up as the stream share
falling while the MLP's write share rises.

CONE / ANISOTROPY (Gao et al. representation degeneration; Ethayarajh). Mean pairwise cosine
between token representations, and the share of the second moment carried by the mean
direction -- computed for the stream AND for the unembedding rows, which is where the
original degeneration claim lives.

NEURAL COLLAPSE, as adapted for language models by Wu & Papyan (arXiv:2405.17767): NC1 via
their class-distance normalised variance with the extra power, NC2 equinormness (CoV of LOG
class-mean norms) and equiangularity (CoV of the interference), their GNC2 hyperspherical
uniformity, NC3 self-duality and their UNC3 uniform duality (CoV of the cosines), and NC4
nearest-class-centre agreement. The textbook forms of NC1 and NC3 are exactly the ones they
had to modify, so using those would not test their claim.

Writes {model: {step: {...}}} to data/results/collapse_geometry.pt.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from utils.model_registry import (_fam, _getattr_path, get_checkpoint_schedule,
                                  get_final_layernorm, get_head_module, get_model_config,
                                  get_num_layers, get_post_mlp_norm, get_token_count, load_model,
                                  delete_cached_revision)
from utils.nullspace import head_subspace

MIXES = {"pythia": "data/mixes/pile_30M_512.pt", "olmo": "data/mixes/olmo_mix_30M_512.pt",
         "nanochat": "data/mixes/fineweb_edu_nanochat_1M_512.pt"}


def _anisotropy(x):
    """Mean cosine between distinct rows, exactly: the classic anisotropy number. 0 is
    isotropic, 1 is a single ray."""
    u = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n = u.shape[0]
    return float((u.sum(0).pow(2).sum() - n) / (n * (n - 1)))


def _cone(x):
    """Share of the second moment carried by the mean direction -- how far off the origin
    the cloud sits, which is what 'cone-shaped' actually means."""
    mu = x.mean(0)
    return float(mu.pow(2).sum() / (x.pow(2).sum() / x.shape[0]))


def _freq_direction(w_eff, log_f):
    """The stream direction whose logits are proportional to log token frequency, by least
    squares, with the r2 of that fit: how well the unembedding can express frequency bias
    at all. w_eff is the centred, norm-folded head."""
    u, s, vh = torch.linalg.svd(w_eff, full_matrices=False)
    keep = s > s[0] * 1e-6
    d = vh[keep].T @ ((u[:, keep].T @ log_f) / s[keep])
    resid = w_eff @ d - log_f
    return d / d.norm(), 1 - float(resid.pow(2).sum() / log_f.pow(2).sum())


def _nc1_cdnv(buf, min_count=30, chunk=2048):
    """Their NC1: class-distance normalised variance with the extra power (k=2),

        sigma_hat(c,c') = (sigma_c^2 + sigma_c'^2) / (2 ||mu_c - mu_c'||^2) / ||mu_c - mu_c'||^2

    reported as the AVERAGE OF THE LOGARITHM, which is what their Figure 5 plots against
    training. Pairs are reduced in chunks, so C can be tens of thousands without the C x C
    matrix ever existing.
    """
    keep = torch.nonzero(buf["n"] >= min_count, as_tuple=True)[0]
    if len(keep) < 5:
        return {}
    n = buf["n"][keep]
    mu = buf["s"][keep].double() / n[:, None]
    sig2 = (buf["sq"][keep] / n - mu.pow(2).sum(1)).clamp_min(0)
    tot_log, tot, npairs = 0.0, 0.0, 0
    for i in range(0, len(keep), chunk):
        d2 = torch.cdist(mu[i:i + chunk], mu).pow(2)
        num = sig2[i:i + chunk, None] + sig2[None, :]
        m = d2 > 0
        val = (num[m] / (2 * d2[m]) / d2[m]).clamp_min(1e-30)
        tot_log += float(val.log().sum()); tot += float(val.sum()); npairs += int(m.sum())
    return {"nc1_log_cdnv": tot_log / npairs, "nc1_cdnv": tot / npairs,
            "nc1_n_classes": len(keep)}


def _collapse(x, labels, w_rows, min_count=30, max_classes=400):
    """Neural collapse as adapted for language models by Wu & Papyan, "Linguistic Collapse:
    Neural Collapse in (Large) Language Models" (arXiv:2405.17767), with next-token identity
    as the class. Their adaptations exist because token classes are extremely imbalanced,
    contexts are ambiguous, and models see few epochs, so the textbook forms do not apply.

      nc1_cdnv     class-distance normalised variance with their EXTRA power (k=2), which
                   downweights well-separated pairs:
                       (sigma_c^2 + sigma_c'^2) / (2 ||mu_c - mu_c'||^2) / ||mu_c - mu_c'||^2
      nc2_equinorm CoV of the LOGARITHMS of the centred class-mean norms (not of the norms)
      nc2_equiangle CoV of the pairwise cosines ("interference"), whose target is -1/(C-1)
      gnc2_uniform hyperspherical uniformity, mean of log ||mu_hat_c - mu_hat_c'||^-1, their
                   relaxation of the simplex ETF, which is unreachable once C > d+1
      nc3_dual     self-duality: mean cosine between an unembedding row and its class mean
      unc3_dual_cv UNIFORM duality: CoV of those cosines, which they introduce because the
                   direct alignment does not emerge with scale
      nc4_agree    nearest-class-centre agreement: the share of tokens where argmax_c <w_c,h>
                   picks the same class as argmin_c ||h - mu_c||
    """
    ids, counts = labels.unique(return_counts=True)
    ok = counts >= min_count
    ids = ids[ok][counts[ok].argsort(descending=True)][:max_classes]   # most frequent classes
    if len(ids) < 5:
        return {}
    means, var = [], []
    for c in ids.tolist():
        xc = x[labels == c]
        means.append(xc.mean(0))
        var.append((xc - xc.mean(0)).pow(2).sum(1).mean())
    m = torch.stack(means)
    sig2 = torch.stack(var)
    off = ~torch.eye(len(m), dtype=torch.bool, device=m.device)

    d2 = torch.cdist(m, m).pow(2).clamp_min(1e-12)
    cdnv = ((sig2[:, None] + sig2[None, :]) / (2 * d2) / d2)[off]       # the extra power

    mc = m - m.mean(0)
    nrm = mc.norm(dim=1).clamp_min(1e-12)
    u = mc / nrm[:, None]
    cos = (u @ u.T)[off]
    uhat = m / m.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gnc2 = -torch.cdist(uhat, uhat).clamp_min(1e-6).log()[off].mean()

    w = w_rows[ids].float()
    wu = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dual = (wu * u).sum(1)

    logits = x @ w.T                                                    # no bias in these heads
    agree = (logits.argmax(1) == torch.cdist(x, m).argmin(1)).float().mean()

    cv = lambda t: float(t.std() / t.mean().abs().clamp_min(1e-12))
    return {"nc1_cdnv": float(cdnv.mean()),
            "nc2_equinorm_cv_log": cv(nrm.log()),
            "nc2_equiangle_cv": cv(cos),
            "nc2_cos_mean": float(cos.mean()), "nc2_cos_target": -1.0 / (len(m) - 1),
            "gnc2_uniformity": float(gnc2),
            "nc3_dual": float(dual.mean()), "unc3_dual_cv": cv(dual),
            "nc4_agree": float(agree), "n_classes": len(m)}


def _capture(model, config, block, tokens, batch_size, dev, vocab, keep_rows):
    """Class means and within-class variances accumulated over EVERY token, streamed into a
    (vocab, d) buffer rather than by retaining sample rows -- retaining rows caps the number
    of usable classes, and the regime that matters here is C >> d+1.

    Also returns a bounded random-ish subset of rows (the first keep_rows) for the metrics
    that need individual samples, plus the anisotropy/cone inputs.
    """
    fam = _fam(config)
    norm: Any = get_final_layernorm(model, config)
    mlp: Any = _getattr_path(model, f"{fam.blocks}.{block}.mlp")
    buf: dict = {}
    kept: dict[str, list] = {"bfn": [], "afn": [], "mlp": [], "lab": []}

    def flat(t):
        t = t[0] if isinstance(t, tuple) else t
        return t.detach().float().reshape(-1, t.shape[-1])

    cur: dict = {}

    def norm_hook(_m, inp, out):        # returns None: a non-None return replaces the output
        cur["bfn"], cur["afn"] = flat(inp[0]), flat(out)

    def mlp_hook(_m, _inp, out):
        cur["mlp"] = flat(out)

    handles = [norm.register_forward_hook(norm_hook), mlp.register_forward_hook(mlp_hook)]
    n_kept = 0
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch_size):
            batch = tokens[i:i + batch_size].to(dev)
            model.forward(batch)
            lab = batch[:, 1:].reshape(-1)                       # next-token identity
            for k in ("bfn", "afn", "mlp"):
                cur[k] = cur[k].reshape(batch.shape[0], -1, cur[k].shape[-1])[:, :-1]
                cur[k] = cur[k].reshape(-1, cur[k].shape[-1])
            h = cur["afn"]
            if not buf:
                buf.update(s=torch.zeros(vocab, h.shape[1], dtype=torch.float32, device=dev),
                           sq=torch.zeros(vocab, dtype=torch.float64, device=dev),
                           n=torch.zeros(vocab, dtype=torch.float64, device=dev))
            buf["s"].index_add_(0, lab, h)
            buf["sq"].index_add_(0, lab, h.pow(2).sum(1).double())
            buf["n"].index_add_(0, lab, torch.ones_like(lab, dtype=torch.float64))
            if n_kept < keep_rows:
                for k in ("bfn", "afn", "mlp"):
                    kept[k].append(cur[k][:keep_rows - n_kept].cpu())
                kept["lab"].append(lab[:keep_rows - n_kept].cpu())
                n_kept += min(h.shape[0], keep_rows - n_kept)
    for hd in handles:
        hd.remove()
    acts = {k: torch.cat(v).to(dev) for k, v in kept.items() if k != "lab"}
    return acts, torch.cat(kept["lab"]).to(dev), buf


def main(out_path: str = "data/results/collapse_geometry.pt",
         model_name: str = "allenai/OLMo-2-0425-1B", max_checkpoints: int = 20,
         spacing: str = "log", num_windows: int = 4096, batch_size: int = 8,
         max_keep: int = 40_000, keep_cached: bool = True, heavy_metrics: bool = False) -> None:
    """num_windows x 512 tokens per checkpoint. 4096 windows = 2.1M tokens, which yields
    ~7-8k classes with >=30 occurrences -- past d+1, the regime GNC2 exists for, while still
    cheap enough to run at every checkpoint. heavy_metrics adds NC2/GNC2/NC3/UNC3/NC4, which
    need retained sample rows and pairwise work; NC1 (their Figure 5 metric) does not."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_model_config(model_name)
    short = model_name.split("/")[-1]
    out = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
    res = out.setdefault(short, {})
    tokens = torch.load(MIXES[config.family], weights_only=True)[:num_windows]

    for step, revision, repo in get_checkpoint_schedule(config, max_checkpoints, spacing):
        if step in res or step == 0:
            continue
        loaded = load_model(config, repo, revision)
        block = get_num_layers(loaded, config) - 1
        model = torch.nn.Module.to(loaded, dev).eval()
        head = get_head_module(model, config).weight.detach().float().to(dev)
        gamma = getattr(get_final_layernorm(model, config), "weight", None)
        w_eff = head - head.mean(0, keepdim=True)
        if gamma is not None:
            w_eff = w_eff * gamma.float().to(dev)

        counts = torch.bincount(tokens.reshape(-1), minlength=head.shape[0]).float().to(dev)
        p = (counts + 1) / (counts + 1).sum()                    # unigram, add-one smoothed
        log_f = p.log() - p.log().mean()                         # = v_freq of the paper
        d_freq, r2 = _freq_direction(w_eff, log_f)
        k = max(1, round(0.01 * head.shape[1]))
        v0 = head_subspace(head, gamma, k)

        acts, labels, buf = _capture(model, config, block, tokens, batch_size, dev,
                                     head.shape[0], max_keep)
        share = lambda x, v: float((x @ v).pow(2).mean() / x.pow(2).sum(1).mean())
        r = {"tokens": get_token_count(model_name, step), "freq_r2": r2,
             "d_freq_null_share": float((v0 @ d_freq).norm()),
             "head_anisotropy": _anisotropy(head[counts > 0][:20_000]),
             "head_cone": _cone(head[counts > 0][:20_000]),
             "freq_corr_head_norm": float(torch.corrcoef(
                 torch.stack([head.norm(dim=1), p.log()]))[0, 1]),
             **{f"{leaf}_{name}": fn(acts[leaf])
                for leaf in ("bfn", "afn", "mlp") for name, fn in
                (("anisotropy", _anisotropy), ("cone", _cone))},
             **{f"{leaf}_freq_share": share(acts[leaf], d_freq) for leaf in ("bfn", "afn", "mlp")},
             **{f"afn_{k2}": v for k2, v in _nc1_cdnv(buf).items()},
             **({f"afn_{k2}": v for k2, v in _collapse(acts["afn"], labels, head).items()}
                if heavy_metrics else {})}
        res[step] = r
        del model, loaded, acts, buf
        torch.cuda.empty_cache()
        if not keep_cached and config.family != "nanochat":   # 7B snapshots are too big to hoard
            delete_cached_revision(repo, revision)
        torch.save(out, out_path)
        print(f"{short} step{step} ({r['tokens']:.2e} tok): freq_r2={r2:.3f} "
              f"stream_freq={r['afn_freq_share']:.4f} mlp_freq={r['mlp_freq_share']:.4f} "
              f"aniso(afn)={r['afn_anisotropy']:.3f} cone(afn)={r['afn_cone']:.3f} "
              f"log-CDNV={r.get('afn_nc1_log_cdnv', float('nan')):.3f} "
              f"over {r.get('afn_nc1_n_classes', 0)} classes", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
