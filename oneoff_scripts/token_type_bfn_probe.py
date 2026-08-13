"""Token-type-resolved before_final_norm geometry (Pythia).

Motivation: the padded (last-token) bfn RankMe collapse pools all document-final tokens, and
the packed 'all tokens' pools everything — but 'all tokens' includes the attention-sink first
token and the newline/delimiter spike tokens. This probe splits a packed pile stream by token
type and measures, per subset, the centered RankMe, top-eigenvalue share, the massive-activation
channel's mean/std, and the mean per-token magnitude. If the collapse is driven by sink/newline
tokens, those subsets collapse (low rank, huge MA channel) while ordinary tokens do not.

Subsets (disjoint): pos0 (first token of each 512-chunk = attention sink) | newline (decoded
contains '\\n') | period (decoded is '.') | other. One checkpoint, no training sweep.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM


def pack(texts, tok, seq_len, n_chunks):
    buf, chunks = [], []
    for t in texts:
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            chunks.append(buf[:seq_len]); buf = buf[seq_len:]
            if len(chunks) >= n_chunks:
                return torch.tensor(chunks, dtype=torch.long)
    return torch.tensor(chunks, dtype=torch.long)


def classify_vocab(tok, V):
    """id -> 1 (newline) / 2 (period) / 0 (other), by decoded string. V must cover every id
    that can appear (pythia emits ids beyond tok.vocab_size, so size by the packed max id)."""
    cls = torch.zeros(V, dtype=torch.long)
    for i in range(V):
        try:
            s = tok.decode([i])
        except Exception:
            continue
        if "\n" in s:
            cls[i] = 1
        elif s.strip() == ".":
            cls[i] = 2
    return cls


def stats(X):
    """X: (N,d) fp32. Returns (rankme, true_rankme, top1_eigval_share, avg_magnitude).
    rankme = exp(entropy of EIGENVALUES) — matches scripts/compute_metrics.rankme_metrics['rankme']
    (the value the sweeps store); true_rankme = exp(entropy of singular values) = Garrido."""
    Xc = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xc.double())          # singular values of centered data
    lam = s * s                                      # covariance eigenvalues
    ent = lambda p: float(-(p * torch.log(torch.clamp(p, min=1e-7))).sum())
    rankme = float(torch.exp(torch.tensor(ent(lam / lam.sum()))))
    true_rankme = float(torch.exp(torch.tensor(ent(s / s.sum()))))
    top1 = float(lam[0] / lam.sum())
    mag = float(X.norm(dim=1).mean())
    return rankme, true_rankme, top1, mag


def main(model="EleutherAI/pythia-1b-deduped", revision="step143000",
         cache="data/filtered_texts/pile_deduped_eleutherai_16384_s42.json",
         ma_channel=1668, n_chunks=5000, n_cap=4000, batch=64, seq_len=512):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model)
    texts = json.load(open(cache))
    ids = pack(texts, tok, seq_len, n_chunks).to(dev)
    print(f"packed {ids.shape[0]} chunks x {seq_len} = {ids.numel()} tokens", flush=True)
    vcls = classify_vocab(tok, int(ids.max().item()) + 1).to(dev)

    m = GPTNeoXForCausalLM.from_pretrained(model, revision=revision,
                                           torch_dtype=torch.float16).to(dev).eval()
    captured = {}
    m.gpt_neox.final_layer_norm.register_forward_pre_hook(
        lambda mod, args: captured.__setitem__("bfn", args[0]))

    names = ["pos0", "newline", "period", "other"]
    bins = {k: [] for k in names}                    # collected bfn rows per subset
    full = set()
    with torch.no_grad():
        for b0 in range(0, ids.shape[0], batch):
            if len(full) == len(names):
                break
            chunk = ids[b0:b0 + batch]
            m(chunk)
            bfn = captured["bfn"].float()            # (B,T,d)
            B, T, d = bfn.shape
            ttype = vcls[chunk]                      # (B,T): 0 other,1 nl,2 period
            pos0 = torch.zeros_like(ttype); pos0[:, 0] = 1
            key = torch.where(pos0 == 1, torch.full_like(ttype, 3), ttype)  # 3=pos0
            flat = bfn.reshape(B * T, d); key = key.reshape(B * T)
            for code, name in ((3, "pos0"), (1, "newline"), (2, "period"), (0, "other")):
                if name in full:
                    continue
                need = n_cap - sum(x.shape[0] for x in bins[name])
                rows = flat[key == code]
                if rows.numel() and need > 0:
                    bins[name].append(rows[:need].cpu())
                if sum(x.shape[0] for x in bins[name]) >= n_cap:
                    full.add(name)

    print(f"\n{'subset':10s} {'N':>7s} {'RankMe':>8s} {'trueRkMe':>9s} {'top1':>6s} "
          f"{'chan%d_mean' % ma_channel:>12s} {'chan_std':>9s} {'avg_mag':>8s}")
    for name in names:
        X = torch.cat(bins[name])[:n_cap]
        rk, trk, top1, mag = stats(X)
        cm = float(X[:, ma_channel].mean()); cs = float(X[:, ma_channel].std())
        print(f"{name:10s} {X.shape[0]:7d} {rk:8.1f} {trk:9.1f} {top1:6.3f} "
              f"{cm:12.1f} {cs:9.1f} {mag:8.2f}", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
