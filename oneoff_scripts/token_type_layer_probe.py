"""Layer-discriminated, token-type-resolved MA accounting across Pythia's FINAL block, for
several datasets, projected onto BOTH a standard-basis channel AND blk3's largest write direction.

For the last block L-1 (parallel: bfn = h_in + attn_out + mlp_out) we report, per token type and
per dataset, the projection of each stage onto:
  ch    = standard-basis coordinate `ma_channel` (the bfn-dominant axis), and
  v1    = the top eigenvector of blk3.mlp's CENTERED write covariance (showcase's sink direction),
so 'reinforce' (final MLP moves the residual further along the incoming direction) vs 'cancel'
can be read for the model's actual write direction, not an arbitrary axis. Columns also include
bfn EV-RankMe (= the sweeps' 'rankme') per type and the mean next-token logit entropy H per type
(to test whether the collapse tracks next-token uncertainty). Token types: pos0 / first_newline /
other_newline / period / other. Packed stream, one checkpoint. py1b runs on every dataset's text.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM

CACHES = {
    "pile":     "data/filtered_texts/pile_deduped_eleutherai_16384_s42.json",
    "fineweb":  "data/filtered_texts/fineweb_5000.json",
    "olmo_mix": "data/filtered_texts/olmo_mix_16384_s42.json",
}


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
    cls = torch.zeros(V, dtype=torch.long)      # 1 newline, 2 period, 0 other
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


def ev_rankme(X):
    Xc = X.double() - X.double().mean(0, keepdim=True)
    lam = torch.linalg.svdvals(Xc) ** 2
    p = lam / lam.sum()
    return float(torch.exp(-(p * torch.log(torch.clamp(p, min=1e-7))).sum()))


def blk3_write_dir(m, ids, blk3, d, dev, batch, seq_len, cov_tokens):
    """Top eigenvector of blk3.mlp's centered write covariance (oriented so its mean projects +)."""
    cap = {}
    h = m.gpt_neox.layers[blk3].mlp.register_forward_hook(
        lambda mod, i, o: cap.__setitem__("w", o[0] if isinstance(o, tuple) else o))
    S1 = torch.zeros(d, dtype=torch.float64); S2 = torch.zeros(d, d, dtype=torch.float64); n = 0
    n_ch = min(cov_tokens // seq_len, ids.shape[0])
    with torch.no_grad():
        for b0 in range(0, n_ch, batch):
            m(ids[b0:b0 + batch])
            w = cap["w"].float().reshape(-1, d).double()
            S1 += w.sum(0).cpu(); S2 += (w.T @ w).cpu(); n += w.shape[0]
    h.remove()
    mean = S1 / n
    cov = S2 / n - torch.outer(mean, mean)
    v1 = torch.linalg.eigh(cov)[1][:, -1]           # top eigenvector (eigh ascending)
    if mean @ v1 < 0:
        v1 = -v1
    return v1.to(dev).float()


def run_dataset(m, tok, vcls_full, name, cache, ma_channel, blk3, d, dev, n_chunks, n_cap,
                batch, seq_len, cov_tokens):
    L = m.config.num_hidden_layers
    ids = pack(json.load(open(cache)), tok, seq_len, n_chunks).to(dev)
    vcls = vcls_full[: int(ids.max().item()) + 1].to(dev)
    v1 = blk3_write_dir(m, ids, blk3, d, dev, batch, seq_len, cov_tokens)
    e = torch.zeros(d, device=dev); e[ma_channel] = 1.0
    cos = float(v1 @ e)                              # alignment of blk3 dir with the ch1668 axis
    pr = float(1.0 / (v1.double() ** 4).sum())       # participation ratio of v1 (1=axis, d=spread)

    cap = {}
    last = m.gpt_neox.layers[L - 1]
    uw = lambda o: o[0] if isinstance(o, tuple) else o
    last.register_forward_pre_hook(lambda mod, a: cap.__setitem__("h_in", a[0]))
    last.attention.register_forward_hook(lambda mod, i, o: cap.__setitem__("a", uw(o)))
    last.mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__("mlp", uw(o)))
    m.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("bfn", a[0]))

    codes = {4: "pos0", 3: "first_nl", 2: "other_nl", 1: "period", 0: "other"}
    bfn_bins = {k: [] for k in codes.values()}
    hin_bins = {k: [] for k in codes.values()}      # residual ENTERING the last block (causation)
    sc_bins = {k: [] for k in codes.values()}       # cols: ch(hin,a,mlp) v1(hin,a,mlp) H
    full = set()
    with torch.no_grad():
        for b0 in range(0, ids.shape[0], batch):
            if len(full) == len(codes):
                break
            chunk = ids[b0:b0 + batch]
            out = m(chunk)
            B, T, dd = cap["bfn"].shape
            lp = F.log_softmax(out.logits.float(), dim=-1)
            H = (-(lp.exp() * lp).sum(-1)).reshape(-1)
            nl = (vcls[chunk] == 1); pd = (vcls[chunk] == 2)
            first = nl & (nl.cumsum(1) == 1)
            key = torch.zeros_like(chunk)
            key[pd] = 1; key[nl & ~first] = 2; key[first] = 3; key[:, 0] = 4
            key = key.reshape(-1)
            hin = cap["h_in"].float().reshape(-1, dd); a = cap["a"].float().reshape(-1, dd)
            ml = cap["mlp"].float().reshape(-1, dd); bfn = cap["bfn"].float().reshape(-1, dd)
            sc = torch.stack([hin[:, ma_channel], a[:, ma_channel], ml[:, ma_channel],
                              hin @ v1, a @ v1, ml @ v1, H], dim=1)
            for code, nm in codes.items():
                if nm in full:
                    continue
                need = n_cap - sum(x.shape[0] for x in bfn_bins[nm])
                sel = (key == code)
                if sel.any() and need > 0:
                    bfn_bins[nm].append(bfn[sel][:need].cpu())
                    hin_bins[nm].append(hin[sel][:need].cpu())
                    sc_bins[nm].append(sc[sel][:need].cpu())
                if sum(x.shape[0] for x in bfn_bins[nm]) >= n_cap:
                    full.add(nm)

    print(f"\n===== {name}  (py1b) =====")
    print(f"blk3 write-dir v1: cos(v1, e{ma_channel})={cos:+.3f}  participation_ratio={pr:.1f}/{d}")
    print(f"{'type':9s} {'N':>5s} {'rank_in':>7s} {'rank_bfn':>8s} {'H_next':>6s} | "
          f"ch{ma_channel}[hin/attn/mlp] | v1_blk3[hin/attn/mlp]")
    period_bfn = None
    for nm in codes.values():
        bfn = torch.cat(bfn_bins[nm])[:n_cap]; hin = torch.cat(hin_bins[nm])[:n_cap]
        sc = torch.cat(sc_bins[nm])[:n_cap]
        rk_in, rk_bfn, Hm = ev_rankme(hin), ev_rankme(bfn), float(sc[:, 6].mean())
        chh, cha, chm = (float(sc[:, i].mean()) for i in (0, 1, 2))
        vh, va, vm = (float(sc[:, i].mean()) for i in (3, 4, 5))
        print(f"{nm:9s} {bfn.shape[0]:5d} {rk_in:7.1f} {rk_bfn:8.1f} {Hm:6.2f} | "
              f"{chh:6.1f}/{cha:5.1f}/{chm:6.1f} | {vh:8.1f}/{va:5.1f}/{vm:9.1f}", flush=True)
        if nm == "period":
            period_bfn = bfn
    if period_bfn is not None:                       # what direction do period tokens collapse in?
        Xc = period_bfn.double() - period_bfn.double().mean(0, keepdim=True)
        u = torch.linalg.svd(Xc, full_matrices=False)[2][0]
        ee = torch.zeros(d, dtype=torch.float64); ee[ma_channel] = 1.0
        print(f"  period bfn top-eigvec: cos(e{ma_channel})={float(u @ ee):+.3f}  "
              f"cos(v1_blk3)={float(u @ v1.double().cpu()):+.3f}  "
              f"participation={float(1.0 / (u ** 4).sum()):.1f}/{d}", flush=True)


def main(model="EleutherAI/pythia-1b-deduped", revision="step143000", ma_channel=1668,
         blk3=3, n_chunks=6000, n_cap=4000, batch=24, seq_len=512, cov_tokens=150000):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model)
    m = GPTNeoXForCausalLM.from_pretrained(model, revision=revision,
                                           torch_dtype=torch.float16).to(dev).eval()
    d = m.config.hidden_size
    vcls_full = classify_vocab(tok, 50400)          # covers pythia's full id range (~50277)
    for name, cache in CACHES.items():
        run_dataset(m, tok, vcls_full, name, cache, ma_channel, blk3, d, dev,
                    n_chunks, n_cap, batch, seq_len, cov_tokens)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
