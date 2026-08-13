"""Comprehensive final-block analysis. Padded DOCUMENT-FINAL tokens (symmetric categories), py1b,
pile / fineweb / olmo_mix. For each dataset and token type (period / newline / other) it prints:
  (A) projection of entering / attn / mlp / bfn onto channel 1668, + verdict + rank_in/bfn/afn + H
  (B) the same projection onto blk3's largest write direction v1
  (C) the final block's WRITE dominant direction (top eigvec of attn+mlp, mean-oriented):
      participation ratio, cos with the bfn direction / channel 1668 / v1, and its final-LayerNorm
      gain + unembedding treatment (logit-norm and top-1/bias share).
Cross-dataset: |cos| of the write direction between datasets, per type.
rank_in/bfn/afn = EV-RankMe (exp H(cov eigenvalues), the repo's 'rankme') entering the last block /
after it / after the final norm. verdict: MLP reinforces (sign(mlp)=sign(entering)) or cancels.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM

CACHES = {"pile": "data/filtered_texts/pile_deduped_eleutherai_16384_s42.json",
          "fineweb": "data/filtered_texts/fineweb_5000.json",
          "olmo_mix": "data/filtered_texts/olmo_mix_16384_s42.json"}


def ev_rankme(X):
    Xc = X.double() - X.double().mean(0, keepdim=True)
    lam = torch.linalg.svdvals(Xc) ** 2
    p = lam / lam.sum()
    return float(torch.exp(-(p * torch.log(torch.clamp(p, min=1e-7))).sum()))


def top_dir(X, orient):
    Xc = X.double() - X.double().mean(0, keepdim=True)
    u = torch.linalg.svd(Xc, full_matrices=False)[2][0]
    return u if (u @ orient.double()) >= 0 else -u


def blk3_write_dir(m, tok, cache, blk3, d, dev, seq_len=512, cov_tokens=120000, batch=24):
    ids, buf = [], []
    for t in json.load(open(cache)):
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        while len(buf) >= seq_len:
            ids.append(buf[:seq_len]); buf = buf[seq_len:]
        if len(ids) * seq_len >= cov_tokens:
            break
    ids = torch.tensor(ids, dtype=torch.long).to(dev)
    cap = {}
    h = m.gpt_neox.layers[blk3].mlp.register_forward_hook(
        lambda mod, i, o: cap.__setitem__("w", o[0] if isinstance(o, tuple) else o))
    S1 = torch.zeros(d, dtype=torch.float64); S2 = torch.zeros(d, d, dtype=torch.float64); n = 0
    with torch.no_grad():
        for b0 in range(0, ids.shape[0], batch):
            m(ids[b0:b0 + batch]); w = cap["w"].float().reshape(-1, d).double()
            S1 += w.sum(0).cpu(); S2 += (w.T @ w).cpu(); n += w.shape[0]
    h.remove()
    mean = S1 / n
    v1 = torch.linalg.eigh(S2 / n - torch.outer(mean, mean))[1][:, -1]
    return (v1 if mean @ v1 >= 0 else -v1).float()


def padded_batches(texts, tok, seq_len, batch, max_docs):
    enc = [tok(t, add_special_tokens=False).input_ids[:seq_len] for t in texts[:max_docs]]
    enc = [e for e in enc if len(e) >= 16]
    enc.sort(key=len)
    pad = tok.eos_token_id or 0
    for i in range(0, len(enc), batch):
        ch = enc[i:i + batch]; L = max(len(e) for e in ch)
        ids = torch.full((len(ch), L), pad, dtype=torch.long)
        mask = torch.zeros((len(ch), L), dtype=torch.long); lengths = []
        for j, e in enumerate(ch):
            ids[j, :len(e)] = torch.tensor(e); mask[j, :len(e)] = 1; lengths.append(len(e))
        yield ids, mask, torch.tensor(lengths)


def classify_last(tok, tid):
    s = tok.decode([int(tid)])
    return "newline" if "\n" in s else ("period" if s.strip() == "." else "other")


def collect(m, tok, cache, d, dev, cap_n=4000, max_docs=8000, batch=24, seq_len=512):
    cap = {}
    L = m.config.num_hidden_layers
    last = m.gpt_neox.layers[L - 1]
    uw = lambda o: o[0] if isinstance(o, tuple) else o
    hs = [last.register_forward_pre_hook(lambda mod, a: cap.__setitem__("hin", a[0])),
          last.attention.register_forward_hook(lambda mod, i, o: cap.__setitem__("attn", uw(o))),
          last.mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__("mlp", uw(o))),
          m.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("bfn", a[0])),
          m.gpt_neox.final_layer_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("afn", o))]
    keys = ("hin", "attn", "mlp", "bfn", "afn")
    bins = {k: {q: [] for q in keys} | {"H": []} for k in ("period", "newline", "other")}
    with torch.no_grad():
        for ids, mask, lengths in padded_batches(json.load(open(cache)), tok, seq_len, batch, max_docs):
            ids, mask, lengths = ids.to(dev), mask.to(dev), lengths.to(dev)
            out = m(ids, attention_mask=mask)
            r = torch.arange(ids.shape[0], device=dev); idx = lengths - 1
            vals = {q: cap[q][r, idx].float() for q in keys}
            lp = F.log_softmax(out.logits[r, idx].float(), dim=-1)
            H = -(lp.exp() * lp).sum(-1)
            last_ids = ids[r, idx]
            for j in range(ids.shape[0]):
                k = classify_last(tok, last_ids[j])
                if len(bins[k]["hin"]) >= cap_n:
                    continue
                for q in keys:
                    bins[k][q].append(vals[q][j].cpu())
                bins[k]["H"].append(float(H[j]))
    for h in hs:
        h.remove()
    return bins


def main(model="EleutherAI/pythia-1b-deduped", revision="step143000", ma=1668):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model)
    m = GPTNeoXForCausalLM.from_pretrained(model, revision=revision,
                                           torch_dtype=torch.float16).to(dev).eval()
    d = m.config.hidden_size
    gamma = m.gpt_neox.final_layer_norm.weight.detach().double().cpu()
    ee = torch.zeros(d, dtype=torch.float64); ee[ma] = 1.0
    v1 = blk3_write_dir(m, tok, CACHES["pile"], 3, d, dev).double()

    def numlast(pat):
        fs = glob.glob(pat)
        return max(fs, key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    Vh = torch.load(numlast("data/results/unembedding_svd/pythia-1b-deduped/step*.pt"),
                    map_location="cpu", weights_only=False).double()
    spec = torch.load("data/results/unembedding_spectra.pt", map_location="cpu", weights_only=False)
    S = spec["pythia-1b-deduped"][max(spec["pythia-1b-deduped"])].double()
    k = min(Vh.shape[0], S.shape[0])

    def unembed(dv):
        proj = Vh[:k] @ dv; en = (S[:k] * proj) ** 2
        return float(en.sum() ** 0.5), float(en[0] / en.sum())

    def verdict(en, ml):
        return "reinforce" if en * ml > 0 else ("cancel" if en * ml < 0 else "~0")

    wdirs = {}
    for name, cache in CACHES.items():
        bins = collect(m, tok, cache, d, dev)
        types = [t for t in ("period", "newline", "other") if len(bins[t]["hin"]) >= 50]
        stk = {t: {q: torch.stack(bins[t][q]) for q in ("hin", "attn", "mlp", "bfn", "afn")} for t in types}
        rk = {t: (ev_rankme(stk[t]["hin"]), ev_rankme(stk[t]["bfn"]), ev_rankme(stk[t]["afn"])) for t in types}
        Hm = {t: sum(bins[t]["H"]) / len(bins[t]["H"]) for t in types}
        wr = {t: top_dir(stk[t]["attn"] + stk[t]["mlp"], (stk[t]["attn"] + stk[t]["mlp"]).mean(0)) for t in types}
        bf = {t: top_dir(stk[t]["bfn"], stk[t]["bfn"].mean(0)) for t in types}
        wdirs[name] = wr

        def proj_table(dsrc, title):
            print(f"\n--- {name}: {title} ---")
            print(f"{'type':8s} {'N':>5s} {'entering':>8s} {'attn':>7s} {'mlp':>8s} {'bfn':>8s} "
                  f"{'verdict':>10s} {'rank_in':>7s} {'rank_bfn':>8s} {'rank_afn':>8s} {'H':>5s}")
            for t in types:
                dv = dsrc[t] if isinstance(dsrc, dict) else dsrc     # per-type direction, or one for all
                en = float((stk[t]["hin"].double() @ dv).mean())
                at = float((stk[t]["attn"].double() @ dv).mean())
                ml = float((stk[t]["mlp"].double() @ dv).mean())
                print(f"{t:8s} {stk[t]['bfn'].shape[0]:5d} {en:8.1f} {at:7.1f} {ml:8.1f} {en+at+ml:8.1f} "
                      f"{verdict(en, ml):>10s} {rk[t][0]:7.1f} {rk[t][1]:8.1f} {rk[t][2]:8.1f} {Hm[t]:5.2f}")

        proj_table(ee, "projection onto CHANNEL 1668")
        proj_table(v1, "projection onto BLK3 WRITE DIR v1")
        proj_table(wr, "projection onto each type's OWN FINAL-BLOCK WRITE direction")
        proj_table(bf, "projection onto each type's OWN BFN COLLAPSE direction")

        print(f"\n--- {name}: WRITE-direction geometry (final-norm gain + unembedding) ---")
        print(f"{'type':8s} {'partic':>7s} {'cos(wr,bfn)':>11s} {'cos(wr,e1668)':>13s} "
              f"{'cos(wr,v1)':>10s} {'wr_lngain':>9s} {'wr_top1':>8s} {'||W_U wr||':>10s}")
        for t in types:
            wn, wt1 = unembed(wr[t])
            print(f"{t:8s} {float(1.0 / (wr[t] ** 4).sum()):7.1f} {float(wr[t] @ bf[t]):11.3f} "
                  f"{float(wr[t] @ ee):13.3f} {float(wr[t] @ v1):10.3f} "
                  f"{float((wr[t] ** 2 * gamma).sum()):9.3f} {wt1:8.3f} {wn:10.2f}")

    print("\n===== cross-dataset WRITE-direction agreement (|cos|, per type) =====")
    print(f"{'type':8s} {'pile~fineweb':>12s} {'pile~olmo':>10s} {'fineweb~olmo':>12s}")
    for t in ("period", "newline", "other"):
        def c(a, b):
            da, db = wdirs.get(a, {}).get(t), wdirs.get(b, {}).get(t)
            return f"{abs(float(da @ db)):.3f}" if (da is not None and db is not None) else "  --"
        print(f"{t:8s} {c('pile','fineweb'):>12s} {c('pile','olmo_mix'):>10s} {c('fineweb','olmo_mix'):>12s}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
