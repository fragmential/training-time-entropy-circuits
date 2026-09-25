"""Is the bfn collapse cosmetic? Padded (document-final) probe, py1b, pile + fineweb.

Per last-token TYPE (period / newline / other — symmetric, all document-final):
  rank_in / rank_bfn / rank_afn  = EV RankMe entering the last block / after it / after the norm
  H_next                         = mean next-token logit entropy
  sig1668                        = fraction of tokens with bfn channel-1668 > 0 (is it 1-signed?)
  sigma_frac_1668                = mean share of the per-token LayerNorm sigma^2 carried by ch1668
  bfn_top_dir cos(e1668)         = alignment of this type's bfn collapse direction with the axis
Then a GEOMETRY section for {e1668, blk3 write-dir v1, pile-period dir, fineweb-period dir}:
  ln_gain = sum_i dir_i^2 * gamma_i           (effective final-norm gain along the direction)
  unembed: cos(dir, Vh[0]) and logit-energy share in the top-1 (bias) unembedding component,
           via the STORED unembedding SVD (Vh, S) — ||W_U dir||^2 = sum (S * (Vh@dir))^2.
No weights are downloaded (model from the local HF cache; unembedding from stored SVD).
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
          "fineweb": "data/filtered_texts/fineweb_5000.json"}


def ev_rankme(X):
    Xc = X.double() - X.double().mean(0, keepdim=True)
    lam = torch.linalg.svdvals(Xc) ** 2
    p = lam / lam.sum()
    return float(torch.exp(-(p * torch.log(torch.clamp(p, min=1e-7))).sum()))


def top_dir(X):
    Xc = X.double() - X.double().mean(0, keepdim=True)
    return torch.linalg.svd(Xc, full_matrices=False)[2][0]


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
    if "\n" in s:
        return "newline"
    if s.strip() == ".":
        return "period"
    return "other"


def run_dataset(m, tok, name, cache, gamma, d, dev, ma=1668, max_docs=8000, batch=24, seq_len=512):
    cap = {}
    L = m.config.num_hidden_layers
    last = m.gpt_neox.layers[L - 1]
    uw = lambda o: o[0] if isinstance(o, tuple) else o
    hs = [last.register_forward_pre_hook(lambda mod, a: cap.__setitem__("hin", a[0])),
          last.mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__("mlp", uw(o))),
          m.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("bfn", a[0])),
          m.gpt_neox.final_layer_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("afn", o))]
    bins = {k: {"hin": [], "mlp": [], "bfn": [], "afn": [], "H": [], "sfrac": []}
            for k in ("period", "newline", "other")}
    with torch.no_grad():
        for ids, mask, lengths in padded_batches(json.load(open(cache)), tok, seq_len, batch, max_docs):
            ids, mask, lengths = ids.to(dev), mask.to(dev), lengths.to(dev)
            out = m(ids, attention_mask=mask)
            idx = (lengths - 1)
            r = torch.arange(ids.shape[0], device=dev)
            hin, mlp = cap["hin"][r, idx].float(), cap["mlp"][r, idx].float()
            bfn, afn = cap["bfn"][r, idx].float(), cap["afn"][r, idx].float()
            lp = F.log_softmax(out.logits[r, idx].float(), dim=-1)
            H = -(lp.exp() * lp).sum(-1)
            bc = bfn - bfn.mean(1, keepdim=True)                       # centered over channels
            sfrac = bc[:, ma] ** 2 / (bc ** 2).sum(1)                  # ch1668 share of per-token sigma^2
            lasttok = ids[r, idx]
            for j in range(ids.shape[0]):
                k = classify_last(tok, lasttok[j])
                b = bins[k]
                b["hin"].append(hin[j].cpu()); b["mlp"].append(mlp[j].cpu())
                b["bfn"].append(bfn[j].cpu()); b["afn"].append(afn[j].cpu())
                b["H"].append(float(H[j])); b["sfrac"].append(float(sfrac[j]))
    for h in hs:
        h.remove()
    print(f"\n===== {name} (py1b, padded document-final) =====")
    print(f"{'type':8s} {'N':>5s} {'rank_in':>7s} {'rank_bfn':>8s} {'rank_afn':>8s} "
          f"{'H_next':>6s} {'sig1668+':>8s} {'sigfrac1668':>11s} {'bfn_dir.e1668':>13s}")
    dirs = {}
    for k in ("period", "newline", "other"):
        b = bins[k]
        if len(b["bfn"]) < 50:
            print(f"{k:8s} {len(b['bfn']):5d}  (too few)"); continue
        HIN, BFN, AFN = (torch.stack(b[x]) for x in ("hin", "bfn", "afn"))
        u = top_dir(BFN); dirs[k] = u
        ee = torch.zeros(d, dtype=torch.float64); ee[ma] = 1.0
        print(f"{k:8s} {BFN.shape[0]:5d} {ev_rankme(HIN):7.1f} {ev_rankme(BFN):8.1f} "
              f"{ev_rankme(AFN):8.1f} {sum(b['H'])/len(b['H']):6.2f} "
              f"{sum(1 for j in range(BFN.shape[0]) if BFN[j, ma] > 0)/BFN.shape[0]:8.2f} "
              f"{sum(b['sfrac'])/len(b['sfrac']):11.3f} {float(u @ ee):13.3f}")
    return dirs.get("period")


def main(model="EleutherAI/pythia-1b-deduped", revision="step143000", ma=1668):
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model)
    m = GPTNeoXForCausalLM.from_pretrained(model, revision=revision,
                                           torch_dtype=torch.float16).to(dev).eval()
    d = m.config.hidden_size
    gamma = m.gpt_neox.final_layer_norm.weight.detach().double().cpu()
    beta = m.gpt_neox.final_layer_norm.bias.detach().double().cpu()

    v1 = blk3_write_dir(m, tok, CACHES["pile"], 3, d, dev)
    per_pile = run_dataset(m, tok, "pile", CACHES["pile"], gamma, d, dev, ma)
    per_fw = run_dataset(m, tok, "fineweb", CACHES["fineweb"], gamma, d, dev, ma)

    # unembedding SVD (stored — never recomputed from weights)
    def numlast(pat):
        fs = glob.glob(pat)
        return max(fs, key=lambda p: int("".join(filter(str.isdigit, os.path.basename(p)))))
    Vh = torch.load(numlast("data/results/unembedding_svd/pythia-1b-deduped/step*.pt"),
                    map_location="cpu", weights_only=False).double()
    spec = torch.load("data/results/unembedding_spectra.pt", map_location="cpu", weights_only=False)
    S = spec["pythia-1b-deduped"][max(spec["pythia-1b-deduped"])].double()

    ee = torch.zeros(d, dtype=torch.float64); ee[ma] = 1.0
    cand = {"e1668": ee, "blk3_v1": v1.double(),
            "pile_period": per_pile.double() if per_pile is not None else None,
            "fineweb_period": per_fw.double() if per_fw is not None else None}
    print(f"\n===== geometry =====\ngamma[{ma}]={float(gamma[ma]):+.3f} "
          f"(|gamma| percentile {float((gamma.abs() < abs(gamma[ma])).float().mean()) * 100:.0f})  "
          f"beta[{ma}]={float(beta[ma]):+.3f}")
    if per_pile is not None and per_fw is not None:
        print(f"cos(pile_period_dir, fineweb_period_dir) = {float(per_pile.double() @ per_fw.double()):+.3f}")
    print(f"{'dir':16s} {'ln_gain':>9s} {'cos(Vh0)':>9s} {'||W_U d||':>10s} {'top1_share':>10s}")
    k = min(Vh.shape[0], S.shape[0])
    for nm, dvec in cand.items():
        if dvec is None:
            continue
        proj = Vh[:k] @ dvec                         # coords in unembedding singular basis
        energy = (S[:k] * proj) ** 2
        print(f"{nm:16s} {float((dvec ** 2 * gamma).sum()):9.3f} {float(dvec @ Vh[0]):9.3f} "
              f"{float(energy.sum() ** 0.5):10.2f} {float(energy[0] / energy.sum()):10.3f}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
