"""Own-generations entropy lens for nanochat (the entropy-lens paper's protocol, softcap
applied): per checkpoint, 64 BOS-seeded generations at temperature 0.7, 32 tokens each;
per-layer mean next-token entropy of the capped logit-lens readout at the generated (last)
positions. This is the DEFAULT entropy-profile protocol (data-free, the model's inherent
profile); the teacher-forced lens collected via `vocab_entropy: true` is the secondary view.

Saves data/results/vocab_entropy_own_gen/results_nanochat-<tag>.npy in the standard results
shape: {step: {"vocab_entropy": {"entropy_lens_own_gen": {"per_layer": [emb, blk0..blkN]}}}}.

Run on a GPU node:  uv run python oneoff_scripts/entropy_own_gen.py --tag d12
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.nanochat_gpt import GPT, GPTConfig, NanochatTokenizer, NANOCHAT_DIR

DEV = "cuda"
NUM_SAMPLES, MAX_TOKENS, TEMPERATURE = 64, 32, 0.7   # the old report's exact spec
SOFTCAP = 15.0


def _steps(ckpt_dir: str) -> list[int]:
    return sorted(int(f.split("_")[1].split(".")[0])
                  for f in os.listdir(ckpt_dir) if f.startswith("model_"))


def _load(ckpt_dir: str, step: int) -> GPT:
    with open(os.path.join(ckpt_dir, f"meta_{step:06d}.json")) as f:
        meta = json.load(f)
    model = GPT(GPTConfig(**meta["model_config"]))
    sd = torch.load(os.path.join(ckpt_dir, f"model_{step:06d}.pt"),
                    map_location="cpu", weights_only=True)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()}, strict=True)
    return model.to(DEV, dtype=torch.bfloat16).eval()


class _Capture:
    """emb (post embed-norm, = blk0 input) + every block output."""

    def __init__(self, model: GPT) -> None:
        self.h: list[torch.Tensor] = []
        self.handles = [model.transformer.h[0].register_forward_pre_hook(
            lambda m, a: self.h.append(a[0].detach()))]
        for blk in model.transformer.h:
            self.handles.append(blk.register_forward_hook(
                lambda m, i, o: self.h.append(o.detach())))

    def pop(self) -> list[torch.Tensor]:
        h, self.h = self.h, []
        return h

    def close(self) -> None:
        for hd in self.handles:
            hd.remove()


def _entropies(model: GPT, hs: list[torch.Tensor], chunk: int = 4096) -> list[float]:
    """Per-layer mean entropy (nats) of softmax(softcap(lm_head(rms_norm(h)))) — fp32."""
    W = torch.as_tensor(model.lm_head.weight)
    out = []
    for h in hs:
        s, n = 0.0, 0
        for i in range(0, h.shape[0], chunk):
            x = F.rms_norm(h[i:i + chunk].float(), (h.shape[-1],))
            logits = SOFTCAP * torch.tanh(x @ W.float().T / SOFTCAP)
            lp = torch.log_softmax(logits, -1)
            s += (-(lp.exp() * lp).sum(-1)).sum().item()
            n += x.shape[0]
        out.append(s / n)
    return out


@torch.no_grad()
def own_gen_profile(model: GPT, bos: int, n_layers: int) -> list[float]:
    cap = _Capture(model)
    per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers + 1)]
    for i in range(NUM_SAMPLES):
        g = torch.Generator(device=DEV).manual_seed(i)
        ids = torch.tensor([[bos]], device=DEV)
        for _ in range(MAX_TOKENS):
            out = model(ids)                             # capped logits, the trained head
            for l, h in enumerate(cap.pop()):
                per_layer[l].append(h[0, -1])
            probs = torch.softmax(out.logits[0, -1].float() / TEMPERATURE, -1)
            ids = torch.cat([ids, torch.multinomial(probs, 1, generator=g)[None]], dim=1)
    cap.close()
    return _entropies(model, [torch.stack(x) for x in per_layer])


def main(tag: str = "d12", out_root: str = "data/results/vocab_entropy_own_gen") -> None:
    ckpt_dir = os.path.join(NANOCHAT_DIR, "base_checkpoints", tag)
    tok = NanochatTokenizer.from_base_dir(NANOCHAT_DIR)
    steps = _steps(ckpt_dir)
    os.makedirs(out_root, exist_ok=True)
    out_path = os.path.join(out_root, f"results_nanochat-{tag}.npy")
    res: dict = np.load(out_path, allow_pickle=True).item() if os.path.exists(out_path) else {}
    for step in steps:
        if step in res:
            continue
        model = _load(ckpt_dir, step)
        prof = own_gen_profile(model, tok.bos_token_id, len(model.transformer.h))
        res[step] = {"vocab_entropy": {"entropy_lens_own_gen": {"per_layer": prof}}}
        print(f"step {step:6d}: emb {prof[0]:.3f}  final {prof[-1]:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
        np.save(out_path, np.array(res, dtype=object), allow_pickle=True)   # incremental: resumable
    print(f"saved {out_path} ({len(res)} checkpoints)")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
