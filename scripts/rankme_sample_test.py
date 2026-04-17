"""Quick test: does Pythia-1b-deduped RankMe peak approach ~800 with 15k samples?

Runs on a few checkpoints around the known 5k-sample peak (step 12000),
comparing 5k vs 15k sample counts using identical FineWeb data loading.
"""

import os, sys, time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_utils import load_and_cache_texts, pad_and_tokenize, compute_token_mask
from utils.model_registry import get_model_config, get_checkpoint_schedule
from transformers import AutoModelForCausalLM
from utils.hooks import setup_identity_head, restore_head
from data import get_loader

HF_HOME = "/projects/prjs1815/hf_cache"
os.environ["HF_HOME"] = HF_HOME

MODEL_NAME    = "EleutherAI/pythia-1b-deduped"
DATASET       = "fineweb"
MAX_LENGTH    = 512
MIN_LENGTH    = 32
BATCH_SIZE    = 256
SAMPLE_COUNTS = [5_000, 15_000]

# Checkpoints to probe (around the known 5k peak at step 12000)
TARGET_STEPS  = [3000, 6000, 9000, 12000, 15000, 18000, 24000]

REF_5K = {          # from existing 5k results for quick sanity check
    0: 171.7, 3000: 372.0, 6000: 473.2, 9000: 498.1,
    12000: 529.3, 15000: 498.0, 18000: 485.5, 24000: 430.2,
}


def centered_rankme(acts: torch.Tensor) -> float:
    """Centered covariance eigenvalue-weighted entropy (Li et al. formula)."""
    mu  = acts.mean(0)
    X   = acts - mu
    cov = (X.T @ X) / acts.shape[0]
    ev  = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0).double().numpy()
    ev  = np.maximum(ev, 0)
    p   = ev / ev.sum()
    eps = 1e-10
    return float(np.exp(-np.sum(p * np.log(np.clip(p, eps, None)))))


def collect_acts(model, texts, tokenizer, device, max_n: int) -> torch.Tensor:
    """Run padded forward passes and return last-token activations (max_n, d)."""
    model.eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, min(len(texts), max_n), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            enc = pad_and_tokenize(batch_texts, tokenizer, MAX_LENGTH)
            input_ids   = enc["input_ids"].to(device)
            attn_mask   = enc["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = out.logits  # (B, T, d) — identity head so this IS the activation

            # Last real token per sequence
            mask = compute_token_mask(
                input_ids, attention_mask=attn_mask, token_selection="last"
            )  # (B, T) bool
            # mask has exactly one True per row
            acts = logits[mask].float().cpu()  # (B, d)
            all_acts.append(acts)
    return torch.cat(all_acts, 0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Load data ONCE (largest sample count needed) ---
    max_n = max(SAMPLE_COUNTS)
    print(f"\nLoading {max_n} FineWeb sequences (min_len={MIN_LENGTH}, max_len={MAX_LENGTH})...")
    t0 = time.time()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME)
    loader_fn = get_loader(DATASET)
    texts = load_and_cache_texts(loader_fn, max_n, MIN_LENGTH, tokenizer, DATASET)
    print(f"  Loaded {len(texts)} texts in {time.time()-t0:.1f}s")

    # --- Get checkpoint schedule ---
    cfg = get_model_config(MODEL_NAME)
    schedule = {s: (r, m) for s, r, m in get_checkpoint_schedule(cfg, None)}

    # --- Header ---
    ns_cols = "  ".join(f"{'RankMe@'+str(n//1000)+'k':>12}" for n in SAMPLE_COUNTS)
    print(f"\n{'Step':>8}  {'Ref5k':>8}  {ns_cols}")
    print("-" * (8 + 2 + 8 + 2 + 14 * len(SAMPLE_COUNTS)))

    for step in TARGET_STEPS:
        if step not in schedule:
            print(f"{step:8d}  (no checkpoint)")
            continue

        revision, hf_repo = schedule[step]
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            hf_repo, revision=revision, cache_dir=HF_HOME,
            torch_dtype=torch.float16,
        ).to(device)
        orig_head, head_attr = setup_identity_head(model)

        results = []
        for n in SAMPLE_COUNTS:
            acts = collect_acts(model, texts, tokenizer, device, n)
            rm   = centered_rankme(acts.float())
            results.append(rm)

        restore_head(model, orig_head, head_attr)
        del model
        torch.cuda.empty_cache()

        ref = REF_5K.get(step, float("nan"))
        cols = "  ".join(f"{r:>12.1f}" for r in results)
        print(f"{step:8d}  {ref:>8.1f}  {cols}  ({time.time()-t0:.0f}s)")

    print("\nDone.")


if __name__ == "__main__":
    main()
