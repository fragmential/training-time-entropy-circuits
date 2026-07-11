"""TriviaQA answer-likelihood memorization metric across a checkpoint sweep.

For each checkpoint we score P(answer | "Question: <q>? Answer:") on the TriviaQA
`rc` validation split and store, per step, the summed answer-token logprob of every
example (float array) plus aggregate scalars. Plots on the same step-axis as the
spectral metrics under data/results/.

Uses model_registry for checkpoint discovery + prefetch, so both Pythia and OLMo-2
sweep uniformly (mirrors scripts/collect.py's loop).
"""
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_registry import (  # noqa: E402
    get_model_config,
    get_checkpoint_schedule,
    load_model,
    load_tokenizer,
    prefetch_checkpoint,
    delete_cached_revision,
)


def _instruction(question, answer):
    q = question if question.endswith("?") else question + "?"
    text = f"Question: {q} Answer: {answer}"
    prompt = text.rsplit(f" {answer}", 1)[0]  # "Question: <q>? Answer:"
    return text, prompt


def prepare_examples(tokenizer, limit=0):
    """Tokenize every validation example once. Returns list of dicts with the full
    token ids and the [start, end) answer span (positions of answer tokens)."""
    # rc.nocontext = same Q/A pairs as "rc" without the multi-GB evidence documents
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    examples = []
    for idx, sample in enumerate(tqdm(ds, desc="Tokenizing", unit="ex")):
        if limit and len(examples) >= limit:
            break
        question = sample["question"]
        answer = sample["answer"]["value"]
        text, prompt = _instruction(question, answer)
        full_ids = tokenizer.encode(text)
        prompt_ids = tokenizer.encode(prompt)
        start, end = len(prompt_ids), len(full_ids)
        if end <= start:  # tokenizer merged across the boundary; skip degenerate
            continue
        examples.append(
            {"idx": idx, "question": question, "answer": answer,
             "input_ids": full_ids, "ans_start": start, "ans_end": end}
        )
    return examples


def score_batch(model, batch, pad_id, device):
    """Summed answer-token logprob for each example in the batch."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    for i, b in enumerate(batch):
        ids = b["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids)
        attn[i, : len(ids)] = 1
    with torch.no_grad():
        logits = model(input_ids.to(device), attention_mask=attn.to(device)).logits
    sums = []
    for i, b in enumerate(batch):
        ids, s, e = b["input_ids"], b["ans_start"], b["ans_end"]
        # token j (s<=j<e) is predicted from logits at position j-1
        pos = torch.arange(s - 1, e - 1, device=device)
        logp = F.log_softmax(logits[i, pos].float(), dim=-1)
        tgt = torch.tensor(ids[s:e], device=device)
        sums.append(logp.gather(1, tgt[:, None]).sum().item())
    return sums


def run_checkpoint(model, examples, pad_id, device, batch_size):
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    seq_logprob = np.empty(len(examples), dtype=np.float32)
    for k in tqdm(range(0, len(order), batch_size), desc="  scoring", leave=False):
        idxs = order[k : k + batch_size]
        sums = score_batch(model, [examples[i] for i in idxs], pad_id, device)
        for i, s in zip(idxs, sums):
            seq_logprob[i] = s
    return seq_logprob


def main(
    model_name: str = None,
    max_checkpoints: int = 50,
    checkpoint_spacing: str = "log",
    batch_size: int = 64,
    max_examples: int = 0,
    output_dir: str = "data/results/memorization_llm",
):
    if not model_name:
        raise SystemExit("--model_name is required")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = model_name.split("/")[-1]
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"results_{short}.npy")
    examples_path = os.path.join(output_dir, f"examples_{short}.npy")

    config = get_model_config(model_name)
    schedule = get_checkpoint_schedule(config, max_checkpoints, checkpoint_spacing)
    print(f"{short}: {len(schedule)} checkpoints, spacing={checkpoint_spacing}")

    tokenizer = load_tokenizer(config, revision=schedule[0][1])
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    examples = prepare_examples(tokenizer, limit=max_examples)
    print(f"{short}: scoring {len(examples)} examples")
    n_answer_tokens = np.array([e["ans_end"] - e["ans_start"] for e in examples], dtype=np.int32)
    # one-time sidecar: strings + token counts, aligned to the seq_logprob arrays
    np.save(examples_path, {
        "idx": np.array([e["idx"] for e in examples]),
        "question": [e["question"] for e in examples],
        "answer": [e["answer"] for e in examples],
        "n_answer_tokens": n_answer_tokens,
    })

    res = {}
    if os.path.exists(results_path):
        try:
            res = np.load(results_path, allow_pickle=True).item()
        except (OSError, ValueError, TypeError):
            pass

    to_process = [s for s in schedule if s[0] not in res]
    print(f"{short}: {len(to_process)} checkpoints remaining")
    executor = ThreadPoolExecutor(max_workers=1)
    for i, (step_num, revision, step_model) in enumerate(tqdm(to_process, desc="Checkpoints")):
        prefetch_future = None
        if i + 1 < len(to_process):
            _, next_rev, next_model = to_process[i + 1]
            prefetch_future = executor.submit(prefetch_checkpoint, next_model, next_rev)
        model = None
        try:
            t0 = time.time()
            model = load_model(config, step_model, revision).to(device).eval()
            seq_logprob = run_checkpoint(model, examples, pad_id, device, batch_size)
            token_logprob = seq_logprob / n_answer_tokens
            res[step_num] = {
                "mean_seq_logprob": float(seq_logprob.mean()),
                "mean_token_logprob": float(token_logprob.mean()),
                "n_examples": int(len(examples)),
                "seq_logprob": seq_logprob,
            }
            np.save(results_path, res)  # incremental
            tqdm.write(f"  step {step_num}: mean_seq_logprob={seq_logprob.mean():.3f} "
                       f"mean_token_logprob={token_logprob.mean():.3f} ({time.time()-t0:.0f}s)")
        except Exception as e:
            tqdm.write(f"  skipping step {step_num}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if prefetch_future is not None:
            prefetch_future.result()
        delete_cached_revision(step_model, revision)
    executor.shutdown(wait=True)
    print(f"Saved {len(res)} checkpoints to {results_path}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
