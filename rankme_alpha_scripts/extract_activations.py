"""Unified activation extraction for Pythia and OLMo model families."""
import gc, json, os, sys, torch, numpy as np
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader, wikitext_loader, lam_loader, sciq_loader
from utils.model_registry import (
    get_model_config, get_checkpoint_schedule, load_model, load_tokenizer,
    prefetch_checkpoint, delete_cached_revision,
)

DATASET_LOADERS = {
    "fineweb": fineweb_loader.get_dataset,
    "wikitext": wikitext_loader.get_dataset,
    "lam": lam_loader.get_dataset,
    "sciq": sciq_loader.get_dataset,
}


def replace_output_head_with_identity(model):
    """Replace model output head with Identity for logit-space extraction."""
    for attr in ("lm_head", "embed_out"):
        if hasattr(model, attr):
            setattr(model, attr, nn.Identity())
            return
    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(nn.Identity())
        return
    raise ValueError("Could not locate output head (lm_head/embed_out/output_embeddings) for identity collection")


def collect_last_token_activations(model, input_ids, attention_mask, collection_method: str):
    """Run forward pass and extract activations at the last real token position.

    Returns a single tensor for identity, or a list of tensors (one per layer) for hf_hidden_states.
    """
    last_indices = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)

    if collection_method == "hf_hidden_states":
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return [hs[batch_indices, last_indices, ...] for hs in outputs.hidden_states]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits[batch_indices, last_indices, ...]


def extract_activations(model, filtered_texts, tokenizer, collection_method,
                         max_length=512, batch_size=128):
    """Run inference on a loaded model.

    Returns a single numpy array for identity, or a list of numpy arrays (one per layer)
    for hf_hidden_states.
    """
    device = next(model.parameters()).device
    per_layer = (collection_method == "hf_hidden_states")
    activations_arr = [] if not per_layer else None

    with torch.no_grad():
        for bidx in tqdm(range(0, len(filtered_texts), batch_size), desc="Inference"):
            batch = filtered_texts[bidx:bidx+batch_size]
            tokenized = tokenizer(batch, padding="longest", return_tensors="pt",
                                  max_length=max_length, truncation=True)
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            result = collect_last_token_activations(
                model=model, input_ids=input_ids,
                attention_mask=attention_mask, collection_method=collection_method,
            )

            if per_layer:
                if activations_arr is None:
                    activations_arr = [[] for _ in result]
                for i, layer_act in enumerate(result):
                    activations_arr[i].append(layer_act.float().cpu().numpy())
            else:
                activations_arr.append(result.float().cpu().numpy())
            del result, input_ids, attention_mask

    if per_layer:
        return [np.vstack(layer) for layer in activations_arr]
    return np.vstack(activations_arr)


def main(
    model_name: str = "EleutherAI/pythia-70m-deduped",
    dataset_name: str = "fineweb",
    dataset_content_key: str = "text",
    num_samples: int = 2000,
    max_length: int = 512,
    min_length: int = 32,
    max_checkpoints: int = 50,
    batch_size: int = 128,
    collection_method: str = "identity",
):
    config = get_model_config(model_name)
    print(model_name, dataset_name)

    # Checkpoint schedule
    schedule = get_checkpoint_schedule(config, max_checkpoints)
    print(f"Total checkpoints: {len(schedule)}")

    # Output directory
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    act_dir = os.path.join("activations", dataset_name, short_name)
    os.makedirs(act_dir, exist_ok=True)
    print(f"Saving activations to {act_dir} (collection_method={collection_method})")

    # Filter to unprocessed checkpoints
    to_process = []
    for step_num, revision, step_model in schedule:
        if collection_method == "identity":
            if os.path.exists(os.path.join(act_dir, f"step{step_num}.npy")):
                continue
        else:
            if os.path.exists(os.path.join(act_dir, f"step{step_num}_hidden_states.npy")):
                continue
        to_process.append((step_num, revision, step_model))

    if not to_process:
        print("All checkpoints already extracted.")
        return

    # Tokenizer (use first checkpoint's revision for OLMo compatibility)
    first_revision = to_process[0][1]
    tokenizer = load_tokenizer(config, revision=first_revision)

    # Filter dataset, caching to disk
    os.makedirs("data/cache", exist_ok=True)
    cache_path = os.path.join("data", "cache", f"filtered_texts_{dataset_name}_{num_samples}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            filtered_texts = json.load(f)
        print(f"Loaded {len(filtered_texts)} cached sequences from {cache_path}")
    else:
        dataset = DATASET_LOADERS[dataset_name]()
        filtered_texts = []
        for seq in tqdm(dataset, desc="Filtering"):
            text = seq[dataset_content_key]
            if tokenizer(text, return_tensors="pt").input_ids.shape[-1] > min_length:
                filtered_texts.append(text)
                if len(filtered_texts) >= num_samples:
                    break
        with open(cache_path, "w") as f:
            json.dump(filtered_texts, f)
        print(f"Filtered and cached {len(filtered_texts)} sequences to {cache_path}")

    # Main extraction loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    executor = ThreadPoolExecutor(max_workers=1)

    print(f"Processing {len(to_process)} remaining checkpoints...")
    prefetch_future = None
    for idx, (step_num, revision, step_model) in enumerate(tqdm(to_process, desc="Checkpoints")):
        # Prefetch next checkpoint in background
        if idx + 1 < len(to_process):
            _, next_rev, next_model = to_process[idx + 1]
            prefetch_future = executor.submit(prefetch_checkpoint, next_model, next_rev)
        else:
            prefetch_future = None

        try:
            model = load_model(config, step_model, revision)
            if collection_method == "identity":
                replace_output_head_with_identity(model)
            model.to(device)

            result = extract_activations(
                model, filtered_texts, tokenizer, collection_method,
                max_length=max_length, batch_size=batch_size,
            )

            if isinstance(result, list):
                stacked = np.stack(result)
                save_path = os.path.join(act_dir, f"step{step_num}_hidden_states.npy")
                np.save(save_path, stacked)
                tqdm.write(f"Step {step_num}: saved {stacked.shape} to {save_path}")
            else:
                save_path = os.path.join(act_dir, f"step{step_num}.npy")
                np.save(save_path, result)
                tqdm.write(f"Step {step_num}: saved {result.shape} to {save_path}")
        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # Wait for prefetch before cleaning cache
        if prefetch_future is not None:
            prefetch_future.result()
        delete_cached_revision(step_model, revision)

    executor.shutdown(wait=True)
    print(f"Completed! Activations saved to {act_dir}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
