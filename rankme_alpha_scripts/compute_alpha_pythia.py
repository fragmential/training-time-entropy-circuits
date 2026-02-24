import gc, os, sys, torch, numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader, wikitext_loader, lam_loader, sciq_loader
from rankme_alpha_scripts.activation_collection import collect_last_token_activations, HOOK_LOCATION_BY_METHOD, replace_output_head_with_identity

DATASET_LOADERS = {
    "fineweb": fineweb_loader.get_dataset,
    "wikitext": wikitext_loader.get_dataset,
    "lam": lam_loader.get_dataset,
    "sciq": sciq_loader.get_dataset,
}


def prefetch_checkpoint(model_name, step_num):
    """Download model weights in background so they're cached for next iteration."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_name, revision=f"step{step_num}")
    except Exception:
        pass


def delete_cached_revision(model_name, revision):
    """Remove a specific revision from the HF cache to free disk space."""
    _force_delete_cached_revision(model_name, revision)


def _force_delete_cached_revision(model_name, revision):
    """Directly remove snapshot, ref, and orphaned blobs.
    Uses inodes to track references (works with both symlinks and hardlinks)."""
    import shutil
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_cache = os.path.join(hf_home, "hub", f"models--{model_name.replace('/', '--')}")
    if not os.path.isdir(model_cache):
        return

    ref_path = os.path.join(model_cache, "refs", revision)
    if not os.path.exists(ref_path):
        return

    try:
        with open(ref_path) as f:
            commit_hash = f.read().strip()

        snapshot_dir = os.path.join(model_cache, "snapshots", commit_hash)
        if os.path.isdir(snapshot_dir):
            shutil.rmtree(snapshot_dir)

        os.remove(ref_path)

        # Collect inodes still referenced by remaining snapshots
        snapshots_base = os.path.join(model_cache, "snapshots")
        referenced_inodes = set()
        if os.path.isdir(snapshots_base):
            for snap in os.listdir(snapshots_base):
                snap_path = os.path.join(snapshots_base, snap)
                if os.path.isdir(snap_path):
                    for root, _, files in os.walk(snap_path):
                        for fname in files:
                            fpath = os.path.join(root, fname)
                            try:
                                referenced_inodes.add(os.stat(fpath).st_ino)
                            except OSError:
                                pass

        # Delete orphaned blobs
        blobs_dir = os.path.join(model_cache, "blobs")
        if os.path.isdir(blobs_dir):
            for fname in os.listdir(blobs_dir):
                blob_path = os.path.join(blobs_dir, fname)
                try:
                    if os.stat(blob_path).st_ino not in referenced_inodes:
                        os.remove(blob_path)
                except OSError:
                    pass
    except Exception as e:
        print(f"Warning: could not clean cache for {revision}: {e}")


def extract_activations(
    model_name: str,
    step_num: int,
    filtered_texts: list,
    tokenizer,
    collection_method: str,
    max_length: int = 512,
    batch_size: int = 128,
) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = f"step{step_num}"

    model = None
    try:
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.float16)
        if collection_method == "identity":
            replace_output_head_with_identity(model)
        model.to(device)

        activations_arr = []
        with torch.no_grad():
            for bidx in tqdm(range(0, len(filtered_texts), batch_size), desc="Inference"):
                batch = filtered_texts[bidx:bidx+batch_size]
                tokenized = tokenizer(batch, padding="longest", return_tensors="pt",
                                      max_length=max_length, truncation=True)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)

                activations = collect_last_token_activations(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    collection_method=collection_method,
                )
                activations_arr.append(activations.float().cpu().numpy())
                del activations, input_ids, attention_mask
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        delete_cached_revision(model_name, revision)

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
    print(model_name, dataset_name)
    if collection_method == "kfac":
        raise NotImplementedError(
            "collection_method='kfac' is not implemented yet. "
            "Use 'identity' or 'hf_hidden_states' for now."
        )

    hook_location = HOOK_LOCATION_BY_METHOD[collection_method]

    dataset = DATASET_LOADERS[dataset_name]()

    # step_nums = [0,1,2,4,8,16,32,64,128,256,512] + list(np.arange(1000,143000+1,10000))
    early_steps = [0,8,16,32,64,128,256,512]
    later_steps = list(np.arange(1000,143000+1,1000))

    # Subsample only the later section to max_checkpoints
    if max_checkpoints and len(later_steps) > max_checkpoints:
        indices = np.linspace(0, len(later_steps) - 1, max_checkpoints, dtype=int)
        later_steps = [later_steps[i] for i in indices]
        print(f"Subsampled later checkpoints to {len(later_steps)}")

    step_nums = early_steps + later_steps
    print(f"Total checkpoints: {len(step_nums)} ({len(early_steps)} early + {len(later_steps)} later)")

    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    act_dir = os.path.join("activations", dataset_name, short_name)
    os.makedirs(act_dir, exist_ok=True)
    print(f"Saving activations to {act_dir} (collection_method={collection_method})")

    to_process = []
    for step_num in step_nums:
        path = os.path.join(act_dir, f"step{step_num}.npy") if collection_method == "identity" else os.path.join(act_dir, f"step{step_num}_{hook_location}.npy")
        if os.path.exists(path):
            continue
        if collection_method == "identity" and os.path.exists(os.path.join(act_dir, f"step{step_num}_{hook_location}.npy")):
            continue
        to_process.append(step_num)

    # Load tokenizer once for filtering (all Pythia checkpoints share the same tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter dataset, caching to disk so subsequent runs skip filtering
    import json
    os.makedirs('data/cache', exist_ok=True)
    cache_path = os.path.join('data', 'cache', f'filtered_texts_{dataset_name}_{num_samples}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            filtered_texts = json.load(f)
        print(f"Loaded {len(filtered_texts)} cached sequences from {cache_path}")
    else:
        filtered_texts = []
        for seq in tqdm(dataset, desc="Filtering"):
            text = seq[dataset_content_key]
            if tokenizer(text, return_tensors="pt").input_ids.shape[-1] > min_length:
                filtered_texts.append(text)
                if len(filtered_texts) >= num_samples:
                    break
        with open(cache_path, 'w') as f:
            json.dump(filtered_texts, f)
        print(f"Filtered and cached {len(filtered_texts)} sequences to {cache_path}")

    executor = ThreadPoolExecutor(max_workers=1)

    print(f"Processing {len(to_process)} remaining checkpoints...")
    for idx, step_num in enumerate(tqdm(to_process, desc="Checkpoints")):
        # Prefetch next checkpoint in background
        if idx + 1 < len(to_process):
            executor.submit(prefetch_checkpoint, model_name, to_process[idx + 1])

        try:
            activations = extract_activations(
                model_name,
                step_num,
                filtered_texts,
                tokenizer,
                collection_method=collection_method,
                max_length=max_length,
                batch_size=batch_size,
            )
            if collection_method == "identity":
                save_path = os.path.join(act_dir, f"step{step_num}.npy")
            else:
                save_path = os.path.join(act_dir, f"step{step_num}_{hook_location}.npy")
            np.save(save_path, activations)
            tqdm.write(f"Step {step_num}: saved {activations.shape} to {save_path}")
        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")

    executor.shutdown(wait=True)

    print(f"Completed! Activations saved to {act_dir}")

if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    CLI(main)
