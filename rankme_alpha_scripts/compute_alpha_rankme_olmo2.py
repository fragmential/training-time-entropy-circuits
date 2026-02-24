import gc, os, sys, torch, numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader, wikitext_loader, lam_loader, sciq_loader
from rankme_alpha_scripts.activation_collection import collect_last_token_activations, HOOK_LOCATION_BY_METHOD, replace_output_head_with_identity

DATASET_LOADERS = {
    "fineweb": fineweb_loader.get_dataset,
    "wikitext": wikitext_loader.get_dataset,
    "lam": lam_loader.get_dataset,
    "sciq": sciq_loader.get_dataset,
}


def prefetch_checkpoint(model_name, revision):
    """Download model weights in background so they're cached for next iteration."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_name, revision=revision)
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


def extract_activations_for_checkpoint(
    model_name,
    revision,
    filtered_texts,
    tokenizer,
    max_length,
    batch_size,
    collection_method: str,
):
    print(f"Running model {model_name}/{revision}")

    try:
        step_num = int(revision.split('-tokens')[0].split('step')[-1])
    except ValueError as e:
        print(f"Could not parse revision {revision} for model {model_name}\n{e}")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision,
            torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        if collection_method == "identity":
            replace_output_head_with_identity(model)
        model.to(device)

        activations_arr = []
        for bidx in tqdm(range(0, len(filtered_texts), batch_size), desc="Inference"):
            batch_prompts = filtered_texts[bidx : bidx+batch_size]
            tokenized = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                  max_length=max_length, truncation=True)
            input_ids = tokenized.input_ids.to(device)
            attention_mask = tokenized.attention_mask.to(device)

            with torch.no_grad():
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

    if not activations_arr:
        print("No valid features extracted.")
        return step_num, None

    return step_num, np.vstack(activations_arr)

def get_available_checkpoints(filepath: str) -> dict:
    print(f"Opening checkpoint file: {filepath}")
    checkpoint_map = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if "step" in line and "-tokens" in line:
                    try:
                        step = int(line.split('-tokens')[0].split('step')[-1])
                        checkpoint_map[step] = line
                    except ValueError:
                        print(f"Skipping line (invalid step): {line}")
                        continue
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
    except Exception as e:
        print(f"Unexpected error: {e}")

    print(f"Found {len(checkpoint_map)} checkpoints")
    return checkpoint_map

def run_all_checkpoints(model_name="allenai/OLMo-1B", dataset_name="fineweb",
                        dataset_content_key="text", min_length=32, max_length=512,
                        batch_size=128, num_samples=2000,
                        max_checkpoints=50,
                        revisions_file="1b_revisions.txt",
                        early_training_model: str = None,
                        collection_method: str = "identity"):
    if collection_method == "kfac":
        raise NotImplementedError(
            "collection_method='kfac' is not implemented yet. "
            "Use 'identity' or 'hf_hidden_states' for now."
        )

    hook_location = HOOK_LOCATION_BY_METHOD[collection_method]

    short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    act_dir = os.path.join("activations", dataset_name, short_name)
    os.makedirs(act_dir, exist_ok=True)
    print(f"Saving activations to {act_dir} (collection_method={collection_method})")

    checkpoint_map = get_available_checkpoints(revisions_file)

    all_steps = sorted(checkpoint_map.keys())
    early_steps = sorted(s for s in all_steps if s <= 10000)
    later_steps = sorted(s for s in all_steps if s > 10000 and s % 10000 == 0)

    # Subsample only the later section to max_checkpoints
    if max_checkpoints and len(later_steps) > max_checkpoints:
        indices = np.linspace(0, len(later_steps) - 1, max_checkpoints, dtype=int)
        later_steps = [later_steps[i] for i in indices]
        print(f"Subsampled later checkpoints to {len(later_steps)}")

    step_nums = early_steps + later_steps
    print(f"Total checkpoints: {len(step_nums)} ({len(early_steps)} early + {len(later_steps)} later)")

    to_process = []
    for step_num in step_nums:
        path = os.path.join(act_dir, f"step{step_num}.npy") if collection_method == "identity" else os.path.join(act_dir, f"step{step_num}_{hook_location}.npy")
        if os.path.exists(path):
            continue
        if collection_method == "identity" and os.path.exists(os.path.join(act_dir, f"step{step_num}_{hook_location}.npy")):
            continue
        to_process.append(step_num)

    # Filter dataset, caching to disk so subsequent runs skip filtering
    import json
    os.makedirs('data/cache', exist_ok=True)
    cache_path = os.path.join('data', 'cache', f'filtered_texts_{dataset_name}_{num_samples}.json')
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            filtered_texts = json.load(f)
        print(f"Loaded {len(filtered_texts)} cached sequences from {cache_path}")
    else:
        dataset = DATASET_LOADERS[dataset_name]()
        first_rev = checkpoint_map[to_process[0]] if to_process else list(checkpoint_map.values())[0]
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=first_rev, trust_remote_code=True)
        filtered_texts = []
        for text in tqdm(dataset[dataset_content_key], desc="Filtering"):
            if tokenizer(text, return_tensors="pt").input_ids.shape[-1] > min_length:
                filtered_texts.append(text)
                if len(filtered_texts) >= num_samples:
                    break
        with open(cache_path, 'w') as f:
            json.dump(filtered_texts, f)
        print(f"Filtered and cached {len(filtered_texts)} sequences to {cache_path}")

    # Load tokenizer once (shared across all revisions)
    first_rev = checkpoint_map[to_process[0]] if to_process else list(checkpoint_map.values())[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=first_rev, trust_remote_code=True)

    executor = ThreadPoolExecutor(max_workers=1)

    def _model_for_step(step):
        if early_training_model and step >= 1000 and step <= 10000:
            return early_training_model
        return model_name

    print(f"Processing {len(to_process)} remaining checkpoints...")
    prefetch_future = None
    for idx, step in enumerate(tqdm(to_process, desc="Extracting activations")):
        revision = checkpoint_map[step]
        step_model = _model_for_step(step)

        # Prefetch next checkpoint in background
        if idx + 1 < len(to_process):
            next_step = to_process[idx + 1]
            next_rev = checkpoint_map[next_step]
            prefetch_future = executor.submit(prefetch_checkpoint, _model_for_step(next_step), next_rev)
        else:
            prefetch_future = None

        try:
            step_num, activations = extract_activations_for_checkpoint(
                step_model,
                revision,
                filtered_texts,
                tokenizer,
                max_length,
                batch_size,
                collection_method=collection_method,
            )
            if activations is not None:
                if collection_method == "identity":
                    save_path = os.path.join(act_dir, f"step{step_num}.npy")
                else:
                    save_path = os.path.join(act_dir, f"step{step_num}_{hook_location}.npy")
                np.save(save_path, activations)
                tqdm.write(f"Step {step_num}: saved {activations.shape} to {save_path}")
        except Exception as e:
            tqdm.write(f"Error at step {step}: {str(e)}")

        # Wait for prefetch to finish before cleaning cache, so blob
        # orphan detection sees all symlinks from the prefetched snapshot
        if prefetch_future is not None:
            prefetch_future.result()
        delete_cached_revision(step_model, revision)

    executor.shutdown(wait=True)

    print(f"Completed! Activations saved to {act_dir}")

if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    CLI(run_all_checkpoints)
