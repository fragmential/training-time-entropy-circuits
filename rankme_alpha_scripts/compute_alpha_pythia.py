import os, sys, torch, numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import datasets
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader
from utils import powerlaw


def prefetch_checkpoint(model_name, step_num):
    """Download model weights in background so they're cached for next iteration."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_name, revision=f"step{step_num}")
    except Exception:
        pass


def delete_cached_revision(model_name, revision):
    """Remove a specific revision from the HF cache to free disk space."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == model_name:
                for rev in repo.revisions:
                    if any(ref == revision for ref in rev.refs):
                        strategy = cache_info.delete_revisions(rev.commit_hash)
                        strategy.execute()
                        return
    except Exception as e:
        print(f"Warning: could not clean cache for {revision}: {e}")


def get_metrics(model_name: str, step_num: int,
                filtered_texts: list, tokenizer,
                max_length: int = 512, batch_size: int = 128) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = f"step{step_num}"

    model = None
    try:
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=revision, torch_dtype=torch.float16)
        model.embed_out = nn.Identity()
        model.to(device)

        activations_arr = []
        with torch.no_grad():
            for bidx in tqdm(range(0, len(filtered_texts), batch_size), desc="Inference"):
                batch = filtered_texts[bidx:bidx+batch_size]
                tokenized = tokenizer(batch, padding="longest", return_tensors="pt",
                                      max_length=max_length, truncation=True)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)

                out = model(input_ids=input_ids, attention_mask=attention_mask)

                last_indices = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(input_ids.shape[0])
                activations = out.logits[batch_indices, last_indices, ...]
                activations_arr.append(activations.cpu().numpy())
                del out, input_ids, attention_mask
    finally:
        del model
        torch.cuda.empty_cache()

    delete_cached_revision(model_name, revision)

    all_activations = np.vstack(activations_arr)
    eigen = powerlaw.get_eigenspectrum(all_activations)
    rankme = powerlaw.rankme(eigen)
    alpha, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11,100))
    return {'eigenspectrum': eigen,
            'rankme': rankme,
            'ypred': ypred,
            'alpha': alpha,
            'r2': fit_r2,
            'r2_100': fit_r2_100}


def main(model_name: str = "EleutherAI/pythia-70m-deduped",
         dataset_name: str = "fineweb",
         num_samples: int = 2000,
         max_length: int = 512,
         min_length: int = 32,
         max_checkpoints: int = 50,
         batch_size: int = 128):
    print(model_name, dataset_name)
    assert dataset_name in ['fineweb'], NotImplementedError

    short_name = model_name.split("/")[-1] if "/" in model_name else model_name

    dataset = fineweb_loader.get_dataset()

    # step_nums = [0,1,2,4,8,16,32,64,128,256,512] + list(np.arange(1000,143000+1,10000))
    step_nums = [0,8,16,32,64,128,256,512] + list(np.arange(1000,143000+1,1000))

    # Uniformly subsample to max_checkpoints if needed
    if max_checkpoints and len(step_nums) > max_checkpoints:
        indices = np.linspace(0, len(step_nums) - 1, max_checkpoints, dtype=int)
        step_nums = [step_nums[i] for i in indices]
        print(f"Subsampled to {len(step_nums)} checkpoints")

    os.makedirs('results', exist_ok=True)
    tmp_save_fname = os.path.join('results', f'results_{short_name}_temp.npy')
    final_save_fname = os.path.join('results', f'results_{short_name}.npy')

    res_dict = {}
    if os.path.exists(final_save_fname):
        try:
            res_dict = np.load(final_save_fname, allow_pickle=True).item()
        except:
            pass

    if os.path.exists(tmp_save_fname):
        try:
            temp = np.load(tmp_save_fname, allow_pickle=True).item()
            res_dict.update(temp)
        except:
            pass

    completed = set(res_dict.keys())
    to_process = [s for s in step_nums if s not in completed]

    # Load tokenizer once for filtering (all Pythia checkpoints share the same tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=f"step{to_process[0]}" if to_process else "step0")
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
            text = seq["text"]
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
            res_dict[step_num] = get_metrics(model_name, step_num, filtered_texts, tokenizer,
                                             max_length=max_length, batch_size=batch_size)
            tqdm.write(f"Step {step_num}: rankme={res_dict[step_num]['rankme']:.3f}, "
                       f"alpha={res_dict[step_num]['alpha']:.3f}, "
                       f"r2_100={res_dict[step_num]['r2_100']:.3f}")
        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")

        np.save(tmp_save_fname, res_dict)

    executor.shutdown(wait=True)

    print(f"Saving results to {final_save_fname}")
    np.save(final_save_fname, res_dict)
    if os.path.exists(tmp_save_fname):
        os.remove(tmp_save_fname)

if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    CLI(main)
