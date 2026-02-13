import os, sys, torch, numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader
from utils import powerlaw


def prefetch_checkpoint(model_name, revision):
    """Download model weights in background so they're cached for next iteration."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_name, revision=revision)
    except Exception:
        pass


def compute_metrics_for_checkpoint(model_name, revision, dataset, dataset_content_key,
                                   min_length, max_length, batch_size, num_samples):
    print(f"Running model {model_name}/{revision}")

    try:
        step_num = int(revision.split('-tokens')[0].split('step')[-1])
    except ValueError as e:
        print(f"Could not parse revision {revision} for model {model_name}\n{e}")
        return None

    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision,
        torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision,
        trust_remote_code=True
    )

    filtered_dataset = [s for s in tqdm(dataset[dataset_content_key], desc="Filtering")
                        if tokenizer(s, return_tensors="pt").input_ids.shape[-1] > min_length]
    if len(filtered_dataset) > num_samples:
        filtered_dataset = filtered_dataset[:num_samples]
    print(f"Using {len(filtered_dataset)} sequences")

    activations_arr = []
    for bidx in tqdm(range(0, len(filtered_dataset), batch_size), desc="Inference"):
        batch_prompts = filtered_dataset[bidx : bidx+batch_size]
        tokenized = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                              max_length=max_length, truncation=True)
        input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = np.arange(attention_mask.shape[0])
        activations = outputs.hidden_states[-1][batch_indices, last_indices, ...]
        activations_arr.append(activations.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    if not activations_arr:
        print("No valid features extracted.")
        return None

    all_activations = np.vstack(activations_arr)
    eigen = powerlaw.get_eigenspectrum(all_activations)
    rankme = powerlaw.rankme(eigen)
    alpha, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11, 100))

    return {
        'step': step_num,
        'checkpoint': revision,
        'eigenspectrum': eigen,
        'rankme': rankme,
        'alpha': alpha,
        'ypred': ypred,
        'r2': fit_r2,
        'r2_100': fit_r2_100
    }

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
                        batch_size=32, num_samples=2000,
                        revisions_file="1b_revisions.txt"):
    short_name = model_name.split("/")[-1] if "/" in model_name else model_name

    os.makedirs('results', exist_ok=True)

    checkpoint_map = get_available_checkpoints(revisions_file)

    all_steps = sorted(checkpoint_map.keys())
    step_nums = []
    for s in all_steps:
        if s <= 10000:
            step_nums.append(s)
        elif s % 10000 == 0:
            step_nums.append(s)

    step_nums = sorted(step_nums)

    tmp_save = os.path.join('results', f'results_{short_name}_temp.npy')
    final_save = os.path.join('results', f'results_{short_name}.npy')

    res_dict = {}
    if os.path.exists(final_save):
        try:
            res_dict = np.load(final_save, allow_pickle=True).item()
        except:
            pass

    if os.path.exists(tmp_save):
        try:
            temp = np.load(tmp_save, allow_pickle=True).item()
            res_dict.update(temp)
        except:
            pass

    completed = set(res_dict.keys())
    to_process = [s for s in step_nums if s not in completed]

    dataset = fineweb_loader.get_dataset()
    executor = ThreadPoolExecutor(max_workers=1)

    print(f"Processing {len(to_process)} remaining checkpoints...")
    for idx, step in enumerate(tqdm(to_process, desc="Calculating metrics")):
        revision = checkpoint_map[step]

        # Prefetch next checkpoint in background
        if idx + 1 < len(to_process):
            next_rev = checkpoint_map[to_process[idx + 1]]
            executor.submit(prefetch_checkpoint, model_name, next_rev)

        try:
            result = compute_metrics_for_checkpoint(model_name, revision, dataset,
                                                    dataset_content_key, min_length,
                                                    max_length, batch_size, num_samples)
            if result:
                res_dict[step] = result
                tqdm.write(f"Step {step}: rankme={result['rankme']:.3f}, alpha={result['alpha']:.3f}, r2_100={result['r2_100']:.3f}")
                np.save(tmp_save, res_dict)
        except Exception as e:
            tqdm.write(f"Error at step {step}: {str(e)}")
            continue

    executor.shutdown(wait=True)

    np.save(final_save, res_dict)
    if os.path.exists(tmp_save):
        os.remove(tmp_save)

    print(f"Completed! Results saved to {final_save}")

if __name__ == "__main__":
    from jsonargparse import CLI
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    CLI(run_all_checkpoints)
