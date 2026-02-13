import os, sys, torch, numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import fineweb_loader
from utils import powerlaw


def prepare_dataset(dataset: datasets.arrow_dataset.Dataset,
                    tokenizer: AutoTokenizer,
                    content_key: str = "text",
                    min_length: int = 10,
                    max_length: int = 128,
                    ) -> list:
    encoded_dataset = []
    for seq in dataset:
        encoded_seq = tokenizer(seq[content_key], max_length=max_length,
                                return_tensors="pt", truncation=True)
        try:
            if encoded_seq['input_ids'].shape[-1] > min_length:
                encoded_dataset.append(encoded_seq)
        except:
            continue

    print(f"Number of valid sequences tokenized: {len(encoded_dataset)}/{len(dataset)}")
    return encoded_dataset


def get_rankme(model_name: str, step_num: int,
               dataset: datasets.arrow_dataset.Dataset) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(model_name, revision=f"step{step_num}")
    model.embed_out = nn.Identity()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=f"step{step_num}")
    tokenized_dataset = prepare_dataset(dataset, tokenizer)

    activations_arr = []
    with torch.no_grad():
        for seq in tokenized_dataset:
            for k, v in seq.items():
                seq[k] = v.to(device)
            out = model(**seq)
            activations_arr.append(out.logits[0, -1].cpu().numpy())

    activations_arr = np.array(activations_arr)
    eigen = powerlaw.get_eigenspectrum(activations_arr)
    rankme = powerlaw.rankme(eigen)
    return {'eigenspectrum': eigen,
            'rankme': rankme}


def main(model_name: str = "EleutherAI/pythia-70m-deduped",
         dataset_name: str = "fineweb"):
    assert dataset_name in ['fineweb'], NotImplementedError

    short_name = model_name.split("/")[-1] if "/" in model_name else model_name

    dataset = fineweb_loader.get_dataset()
    step_nums = list(range(5000, 1000000, 5000))

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
    to_process = [step for step in step_nums if step not in completed]

    print(f"Processing {len(to_process)} remaining checkpoints...")
    for step_num in tqdm(to_process):
        try:
            res_dict[step_num] = get_rankme(model_name, step_num, dataset)
            tqdm.write(f"Step {step_num}: rankme = {res_dict[step_num]['rankme']:.3f}")
        except Exception as e:
            tqdm.write(f"Skipping step {step_num}: {e}")

        np.save(tmp_save_fname, res_dict)

    print(f"Saving results to {final_save_fname}")
    np.save(final_save_fname, res_dict)
    if os.path.exists(tmp_save_fname):
        os.remove(tmp_save_fname)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
