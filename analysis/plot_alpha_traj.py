import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import seaborn
seaborn.set_style('whitegrid')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.checkpoint_info import get_token_count


def load_results(model_name: str, dataset_name: str = 'fineweb'):
    new_path = os.path.join('results', dataset_name, f'results_{model_name}.npy')
    old_path = os.path.join('results', f'results_{model_name}.npy')
    path = new_path if os.path.exists(new_path) else old_path
    res = np.load(path, allow_pickle=True).item()
    step_nums = sorted(res.keys())
    return res, step_nums


def get_xs_steps(model_name, step_nums):
    return step_nums


def get_xs_tokens(model_name, step_nums):
    return [get_token_count(model_name, s) for s in step_nums]


def get_xs_flops(model_name, step_nums):
    num_params_str = model_name.split('pythia-')[-1].split('-deduped')[0]
    num_params_val = float(num_params_str[:-1])
    num_params_order = 1e6 if num_params_str[-1] == 'm' else 1e9
    flops_per_step = num_params_val * num_params_order
    return [flops_per_step * s for s in step_nums]


XVAR_FNS = {
    'steps': get_xs_steps,
    'tokens': get_xs_tokens,
    'flops': get_xs_flops,
}

XVAR_LABELS = {
    'steps': 'Steps',
    'tokens': 'Pretraining tokens',
    'flops': 'Flops',
    'isoflops': 'Parameters',
}

YVAR_LABELS = {
    'alpha': r'$\alpha$',
    'rankme': 'RankMe',
}


def plot_model_training(model_name: str, xvar: str = 'steps', yvar: str = 'alpha',
                        dataset_name: str = 'fineweb',
                        color: str = 'k', ls: str = '-', marker: str = 'o'):
    res, step_nums = load_results(model_name, dataset_name=dataset_name)
    xs = XVAR_FNS[xvar](model_name, step_nums)
    ys = [res[s][yvar] for s in step_nums]
    plt.plot(xs, ys, marker=marker, color=color, ls=ls, lw=3, label=f'{model_name}')


def plot_model_training_isoflops(model_name: str, yvar: str = 'alpha',
                                 dataset_name: str = 'fineweb',
                                 marker: str = 'o', max_flops: int = 2e15):
    res, step_nums = load_results(model_name, dataset_name=dataset_name)
    num_params_str = model_name.split('pythia-')[-1].split('-deduped')[0]
    num_params_val = float(num_params_str[:-1])
    num_params_order = 1e6 if num_params_str[-1] == 'm' else 1e9
    flops_per_step = num_params_val * num_params_order
    isoflops = np.array([1e13, 5e13, 1e14])
    isoflops_steps = (isoflops / flops_per_step // 1000 * 1000).astype(int)
    plot_steps = [s for s in isoflops_steps if s < max(step_nums) and s > min(step_nums)]
    ys = [res[s][yvar] for s in plot_steps]
    xs = [flops_per_step] * len(ys)
    plot_colors = colors[:len(plot_steps)]
    plt.scatter(xs, ys, marker=marker, color=plot_colors)


model_names = [
    'pythia-14m', 'pythia-31m',
    'pythia-70m', 'pythia-70m-deduped',
    'pythia-160m', 'pythia-160m-deduped',
    'pythia-410m', 'pythia-410m-deduped',
    'pythia-1b', 'pythia-1b-deduped',
    'pythia-1.4b', 'pythia-1.4b-deduped',
    'pythia-2.8b', 'pythia-2.8b-deduped',
    'pythia-6.9b', 'pythia-6.9b-deduped',
    'pythia-12b', 'pythia-12b-deduped',
    'OLMo-2-0425-1B',
    'OLMo-2-1124-7B',
]

colors = [
    'turquoise', 'cornflowerblue',
    'dodgerblue', 'dodgerblue',
    'gold', 'gold',
    'lime', 'lime',
    'darkgreen', 'darkgreen',
    'magenta', 'magenta',
    'deeppink', 'deeppink',
    'purple', 'purple',
    'brown', 'brown',
    'blue',
    'red',
]

filter_model_names = [
    # 'pythia-1.4b-deduped',
    # 'pythia-70m-deduped',
    # 'pythia-160m-deduped',
    # 'pythia-410m-deduped',
    # 'pythia-1b-deduped',
    # 'pythia-2.8b-deduped',
    # 'pythia-6.9b-deduped',
    # 'pythia-12b-deduped',
    'pythia-14m',
    'pythia-31m',
    'pythia-70m',
    'pythia-160m',
    'pythia-410m',
    'pythia-1b',
    'pythia-1.4b',
    'pythia-2.8b',
    'pythia-6.9b',
    # 'pythia-12b',
    # 'OLMo-2-0425-1B',
    # 'OLMo-2-1124-7B',
]

def main(xvar: str = 'steps', yvar: str = 'alpha', dataset_name: str = 'fineweb'):
    for midx, model_name in enumerate(tqdm(model_names)):
        if len(filter_model_names) and model_name not in filter_model_names: continue
        color = colors[midx]
        ls = '--' if 'deduped' in model_name else '-'
        marker = '*' if 'deduped' in model_name else 's'
        try:
            if xvar == 'isoflops':
                plot_model_training_isoflops(model_name, yvar=yvar, dataset_name=dataset_name, marker=marker)
            else:
                plot_model_training(model_name, xvar=xvar, yvar=yvar,
                                    dataset_name=dataset_name,
                                    color=color, ls=ls, marker='')
        except:
            continue

    plt.xscale('log')
    plt.xlabel(XVAR_LABELS[xvar], fontsize=14)
    plt.ylabel(YVAR_LABELS.get(yvar, yvar), fontsize=14)
    plt.legend()
    plt.show()

if __name__=='__main__':
    from jsonargparse import CLI

    CLI(main)
    # To run (e.g.):
    #  python analysis/plot_alpha_traj.py --xvar tokens --yvar alpha
    #  python analysis/plot_alpha_traj.py --xvar steps --yvar rankme
