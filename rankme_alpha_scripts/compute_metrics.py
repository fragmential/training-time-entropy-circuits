import os, sys, re, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import powerlaw


def compute_metrics_for_file(args):
    step, hook, path = args
    activations = np.load(path)
    eigen = powerlaw.get_eigenspectrum(activations)
    rm = powerlaw.rankme_metrics(eigen)
    alpha, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11, 100))
    return step, hook, {
        'eigenspectrum': eigen,
        **rm,
        'alpha': alpha,
        'ypred': ypred,
        'r2': fit_r2,
        'r2_100': fit_r2_100,
    }


def main(model_name: str, dataset_name: str = "fineweb", collection_method: str = "identity",
         num_workers: int = None, recompute: bool = False):
    act_dir = os.path.join('activations', dataset_name, model_name)
    if not os.path.isdir(act_dir):
        print(f"No activations directory found: {act_dir}")
        return

    step_hook_files = {}
    for fname in os.listdir(act_dir):
        if fname.endswith('_grads.npy'):
            continue
        m = re.match(r'step(\d+)\.npy$', fname)
        if m:
            step = int(m.group(1))
            step_hook_files.setdefault(step, {})['identity'] = os.path.join(act_dir, fname)
            continue
        m = re.match(r'step(\d+)_(.+)\.npy$', fname)
        if m:
            step = int(m.group(1))
            hook = m.group(2)
            if hook == 'identity' and 'identity' in step_hook_files.get(step, {}):
                continue
            step_hook_files.setdefault(step, {})[hook] = os.path.join(act_dir, fname)

    if collection_method == "identity":
        tasks = []
        for step, hooks in step_hook_files.items():
            if 'identity' in hooks:
                tasks.append((step, 'identity', hooks['identity']))
    else:
        tasks = []
        for step, hooks in step_hook_files.items():
            for hook, path in hooks.items():
                if hook != 'identity':
                    tasks.append((step, hook, path))

    if not tasks:
        print(f"No activation files found in {act_dir} for collection_method={collection_method}")
        return

    os.makedirs(os.path.join('results', dataset_name), exist_ok=True)
    if collection_method == "identity":
        results_path = os.path.join('results', dataset_name, f'results_{model_name}.npy')
    else:
        results_path = os.path.join('results', dataset_name, f'results_{model_name}_{collection_method}.npy')

    res_dict = {}
    if os.path.exists(results_path):
        try:
            res_dict = np.load(results_path, allow_pickle=True).item()
        except (OSError, ValueError, TypeError):
            pass

    if collection_method == "identity":
        if not recompute:
            tasks = [(s, h, p) for (s, h, p) in tasks if s not in res_dict]
    else:
        if not recompute:
            pending = []
            for step, hook, path in tasks:
                if step not in res_dict or hook not in res_dict[step]:
                    pending.append((step, hook, path))
            tasks = pending

    if not tasks:
        print(f"All available items already have metrics in {results_path}")
        return

    print(f"Computing metrics for {len(tasks)} items "
          f"(dataset={dataset_name}, collection_method={collection_method}) "
          f"using {num_workers or os.cpu_count()} workers...")
    with Pool(processes=num_workers) as pool:
        for step, hook, metrics in pool.imap_unordered(compute_metrics_for_file, tasks):
            if collection_method == "identity":
                res_dict[step] = metrics
                print(f"  Step {step}: rankme={metrics['rankme']:.3f}, "
                      f"alpha={metrics['alpha']:.3f}, r2_100={metrics['r2_100']:.3f}")
            else:
                if step not in res_dict:
                    res_dict[step] = {}
                res_dict[step][hook] = metrics
                print(f"  Step {step} [{hook}]: rankme={metrics['rankme']:.3f}, "
                      f"alpha={metrics['alpha']:.3f}, r2_100={metrics['r2_100']:.3f}")

    np.save(results_path, res_dict)
    print(f"Saved {len(res_dict)} results to {results_path}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
