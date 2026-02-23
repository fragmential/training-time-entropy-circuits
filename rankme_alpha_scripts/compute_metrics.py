import os, sys, re, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import powerlaw


def compute_metrics_for_file(args):
    step, path = args
    activations = np.load(path)
    eigen = powerlaw.get_eigenspectrum(activations)
    rm = powerlaw.rankme_metrics(eigen)
    alpha, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11, 100))
    return step, {
        'eigenspectrum': eigen,
        **rm,
        'alpha': alpha,
        'ypred': ypred,
        'r2': fit_r2,
        'r2_100': fit_r2_100,
    }


def main(model_name: str, num_workers: int = None, recompute: bool = False):
    act_dir = os.path.join('activations', model_name)
    if not os.path.isdir(act_dir):
        print(f"No activations directory found: {act_dir}")
        return

    # Find all step*.npy files
    step_files = {}
    for fname in os.listdir(act_dir):
        m = re.match(r'step(\d+)\.npy$', fname)
        if m:
            step_files[int(m.group(1))] = os.path.join(act_dir, fname)

    if not step_files:
        print(f"No activation files found in {act_dir}")
        return

    # Load existing results
    os.makedirs('results', exist_ok=True)
    results_path = os.path.join('results', f'results_{model_name}.npy')
    res_dict = {}
    if os.path.exists(results_path):
        try:
            res_dict = np.load(results_path, allow_pickle=True).item()
        except:
            pass

    # Determine which steps need computation
    if recompute:
        to_compute = list(step_files.items())
    else:
        to_compute = [(s, p) for s, p in step_files.items() if s not in res_dict]

    if not to_compute:
        print(f"All {len(step_files)} steps already have metrics in {results_path}")
        return

    print(f"Computing metrics for {len(to_compute)}/{len(step_files)} steps "
          f"using {num_workers or os.cpu_count()} workers...")

    with Pool(processes=num_workers) as pool:
        for step, metrics in pool.imap_unordered(compute_metrics_for_file, to_compute):
            res_dict[step] = metrics
            print(f"  Step {step}: rankme={metrics['rankme']:.3f}, "
                  f"alpha={metrics['alpha']:.3f}, r2_100={metrics['r2_100']:.3f}")

    np.save(results_path, res_dict)
    print(f"Saved {len(res_dict)} results to {results_path}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
