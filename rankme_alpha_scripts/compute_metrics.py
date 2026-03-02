import os, sys, re, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import powerlaw

_IDENTITY_RE = re.compile(r'step(\d+)\.npy$')
_HIDDEN_STATES_RE = re.compile(r'step(\d+)_hidden_states\.npy$')


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


def compute_metrics_for_hidden_states(args):
    """Process a single step's hidden_states file (n_layers, n_samples, hidden_dim).
    Returns (step, {layer_idx: metrics_dict})."""
    step, path = args
    all_layers = np.load(path)  # (n_layers, n_samples, hidden_dim)
    layer_metrics = {}
    for i, layer_act in enumerate(all_layers):
        eigen = powerlaw.get_eigenspectrum(layer_act)
        rm = powerlaw.rankme_metrics(eigen)
        alpha, ypred, fit_r2, fit_r2_100 = powerlaw.stringer_get_powerlaw(eigen, np.arange(11, 100))
        layer_metrics[i] = {
            'eigenspectrum': eigen,
            **rm,
            'alpha': alpha,
            'ypred': ypred,
            'r2': fit_r2,
            'r2_100': fit_r2_100,
        }
    return step, layer_metrics


def discover_activation_files(act_dir, collection_method):
    """Return {step: path} for activation files matching the collection method."""
    pattern = _IDENTITY_RE if collection_method == "identity" else _HIDDEN_STATES_RE
    step_files = {}
    for fname in os.listdir(act_dir):
        m = pattern.match(fname)
        if m:
            step_files[int(m.group(1))] = os.path.join(act_dir, fname)
    return step_files


def main(model_name: str, dataset_name: str = "fineweb", collection_method: str = "identity",
         layer: int = None, num_workers: int = None, recompute: bool = False):
    act_dir = os.path.join('activations', dataset_name, model_name)
    if not os.path.isdir(act_dir):
        print(f"No activations directory found: {act_dir}")
        return

    step_files = discover_activation_files(act_dir, collection_method)
    if not step_files:
        print(f"No activation files found in {act_dir} for collection_method={collection_method}")
        return

    results_dir = os.path.join('results', dataset_name)
    os.makedirs(results_dir, exist_ok=True)

    if collection_method == "identity":
        _compute_identity(step_files, results_dir, model_name, num_workers, recompute)
    else:
        _compute_hidden_states(step_files, results_dir, model_name, layer, num_workers, recompute)


def _compute_identity(step_files, results_dir, model_name, num_workers, recompute):
    results_path = os.path.join(results_dir, f'results_{model_name}.npy')
    res_dict = _load_existing(results_path)

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


def _compute_hidden_states(step_files, results_dir, model_name, layer, num_workers, recompute):
    """Compute metrics for hidden state files. If layer is specified, only that layer;
    otherwise all layers found in the files."""
    # Peek at first file to discover layer count
    first_path = next(iter(step_files.values()))
    n_layers = np.load(first_path, mmap_mode='r').shape[0]

    if layer is not None:
        layers = [layer]
    else:
        layers = list(range(n_layers))
        print(f"Found {n_layers} layers to process")

    for l in layers:
        results_path = os.path.join(results_dir, f'results_{model_name}_layer{l}.npy')
        res_dict = _load_existing(results_path)

        if recompute:
            to_compute = list(step_files.items())
        else:
            to_compute = [(s, p) for s, p in step_files.items() if s not in res_dict]

        if not to_compute:
            print(f"[layer {l}] All {len(step_files)} steps already have metrics")
            continue

        print(f"[layer {l}] Computing metrics for {len(to_compute)}/{len(step_files)} steps "
              f"using {num_workers or os.cpu_count()} workers...")

        # Extract single layer and compute
        tasks = [(s, p, l) for s, p in to_compute]
        with Pool(processes=num_workers) as pool:
            for step, metrics in pool.imap_unordered(_compute_single_layer, tasks):
                res_dict[step] = metrics
                print(f"  [layer {l}] Step {step}: rankme={metrics['rankme']:.3f}, "
                      f"alpha={metrics['alpha']:.3f}, r2_100={metrics['r2_100']:.3f}")

        np.save(results_path, res_dict)
        print(f"[layer {l}] Saved {len(res_dict)} results to {results_path}")


def _compute_single_layer(args):
    """Load one layer from a hidden_states file and compute metrics."""
    step, path, layer_idx = args
    activations = np.load(path, mmap_mode='r')[layer_idx]
    activations = np.array(activations)  # read into memory from mmap
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


def _load_existing(results_path):
    if os.path.exists(results_path):
        try:
            return np.load(results_path, allow_pickle=True).item()
        except (OSError, ValueError, TypeError):
            pass
    return {}


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from jsonargparse import CLI
    CLI(main)
