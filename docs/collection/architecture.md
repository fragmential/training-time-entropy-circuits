# Architecture

Tracks how LLM representations evolve during pretraining via spectral methods (RankMe,
α power-law) and K-FAC curvature. Two families: Pythia (GPTNeoX) and OLMo-2 (LLaMA-style).

The whole system is one data model — the **address axis** — plus three processes that read
and write it: **collection**, **metric computation**, and **storage operations**.

---

## The address axis

Every value has one address: **`leaf . quantity . component`**, optionally viewed in a
`reference`'s basis (a projection).

- **leaf** — a dotted capture path, e.g. `blk3.mlp.up.in`, `blk3.attn.out`, `after_final_norm`.
  On the collection side a leaf is a *hook* (one physical capture point); on the accessor side
  it is a *node in the tree*. Same string, two contexts.
- **quantity** — `acts` or `grads` (what flowed through the leaf).
- **component** — a representation of that quantity:
  `gram` (Σxxᵀ, unnormalized), `n`, `mask`, `samples`, `mean`, `cov` (=gram/n), `cov_centered`,
  `eigvals`/`eigvecs` (+ `_centered`), `lvecs` (left singular vectors). See `COMPONENTS` in
  `utils/accessor.py`.

**Quantities live at leaves; metrics live at nodes.** A metric is an interaction *across* a
node's children (K-FAC = acts at one child ⊗ grads at another) and is never stored in the
collected `.pt` — it is computed on read and written only to the results `.npy`, keyed by node path.

---

## Dependency spine (enforced)

One-way: foundation → library top → consumers. `utils/accessor.py` is the top of the library;
`scripts/` are its consumers.

```
scripts/collect.py ─┐                       scripts/compute_metrics.py ─┐
  model_registry    │                         accessor                  │
  hooks             ├── consumers             gpu                       ├── consumers
  hook_specs        │                         model_registry            │
  accessor          │                                                   │
  data_utils       ─┘                                                  ─┘
        │                                            │
        ▼                                            ▼
   utils/accessor.py  ── the library top (imports gpu, hook_names, model_registry) ──
        │
        ▼   foundation (no intra-utils deps): model_registry · hooks · hook_names · data_utils · gpu
   utils/hook_specs.py → (hook_names, model_registry)
```

Two `import-linter` contracts in `pyproject.toml` make this machine-checked (`uv run lint-imports`):
1. nothing in `utils` imports `utils.accessor` (it sits at the top),
2. `utils` never imports `scripts` (the library never imports its consumers).

---

## Module responsibilities

| Module | Owns |
|--------|------|
| `utils/model_registry.py` | Model/family facts (`ModelConfig`, the `_Family` table), checkpoint discovery, model/tokenizer loading, weight providers (`ModelWeights` live / `LazyWeights` lazy+cache), derivation transforms (`Linear`/`Norm`), the on-disk weight cache. |
| `utils/hooks.py` | Real-time capture during forward/backward: `HookCollector` (one leaf) and `MultiHeadOVDispatcher` (per-OV-head), both `Capturer`. Accumulates `gram`/`samples`/`mean`; knows nothing about storage formats. |
| `utils/hook_specs.py` | Resolves the `hooks` config patterns (`<leaf>[:acts\|:grads\|:both]`, `preset:kfac`) against the model's family-aware leaf universe → `SingleHookSpec`/`OVHeadSpec`. |
| `utils/hook_names.py` | Pure leaf-name grammar (parse / classify / Pythia-alias `canonical`). No model, no torch. |
| `utils/data_utils.py` | Text load+cache, packing/padding, token masks, labels. |
| `utils/gpu.py` | GPU-execution policy (see below). |
| `utils/accessor.py` | The reader/converter/saver: one `_resolve` over `CONVERSIONS`/`FORMATS`/`PROJECTIONS` tables; the `Node`/`View` tree; `save`; projections; identity/metadata; the `python -m utils.accessor` CLI. |
| `scripts/collect.py` | Orchestration: data → checkpoint loop → register hooks → forward/backward → `captured()` → `DataAccessor.save`. |
| `scripts/compute_metrics.py` | Walks the tree, computes spectral + node metrics, writes results `.npy`. |

---

## Process 1 — Collection (`scripts/collect.py`)

```
config YAML ─▶ CollectConfig
  data_utils (texts, packing, masks)
  model_registry (checkpoint schedule, load_model)
  hook_specs.resolve(cfg.hooks) ─▶ SingleHookSpec / OVHeadSpec
  hooks: HookCollector / MultiHeadOVDispatcher  ── forward (+ backward if any :grads)
  captured()  ─▶ {leaf: {acts_gram, acts_n, acts_mean?, grads_gram, ...}}
  DataAccessor(captured, config=, weights=ModelWeights(model,config), identity=(model,rev,run))
       .stamp(token_filter=…, n_chunks=…)
       .save(path, format=<base>, overrides=<±b/±o/…>)
  [optional inline] compute_metrics_for_checkpoint(acc) ─▶ save_step_metrics(...)
```

A K-FAC pair is just two ordinary leaves (`…in:acts` + `…out:grads`, or `preset:kfac`); the
projection's output acts (old "B") are *derived* on read, never collected. `fast_final_norm`
swaps the output head for `Identity()` to capture `after_final_norm` acts forward-only (errors
if any hook needs grads). Token masking is per-spec: MLP-projection leaves force `all` tokens;
residual/boundary/OV leaves use the config's `token_selection`.

## Process 2 — Metric computation (`scripts/compute_metrics.py`)

```
slurm/compute_metrics.sh ─▶ main() ─▶ run_pipeline (threads, RLock GPU funnel, checkpoint prefetch)
  per file: DataAccessor(data, config, weights=LazyWeights|abort)
    prewarm("eigvecs","eigvecs_centered")   # warm the shared eigendecompositions
    get_metrics(acc.v)                        # recursive tree walk, results keyed by node.path
    _to_numpy(...)                            # the single torch→numpy boundary
    save_step_metrics(results_path, step, metrics)   # the ONE results-.npy writer
```

`get_metrics` writes, per node: the per-quantity spectral family (`{q}_uncentered`,
`{q}_centered`, `{q}_mean_metrics`, `{q}_mean_vec`, cross-checkpoint `{q}_cross_*`), plus any
`METRICS` whose operands resolve there — `gen` (a leaf's acts vs grads), `kfac` (proj node:
`in.acts`⊗`out.grads`), `projections_kfac` (mlp node: `up.in.acts`⊗`down.out.grads`),
`mean_metrics_blk_vs_res` (residual sub-block nodes). `save_step_metrics` is the single owner
of results-`.npy` writes, called by both `main()` and collect's inline path so they can't diverge.

## Process 3 — Storage operations (`python -m utils.accessor`)

`slurm/storage.sh` → the accessor CLI: `info` (inspect), `convert --to <format> [--materialize ±b/±o/regex]`
(re-materialize a format), `project --onto-file <ref>` (persist each view's eigvals in a
reference checkpoint's basis). All three share `_pt_files`/`_out_path`/`_run_pool`.

---

## On disk

A collected `.pt` is `{leaf: {"<q>_<component>": tensor, ...}, "__meta__": ...}`. Metadata keys
are dunder (`__hf_model__`, `__revision__`, `__run__`, `__format__`, `__token_filter__`,
`__n_chunks__`); persisted projections live under `__<q>_projections__`. A **format** is just a
named set of components to write (`FORMATS` in `utils/accessor.py`: `acts`, `acts_svd`, `cov`,
`cov_svd`, `eigenvalues`); `save` materializes those and anything a reload couldn't reconstruct.
On read, `_resolve` produces any requested component: **stored → convert (same leaf/quantity via
`CONVERSIONS`) → derive (another leaf via a `model_registry` transform + weights)**, memoized and
cycle-guarded. Derivation is what turns `.in` acts into `.out` acts (via the projection weight),
head `.slice` into `.contrib`, and `before_final_norm` into `after_final_norm`.

---

## GPU execution (`utils/gpu.py`)

Not an accessor concern. Every GPU op (eigh, svd, `W@C@Wᵀ` derivation) funnels through
`prefer_gpu`/`require_gpu`, which run on-device under a process-wide lock and return to CPU.
`run_pipeline` is the one shared parallel-over-files mechanism (a producer thread prefetches the
next checkpoint; `workers` consumer *threads* share one CUDA context; GPU work serializes through
a **reentrant** `RLock` — derivation nests GPU ops, so a plain `Lock` self-deadlocks). Both
`compute_metrics.main()` and the accessor CLI use it.

---

## Structural enforcement (how to keep it from rotting)

- **Imports** — `uv run lint-imports` (contracts above).
- **Types** — pyright (`typeCheckingMode = "basic"`, py3.14). `reportPrivateUsage` is off inside
  `utils` (Node/View/DataAccessor are one unit) but `error` in `scripts/` — a consumer boundary:
  scripts read through the tree, never reach into protected members.
- **Rule-sets** — `Capturer` (ABC over the two collectors), `WeightProvider` (Protocol), `Transform`
  (ABC: `Linear`/`Norm`).
- **Behaviour** — `pytest -m "not e2e"` (login) + e2e on GPU via `srun`; snapshot values are frozen
  (relabel, never regenerate).
