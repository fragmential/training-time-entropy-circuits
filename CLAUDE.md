# Codebase Guide

## What this project does
Tracks how LLM residual-stream representations evolve during pretraining: spectral methods
(RankMe, alpha), the exact per-block rank ledger (ΔS = χ + quality + interference via
in-memory cross-covariances), and cross-population generalized eigenanalysis. Model families:
Pythia (14m–12b), OLMo-2 (1B, 7B), nanochat-d12 (worktree), plus a toy classifier platform.

## Research state (read these before assuming anything)
- `docs/final_report.md` — the synthesis: headline results, hypothesis scoreboard, in-flight work.
- `docs/research_questions.md` — RQ1–RQ3 + hypotheses with decision criteria and status.
- `docs/dig_findings.md` (RQ1 evidence), `docs/rq2_methods.md` + `docs/splithalf_note.md`
  (RQ2 methods), `docs/rank_ledger_notes.md` / `docs/ledger_symbols.md` (the ledger),
  `docs/toy_model_report.md` (RQ3), `docs/values_prompt.md` (code style values).

## Project structure

```
scripts/
  collect.py            # Main collection: activations/covariances at hook leaves + inline metrics
  compute_metrics.py    # Unified metrics incl. ledger, crosses, gen_vs_ref (--ref); metric fns live here
  build_shuffled_mix.py # Pre-shuffled token mixes (data/mixes)
  activation_ratio.py, merge_*.py, migrate_*.py, profile_sweep.py   # utilities

configs/                # YAML for collect.py (--config); output_dir: data/inferences/<config_name>
  block_representations_samples[_padded|_swap].yaml  # the main sweeps (samples-mode, ledger)
  rq2_*.yaml            # RQ2 populations (general/math/quotes/memorized × packed/padded × seeds)
  block_rogue_id.yaml, block_write_id_olmo*.yaml     # raw-sample token-attribution runs
  reproduce_*.yaml, full*.yaml, kfac_merullo.yaml    # older reproduction configs

utils/
  model_registry.py     # ModelConfig, checkpoint discovery/loading, token counts
  hooks.py              # HookCollector (acts via fwd hooks; input-side grads via TENSOR hooks —
                        #   module backward hooks get empty grad_input on kwarg-called models)
  accessor.py           # DataAccessor: leaf × quantity × format resolver, View.cross, projections
  data_utils.py         # dataset registry (incl. rq2 datasets), packing/padding, text caches
  gpu.py, hook_names.py, hook_specs.py, revisions/

toy/                    # Li et al toy classifier + residual/architecture-knob variants;
                        # dumps in accessor format so compute_metrics runs on toys UNMODIFIED

analysis/               # jupytext-paired notebooks (edit the .py, `jupytext --sync`)
  showcase.ipynb        # one-figure-per-finding companion to the final report
  experiments.ipynb     # main grids (Exp 4.1–4.10) + padded twin experiments_padded.ipynb
  rq2_results.ipynb     # RQ2 depth profiles, split-half, whitened H2.1
  vocab_entropy.ipynb   # entropy-lens staging (nanochat; the collection flag lives in the worktree)
  experiments_lib.py    # get_ys/plot_group/virtual hooks (tail_rankme, alpha_window, sub_* ledger)

memorization_scripts/   # TriviaQA likelihood/infgram scripts (other session)
data/  mixes/ (pre-shuffled token tensors incl. cross-tokenized swap mixes)
       filtered_texts/ (text caches)  inferences/ → /projects/prjs1815/inferences  results/
slurm/ collect.sh (the submission wrapper — array over models), compute_metrics.sh, storage.sh
tests/                  # unit suite + e2e (GPU) — includes toy-format and resolver-cross tests
```

## Running

```bash
python scripts/collect.py --config configs/<cfg>.yaml --model_name <hf_name>   # single
./slurm/collect.sh configs/<cfg>.yaml --models "pythia-1b-deduped OLMo-2-0425-1B" --time 4:00:00
python scripts/compute_metrics.py <config_dir> <model_short> --num_workers 4
python scripts/compute_metrics.py <T_dir> <model> --recompute true --ref data/inferences/<G>/<model>/step<N>.pt   # gen_vs_ref
```

Config essentials: `model_name` scalar/list; vectorizable per-model fields (batch_size,
max_tokens, num_samples, dataset_name, drift_metrics, ...); `checkpoints: "final"` or list;
`collect_format` (acts|cov) decoupled from `storage_format`; `max_bytes` byte-budgeted text.
**`keep_cached`**: True = keep HF checkpoint cache (REQUIRED for concurrent same-model jobs —
a finishing job otherwise deletes the cache under the others); False = delete after use
(REQUIRED for multi-checkpoint sweeps — True hoards ~2–4 GB × checkpoints and blows the quota).

## Key concepts

### Hooks — a hook IS a leaf
`hooks:` lists `<leaf-pattern>[:acts|:grads|:both]` (default `:acts`) or `preset:<name>`.
Leaves: `blk{i}.mlp.{up,down,gate}.{in,out}`, `blk{i}.{attn,mlp}.{in,out,raw_out}`,
`blk{i}.attn.head{h}.slice`, `before/after_final_norm`. `:grads`/`:both` trigger a backward
pass (~2–3× forward VRAM). `fast_final_norm: true` only with all-`:acts`. Pythia exposes 3
boundary leaves per block (mlp.in aliases attn.in on read); OLMo-2 exposes the superset.
Input-side grads are the full dL/dh at the hookpoint (tensor hooks, all consumers).

### Data modes
packing packed (concatenated; `packed_data_path` mix or `dataset_name` streaming+pack) vs
padded (per-doc, `token_selection: last` for last-token geometry). `text_shuffle_seed`
de-biases streamed text (cache key includes it). `dataset_content_key: "a+b"` joins fields.

### Storage formats
acts | acts_svd | cov | cov_svd | eigenvalues (+`+b/+m/+o` modifiers). Samples-mode runs
persist tiny `eigenvalues` files; crosses/ledger are computed inline and are NOT recomputable
offline (samples aren't persisted). cov_svd is required for geneig work (needs eigvecs).

### nanochat family
Vendored (`utils/nanochat_gpt.py`), locally trained Karpathy-nanochat fork: 2 gateless MLP
projections, param-free RMSNorm, logit softcap. Checkpoints + tokenizer load from
`$NANOCHAT_DIR` (default `/projects/prjs1815/nanochat`), no HF hub; names `nanochat-<tag>`
(e.g. `nanochat-d12`); packed collection only (no padding-mask support).

### Vocabulary-entropy lens
`vocab_entropy: true` in a collect config (requires `compute_metrics`) rides the collection
forward passes and stores per-layer mean next-token-distribution entropy (`utils/entropy_lens.py`)
under node `vocab_entropy`, metric `entropy_lens`, in the results .npy. `configs/vocab_entropy.yaml`
runs it over the block_representations_all sweep; fold into existing results with
`scripts/merge_results.py`.

### DataAccessor
Address = `leaf.quantity.format` (formats: cov, eigvals, eigvecs, eigh, *_centered, mean,
samples, n); resolver derives greedily; `View.cross(partner)` gives in-memory cross-covariances
(pair-valid comps only), cached, never persisted. `acc.v.blk3.mlp.out.acts.eigvals_centered`,
`acc["before_final_norm"].acts.cov`. Storage CLI: `python -m utils.accessor info|convert|project`.

### Precision / weight cache
accumulation fp64, activations fp32 (bf16 fine for samples-mode). On-disk weight cache under
`data/weight_cache` feeds derived factors (B, O) without full snapshot downloads.

## Testing
```bash
pytest tests/ -m "not e2e"     # unit, login node OK
export HF_HOME="/projects/prjs1815/hf_cache"
srun --partition=gpu_a100 --gpus=1 --ntasks=1 --cpus-per-task=18 --time=00:40:00 \
    uv run pytest tests/ -m e2e
```
Standard checks after edits: `uv run --with pyright pyright <files>` and `uv run lint-imports`.

## Cluster notes
No compute on login nodes; `uv run`, `python` not `python3`. **Eigendecomposition-heavy work
(compute_metrics passes, geneig analyses) goes on a GPU node (gpu_h100 preferred) — never
staging CPU.** Staging is for I/O-light readouts of already-computed results files only. HF_HOME on /projects (scratch-shared is ~20× slower). VRAM estimator
underestimates backward-pass runs 2–3×. $TMPDIR on gcn nodes is RAM-backed tmpfs.

## Dependencies
torch, transformers, numpy, sklearn, datasets, tqdm, jsonargparse, matplotlib, seaborn
