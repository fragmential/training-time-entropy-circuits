# Entropy neurons and the final-layer handover

Thread from Stolfo, Wu, Gurnee, Belinkov, Song, Sachan and Nanda, *Confidence Regulation
Neurons in Language Models* (arXiv:2406.16254). An **entropy neuron** writes into the effective
null space of the unembedding — the span `V0` of its bottom-k right singular vectors — so it
moves the residual norm, and through the final norm the logit temperature, without moving the
logits. Their statistic, with a high weight norm and low LogitVar:

    rho_i = ||V0^T w_out^(i)|| / ||w_out^(i)||,     k ~ 0.01 d_model

They never ask *when* these form. That is what this thread measures.

## Findings

1. **rho has a sharp onset in every model** — flat at the random level, then one decade of
   rise to saturation. nanochat has no LR warmup and shows it too, so it is not a warmup
   artifact. Onsets: nanochat 1.3e8, Pythia 2-4e9 (all scales), OLMo 1e10.
2. **The transition is `W_U` growing a null space, not neurons finding one.** Before it the
   bottom-k subspace sits at its initialisation (overlap with final = chance) and the spectral
   tail is flat; at 1-2e9 tokens it rotates almost completely between checkpoints (0.22), then
   locks in as the bottom singular values separate. Same timing in all four Pythias.
3. **The population is mixed, not clean entropy neurons.** rho>0.5 neurons have LogitVar
   0.67-0.77x the layer median (1.38x in 160m) and ordinary weight norms. All three of the
   paper's criteria at once leaves 2-57 neurons. Their one named Pythia neuron (23.417) is a
   *token frequency* neuron and reproduces only in the non-deduped model — indices do not
   transfer across runs.
4. **Ablation does not drive the rank collapse.** Removing the top-256 by rho takes out 77% of
   the last MLP's output variance but moves RankMe by -3.4%, the *wrong* way (compression
   would predict a rise), while output entropy rises 26.5%. Matched random controls: 0.0%.
5. **The direction the last block adds is the top direction of the final stream** —
   cos(gain, afn) = 0.89, 8-10% of stream variance, null share 0.84 vs 0.099 chance, stable
   across data halves. **Open contradiction with (4)**: likely rho-ranking does not select the
   neurons that write it (contribution ranks 6016 at blk15, 7446 at blk14).
6. **Not a sink** — spread over 24-53% of tokens. `gain` is carried by discourse connectives
   (And/But/Although/Maybe), `afn` by sentence terminators. Both are where prediction gets hard.

Paper gaps: no timing, no baseline for rho, no k sensitivity, almost nothing on activations,
and **it never asks whether `W_U` itself encodes frequency** — which is where the handover
theory sits. LayerNorm mediation is 30-40% in Pythia vs ~80% in GPT-2.

## Open

- **Handover theory**: `W_U` stops doing frequency bias and switches to reducing interference
  while the final layers take over, because equinormness under neural collapse
  (arXiv:2405.17767) is incompatible with a frequency-biased unembedding. → `collapse_geometry.py`
- Which neurons *grow* their entropy effect between 1e10 and 1e12. → `entropy_effect.py`
- Ablate by contribution to the top direction rather than by rho.
- Do Pythia/OLMo have the spectral gap the paper's k-selection assumes?
- The pre-norm mean direction is excluded everywhere (covariances are centred); it may be more
  null-space-aligned than the top variance direction.

## Scripts (all in `oneoff_scripts/`, output to `data/results/`)

| script | what | output |
| --- | --- | --- |
| `entropy_neurons.py` | rho / weight norm / LogitVar per neuron per checkpoint | `entropy_neurons{,_paper,_nondedup}.pt` |
| `nullspace_stability.py` | is the bottom-k subspace a stable target? | `nullspace_stability.pt` |
| `entropy_neuron_activations.py` | per-neuron activation mean/RMS/frac-zero | `neuron_write_stats.pt` |
| `grown_direction_neurons.py` | the direction each block adds; null/top shares, magnitude, split-half, token concentration, predictions, writing neurons | `grown_direction_neurons.pt` |
| `entropy_effect.py` | per-neuron dH/da by autograd, all neurons all checkpoints | `entropy_effect.pt` |
| `collapse_geometry.py` | frequency handover, cone/anisotropy (uncentred), NC1/NC2/NC3 (centred) | `collapse_geometry.pt` |
| `analysis/entropy_neurons.py` | the notebook | figures |

Ablations run through `collect.py`'s `ablate_neurons` (`{block, rank, n}`, negative indices,
rank in rho/wnorm/random, pooled when `block` is a list); configs `configs/entropy_ablate*`.
rho lives in `utils/nullspace.py`, shared by the sweep and the ablation selection.

## Runs

| job | what | state |
| --- | --- | --- |
| 25068489 | two-block ablation, 25 interventions x 28-61 ckpts, 4 models | running |
| 25068735 | last-quarter pooled ablation | running |
| 25070541 | grown-direction + entropy effects, 4 OLMo checkpoints | queued |
| 25070983 | collapse geometry, OLMo-1B / pythia-1b / nanochat | queued |
