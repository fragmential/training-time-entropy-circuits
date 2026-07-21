# Model init details

Measured from the earliest checkpoints (head spectra: `data/results/unembedding_spectra.pt`;
nanochat step-0 checkpoint; OLMo-7B step-150 weights via the selective loader).

| model               | embed / head init                          | head fro (earliest)      | head RankMe (earliest) |
|---------------------|--------------------------------------------|--------------------------|------------------------|
| pythia-160m/410m/1b | small-init, σ=√(2/5d)                      | 141.9 = √(0.4·V)         | 762 / 1014 / 2007      |
| pythia-6.9b         | plain normal, σ=0.02                       | 287.6 = 0.02·√(d·V)      | 3933                   |
| OLMo-2-1B           | truncated normal(0.02), 3σ cutoff          | 283.0 = 0.9865·0.02·√(d·V) | 2027 / 2048          |
| OLMo-2-7B (step150) | trunc-normal(0.02) marginals, rank ~380    | 400.1                    | 374 of 4096            |
| nanochat-d12        | wte N(0,1); all c_proj + lm_head zero      | 0                        | 1                      |

- **nanochat-d12**: init by construction — in the d12 step-0 checkpoint, every write projection
  is exactly zero (all 12 blocks' `attn.c_proj` and `mlp.c_proj` have std 0.0) and `wte` is
  drawn with std ≈ 1.0. At step 0 the residual stream is the (RMS-normed) embedding, writes
  grow from zero, and the embedding keeps its lead early because its learning rate is 10× the
  matrix LR (0.2 vs 0.02 in the run config). The `lm_head` is zero-init too (its unembedding
  RankMe is literally 1.0 at step 0).

- **Pythia 160m/410m/1b**: head Frobenius norm is 141.8/141.9/141.9 — exactly √(0.4·V), i.e.
  the GPT-NeoX "small init" σ=√(2/5d), which makes ‖W‖_F width-independent. Full rank
  (RankMe ≈ 0.99·d).

- **Pythia 6.9b**: fro 287.6 = 0.02·√(d·V) — plain σ=0.02, a different scheme than its smaller
  siblings. Full rank.

- **OLMo-2 1B**: step-0 head matches truncated normal(0.02) with 3σ cutoff to three digits
  (fro 283.0 vs predicted 282.9). Full rank (2027/2048).

(Only heads were collected in `unembedding_spectra.pt`; embeddings of Pythia/OLMo-1B weren't
checked.)

**OLMo-2 7B: the low early unembedding rank is real, and the init is not what's documented.**
The earliest public checkpoint is step 150 (no step 0 exists). At step 150 the head is
numerically rank ≈ 380 of 4096: a smoothly decaying top spectrum (26.8 → 15.1 over ~380
directions), then a cliff to bf16 noise, 99.999% of energy in the top 400. Yet its marginal
statistics are exactly truncated-normal(0.02): row norms 1.263 ± 1% (the precise χ(4096)
width, with the 0.9865 truncation factor), fro 400.1, coordinate kurtosis 2.83, hard cutoff
at 3.1σ. The embedding has the identical structure (independent draw — the two matrices are
uncorrelated, cos ≈ 0). This cannot be 150 steps of training on an iid init (LR is still
~1/13 of peak mid-warmup; updates are ~10⁻³ of coordinate scale), and it contradicts the
paper/config's stated plain truncated-normal init. It's also mathematically impossible for
rows genuinely Gaussian inside a ~380-dim subspace to have norms that tight — the rows have
full-dimensional-looking norms but confined directions, i.e. something in their init stack
produced norm-preserved, direction-collapsed matrices (deliberate trick or sharding/RNG
artifact; not determinable from here). Consistent with this, the head rank rises early
(374 → 421 by step 850) as training fills in directions, before the familiar long-run
collapse (187 at the end).
