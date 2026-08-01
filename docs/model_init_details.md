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

## Note: OLMo's blk0 cliff and the depth climb are the write-norm clamp

OLMo-2's depth profile in early/mid training — high at the embedding, cliff at blk1, monotone
climb through the early layers to a mid-stack maximum — is the reordered norm. RMSNorm on the
sublayer output pins every write to ‖w‖² ≈ d, which is *enormous* next to OLMo's embedding,
not small. Measured on OLMo-2-1B, `block_representations_samples` (uncentered trace = E‖·‖²,
d = 2048):

| step | tokens | embedding (blk0.attn.in) trace | blk0.attn.out trace | blk0.mlp.out trace |
|---|---|---|---|---|
| 2000 | 5B | **0.83** | **1.94e3** | **1.95e3** |
| 7000 | 15B | 1.67 | 1.01e3 | 1.30e3 |
| 100000 | 210B | 10.2 | 2.12 | 1.23 |
| 1900000 | 3985B | 29.6 | 7.96 | 6.84 |

- At step 2000 *every* write in *every* block has trace 1.94–1.96e3 = d·(gain≈1) — the
  signature of RMS-1 output. The stream entering block 0 has trace 0.83. Write:stream energy
  at block 0 is ~2400:1.

- So block 0 does not perturb the embedding's geometry, it **replaces** it. Block 0's write has
  centered RankMe ~70 (attn) / ~200 (mlp) at that point, and that — not the embedding's ~700 —
  is what `blk1.attn.in` measures:

  ```
  step 2000 (5B), centered RankMe by depth:
  648(emb) → 152 → 201 → 250 → 289 → 356 → 408 → 469 → 519 → 564 → 599 → 651 → 654 → 648 → 619 → 550 → 499
  ```

- Why the early layers push it back up: the clamp applies to all blocks equally, so no block
  can outweigh another. After k blocks the stream is a near-equal-weight sum of 2k writes in
  different subspaces, and effective rank grows roughly with the number of accumulated writes.
  The embedding contributes essentially nothing to the climb — it was drowned at block 0.

- Same object as the no-valley result (dig_findings.md): equal-magnitude writes make a rogue
  write architecturally impossible. Pythia-1b final, for contrast: `blk3.mlp.out` trace 1.42e4
  with centered RankMe **1.2**, against a stream of trace 1.4e3 — one rank-1 write carrying 10×
  the whole stream. Nothing under write-norm can do that.

- It disappears late in training, and what replaces it is a different phenomenon. The learned
  norm gains shrink while the embedding grows (0.83 → 29.6); by step 100k the block-0 write is
  smaller than the stream and the profile flattens:

  ```
  step 1.9M (3985B), centered:   1345 → 1400 → 1393 → 1357 → 1265 → ... → 975 → 740 → 447
  step 1.9M (3985B), uncentered: 1127 →  778 →  528 →  376 →  292 → ... → 618 → 491 → 267
  ```

  The centered cliff at blk1 is gone at the end; what remains is purely an **uncentered**
  decline, i.e. the mean vector growing (‖μ‖ 1.24 → 19.4 down the stack), not a variance
  collapse. Early-training rank *replacement* by a giant normed write and late-training mean
  accumulation are different events — plots must state which of the two they show.

- Why the other families don't do this. Pythia: pre-norm, no clamp on the write. Block 0 still
  swamps the embedding energetically (0.96 → 541 at step 143000 / 300B), but the write is
  high-rank plus a large mean, so uncentered crashes (618 → 57) while centered barely moves
  (888 → 746) — no cliff, no depth climb, and one write (blk3) eventually owns everything.
  nanochat-d12: zero-init `c_proj` (above), so at step 0 the stream *is* the embedding at every
  depth and there is nothing to kick it down until the writes grow in.

- Open, testable: the toy `writenorm` knob (toy.md) should reproduce the blk1 cliff + monotone
  depth climb from norm placement alone, with a scale-matched embedding as control.

## OLMo-2 7B

**The low early unembedding rank is real, and the init is not what's documented.**
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
