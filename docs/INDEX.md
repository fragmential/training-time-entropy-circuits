# Docs index — where everything lives

Four layers, one job each:

- **registry** (research_questions.md) — every question and hypothesis, one-line verdict,
  pointer to the evidence. Start here to find anything.

- **synthesis** (final_report.md) — the whole story in one file, current belief only;
  superseded readings live in its corrections-record appendix.

- **plan** (plan.md) — the only forward-looking doc.

- **topic docs** — detailed evidence and methods; the thesis' appendix material.

Figures: every headline finding has a showcase figure (analysis/showcase.ipynb, organised
Question → Finding → evidence per section); appendix-grade figures in
analysis/showcase_appendix.ipynb; every multi-layer toy ablation in
analysis/toy_multilayer.ipynb.

## The documents

| doc | kind | contents |
|---|---|---|
| research_questions.md | registry | RQ1/RQ1b/RQ2/RQ3 + hypotheses, verdicts, pointers |
| final_report.md | synthesis | executive summary; RQ1/RQ3 results; RQ2 parked status; caveats; open questions; corrections record |
| plan.md | plan | in-flight jobs, ranked next steps, deferred decisions |
| dig_findings.md | evidence (RQ1) | the samples-sweep evidence: head/bulk, sink-write identification (two slots), interventions, valleys, dataset-swap controls, metric→finding table |
| sink_literature.md | literature | sink-paper summaries mapped to our results (+ the original deposit-test negative, filed there; figure: showcase_appendix §F) |
| ablations.md | evidence | robustness/ablation table (what was checked, how) |
| toy.md | evidence (RQ3) | the whole toy arc: single-layer reproduction + controls (§2), fourth condition / kick / lag / transience (§3), depth pattern conditions + ablation table (§4), decomposition signature (§5), architecture knobs (§6) |
| ledger.md | methods | the decomposition: derivation, symbols, term meanings, exactness caveats, alignment-metric taxonomy |
| rq2.md | methods (RQ2) | geneig/excess-mass methods, split-half estimator (v1 flaw + v2 + contamination audit), population inventory — the redo's blueprint |
| collection/ | archive-adjacent | ops/process docs kept out of the main flow: architecture.md (codebase), model_architectures.txt, next_run_additions.md (next-sweep metric additions), values_prompt.md (code-values charter) |
| archive/ | archive | pre-RQ-era scratch (profiling, old TODOs) |

## The notebooks

| notebook | contents |
|---|---|
| showcase.ipynb | the headline figures: §1 reproduction, §2 spectral locality, §3 the decomposition, §4 Pythia's sink write, §5 OLMo-2's aligned interference, §6 the minimal model (6.1 single-layer, 6.2 depth conditions, 6.3 decomposition signature, 6.4 architecture knobs), §7 RQ2 stub (parked), §8 side findings + compression valleys |
| showcase_appendix.ipynb | §A dataset swap, §B single-layer toy edge cases, §C per-depth uncentered-vs-centered (pattern runs + the 32-class artifact), §D valleys packed vs padded, §E sample-count validity, §F sink-variance deposit test |
| toy_multilayer.ipynb | every experiment behind toy.md §4 (the depth ablations) |
| experiments.ipynb / experiments_padded.ipynb | full per-model grids (Exp 4.x) |
| rq2_results.ipynb | RQ2 machinery (parked with RQ2) |
| vocab_entropy.ipynb | entropy-lens staging |
