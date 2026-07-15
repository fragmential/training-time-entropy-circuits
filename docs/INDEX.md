# Docs index — where everything lives

Three layers: **report** (headline results), **plan** (forward-looking, the only such doc),
**topic docs** (detailed evidence; the thesis' appendix material). Figures: every headline
finding has a showcase figure (analysis/showcase.ipynb); appendix-grade figures live in
analysis/showcase_appendix.ipynb.

## The documents

| doc | kind | contents |
|---|---|---|
| final_report.md | report | executive summary; RQ1/2/3 results + hypothesis scoreboard; caveats; open questions; appendix (side findings) |
| plan.md | plan | in-flight jobs, ranked next steps, deferred decisions |
| research_questions.md | registry | RQ1–RQ3 + hypotheses with decision criteria and status |
| dig_findings.md | evidence (RQ1) | the samples-sweep findings log: head/bulk, rogue identification (sink slots), interventions, valleys, metric→finding table |
| rq2_methods.md | methods+evidence (RQ2) | populations, excess-mass, split-half design and numbers |
| splithalf_note.md | methods (RQ2) | split-half estimator details |
| swap_run_findings.md | evidence | dataset-swap control: full multi-metric verdict table |
| toy_model_report.md | evidence (RQ3) | toy reproduction + residual/knob variants, H3.x verdicts |
| toy_fig4_addendum.md | evidence (RQ3) | the fourth init condition; covariance-collapse mechanism; lag; transience proof sketch |
| toy_multilayer.md | evidence (RQ3) | why multi-layer toys lacked the phases + the fix: dS/dt = −Cov_p(g, log p), print conditions (timing/kick/transmission), full ablation table |
| sink_literature.md | literature | Sun et al / spike-sparse-sink / sinks-valleys summaries, claims tagged per paper w/ file:line; our position |
| rank_ledger_notes.md, ledger_symbols.md | methods | the ledger derivation and symbol glossary |
| ablations.md | evidence | robustness/ablation table (what was checked, how) |
| next_run_additions.md | plan (annex) | metric additions/fixes for the next collection sweep |
| architecture.md, values_prompt.md | meta | codebase architecture; code-style values |

## The major findings (finding → doc section → figure)

| finding | doc | figure |
|---|---|---|
| Li et al phases reproduce (both families, both data modes) | final_report §2.1 | showcase §1 |
| Compression is a head event; the deep spectrum keeps expanding | dig_findings "head phenomenon" | showcase §2 |
| Ledger split: Pythia quality-carried, OLMo-2 interference-carried | dig_findings headline; final_report §2.2 | showcase §3 |
| The rogue write is a sink direction (two slots: window start + first newline) | dig_findings "rogue direction identified" | showcase §4 |
| Rogue direction is not inert at the output; its per-occurrence content is read at the next position (H1.5) | dig_findings "interventions" | — (text) |
| OLMo-2: distributed aligned late writes carry the interference | final_report §2.2 | showcase §5 |
| OLMo-2's norm package selects the mechanism; nanochat (QK-norm only) shows write-norm is the operative lever (H1.4) | final_report §0.3; sink_literature "norm placement" | showcase §6 arch grid |
| Dataset-swap: mechanism follows model, not data | swap_run_findings.md | showcase_appendix A |
| Toy compression needs the paper's unstated init; decline is a fading covariance event | toy_fig4_addendum.md (mechanism; transience proof sketch in its appendix) | showcase §6 + appendix B (B5 = MSE-starvation check) + appendix C (depth × centering) |
| The phase curve passes through deep stacks (residual AND identity-init plain) iff event-after-saturation + near-identity transmission | toy_multilayer.md | showcase §6 "multi-layer resolution" |
| Valleys emerge AT the RankMe peak, with the rogue collapse; OLMo-2 has none | dig_findings "compression valleys" | showcase §8 |
| RQ2 (ALL of it): ⚠️ UNDER REVISION — documented aggregates mixed acts+grads; verdicts suspect, parked | final_report §4 banner; plan.md item 1 | figures not to be trusted until redo |
| Vocab-entropy: old flat final layer was a softcap-omission artifact | vocab_entropy.ipynb header | vocab_entropy figures |
