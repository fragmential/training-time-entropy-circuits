


"""Do not use this"""



# """The warm-up alignment tables, as LaTeX: one for the body, two for the appendix.

# One case per ablated group of layers. Ablating layers 5--8 hands layer 9 the stream that layer 5
# received in the baseline, so layer 5 as it is in the baseline is what layer 9 is compared
# against. Three Delta S components of Section 3.6 are read off each: the update entropy, the
# overlap, and the interference.

#   --body  the two differences per component, one row per case
#   --full  every value and difference, one row-section per component

# beta itself is not reported: the update carries a few percent of the trace, so w scales every
# gap to about a hundredth and hides the movement. The unweighted entropies show it.

# S_u is the trace-weighted entropy of the block's updates, sum_i>0 w_i S_i / sum_i>0 w_i, which
# is the S_u of eq:beta when attn and mlp are pooled: beta = W (S_u - S_in), W = sum_i>0 w_i.

# Reads data/results/{block_representations_samples,nanochat_samples,ablate_*}.
# Writes to stdout, and every measured number, shown or not, to DUMP. The tables in the
# thesis are hand-maintained: what is worth taking from here is the numbers, not the LaTeX.
# CPU, seconds.
# """
# import ast
# import json
# import os
# import sys

# import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# NB = "analysis/standalone_figures.py"
# DUMP = "data/results/warmup_alignment.json"
# LABELS = {"body": "tab:warmup_alignment", "full": "tab:apx:warmup_alignment_full"}
# # (results dir, first ablated block, measured block); 0-indexed blocks, 1-indexed in the tables
# ABLATIONS = {"pythia-1b-deduped": [("ablate_blk4-7", 4, 8), ("ablate_blk8-11", 8, 12)],
#              "OLMo-2-1124-7B": [("ablate_blk8-15", 8, 16), ("ablate_blk16-23", 16, 24)],
#              "nanochat-d12": [("ablate_blk3-5", 3, 6), ("ablate_blk6-8", 6, 9)]}
# # rankme_phases puts the trough at the middle of the first run below 10% of d_model. For a
# # 4096-wide model that band is 410 wide, which swallows 7B's whole climb out of the crash and
# # lands the trough at 3e9, where the depth profile has already flattened. Its trough is the
# # first checkpoint it records.
# FIRST_CKPT = ("OLMo-2-1124-7B",)
# # (key, heading, scale, decimals); overlap runs two orders below the entropies, so it is scaled
# # rather than given more decimals
# TERMS = (("s_u", r"$S_{\mathbf u_\ell}$", 1, 2),
#          ("chi", r"$\chi_\ell \times 10^2$", 100, 1),
#          ("interference", r"$I_\ell$", 1, 2))
# STUB = "Placeholder. Both tables are hand-maintained in the thesis; this is the "\
#        "measurement, not their text."
# NOTE_OLMO = ("For the middle layers that we have ablation data for, OLMo~2~1B's input entropy "
#              "to the ablated layers is already near to the floor, so the ablated layers barely "
#              "affect entropy ($S_4 - S_8 = 0.10$). For this reason, we measure OLMo~2~7B, which "
#              "starts training with higher-entropy early layers.")
# NOTE_SU = ("We take $S_{\\mathbf u_\\ell}$ rather than $\\beta_\\ell$ because "
#            "$\\beta_\\ell$ is weighted by relative output trace and more importantly is taken "
#            "relative to its input. Here we want to compare the change in output based on input, "
#            "so taking $\\beta_\\ell$ would normalise by what we intend to measure.")
# MARK = "OLMo-2-1124-7B"        # the model whose name carries the OLMo note


# def notebook_globals(stop="ledger_bars"):
#     """The notebook's definitions, without running its figure cells."""
#     import matplotlib
#     matplotlib.use("Agg")
#     tree = ast.parse(open(NB).read(), NB)
#     end = max(i for i, st in enumerate(tree.body)
#               if isinstance(st, ast.FunctionDef) and st.name == stop)
#     g = {"__name__": "__main__", "__file__": NB}
#     for st in tree.body[:end + 1]:
#         if isinstance(st, ast.Expr) or (isinstance(st, ast.Assign) and len(st.targets) == 1
#                                         and isinstance(st.targets[0], ast.Name)
#                                         and st.targets[0].id == "_"):
#             continue
#         exec(compile(ast.Module(body=[st], type_ignores=[]), NB, "exec"), g)
#     return g


# def terms(g, model, src, blk, tok):
#     """Every component of one block at the checkpoint nearest tok."""
#     out = {}
#     for q in ("w", "s", "chi", "quality", "interference", "delta_s"):
#         ys, steps = g["_lib"].get_ys(src, model, (f"blk{blk}", "block_ledger"), q)
#         xs = np.asarray(g["_lib"].get_xs_tokens(model, steps), float)
#         out[q] = np.asarray(ys[int(np.argmin(np.abs(np.log(np.maximum(xs, 1)) - np.log(tok))))],
#                             float)
#     w, s = out["w"], out["s"]
#     return dict(s_in=float(s[0]), s_u=float((w[1:] * s[1:]).sum() / w[1:].sum()),
#                 chi=float(out["chi"]), beta=float(out["quality"]),
#                 interference=float(out["interference"]), delta_s=float(out["delta_s"]),
#                 w_update=float(w[1:].sum()))


# def measure(g):
#     rows = []
#     for model, ablations in ABLATIONS.items():
#         base = g["cfg_of"](model)
#         xs, _ = g["_lib"].rankme_series(base, model, g["HOOK"], include_init=False)
#         tok = xs[0] if model in FIRST_CKPT else \
#             xs[g["_lib"].rankme_phases(base, model, g["HOOK"], include_init=False)[0]]
#         for src, first, meas in ablations:
#             rows.append(dict(model=model, label=g["model_label"](model), tokens=float(tok),
#                              ablated=f"{first + 1}--{meas}", measured=meas + 1,
#                              base=terms(g, model, base, meas, tok),
#                              abl=terms(g, model, src, meas, tok),
#                              first=terms(g, model, base, first, tok)))
#     return rows


# def num(v, d, signed=True, bold=False):
#     """Math mode throughout: a signed number in text mode gets a hyphen rather than a minus."""
#     t = f"{v:+.{d}f}" if signed else f"{v:.{d}f}"
#     return f"$\\mathbf{{{t}}}$" if bold else f"${t}$"


# def cells(row, key, scale, d):
#     """(baseline, ablated, first ablated, difference to each), formatted, differences bolded."""
#     b, a, f = (row[w][key] * scale for w in ("base", "abl", "first"))
#     db, df = a - b, a - f
#     return ([num(v, d, signed=False) for v in (b, a, f)]
#             + [num(db, d, bold=abs(db) < abs(df)), num(df, d, bold=abs(df) <= abs(db))])


# def model_col(rows, i, mark=""):
#     """A model's name, once per model, with its note mark on the first row that carries it."""
#     if i and rows[i - 1]["model"] == rows[i]["model"]:
#         return ""
#     return rows[i]["label"] + (mark if rows[i]["model"] == MARK else "")


# def wrap(body, label, caption, notes, colsep=5):
#     """notes: (mark, text) pairs, lettered in the order the marks are read."""
#     out = ["\\begin{table}[ht]", "\\centering", "\\begin{threeparttable}", "\\small",
#            f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}", f"\\caption{{{caption}}}",
#            f"\\label{{{label}}}"] + body + ["\\end{tabular}"]
#     out += ["\\begin{tablenotes}[flushleft]\\footnotesize"]
#     out += [f"  \\item [{m}] {t}" for m, t in notes]
#     out += ["\\end{tablenotes}"]
#     return "\n".join(out + ["\\end{threeparttable}", "\\end{table}"])


# def emit_body(rows):
#     # the label columns need air the numeric ones do not: tabcolsep is set tight for those
#     body = ["\\begin{tabular}{@{}l c@{\\hspace{0.7em}}c@{\\hspace{1em}} "
#             + " ".join(["rr"] * len(TERMS)) + "@{}}", "\\toprule",
#             "& & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{h}{'\\tnote{a}' if i == 0 else ''}}}"
#                                   for i, (_k, h, _s, _d) in enumerate(TERMS)) + " \\\\",
#             "".join(f"\\cmidrule({'lr' if i < len(TERMS) - 1 else 'l'})"
#                     f"{{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(TERMS))),
#             "Model & Ablated & Measured & " + " & ".join(["baseline & first ablated"] * len(TERMS))
#             + " \\\\", "\\midrule"]
#     for i, r in enumerate(rows):
#         vals = [c for k, _h, s, d in TERMS for c in cells(r, k, s, d)[3:]]
#         body.append(f"{model_col(rows, i, chr(92) + 'tnote{b}')} & {r['ablated']} & "
#                     f"{r['measured']} & " + " & ".join(vals) + " \\\\")
#     body.append("\\bottomrule")
#     return wrap(body, LABELS["body"], STUB, [("a", NOTE_SU), ("b", NOTE_OLMO)],
#                 colsep=2)


# def emit_full(rows):
#     body = ["\\begin{tabular}{@{}lcc rrr rr@{}}", "\\toprule",
#             "& & & \\multicolumn{3}{c}{Value} & \\multicolumn{2}{c}{Difference to} \\\\",
#             "\\cmidrule(lr){4-6}\\cmidrule(l){7-8}",
#             "Model & Ablated & Measured & baseline & ablated & first ablated & baseline & "
#             "first ablated \\\\"]
#     for key, heading, scale, d in TERMS:
#         body += ["\\midrule", f"\\multicolumn{{8}}{{@{{}}l}}{{\\textit{{{heading}}}}} \\\\"]
#         for i, r in enumerate(rows):
#             mark = chr(92) + "tnote{a}" if key == TERMS[0][0] else ""
#             body.append(f"{model_col(rows, i, mark)} & {r['ablated']} & {r['measured']} & "
#                         + " & ".join(cells(r, key, scale, d)) + " \\\\")
#     body.append("\\bottomrule")
#     return wrap(body, LABELS["full"], STUB, [("a", NOTE_OLMO)])



# if __name__ == "__main__":
#     G = notebook_globals()
#     ROWS = measure(G)
#     os.makedirs(os.path.dirname(DUMP), exist_ok=True)
#     with open(DUMP, "w") as fh:
#         json.dump(ROWS, fh, indent=1)
#     want = [a.lstrip("-") for a in sys.argv[1:]] or ["body", "full"]
#     print("\n\n".join({"body": emit_body, "full": emit_full}[w](ROWS) for w in want))
#     print(f"% every measured number, shown or not: {DUMP}", file=sys.stderr)



"""Do not use this"""