# DataAccessor — desired interface (scaffold)

> **Temporary.** Formalizes the *desired* public surface to lock before touching internals.
> The durable form is this interface **expressed in code** — typed `Protocol`s / ABCs + full
> type hints — which is also the enforcement. Delete this `.md` once that exists.
> Source of intent: `docs/values_prompt.md`.

## Principles this encodes
- **Tree is the only read path.** Reading goes through `.v` / `acc[path]` → `Node` → `View`. `resolve`, `view`, `leaves`, `leaf_quantities`, `tree_children`, `to_dict`, `cached_references` are **private** (the daughter classes use them).
- **Resolution is fmt-level.** Any format may be `None`, and a `View` may be **unbacked** (no accessor) — both are normal, not errors. A View is a *named quantity*, not a guarantee of data.
- **Components are finite & inherent.** A quantity's formats are a fixed set. A **projection** is not a component — it's a `reference` coordinate on a View, served by the *one* resolver.
- **Capability narrowing.** The accessor depends on a minimal `WeightProvider`; it never sees a model *name to resolve* or a model *to introspect*. Identity labels + metadata are written by the producer, never reverse-engineered from the file path.
- **Types are the interface.**

## DataAccessor — public surface
| Member | Signature | Contract |
|---|---|---|
| construct | `DataAccessor(data: dict \| str, *, weights: WeightProvider \| None = None, config: ModelConfig \| None = None, derive: bool = True)` | `data` is a dict or a path. No `model_name`. Identity (model/revision/run) is read from metadata. `derive=False` ⇒ weight-derived quantities resolve to `None` (never loads). |
| `save` | `save(path: str, format: str \| None = None, *, storage_dtype=None, token_filter: dict \| None = None, n_chunks: int \| None = None, persist_projections: bool = False) -> str` | `format=None` = **preserve** (current data + corrected metadata). Computed projections live in the resolver cache; `persist_projections=True` writes the cached ones into `__projections__` (opt-in). |
| read entry | `v -> Node` ; `__getitem__(path: str) -> Node` | Only public read entry. |
| `prewarm` | `prewarm(fmts: list[str] \| None = None) -> None` | `None` warms everything, else the listed formats. **Owns the GPU context internally** (no external `_gpu_ctx`). |
| `needs_model_weights` | `() -> bool` | Any required derived `acts` needs weights not already stored. |
| `info` | `() -> str` | Human-readable summary. |

**Private** (`_`-prefixed; consumers migrate to the tree): `_resolve(leaf, q, fmt, reference=None)` — the one resolver, now reference-aware; `_view`, `_leaves`, `_leaf_quantities`, `_tree_children`, `_to_dict`. `View.projections()` inlines the `__projections__` scan for now (extract a helper later only if a second caller appears).

## Node — a path cursor
| Member | Signature | Contract |
|---|---|---|
| descend | `node[seg] -> Node` ; `node.<seg> -> Node` | Pure path navigation. |
| quantity | `node.acts -> View` ; `node.grads -> View` | **Strict**: raises if unresolvable (the human path; catches typos). |
| `children` | `() -> list[Node]` | Immediate children among present+derivable leaves. |
| `get` | `get(path: str) -> View \| None` | **Soft**: dotted relative path ending in a quantity (`"in.acts"`) → `View`/`None`. The metric engine's absence-tolerant primitive. No `enforce_type` flag — strictness lives on the attribute path. |

## View — one (leaf, quantity), optionally projected
```python
Identity = namedtuple("Identity", "model revision run")   # in-memory; assembled from
                                                          # separate metadata keys (below)
class View:
    def __init__(self, acc: "DataAccessor | None", identity: Identity,
                 leaf: str, q: str, reference: "View | None" = None) -> None: ...
    def __getattr__(self, fmt: str): ...        # acc.resolve(leaf, q, fmt, reference); None if unbacked
    def in_basis(self, reference: "View") -> "View": ...
    def projections(self) -> "list[View]": ...
```
| Member | Contract |
|---|---|
| `view.<format>` | Finite set: `cov, eigvals, eigvecs, eigh, eigvals_centered, eigvecs_centered, mean, samples, svd, n` → `Tensor`/`tuple`/`None`. Unbacked (`acc is None`) → always `None`. |
| `in_basis(reference: View) -> View` | This quantity in `reference`'s basis. **Sole** projection door; cache-aware/memoized via the resolver. `reference` data-backed ⇒ may compute; unbacked ⇒ only a cached hit resolves (else `None`). Local & cross-checkpoint are the same call. |
| `projections() -> list[View]` | The cached projections of this quantity, each a **readable** projected View (their `reference` is an unbacked identity View from cache → no reference files loaded). |
| `.identity -> Identity` ; `.reference -> View \| None` | Own identity; the basis it's projected onto (`None` for a plain quantity). |

A projection is just `resolve(leaf, q, fmt, reference=…)` — one resolver. On miss it computes `cross_eigvals(resolve(leaf,q,'cov'), reference.eigvecs)`; `reference.eigvecs` is `None` for an unbacked reference, so an uncached projection onto an unloaded checkpoint yields `None`. No new class, no second path. Cache key = `(leaf, q, fmt, reference.identity, ref_leaf, ref_q)`.

## Supporting types
- **`WeightProvider`** (Protocol): `weight(sd_prefix: str) -> LinearLike | None`, `final_norm() -> Module | None`. The minimal surface for derivation — no model-name / HF logic.
- **`Identity`** = `(model, revision, run)`, assembled **in-memory** from the **separate** metadata keys `__hf_model__` / `__revision__` / `__run__` — *not* stored fused (don't lose one by tupling). `run` = run/output-dir name; **no config hash** (so compatible runs — e.g. filling a sparse checkpoint schedule — stay mergeable).

## Internals — direction only (deferred; "first formalization")
- **Storage = pure identity** `{q}_{component}`. Transforms move to `FORMAT_CONVERSIONS`: normalized `cov ← (gram, n)`; `svd` reuses `eigvecs`(=V) + `√(eigvals·n)`(=S), only `U` distinct. Raw covariance stored as `gram` (+`n`). Net: removal.
- **Projections** live in a separate `__projections__` namespace keyed by reference identity (`Identity` + ref leaf/q), in the source file; never in the flat component keys. `resolve(..., reference)` checks it first.
- **Identity stamped at collection** — `collect.py` writes `__hf_model__`/`__revision__`/`__run__`. This **removes `_stamp_from_path`** (no filename reverse-engineering) and `_resolve_hf_name` (the model label is metadata; *loading* is the `WeightProvider`'s job).
- **`decomp_profiler`** stays active as a profiling side-channel — **never written into results**. The `gpu_wait` field and external `_gpu_ctx` monkey-patch are dropped (prewarm owns the context).

## Resolved
`reference` field name = `reference`. `save(persist_projections: bool=False)`. `projections()` on `View`, inlined (no helper yet). `run` = output-dir name.

## Open / TBD at implementation
- `WeightProvider` exact methods; how hard-abort-on-load folds in.
- pyright scope (start at `utils/accessor.py` + consumers, widen later).
