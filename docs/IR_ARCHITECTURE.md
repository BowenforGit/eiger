# GaMax IR / Lowering / Rules / Execution — Architectural Reference

This doc is the architectural reference for the query-processing
stack introduced between commits `a2209a4` ("Add IR foundations")
and `89e7164` ("Add Concat IR/phys + Q13 outer-join-by-
decomposition"). It documents what every layer is, why it exists,
what invariants hold, and where the boundary between layers sits.
Readers who want the *story* of how the stack grew — design
decisions, corners cut, footguns — should pair this with
`docs/TPCH_JOURNEY.md`.

## Pipeline at a glance

```
  SQL (future)
       │
       │  frontend — not part of GaMax today; builders hand-write
       │  PlanNode DAGs directly in `src/cuda/tpch/tpch_ir.cu`.
       ▼
  IR (PlanNode DAG)                     ← declarative, rule-rewritable,
       │                                  ColumnId-keyed.
       │  apply_rules(plan, RuleConfig) ← semantic / structural rewrites.
       ▼
  IR' (rewritten)
       │
       │  lower_to_exec_plan(plan, cat, LowerOptions)
       │                                ← all host-side prep:
       │                                  ColumnId→ordinal resolution,
       │                                  Predicate / ExprTree build,
       │                                  join key extraction, reshape
       │                                  ordinals, Sort+Limit fusion.
       ▼
  ExecPlan (ordered SSA-like Steps)     ← positional, payloads baked.
       │
       ├─→ exec::execute                ← walks Steps, dispatches to
       │                                  phys::* via std::visit.
       └─→ (future) CodeGenerator       ← emits C++ that calls phys::*
                                          (Phase 3 in the original
                                          plan doc; not started).

  gamax::phys::* ← curated operator contract (reorder / filter /
                   project / eq_join / semi_join / anti_join /
                   cross_join / scalar_aggregate / group_by / sort /
                   top_k / distinct / concat). STRING passthrough,
                   empty-input short-circuits, count-then-probe tid
                   sizing — backend quirks absorbed here so every
                   caller above sees one uniform surface.
```

Three load-bearing invariants cut across every layer:

- **Signed `gamax::size_type` (int64) everywhere; `-1` sentinel.**
  Row counts, column ordinals, ColumnIds, step indices — all
  signed. Rules out the signed/unsigned mix-ups that bite
  DataFrame code when a size is compared against a negative
  delta.
- **No NULL support.** TPC-H has none; every gap that
  superficially required NULLs (Q13 outer join, Q21 anti-join
  with non-equi residual) got builder-level rewritten (see
  `docs/TPCH_JOURNEY.md`).
- **EXEC schema is authoritative at execute time** — not IR
  schema. See "Lowering" below.

## Layer 1 — IR (`src/cuda/ir/`)

The IR is a declarative DAG of typed operators. Every node carries
a full `Schema` (its declared output type); expressions reference
columns by `ColumnId` (global identity, not ordinal); factories run
type / ref validation at construction time, so a successfully-
constructed IR is always well-typed and well-formed.

### Data model

#### `ColumnId` (`ir/column_id.hpp`)
Global identity handle for a column: a plain `int64_t value` with
`-1` as the unset sentinel. Allocated from an `IdGenerator` (a
counter). A column's ColumnId is the correctness-bearing handle —
names are cosmetic and may be duplicated after joins.

#### `Schema` / `Field` (`ir/schema.hpp`)
A `Schema` is an ordered list of `Field { id, type, nullable,
name }`. Position matters (physical operators consume by ordinal);
`id` is the handle expressions use. Two lookup methods exist:
`index_of(ColumnId)` (correctness) and `index_of(std::string)`
(frontend/builder convenience).

#### `Catalog` (`ir/catalog.{hpp,cpp}`)
Table registry. One `CatalogTable { name, column_names,
column_types, column_nullable }` per registered source. Held by
the `QueryBuilder` and consulted when a `scan(...)` resolves
column names to types. GaMax has a single TPC-H catalog today
(`tpch/tpch_ir.cu::make_tpch_catalog`).

#### `Expr` (`ir/expr.{hpp,cpp}`)
Typed scalar expression tree. `ExprKind`:

| Kind | Payload | Notes |
|---|---|---|
| `ColRef` | `col_ref` (ColumnId) | pointer into the declared input schema. |
| `Literal` | `literal` (`variant<int64_t, double, bool, std::string>`) | integer literals widen at lowering; string literals become `string_scalar` only when a physical string op consumes them. |
| `BinArith` | `arith_op` + two children | `Add`/`Sub`/`Mul`/`Div`. Numeric promotion applied. |
| `BinCompare` | `compare_op` + two children | `Eq`/`Ne`/`Lt`/`Le`/`Gt`/`Ge`. Strict type match; no implicit widening. |
| `BinLogical` | `logical_op` + two children | `And`/`Or`. BOOL8 only. |
| `Unary` | `unary_op` + one child | `Not` only today. |
| `StringOp` | `string_op` + `string_values` + one ColRef child | `Equal`/`Startswith`/`Endswith`/`Contains`/`ContainsMultiple`/`In`. See "Name conventions" below. |
| `Case` | three children (cond BOOL8, then, else) | `then`/`else` must share a common numeric type; result carries that type. |
| `FunctionCall` | `function_name` + args | escape hatch; only `year` and `substring` recognized at lowering today. |

Factories live in `expr.cpp` and do type inference bottom-up:
construction-time validation yields a concrete `type` and
`nullable` for the whole tree.

**Name convention trap:** `StringOpKind::ContainsMultiple` is
**sequenced AND** — matches SQL `LIKE '%p1%p2%...%'`. Predecessor
was `ContainsAny` which wired to the same kernel. The current
name is accurate; the history is in `docs/TPCH_JOURNEY.md` under
"Footguns."

#### `PlanNode` (`ir/plan_node.{hpp,cpp}`)
Operator tree. One `PlanNode` per logical operator; the tree is
held by `std::shared_ptr<PlanNode>`, so a sub-plan may feed
multiple parents (CTE / self-join / shared sub-expression).

`NodeKind`: `Scan`, `Filter`, `Project`, `Aggregate`, `Sort`,
`Limit`, `Join`, `Distinct`, `Concat`. `JoinKind`: `Inner`,
`LeftOuter`, `LeftSemi`, `LeftAnti`, `Cross`. `AggKind`: `Sum`,
`Avg`, `Min`, `Max`, `Count`, `CountAll`, `CountDistinct`.

Per-kind payload lives inline as named fields (`scan_table`,
`filter_predicate`, `project_items`, `agg_keys`, `agg_items`, etc).
`kind` selects which payload is meaningful; others are left
default-constructed. Kept as named fields rather than a variant to
match the existing `Predicate` / `aggregation_request` style.

### Factories: construction-time validation

Every factory (`scan`, `filter`, `project`, `aggregate`, `sort`,
`limit`, `inner_join`, `left_outer_join`, `left_semi_join`,
`left_anti_join`, `cross_join`, `distinct`, `concat`) runs:

1. **Input validity** — no null inputs, at least one column, etc.
2. **Expression validation** — referenced ColumnIds are in the
   input schema; predicate is BOOL8; join condition is BOOL8 over
   the union schema; etc.
3. **Output schema derivation** — bottom-up from child schemas
   and the operator's semantics. Fresh ColumnIds from
   `IdGenerator` for newly-computed columns; input ColumnIds
   preserved for pure ColRef passthroughs.

A successfully-constructed `PlanPtr` is therefore always
well-typed and well-formed. Invalid constructions throw
`std::runtime_error` at factory-call time with a message naming
the specific invariant that failed.

**ColumnId preservation for ColRef passthroughs** is explicit:
`project({"x", col_ref(...)})` keeps the input's ColumnId, so
"rename / reorder / subset" style projects don't renumber the
downstream references. A rule that mutates the aggregate's
bodies in place (G4a) doesn't break ancestors; a rule that
structurally replaces a node (CountDistinct rewrite) has to
preserve ids explicitly — see `rules.cpp`.

### Builder API (`ir/builder.{hpp,cpp}`)

Fluent surface for constructing IR. Two types:

- `QueryBuilder(catalog, gen)` — top-level; owns the catalog
  reference and the IdGenerator. `scan(table, cols)` begins a
  plan chain.
- `PlanBuilder` — wraps a `PlanPtr` and resolves column references
  by name against the current `output_schema`. Every method is
  `const` and returns a new `PlanBuilder`; sharing a builder
  across branches yields a DAG naturally
  (`li.inner_join(li, ...)`).

`.c(name)` resolves a column reference to a typed `Expr` leaf
(the first match when names repeat after a join; callers that
need a specific side should capture the builder pre-join).

Standard TPC-H `tpch_ir.cu` builders use `PlanBuilder` throughout;
the rule-test and unit-test harnesses use the lower-level
`plan_node.cpp` factories directly.

## Layer 2 — Liveness analysis (`ir/liveness.{hpp,cpp}`)

Given a root node, compute the set of ColumnIds live at every
node's output. Used by:

- **Scan column pushdown rule** — drops `scan_columns` entries not
  live at the Scan's output.
- **`prune_filter_materialization` lowering option** — Filter
  emits narrow `output_ordinals` so columns no ancestor uses
  aren't materialized after the bitmask.

Algorithm:

1. Root's live set = every field in its declared `output_schema`.
2. Worklist-driven propagation: for each node, compute the live
   set at each child's output via `child_live_sets(node,
   live_out)`. If a child's live set grows, re-enqueue.
3. Monotone lattice (only adds ids), so termination is
   guaranteed.

Per-kind propagation rules live in `child_live_sets`. A few
interesting ones:

- **Scan, Limit, Sort**: pass-through (schema shape is
  preserved), plus any columns the operator's own payload
  references (sort keys).
- **Project**: for each item, if the output ColumnId is live, pull
  refs from the item's Expr.
- **Aggregate**: grouping keys are always needed (even if their
  output slot isn't live, the groupby code path consumes them);
  agg bodies contribute refs only if their output slot is live;
  `CountAll` has no body.
- **Join**: split `live_out` into parts from L's schema and parts
  from R's schema; union with side-specific refs from the join
  condition.
- **Concat**: positional — if output.col[i] is live, every
  input.col[i] is live (inputs don't share ColumnIds with each
  other or with the Concat output).

## Layer 3 — Rule engine (`ir/rules.{hpp,cpp}`)

`apply_rules(root, RuleConfig = {})` returns a (possibly identical)
new plan. Rules are the designated place for *semantic*
rewrites — `COUNT → SUM(1)`, non-equi-in-inner-join lifting,
multi-key decomposition — as distinct from *backend quirks*,
which live inside `phys::*`.

### `RuleConfig` knobs (most default-on)

| Rule | Field | Default | Target | Effect |
|---|---|---|---|---|
| **CountDistinct rewrite** | `rewrite_count_distinct` | on | single-agg `Aggregate` with `CountDistinct` over a ColRef body | emit `Aggregate(K, [CountAll], Distinct(K∪{x}))`, preserving the original agg-output ColumnId. Runs BEFORE G4a (which is now off, so the emitted `CountAll` flows directly to the phys layer's native `aggregation_id::COUNT_ALL`). |
| **G4a** | `rewrite_count_to_sum` | **off** | `Aggregate` items with `Count` / `CountAll` | (disabled) Used to rewrite the body to `Sum(lit(int64_t{1}))`. Disabled in favor of native `aggregation_id::COUNT_ALL` in the phys agg layer — the Sum(lit(1)) rewrite forced a full-cardinality int64 column to be materialized (4.5 GiB at SF=100 for Q1) just to be summed. The rule body is preserved in the source as a fallback for null-aware semantics that need to split `Count(x)` from `CountAll`. |
| **G7** | `split_non_equi_join_conditions` | on | **Inner** join whose ON is a conjunction with at least one non-equi clause | lift non-equi residuals into a post-join `Filter`; keep pure-equi clauses in the join. |
| **G1-rule** | `split_multi_key_equi_join` | on | **Inner** join whose ON is 2+ pure-equi conjuncts | keep first equi in the join, lift the rest into a post-join `Filter`. |
| **Scan pushdown** | `scan_column_pushdown` | on | `Scan` | drop `scan_columns` entries whose output ColumnId is not live at this Scan's output; narrow the Scan's `output_schema` to match. |

Ordering within `apply_local_rules`: CountDistinct rewrite runs
FIRST so the emitted `CountAll` can be picked up by G4a on the
same pass. G7 runs BEFORE G1-rule so any non-equi residual is
lifted out first; G1-rule then operates on a pure-equi
conjunction. Scan-pushdown runs last because it reads liveness
over the (possibly rewritten) tree.

**Deliberate non-behaviors:**

- G7 / G1-rule **only fire on Inner joins**. Semi / Outer / Anti
  residuals affect match semantics rather than the output rows,
  and need a different treatment (push residual INTO the match
  predicate). Tracked as a follow-up gap; Q21 sidesteps it at
  the builder level via the count-distinct-groupby trick.
- CountDistinct rewrite **only fires for single-agg, ColRef body**.
  Mixed shapes (e.g., `COUNT(DISTINCT x), SUM(y)`) pass through;
  lowering throws with an actionable message that names the
  narrow scope.

### Engine

`apply_rules(root, cfg)`:

1. Compute liveness over the current tree.
2. Walk bottom-up, memoizing on pointer identity so a shared
   sub-plan is rewritten exactly once.
3. For each node: rewrite children; if any changed, clone the
   parent with new inputs; then `apply_local_rules(node, cfg,
   liveness)` runs every enabled rule in sequence on the node.
4. Pass-level: repeat until the pass returns a pointer-identical
   tree (fixpoint), capped at 8 passes.

Fixpoints are needed because G7 / G1-rule produce new Filter
wrapping a freshly-created Join, and the next pass's walk needs
to examine that new Join. CountDistinct rewrite's output
contains a new Aggregate that G4a catches **in the same pass**
(hence the ordering above); no second pass needed for that pair.

## Layer 4 — Lowering (`ir/lower_to_exec_plan.{hpp,cpp}`)

Input: a rewritten `PlanPtr` + `Catalog` + `LowerOptions`.
Output: an `exec::ExecPlan` — a flat vector of `Value`s and a
flat vector of `Step`s.

### The EXEC schema invariant

Each `exec::Value` carries a `schema` field that reflects the
**actual runtime column layout** of that Value — not the IR's
declared `output_schema`. Column pruning rules (scan-pushdown,
`prune_filter_materialization`) narrow the EXEC schema while the
IR's declared schema is left stale. Every lowering helper that
resolves a ColumnId to an ordinal reads from `lo.exec_schema(id)`,
so upstream pruning propagates correctly through the DAG.

Without this invariant, a column pruned at the Scan would
require every ancestor's IR schema to be rebuilt, rippling
changes through the whole tree. With it, rules can narrow
locally and everyone else just reads Value schemas.

### `LowerOptions`

| Field | Effect |
|---|---|
| `prune_filter_materialization` | When true (default), Filter steps emit non-empty `output_ordinals` so columns no ancestor uses aren't materialized after the bitmask. Same liveness analysis as `scan_column_pushdown` but applied at lowering time. |

This is the boundary where a single "option" field lives that
isn't a rule. The rule / option separation: **rules rewrite the
IR tree** (semantic); **options control how a fixed IR gets
lowered** (codegen).

### Per-node lowering

Each `NodeKind` has a `lower_*` function that:

1. Recursively lowers children (memoized via `Lowering::memo`).
2. Reads the children's EXEC schemas to resolve any ColumnIds.
3. Builds the step's payload (`ExprTree` / `Predicate` / ordinal
   vectors / `AggSpec` / `SortSpec` lists).
4. Allocates a new `ValueId` with the correct EXEC schema.
5. `push_step(kind, inputs, output, payload)` — also updates
   each input's `last_used_step` to drive freeing.

Notable special cases:

- **Scan → single SourceTable step.** No follow-up Reorder; the
  SourceTable payload carries the requested column names in
  output order, and the resolver is contractually obliged to
  surface those columns in that order.
- **Sort + Limit fusion.** `lower_limit` looks at its child; if
  it's a Sort, emit one `TopK` step keyed on the Sort's keys
  over the Sort's INPUT — skip lowering the Sort node. Limit
  without a preceding Sort is `not_implemented` (TPC-H never
  uses that shape).
- **Aggregate** materializes `[keys..., agg-body-columns...]` as
  an intermediate Value via Reorder (pure ColRefs) or Project
  (mixed / computed), then emits `ScalarAgg` (0 keys) or
  `GroupBy` (≥1 keys) consuming that intermediate. Decouples
  ColumnId resolution from the phys layer's positional expectation.
  Two refinements live here:
  * **ColRef-body dedup** (`dedup_agg_bodies`): if multiple agg
    items have ColRef bodies pointing at the same `ColumnId`,
    only the first emits a fresh body column; subsequent items
    share its `body_position`. This pairs with `groupby_noskf`'s
    existing AVG-reuses-SUM dedup at the phys layer — same
    `column_index` → same internal SUM result. So Q1's natural
    `sum(l_quantity) + avg(l_quantity) + avg(l_disc) + sum(...)`
    form materializes l_quantity exactly once.
  * **CountAll / Count** items are routed directly to the phys
    layer's `aggregation_id::COUNT_ALL` — `to_physical_agg`
    maps both to the same kind. The materialize step emits no
    body column for them (count comes from the internal "ones"
    column the phys layer already builds for AVG's denominator),
    and `lower_aggregate` sets `spec.column = -1` (sentinel) on
    those agg specs so phys::group_by knows not to read an
    input column.
- **Inner join** assembles single-key equi (G1-rule already
  split multi-key); emits `EqJoin` followed by a `Reorder` that
  restores the IR-declared column order.
- **Concat** lowers every input, re-checks EXEC schema
  compatibility (arity + types) as a belt-and-suspenders
  against post-rules mismatches, and emits a single `Concat`
  step with N `Step::inputs`.

### ColumnId → ordinal resolution

`resolve_ordinal(schema, id)` is the single entry point for
"find the EXEC ordinal of this ColumnId." Lowering helpers never
open-code the lookup; a missing id throws with a clear
`ColumnId N not in schema` message.

## Layer 5 — ExecPlan (`exec/exec_plan.hpp`)

Positional, SSA-like:

- `Value { id, schema, last_used_step }` — an intermediate
  result. `schema` is the EXEC schema. `last_used_step` is filled
  in at lowering and consumed by the executor to drop the value
  as soon as no later step needs it.
- `Step { kind, inputs, output, payload }` — one operator per
  step. `inputs` is a vector of `ValueId`; `output` is a single
  `ValueId`; `kind` is the dispatch tag and `payload` is the
  per-kind variant (`StepPayload`).

`StepKind` / `StepPayload` variants (alphabetical):

| Kind | Payload | Phys entry point |
|---|---|---|
| `AntiJoin` | `{lkeys, rkeys}` | `phys::anti_join` |
| `Concat` | `{}` (N inputs live on `Step::inputs`) | `phys::concat` |
| `CrossJoin` | `{}` | `phys::cross_join` |
| `Distinct` | `{key_ordinals}` | `phys::distinct` |
| `EqJoin` | `{lkeys, rkeys}` | `phys::eq_join` |
| `Filter` | `{predicate, output_ordinals}` | `phys::filter` |
| `GroupBy` | `{n_keys, specs}` | `phys::group_by` (executor splits keys/values at `n_keys`) |
| `Project` | `{items}` (mixed `ExprTree` / `Predicate` / `SubstringItem`) | `phys::project` |
| `ScalarAgg` | `{specs}` | `phys::scalar_aggregate` |
| `SemiJoin` | `{lkeys, rkeys}` | `phys::semi_join` |
| `Sort` | `{keys}` | `phys::sort` |
| `SourceTable` | `{table_name, columns}` | resolver-served (no phys call) |
| `TopK` | `{keys, k}` | `phys::top_k` |
| `Reorder` | `{ordinals}` | `phys::reorder` |

Dispatch is by `std::visit` over the payload variant inside
`run_step`; `StepKind` exists alongside the variant as a tag for
printing and occasional switch-based lookups.

## Layer 6 — Executor (`exec/executor.{hpp,cpp}`)

`execute(plan, source_resolver, ctx)`:

1. For each step in order:
   1. SourceTable special case: `Slot.view = source_resolver(name,
      cols)` — aliased caller-owned view, Executor doesn't take
      ownership.
   2. Reorder / Project special cases: see "Aliasing fast paths"
      below.
   3. Everything else: `Slot.owned = run_step(step, live, ...)`.
   4. **Value lifecycle.** After the step runs, drop any input
      `live` entry whose `last_used_step ≤ current_step_index`.
      Skip source views (aliased). Returned blocks re-enter the
      memory pool's free list; no `cudaFree` in the hot path.
2. At end: move the final Value's `owned` table out to the caller
   (or materialize the aliased view via a pure Reorder, if the
   final Slot has an explicit view from a SourceTable or aliasing
   step).

**Table ownership boundaries:**

- **Source tables** — caller-owned via `SourceResolver`, Executor
  holds only `table_view`.
- **Intermediates** — Executor owns `unique_ptr<table>` per
  `ValueId` in a `live` map; drops it as soon as the current step
  index passes each Value's `last_used_step`.
- **Final result** — `std::move`'d out of the `live` map.

**Slot states.** A `Slot` carries `unique_ptr<table> owned` and
`table_view view`. The two fields define three valid states:

| State | `owned` | `view` | When | view_of() returns |
|---|---|---|---|---|
| SourceTable | null | set (≥1 col) | resolver hand-off | `view` |
| Standard | set | default (0 cols) | most ops; produced by `run_step` | `owned->view()` |
| Aliasing | set | set (≥1 col) | Reorder / mixed-Project alias paths | `view` (pinned by `owned`) |

`view_of()` prefers an explicit view (>=1 column) over
`owned->view()`. Backwards-compatible because non-aliasing steps
leave `view` default-constructed (0 cols) and fall through to
`owned->view()`.

### Aliasing fast paths

Two step kinds get special-cased in `step_dispatch` to skip the
gather/copy that the corresponding `phys::*` call would normally
do, at the cost of pinning their upstream's data slightly longer.

**Reorder.** When `plan.values[in].last_used_step == step_index`
(i.e. this Reorder is the upstream's last consumer) AND the
upstream Slot is pure-owned (has `owned`, no explicit `view`),
move the upstream's `owned` table into the Reorder's Slot and
build the Reorder's `view` as `select_columns(ordinals)` over the
moved-in table. Output is the same `view` downstream consumers
would have read from a copying `phys::reorder`, but the gather
kernel doesn't run and no allocations happen.

If the upstream has further consumers (DAG share — fairly rare in
TPC-H but the lowering is permitted to share), or the upstream
is a SourceTable (no `owned` to move) or itself an aliasing
Reorder (would need ordinal remapping that the Slot doesn't
currently carry), fall back to `phys::reorder`'s gather copy.
Both fallback cases preserve correctness; they just pay the copy.

**Project.** Same shape as Reorder but mixed: the agg items are
classified into ColRef passthroughs vs. non-trivial (Computed
ExprTree / Predicate / Substring). The non-trivial subset goes
through `phys::project` on the *intact* upstream view (must run
first — it reads source columns by ordinal); then the ColRef
columns are *moved* out of upstream's owned table directly into
their target output positions. Same single-consumer + pure-owned
guards as Reorder; same fallback to today's full-copy
`phys::project` when either fails. Two ColRef items hitting the
same source ordinal: first ref moves, subsequent refs go through
`phys::project`'s copy path, deduped via a small seen-set.

The structural consequence of "alias only when upstream is
pure-owned" is that the database convention "scan output is
untouched" is enforced *for free*: SourceTable upstreams have no
`owned`, so Project-after-Scan always copies. No config flag
needed.

**Debug hook.** If `GAMAX_PHYS_DEBUG` lists a kind (case-
insensitive, comma-separated), the executor prints a summary +
first N rows of that step's output to stderr after the step
runs. Useful for bisecting wrong results without recompiling.

## Layer 7 — Phys (`phys/phys.{hpp,cu}`)

The curated operator contract. Every function takes a `Context`
(carries the CUDA stream) and returns a `table_ptr`. This is the
**only** layer that knows about the incomplete / quirky surfaces
of the raw `gamax::*` operators underneath.

### Surface

```
table_ptr reorder(in, ordinals, ctx);
table_ptr filter(in, pred, ctx);
table_ptr filter(in, pred, output_ordinals, ctx);  // selective
table_ptr project(in, items, ctx);                 // ExprTree / Predicate / Substring
table_ptr eq_join(L, lkeys, R, rkeys, ctx);
table_ptr semi_join(L, lkeys, R, rkeys, ctx);
table_ptr anti_join(L, lkeys, R, rkeys, ctx);
table_ptr cross_join(L, R, ctx);                   // via fake-key + eq_join
table_ptr scalar_aggregate(values, specs, ctx);
table_ptr group_by(keys, values, specs, ctx);
table_ptr distinct(keys, ctx);
table_ptr concat(inputs, ctx);                     // N-ary
table_ptr sort(in, keys, ctx);
table_ptr top_k(in, keys, k, ctx);
table_ptr left_outer_join(L, lkeys, R, rkeys, ctx); // throws "not yet implemented"
```

### Responsibilities

- **STRING passthrough.** Most raw `gamax::*` operators (project,
  compact, eq_join's internal `inner_eq_join`) were built
  numeric-first. Phys wraps them with STRING-aware gather paths
  (`simple_materialize` in `operators/join/materialize.cu`,
  `gather_strings` in `string/string_sort.cu`). A future caller
  never has to know.
- **Empty-input short-circuits.** Every phys op handles
  zero-row inputs: sort → zero-rows-like copy; group_by →
  correctly-typed empty table with keys + agg output types;
  eq_join / semi_join → empty output with the right raw schema.
  The `prefix_sum` empty-input guard
  (`utils/transform.cu::prefix_sum_impl`) was added specifically
  because `thrust::inclusive_scan` on a zero-element range
  raised `cudaErrorInvalidDevice` on some CUDA/thrust combos
  during temp-storage sizing.
- **Native `aggregation_id::COUNT_ALL`.** The agg layer's
  `aggregation_id` enum has a `COUNT_ALL` kind that produces
  per-group row counts (or, in the scalar-aggregate path, a
  single INT64 = input row count). The grouped path reuses the
  internal "ones" column that `groupby_noskf` already builds for
  AVG's denominator: when `has_avg || has_count_all || hint.agg_count`,
  one ones-column + one internal SUM serves all three purposes.
  COUNT_ALL agg requests pass `column_index = -1` (sentinel —
  not read; the kind alone disambiguates) and are routed to the
  count column at result-build time. `agg_output_data_type`
  short-circuits COUNT_ALL → INT64 before its `type_dispatcher`
  so callers don't need a valid input type. Critical sizing
  detail in `groupby_noskf`: `ones` is allocated by
  `key.num_rows()`, not `val.num_rows()` — `val` has 0 columns
  when COUNT_ALL is the only agg with no other val-bearing
  sibling, and `val.num_rows()` returns 0 in that case.
- **Tight output bounds.** `ieq_hash_join` previously allocated
  1B-row tid buffers by default. Now calls `multiset::count`
  first (when no tight hint is supplied), then allocates tids at
  exact size. Hinted callers (microbenchmarks) bypass the extra
  pass — **do not remove the hint settings in
  `benchmarks/join/ieq_join_nvbench.cu` without also lifting the
  count-probe protection**.
- **Cross-join.** Internally via a fake-key `eq_join`: broadcast
  a constant key column over each side so the two rows hash to
  the same group. Output shape matches the IR's declared L ++ R.
- **Concat.** N-ary at the surface. Numeric columns copy via
  per-input `cudaMemcpyAsync`; STRING columns concat the byte
  buffer and shift per-input offsets via a single-block-per-input
  kernel. Zero-row inputs short-circuit.

### What's not in phys (deliberate)

- **`left_outer_join`**: declared as a placeholder, throws at
  call-time. Q13's outer-join shape is served via a builder-level
  decomposition (`inner + anti + concat`). Needs a NULL scheme
  or matched-mask column to land for real.
- **Multi-key `eq_join` / `semi_join`**: G1-rule splits multi-key
  inner joins at the IR layer; phys receives single-key pairs.
  Multi-key for semi/anti is out of scope (no TPC-H query needs
  it).
- **String join keys**: `cuco::is_bitwise_comparable_v` gates the
  hash set's Key type. No TPC-H query joins on a STRING key.

## What the stack is NOT

Things a reader might expect that aren't here:

- **No join-order optimizer.** TPC-H join orders are hand-written
  in `tpch_ir.cu`. Q5's SF10 OOM motivated a manual reorder;
  future queries at larger scale will likely need the same.
- **No catalog statistics.** No NDV estimates, no histograms,
  no cardinality-based decisions. Some micro-heuristics exist in
  `groupby::auto_select_groupby_algo` but nothing cross-operator.
- **No native NULLs.** All operators assume every input column
  is non-nullable. Schema tracks a `nullable` flag but nothing
  enforces it or produces NULLs.
- **No spilling.** Intermediates live entirely on the GPU pool;
  SF100-scale queries that exceed 88 GiB would OOM.
- **No CodeGen yet.** The ExecPlan was designed to feed either
  the Executor or a CodeGenerator. Only the Executor is in place
  (Phase 3 in the original plan doc has not started).

## Reading order for a new contributor

1. **This doc** — skim for the layer map and invariants.
2. **`docs/TPCH_JOURNEY.md`** — the narrative of how the stack
   grew and every footgun we've hit. If you're going to touch
   joins, aggregates, the rule engine, or the phys layer, read
   this first.
3. **`src/cuda/ir/plan_node.hpp`** — one-file summary of every
   IR operator + factory contract.
4. **`src/cuda/exec/exec_plan.hpp`** — one-file summary of every
   ExecPlan shape.
5. **`src/cuda/phys/phys.hpp`** — the phys surface.
6. **A rule of your choice** (e.g., `rule_count_distinct_to_distinct_plus_count`
   in `ir/rules.cpp`) — shows how the pieces plug together for one
   semantic rewrite.

## Commit anchors

The IR stack was built across the following commits
(chronological, from `a2209a4` to `89e7164`):

- **Foundations (a2209a4 … aa536a4).** ColumnId, Schema,
  Catalog; Expr tree with type inference; PlanNode DAG with full
  TPC-H operator coverage; fluent PlanBuilder / QueryBuilder
  surface.
- **Lowering (412b1d5 … 8fd0035).** Expression lowering
  (`Expr → ExprTree` / `Predicate`); per-NodeKind lowering; Q6
  end-to-end; Inner/LeftSemi join, Sort+Limit fusion.
- **Pretty-printer (8876852).**
- **Exec split (c8817eb).** Emit split into phys layer +
  ExecPlan + Executor. The three-layer pipeline since.
- **LowerOptions + rules (49c42f7, bbbdb39, a283e19, b84056e).**
  Column-pruning via liveness; pluggable rule engine with the
  three Tier-1 rewrites; `prune_filter_materialization` knob;
  scan-pushdown.
- **TPC-H reader wiring (72119a9).**
- **Feature pile-on (across 30+ commits).** STRING groupby
  (`be1104b`), `string_in` (`f18e3fc`), anti-join (`97aacc3`),
  DATE encoding (`3077975`), `year_of` (`3a12eb1` + `b2a7dc7`),
  all-22 wired into runner (`5f2a50e`), `LeftAnti` dispatch
  (`82ccd43`), Q12/Q8 pre-split (`3d41590`), scalar AVG
  (`1b8b437`), `substring` (`d22c694`), CountDistinct rewrite +
  `Distinct` lowering + `phys::distinct` (`abcf64b`).
- **Verification infra (f050fcb).** DuckDB golden generator +
  `run_tpch_ir --verify`.
- **Builder + engine polish (238bcc0 … d24aa2f).** Date-window
  fixes (Q3/Q4/Q5/Q6/Q10/Q15/Q20); `prefix_sum` empty-input
  guard; StringInThrows test fix; `ContainsAny → ContainsMultiple`
  rename + Q16 predicate correction; Q21 count-distinct rewrite;
  `ieq_hash_join` count-then-probe; Q5 SF10 join-order rewrite.
- **Q13 landing (89e7164).** `NodeKind::Concat` + `phys::concat`
  (N-ary UNION ALL, STRING-aware) + Q13 outer-join
  decomposition.

Every one of those commits has a substantial body message
explaining its specific concern; the journey doc has the big-
picture arc.
