# GaMax TPC-H — the whole journey

**End state (2026-04-24):** all 22 TPC-H queries execute on the GaMax
GPU engine and match DuckDB-generated Parquet goldens byte-for-byte
at both SF1 and SF10, inside the verifier's per-type tolerances
(exact for ints / strings; `|Δ| ≤ max(0.01, 1e-6·|expected|)` for
floats). Zero engine-level divergences. The verifier, goldens, and
runner driver are all committed.

**Starting state (week one):** Q6 matched pyarrow. Q1 crashed on
STRING groupby. Most of Q2–Q22 either threw `not yet implemented`
somewhere or produced garbage. The engine had no rule machinery, no
DuckDB oracle, no dedicated TPC-H verify path — just a pipeline and
ambition.

This doc is the story in between: what we did, why, what we decided
not to do, what assumptions are load-bearing, and the footguns a
future maintainer is most likely to step on.

## Headline numbers

| Metric | Start | End |
|---|---|---|
| Queries that run end-to-end | 1 (Q6) | 22 / 22 |
| Queries verified against a canonical oracle | 1 (Q6 vs pyarrow, manual) | 22 / 22 (DuckDB golden, automated `--verify`) |
| IR rules | 0 | 5 (G4a, G7, G1-rule, scan-pushdown, CountDistinct rewrite) |
| Phys ops | reorder/filter/project/eq/semi/cross/scalaragg/groupby/sort/topk | + anti, distinct, concat, substring, year_of, in_multiple, and more |
| Unit tests | ~220 | 620+ across ~45 binaries |
| Pool OOMs on runnable queries | Multiple | 0 |

## Architecture (inherited, preserved)

The three-layer pipeline that pre-existed this arc stayed intact:

```
  IR (PlanNode DAG)  —  declarative, rule-rewritable, ColumnId-keyed
      │
      │  apply_rules(plan, RuleConfig)   ← semantic IR → IR rewrites
      ▼
  IR' (rewritten)
      │
      │  lower_to_exec_plan(plan, cat, LowerOptions)
      ▼
  ExecPlan (ordered SSA Steps)   ← positional, payloads baked
      │
      ├─→ exec::execute          ← walks steps, dispatches to phys::*
      └─→ (future) CodeGenerator ← emits C++ that calls phys::*
  phys::* ← curated operator surface; backend quirks absorbed here
```

Invariants that survived (and shaped) every gap-closing decision:

- **Signed `gamax::size_type` (int64) everywhere with `-1` sentinel.**
  No uint*/0 sentinel. Keeps row counts interchangeable with
  SQL-sized arithmetic; avoids the signed-unsigned mix-up trap.
- **No NULL support.** TPC-H has none; we don't introduce a NULL
  scheme lightly. Every gap that superficially required NULLs (Q13
  outer join, Q21 anti-join with non-equi residual) got
  builder-level rewritten instead.
- **EXEC schema, not IR schema, is authoritative at execute time.**
  `lo.exec_schema(value_id)` is the right lookup; stale IR
  `output_schema` fields survive after rules prune columns, and
  lowering flows through actual Value schemas.
- **DAG sharing is honored.** Plan factories return `shared_ptr`;
  rules memoize on pointer identity and copy-on-write. A node
  reachable from multiple parents is rewritten once.
- **Phys layer absorbs backend quirks.** STRING passthrough,
  multi-key fusion, empty-input short-circuits, cross-join-via-fake-
  key. Callers above see one complete interface; raw
  `gamax::{inner_eq_join, groupby, ...}` peculiarities don't leak.
- **Rules vs. LowerOptions are distinct concerns.** Rules rewrite
  the IR tree (semantic); LowerOptions controls how a fixed IR
  gets lowered (codegen / pruning).
- **One stream per Executor.** `phys::Context` carries it; all phys
  ops take a stream. Stream pools are a future additive extension.

## Timeline — the arcs

Not a strict chronology, but the five problem arcs each closed
over one or two focused sessions.

### 1. Foundations: rule engine, scan pushdown, DuckDB reader

**Problem:** pyarrow-as-oracle, no automation, an `emit.cpp` that
wired TPC-H-shape queries by hand. Every new query was bespoke.

**What shipped:**
- Pluggable IR rule engine (`apply_rules` + `RuleConfig`) with G4a
  (Count → Sum(lit(1))), G7 (inner-join non-equi residual →
  post-filter), G1-rule (multi-key inner join → single-key + filter),
  and scan column pushdown (drops unused catalog columns at the
  Scan boundary).
- Parquet reader (`tpch_reader.cpp`) ingesting into pinned-host
  caches, with Arrow DATE32 → INT32 YYYYMMDD coercion and
  decimal128 → float64 flattening. **Date encoding as packed
  YYYYMMDD** was the single biggest leverage decision: lex
  comparison and equality work, integer arithmetic (year extract)
  is `value / 10000`, and the storage is int32. Tradeoff: no
  date-difference arithmetic (we don't need it for TPC-H).
- `run_tpch_ir.cpp` command-line driver.

**Corner cut:** the legacy `emit.cpp` was NOT retired. It's still
present as a thin wrapper around `lower_to_exec_plan` + `execute`,
because some tests rely on it. If the CodeGenerator ever lands
(Phase 3 in the original plan doc), `emit.cpp` should go with it.

### 2. STRING-groupby + multi-col routing (Q1, Q4, Q18, Q20)

**Problem:** `hash_groupby_multi_key` threw `Invalid type_id: 12`
before it could touch STRING keys. `auto_select_groupby_algo` only
handled single-column numeric keys. Q1, Q4, Q5, Q10, Q16, Q21, Q22
all needed STRING keys somewhere; Q18 / Q20 needed multi-col
numeric.

**What shipped:**
- `string_column_impl` ctor fix (untyped `.data()` → `.data<char>()`).
- Leader-tid gather rewrite of `compute_groupid_kernel` +
  `hash_groupby_multi_key`: the kernel writes one leader_tid per
  group, and key columns are materialized POST-kernel via
  `simple_materialize` (STRING-safe) rather than inline (which
  would require Arrow offsets we can't produce in-kernel).
- `auto_select_groupby_algo` routes STRING → HASH and multi-col →
  HASH; neither falls through to `AUTO_ALGO` anymore.
- Per-callsite STRING dispatch in `simple_materialize`,
  `util::device::compact(table)`, `multikey_compress` probe.
- `atomic_add<double>` specialization in `atomics/atomics.cuh` —
  the default 8-byte generic path was doing
  `static_cast<unsigned long long>(val)` on a double, truncating to
  integer bytes. Broke Q11 / Q15 (both yielded 0 rows, or nonsense
  sums, depending on data).

**Design decision preserved:** keys are materialized post-kernel,
numeric as well as STRING, for uniformity. Slight overhead on
numeric keys; keeps the code path single. Good trade.

**Documentation:** see `docs/GROUPBY_STRING_KEYS.md` for the full
write-up of the STRING groupby work.

### 3. Structural gaps (Q7/Q8/Q9/Q12/Q14/Q16/Q22) — one rule or op per gap

Each of these was a narrow, self-contained gap; the pattern was
"find the exact thing SQL needs, add the smallest thing that
covers it, prove it against pyarrow/DuckDB." In order of landing:

- **`year()` scalar function** (Q7/Q8/Q9): new
  `ExprId::YearOf` + `ExprTree::year_of(child)` evaluators
  (fused and no-fused). Behaviour is `value / 10000` on the packed
  YYYYMMDD. IR-level: `FunctionCall("year", date)` rewritten in
  `lower_expr.cpp`.
- **StringOp-in-CASE body pre-split** (Q12, Q8): builder shape —
  Q12/Q8's CASE body had a StringOp equality that needs BOOL8, but
  the surrounding numeric CASE wanted a pure ExprTree. Pre-split
  the StringOp into a prior Project that emits a BOOL8 intermediate
  (`is_high`, `is_brazil`), then CASE references it as a ColRef.
  No engine change. Same pattern Q14 already used.
- **`phys::anti_join`** (Q16, Q21, Q22): shares a
  `semi_hash_membership_mask` helper with semi-join; same output
  shape (L-only), different membership predicate. IR-dispatch
  through `lower_join` + new `AntiJoinPayload`.
- **`substring()` scalar function** (Q22): new
  `gamax::substring(in, offset_1based, length, stream)` in
  `string/string_sort.cu` — two-pass algorithm (per-row length via
  `thrust::transform` → `prefix_sum` → single copy kernel). IR
  lowering: `FunctionCall("substring", [ColRef, offset_lit,
  length_lit])` → `SubstringItem` variant of `ProjectItem` (since
  substring produces STRING, neither `ExprTree` nor `Predicate`
  fit).
- **Scalar AVG** (Q22's inner `avg(c_acctbal)`): new AVG branch in
  `scalar_aggregate` using a host-side div-by-n, reusing `Sum<T>`'s
  output-type widening. Not bolted into the monoid
  `DISPATCH_AGGREGATE` macro since AVG isn't a monoid.
- **`STRING_IN` predicate** (Q7/Q12/Q19/Q22): `in_multiple` kernel
  reusing `multi_pattern_helper` from `contains_multiple`;
  length-pre-check + full-string compare, short-circuit OR.
- **`CountDistinct` rewrite** (Q16): new IR rule
  `rule_count_distinct_to_distinct_plus_count` emits
  `Aggregate(K, [CountAll], Distinct(K∪{x}))`, relying on G4a to
  compose the emitted `CountAll → Sum(lit(1))` on the same pass.
  New `NodeKind::Distinct` lowering + `phys::distinct` (sharing the
  cuco-set/leader-tid phase with `hash_groupby_multi_key` via an
  extracted `compute_group_ids` helper).

**Design decision (recurring):** for TPC-H-peculiar shapes, prefer
a **narrow, semantic rule** + minimal phys support over a
general-purpose operator that has to handle every corner. E.g.
`rewrite_count_distinct` fires only for `agg_items.size() == 1 &&
body.kind == ColRef`; any mixed Aggregate (e.g. `COUNT(DISTINCT x),
SUM(y)`) throws at lowering with an actionable message. TPC-H
doesn't need mixed-agg CountDistinct; rather than build
grouping-sets machinery for one hypothetical use, we defer.

### 4. Verification infra + builder corrections

**Problem:** the 9–13 queries that "should run" produced
plausible-looking outputs, but we had no way to confirm they
matched the spec at scale. Manual pyarrow diffs are fragile and
don't scale to 22 queries.

**What shipped:**
- `script/gen_tpch_golden.py` — registers existing
  `/scratch/wubo/sfN/*.parquet` tables as DuckDB views (no data
  regeneration), runs the spec-compliant SQL served by DuckDB's
  `tpch_queries()`, and writes one Parquet golden per query into
  `sfN/golden/qNN.parquet`. Pre-query coercion: DATE → INT32
  YYYYMMDD, DECIMAL → DOUBLE (matching GaMax's reader).
- `src/cuda/tpch/verify.{hpp,cpp}` — column-wise comparison with
  per-type tolerances. **Integer widening is built in** (INT32 got
  vs INT64 golden compares losslessly) because DuckDB's
  `EXTRACT(YEAR ...)` returns INT64 while GaMax's `year_of` returns
  INT32, and we didn't want false type-mismatch noise.
- `run_tpch_ir --verify` flag wires it up. DuckDB stays out of the
  C++ build — only Arrow/Parquet (already linked) are needed at
  runtime.

**What it surfaced:**
- Q3: column-order bug — gamax emitted `[l_orderkey, o_orderdate,
  o_shippriority, revenue]` (aggregate keys-first); spec wants
  `[l_orderkey, revenue, o_orderdate, o_shippriority]`. Fixed with
  a pure-ColRef Project between the aggregate and the sort.
- Q4/Q6/Q15/Q20: **date-window constants were one or more years
  off the spec**. This was the "the builder was written against
  what looked like TPC-H" trap — tiny SQL transcription errors
  that were invisible without a canonical oracle.
- Q5: revenue drift + row reorder — also a date-window bug.
- Q20: `p_name LIKE 'forest%'` was implemented as
  `Contains("forest")` instead of `Startswith("forest")`,
  overmatching on non-prefix occurrences (supplier comments like
  "rainforest") and ~doubling the output.
- Q10: the date window produced an empty intermediate which
  exposed a **separate** engine bug (see §5).

All fixes were builder-side; no engine changes.

### 5. Engine scale fixes (SF10 readiness)

**Problem:** many queries that worked at SF1 either OOMed the 88 GiB
pool at SF10 (Q5) or crashed inside kernel-level empty-input paths
(Q10 with its fixed date window). The root causes were
**preallocation-by-upper-bound** and **empty-input CUDA quirks**.

**What shipped:**

- **`prefix_sum_impl` empty-input guard.** `thrust::inclusive_scan`
  on a 0-element range raises `cudaErrorInvalidDevice` on some
  CUDA/thrust combinations during temp-storage sizing. The error
  was misleadingly named; the fix is defensive
  (`if (n == 0) { memset(out+1, 0, ...); return; }` for
  `INCLUSIVE_PAD_ZERO`; direct return for INCLUSIVE/EXCLUSIVE).
  Callers don't need to guard individually.

- **`ieq_hash_join` count-then-probe.** `multiset::probe`
  pre-allocated its left/right tid buffers at
  `MAX_JOIN_OUTPUT_SIZE = 2^30` rows = **8 GB per buffer × 2 = 16 GB
  per hash join**, trimming logical size via `set_size` but NOT the
  physical backing buffer. Chain a few of those and the pool is
  toast. Now, when the caller supplies no tight bound (no
  `max_out` / `unique_left` / `unique_right`), we call the
  multiset's `count()` first (cheap second pass, same asymptotic
  work as probe minus the tid writes), then allocate tids at the
  exact match count. **Benchmarks must keep supplying a tight
  hint** — see the in-code warning and footgun list.

- **Q5 join-order reorder.** Builder-side, not engine. The old
  plan joined
  `region ⋈ nation ⋈ supplier (nation-keyed)` × `customer
  (nation-keyed)` as a many-to-many on nationkey, producing a
  1.2B-row intermediate at SF10 (~87 GB). Rewrote to build the
  two sides of the FK chain independently and bridge them once
  at the end via `(l_suppkey, c_nationkey) = (s_suppkey,
  s_nationkey)`. G1-rule splits the bridge into suppkey-first
  (PK-unique) + nationkey-filter, bounding the intermediate by
  the fact side. Peak drops 1.2B → ~1.8M rows.

### 6. The hardest cases: Q21 (non-equi) and Q13 (outer join)

These needed the most thought because they'd normally demand
significant engine investment (non-equi membership semantics,
NULL support). We dodged both with builder-level rewrites that
lean on existing machinery.

**Q21 — EXISTS / NOT EXISTS with `<>` residuals.** The canonical
query has two correlated subqueries whose match predicates each
carry a `<>` clause against the outer row. Gamax's semi/anti joins
require pure-equi conditions; `<>` doesn't lower.

Maximus's Q21 implementation turned out to contain the trick we
needed for the EXISTS half (we found it in their
`tpch_queries.cpp`): observe that
`EXISTS (l2 | l2.orderkey=l1.orderkey AND l2.suppkey<>l1.suppkey)`
is equivalent to "this order has > 1 distinct supplier", which is
purely order-side and precomputable as
`COUNT(DISTINCT l_suppkey) GROUP BY l_orderkey > 1`. Membership
against that precomputed set is a pure equi-semi-join on
`l_orderkey`.

Maximus's implementation stops there — it omits the NOT EXISTS
half. We added the parallel rewrite: `NOT EXISTS (l3 | ... AND
l3.suppkey<>l1.suppkey AND l3.late)` ⟺ "exactly 1 distinct failed
supplier on this order" (since l1 itself is failed, that supplier
must be l1.suppkey). Two `count_distinct` aggregates feeding two
pure-equi semi-joins. The existing `rewrite_count_distinct` rule
lowers both. No new engine work.

**Q13 — LEFT OUTER JOIN with predicate-filter on the ON clause.**
Two lifts at once: the filter pushdown (spec form has the predicate
inside the ON; pushing it to filter orders first is semantics-
preserving for LEFT OUTER whenever the predicate references only
the right side — DuckDB does it too) and the outer-join
decomposition.

The decomposition: every customer either has ≥1 matching clean
order (inner branch) or 0 matching clean orders (anti branch).
Compute each branch's per-customer `c_count` independently — matched
via `aggregate(inner_join, {c_custkey}, [count_all])`, unmatched
via `project(anti, {c_custkey, c_count = lit(0)})` — then UNION
ALL. Needed a new `NodeKind::Concat` / `phys::concat` primitive.

A key optimization here: **push aggregation below the concat**.
The naive decomposition concats `matched_join` (13.5M rows at SF10,
one per matching order) with `anti` (0.5M rows, one per unmatched
customer), then aggregates. ColRef passthroughs in `phys::project`
are deep-copies — so that concat + project alone would allocate
~224 MB at SF10 for no reason. Aggregating on the matched side
FIRST collapses it to ~1M rows before concat; the concat input
volumes are balanced and small, and the only remaining copy is a
negligible 4 MB on the anti side.

## Corners cut (deliberate)

- **No `phys::left_outer_join`.** Never wrote it. The decomposition
  covers Q13; if a future frontend query needs a true outer join
  with ON-clause predicates that reference both sides (i.e., can't
  be pushed down), we'll need to do real work. Until then, the
  "no NULLs" architectural invariant stands.
- **No real non-equi residual in semi/anti joins.** Same story —
  covered for TPC-H via Q21's count-distinct trick. A rule-engine
  pass that pushes a non-equi clause INTO the match predicate is
  still an open work item for the general case.
- **No general CountDistinct.** The rewrite fires only for single-
  agg Aggregate with a ColRef body. Mixed shapes (`COUNT(DISTINCT x),
  SUM(y)`) throw at lowering with an actionable message. Not
  required by TPC-H.
- **STRING join keys not supported.** `cuco::is_bitwise_comparable_v`
  requires numeric types. No TPC-H query joins on a STRING key
  (all are INT64/INT32 keys). Would need STRING key hashing and
  a STRING hash set to lift.
- **Q13's filter-pushdown is manual, not a rule.** Worked into
  `build_q13` directly. If a second query ever needs "push
  right-only filter past LEFT OUTER JOIN", extract into a rule then.
- **Q5's join order is manual.** No join-order optimizer exists.
  The other 21 queries' current orderings happen to be OK on the
  data we have, but SF100 or different PK/FK cardinalities could
  re-expose the same Cartesian trap.
- **No multi-GPU, no multi-node.** Single-stream Executor is the
  rule.
- **Scale factor support capped at SF10.** SF100 (~100 GB) doesn't
  fit in the 88 GiB pool; we haven't benchmarked what breaks first.
- **The verifier widens integer pairs INT32↔INT64 silently.** This
  buys us false-positive-free checks, but it would also hide a
  real-world case where gamax emits INT32 and should have emitted
  INT64 (e.g. a sum that overflows int32 at scale). Haven't hit it
  yet; watch for it if a future query chain aggregates huge counts
  into the int32 space.
- **Concat is materializing.** The phys op allocates `sum(rows)` of
  output column and copies each input in. Zero streaming; 2× memory
  during the op. Fine for our scales, wasteful for future 100B-row
  unions. Streaming Concat (with a late-bound output offset
  producer) is a future exercise.
- **`copy_any_column` deep-copies ColRef passthroughs.** This
  surprises readers of `phys::project` — Q13's builder was written
  with the assumption that "push aggregation below the concat"
  pattern is important precisely because of this. Future
  optimization: a zero-copy "append constant column" primitive
  would make the anti branch essentially free.
- **`emit.cpp` still exists.** Legacy two-line wrapper around
  `lower_to_exec_plan` + `execute`. Some tests use it. Deleting it
  is risk-free but hasn't happened.

## Load-bearing assumptions

These are things the code relies on. If any change, expect
breakage:

- **TPC-H dbgen data has no NULLs.** Every Parquet column we
  produce has `nullable = false` in the Arrow schema. Any
  mechanism that produced NULLs (e.g., outer join) would need
  explicit handling we don't have.
- **`l_returnflag = 'R'` is set only for lineitems with
  `l_receiptdate ≤ 1995-06-17`** (dbgen's simulation "now"). Q10
  and similar queries rely on this implicitly when they filter
  orders into a future date window and expect returnflag='R' to
  find rows.
- **All monetary columns can be safely flattened to FLOAT64.**
  TPC-H's DECIMAL(15, 2) values fit losslessly in a double for
  individual rows; aggregated sums over SF1/SF10 stay within
  mantissa precision per the verifier's `1e-6` rel-tol.
- **All date comparisons are
  lex-equivalent on packed YYYYMMDD int32.** True because 10 >
  months > 0, 100 > days > 0, and YYYY*10000 + MM*100 + DD
  preserves chronological order. Breaks if you ever want date
  arithmetic (add 30 days etc.) — you'd need real date
  encoding then.
- **All join outputs have exactly one matching-key column** (the
  shared key). `phys::eq_join` enforces this via the raw
  `inner_eq_join`'s "column 0 is the key" convention. Multi-key is
  handled by G1-rule splitting at the IR layer.
- **The DuckDB goldens reflect the Parquet data at `$TPCH_DATA_PATH`.**
  If anyone regenerates the Parquet with a different dbgen seed,
  re-run `script/gen_tpch_golden.py`. Otherwise verifies will
  diverge for all queries at once.
- **The memory pool is global and oversubscribed** (~88 GiB on H100,
  allocated up-front). If another CUDA library or process grabs
  device memory during a query, the pool's free-list sum can
  appear adequate while contiguous allocation fails. We don't
  guard against this; reset the pool between queries if strange
  OOMs appear.

## Potential footguns (for maintainers)

- **`StringOpKind::ContainsMultiple` is sequenced AND, NOT `contains
  any`.** The name's history: it was once `ContainsAny` and
  wired to `contains_multiple` anyway, which produced the opposite
  meaning. Renamed. Future readers: the phys op's semantic is
  "contains all patterns, in order, non-overlapping" — matches
  SQL `LIKE '%p1%p2%...%'`.
- **`ieq_hash_join`'s count-then-probe adds ~2× to the hash-join
  hot path when the caller supplies no tight bound.** The nvbench
  microbenchmarks (`benchmarks/join/ieq_join_nvbench.cu`) set
  `hint.max_out = right_num_items` and `unique_left = true`, so
  their numbers are unaffected. **Removing those hint settings
  without also disabling the count-probe protection will silently
  regress benchmark numbers by ~2×.** The in-code comment spells
  this out.
- **`phys::project` deep-copies ColRef passthroughs.** A project
  that looks like a no-op (pure rename / reorder) isn't; every
  column is gathered or copied. Plans that do "aggregate → project
  → concat → aggregate" pay for the middle copy. Push aggregation
  below concat or use Reorder instead of Project for pure-ColRef
  reshapes.
- **Concat produces fresh ColumnIds.** Any downstream reference
  that was built against an input's ColumnId will fail to resolve.
  Builder-side: look up by name via `.c("name")` after the concat.
- **`phys::distinct`'s output row order is insertion order, not
  specified.** Callers that expect a particular order MUST sort.
  Q16, Q21 rely on a downstream sort; don't write a new caller
  without one.
- **The CountDistinct rewrite preserves the original aggregate
  output's ColumnId** (see `rule_count_distinct_to_distinct_plus_count`
  in `rules.cpp`). Skipping that preservation breaks every
  ancestor Sort/Project that referenced the original id. Similar
  care is needed for any future structural rewrite that replaces
  a node wholesale.
- **Q21's `count_distinct over all lineitem > 1` and `count_distinct
  over failed lineitem == 1` rewrites are correct only because**
  (a) l1 itself is a failed lineitem on the order (so the failed
  set is non-empty, forcing `n_failed_supp ≥ 1`), and (b) the
  "no other supplier failed" semantic collapses to `= 1` rather
  than `= 0`. If a future query's NOT EXISTS doesn't share these
  conditions, the rewrite doesn't transfer directly.
- **Q16 shipped initially with `Contains("Customer") &&
  Contains("Complaints")` instead of `ContainsMultiple(...)`.** It
  passed verification by coincidence (no supplier in the data had
  "Complaints" before "Customer"). The current builder uses the
  correct sequenced predicate, but the episode is a reminder that
  **data-coincidence silences wrong rewrites**. Any `Contains` +
  `&&` pattern in a builder should be re-checked against the
  spec's LIKE shape.
- **`atomic_add<double>` used to truncate**. Specialized now. If a
  new numeric type ever gets added to the atomics helpers, its
  generic `unsigned long long` fallback must not be a silent bit
  reinterpretation. The old bug produced plausible-but-wrong
  output for two queries before anyone noticed.
- **`prefix_sum_impl` short-circuits on `n == 0`.** Callers that
  assume "the scan always runs and writes something" are wrong.
  For `INCLUSIVE_PAD_ZERO` the single zero at out[0] is still
  written via `cudaMemsetAsync`; the scan is skipped.
- **Q4's EXISTS is a left semi-join on lineitem with a late
  commit/receipt filter.** Works because dbgen-generated lineitem
  has enough late rows per order within the 1993-07-01..10-01
  window. For custom data or different windows, the semi-join
  might produce empty output when the spec expects non-empty, or
  vice versa — the rewrite is sound but behaviour depends on data.
- **DuckDB golden generation casts DECIMAL to DOUBLE via a script
  parameter.** If a new query or aggregation produces a column
  DuckDB infers as DECIMAL(>, 0) with integer semantics, the
  generated golden will be DOUBLE-typed and the verifier's
  widening lets an INT64 gamax result pass — which is usually
  fine, but in principle mis-promotion could hide a type bug.
  Review `gen_tpch_golden.py`'s projection if adding queries.
- **The `distinct()` IR factory uses a single-factory pattern where
  output schema depends on whether `keys` cover all input
  columns.** This was deliberately chosen over a
  `DistinctVariant` enum (see the session design discussion).
  SQL `DISTINCT ON (subset) keep all cols` is not expressible;
  add it as a separate factory if ever needed.
- **The runner's `print_result` reads STRING columns via direct
  `cudaMemcpy`** of the helper [data_ptr, meta_ptr] buffer —
  bypassing `string_column_view`'s host ctor because the latter
  NVCC-compiled with a void*-arithmetic codegen quirk that
  segfaulted at runtime when called from that TU. If you refactor
  the reader, don't "simplify" by calling the ctor — test end-to-
  end first.

## What's next (if you're picking up the work)

Nothing is "required" for TPC-H. The engine is complete for that
workload. Strong next directions, in rough priority:

1. **SF100 readiness.** Q5's join-order fix was manual; a join-
   order optimizer would be the cleanest way to make arbitrary
   future queries scale to SF100 and beyond. Also: the 88 GiB
   pool is a hard ceiling — spilling intermediates to host
   memory is a precondition for SF100.

2. **Non-equi residual rule for semi/anti joins (the general
   case).** Q21's rewrite covers a specific algebraic shape. A
   true rule pass that detects non-equi residual + pushes it INTO
   the match predicate would unblock frontend queries with
   arbitrary correlated-subquery shapes.

3. **Real `phys::left_outer_join`.** Paired with a NULL scheme
   (either native nullability or a matched-mask column). Would
   let the builder-side decomposition retire, and would unblock
   outer joins whose ON clause genuinely references both sides.

4. **CodeGen (Phase 3 in the original plan).** The ExecPlan was
   designed to feed either the Executor or a CodeGenerator. Only
   the Executor is in place.

5. **Multi-GPU / multi-node.** The long-term scope. Single-stream
   single-GPU is a steady floor to build on; scale-out is a from-
   scratch layer above the phys op set.

## Reference pointers

- `docs/IR_ARCHITECTURE.md` — architectural reference for the whole
  IR → ExecPlan → phys stack. Read for the layer map and invariants;
  read this doc for the story of how they got there.
- `docs/TPCH_GAPS.md` — the live gap tracker. Every closed gap
  has a dated entry; every remaining gap (if any) has a status +
  fix sketch. Authoritative for "what's the current state."
- `docs/TPCH_DISCREPANCIES.md` — per-query divergences between
  GaMax's builders, Maximus, and the TPC-H spec.
- `docs/GROUPBY_STRING_KEYS.md` — deep dive on the STRING-groupby
  rewrite (the single largest gap-closing arc prior to this one).
