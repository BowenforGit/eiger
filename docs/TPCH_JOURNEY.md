<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# GaMax TPC-H — the whole journey

**End state (2026-04-25):** all 22 TPC-H queries execute on the GaMax
GPU engine and match DuckDB-generated Parquet goldens byte-for-byte
at SF1, SF10, and SF30, inside the verifier's per-type tolerances
(exact for ints / strings; `|Δ| ≤ max(0.01, 1e-6·|expected|)` for
floats). Zero engine-level divergences. Beyond correctness, the
engine now has end-to-end measurement infrastructure: NVTX timeline
instrumentation, an IR-driven NVBench harness with operator-hint
sweeping, a fast failure-mode probe binary, and Caliper-style
per-step CSV breakdown. The verifier, goldens, runner driver, and
all measurement tooling are committed.

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
| Queries verified against a canonical oracle | 1 (Q6 vs pyarrow, manual) | 22 / 22 (DuckDB golden, automated `--verify`) at SF1/10/30/100; 20 / 22 at SF300 (Q1/Q21 OOM). 108/110 default-config OK across all 5 scales, 0 mismatch |
| IR rules | 0 | 5 (G4a, G7, G1-rule, scan-pushdown, CountDistinct rewrite) |
| Phys ops | reorder/filter/project/eq/semi/cross/scalaragg/groupby/sort/topk | + anti, distinct, concat, substring, year_of, in_multiple, and more |
| Operator-hint sweeping | none | per-kind axes via `LowerOptions` + NVBench fixture; one-hot break-down + `bench_defaults` baseline |
| Per-step GPU timing | none | `StepProfileSink` (cudaEvent pairs) + Caliper-style CSV |
| Axis-probe correctness coverage | none | 283/283/279 cells verified vs DuckDB golden at SF=1/10/30 (`tpch_ir_axis_probe --verify`) |
| Q1 SF10 wall (cold) | 27.6 s | 6.6 s (Parquet column projection + threaded decode) |
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

### 7. Performance + measurement infrastructure

**Problem:** with all 22 queries correct, the next questions were
"how long does each query take?", "where does the time go?", and
"what changes when we pick a different operator implementation?"
The engine had no per-step timing, no operator-hint sweeping at the
IR level, and the Parquet reader was a bottleneck nobody had
profiled.

**What shipped (in rough dependency order):**

- **Parquet reader column projection + threaded decode**
  (`tpch_reader.cpp`). Was: `reader->ReadTable(&table)` decoded all
  16 lineitem columns even when the query needed 4, single-threaded.
  Now: cache `parquet::arrow::FileReader` per table, configure
  `ArrowReaderProperties(use_threads=true)`, and call
  `ReadTable(missing_col_indices, &tbl)` to project + parallel-
  decode only the missing columns. **3–4× end-to-end speedup at
  SF10.** Q6: 24.2 s → 5.8 s. Q1: 27.6 s → 6.6 s.

- **Chunked-string path for SF30+** (same file). Arrow's
  `StringArray` uses int32 offsets; columns over 2 GiB total
  (orders.o_comment / lineitem.l_shipinstruct at SF30) made
  `arrow::Concatenate` overflow. Walked Arrow's chunks directly
  into the int64-offset pinned cache we were going to build anyway.
  All 22 queries verify at SF30.

- **NVTX + per-step GPU timing** (`exec/step_profile.{hpp,cpp}` +
  `executor.cpp`). Per-step cudaEvent pairs recorded on the
  executor's stream, resolved at finalize. NVTX ranges
  unconditionally; events only when `phys::Context::profile_sink`
  is non-null. The legacy `query_context::scoped_op` NVTX wrapper
  in `tpch.hpp` was the bootstrap; the executor-side instrumentation
  generalises it. `run_tpch_ir --profile-csv` exposes per-step CSV.

- **IR-driven NVBench harness** (`benchmarks/tpch/tpch_ir_nvbench.cu`).
  Single `NVBENCH_BENCH(tpch_ir)` over `(query, scale, filter,
  project, sort, group_by, eq_join)` axes. Methodology:
  `bench_resolve_sources` (Phase 1, untimed — H2D + Parquet decode)
  + `bench_run_steps` (Phase 2, timed via NVBench's
  `exec_tag::timer`). Skip-by-`n/a`: each operator-kind axis includes
  an `n/a` sentinel; the fixture inspects the lowered plan's
  `StepKind` set and skips cells that don't match. Process-wide
  reader cache keyed by scale, per-cell IrSourceResolver bounds
  device memory. `release_all_mem()` at cell boundary keeps the slab
  pool's high-water mark predictable.

- **Per-payload typed hints + LowerOptions overrides** (the v2 of
  the bench harness). Every ExecPlan payload that has a runtime
  implementation choice gained a typed `hint` field
  (Project → `project_options`, Filter → `select_options`, EqJoin →
  `join_hint`, GroupBy → `groupby_hint`, Sort → `sort_table_options`,
  ...). `LowerOptions` carries one `std::optional<KindHint>` per
  kind; lowering sets `payload.hint = opts.kind.value_or(default)`.
  Phys op signatures grew matching `const HintT& = default` params;
  the executor's `step_dispatch` walker forwards `payload.hint`. The
  bench fixture wires NVBench string axes through `tpch_params::
  parse_algo_value` to typed hints.

- **`Predicate` purely semantic.** `select_options` (the fuse vs
  nofuse axis for filter) used to be baked at construction time
  inside Predicate leaves. Stripped — `Predicate::expr(tree)` no
  longer takes options; `select_to_bitmask(table, pred, options,
  stream)` takes them at evaluation time; `FilterPayload::hint`
  carries them through the executor. Mirrors how `EqJoinPayload`
  carries structural keys but the new hint is a separate field.
  Closes the doc's open question #2 about `select_to_bitmask`'s
  surface.

- **`project=nofuse` Select + materialization fixes**
  (`expr_nofuse.cu`). `evaluate_one_expr`'s switch had explicitly
  skipped `ExprId::Select` (`// skip for now`); wired it up to the
  existing `launch_select` infrastructure (which had a latent
  T1/T2 deduction bug — fixed). `evaluate_expr_nofuse` also
  materialises non-`column_ptr` final results: `column_view` →
  `copy_column`, `scalar` → broadcast via `thrust::fill_n`. All 22
  queries verify with `--no-fuse-project --verify`.

- **Axis-probe binary**
  (`benchmarks/tpch/tpch_ir_axis_probe.cu`). NVBench has ~1 s
  per-cell overhead; for failure-mode coverage where the question
  is "which (q × hint) cells fail at all", that adds up to multi-
  hour sweeps. The probe enumerates the same valid axis configs the
  bench fixture sweeps (one-hot + bench-local defaults, see below),
  calls IR `lower → execute` once per valid cell with try/catch,
  emits one CSV row per cell. **Full SF=1 sweep runs in ~28 s**.
  Wrapper script `script/run_tpch_axis_probe.sh` spawns one subprocess
  per query so a CUDA-context-corrupting cell only contaminates that
  query's remaining cells.

- **Bench breakdown one-hot + bench-local defaults**. Five
  operator-hint axes fully crossed gave up to 240 cells per query
  (2·2·2·6·5 with the supported combinations) — most were redundant
  for break-down analysis since operators are independent in the
  plan. Three changes collapsed this:

  - Each axis grew a `default` sentinel; a new `bench_defaults`
    namespace pins the baseline locally (filter/project=nofuse,
    sort=no_skf, group_by=HASH, eq_join=HASH+GFUR) so the baseline
    is stable regardless of changes to the hidden engine defaults.
  - At-most-one-set rule: any cell with two or more axes in algo
    (non-default, non-n/a) mode is skipped. A query with all five
    StepKinds present yields 1 + 2+2+2+6+5 = 18 cells per scale.
  - Dropped `hash_gftr` from the eq_join axis values + parse table
    — HASH+GFTR is not a supported (algo, mat) pair on the
    inner_eq_join path.

- **Multi-scale axis-probe verification (SF=1 / SF=10 / SF=30)**.
  Probe gained `--verify` (compares each passing result against the
  $TPCH_DATA_PATH/sf<N>/golden/q<NN>.parquet via the existing
  `verify_result`) and `--use-cuda-mm`. Per-query resolver scoping
  fixes the previously-broken pool reset path (the prior single-
  resolver-per-process held device buffers carved from the slab;
  `release_all_mem()` between cells freed the slab while the
  resolver still pointed into it → cudaErrorIllegalAddress →
  poisoned context → "Initializing memory pool fails"). Together
  with `get_mm()`'s cold-context init (cudaMalloc(4)+cudaFree to
  force the lazy-init driver to create the device context before
  cudaMemGetInfo, which otherwise returns (0,0) and sizes the slab
  to 0), the probe runs cleanly at SF=10/30 under both UserSpaceMM
  and CudaMM. Findings:

  | Scale | pass | verify_ok | mismatch | fail | runtime |
  |-----:|----:|---------:|--------:|----:|-------:|
  | SF=1  | 283 | 283 | 0 | 65 | 28 s |
  | SF=10 | 283 | 283 | 0 | 65 | 51 s |
  | SF=30 | 279 | 279 | 0 | 69 | 113 s |

  All passing cells match the golden parquet within
  `verify_result`'s default tolerances (abs_tol=0.01 / rel_tol=1e-6,
  the TPC-H monetary spec). Failed cells are all the documented
  categories: 24 sort-groupby multi-key, 15 sort-groupby STRING,
  16+10 partition-groupby multi-key/STRING (tactical guards), and
  at SF=30 only, 4 partition_hash_join build-bucket overflows on
  Q9 / Q10 / Q21 (5540 / 12202 rows vs the cap of 4096) — same
  documented kernel limitation; key skew per partition is dataset-
  dependent so it doesn't trip at smaller scales.

- **`use_cuda_mm()` runtime memory-manager switch**
  (`memory/mem_manager.{hpp,cu}`). Replaced the old
  `USE_DEFAULT_MEMALLOC` compile-time macro: now a runtime function
  that swaps from `UserSpaceMM` (88 GiB pre-allocated slab) to
  `CudaMM` (per-allocation `cudaMallocAsync`/`cudaFreeAsync`)
  before first allocation. Used to debug kernel OOBs that
  UserSpaceMM's slab boundary would have hidden — a kernel
  writing past its logical bucket but inside the slab looked
  legal. `run_tpch_ir --use-cuda-mm` exposes it.

- **End-to-end + per-step bench at SF=10 (UserSpaceMM)**. NVBench
  sweep with `--csv` for e2e GPU time and `GAMAX_BENCH_STEP_CSV` for
  per-step breakdown. 283 sample-gathering cells, ~95 min wall.
  Pre-flight try/catch in the bench fixture turns throwing cells
  (multi-key/STRING groupby etc.) into clean state.skip without
  poisoning NVBench's stream. Per-query baseline-vs-best summary
  (best is the fastest single-axis variation under the one-hot
  sweep — eq_join dominates; sort_merge / partition_hash variants
  often beat the HASH+GFUR baseline):

  | Q | baseline (ms) | best (ms) | win | best cfg |
  |--:|--:|--:|--:|:--|
  |  1 | 34.86 | 34.85 |  0.0% | filter=nofuse |
  |  2 | 14.69 | 14.31 |  2.6% | eq_join=sort_merge_gfur |
  |  3 |  8.98 |  7.75 | 13.7% | eq_join=partition_hash_gftr |
  |  4 |  4.77 |  4.77 |  0.0% | baseline |
  |  5 | 16.44 | 13.69 | 16.7% | eq_join=sort_merge_gftr |
  |  6 |  1.68 |  1.68 |  0.0% | baseline |
  |  7 | 11.22 | 10.54 |  6.0% | eq_join=sort_merge_gfur |
  |  8 | 14.62 | 12.32 | 15.7% | eq_join=sort_merge_gftr |
  |  9 | 23.77 | 20.81 | 12.5% | eq_join=sort_merge_gfur |
  | 10 | 22.67 | 22.58 |  0.4% | filter=nofuse |
  | 11 |  7.43 |  3.67 | 50.5% | eq_join=sort_merge_gfur |
  | 12 |  5.42 |  5.40 |  0.4% | project=nofuse |
  | 13 |  8.78 |  7.28 | 17.1% | eq_join=sort_merge_gfur |
  | 14 |  2.29 |  2.29 |  0.2% | project=nofuse |
  | 15 |  2.41 |  2.39 |  0.6% | project=nofuse |
  | 16 |  6.81 |  6.73 |  1.1% | eq_join=sort_merge_gfur |
  | 17 | 19.12 | 16.07 | 15.9% | eq_join=partition_hash_gftr |
  | 18 | 16.96 | 13.92 | 18.0% | eq_join=partition_hash_gftr |
  | 19 | 70.93 | 64.10 |  9.6% | eq_join=sort_merge_gfur |
  | 20 |  9.46 |  8.05 | 14.9% | eq_join=sort_merge_gftr |
  | 21 | 48.51 | 48.48 |  0.1% | sort=skf |
  | 22 |  2.34 |  2.34 |  0.0% | filter=nofuse |

  Plotted by `tools/visualize/plot_tpch_bench.py` as a 2×11 grid of
  per-query bar charts (e2e and per-step-stacked breakdown). The
  breakdown plot adds a "best-of-step" composite bar per query —
  the per-step minimum across configs, summed — to show the lower
  bound the engine could reach if it picked the best operator
  implementation per step. The 50% win on Q11 surfaces a clear
  case where HASH+GFUR (the bench-local default) is markedly
  worse than SORT_MERGE+GFUR for this query's join shape.

**Operator-hint sweeping found real bugs:**

The IR-bench probe at SF=1 surfaced six distinct failure patterns
across 2400+ cells under the original full-Cartesian sweep. (The
current one-hot probe yields 65 fails at SF=1/10 and 69 at SF=30 —
same root patterns, narrower projection.) After triage:

- *Pattern A* — sort-based groupby (`sort_gftr`,
  `sort_gftr_opt`, `sort_gfur`) fails on multi-key or STRING groups
  ("Only single-column group keys are supported" / "Invalid
  type_id: 12"). Out of scope per the directive "let groupby_hash
  handle these cases."

- *Pattern B* — partition-based groupby (`partition_gftr`,
  `partition_gfur`) ignores keys past column 0 (the
  `type_dispatcher` only sees `key.get_column(0).type()`); the
  caller's post-processing in `groupby_noskf` then indexes past
  the result's columns. Tactical guard in place; real fix is
  `multikey_compress` inside the partition path. See
  `docs/PARTITION_KERNEL_BUGS.md`.

- *Pattern C* — partition_hash_join's
  `join_copartitions_arr` writes past shared-memory `elem`/
  `payload`/`next` buffers when any partition has more than
  `max_build_bucket_size + 512 = 4608` rows, corrupting adjacent
  shared-mem regions and producing non-deterministic output
  (sometimes wrong values, sometimes `cudaErrorIllegalAddress`).
  Two pieces of fix:

  - Host-side precondition check in `create_work_units`:
    `generate_work_units` now also `atomic_max`'s the per-partition
    build size into a 1-element scalar; `create_work_units` reads
    it back and throws cleanly if any partition exceeds
    `max_build_bucket_size`.

  - Build-vs-probe flips in `tpch_ir.cu` for Q7/Q9/Q10/Q21: each
    had a big intermediate (lineitem-side join result) on the
    LEFT (build) and a small dimension table (`nation`,
    `nation_filt`) on the RIGHT (probe), joined on a 25-distinct-
    value key (n_nationkey). With the dimension table on the
    right, the partition_hash_join hashed the big-side keys
    into 65536 partitions, with the 25 distinct values landing
    in 25 partitions of `big_size / 25` rows each — way past the
    4096 cap. Flipping to put the dimension table on the build
    side is correct semantically (IR uses ColumnIds, not ordinals,
    so downstream references are unaffected) and avoids the
    overflow.

  Together these eliminated all 384 SF=1 overflow failures.

- *Probe diagnostics at SF=1*: 4516 cells × all axes →
  pass=2116 / fail=2400. Every remaining failure is a clean
  unsupported-shape rejection with an actionable message. No
  CUDA-context-corruption cascades.

**What localized the partition_hash_join bug.** `compute-sanitizer`
could not attach to a libcudart-linked binary on this build
(libcudart's static initializers init CUDA before `main`,
disabling the sanitizer hooks). Workaround: `printf` instrumentation
in `generate_work_units` (per-partition `r_len`) and the kernel
entry (work-unit shape). The first oversize bucket showed up in
the first run: `len_R=6100, max_build_bucket_size=4096`. Total
debug time from `printf` to root cause: ~30 minutes.

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
- **`partition_hash_join`'s build-bucket overflow check is host-
  side only.** The kernel still has the latent OOB shared-memory
  write; we throw before launching when any partition exceeds the
  cap. Anyone who widens the cap or adds a workload with even more
  skewed keys re-discovers the same OOB. Real fix tracked in
  `docs/PARTITION_KERNEL_BUGS.md` (kernel bounds-check + spill
  path; or dynamic n_partitions sizing).
- **`partition::groupby` and `partition::groupby_gfur` are
  single-key-only.** Multi-key / STRING inputs now throw a clear
  error. The proper fix (multikey_compress inside the partition
  path) is sketched but not implemented. The `_sort` variants are
  similarly single-key-only and will likely stay so — the user
  directive was "let groupby_hash handle these cases."
- **`tpch_params::algo_map`** (a string→int map keyed by op name)
  remains in use only for the legacy hand-written queries and
  legacy `tpch_nvbench.cu`. The IR pipeline doesn't read it; IR
  hint-sweeping goes through `LowerOptions`. The map can retire
  once the legacy queries either go away or migrate to IR.
- **`compute-sanitizer` does not attach** to any binary that
  dynamically links libcudart on this build environment (CUDA
  initialises before sanitizer's hooks). The `use_cuda_mm()` switch
  was the workaround for one investigation; future kernel
  debugging may need `printf` instrumentation instead. If you
  ever land a fix that lets the sanitizer attach, please update
  `docs/PARTITION_KERNEL_BUGS.md`.

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
- **The `inner_join` IR factory's L/R order matters for performance
  even though it doesn't affect correctness.** The IR uses
  ColumnIds so flipping L/R preserves downstream references, but
  `phys::eq_join` lets `inner_eq_join` use the LEFT table as the
  build side and the RIGHT as the probe side. For partition_hash,
  build > 4096 rows per partition crashes the kernel; for hash, a
  large build is still a memory hit. **Put the smaller table on
  the LEFT.** Q7/Q9/Q10/Q21 had this backwards and silently
  worked on the default hint (because the auto algo picked HASH);
  the bug only surfaced when the bench harness asked for
  partition_hash explicitly. If you write a new IR builder, default
  to `small.inner_join(big, predicate)`.
- **`UserSpaceMM`'s slab hides kernel OOBs.** The 88 GiB
  pre-allocated slab makes any out-of-bounds write inside the
  slab read as legal memory. compute-sanitizer (when it can
  attach) and runtime crashes both miss them — you only see
  garbage output. If a new query produces non-deterministic
  results, swap to `--use-cuda-mm` (or call `use_cuda_mm()`) to
  unmask OOBs that UserSpaceMM is hiding.
- **`StepProfileSink` skips the first sample in aggregating mode.**
  NVBench's `cold` mode doesn't run a separate warmup, so the
  first invocation carries JIT and one-time costs that would
  inflate the per-step mean by 5–50×. The bench fixture's
  `discard_pending` call drops the first sample's events; the
  CSV stats reflect the steady state.
- **`bench_run_steps` requires a sentinel `SourceResolver`.** The
  resolver passed to `bench_resolve_sources` is consumed in
  Phase 1; Phase 2 must NOT re-call it. The executor enforces
  this with a throwing sentinel, but a future caller that
  reuses the resolver across phases will hit the throw and
  should instead split the calls.
- **`run_tpch_ir --eq-join-algo partition_hash_gfur`** picks the
  PARTITION+GFUR pair, but for inputs that match the
  `early_materialization` shape (caller's wrapper builds a
  [key, row_id] table — true for every IR-driven join), the
  partition_hash_join routes to `ieq_partition_hash_join_narrow`
  which is GFTR-equivalent regardless of the requested mat_algo.
  So the bench cell labelled `partition_hash_gfur` is actually
  measuring the GFTR-equivalent narrow path. Worth flagging
  if you ever need a clean GFUR vs GFTR comparison.

## SF=100 readiness — engine optimizations (2026-04-28..29)

The SF=100 axis probe (`tpch_ir_axis_probe --scale 100 --verify`)
turned out to surface fragmentation OOMs rather than capacity
problems on its own. The 88 GiB user-space slab couldn't satisfy
contiguous-allocation requests of 4–5 GiB even when 10–13 GiB of
total free space remained. Pulling on that thread led to a chain
of engine-level optimizations that pay off across all scales,
not just SF=100.

**SF=100 probe baseline (2026-04-28).** 267 / 348 non-skipped
cells pass · 81 fail · 0 mismatches · 460 s wall. The 81 fails
break down as: 21 multi-key/STRING groupby on `sort_*`/`partition_*`
algos (pre-existing limitations, same as SF=30); 18 PARTITION
groupby on multi-col / STRING keys (same); 15 `Invalid type_id: 12`
under `project=fuse` (pre-existing); 8 `partition_hash_join`
build-bucket overflows on Q8/Q11/Q12/Q20 (different queries
than SF=30 — the cardinality-driven bucket skew shifts);
**13 `std::bad_alloc` on Q1**, the new SF=100-specific failure.

### Q19 plan rewrite (commit `09021a0`)

Q19's predicate factors as `shipmode ∈ {AIR,AIR REG} ∧ shipinstruct
= 'DELIVER IN PERSON' ∧ (three brand/container/qty/size disjunctive
branches)`. The shared shipmode/shipinstruct prefix is highly
selective (~0.3% of lineitem at SF=100) and depends only on
lineitem columns. The original plan applied it post-join, which
forced the eq-join to build/probe over the full 600 M lineitem
rows.

Pushed the prefix down to a pre-scan filter; followed by a project
that drops `l_shipmode` / `l_shipinstruct` from the join payload.
The disjunctive branches still touch both sides post-join and stay
post-join. Wall: SF=10 baseline 70.93 → 19.84 ms (3.6×); SF=30
baseline 215 → 31 ms (~7×). The SF=30 hash_gfur baseline now
beats every alternative the operator-hint sweep had previously
found at SF=30 — the filter pushdown saves so much that the
default join algo wins outright.

### Q1 OOM root-cause and the four engine fixes that came out of it

`Q1 SF=100 --verify` failed with a 4514 MiB allocation that the
slab couldn't satisfy. The OOM-dump diagnostic (`mem_manager.cu`,
extended to print requested-size + total/largest free; commit
`1d84c77`) made the failure mode visible: 6 GiB free across 8
holes, largest 2.3 GiB, so any single 4.5 GiB column allocation
is unsatisfiable. NSight Systems profiling showed the failing
step was step[3] — the synthetic Project before GroupBy that the
lowering inserts via `materialize_keys_and_values`.

Step[3] existed because the G4a rule (`rule_count_to_sum`)
rewrites `count_all(*)` to `Sum(lit(int64_t{1}))`, which has a
non-ColRef body. `materialize_keys_and_values` then picked the
Project path (mixed bodies) instead of Reorder, materializing
8 fresh ~4.5 GiB columns at filter cardinality. Fixing this
turned into four cumulative engine changes:

**1. Aliasing Reorder (commit `d32f5fe`).** `phys::reorder` used
to gather every requested ordinal into a fresh column —
`copy_any_column` per ordinal. But `Reorder` is purely a
column-set / column-order rearrangement; it doesn't transform
data. Refactored the executor so that when a Reorder step is
its upstream's last consumer (`plan.values[in].last_used_step ==
step_index`), we *move* the upstream's `owned` table into the
Reorder slot and present a `select_columns` view of it. Slot's
`view` field becomes load-bearing for non-SourceTable steps for
the first time; `view_of()` prefers an explicit view (>=1 column)
over `owned->view()`. DAG-shared upstreams fall back to the
gather copy. Multi-join queries benefit by 1.05–1.45× on
post-join reshape (Q2 1.36×, Q10 1.45× at SF=10).

**2. Aliasing Project (commit `fe41631`).** Same trick, applied
to mixed Projects. When the Project is single-consumer and the
upstream is pure-owned, classify items into ColRef passthroughs
vs. non-trivial (Computed / Predicate / Substring); evaluate the
non-trivial subset via a slimmed `phys::project` call on the
intact upstream view, then *move* the ColRef columns out of
upstream's owned table into the output's column slots. The
non-trivial pass runs *first* so the view it sees is unmodified.
Two ColRef items hitting the same source ordinal: first ref
moves, subsequent refs go through the original copy path
(tracked via a small seen-set). SourceTable upstreams have no
`owned` to move, so the alias path naturally falls back to copy
for Project-after-Scan — preserving the database convention
"scan output is untouched" without a config flag. Q1 SF=10
baseline 31.09 → 22.78 ms (1.36×); the wide pre-groupby Project
is mostly ColRef passthroughs.

**3. Native `aggregation_id::COUNT_ALL` (commit `c253888`).**
The G4a rewrite was disabled by default; `to_physical_agg` now
maps both `AggKind::Count` and `CountAll` to a new phys agg
kind. Phys side: `groupby_noskf` already builds an internal
"ones" column for AVG's denominator (`needs_count = hint.agg_count
|| has_avg`), so we just thread `has_count_all` through the same
flag — no new allocation. COUNT_ALL items skip
`internal_agg_requests` entirely and are routed to the count
column at result-build time via a placeholder + patch-up pattern.
`scalar_aggregate` short-circuits COUNT_ALL to an INT64 [N]
column directly (no input column access). Critical fix in
`groupby_noskf`: size the ones column by `key.num_rows()` instead
of `val.num_rows()` — when COUNT_ALL is the only agg with no AVG
sibling, `val` has 0 columns and `val.num_rows()` returns 0,
which would zero out every count. `materialize_keys_and_values`
emits no body column for CountAll/Count items; `lower_aggregate`
sets `spec.column = -1` (sentinel) for them. Lowering's previous
"rebase specs onto value-only view" loop is gone — specs are now
built val-relative directly. Disabled `rule_count_to_sum` in
`RuleConfig` (kept the rule defined for null-aware-future use).

**4. Materialize-time ColRef body dedup (commit `3c60a14`).**
`materialize_keys_and_values` used to emit one body column per
agg item, even when two items targeted the same source. Q1's
natural form has `sum(l_quantity) + avg(l_quantity)` — the same
l_quantity got materialized twice (a 4.4 GiB cost at SF=100).
`groupby_noskf`'s existing AVG-reuses-SUM dedup at the algorithm
layer was a no-op because the materialize step had already
made them physically distinct columns at different
`column_index`. New `dedup_agg_bodies()` walks `agg_items` once,
builds a `ColumnId → body-position` map; subsequent ColRef
references reuse the prior position. Computed bodies are not
deduped (no expression-equivalence test attempted). The two
layers now coordinate: same `body_position` → same `spec.column`
→ algorithm-layer dedup fires. Q1's IR reverts to the natural
three-avg form without the post-aggregate-divide hack.

### Net

SF=1 / SF=10 / SF=30 axis probes match prior baselines cell-for-
cell (277 / 277 / 273 pass, 0 mismatches). The 6-cell delta vs the
283 / 283 / 279 prior counts is cosmetic — queries whose Aggregate
previously synthesized a `Project(items=[..., lit(1)])` (forcing
the project axis to apply) now synthesize a `Reorder` instead, so
the project-axis labels for those queries are `n/a` rather than
`fuse`/`nofuse`. Same coverage; just labelled differently.

SF=10 baselines, head-to-head against pre-work HEAD on the same
machine: 21 / 22 queries faster, 1 within noise (Q11 0.98×), 0
regressed. Total wall 947 → 791 ms (1.20×) — Q19's 5× pulls
most of that, but Q1 1.07×, Q2 1.36×, Q7 1.26×, Q10 1.45×,
Q12 1.24×, Q14 1.24×, Q15 1.24× are all genuine engine wins.

**Q1 SF=100 still doesn't fit.** The four fixes above shift the
failing allocation from step[3] (the synthetic Project's lit-1
column) to step[4] (the GroupBy's `ones` column for AVG's
denominator) — the same 4.5 GiB shape, same fragmentation wall.
The lit-1 cost is genuinely gone, but Q1 has `avg(l_discount)`,
which independently requires a full-cardinality ones column for
the AVG denominator. The per-group count is implicit in the
hash/sort/partition group structure; eliminating the explicit
ones column is an algorithm-level change inside each groupby
implementation. Out of scope for this iteration.

### Memory pool sizing knobs

`UserSpaceMM` was sized at `0.95 × cudaMemGetInfo's free`. On a
94 GiB H100 NVL that's ~4.7 GiB of headroom the slab didn't need;
the explicit 256 MiB margin (later 128 MiB) covers cudaMalloc /
NCCL scratch with room to spare. The OOM-dump diagnostic now
prints requested-size + total/largest free alongside the existing
free-list dump, so future fragmentation OOMs are diagnosable
without instrumentation. Both changes ship with `1d84c77` /
`fe41631`.

### What didn't work, briefly

- **Bumping pool to 0.99×free / 1.0×free.** Made the failing
  allocation 88 bytes short instead of 800 MiB short. The
  limiting hole was bounded by adjacent allocations, not by
  the pool tail.
- **Pre-projecting `count_one` (a lit-1 column in step[2])** to
  force a Reorder instead of Project at step[3]. Reorder COPIES
  by default — at the time. Net effect was just shifting the
  4.5 GiB allocation from step[3] to step[2]. Subsumed by the
  Reorder-aliasing change.
- **`--use-cuda-mm` at SF=100.** Fails with `cudaErrorInvalidDevice:
  invalid device ordinal` from `fill_n` inside CUB. Separate
  CudaMM bug, not pursued.

## Q20 plan rewrite — semi-join reduction (2026-07-05)

The original Q20 (§2) aggregated *all* date-filtered lineitem into a
per-`(l_partkey, l_suppkey)` sum, then pruned the result down to
forest-part / CANADA-supplier pairs. At SF=100 that group-by runs over
~86 M rows and materializes a table keyed by essentially the whole
partsupp space — even though the query only needs the sum for the
handful of `(part, supplier)` pairs that survive the forest+CANADA
filters.

The current `build_q20` is a semi-join-reduction reformulation. It
builds the small `partsupp_forest_ca` relation first — partsupp
restricted to CANADA suppliers (semi-join against
`supplier ⋈ nation='CANADA'`) and to forest parts (semi-join against
`part LIKE 'forest%'`) — and reuses it in two places:
1. to prune lineitem *before* the per-pair sum (an inner join on
   `(l_partkey, l_suppkey)`; see the multi-key note below), and
2. as one side of the `availqty > 0.5·sum` threshold join.

The CANADA-supplier relation `supplier_ca` is likewise reused: once as
the partsupp reducer, once as the final output side.

### Reusable sub-DAGs

`supplier_ca` and `partsupp_forest_ca` are each consumed by two
parents, so the IR is a DAG, not a tree. This is honored end-to-end
and requires nothing special from the query author — just reuse the
`PlanBuilder` variable:

- **Builder** (`builder.hpp`): a `PlanBuilder` wraps a
  `shared_ptr<PlanNode>`. Passing the same builder to two operators
  makes both parents' `inputs` point at the *same* `PlanNode` address.
- **Rules** (`rules.cpp`): the rewrite engine memoizes on `PlanNode*`
  (`_memo`) and is copy-on-write, so a node reachable from multiple
  parents is rewritten once and the sharing is preserved.
- **Lowering** (`lower_to_exec_plan.cpp`): `lower_node` keeps a
  `unordered_map<const PlanNode*, ValueId> memo`; the first visit emits
  the subtree's steps and records its output `ValueId`, later visits
  return that id without re-emitting. A shared subtree becomes **one**
  chain of ExecPlan steps feeding multiple consumers.
- **Lifetime** (`executor.cpp`): `Value::last_used_step` is set to the
  max step index over *all* consumers (`record_use`). A step may only
  steal/free an input when it is that input's last consumer
  (`last_used_step == step_index`); a shared value therefore stays live
  until its final consumer runs, then is freed. No double-compute, no
  premature free, no UAF.

So the diamond here — `partsupp_forest_ca` feeding both the lineitem
prune and (transitively, via `qty_threshold`) the threshold join —
computes `partsupp_forest_ca` once and holds it until the later of the
two joins.

### gamax mapping points

- **Multi-key semi-join → multi-key inner join.** gamax lowers only
  *inner* multi-key equi-joins (the G1/G7 split rules are Inner-only);
  a multi-key semi-join hits `not_implemented`. The lineitem-pruning
  step is written as an inner join instead. It is exactly equivalent
  because partsupp's `(ps_partkey, ps_suppkey)` is a primary key — so
  `partsupp_forest_ca` has unique join keys and the inner join neither
  drops nor multiplies lineitem rows relative to the semi-join — and
  the subsequent group-by consumes only `l_*` columns. (Same trick the
  old Q20's `with_thr` inner join relied on.)
- **Build/probe order matters a lot.** Each inner join puts the
  smaller relation on the build (left) side. The dominant effect is the
  `supplier ⋈ nation` join: building on `nation='CANADA'` (1 row) and
  streaming supplier is ~2× faster than the reverse. With the large
  side as build, the whole rewrite is actually *slower* than the
  original Q20 — the plan shape only wins once the join orders are
  right.

Wall (hash_gfur baseline, GPU time, source-load excluded):

| SF  | old Q20   | new Q20  | speedup |
|-----|-----------|----------|---------|
| 10  |  11.75 ms |  6.55 ms | 1.8×    |
| 30  |  32.87 ms | 15.1 ms  | 2.2×    |
| 100 | 106.2 ms  | 43.6 ms  | 2.4×    |

The speedup grows with scale because the avoided full-lineitem group-by
is the part that scales with SF. Verified bit-identical to the q20
golden at SF=1 / 10 / 100.

## SF=300 readiness — single-GPU ceiling + the phj >2³⁰ crash (2026-06-22, GB200)

Five-scale single-GPU benchmark on **GB200 (Grace-Blackwell, sm_100,
aarch64, CUDA 13.0, ~186 GiB)**: SF1/10/30/100 run all 22 queries;
**SF=300 is the single-GPU ceiling at 20/22.** Built with
`-DBUILD_TPCH=ON -DBUILD_ALPS=ON -DCMAKE_CUDA_ARCHITECTURES=100`,
Arrow/Parquet 24.0 from the conda `dev` env (see `docs/BENCHMARKING.md`
§1–2). No engine code changed — this arc is a scaling + correctness
campaign, and it surfaced one real bug.

### The phj `>2³⁰`-row illegal-access crash

The SF=300 NVBench single-sweep died after only ~35 valid cells with
`cudaErrorIllegalAddress`. Root cause: **`partition_hash` join faults
at `>2³⁰` rows** — SF=300 lineitem ≈ 1.8 B crosses 2³⁰ = 1.07 B, where
SF=100 (600 M) stayed under. This is the same class as the earlier
"join: fix output under-allocation (OOB) for unique-side joins above
2³⁰ rows" fix — that closed one site; `partition_hash` still faults.
Unlike SF=100's `bad_alloc` throws (host-side, recoverable), an illegal
access is a **sticky CUDA error that poisons the whole process**: every
subsequent cell's `cudaMemGetInfo`/`cudaMalloc` returns 0
(`Memory pool has initial size 0 bytes` cascade) → exit 1. So the
"all SF=100 failures are recoverable" property (BENCHMARKING §11) does
**not** hold at SF=300.

**Fix (no code change): exclude the partition variants** from the sweep
so the faulting cell never runs — `default` (hash) + `sort_*`
alternatives stay:
`-a "eq_join=[sort_merge_gfur,sort_merge_gftr,default,n/a]"
-a "group_by=[sort_gftr,sort_gftr_opt,sort_gfur,default,n/a]"`. Run
**per-query** (`-a query=N`, one process each) distributed across the
reserved nodes for crash isolation, then merge the per-query CSVs.
Merge trap: an all-OOM query (Q1/Q21) emits a 12-column nvbench e2e
header (no timing cols) vs 17 for measured queries — drop those when
merging or pandas chokes. (Also: `pkill -x benchmark_tpch_ir` silently
no-ops — the name is 17 chars, past `pgrep`'s 15-char comm limit; kill
by PID or `pkill -f` a unique substring.)

phj has a **second, scale-independent** limit too: a build-side
**bucket overflow** on skewed FKs (`max_build_bucket_size=4096`), a
clean refuse-to-run throw — seen for Q8/Q10/Q12/Q18/Q20 even at SF=100
(largest partition ~9k–18k rows). Not a wrong answer; HASH/SORT are
correct there.

### The single-GPU memory ceiling (Q1, Q21)

Both OOM at **every** config at SF=300 — not a phj issue, a genuine
working-set-exceeds-186-GiB wall. **Q1**: the project materializes a
~14 GiB full-cardinality `double` column over 1.8 B lineitem rows; it
fails (with fragmentation) at ~161 GiB used. This is the *same*
AVG-ones / full-cardinality-column shape that blocked Q1 at SF=100 in
the previous arc — now the inputs are 3× larger, so even the hash path
that fit at SF=100 doesn't fit at SF=300. **Q21**: ~21 GiB allocation
at ~171 GiB used. Sort-based group_by doesn't rescue them (the pressure
is the project / join output, not the group structure), so they're
shown "no data" at SF=300.

### Correctness (DuckDB goldens, `run_tpch_ir --verify`)

- **Default config** (hash join / hash groupby — the production path),
  all 5 scales: **108/110 OK**, the only 2 non-OK being Q1/Q21 @ SF=300
  (OOM, not wrong). **0 mismatches.**
- **Alternative eq-join algos** (`hash_gftr`, `sort_merge_{gfur,gftr}`,
  `partition_hash_{gfur,gftr}`; phj excluded @ SF=300) at SF=100+SF=300:
  **128 OK, 10 phj-bucket-overflow, 6 OOM, 0 mismatch.** Every algo that
  runs is correct.
- Coverage gap: `run_tpch_ir` can only override `--eq-join-algo` and
  `--no-fuse-project`; it has no flag for group_by/sort/distinct algos,
  so those alternatives are timed-but-unverified. Verifying them needs
  a small runner extension (mirror the bench's `LowerOptions` axes,
  then `--verify`).

## Current limitations and maintainer guidance

The chronology above records the machine and allocator conditions under which
each result was observed. In particular, statements about an SF100 or SF300
"ceiling" are not portable capacity claims: later GB200 and distributed runs
have much more memory and a different execution shape.

The remaining durable limitations are:

1. **Partition kernels.** `partition_hash_join` still has guarded bucket and
   very-large-input failure modes, and partition group-by supports fewer key
   shapes than the default HASH implementation. Keep unsupported combinations
   as explicit rejections until the kernel designs in
   `docs/PARTITION_KERNEL_BUGS.md` are fixed and validated.
2. **Full-cardinality aggregate workspace.** Some single-GPU group-by paths
   build an explicit ones column for AVG/COUNT state. It can create a large
   contiguous allocation and fragmentation pressure even when the logical
   group count is small. Distributed partial aggregation changes the scale at
   which this appears but does not make the local algorithmic cost disappear.
3. **General non-equi semi/anti joins.** Q21's group-by rewrite handles its
   particular EXISTS/NOT EXISTS shape. Arbitrary correlated residuals still
   need a semantics-preserving match-predicate implementation.
4. **True outer joins and NULLs.** Q13's inner-plus-anti-plus-concat rewrite is
   sufficient for TPC-H but is not a general left-outer-join implementation.
5. **Verification of non-total orderings.** Some TPC-H ORDER BY clauses permit
   ties. A row-position comparison can reject two semantically valid outputs;
   distributed Q18 has exposed this. Verification should compare equivalence
   classes under the declared ordering.
6. **Alternative-algorithm verification.** Performance sweeps exercise
   configurations that the ordinary runner does not always expose. Any newly
   enabled hint path should receive golden verification, not merely a
   successful benchmark cell.
7. **Code generation.** ExecPlan still feeds the executor; a separate compiled
   CodeGenerator remains future architecture work.

Multi-GPU and multi-node execution are now implemented. Its current placement,
exchange, failure, and benchmark contracts are documented in
`docs/DISTRIBUTED_TPCH_ARCHITECTURE.md` rather than in this single-GPU history.

## Reference pointers

- `docs/IR_ARCHITECTURE.md` — architectural reference for the whole
  IR → ExecPlan → phys stack. Read for the layer map and invariants;
  read this doc for the story of how they got there.
- `docs/TPCH_DISCREPANCIES.md` — per-query divergences between
  GaMax's builders, Maximus, and the TPC-H spec.
- `docs/GROUPBY_STRING_KEYS.md` — deep dive on the STRING-groupby
  rewrite (the single largest gap-closing arc prior to this one).
- `docs/PARTITION_KERNEL_BUGS.md` — tracking doc for the partition-
  based operator bugs surfaced by the IR-bench axis probe. Bug 2
  (partition_hash_join build-bucket overflow) has a host-side
  precondition fix; Bug 1 (partition::groupby single-key-only)
  has a tactical guard. Both have real-fix sketches.
- `script/run_tpch_axis_probe.sh` — wrapper for the operator-axis
  probe; full SF=1 sweep finishes in ~60 s.
- `benchmarks/tpch/tpch_ir_nvbench.cu` — IR-driven NVBench harness
  with operator-hint axes.
- `docs/DISTRIBUTED_TPCH_ARCHITECTURE.md` — current distributed
  placement, exchange, operator, failure, and timing contracts.
- `docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md` — current distributed
  setup and campaign workflow.
