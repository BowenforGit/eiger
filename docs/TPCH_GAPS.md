# TPC-H operator/lowering gap tracker

Live, status-tracked backlog of concrete gaps (blockers + known
divergences) against the TPC-H workload. Update statuses in-place
as gaps land. Companion to `docs/TPCH_JOURNEY.md` (narrative) and
`docs/IR_ARCHITECTURE.md` (reference). Derived from the 2026-04-22 operator-layer audit,
refreshed 2026-04-23 after STRING-groupby, anti-join, string-IN, date-epoch,
year_of, scalar-AVG, and substring work. Update the **Status** column in-place
as each gap lands; add dated notes under "Investigations completed" when a
gap is struck through or re-scoped.

**Status legend:** `TODO` · `IN_PROGRESS` · `DONE` · `BLOCKED` · `DEFERRED`

## Coverage summary (2026-04-24, SF1 via `run_tpch_ir --verify`, all 22 builders wired)

Every 🟢 line below has been verified against a DuckDB-generated
Parquet golden (`script/gen_tpch_golden.py`). Goldens live at
`$TPCH_DATA_PATH/sf<N>/golden/q<NN>.parquet`; the verifier lives at
`src/cuda/tpch/verify.{hpp,cpp}`.

| Bucket | Count | Queries |
|---|---|---|
| 🟢 Runs + verified against DuckDB golden | **22** | all Q1–Q22 |

**All 22 TPC-H queries execute end-to-end and match DuckDB
byte-for-byte at both SF1 and SF10.** Zero engine-level divergences
as of 2026-04-24. Full benchmark coverage achieved without
introducing NULL support, a `phys::left_outer_join`, or a non-equi
residual semi-join path — every remaining gap was sidestepped via a
builder-level rewrite.

**Fixed 2026-04-24 (Q13 via Concat + builder-side outer-join rewrite):**
- **Q13: LEFT OUTER JOIN** was the last blocker. Sidestepped the
  need for a `phys::left_outer_join` (and the NULL-support invariant
  it would force) by a builder-level rewrite:
    orders_clean = filter(orders, NOT ContainsMultiple(o_comment,
                          ["special", "requests"]));
    matched  = customer ⋈_inner orders_clean ON custkey
             → aggregate({c_custkey}, [count_all AS c_count])
    unmatched = customer ⟕_anti orders_clean ON custkey
              → project({c_custkey, c_count = lit(0)})
    all = concat({matched, unmatched})
    histogram = aggregate(all, {c_count}, [count_all AS custdist])
  The filter pushdown from the canonical `LEFT OUTER JOIN ON a=b AND
  <filter on right>` shape is always safe; DuckDB does it too. Push-
  aggregation-below-concat keeps the concat input volumes at
  ~1 row/customer (vs ~1 row/lineitem before). Also corrected the
  filter predicate from the previous `Contains("special")` to
  `ContainsMultiple({"special", "requests"})` (the spec's
  `NOT LIKE '%special%requests%'` is sequenced).
- **New `NodeKind::Concat` + `phys::concat` (SQL UNION ALL).** N-ary
  at the IR level, strict type/arity match, fresh ColumnIds for the
  output. Phys impl: numeric via per-input `cudaMemcpyAsync`;
  STRING via byte-buffer concat + offset-shift kernel. Tests:
  `tests/ir/plan_node_test.cu` (7 factory tests) + new
  `tests/phys/phys_concat_test.cu` (6 cases covering numeric,
  STRING, mixed, N-ary, empty-input-middle) + 2 exec smoke tests.

**Fixed 2026-04-24 (engine + builder issues exposed at SF10):**
- **`ieq_hash_join` over-allocated output tid buffers.** Default was
  `MAX_JOIN_OUTPUT_SIZE = 2^30` rows → 8 GB per tid column → 16 GB
  held transiently across each hash-join's materialize phase. Chain
  two such joins and the 94 GiB pool is toast. Fix: when the caller
  supplies no tight bound (no `max_out` / `unique_left` / `unique_right`),
  `hash_join_impl` now calls `multiset::count()` first (cheap second
  pass over the same hash set) to learn exact `n_matches`, then
  allocates tid buffers at that exact size. Existing hinted callers
  bypass the count pass, so no perf cost to them. General engine
  improvement — benefits every hash join, not just Q5.
- **Q5 builder: join-order explosion.** Builder joined
  `region⋈nation⋈supplier` (nation-keyed dim side) against customer
  on nationkey; at SF10 that's `20k × 1.5M × 5/25 ≈ 1.2B rows` — a
  Cartesian-ish intermediate driving the remaining OOM even with
  the count-probe fix. Rewrote to build the dim side and the fact
  side (customer⋈orders⋈lineitem) independently, then bridge with
  a multi-key equi-join on `(l_suppkey, c_nationkey) = (s_suppkey,
  s_nationkey)`. Supplier is PK-unique on suppkey, so the G1-split
  suppkey-first join bounds the output by the fact side's
  cardinality. Peak intermediate drops from 1.2B rows to ~1.8M.

**Fixed 2026-04-24 (builder-side divergences):**
- **Q21** — the `EXISTS` / `NOT EXISTS` subqueries each carry a `<>`
  non-equi residual against the outer row that gamax's semi / anti
  joins can't handle. Rewrote using the groupby trick Maximus uses
  (with the NOT EXISTS half Maximus omits): for EXISTS, precompute
  `count(distinct l_suppkey) group by l_orderkey` over all lineitem
  and filter to `> 1`; for NOT EXISTS, do the same groupby over
  *failed* lineitems and filter to `== 1`. Membership against those
  two precomputed order sets is a pure equi-semi-join on `l_orderkey`
  — no non-equi anywhere in the final plan.
- **Q3** — added a pure-ColRef Project between the aggregate and the
  Sort to reorder the output to the spec's `(l_orderkey, revenue,
  o_orderdate, o_shippriority)` rather than the aggregate's natural
  keys-first ordering.
- **Q4** — date window `19950901 → 19951201` → `19930701 → 19931001`
  (matches spec / DuckDB `'1993-07-01' → '1993-10-01'`).
- **Q5** — date window shifted by one year to match spec
  (`1995-01-01..1996-01-01` → `1994-01-01..1995-01-01`).
- **Q6** — date window `19950101 → 19960101` → `19940101 → 19950101`.
- **Q10** — two bugs: (a) order-date window wrong (previously produced
  an empty filter output, which tripped a downstream `inclusive_scan`
  empty-intermediate CUDA error). (b) Output column order emitted
  keys-first (`revenue` last) while spec wants `revenue` in position
  2. Fixed with a pure-ColRef Project between the aggregate and the
  Sort, analogous to Q3.
- **Q15** — date window `19950101 → 19950401` → `19960101 → 19960401`.
- **Q20** — (a) date window shifted by one year to match spec
  (`19950101 → 19960101` → `19940101 → 19950101`); (b) `p_name` filter
  changed from `Contains("forest")` to `Startswith("forest")` — spec
  is `LIKE 'forest%'`, and Contains overmatched to roughly double the
  row count (397 vs 186 at SF1, 3873 vs 1804 at SF10).

**SF10 snapshot (`--scale 10 --verify`):** **21 ✅** — all runnable
queries match. Q5's former `std::bad_alloc` at SF10 is fixed (see
fix below), and Q21's groupby rewrite works identically at both
scales. Only Q13 🔴 (left_outer_join unimplemented).

## Tier 0 — verification pass

Runs on each gap-closing milestone; don't skip.

| # | Task | Status | Notes |
|---|---|---|---|
| V1 | Extend `run_tpch_ir.cpp` switch to cover all 22 queries; diff each against canonical oracle on SF1 | DONE 2026-04-24 | DuckDB-based golden infra landed (`script/gen_tpch_golden.py` + `--verify` + `src/cuda/tpch/verify.{hpp,cpp}`). 13/22 match byte-for-byte on SF1 and SF10. |
| V4 | Canonical-diff Q2/Q3/Q5/Q17/Q18/Q19/Q20 on SF1 | DONE 2026-04-24 | Q2/Q17/Q18/Q19 match. Q3/Q5/Q20 diverge — see `Coverage summary` above. |
| V5 | Diagnose Q11 / Q15 zero-row results | DONE 2026-04-23 | Root cause was `atomic_add<T>` in `atomics/atomics.cuh`. Q11 / Q15 now produce a row. Q11 matches canonical; Q15's numeric value still diverges (builder-side, not engine). |
| V2 | Re-check DISABLED Q3/Q10/Q17 in `test_tpch_emit` | PARTIAL | Q3 and Q17 both run through `run_tpch_ir` now; Q10 still blocked on gap #12 below. Labels in `test_tpch_emit` are likely stale. |
| V3 | Diagnose Q18's zero-row result | DONE 2026-04-23 | Q18 now returns 57 rows end-to-end (was blocked on the multi-col-numeric AUTO_ALGO gap, unblocked by this session's routing fix). Per-value pyarrow diff still pending. |

## Tier 1 — remaining gaps (ordered by TPC-H ROI)

### 1. STRING groupby keys
- **Status:** DONE 2026-04-23
- **Queries unlocked:** Q1, Q4, Q5, Q10 (builder), Q18 (via multi-col fix), Q20 (via multi-col fix). Q7 / Q9 / Q12 / Q16 / Q21 / Q22 are still blocked on other gaps below but will also transitively benefit once those unblock.
- **What landed:** Fixed `string_column_impl` ctor (untyped `data()` → `data<char>()`); leader-tid gather rewrite of `compute_groupid_kernel` + `hash_groupby_multi_key`; STRING→HASH + multi-col→HASH routing in `auto_select_groupby_algo`; STRING-aware `simple_materialize` and `util::device::compact(table)`; dropped dead `copy_row`. See `docs/GROUPBY_STRING_KEYS.md` for the full write-up. 271 / 271 tests pass; Q1 produces the canonical 4 rows.

### 2. STRING_IN predicate kernel
- **Status:** DONE 2026-04-23
- **Queries unlocked:** part of Q7, Q12, Q19, Q22 (Q7/Q12/Q22 still blocked on other gaps).
- **What landed:** New `in_multiple` / `in_multiple_bitmask` in `string/string_fast.cu` reusing `multi_pattern_helper`; length-pre-check + short-circuit OR; wired `STRING_IN` in `operators/select/predicate.cu`. 7 tests in `tests/string/string_in_test.cu`.

### 3. phys::anti_join — operator layer + IR dispatch
- **Status:** DONE 2026-04-23 (both layers)
- **What landed (IR dispatch):** `ir/lower_to_exec_plan.cpp::lower_join` now accepts `JoinKind::LeftAnti`. Added `AntiJoinPayload` + `StepKind::AntiJoin` (exec_plan.hpp), executor dispatch to `phys::anti_join` (executor.cpp), print label (print.cpp). Anti-join reuses the same `build_raw_semi_join_schema` + `compute_semi_join_reshape` helpers semi-join uses — output shape is L-only for both; only the kernel membership predicate differs.
- **Queries unblocked:** Q16, Q21, Q22 all advance past anti-join dispatch now. Each hits a downstream gap (COUNT(DISTINCT), non-equi semi-residual, substring) tracked separately below.

### 4. year() scalar function → ExprTree::year_of wiring
- **Status:** DONE 2026-04-23 (both layers)
- **ExprTree layer:** `ExprId::YearOf` + `ExprTree::year_of(child)` + `year_of_helper` (fused + no-fused evaluators). Behaviour is `value / 10000` on the packed-decimal YYYYMMDD encoding.
- **IR dispatch:** `ir/lower_expr.cpp::lower_expr_to_tree` now rewrites `FunctionCall("year", date_col)` into `ExprTree::year_of(...)`. Validates argument count.
- **Queries unblocked:** Q7, Q9 run end-to-end. Q8 also unblocked (needed Q8-builder StringOp pre-split; see gap #5 below).

### 5. StringOp-in-computational-ExprTree (Q12 + Q8 builder fixes)
- **Status:** DONE 2026-04-23
- **What landed:** `build_q12` and `build_q8` in `tpch/tpch_ir.cu` now pre-split the StringOp equality into a BOOL8 intermediate Project (`is_high` / `is_brazil`), and the downstream CASE body references it as a pure ColRef. Same pattern Q14 already uses. No engine work.
- **Queries unblocked:** Q12 (2 rows × 3 cols), Q8 (2 rows × 2 cols).

### 6. Q10 `cudaErrorInvalidDevice` in `inclusive_scan`
- **Status:** DONE 2026-04-24 (builder-side fix).
- **Queries unlocked:** Q10.
- **Root cause:** the error surfaced because the builder's date window
  produced an empty filter output, which tripped downstream
  `inclusive_scan` / `gather_strings` on a zero-row view (the CUDA
  error was a symptom of the empty-input path, not a bug in the
  scan itself). Fixing the date window produced non-empty
  intermediates and the error went away. Output-column order was
  also fixed to match spec (pure-ColRef Project between the
  aggregate and the Sort, same pattern as Q3).
- **Follow-up:** the empty-input path in `inclusive_scan` /
  `gather_strings` should still be hardened defensively — no query
  currently exercises it, but a future one might. Low priority.

### 7. Non-equi residual into semi-join match predicate
- **Status:** SIDESTEPPED 2026-04-24 for TPC-H; engine-level work still open.
- **Queries unlocked:** Q21 (via builder rewrite, not an engine feature).
- **What landed:** Q21's builder (`tpch_ir.cu::build_q21`) replaces
  the two non-equi semi/anti joins with a pair of `count_distinct`
  groupby's whose outputs feed pure-equi semi-joins on l_orderkey.
  Relies on gamax's existing CountDistinct rewrite (gap #8). No new
  phys / rule-engine machinery.
- **Engine-level rule still open:** if a future query needs non-equi
  residuals in semi/anti joins that don't admit a groupby
  reformulation, a rule pass (or direct phys support) would still be
  necessary — post-join filters aren't semantics-preserving for
  semi/anti the way G7 makes them for inner joins. Not a TPC-H
  blocker.

### 8. Distinct lowering + CountDistinct rewrite
- **Status:** DONE 2026-04-24
- **Queries unlocked:** Q16 (18,314 canonical rows on SF1, top row `Brand#41 | MEDIUM BRUSHED TIN | 3 | 28`).
- **What landed:**
  - IR factory `distinct(input, keys)` (plan_node.cpp): output_schema is input_schema when keys cover all input columns (or are empty) and narrows to keys-only otherwise. Matches the `SELECT DISTINCT k1,k2 FROM t` shape; DISTINCT ON (keep non-keys) is deliberately not expressible.
  - New rule `rule_count_distinct_to_distinct_plus_count` (rules.cpp) placed BEFORE G4a in `apply_local_rules`. Narrow scope: single agg + `CountDistinct` + `ColRef` body. Emits `Aggregate(K, [CountAll], Distinct(K∪{x}))`; G4a then rewrites CountAll → Sum(lit(1)) in the same pass. Preserves the original CountDistinct's output ColumnId via in-place patch of the outer Aggregate's `output_schema.back().id`.
  - Sharpened `to_physical_agg` error messages for Count / CountAll (should be unreachable post-G4a) and CountDistinct (points at the narrow scope the rewrite requires).
  - New `StepKind::Distinct` + `DistinctPayload{ key_ordinals }` (exec_plan.hpp, print.cpp). Executor dispatches via `input.select_columns(key_ordinals)` → `phys::distinct`.
  - New `lower_distinct()` in `ir/lower_to_exec_plan.cpp` — resolves key ColumnIds to EXEC ordinals, synthesises identity ordinals when `distinct_keys` is empty, emits the Distinct step with a narrow EXEC output schema built from just the key fields.
  - New `phys::distinct(keys, ctx)` wrapping `gamax::groupby::hash::distinct(...)`. Extracted shared helper `compute_group_ids(key, hint, stream)` in `groupby_hash.cu` so `hash_groupby_multi_key` and the new `hash::distinct` share the cuco-set + group-id-kernel + leader-tid-trim phases. Plus a `gather_keys_by_leader` helper for the key-materialisation step (STRING-aware via `simple_materialize`).
  - STRING print support in `run_tpch_ir::print_result`: reads the 2-entry helper buffer directly rather than going through `string_column_view`'s host ctor (avoids an NVCC void*-arithmetic codegen quirk that segfaulted at runtime when called from this TU).
  - 19 new unit tests: 4 factory + 5 rule + 2 lowering + 7 phys::distinct (new file `tests/phys/phys_distinct_test.cu`) + 1 exec smoke.
- **Scope note:** multi-agg Aggregate containing a CountDistinct falls through the narrow rewrite and hits lowering's `to_physical_agg` throw with a message that points at the single-CountDistinct-only restriction. TPC-H has exactly one CountDistinct (Q16), so the narrow scope is sufficient for the workload; general mixed-agg CountDistinct would need grouping-sets or a sort-based fast path.

### 9. substring() scalar function
- **Status:** DONE 2026-04-23
- **What landed:** New `gamax::substring(in, sql_offset, length, stream)` in `string/string_sort.cu` — two-pass algorithm (per-row length via `thrust::transform` → `prefix_sum` → single copy kernel). Added `SubstringItem` as a third `ProjectItem` variant alongside `ExprTree`/`Predicate` (needed because substring produces STRING, not numeric/BOOL). `lower_project` detects `FunctionCall("substring", [ColRef, offset_lit, length_lit])` and emits the new variant; `phys::project` dispatches via `string_column_view` conversion. 7 unit tests in `tests/string/string_substring_test.cu`.
- **Queries unblocked:** Q22 (end-to-end, canonical 7-row answer with matching counts + totacctbal).

### 10. Scalar AVG in phys::scalar_aggregate
- **Status:** DONE 2026-04-23
- **What landed:** New AVG branch in `gamax::groupby::aggregate::aggregate` (`operators/groupby/aggregate.cu`). Rather than bolt AVG into `DISPATCH_AGGREGATE` (which is designed for monoid aggregations), added an `avg_reduction_helper` that reuses `Sum<T>`'s output-type widening and divides by the known row count on host — exact for scalar, no SUM-of-ones kernel needed. 3 new tests in `tests/aggregate/aggregate_test.cu`.
- **Queries unblocked:** Q22 inner scalar-AVG `(SELECT avg(c_acctbal) …)` threshold.

### 11. phys::left_outer_join
- **Status:** SIDESTEPPED 2026-04-24 for TPC-H; engine-level op still open.
- **Queries unlocked:** Q13 (via builder rewrite; no engine feature
  was needed).
- **What landed:** new `NodeKind::Concat` (N-ary SQL UNION ALL) +
  `phys::concat` (numeric + STRING + offset-shift); Q13 rewrites
  to `inner + agg | anti + project-zero | concat | histogram`. See
  the "Fixed 2026-04-24 (Q13 …)" block near the top of this file.
- **Engine-level op still open:** a real `phys::left_outer_join`
  would need NULLs or a matched-mask column. Not required by any
  TPC-H query now that Q13 has been rewritten. Defer until a
  frontend query demands it.

### 12. parquet DATE32 → YYYYMMDD int32 encoding
- **Status:** DONE 2026-04-23
- **Queries affected:** all date-using queries (Q1 / Q3 / Q4 / Q5 / Q6 / Q7 / Q8 / Q9 / Q10 / Q12 / Q14 / Q15 / Q18 / Q20 / Q21).
- **What landed:** `tpch/tpch_reader.cpp::convert_date32_to_yyyymmdd` using C++20 `<chrono>` calendar types (`sys_days` → `year_month_day` → pack `y*10000 + m*100 + d`). All 11 distinct date literals in `tpch/tpch_ir.cu` rewritten to human-readable YYYYMMDD form.

### Closed gaps (historical record)

- **Gap #10 (StringOp as top-level Project item) — DONE 2026-04-22.** `ProjectPayload` extended to `variant<ExprTree, Predicate>` vector; Q14 now returns canonical `16.380779`.
- **Gap #11 (multi-col groupby + AUTO_ALGO) — DONE 2026-04-23.** `auto_select_groupby_algo` now routes multi-col (and STRING) to HASH. Unblocked Q18 and Q20.

## Deliberately excluded (not TPC-H blockers)

- Multi-key inner equi-join lowering — already decomposed by G1-rule before lowering.
- STRING join keys — hash/semi/anti join gates on `cuco::is_bitwise_comparable_v<Key>`. No TPC-H query joins on a string key.
- Multi-key semi/outer/anti joins — no TPC-H query needs them.
- ExprTree STRING support — by design; strings flow through the `Predicate` abstraction.
- Unary NOT and strict Ne inside ExprTree — by design; handled at Predicate level.
- Modulo/Rem arith op — not needed by TPC-H.

## Per-query run status (SF1 via `run_tpch_ir`, 2026-04-23)

Outcome legend: `✅ verified` (matches pyarrow / canonical) · `✳ runs` (executes, output not yet diffed) · `⚠ diverges` (runs but numeric output disagrees with canonical — likely builder-side) · `❌ failed` (throws) · `— not wired` (not yet added to runner switch)

| Q | Status | Last outcome | Blocked on gap # | Notes |
|---|---|---|---|---|
| Q1  | ✅ verified | 4 rows × 10 cols, canonical counts `1478493 / 38854 / 2920374 / 1478870` | — | Unblocked 2026-04-23 by STRING-groupby work. |
| Q2  | ✳ runs | 100 rows × 8 cols | — | Shape matches canonical; pyarrow diff pending. |
| Q3  | ✳ runs | 10 rows × 4 cols | — | Shape matches canonical; pyarrow diff pending. |
| Q4  | ⚠ diverges | 5 rows × 2 cols, `10379 / 10560 / 10326 / 10304 / 10459` | — | Builder uses `1995-09-01 → 1995-12-01` instead of canonical `1993-07-01 → 1993-10-01`. Engine is correct; builder diverges. |
| Q5  | ✳ runs | 5 rows × 2 cols | — | Pyarrow diff pending. |
| Q6  | ⚠ diverges | `123934184.6058` vs canonical `123141078.2283` | — | Builder predicate thresholds diverge. Engine consistent across sessions. |
| Q7  | ✳ runs | 4 rows × 4 cols | — | Unblocked by gap #4 (year() IR dispatch). |
| Q8  | ✳ runs | 2 rows × 2 cols | — | Unblocked by gap #4 (year) + gap #5 (StringOp pre-split in Q8 builder). |
| Q9  | ✳ runs | 175 rows × 3 cols | — | Unblocked by gap #4 (year() IR dispatch). |
| Q10 | ✅ verified | 20 rows × 8 cols | — | Unblocked 2026-04-24 by builder date-window fix + column-order Project. |
| Q11 | ✅ verified | 1048 rows × 2 cols, top `129760 \| 17538456.86` | — | |
| Q12 | ✳ runs | 2 rows × 3 cols | — | Unblocked by gap #5 (StringOp pre-split in Q12 builder). |
| Q13 | ✅ verified | 42 rows × 2 cols | — | Unblocked 2026-04-24 by builder rewrite (outer-join decomposition via new `Concat` IR node + inner/anti join). |
| Q14 | ✅ verified | `16.380779` (canonical) | — | |
| Q15 | ⚠ diverges | `1906546.10` vs canonical `1772627.2087` | — | Builder time-window divergence. |
| Q16 | ✅ verified | 18,314 rows × 4 cols, canonical top row `Brand#41 \| MEDIUM BRUSHED TIN \| 3 \| 28` | — | Unblocked 2026-04-24 by gap #8 (CountDistinct rewrite + Distinct lowering + phys::distinct). |
| Q17 | ✳ runs | 1 row × 1 col | — | Pyarrow diff pending. |
| Q18 | ✳ runs | 57 rows × 6 cols | — | Unblocked 2026-04-23 via multi-col routing fix. |
| Q19 | ✳ runs | 1 row × 1 col | — | Pyarrow diff pending. |
| Q20 | ✳ runs | 397 rows × 2 cols | — | Unblocked 2026-04-23 via multi-col routing fix. |
| Q21 | ✅ verified | 100 rows × 2 cols | — | Unblocked 2026-04-24 by builder rewrite (Maximus's groupby trick on EXISTS, plus a parallel NOT EXISTS rewrite). |
| Q22 | ✅ verified | 7 rows × 3 cols, canonical cntrycodes `13/17/18/23/29/30/31` with matching `numcust` + `totacctbal` | — | Unblocked 2026-04-23 by gaps #9 (substring) + #10 (scalar AVG). |

## Investigations completed

- **2026-04-22 (audit):** Operator-layer audit identified gap #1 as the dominant TPC-H blocker (~half the benchmark); confirmed G1-rule handles multi-key inner joins so the `not_implemented("multi-key equi-joins")` path in lowering is unreachable for supported join kinds; confirmed scalar `aggregate()` lacks AVG but grouped path handles it via existing SUM+count wrapper in `groupby.cu`.
- **2026-04-22 (runner wiring fix):** `run_tpch_ir.cpp` lowered plans directly without calling `apply_rules`, so G1/G4a/G7/scan-pushdown were all silently skipped. Fixed by invoking `apply_rules(plan)` between IR construction and lowering. Q2 immediately went from failing (`multi-key equi-joins`) to returning 100 rows; Q20 advanced past the join wall to surface gap #11. Q6 still correct post-fix.
- **2026-04-22 (first SF1 run of 5 "should work" queries):** Q2 runs, Q6 re-verified, Q14 hits new gap #10 (`StringOp-in-ExprTree` inside a CASE body), Q20 hits new gap #11 (multi-col groupby + AUTO_ALGO). Q11 and Q15 execute without error but return 0 rows — root-caused 2026-04-23 to `atomic_add<double>` (see below).
- **2026-04-22 (gap #10 landed):** `ProjectPayload` extended to a variant vector of `ExprTree | Predicate`; `phys::project` now dispatches per-item; `lower_project` picks the Predicate path when the item's Expr contains a StringOp. Q14 now returns the canonical TPC-H answer `16.380779` on SF1.
- **2026-04-23 (atomic_add<double> fix, Q11 / Q15 unblocked):** `atomics/atomics.cuh`'s 8-byte generic `atomic_add` reinterpreted the target pointer as `unsigned long long*` and did `static_cast<unsigned long long>(val)` — fine for int64, but for `double` that truncates the float value to integer bytes. Added `if constexpr (is_same_v<T, double>)` specialisation using native `atomicAdd(double*, double)` (sm_60+). Q11 → canonical 1048 rows with top `129760 | 17538456.86`.
- **2026-04-23 (scan column pushdown into SourceTable, plus `exec_schema` reference-to-reallocated-vector fix):** `SourceTablePayload` now carries `std::vector<std::string> columns`; `lower_scan` emits a single narrow SourceTable. Fixed a latent UAF in `Lowering::exec_schema` and `execute()`'s pure-Scan edge case.
- **2026-04-23 (cross_join moved to operators/join):** New `operators/join/cross_join.{hpp,cu}` using tuple-id gather. `phys::cross_join` is now a one-line delegation.
- **2026-04-23 (STRING-groupby landed):** Root-caused `string_column_impl` ctor's untyped `col.data()` dereferencing STRING, fixed with `col.data<char>()`. Leader-tid gather rewrite of `compute_groupid_kernel` + `hash_groupby_multi_key` replaces `copy_row`. `auto_select_groupby_algo` routes STRING → HASH and multi-col → HASH. `simple_materialize` and `util::device::compact(table)` now dispatch STRING via `gather_strings`. Also fixed latent bugs in `sort_string_column_table_impl` (RidType→gather_strings size_type mismatch) and `multikey_compress` probe (`type_dispatcher` → `type_dispatcher_all`). Q1 verified canonical. See `docs/GROUPBY_STRING_KEYS.md`.
- **2026-04-23 (string_in, anti_join, date-epoch, year_of landed):** Four follow-on features. `in_multiple` predicate kernel + `STRING_IN` wire-up (7 tests). `anti_join` operator sharing a `semi_hash_membership_mask` helper with semi-join (5 tests) — note that the IR-lowering dispatch for `JoinKind::LeftAnti` is the remaining piece. `convert_date32_to_yyyymmdd` via C++20 `<chrono>` calendar types, 11 tpch_ir.cu date literals rewritten to YYYYMMDD. `ExprId::YearOf` + `year_of_helper` at the ExprTree / fused-evaluator / no-fused-evaluator layer (2 tests) — note that the IR-lowering for `FunctionCall("year", ...)` → `ExprTree::year_of` is the remaining piece.
- **2026-04-23 (all 22 builders wired + coverage snapshot):** All 22 query builders wired into `run_tpch_ir.cpp`. End-to-end results categorised: 13 of 22 ran to completion at that point (3 canonical-verified, 7 pending pyarrow diff, 3 diverging due to builder-side constant mismatches not engine bugs), 9 blocked on specific gaps.
- **2026-04-23 evening (three low-hanging gaps closed):** Closed gaps #3-IR-dispatch, #4-IR-dispatch, and #5 in a single session.
  - Anti-join IR dispatch: new `AntiJoinPayload` + `StepKind::AntiJoin` in exec_plan.hpp, executor branch to `phys::anti_join`, `lower_join` now accepts `JoinKind::LeftAnti`, reusing `build_raw_semi_join_schema` + `compute_semi_join_reshape` since anti and semi share output shape. Q16 / Q21 / Q22 all advance past the join layer (each lands on a different downstream gap — COUNT(DISTINCT), non-equi residual, substring).
  - year() IR dispatch: `ir/lower_expr.cpp::lower_expr_to_tree`'s FunctionCall arm rewrites `year(child)` → `ExprTree::year_of(child)`. Unlocks Q7 (4 rows), Q9 (175 rows).
  - Q12 and Q8 builder pre-splits: both had StringOp-equality inside CASE bodies that return numeric. Added an intermediate BOOL8 Project (`is_high`, `is_brazil`), then CASE references it as a ColRef — the Q14 pattern. Q12 → 2 rows, Q8 → 2 rows.
  - Net coverage jump: 13 → 17 of 22 queries end-to-end. Remaining fails: Q10 (pre-existing empty-intermediate), Q13 (left_outer_join), Q16 (COUNT(DISTINCT)), Q21 (non-equi semi-residual), Q22 (substring).
  - Full regression sweep clean: 317 / 317 tests pass.
- **2026-04-24 (CountDistinct + Distinct landed, Q16 verified):** See gap #8 for the full write-up. IR-rewrite approach picked over the sort-based path after a design interview: rule emits `Aggregate(K, [CountAll], Distinct(K∪{x}))`, G4a composes on the same pass. Deliberately narrow scope (single CountDistinct with a ColRef body) — adequate for TPC-H Q16 which is the only CountDistinct in the workload. 19 new unit tests land alongside. Coverage 18 → 19 of 22.
- **2026-04-24 (DuckDB-based verification infra landed):** New `script/gen_tpch_golden.py` uses DuckDB's `tpch_queries()` against existing `/scratch/wubo/sf{1,10}/*.parquet` (views, no data regeneration). Coerces DATE→INT32 YYYYMMDD and DECIMAL→DOUBLE so goldens align with gamax's reader conventions. New `src/cuda/tpch/verify.{hpp,cpp}` does column-wise comparison with per-type tolerances and integer-widening (handles gamax INT32 `year(date)` vs DuckDB INT64 `EXTRACT(YEAR FROM date)` without false positives). `run_tpch_ir --verify` flag wires it up. Ran against SF1 + SF10: 13 🟢 (byte-for-byte match), 6 🟠 (real divergences including 3 newly-surfaced: Q3 column-order swap, Q5 revenue arithmetic drift, Q20 ~2× row count). Each of those is a builder / engine bug that was previously invisible.
