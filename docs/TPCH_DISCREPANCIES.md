# TPC-H builder discrepancies — GaMax vs Maximus vs TPC-H spec

**Purpose.** When a query's output disagrees with pyarrow (spec) or with Maximus,
check here *first*. This is a frozen snapshot of structural/constant/semantic
differences between GaMax's builders and the Maximus reference, audited
2026-04-22. Update the **Status** column in-place as items are resolved.

**Sources:**
- GaMax builders: `src/cuda/tpch/tpch_ir.cu` (per-query line numbers in table
  below).
- Maximus reference: Maximus GitLab, `src/maximus/tpch/tpch_queries.cpp` —
  `q1()`..`q22()` functions. Snapshot cached during audit at
  `/tmp/maximus_tpch_queries.cpp` (volatile — re-fetch if needed).
- TPC-H spec: dbgen defaults. Relevant constants enumerated in the TPC-H
  Benchmark Standard Specification revision 3.0.

**How to read this file when debugging a wrong result:**
1. Find the query's section.
2. Look at every discrepancy for that query — pick the one whose column
   (`l_shipdate`, `r_name`, join key, etc.) relates to your symptom.
3. Each discrepancy records what GaMax does, what Maximus does, and (when
   they differ) what the TPC-H spec says. The value you're getting probably
   matches one of the three. That tells you which "truth" you're on.

**Day-serial cheat sheet** (days since 1970-01-01, used in GaMax date
literals):

| Date | Day | | Date | Day |
|---|---|---|---|---|
| 1993-07-01 | 8582 | | 1995-09-01 | 9374 |
| 1993-10-01 | 8674 | | 1995-10-01 | 9404 |
| 1994-01-01 | 8766 | | 1995-12-01 | 9465 |
| 1994-04-01 | 8856 | | 1996-01-01 | 9496 |
| 1995-01-01 | 9131 | | 1996-04-01 | 9587 |
| 1995-03-15 | 9204 | | 1996-12-31 | 9861 |

---

## Category A — unambiguous bugs (GaMax wrong vs BOTH Maximus and spec)

These produce incorrect SF1 results. Fix on sight.

| Q | GaMax loc | Field | GaMax value | Maximus / spec |
|---|---|---|---|---|
| Q4  | `:352-353` | `o_orderdate` range | `[9374, 9465)` = 1995-09-01…1995-12-01 | `[8582, 8674)` = 1993-07-01…1993-10-01 |
| Q5  | `:383-384` | `o_orderdate` range | `[9131, 9496)` = 1995 | `[8766, 9131)` = 1994 |
| Q6  | `:415-416` | `l_shipdate` range  | `[9131, 9496)` = 1995 | `[8766, 9131)` = 1994 |
| Q8  | `:519`     | `r_name` filter      | `'AMERICA'` | `'AFRICA'` (Maximus); **spec is `'AMERICA'`** — if following spec, GaMax is correct here. |
| Q10 | `:635-636` | `o_orderdate` range | `[9374, 9465)` = Sep–Dec 1995 | `[8674, 8766)` = 1993-10-01…1994-01-01 |
| Q15 | `:833-834` | `l_shipdate` range  | `[9131, 9221)` = Jan–Apr 1995 | `[9496, 9587)` = Jan–Apr 1996 |
| Q20 | `:1045-1046` | `l_shipdate` range | `[9131, 9496)` = 1995 | `[8766, 9131)` = 1994 |

**Note on Q8:** Maximus uses `'AFRICA'` but TPC-H spec defaults to `'AMERICA'`.
GaMax currently matches the spec, not Maximus. This row is "unambiguous" only
if Maximus is the oracle; under a spec-based oracle, GaMax is correct.

**Status:** all TODO.

---

## Category B — Maximus diverges from spec; policy decision needed

These are structural differences where GaMax currently matches the TPC-H spec,
and Maximus does something else. Which is "right" depends on which is the
correctness oracle (pyarrow-on-spec vs. Maximus runs).

| Q | Aspect | GaMax (spec) | Maximus |
|---|---|---|---|
| **Q12** | Date bound column | `l_receiptdate` between `[1994-01-01, 1995-01-01)` (spec) | `l_commitdate` between same range |
| **Q17** | avg(l_quantity) subquery scope | over full lineitem per partkey (spec) | over the brand-filtered `part ⋈ lineitem` only |
| **Q20** | Threshold grouping keys | `(l_partkey, l_suppkey)` (spec); threshold join is multi-key | `(l_suppkey)` only; threshold join is single-key |
| **Q22** | `c_acctbal > 0` scope | applied only to avg subquery (spec) | applied to both subquery and the outer `in_set` |

**Status:** all PENDING — awaiting policy decision on oracle.

---

## Category C — filter-predicate semantic drift

String predicates whose form differs between the two. Usually visible as
wrong row counts.

| Q | GaMax | Maximus | TPC-H spec |
|---|---|---|---|
| Q13 | `NOT contains "special"` on `o_comment` | `NOT LIKE '%special%requests%'` | `NOT LIKE '%special%requests%'` |
| Q16 | `contains "Customer" AND contains "Complaints"` on `s_comment`, used in a `left_anti_join` from partsupp | `NOT LIKE '%Customer%Complaints%'` keeps suppliers, then `inner_join` partsupp | `LIKE '%Customer%Complaints%'` is the unwanted pattern; spec matches Maximus |
| Q20 | `contains "forest"` on `p_name` | `LIKE 'forest%'` (starts_with) | `LIKE 'forest%'` |

GaMax's `contains` without ordering means:
- Q13: includes orders whose comment has "special" anywhere, even without
  "requests" following — too strict (filters more than spec) because spec
  wants to *keep* orders whose comment does NOT match the ordered pair.
- Q16: unordered AND — includes suppliers with "Complaints ... Customer"
  which spec excludes.
- Q20: any part whose name contains "forest" (not just prefixed) — too loose.

**Status:** all TODO.

---

## Category D — join-order divergences (semantically equivalent)

Results identical under the TPC-H cardinality model. Listed for completeness
— not bugs, but differ in plan shape if you're tracing steps.

| Q | GaMax shape | Maximus shape |
|---|---|---|
| **Q11** (user-flagged) | `(supplier ⋈ nation_filt) ⋈ partsupp` | `(partsupp ⋈ supplier) ⋈ nation_filt` |
| Q2  | `region ⋈ nation ⋈ supplier ⋈ partsupp ⋈ part_filt`, min-subplan rescans all 4 base tables | `part_filt ⋈ partsupp ⋈ supplier`; nation/region joined separately; min-subplan shares the joined subgraph |
| Q5  | spine `region_filt ⋈ nation ⋈ supplier ⋈ customer ⋈ orders_filt ⋈ lineitem`, final multi-key on `(o_orderkey, s_suppkey) = (l_orderkey, l_suppkey)` | spine `nation ⋈ region_filt ⋈ supplier ⋈ lineitem`, final multi-key on `(l_orderkey, s_nationkey) = (o_orderkey, c_nationkey)` |
| Q7  | `supplier ⋈ n1_filt` separately from `customer ⋈ n2_filt`, then joined via lineitem+orders | materialized projections after each nation join, different intermediate shapes |
| Q9  | `part_filt ⋈ lineitem ⋈ supplier ⋈ partsupp (multi-key) ⋈ orders ⋈ nation` | `partsupp ⋈ part_filt ⋈ supplier ⋈ nation ⋈ lineitem ⋈ orders`, join to lineitem uses multi-key `(l_suppkey, l_partkey) = (ps_suppkey, ps_partkey)` |
| Q10 | joins first, aggregates last with 7-column key | aggregates early (group by o_custkey) before customer/nation join |

**Status:** all PENDING — equivalent results, update only if Maximus-parity is a goal.

---

## Category E — radically different plan shapes (rewrites)

These are not bugs; they are different algorithmic decompositions. Results
should still match. Listed for reference when comparing step traces.

| Q | Difference |
|---|---|
| Q18 | Maximus: `inner_join(orders, bigorders_subquery)` forwards `sum(l_quantity)` into the outer project — no second lineitem scan. GaMax: `left_semi_join` to prune, then re-scans lineitem and re-aggregates in the outer group_by. |
| Q21 | Maximus `q21()` uses count_distinct + fail-flag-sum with multiple group_bys. Maximus's driver actually calls `q21_optimized` at runtime (line ~2030). GaMax uses semi + anti joins with an inequality condition — closer to the SQL spec. |
| Q22 | Maximus fakes cross-join via `left_outer_join` on a synthetic `join_id=1`. GaMax uses `cross_join` directly. Combined with the Category-B issue, Q22 has multiple layered divergences. |

**Status:** leave alone unless results diverge in ways not explained above.

---

## Queries that structurally match

Q1, Q3, Q14, Q19 — modulo date-constant arithmetic (day-serial vs literal),
no material differences.

---

## Resolution log

When a discrepancy is fixed, move its entry to this section with a dated note
(e.g., "2026-05-03: Q5 date range corrected to `[8766, 9131)`; verified
against pyarrow SF1"). Keep the original row in its section with `Status: DONE`
for history.

_(empty)_
