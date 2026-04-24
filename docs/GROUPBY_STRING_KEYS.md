# Group-by with string keys

## Problem

GaMax's hash-based `groupby` path could not handle STRING key columns.
Any TPC-H query that grouped by a string (Q1, Q4, Q5, Q10, Q16, Q21, Q22)
either threw `Unsupported groupby algorithm` or `Invalid type_id: 12`
before producing a single row. This was the single largest gap in the
TPC-H coverage tracker.

A second, independent gap blocked some numeric queries too: multi-column
numeric keys also came out of `auto_select_groupby_algo` still tagged
`AUTO_ALGO` (the existing heuristic only handled single-column keys)
and hit the same `Unsupported groupby algorithm` default case. Q18,
Q20, and the multi-col path in Q1 all tripped on that.

## Approach

We kept the existing multi-col hash groupby (`operators/groupby/groupby_hash.cu::hash_groupby_multi_key`) as
the functional target and taught every layer it calls through to
handle STRING as a first-class key type. Sort / partition paths remain
numeric-only; any STRING-containing key is routed to HASH.

The design has two parts: (a) make the device-side STRING comparison
path actually work end-to-end, and (b) replace in-kernel key
materialisation with a post-kernel gather so the output's STRING key
columns are produced via the existing STRING-safe `gather_strings`
helper instead of being written inline (which would require precomputed
offsets the kernel can't produce).

### (a) Make STRING hashable / comparable on device

`row_hasher` and `row_equal` in `memory/row_utils.cuh` already had
STRING specialisations in `element_hash_functor` / `element_equal_functor`,
dispatched via `type_dispatcher_all`. The specialisations constructed
a `gamax::string_column_impl(string_column_view)` on device. That
constructor was broken for STRING: it called `col.data()` (untyped),
which in `column_view::data()` computes `_data + _offset * size_of(_type)` —
and `size_of(STRING)` routes through the numeric-only `type_dispatcher`,
which has no STRING case and traps on device (and throws on host).

The fix was to switch `string_column_impl`'s `string_column_view` and
`mutable_string_column_view` constructors to use `col.template data<char>()`.
The templated `data<T>()` returns `_data + _offset` directly in units
of T — no `size_of` call, no trap, correct for STRING columns where
`_offset` is always 0.

### (b) Leader-tid gather for key materialisation

`compute_groupid_kernel` used to call `gamax::copy_row(keys_in, keys_out, idx, gid)`
inline on each new-key insertion. That scatters per-column writes into
the output key table — fine for numeric columns, but unusable for
STRING output columns because STRING writes require precomputed
offsets (which can't be produced inside an in-flight insert-and-find
kernel).

Replace: the kernel now writes `leader_tid[gid] = idx` per new group —
a single `size_type` write recording the row index that became the
group's first inserter. `hash_groupby_multi_key` trims `leader_tid` to
`num_groups` after the kernel runs and uses it as a gather map. Each
input key column is materialised via `join::detail::simple_materialize`,
which dispatches STRING to `gather_strings` and numeric to
`util::device::gather`.

The same pattern already existed in the codebase: the SKF compression
path in `operators/groupby/groupby.cu::groupby()` smuggles a tid
through an `ANY(tids)` aggregation request and gathers original keys
afterwards. Our change does the same shape but without SKF's
ANY-aggregation overhead, and inside `hash_groupby_multi_key` so the
public contract of that function is unchanged.

### Auto-algo routing

`auto_select_groupby_algo` in `operators/groupby/groupby.cu` gained two
guards before the existing single-column cardinality heuristic:

1. Any key column is STRING → `HASH`. Runs first so the STRING short-
   circuit precedes `compute_ndv_approx` (which goes through
   STRING-unaware `type_dispatcher`).
2. More than one key column → `HASH`. Multi-col numeric keys fall
   through the SKF compression path first if the caller sets
   `enable_skf`; anything that reaches `auto_select_groupby_algo`
   multi-col means compression either failed or wasn't attempted,
   and HASH is the safe functional default.

Single-column numeric keys still get the original cardinality-based
`HASH` vs `PARTITION` pick, unchanged.

## Surrounding fixes that fell out

The STRING groupby path traverses code outside the groupby operator,
and those places had their own STRING gaps:

- **`util::device::compact(table)`** was numeric-only; semi-joins whose
  left side had a STRING column (e.g. Q4's `o_orderpriority` carried
  through `left_semi_join`) died on the compact step. Extended to
  materialise surviving tids once and route STRING columns through
  `gather_strings` while leaving the fast pure-numeric path unchanged.

- **`sort_string_column_table_impl`** (in `operators/sort/sort_table.cu`)
  had a pre-existing bug that wasn't reached until the groupby started
  producing STRING outputs. The function called
  `apply_sorted_indices_to_columns` with the RidType-typed `sorted_B`,
  and the inner loop then passed that map into `gather_strings` which
  reads it as `size_type` — over-reads the buffer end, causing
  `cudaErrorIllegalAddress`. Fixed by skipping STRING columns in the
  apply loop (they're re-gathered with the correct `size_type` map
  immediately after).

- **`simple_materialize`** (in `operators/join/materialize.cu`) was
  numeric-only too. Extended to branch on STRING → `gather_strings`.
  Side benefit: the recently-landed `cross_join` was latently broken
  for STRING columns through the same path, and is now correct.

- **`multikey_compress`**'s fusion-compatibility probe used the
  numeric-only `type_dispatcher` to check every column — so a STRING
  column threw "Invalid type_id" instead of letting the functor's
  existing STRING specialisation return "not compatible." Switched to
  `type_dispatcher_all`.

- **`groupby_noskf`**'s key copy used `column::deep_copy` which hits
  `size_of(STRING)` and trips the same trap as `col.data()`. Changed
  to `release_column` — we're discarding `internal_result` at return
  anyway, so taking ownership of the columns is both STRING-safe and
  cheaper than the D2D copy.

- **Dead code removed**: `gamax::copy_row` and `copy_element_functor`
  in `memory/row_utils.cuh`. The insert-kernel's call was the only
  user, and it's gone now.

## Testing

A new test file `tests/groupby/groupby_string_test.cu` covers ten
scenarios:

- Single STRING key, SUM aggregation
- Mixed `{STRING, INT64}` keys
- Per-group value identity (not just total-sum)
- Prefix-overlapping keys (`"abc"`, `"abcd"`, `"abcde"`) must stay distinct
- Long keys beyond any inlined prefix (60+ chars)
- Two STRING keys together (stresses multi-column row hashing)
- Empty strings as a valid key
- SUM + MIN + MAX over the same values
- 1000 rows / ~100 groups with CPU cross-check
- Extreme single-key skew (dominant + rare)

Full regression sweep: 271 / 271 tests pass across `test_exec`,
`test_rules`, `test_emit`, `test_tpch_emit`, `test_groupby_string`,
`test_aggregate`, `test_groupby_hash`, `test_groupby_sort`,
`test_groupby_multikeys`.

TPC-H end-to-end: **Q1** (2 STRING keys) and **Q4** (single STRING
key) now produce canonical SF1 answers. **Q18** and **Q20**
(previously blocked on the multi-col-numeric auto-algo gap) now run
end-to-end as a side effect of the same routing change. Q6 / Q11 /
Q14 / Q15 / Q2 are unchanged. Q10 remains blocked on a pre-existing,
unrelated issue (`cudaErrorInvalidDevice` in an `inclusive_scan`
during an empty-intermediate path).

## What's deliberately not done

- **SKF (smart-key-fusion)** is kept OFF at the phys boundary. Turning
  it on appeared to introduce memory regressions (SKF adds a tids
  value column and an ANY-aggregation request that blow up the
  memory pool on some queries). The `groupby_noskf` path with the
  new routing is sufficient to unblock every STRING-key TPC-H query
  we care about; re-enabling SKF is a separate perf investigation
  once multi-numeric-key workloads actually demand it.
- **Sort-based and partition-based STRING groupby** are unimplemented.
  The routing guarantees any STRING key lands in HASH, and TPC-H
  group cardinalities don't motivate partitioning today. Factoring
  a shared key-materialiser interface across all three backends is
  explicit future work.
- **`compact` / `simple_materialize` perf optimisations** — the
  STRING paths materialise gather maps eagerly; for pure-numeric
  workloads we preserve the faster two-phase `compact` that skips
  the separate tid materialisation.
