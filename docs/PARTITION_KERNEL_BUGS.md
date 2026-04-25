# Partition-based operator bugs surfaced by the IR-bench axis probe

This is a tracking doc for kernel-level bugs in the
`gamax::groupby::partition::*` and `gamax::join::partition_hash_join`
implementations that the IR-bench operator-axis probe surfaced at SF=1.
Cosmetic input-shape guards (the "(A) tactical" fix in the bench
discussion) are already in place; this doc captures the underlying
kernel issues that need a real fix.

## Status: open. Real fixes deferred.

The tactical guards convert what used to be cryptic kernel crashes
(`column out of range`, `vector::_M_range_check`,
`cudaErrorIllegalAddress`) into clean `std::runtime_error`s thrown
before the kernel runs. The bench harness now reports them as
"unsupported input shape" failures and continues running the rest of
the sweep without context corruption.

The actual kernels still have the bugs. They will resurface as soon
as someone (a) widens the input acceptance check, (b) lowers the
input-size threshold, or (c) routes a different workload through these
paths.

---

## Bug 1 — `gamax::groupby::partition::groupby` ignores keys past the first

**File:** `src/cuda/operators/groupby/groupby_partition.cu`

**Symptoms before the (A) guard:**

* Multi-key groupby (e.g. Q3's `(l_orderkey, o_orderdate, o_shippriority)`,
  Q1's `(l_returnflag, l_linestatus)`): `column out of range` thrown by
  `gamax::table::release_column` in the post-processing loop in
  `groupby_noskf` (`groupby.cu:265-294`).

* Two-key groupby with a specific shape (Q20's `s_nationkey` + count
  agg): `vector::_M_range_check: __n (which is 2) >= this->size()
  (which is 2)` from a `std::vector::at()` call deeper in the
  partition path.

**Root cause:**

`partition::groupby` and `partition::groupby_gfur` type-dispatch on
`key.get_column(0).type()` only:

```cpp
return type_dispatcher(
    key.get_column(0).type(),
    partition_groupby_impl<...>{},
    key, val, agg_requests, hint, stream);
```

The downstream `partition_groupby_impl` produces a result with a
**single** "encoded key" column regardless of caller-provided key
cardinality. The caller `groupby_noskf` (line 265) then iterates as if
the result had `key.num_columns()` keys at the front:

```cpp
gamax::size_type internal_col_idx = num_key_cols + result_col_mapping[i];
auto col = internal_result->get_column(internal_col_idx);  // OOB
```

…which indexes past the end of `internal_result->columns()`.

**Tactical guard now in place** (`groupby.cu`, before the dispatch
switch): if `hint.groupby_algo ∈ {PARTITION, PARTITION_GFUR}` and the
input has more than one key column or any STRING key, throw
`"PARTITION groupby: only single-column keys supported"` /
`"PARTITION groupby: STRING keys not supported"`.

**Real fix sketch:**

Two paths:

1. **Run multikey_compress inside `partition::groupby`** the same way
   the SKF top-level does (`groupby.cu:317`). On entry, if
   `key.num_columns() > 1`, fuse keys into a single compressed column
   via `gamax::compress::multikey_compress`, run partition on that,
   then gather original key columns at the end (using a tids ANY
   aggregation, same pattern as the SKF wrapper). This makes the
   partition path multi-key-aware natively.

2. **Have `groupby_noskf` route multi-key/STRING to a different
   internal algo** (HASH) when the user requests PARTITION. Hides the
   limitation but defeats the purpose of explicit hint sweeping in the
   bench harness. **Don't pick this** — it lies to the harness.

Path 1 is the right answer.

---

## Bug 2 — `partition_hash_join` writes out of bounds on tiny inputs

**File:** `src/cuda/operators/join/partition_hash_join.cu`

**Symptoms before the (A) guard:**

Q7 + `eq_join = partition_hash_gfur` triggers
`cudaErrorIllegalAddress: an illegal memory access was encountered`
on a downstream `cudaStreamSynchronize`. Once that fires, the CUDA
context is unrecoverable — the IR-bench probe at SF=1 saw 1132 cells
cascade with `cudaErrorInvalidDevice` because they ran in the same
process after the bad cell.

**Root cause hypothesis:**

`partition_hash_join_params` (`partition_hash_join.cu:26-67`) hard-codes:

* `n_partitions = 1 << 16 = 65536`
* `max_build_bucket_size = 4096`
* `max_probe_bucket_size = 8192`
* `hash_table_size = 1 << 11 = 2048`

When the build or probe side has far fewer rows than `n_partitions`
(Q7 joins through `nation` (25 rows) and `region` (5 rows)), hash
distribution is essentially random with most buckets empty and a few
heavily skewed. Two plausible ways to get an OOB write:

1. **Build-bucket overflow.** A single partition receives more than
   `max_build_bucket_size = 4096` rows; the kernel writes past its
   shared-memory or per-partition buffer. Most likely on inputs whose
   row count happens to land within the `4096`-row range with skew.

2. **Output-buffer underestimate.** `get_ieq_join_output_alloc_size`
   returns a value that under-counts the actual join output for some
   small-input shapes; the join kernel writes past `max_output_size`.
   Less likely than (1) given the params already cap output via
   `MAX_JOIN_OUTPUT_SIZE = 1<<30`.

**Tactical guard now in place** (`partition_hash_join.cu`, in
`check_partition_hash_join_input` called from both
`ieq_partition_hash_join_gftr` and `ieq_partition_hash_join_gfur`): if
`min(left.num_rows(), right.num_rows()) < n_partitions = 65536`, throw
`"PARTITION hash join: input too small for n_partitions"` before any
kernel runs. Eliminates the cascade.

**Real fix sketch (B in the bench-discussion):**

This bug is a kernel-level out-of-bounds access. To find the actual
buggy line, run with NVIDIA's compute-sanitizer:

```bash
# Build the relevant binary in Debug or with -G for line info:
cmake -DCMAKE_CUDA_FLAGS_DEBUG="-G -lineinfo" -DCMAKE_BUILD_TYPE=Debug ..

# Drive the bad cell deterministically. run_tpch_ir doesn't (yet)
# expose --eq-join-algo, so either:
#  (a) add such a flag (one-line LowerOptions.eq_join setter), or
#  (b) write a tiny direct-call test that builds the failing
#      shape (e.g. nation × region) and calls
#      gamax::join::inner_eq_join with hint.join_algo=PARTITION,
#      hint.mat_algo=GFUR.
compute-sanitizer --launch-timeout=300 --tool=memcheck \
    ./run_tpch_ir 7 --scale 1   # with the partition_hash_gfur hint set

# memcheck will print the kernel name, function name, and the
# precise line of the OOB access along with a diff between the
# accessed address and the allocation it strayed from. From there,
# the fix is one of:
#   * Bounds-check the hot kernel write (`if (slot < cap) ...`);
#     spill an overflow flag and fall back to a slower path on hit.
#   * Size n_partitions / max_*_bucket_size dynamically based on the
#     input row count (e.g. n_partitions = clamp(rows / 8, 64,
#     65536); max_build_bucket_size = ceil(rows / n_partitions) * 4).
#   * Drop down to a single-bucket impl when min(rows) is small
#     enough that partitioning is pointless.
```

Once the kernel is fixed, **lower the threshold in
`check_partition_hash_join_input`** to whatever the new safe minimum
is (or remove the check entirely if the kernel becomes general).
