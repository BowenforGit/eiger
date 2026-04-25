# Partition-based operator bugs surfaced by the IR-bench axis probe

Tracking doc for kernel-level bugs in the
`gamax::groupby::partition::*` and `gamax::join::partition_hash_join`
implementations that the IR-bench operator-axis probe surfaced at SF=1
and again (with new shapes for Bug 2) at SF=30.

## Status

* **Bug 1 (PARTITION groupby ignores keys past the first):** open. A
  tactical input-shape guard converts the kernel crash into a clean
  `runtime_error` so the bench harness reports an actionable
  "PARTITION groupby: only single-column keys supported" /
  "STRING keys not supported". The actual kernel is still
  single-key-only.

* **Bug 2 (partition_hash_join build-bucket overflow):** **fixed**.
  The kernel still has the OOB-on-overflow shared-memory write, but
  the host now precondition-checks the maximum build-side bucket
  size after `partition_transform` (via an `atomic_max` written by
  `generate_work_units`) and throws a clear error before launch when
  any partition exceeds `max_build_bucket_size`. No more crashes; no
  more CUDA-context-corruption cascade. SF=30 sweeps surface this
  cleanly on Q9, Q10, and Q21 (largest partitions of 5540 and 12202
  rows vs the cap of 4096) — same root cause; only shows up at
  higher scales because key skew per partition is dataset-dependent.

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

Run `multikey_compress` inside `partition::groupby` the same way
the SKF top-level does (`groupby.cu:317`). On entry, if
`key.num_columns() > 1`, fuse keys into a single compressed column,
run partition on that, then gather original key columns at the end
(using a tids ANY aggregation, same pattern as the SKF wrapper).

---

## Bug 2 — `partition_hash_join` shared-memory overflow on skewed buckets — **fixed (host-side check)**

**File:** `src/cuda/operators/join/partition_hash_join.cu`

**Symptoms before the fix:**

Q7 + `eq_join = partition_hash_gfur` triggered
`cudaErrorIllegalAddress: an illegal memory access was encountered`
on a downstream `cudaStreamSynchronize`. Once that fired, the CUDA
context was unrecoverable — the IR-bench probe at SF=1 saw 1132
cells cascade with `cudaErrorInvalidDevice` because they ran in the
same process after the bad cell.

The bug was **non-deterministic**: three back-to-back invocations of
the same Q7 cell under `--use-cuda-mm` produced (a) garbage values,
(b) different garbage values, (c) `cudaErrorIllegalAddress`. Classic
out-of-bounds shared-memory write hitting different downstream data
on each run.

**Root cause:**

`join_copartitions_arr`'s shared-memory layout is sized for
`max_build_bucket_size + 512 = 4608` build rows per partition:

```cpp
extern __shared__ int16_t temp[];
KeyT* elem    = (KeyT*)temp;
ValT* payload = (ValT*)&elem   [max_build_bucket_size + 512];
int16_t* next = (int16_t*)&payload[max_build_bucket_size + 512];
int32_t* head = (int32_t*)&next   [max_build_bucket_size + 512];
// ... shuffle space follows ...
```

The build loop writes to `elem[base_r + k*blockDim.x + tid]` for
`base_r = 0, VT*blockDim.x, 2*VT*blockDim.x, ...` up to `len_R`. The
write is gated only by `k < l_cnt_R` (per-thread tail bound), not by
`base_r + k*blockDim.x + tid < max_build_bucket_size + 512`.

When `len_R > max_build_bucket_size`, writes at index
`>= max_build_bucket_size + 512` corrupt `payload`, `next`, `head`
and the `shuffle_space` region. Probe then reads a corrupted
`head[hval]` and follows a corrupted `next` chain — produces wrong
matches at best, dereferences a poisoned register at worst.

`partition_hash_join_params` hard-codes `max_build_bucket_size =
4096` based on n_partitions=65536. With low-cardinality keys (e.g.
`s_nationkey ∈ [0, 24]` in Q7's join chain), 65536 partitions are
useless — most are empty and a few absorb thousands of rows. The
bench probe saw build-side bucket sizes of 4227, 4861, 6161, and
14011 (3.4× the cap) for various Q7 / Q9 / Q10 / Q21 join shapes.

**Fix in place** (host-side precondition check):

`generate_work_units` now also tracks the maximum build-side bucket
size via `atomic_max` into a 1-element scalar:

```cpp
gamax::atomic::atomic_max(
    max_r_len_out, static_cast<gamax::size_type>(r_len));
```

`create_work_units` reads that scalar back to host after the kernel
syncs and throws if it exceeds `params.max_build_bucket_size`:

```cpp
if (host_max_r_len > params.max_build_bucket_size) {
    throw std::runtime_error(
        "PARTITION hash join: build-side bucket overflow "
        "(largest partition has " + std::to_string(host_max_r_len) +
        " rows, exceeds max_build_bucket_size=" +
        std::to_string(params.max_build_bucket_size) +
        "); use HASH or SORT for skewed key distributions");
}
```

This **replaces** the prior conservative `min(rows) < n_partitions`
input-size guard, which over-rejected (96 cells/query × 4 queries)
moderately-sized inputs whose buckets were actually well-distributed.
Probe diff at SF=1: 384 cells now report the precise per-shape
overflow message; cells with skew-free distributions now run
correctly through partition_hash_join.

**Why this isn't yet the full fix:**

The kernel still has the OOB-write hazard. The host check prevents
it from being hit, but anyone who:

* widens `max_build_bucket_size`,
* lowers `n_partitions`,
* adds a workload with even more skewed keys than current,

…will re-discover the same OOB write. Real fix in the kernel:

1. **Bounds-check the shared-memory writes**:
   `if (base_r + k*blockDim.x + tid < max_build_bucket_size + 512) elem[...] = val;`
   — but this drops rows past the cap, so the join misses matches.
   Bad on its own; must be combined with (2) or (3) below.

2. **Spill overflowing buckets to a global-memory build path**: when
   a partition exceeds `max_build_bucket_size`, fall back to
   building the hash table in global memory for that partition only.
   Slower for those buckets but correct.

3. **Dynamic n_partitions / bucket-cap sizing**: pre-scan the input
   to estimate distribution, choose `n_partitions` and
   `max_build_bucket_size` so no bucket overflows. The current
   hard-coded 65536 / 4096 pair is a one-size-fits-none choice.

(2) is the right answer long-term and the same code path could
serve as the "tiny-input" fallback too (when no partition has
enough rows to make the partition sweep worthwhile).

## Debugging notes for the next person

The investigation that found Bug 2 used:

* `use_cuda_mm()` runtime switch to swap the global manager from
  `UserSpaceMM` (88 GiB pre-allocated slab — hides any OOB inside
  the slab boundary from `compute-sanitizer`) to `CudaMM`
  (`cudaMallocAsync`/`cudaFreeAsync` per allocation — each
  allocation is a separately-tracked range). Replaces the prior
  compile-time `USE_DEFAULT_MEMALLOC` macro; available via
  `--use-cuda-mm` on `run_tpch_ir`.

* Driving the failing cell deterministically via
  `run_tpch_ir 7 --scale 1 --use-cuda-mm --eq-join-algo partition_hash_gfur`
  — the bench harness wires `--eq-join-algo` straight to
  `LowerOptions::eq_join`.

* `compute-sanitizer` was **unable to attach** because libcudart's
  static initializers init CUDA before main runs (sanitizer reports
  "CUDA initialized before the Sanitizer. The Sanitizer will be
  disabled"). Worked around by adding `printf` instrumentation to
  `generate_work_units` (printing `r_len` per partition) and the
  start of `join_copartitions_arr` (printing the work unit shapes).
  The first oversize bucket showed up immediately. ~30 minutes from
  printf to root cause.

If you ever land a fix that lets compute-sanitizer attach to a
libcudart-linked binary on this system, please remove this section
and document the proper sanitizer flow.
