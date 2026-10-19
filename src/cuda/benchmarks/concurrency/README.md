<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Complete-operation concurrency benchmark

`benchmark_concurrency` measures two registered GPU operations with identical
event/gate timing in five whole-operation modes: A isolated, B isolated,
direct A→B, direct B→A, and concurrent. Supplying both chunk-size options adds
chunked A isolated, chunked B isolated, and chunked concurrent samples to the
same randomized rounds. The registry contains:

- `expr.two_col_add.nofuse`: `int64_t`, `col0 + col1`, one vectorized kernel
- `misc.gather.thrust`: `int64_t`, equal input/map sizes
- `containers.multiset.insert`: I64 keys, CG size 2, build duplication 1,
  load factor 0.5; times only `insert_async` into a table cleared before timing
- `containers.multiset.retrieve`: I64 keys, CG size 2, build duplication 1,
  load factor 0.5, hit ratio 1.0; times a heterogeneous `find_async` with
  `unique_keys = probe_keys / 4`

The default run is the authoritative 10×10 sweep (`2^20` through `2^29`), with
five warmups and 30–100 samples per mode/cell. Sampling continues until both
requested makespan-ratio bootstrap CIs have at most 2% relative half-width or
the 100-sample cap is reached.

The implementation is split into benchmark-internal modules: `operation.hpp`
defines the operation contract, `operations.cu` contains the four adapters,
`timing.cu` owns streams/gates/events, `reporting.cpp` owns statistics and
serialization, and `benchmark_concurrency.cu` orchestrates cells.

## Build and validate the gate

```bash
cmake --build build-rmm3 --target benchmark_concurrency -j8
build-rmm3/src/cuda/benchmarks/benchmark_concurrency --self-test-gate
```

The self-test prequeues 20 ms and 30 ms synthetic kernels behind the shared
mapped-host release gate and checks duration, makespan, and start skew. The
existing `ExprFastTest.NoFuseIsAsyncAndIntermediateDestructionIsStreamOrdered`
is the nonblocking/lifetime regression for the no-fuse adapter prerequisite.
The gate test also deliberately synchronizes a blocked stream and verifies that
the timeout path rejects it.

The prepared expression adapter launches one vectorized addition kernel and
returns its owned output column. It has no intermediate allocation, so the
timed `enqueue(stream)` path remains nonblocking without deferred cleanup.

## Smoke and authoritative runs

```bash
# Fast 2x2 functional smoke (explicitly non-authoritative).
build-rmm3/src/cuda/benchmarks/benchmark_concurrency \
  --a-exp-min 20 --a-exp-max 21 --b-exp-min 20 --b-exp-max 21 \
  --warmups 1 --min-samples 3 --max-samples 3 \
  --bootstrap-iterations 200 --output-prefix exp_results/concurrency_smoke/run

# Full authoritative sweep. Run on an otherwise idle GPU.
build-rmm3/src/cuda/benchmarks/benchmark_concurrency \
  --op-a expr.two_col_add.nofuse --op-b misc.gather.thrust \
  --output-prefix exp_results/concurrency_gh200_full/run

# Full authoritative 9x9 multiset sweep (2^20 through 2^28).
build-rmm3/src/cuda/benchmarks/benchmark_concurrency \
  --op-a containers.multiset.insert --op-b containers.multiset.retrieve \
  --a-exp-min 20 --a-exp-max 28 --b-exp-min 20 --b-exp-max 28 \
  --output-prefix exp_results/multiset_concurrency_gh200_full/run
```

## Chunked execution

Chunking is opt-in and requires both `--a-chunk-size` and `--b-chunk-size`.
Chunk sizes are item counts, need not be powers of two, and are clamped to the
operation's problem size. The vectorized expression operation additionally
requires every `int64` chunk boundary to preserve 16-byte alignment, so its
chunk size must be a multiple of two items. The host rejects an expression
launch whose input or output address is not suitably aligned. Each
output-producing operation performs exactly one full-size allocation at its
existing timing boundary; chunks write disjoint output slices. A chunked
concurrent sample naturally submits `A0, B0, A1, B1, ...` (or the reversed
order) without barriers between chunk rounds. The first operation alternates
between measured concurrent samples.

```bash
build-rmm3/src/cuda/benchmarks/benchmark_concurrency \
  --op-a expr.two_col_add.nofuse --op-b expr.two_col_add.nofuse \
  --a-exp-min 29 --a-exp-max 29 --b-exp-min 29 --b-exp-max 29 \
  --a-chunk-size 1048576 --b-chunk-size 1048576 \
  --output-prefix exp_results/expr_chunking/run
```

The primary added summary metrics are:

- `chunked_isolated_slowdown_{a,b}`: chunked isolated time divided by whole
  isolated time; values above one are chunking overhead.
- `chunked_concurrent_speedup`: whole/whole concurrent makespan divided by
  chunked/chunked concurrent makespan; values above one favor chunking.
- `chunked_interference_slowdown_{a,b}`: chunked concurrent operation time
  divided by its chunked isolated time.

Every run writes `<prefix>_raw.csv`, `<prefix>_summary.csv`, and
`<prefix>_metadata.json`. Warmups are retained in raw CSV. Each successful cell
performs full-device semantic validation of one isolated output from each
operation and of both outputs from a concurrent launch before measurement.
The L2-sized RMM buffer is purged outside timing before every warmup and
measured sample; validation is outside timing and is followed by the next
required purge.

The multiset retrieve operation intentionally specializes the fixture's
`build_dup=1`, full-hit case to one aligned build-TID result per probe. It uses
the same cuCollections `find_async` kernel with hash/equality rebound to the
probe-key column. This avoids retrieve's atomic compaction counter and mandatory
host readback; it is not a measurement of the multi-match `retrieve()` API.

## Nsight Systems ranges

Each timed sample has one host NVTX range whose name is its sampling mode:
`a_isolated`, `b_isolated`, `sequential_a_then_b`,
`sequential_b_then_a`, or `concurrent`. The five modes also use distinct colors.
Chunked samples use `chunked/a_isolated`, `chunked/b_isolated`, and
`chunked/concurrent` with lighter corresponding colors.
The range is tight around the timed operation(s), excluding cell preparation,
validation, reset, L2 purge, and CSV output.

Include NVTX when collecting a report:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output concurrency_profile \
  build-rmm3/src/cuda/benchmarks/benchmark_concurrency \
  --a-exp-min 20 --a-exp-max 20 --b-exp-min 20 --b-exp-max 20 \
  --warmups 1 --min-samples 3 --max-samples 3 \
  --bootstrap-iterations 200 --output-prefix /tmp/concurrency_profile
```

Generate the four PNG and PDF heatmaps without rerunning CUDA work:

```bash
# Rebuild every derived statistic and bootstrap CI from raw samples.
python3 src/cuda/benchmarks/concurrency/summarize_concurrency.py \
  exp_results/concurrency_gh200_full/run_raw.csv \
  exp_results/concurrency_gh200_full/run_summary_regenerated.csv

# Plot either the executable-produced or regenerated summary.
python3 src/cuda/benchmarks/concurrency/plot_concurrency.py \
  exp_results/concurrency_gh200_full/run_summary.csv \
  --output-dir exp_results/concurrency_gh200_full/figures
```

The plotter requires NumPy and matplotlib. On the current development host,
`/usr/bin/python3` provides both packages.

## NVBench timing-boundary comparison

For the `2^20` smoke cell on the GH200, isolated medians from this harness were
about 19 µs (expression) and 18 µs (gather). Filtered existing NVBench runs
reported about 26 µs and 24 µs respectively. The fixtures explicitly call
`cudaStreamSynchronize()` inside their timed callback, so NVBench's stop event
is submitted after a host synchronization gap; the standalone harness submits
its stop event directly behind the complete operation. The direction and
magnitude of the difference are therefore consistent with the intended timing
boundary rather than missing GPU work.

Raw mode order is deterministic from `--seed`; the data fixtures retain seeds
42 and 43. Size ranges, warmups, sample bounds, bootstrap count, CI threshold,
device, and gate timeout are CLI-configurable (`--help` lists all options).
