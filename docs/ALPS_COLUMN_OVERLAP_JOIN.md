<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# ALPS column-overlapped distributed equality join

**Status:** implementation, tuning study, profiling, and required validation
completed on 2026-07-22.

## Objective

This design provides a column-granular communication/computation-overlapped implementation of
the distributed shuffle equality join. This is an alternative to, not a
replacement for, the existing full-table shuffle followed by the monolithic
local join.

Row chunking is intentionally out of scope. It complicates coordination and
can reduce NVLink efficiency by turning large transfers into small messages.
The alternative keeps each column transfer whole.

## Current contract

- The existing implementation remains the default and baseline.
- Select the alternative per execution through `DistributedExecOptions` with
  `DistributedEqJoinImplementation::{Legacy, ColumnOverlap}`.
- The implementation covers shuffle inner equality joins using the local HASH
  join, a single fixed-width key, and fixed-width or Arrow-layout `STRING`
  payloads. Column exchange may use NCCL or the opt-in CE-pull full-staging
  transport. Unsupported configurations fail explicitly rather than silently
  using the legacy path.
- The operator independently supports `partition_left` and `partition_right`,
  including either or both inputs already being correctly distributed. The
  representative SF300 shapes shuffle both inputs. Broadcast joins continue
  through their existing implementation regardless of this selector.
- Output correctness requires identical schema, row count, and row multiset,
  including duplicates. Output row order is unspecified.
- Add a reusable local `eq_join_indices()`-style primitive for HASH join. It
  returns left and right match-index columns without materializing payloads.
  The existing local `eq_join()` remains unchanged and can reuse the primitive
  internally where appropriate.
- Use one key-derived destination permutation and communication pattern per
  input side. Apply that same permutation independently to every column; do
  not recompute a partition for each payload column.
- Use three logical internal streams: one NCCL communication stream, one
  payload preparation stream, and one key-match/output-materialization stream.
  The existing executor stream is the entry/exit dependency boundary. CUDA
  events express readiness and ownership dependencies.
- Own the extra streams and reusable events in a lazily initialized
  column-join runtime attached to `alps::Context`. It is created only when the
  alternative is selected and destroyed with the context.
- Eagerly enqueue whole-column work and retain intermediate buffers until final
  completion. Peak allocator usage is part of acceptance reporting. The
  measured early-reclamation experiment regressed latency, so a rolling column
  window remains out of scope for this milestone.
- Compute both complete hash-partition plans before starting the overlapped
  pipeline. This retains the existing device-to-host count readback and MPI
  count exchange as an up-front synchronization point. Asynchronous partition
  planning is a possible later optimization, not prototype scope.
- Use the existing NCCL communicator on the dedicated communication stream.
  Give that stream elevated priority in the selected `ColumnOverlap` mode;
  measurements showed a decisive improvement over normal priority. Keep this
  configurable for controlled experiments. The legacy path is unaffected.
- Enqueue the complete column preparation and exchange DAG before invoking the
  initially host-synchronizing `eq_join_indices()` primitive. Already-enqueued
  payload communication can then progress while the host waits for the match
  count and index columns.
- Issue key columns in left/build then right/probe order. Use the `Interleave`
  payload schedule by default, alternating right/probe and left/build columns
  while preserving schema order within each side. Assemble completed columns
  into canonical output-schema positions rather than communication order.
- All ranks issue the same ordered column collectives, including zero-count
  peer transfers. If either received key input is empty, matching returns
  correctly typed empty index columns and the final result has the normal
  empty-join schema.
- Validate the configuration and allocate all statically knowable permutation,
  send, and receive storage before the first NCCL operation. Establish
  rank-wide validation success before communication so one rank does not fail
  while its peers enter collectives.
- Use the same arena/UserSpaceMM configuration for legacy and overlap benchmark
  cells. CudaMM allocation stalls must not be allowed to bias the comparison.

## Accepted configuration and why

The accepted NCCL configuration uses a high-priority communication stream, the
`Interleave` payload schedule, and retains intermediates through final
completion. Normal priority and probe/build-first schedules remain experiment
controls. Early event-safe reclamation was rejected because it materially
increased latency, despite reducing lifetime. The opt-in path accepts the
measured workspace increase instead of adding a rolling column window.

`Legacy` remains the overall executor default. There is no automatic
shape-dependent selector: callers choose ColumnOverlap when the input shape
offers enough communication/match overlap and sufficient memory. Unsupported
explicit requests fail clearly rather than silently changing algorithm.

Promotion required visible NCCL/HASH overlap, at least 10% complete-join
improvement on two of the three representative Q5/Q8/Q9 shapes, no correctness
failure, no more than 3% regression on low-overlap controls, and recorded peak
memory without OOM. Multi-stream phase accounting was part of the same gate;
overlapping child intervals are unioned rather than added.

## Intended Dataflow

1. Derive each participating side's destination vector, row permutation, peer
   counts, and offsets from its key.
2. Permute and exchange key columns first.
3. While key exchange runs, apply the saved permutation to payload columns.
4. When both received keys are ready, run the narrow HASH match and produce
   `(left_index, right_index)` columns.
5. Continue exchanging whole payload columns in one deterministic NCCL order
   on every rank.
6. As each received payload becomes ready and match indices exist, materialize
   its output column independently.
7. Assemble columns in the legacy output schema and make the executor stream
   wait for the alternative join's final completion event.

The scheduler must not derive NCCL issue order from rank-local sizes because
skew could make ranks choose different orders and deadlock.

## Internal decomposition

Factor the existing hash-partition machinery into a reusable plan without
changing legacy behavior. The intended conceptual API is:

```cpp
struct HashPartitionPlan {
    std::unique_ptr<gamax::column> permutation;
    CommPattern pattern; // completed send and receive counts/offsets
};

HashPartitionPlan make_hash_partition_plan(...);
gamax::column_ptr apply_partition(
    gamax::column_view input,
    const HashPartitionPlan& plan,
    cudaStream_t stream);
```

The legacy full-table `hash_partition()` may use these primitives internally,
but the legacy distributed-join algorithm, public behavior, and default
selection remain unchanged.

Factor the HASH join at the point where it already obtains matching TID pairs:

```cpp
struct EqJoinMatches {
    std::unique_ptr<gamax::column> left_indices;
    std::unique_ptr<gamax::column> right_indices;
};

EqJoinMatches eq_join_indices(...);
```

The prototype retains the current host synchronization needed to obtain and
apply the actual match count. A device-resident count contract and fully
asynchronous count-aware materialization are deferred.

## Scope and failure behavior

The first promotion deliberately covered fixed-width payloads over NCCL. The
same scheduling contract was later extended to Arrow STRING payloads and the
CE-pull column transport. Keys remain single and fixed-width, the local join
must be HASH, and the selector applies only to shuffle equality joins.
Broadcast joins retain their existing implementation.

`Legacy` remains the default and there is no automatic cost model. An explicit
unsupported request is rejected rather than silently falling back, because a
fallback would make benchmark labels and physical-plan intent untrustworthy.
An already-distributed left or right input is treated as ready instead of being
reshuffled.

## Correctness contract

- At small and medium sizes, compare legacy and overlap results as exact
  multisets after canonical sorting.
- Cover duplicates, misses, empty local inputs, zero-count peers, skewed
  partitions, and both `int32` and `int64` fixed-width keys and payloads.
- At the largest synthetic sizes, validate cardinality plus multiple
  order-independent column and row hashes. These hashes supplement rather than
  replace exact tests.
- After production integration, SF100 and SF300 distributed TPC-H executions
  must pass the existing golden Parquet verification.

## Evaluation methodology

The dedicated benchmark should include synthetic schemas and PK/FK ratios
modeled on the target Q5, Q8, and Q9 joins, scaled size sweeps, the largest
practical SF300-equivalent cases, and payload-count/width sweeps. Include
narrow and low-overlap shapes to quantify regressions.

The primary metric is complete critical-path distributed-join latency. Also
record:

- partition-plan time;
- key preparation and exchange intervals;
- payload preparation and exchange intervals;
- HASH match interval;
- output materialization intervals;
- the union of active intervals and visible overlap; and
- peak allocator usage.

The implementation uses dedicated cross-stream CUDA events, named streams, and
NVTX ranges for these measurements. Production phase rows share a common step
origin, and residual attribution uses the union of overlapping intervals.

## Promotion gate

Use an isolated distributed-join benchmark modeled on the large SF300 shuffle
joins in Q5 step 6, Q8 step 13, and Q9 step 4. Promote the design into the
production executor only after it demonstrates:

- visible NCCL/HASH overlap in an NSYS timeline;
- at least 10% lower complete distributed-join latency on two of the three
  representative shapes;
- no correctness failures;
- no more than 3% regression on shapes with little overlap opportunity; and
- recorded peak memory, with OOM treated as a failure.

No automatic selector is added at promotion time. After collecting full-query
results, a future `Auto` mode may use a cost model to choose overlap only when
the predicted communication and HASH windows justify it.

### Acceptance result

The promoted high-priority `Interleave` configuration passed every gate:

| SF300-shaped join | Legacy median ms | Column-overlap median ms | Improvement |
|---|---:|---:|---:|
| Q5 | 205.440 | 145.102 | 29.4% |
| Q8 | 206.254 | 146.170 | 29.1% |
| Q9 | 198.857 | 121.987 | 38.7% |

The full-width, key-only low-overlap control improved from 138.266 ms to
129.610 ms (6.3%). A smaller divisor-1024 control improved from 0.566 ms to
0.538 ms (5.0%). Thus the selected alternative did not regress either
low-overlap control. Peak allocator usage for the SF300-shaped overlap cases
was 66.0 GB for Q5, 91.0 GB for Q8, and 75.7 GB for Q9 per rank; none OOMed.

The tuned SF300-shaped Q5 NSYS trace measured 51.336 ms mean overlap between
NCCL kernels and match/materialization kernels across four ranks. This was
74.5% of NCCL busy time. The isolated stage timing and NSYS timing differ
because instrumentation changes synchronization and capture overhead; clean
wall-clock results above are the performance gate.

Clean production query measurements also improved at both validated scales:

| Query | SF100 legacy ms | SF100 overlap ms | Improvement | SF300 legacy ms | SF300 overlap ms | Improvement |
|---|---:|---:|---:|---:|---:|---:|
| Q5 | 69.928 | 53.916 | 22.9% | 203.206 | 155.423 | 23.5% |
| Q8 | 77.478 | 56.258 | 27.4% | 224.584 | 162.325 | 27.7% |
| Q9 | 116.800 | 81.735 | 30.0% | 343.326 | 234.968 | 31.6% |

All six production timing runs passed golden verification. The explicitly
selected alternative also passed targeted Q5/Q8/Q9 golden verification at
SF100 and SF300. The unchanged default legacy path passed Q1-Q22 at both
scales after the partition and profiling refactors.

## Multi-stream profiling contract

Production promotion included the required profiler correction. Dedicated
stage events, named streams, and NVTX markers remain available, while phase CSV
attribution now represents overlapping child intervals without treating them
as additive.

The completed implementation:

- extends `StepProfileSink` to represent more than one simultaneously active
  phase through explicit phase handles rather than a single implicit active
  phase;
- captures phase intervals from their actual CUDA streams against a common
  per-step device-time origin;
- derives attributed step time from the union of child intervals, not the sum
  of child durations. Overlapped compute and communication remain separately
  reported, while residual/unattributed time is `step interval - union(child
  intervals)`;
- synchronizes/finalizes every stream that recorded phase events before
  resolving timestamps;
- preserves nested NVTX labeling so NSYS directly shows key exchange, payload
  preparation/exchange, key matching, and per-column materialization overlap;
- adds profiler tests for disjoint, nested, partially overlapping, and fully
  overlapping intervals, including tolerance behavior at interval boundaries;
- updates the phase CSV documentation and plotting consumer so overlapping
  category durations are not added and interpreted as serial elapsed time.

Fresh Q5 production phase data validates the accounting. On the larger shuffle
step, direct phase sums were 1.438-1.471 ms per rank, interval unions were
0.961-0.985 ms, and every derived residual exactly matched outer step time
minus the union without a negative clamp.

## Delivery record

1. Factored the partition-plan and HASH-index primitives and protected legacy
   behavior with regression tests and baseline timings.
2. Added the isolated column-overlap operator and dedicated benchmark.
3. Ran correctness, latency, memory, and NSYS studies and applied the prototype
   acceptance gate.
4. Added production dispatch after the gate passed and supported inputs whose
   distribution already matches the join key.
5. Made production phase profiling multi-stream correct as specified above.
6. Ran SF100/SF300 golden verification, clean end-to-end measurements, and an
   NSYS study confirming overlap.

The first tuning study resolved the evidence-sensitive choices for this
milestone: high communication priority, `Interleave` payload issue order, and
early reclamation disabled. Future experiments may revisit them behind the
existing controls, but the correctness contract and explicit
alternative/default behavior remain fixed.

## Artifacts

Results are under
`/storage/eiger/exp_results/alps_column_overlap_20260722`:

- `final_legacy.csv` and `final_overlap_high_interleave.csv`: clean isolated
  SF300-shaped latency samples;
- `tune_*.csv`: communication-priority, scheduling-order, and reclamation
  experiments;
- `final_overlap_memory.csv`: peak allocator measurements;
- `low_overlap_narrow_full.csv` and `low_overlap_narrow_div1024.csv`:
  low-overlap controls;
- `final_production_e2e.csv`: clean SF100/SF300 production timing samples;
- `q5_sf1_final_{steps,phases}.csv`: final common-origin interval and
  interval-union profiler validation;
- `q5_sf300shape_interleave_high.nsys-rep` and its exported SQLite database:
  final tuned overlap trace;
- `targeted_sf100_sf300_golden.log`: Q5/Q8/Q9 alternative-path goldens; and
- `full_golden_legacy_sf{100,300}`: 22-query default-path golden suites.

## Arrow STRING Payload Extension

The follow-up implementation completes TPC-H shuffle-join coverage for
fixed-width keys with variable-length payloads. It does not add string-key
joins and does not change `Legacy` as the default.

Each partitioned Arrow string is handled as two whole-column communication
stages. Row lengths are exchanged first and scanned into receiver offsets on
the preparation stream. One batched host-side `MPI_Alltoall` per join resolves
the per-peer character counts for every string column, after which character
buffers are exchanged through NCCL. Fixed/local-ready outputs are enqueued
before string outputs; string gather planning only waits for offsets, allowing
character communication to continue while output lengths and offsets are
computed. Exact character allocation occurs at one coordinated size readback.
Temporary length columns are retired promptly, while the established
whole-column lifetime and event contract remains in force.

The profiler exposes the additional work as
`join.column.string_lengths_exchange`,
`join.column.string_chars_exchange`, and
`join.column.string_offsets_rebuild`. Transfer rows report logical row counts,
character/length bytes, and peer counts; the host MPI metadata exchange remains
visible in the surrounding NVTX coordination range. Key and payload exchange
records identify their source with `column_side` and the zero-based
input-table `column_ordinal`; key exchanges are emitted as one profiling row
per transferred key column.

Focused tests cover both-side string payloads, empty and long strings,
rank-local empty inputs, zero-count peers, duplicates, misses, all three
schedules, and shuffled/left-ready/right-ready/both-ready inputs at two, three,
and four ranks. Q2, Q7, Q10, Q12, Q14, Q16, Q18, and Q21 passed exact SF1,
SF100, and SF300 golden verification on four GB200 GPUs without OOM.

Clean five-sample post-load medians were:

| Query | SF100 legacy ms | SF100 overlap ms | Improvement | SF300 legacy ms | SF300 overlap ms | Improvement |
|---|---:|---:|---:|---:|---:|---:|
| Q2 | 17.878 | 16.457 | 8.0% | 31.489 | 28.358 | 9.9% |
| Q7 | 43.135 | 40.086 | 7.1% | 111.010 | 102.788 | 7.4% |
| Q10 | 41.977 | 36.558 | 12.9% | 105.257 | 91.267 | 13.3% |
| Q12 | 29.830 | 27.060 | 9.3% | 80.886 | 74.902 | 7.4% |
| Q14 | 12.088 | 11.330 | 6.3% | 33.683 | 31.701 | 5.9% |
| Q16 | 23.459 | 22.887 | 2.4% | 49.533 | 48.065 | 3.0% |
| Q18 | 42.433 | 39.077 | 7.9% | 116.144 | 105.911 | 8.8% |
| Q21 | 212.890 | 211.773 | 0.5% | 621.933 | 618.941 | 0.5% |

The geomean improvement is 6.9% at SF100 and 7.1% at SF300; no individual
query regressed. A production Q10 SF300 NSYS capture measured the string-bearing
join's mean communication/match-materialization overlap at 2.260 ms across
four ranks, or 20.0% of communication-stream kernel-busy time. The other Q10
shuffle join, which has only fixed-width payloads, overlapped about 60.5%.
The original follow-up artifacts were written under
`/tmp/column_overlap_string_*`; that path was ephemeral and must not be treated
as a durable archive. Repeat or archive the experiment before relying on those
raw files.
