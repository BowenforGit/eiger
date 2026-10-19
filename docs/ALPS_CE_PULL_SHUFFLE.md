<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# ALPS CE-pull shuffle design and evaluation

Status: implemented and evaluated as an opt-in transport
Last evaluated: 2026-08-09

CE-pull moves partitioned columns through sender-owned fabric VMM using
receiver-initiated copy-engine transfers. Final results remain ordinary
RMM-backed GaMax columns. It is available through two explicit execution
models:

- Legacy/global CE-pull adapters use a bounded two-slot ring and synchronize
  the complete collective before downstream compute.
- ColumnOverlap equality join, GroupBy, and SemiJoin use full staging, assigning
  every live column a distinct arena lease so preparation can finish before the
  intended compute-overlap window.

Both use round-robin sender waves and stream-memory doorbells. CE-pull does not
replace NCCL as the default because its benefit and memory cost depend on
operator shape, rank count, and deployment.

## 1. Why a pull transport exists

The earlier CE-push collective writes into receiver-owned VMM staging and then
copies that staging buffer into a normal column. The final local copy consumes
time and SM resources, while protecting a single cross-rank staging buffer
requires global reuse barriers.

CE-pull reverses ownership:

1. A sender partitions remote rows into a short-lived fabric-mapped VMM source
   region.
2. The self segment is placed directly into its final RMM output.
3. Each receiver pulls remote regions with `cudaMemcpyBatchAsync` directly into
   its final RMM output.
4. Device/stream-ordered doorbells publish source readiness and completion.

Removing communication kernels from SMs does not guarantee a faster query.
Copy engines still compete for HBM, NVLink/NVSwitch, and fabric resources.
Profiler overlap is necessary diagnostic evidence, but promotion decisions use
clean end-to-end latency and correctness.

## 2. Scope and deployment boundary

Supported behavior includes:

- one MPI rank per GPU and arbitrary positive world sizes;
- sender-owned fabric VMM mapped by every participating rank;
- fixed-width and Arrow-layout STRING columns;
- direct self placement and receiver pulls into ordinary RMM output;
- a bounded slot ring and a full-staging lease mode;
- stream-memory doorbells with a polling-kernel diagnostic wait mode;
- global/Legacy shuffle and all-gather adapters; and
- ColumnOverlap equality join, GroupBy, and SemiJoin consumers.

CE-pull does not alter the older CE-push backend, make `device_buffer` or
`column` own VMM, silently fall back payload traffic to NCCL, or establish
NVL72-wide performance from four/eight-GPU measurements. AntiJoin and arbitrary
broadcast/replication paths are not ColumnOverlap CE-pull consumers.

Fabric VMM requires working IMEX export/import across all processes and nodes.
Local CUDA peer access is not an adequate deployment test. A new host set must
pass the destructive cross-rank capability test before a benchmark; otherwise
an asynchronous visibility defect can strand a device-side wait indefinitely.

## 3. Stable design decisions

1. `CE_PULL` is a distinct, opt-in backend. NCCL remains the portable baseline.
2. VMM is transient sender storage only. Final outputs are normal RMM-backed
   columns.
3. Host planning, validation, mapping, descriptor creation, and allocation
   finish collectively before device enqueue.
4. Every rank submits the same logical operations in the same order, including
   empty columns and zero-byte peer transfers.
5. Per-column readiness is distinct from source reclamation.
6. Round-robin sender waves are the default. Unordered batch descriptors once
   concentrated traffic on a receiver and materially regressed latency.
7. Doorbell mode, schedule, slot count, and full-staging choice are explicit and
   recorded in experiment configuration.
8. Arena growth is allowed during warm-up and forbidden during measurement.
9. A caller receives an output only with an event/lifetime edge that protects
   every producer and remote reader.
10. End-to-end benefit and golden correctness decide promotion; overlap alone
    does not.

## 4. Host-planned session contract

Exact declarations live in `src/cuda/alps/collectives/ce_pull_options.hpp` and
`ce_pull_shuffle.hpp`. The durable boundary is a host-planned shuffle session
with separate prepare, pull, consumer-wait, output-release, and reclamation
actions.

Session creation is collective. It:

- validates schema, options, and logical operation order;
- completes receive patterns and exchanges remote source offsets;
- batches STRING character-count coordination;
- allocates final RMM outputs and receive workspaces;
- computes each column's VMM footprint and reserves bounded slots or
  full-staging leases;
- creates reusable copy descriptors; and
- agrees on rank-local validation/allocation errors before communication.

After that boundary, column preparation and pull enqueue do not allocate, call
MPI, map VMM, or synchronize the host. Consumers wait on an output-ready CUDA
event and release ownership onto a stream ordered after their last use.

## 5. Persistent context and memory

`alps::Context` lazily owns one CE-pull exchange context. It is freed
collectively before the NCCL communicator, MPI, and primary CUDA stream are
destroyed. It owns:

- a fabric VMM source arena;
- a small fabric control mapping;
- a high-priority transfer stream and a reclamation stream;
- bounded-slot events and one full-batch arena-reuse event;
- monotonic operation/epoch values;
- reusable host copy descriptors; and
- capacity, overflow, byte, and reuse telemetry.

The arena is separate from the older receiver-owned push scratch because the
two designs have incompatible ownership and synchronization semantics.

All regions are 256-byte aligned and the backing allocation is rounded to CUDA
fabric granularity. Remote storage excludes the self segment. A fixed-width
lease packs remote peer segments; a STRING lease packs row lengths and
character bytes, with the receiver rebuilding 64-bit Arrow offsets locally.

### Bounded mode

The default generic adapter uses a two-slot ring. Operation `k` uses
`k % slots`; reuse waits until every receiver has finished reading the old
source. Two slots were retained because three improved the original result by
only about 1.2% while increasing arena capacity by 50%.

Bounded mode is a useful lower-memory transport, but its backpressure can starve
the producer when long compute kernels begin before later payload columns have
been staged.

### Full staging

ColumnOverlap creates a full-staging batch. Every live logical column receives
a distinct aligned lease sized to its remote footprint. Capacity is therefore
the sum of simultaneously live leases rather than `slot_count` times the
largest column.

The larger arena is deliberate: all payload splits can be submitted before the
long match/grouping kernel starts, so the copy engine has READY work during the
useful compute window. Arena growth is a warm-up activity. A measured run
freezes growth and treats an insufficient arena as an explicit capacity error.

## 6. Device state machine and ordering

### Preparation

For each column, the preparation stream waits for its lease to be reusable,
partitions remote rows into sender VMM, writes the self segment to final output,
and publishes a monotonically increasing READY value. Publishing is ordered
after all source writes.

### Pull

The receiver transfer stream waits for READY from every sender, submits
round-robin `cudaMemcpyBatchAsync` waves, performs any receiver-local STRING
offset reconstruction, and records the column's output-ready event. Empty and
zero-byte operations follow the same state transitions.

### Reclamation

In bounded mode, each receiver publishes per-operation consumption after its
last source read; a sender reuses the slot only after observing consumption
from every receiver.

Full staging never reuses an individual column lease during the same shuffle,
so per-column consumption is redundant. After the final pull, each receiver
publishes one `SHUFFLE_DONE` value to every sender on its transfer stream. A
sender records the arena-reusable event only after observing completion from
every receiver. The next full-staging batch waits on that event before writing
the arena.

Eliminating the completion handshake entirely would be incorrect: a fast rank
can begin the next shuffle while another rank is still reading the previous
arena. Replacing it with a host barrier would discard the stream-ordered
overlap the protocol exists to provide.

### Doorbell modes

The primary path uses 64-bit CUDA stream-memory waits and writes on fabric
control memory. The polling mode replaces waits with one-warp kernels for
diagnosis, but still needs supported 64-bit stream-memory publication. Values
are monotonic so a late write from a previous arena use cannot satisfy a later
wait.

## 7. STRING coordination

STRING columns move as lengths plus characters:

1. remote row lengths are staged and pulled;
2. receiver offsets are reconstructed from lengths;
3. per-peer character counts are coordinated in the host planning phase;
4. character bytes are pulled into an exactly sized buffer; and
5. the normal Arrow-layout output takes ownership of offsets and characters.

The host character-count exchange is batched across the session. It must occur
before device enqueue; introducing a rank-local host decision between READY and
pull could make collective ordering diverge.

## 8. ColumnOverlap schedules

### Equality join

Keys are prepared and pulled first. Every payload split is then submitted
before matching begins. The transfer release occurs immediately before the
long `retrieve`/materialization work, allowing remote payload copies to overlap
useful HASH join computation. Each payload is waited on only when its output is
materialized.

### GroupBy

Local partial aggregation and the hash-partition plan are reused. Keys are
prepared first, followed by every expanded aggregate-state column, including
SUM/COUNT state for AVG. State pulls are released immediately before
destination group-ID construction; final merge waits on each required state.

### SemiJoin

The right/build batch contains its key. The left/probe batch contains the key
and retained payloads. Probe pulls are released immediately before building the
right-side hash set, the probe key is waited on before membership testing, and
payloads are waited on during result materialization. If neither side needs
partitioning, no CE session is created and telemetry reports zero traffic.

Unsupported operator/algorithm shapes retain their documented Legacy path.
Selecting a column exchange backend does not turn a semantic fallback into a
transport error.

## 9. Failure and lifetime model

- Capability and option mismatches fail at the collective host-planning
  boundary.
- Rank-local planning/allocation errors are agreed before a later collective or
  doorbell is entered.
- Errors after asynchronous publication cannot always be recovered without a
  job abort; peers may already be waiting on a value that will never arrive.
- Arena overflow while growth is frozen is a measured capacity failure, not a
  reason to resize silently.
- Session destruction does not prove remote reads completed. Normal completion
  or an explicit synchronized abort path must establish reclamation.
- Output release transfers lifetime to a stream; changing an owner's stream
  without an event does not create ordering.
- Context teardown is collective and occurs before MPI finalization.

These rules align with the allocator contract in `docs/MEMORY_ALLOCATOR.md`.

## 10. Instrumentation contract

Telemetry records arena capacity, control capacity, peak requested lease,
growth/overflow counts, enqueued columns, remote bytes, and slot reuse. A
successful measured repetition must report zero growth and zero overflow.

NVTX distinguishes preparation, READY waits, batch copies, offset rebuild,
completion publication, and consumer compute. Phase CSVs use common-origin
CUDA-event intervals. Overlapping phase durations are not additive; attribution
uses their interval union.

For overlap claims, distinguish:

- the fraction of copy time inside a consumer kernel; and
- the fraction of the consumer kernel covered by copy activity.

The first number can be high even when only a small part of a long kernel hides
communication. CE's own partition kernels are preparation, not useful consumer
compute.

## 11. Evidence and design evolution

### The bounded two-slot result

The first implementation used two globally shared slots. With round-robin
waves it improved all Q5/Q8/Q9 SF100/SF300 cells versus NCCL ColumnOverlap and
reduced combined RMM+VMM peak by 15.5-22.0%. It passed the predeclared 10% gate
on two of three queries at eight GPUs but not at four GPUs.

The initial unordered schedule had regressed by as much as 21.6%. Nsight
Systems showed receiver concentration: round-robin waves reduced the captured
Q8/SF300 P2P activity union from 23.6 to 13.8 ms and raised effective activity
bandwidth from 355.6 to 609.3 GB/s.

Despite the improved latency, the bounded design did not achieve its intended
overlap. In the critical Q8 join, only 2.253 ms of CE payload activity fell
inside a 23.538 ms retrieve kernel. The two key columns occupied both slots;
long match kernels then starved the preparation stream before most payloads
could become READY. This was the reason to accept a larger full-staging arena.

### Full-staging result

Paired four- and eight-GPU SF300 runs used the same binary and passed golden
verification. Medians are maximum-rank end-to-end latency after adaptive
warm-up:

| GPUs | Query | NCCL ColumnOverlap | Full-stage CE | Reduction |
|---:|:---:|---:|---:|---:|
| 4 | Q5 | 155.728 ms | 140.446 ms | 9.81% |
| 4 | Q8 | 162.197 ms | 145.640 ms | 10.21% |
| 4 | Q9 | 242.478 ms | 219.988 ms | 9.28% |
| 8 | Q5 | 83.436 ms | 73.155 ms | 12.32% |
| 8 | Q8 | 86.510 ms | 76.816 ms | 11.21% |
| 8 | Q9 | 130.377 ms | 113.442 ms | 12.99% |

On the critical joins, 95-98% of payload-copy time fell inside retrieve and no
copy tail remained. Copy activity covered 39-75% of the retrieve interval,
which is the more conservative measure of how much compute hid transport.

The capacity cost was material. Full staging used 2.08x, 2.56x, and 3.00x the
two-slot VMM capacity for the four-GPU Q5, Q8, and Q9 shapes respectively. All
measured repetitions nevertheless reported zero growth and zero overflow.

### Completion simplification

The first full-staging version retained per-column CONSUMED signaling inherited
from slot reuse. Replacing it with one `SHUFFLE_DONE` handshake removed 42-46%
of stream-memory wait/write calls in the evaluated queries without changing
copy submission count/order or overlap. Four/eight-GPU SF300 medians showed no
material regression, and repeated back-to-back joins verified arena reuse
without a host barrier.

### GroupBy and SemiJoin integration

Focused tests cover STRING keys/payloads, empty ranks, already-partitioned
zero-traffic input, caller-stream lifetimes, and frozen arena reuse. Q1-Q22 at
SF1/SF100 and targeted SF300 queries passed with ColumnOverlap operators using
CE-pull.

Eight-GPU SF300 operator-isolated medians were:

| Operator | Query | NCCL | CE-pull | Change |
|:---|:---:|---:|---:|---:|
| GroupBy | Q1 | 58.060 ms | 58.628 ms | +0.98% |
| GroupBy | Q9 | 915.884 ms | 916.995 ms | +0.12% |
| GroupBy | Q16 | 69.397 ms | 69.446 ms | +0.07% |
| SemiJoin | Q4 | 99.718 ms | 23.632 ms | -76.30% |
| SemiJoin | Q20 | 133.595 ms | 113.453 ms | -15.08% |
| SemiJoin | Q21 | 1,050.421 ms | 918.752 ms | -12.53% |

GroupBy was neutral at query scope; SemiJoin materially benefited. These are
results for the recorded GB200 topology, not an automatic-selection rule.

## 12. Limitations and future boundaries

- Full staging trades memory capacity for overlap. A byte-budgeted staging
  window may be needed for wider schemas or smaller-memory GPUs.
- The remaining critical cost in Q5/Q8/Q9 was payload split/preparation before
  count/retrieve, not a copy-engine tail. Earlier liveness-aware prefetch is a
  more promising boundary than additional sender scheduling for those shapes.
- Very short consumer kernels cannot hide a longer copy. Another independent
  operator is required; protocol tuning cannot create compute time.
- GroupBy's neutral result shows that removing SM communication is insufficient
  when other work dominates or fabric/HBM contention offsets the gain.
- Deployment must revalidate IMEX, fabric visibility, stream-memory attributes,
  copy-engine behavior, and memory reserve on every platform.
- No automatic NCCL/CE selector exists. It requires shape, topology, capacity,
  and overlap-cost modeling backed by measurements beyond the current grid.

## 13. Reproduction records

The self-contained experiment records are:

- `exp_results/ce_pull_tpch_20260807/` — bounded slots and sender scheduling;
- `exp_results/ce_full_staging_overlap_20260807/` — full-staging overlap;
- `exp_results/ce_full_staging_completion_20260808/` — completion protocol;
- `exp_results/ce_relational_tpch_20260809/` — GroupBy/SemiJoin integration.

For current benchmark setup and campaign semantics, use
`docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md`. For the higher-level
ColumnOverlap equality-join design, use `docs/ALPS_COLUMN_OVERLAP_JOIN.md`.
