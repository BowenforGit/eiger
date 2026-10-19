<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# ALPS communication studies

Status: consolidated design and benchmark record

This document preserves the durable conclusions from the original ALPS
component, shuffle, all-gather, direct-P2P, broadcast-join, and distributed-join
benchmark plans. Those files were useful while the code was being built, but
their API sketches, directory layouts, and implementation checklists now
duplicate or contradict the implementation.

The current implementation is under `src/cuda/alps`. This document explains
why the communication paths have their present shape, what the early studies
established, and which results must constrain future changes.

## 1. Stable design principles

### One rank per GPU

ALPS assumes one MPI process per GPU. Every host thread that touches CUDA must
select that rank's device. MPI and NCCL initialization, collective execution,
and teardown must occur in the same order on every rank.

Collective validation is itself collective: validate and agree on success
before any rank enters NCCL or a fabric operation. A rank-local early return
can strand peers in a collective. Unrecoverable asymmetric failures should
abort the MPI job instead of attempting an unmatched reduction.

### Row ownership and rank mapping

Hash exchange maps a 32-bit ownership hash into any world size with
multiplicative range reduction:

```text
destination = (uint32(hash) * world_size) >> 32
```

This is uniform for non-power-of-two rank counts and is equivalent to selecting
the high bits when the rank count is a power of two. The local radix partition
sorts a separate destination-rank column by its low bits. Sorting the high bits
of a signed raw hash is unsafe because CUB transforms the sign bit for signed
radix order while a raw-bit histogram does not. The old approach produced
plausible partition offsets but sent rows to the wrong ranks.

The current production exchange applies an independently seeded XXH32 avalanche
before rank reduction. The rationale and downstream-hash failure that led to
that versioned ownership hash are documented in
`docs/DISTRIBUTED_TPCH_ARCHITECTURE.md`.

### Deterministic collective order

All ranks must issue per-column and per-peer communication in the same order,
including zero-count transfers. The schedule may not depend on rank-local
sizes: skew could otherwise make ranks choose different orders and deadlock.

### Allocation and measurement

Communication libraries and fabric/VMM allocations live partly outside the
project RMM arena. Leave an explicit reserve and prewarm the resource before
measurement. Allocate reusable buffers for the largest tested case outside the
timed loop; repeated `cudaMalloc` or arena growth can dominate small-message
results and distort comparisons between transports.

`CudaMM` produced large and configuration-dependent allocation stalls in the
original join study. The comparable join results therefore used the same
persistent user-space/RMM arena for every strategy. See
`docs/MEMORY_ALLOCATOR.md` for the current allocator contract.

## 2. Benchmark methodology

### Per-peer size is the communication variable

Collective microbenchmarks sweep bytes per peer rather than total rows. Holding
per-peer size fixed avoids mixing message-size effects with changes in rank
count. Allocate once at the maximum size, warm up, time repeated operations,
and reduce per-iteration latency with `MAX` across ranks. The maximum rank is
the distributed critical path.

Report both:

- total processed bandwidth, including loopback/self traffic when relevant;
- cross-GPU bandwidth, excluding self traffic; and
- exact logical bytes, not an assumed aligned size.

The figures are application-payload rates. They are not protocol or wire-byte
measurements.

### Distributed-join data generation

The original PK/FK generator assigned each rank a disjoint primary-key block
and tiled foreign keys over the global primary-key range. That gives an exact
match ratio, exact global output cardinality, and balanced rank ownership for
uniform inputs. Keep data generation and query execution outside the timing
loop, but rerun the complete distributed join for every algorithm cell. A
shuffle-once comparison would measure only the local join and hide the real
strategy cost.

The historical `fk_uniform_id` helper printed a warning to stdout. Benchmark
drivers that still use it must keep machine-readable CSV on a separate stream
or suppress/redirect the warning.

### Alignment experiment

The shuffle study deliberately used an odd per-peer count so packed chunk
offsets became misaligned. This conflates misaligned offsets with counts not
divisible by the vector width, but both arise naturally from hash-partitioned
variable counts and therefore represent the real operator.

The follow-up `{pre, main, post}` split aligned the large middle portion of an
all-to-all transfer. It recovered about 1.5-2.3x in the bandwidth-bound shuffle
regime, but applying the same technique to broadcast-based all-gather made that
path about 30% slower. Alignment is therefore a collective-specific policy,
not a universally beneficial wrapper.

## 3. NCCL all-to-all and shuffle findings

On eight H200 GPUs, the aligned uniform all-to-all reached approximately
2.7 TB/s aggregate cross-GPU bandwidth at the largest tested size. Small
messages were latency-bound; large messages saturated the fabric.

The unaligned study reduced large-message per-GPU bandwidth to approximately
52%, 58%, and 65% of aligned bandwidth at 2, 4, and 8 GPUs. The penalty appeared
only after the latency-dominated regime and varied at NCCL protocol thresholds.
Consequently, uniform-message microbenchmarks are an upper bound for a real
hash shuffle, whose peer segments are unequal and usually unaligned.

The distributed-join study weak-scaled from 5.23 to 25.14 billion tuples/s at
one through eight H200 GPUs for its largest uniform shape. Local join time
stayed approximately constant while shuffle cost grew modestly. It also
exposed the partition-hash join's fixed 4096-row build-bucket limit: the
post-shuffle distribution crossed the limit at configurations that passed on
one GPU. Unsupported cells were rejected collectively so the sweep could
continue.

## 4. Direct fabric P2P

Direct P2P uses CUDA fabric-exportable VMM allocations so independently
launched processes can map one another's memory across nodes. It requires a
working IMEX deployment and explicit device/stream ordering. Ordinary project
columns are not automatically fabric-exportable; production integration needs
an explicit staging or fabric-backed ownership boundary.

Two ownership directions were studied:

- push: read local memory and write remote memory;
- pull: read remote memory and write local memory.

Push requires a system-scope visibility edge before completion is published;
the original SM path used `__threadfence_system`. Pull naturally keeps writes
local and avoids that fence. Large-message push and pull were within roughly
5%; pull was 1.1-1.4x faster for some small cross-node messages, while push was
slightly faster at saturation. The difference is partly the correctness cost
of the push fence, not a pure fabric read-versus-write comparison.

On GB200, direct SM or copy-engine P2P delivered roughly 1.5-2.4x the NCCL
all-to-all bandwidth in the tested configurations. Copy-engine transfers were
competitive or best on one and two nodes but plateaued near 405-430 GB/s at
three nodes, while SM transfers remained near 680 GB/s. The same behavior
appeared in all-gather. Use SM as the robust multi-node baseline; do not infer a
general CE advantage from single-node results.

These microbenchmarks established transport potential, not a promise that a
full relational operator will see the same speedup. Partitioning, staging,
memory capacity, host rendezvous, and downstream compute remain on the
critical path.

## 5. All-gather and broadcast join

The table all-gather must support unequal row counts. The NCCL implementation
therefore uses one broadcast per root with that root's count rather than fixed
count `ncclAllGather`. The original direct-P2P benchmark covered uniform
messages only and was not promoted as a variable-count table operator.

The GB200 all-gather study found direct P2P roughly 1.6-2.4x faster than the
NCCL broadcast-per-root implementation. SM push and pull were again close.
Copy-engine push led on one and two nodes but collapsed to approximately NCCL
performance at three nodes.

Broadcast join replicates the smaller input and joins it with each rank's
disjoint large-side shard. The union of per-rank outputs is therefore the
global result without double-counting. It is advantageous only when the small
side is sufficiently small and rank count is modest:

- at FK:PK 4, broadcast was already worse by 8-12 GPUs;
- at FK:PK 16, the crossover was around 8 GPUs;
- at FK:PK 64, broadcast remained beneficial through 12 GPUs in the tested
  weak-scaling cases.

Shuffle weak-scaled nearly flat because each rank retained similar work.
Broadcast degraded as the replicated table grew with world size and imposed a
fixed floor under strong scaling. A real automatic selector therefore needs
global byte estimates, world size, existing placement, and a calibrated
crossover model. The early `autoselect` value was intentionally a shuffle
stub, not a cost model.

## 6. Operational findings

The early container and cluster work exposed configuration failures that can
look like algorithm bugs:

- NVLink-SHARP/NVLS initialization failed in a container without the required
  fabric-manager/MNNVL setup; disabling NVLS allowed NCCL ring/tree paths to
  initialize.
- HPC-X UCC occasionally faulted during `MPI_Init` while allocating host shared
  memory; excluding the UCC collective component avoided the unrelated crash.
- An arena that reserves nearly all HBM can starve NCCL or fabric scratch.
  Configure the reserve before the first allocation and use the same policy on
  every rank.
- A fabric P2P result is meaningful only when the VMM export/import and IMEX
  path is the one actually exercised; a local CUDA peer-access result is not a
  multi-node substitute.

Current launch scripts own the supported environment settings. Treat the items
above as diagnosis guidance rather than commands to copy blindly onto a new
platform.

## 7. Scope and later work

The original studies deliberately did not solve skew-aware partition
assignment, packed wide-table transfer, a local self-copy shortcut, a complete
broadcast/shuffle cost model, or variable-count direct-P2P all-gather. Some of
those concerns are now addressed by the distributed TPC-H planner and
ColumnOverlap/CE-pull work; others remain workload-dependent optimization
opportunities.

For the current operator and transport designs, see:

- `docs/DISTRIBUTED_TPCH_ARCHITECTURE.md`
- `docs/ALPS_COLUMN_OVERLAP_JOIN.md`
- `docs/ALPS_CE_PULL_SHUFFLE.md`
- `docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md`
