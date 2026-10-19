<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# Distributed TPC-H architecture

Status: implemented architecture and maintainer contract

This document describes the durable architecture, invariants, and design
decisions for distributed TPC-H. It replaces the original implementation plan,
plan re-evaluation, chronological process log, and general implementation log.
Those records mixed proposed APIs, temporary query-specific fallbacks, and
current behavior. The source tree is authoritative for exact types and flags;
this document explains why the system is structured this way and which
correctness boundaries must be preserved.

## 1. Execution model

The distributed runtime uses one MPI rank per GPU:

```text
TPC-H logical IR
  -> ordinary local rewrite and lowering
  -> distribution-aware physical-plan rewrite
  -> shared local/distributed executor
  -> root-complete result and golden verification
```

The logical TPC-H builders continue to express relational semantics. Placement
is a physical concern represented on ExecPlan values and steps. Keeping
distribution below logical IR lets the single-GPU and distributed runners use
the same builders and local rewrite rules, and lets physical strategy change
without changing query meaning.

The MPI runner therefore:

1. builds and lowers the same Q1-Q22 logical plans as the local runner;
2. applies a distribution-aware physical rewrite;
3. resolves each source according to the distributed source policy;
4. executes local and collective steps in one ordered executor; and
5. returns a complete result on the coordinator for verification.

There is no active query-specific result-merge or rank-zero full-query fallback.

## 2. Placement contract

ExecPlan values can describe these placements:

- `Unknown`: no safe ownership claim is available;
- `LocalShard`: the rank owns a disjoint but otherwise arbitrary shard;
- `HashPartitioned`: equal values of the recorded keys belong to one rank;
- `RangePartitioned`: rows are divided by recorded ordering boundaries;
- `Replicated`: every rank owns the same complete table; and
- `RootOnly`: the coordinator owns the complete table and peers hold an empty
  schema-compatible placeholder.

Placement is a correctness property, not only a performance hint. Local
operators may preserve it, remap key ordinals, or invalidate it. In particular,
a projection/filter that drops a partition key must not retain the old hash or
range claim. That bug once caused a later semi-join to skip a required shuffle
and accept only rank-local membership matches.

Replicated inputs require equal care. Re-running a distributed aggregate,
distinct, sort, or TopK over replicated data multiplies state or duplicates
results by world size. Placement-preserving local operations retain
replication; collective rewrites are suppressed when the requested placement
already holds.

`Concat` preserves placement only when its inputs have compatible global
multiplicity. Combining replicated and disjoint inputs cannot safely inherit
either property and is rejected or downgraded rather than guessed.

## 3. Source ownership

The shared source policy marks `nation` and `region` as replicated and the
remaining TPC-H base tables as sharded. The same immutable policy configures
the Parquet reader and physical planner so they cannot disagree about row
ownership.

For single-file datasets, sharded tables are assigned by Parquet row group.
For manifest-controlled multi-file datasets, assignment occurs at file
granularity and is not followed by another row-group partition. Replicated
tables are read completely on every rank. Cache identity includes placement
policy and, for sharded reads, rank and world size.

File assignment is still `LocalShard`. A contiguous sequence of files is not
`RangePartitioned` unless the manifest provides proven key boundaries and the
planner records those keys. See `docs/TPCH_MULTIFILE_DATASET.md` for the data
publication and assignment contract.

## 4. Generic table exchange

Distributed data movement is split into two reusable concerns:

- local row partitioning computes a deterministic destination, permutation,
  and send pattern without communication;
- table exchange negotiates receives and moves arbitrary schemas according to
  that pattern.

Fixed-width columns use the selected collective transport. Arrow-layout STRING
columns exchange row lengths and character bytes separately, rebuild 64-bit
offsets on the receiver, and preserve empty/zero-row schemas. `replicate` and
`gather_to_root` are table-exchange patterns, not query-specific serializers.

Composite ownership hashing uses the same mixed numeric/STRING row semantics
as local equality/grouping. A critical correctness fix made exact STRING
equality check length before bytes: without it, an empty string compared equal
to every string and same-prefix values of different lengths could compare
equal.

Zero-row inputs are first-class. Every rank still executes the same collective
sequence, and all operators must construct correctly typed empty outputs.

## 5. Versioned shuffle ownership hash

The current ownership scheme is named `xxh32-seeded-v2` and is included in
physical-plan fingerprints and campaign compatibility checks. It applies an
independently seeded XXH32 avalanche to the folded composite row hash before
multiply-high rank reduction.

The extra avalanche is not cosmetic. Direct rank selection once conditioned
each rank's rows to a narrow band of the same hash consumed by a downstream
CUCO linear-probing group-by. At SF3000 Q18, row counts and shuffle bytes were
balanced, yet the conditioned starting-slot domain was smaller than the key
count and produced an extreme probe chain. Independent ownership mixing keeps
the downstream hash domain well distributed on each rank.

Changing the ownership hash can alter placement, performance, saved physical
plans, and campaign comparability. Any future change must be versioned, include
fixed hash vectors, cover non-power-of-two rank counts, and test independence
from downstream hash-table indexing.

## 6. Distributed relational operators

### Scalar aggregation

Scalar SUM, COUNT, MIN, and MAX reduce mergeable local states. AVG carries SUM
and COUNT until after the global merge; averaging rank-local averages is
incorrect. Empty ranks contribute the operation's identity.

### Group-by and distinct

Distributed group-by performs local partial aggregation, hash exchange by all
group keys, destination-local merge, and finalization. AVG again remains
`(sum, count)` until the final stage. COUNT partials merge with SUM.

Distributed distinct optionally removes local duplicates, shuffles by every
distinct key, and performs destination-local distinct. Equal keys must be
co-located before the final local operation.

### Ordering

Distributed TopK takes local candidates, gathers at most `k` rows per rank, and
computes the final result on the coordinator. Full distributed sort uses
sampling, splitter replication, mixed-type row classification, range exchange,
local sort, and rank-ordered gather.

TPC-H ordering is not always total. Q18 omits `o_orderkey` from its ORDER BY, so
two valid implementations can return tied rows in a different order. The
current row-position verifier can report such a result as a mismatch even when
all ordered keys and values are valid. A tie-aware verifier remains needed.

### Joins

Distributed equality join supports shuffle and broadcast placement. Semi- and
anti-join may shuffle both sides or broadcast the retained-membership right
side. Broadcasting the retained left side of a semi/anti join is invalid
because it changes result multiplicity. Cross join broadcasts one input and
reuses an already replicated scalar/table where possible.

The local physical join remains responsible for raw output layout and
materialization. Distributed planning only establishes compatible input
placement and propagates the resulting ownership conservatively.

TPC-H-specific Spark-derived strategy overrides use semantic join signatures,
not raw step numbers. Planning rejects unmatched or ambiguous overrides before
rewriting any step. This avoids silently applying an old decision after a local
rewrite changes plan numbering.

## 7. Collective and failure invariants

- Every rank walks the same physical step sequence.
- Validation and allocation that can fail must complete rank-wide before the
  first collective in an operation.
- Zero-row ranks participate in every collective.
- Rank-local early returns around collective work are forbidden.
- An asymmetric runner exception calls `MPI_Abort`; entering an error
  `MPI_Allreduce` can itself mismatch a peer already inside MPI or NCCL.
- CUDA streams must be joined before memory used by another stream is retired,
  remapped, or returned to an arena.
- Collective resources are destroyed before `MPI_Finalize`.
- A sticky CUDA execution error is not allocator OOM and must not be hidden by
  resource reset.

These requirements are more important than graceful single-rank recovery. A
prompt, diagnosable job abort is preferable to an indefinite collective hang.

## 8. Benchmark contract

The distributed benchmark resolves and uploads projected sources once, then
adaptively warms the post-load plan. Clean end-to-end samples and instrumented
step/phase samples are separate passes so profiling does not contaminate the
headline latency.

End-to-end latency is maximum-rank monotonic wall time around the complete
post-load plan and final stream synchronization. It excludes loading, planning,
verification, output, and result destruction. Step and phase timings are GPU
event intervals. CPU orchestration and host MPI latency require Nsight Systems.

Direct phases on different streams may overlap. Their durations are useful as
work measures but must not be added as serial elapsed time. Attribution uses
the union of child intervals; residual time is the parent interval minus that
union. Logical byte counters describe Arrow table payload, including string
offsets and characters, rather than NCCL wire traffic.

Campaign configuration is immutable. Binary hashes, dataset metadata,
topology, environment, physical-plan fingerprints, and ownership-hash version
are frozen so resume/retry cannot silently change the experiment. See
`docs/DISTRIBUTED_TPCH_BENCHMARK_GUIDE.md` and
`docs/DISTRIBUTED_TPCH_PROFILING.md`.

## 9. Specialized communication paths

The legacy NCCL path remains the portable baseline. ColumnOverlap separates
key matching and per-column payload movement so communication can overlap HASH
join work. CE-pull uses fabric VMM staging and copy engines as an opt-in column
transport. These paths deliberately remain explicit alternatives because their
benefit depends on shape, node count, workspace capacity, and deployment.

Their detailed scheduling, memory-ordering, acceptance data, and limitations
are documented in:

- `docs/ALPS_COLUMN_OVERLAP_JOIN.md`
- `docs/ALPS_CE_PULL_SHUFFLE.md`
- `docs/ALPS_COMMUNICATION_STUDIES.md`

## 10. Validation boundary

The generic distributed executor is tested at power-of-two and non-power-of-two
rank counts, with mixed fixed-width/STRING schemas, duplicates, zero-row ranks,
and already-satisfied placement. End-to-end acceptance gathers a complete
result to rank 0 and compares Q1-Q22 with DuckDB-generated Parquet goldens.

Published milestones include complete Q1-Q22 golden verification at SF1,
SF100, and SF300 on the documented GB200 configurations, plus larger targeted
runs. Historical measurements remain evidence for those specific machines and
dates, not timeless performance promises.

The durable correctness boundary is broader than TPC-H: placement propagation,
STRING equality, collective order, empty-rank behavior, and ownership-hash
independence must remain valid for any distributed plan.
