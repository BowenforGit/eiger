<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# ALPS TPC-H Development Notes - 2026-07-10

These notes describe the implementation and development priorities as of
July 10, 2026. They are a historical snapshot, not a statement of current
validation results or remaining work.

## Scope

The MPI runner supported all 22 TPC-H queries through SF100, but much of that
support was still hand-written per query. The development focus was to replace
query-specific implementations with reusable distributed physical planning
and operator machinery.

## Reference Artifacts

- Current architecture and preserved implementation rationale:
  `/storage/alps-tpch/docs/DISTRIBUTED_TPCH_ARCHITECTURE.md`
- Historical single-GPU decisions and limitations:
  `/storage/alps-tpch/docs/TPCH_JOURNEY.md`
- Main code path: `/storage/alps-tpch/src/cuda/tpch/run_tpch_ir_mpi.cpp`
- Local IR definitions: `/storage/alps-tpch/src/cuda/tpch/tpch_ir.cu`
- Spark plan references: `/home/scada/spark/plans/sf100/cbo_on/q*.static.txt`

## Recorded Functional Status

- `run_tpch_ir_mpi` accepted Q1-Q22.
- Rank-0 full-plan fallback was empty as of `82e9d29`.
- Full Q1-Q22 SF100 verification was recorded as passing on two nodes with 8 ranks total.
- SF300 was not evaluated in this milestone.
- Data lived under `/storage/tpch`; golden outputs were under `/storage/tpch/sf*/golden`.

## Design Direction

- Implement full distributed TPC-H support without relying on query-specific tricks.
- Introduce new distributed concepts at the physical plan step, not necessarily as IR nodes.
- Separate `Partition` from `Shuffle` so hash partition, range partition, and other strategies can be represented cleanly.
- Use Spark plans to guide rough shape, join order, and broadcast-vs-shuffle decisions, while keeping GPU-local workarounds as implementation details.
- Prioritize correctness and verify against golden Parquet results.

## Development Priorities at This Milestone

- Replace query-specific merge functions in `run_tpch_ir_mpi.cpp` with reusable distributed physical plan steps.
- Define physical steps for `Partition`, `Shuffle/Exchange`, `Broadcast`, distributed aggregate, distributed join, distributed semi/anti join, and distributed TopK/range sort.
- Generalize fixed-width distributed aggregation helpers beyond hand-coded uses for Q2/Q16/Q20/Q21.
- Add a general strategy for moving string columns, likely via dictionary encoding or a string-capable exchange path, instead of Q2/Q16-specific packing and rank-0 string attachment.
- Move final payload assembly and TopK away from rank 0 where practical.
- Keep row-group sharded reads for now; revisit file-part manifests or regenerated sharded Parquet only when scaling or ingest cost requires it.

## Runtime Note

- Multi-node launches need the build library paths exported, including `build/src/cuda/alps`, otherwise remote ranks may fail to find `libgamax_alps.so`.
