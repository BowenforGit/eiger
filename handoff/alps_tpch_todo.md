<!--
Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
See LICENSE and THIRD_PARTY_NOTICES.md for license information.
-->

# ALPS Distributed TPC-H TODO

This is the living task list for the next implementation batch. Durable
completed-work rationale is recorded in
`docs/DISTRIBUTED_TPCH_ARCHITECTURE.md`; benchmark operation and profiling live
in their topical guides rather than in a chronological implementation log.

## Pending Tasks

### Make partition GroupBy choose `log_num_partitions` adaptively

The partition GroupBy currently fixes `log_num_partitions = 16`. At SF=1000,
Q21 can place more than the 8,209-entry shared-memory table capacity in one
partition even after key bit-mixing. Do not change this yet: the partition
count affects allocations, radix-sort work, kernel geometry, timings, and the
comparability of existing artifacts.

The intended design is a pure, unit-tested selector based on
`min(input_rows, groups_upper_bound)`. It should target a conservative fraction
of the slot-table capacity (roughly 60%), round the required partition count up
to a power of two, and clamp it to audited minimum and maximum partition-bit
limits. Use unsigned 64-bit shifts, audit partition-offset/work-unit memory at
the maximum, retain the runtime overflow guard, and report the chosen value in
telemetry. If the required count exceeds the supported maximum, return a typed
capacity failure so synthesis can explicitly fall back to HASH rather than run
an unsafe clamped configuration. Validation must include selection-boundary
tests, skewed-key correctness against HASH, and the 8-GPU SF=1000 Q21 case.

The compute/communication phase breakdown task was completed on 2026-07-21.
Its timing contract is in `docs/DISTRIBUTED_TPCH_ARCHITECTURE.md` and its Nsight
interpretation is in `docs/DISTRIBUTED_TPCH_PROFILING.md`.
