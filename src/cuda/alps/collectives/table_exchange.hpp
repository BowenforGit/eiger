// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "alps/collectives/alltoall.hpp"
#include "alps/context.hpp"
#include "memory/table.hpp"

namespace alps::collectives {

// A locally reordered table whose rows are contiguous by destination rank.
// `pattern` describes only the send side until exchange_partitions() runs.
struct PartitionedTable {
    std::unique_ptr<gamax::table> table;
    // Set for an already-contiguous borrowed input (replicate/gather). When
    // `table` is present its owned view takes precedence.
    gamax::table_view borrowed;
    CommPattern pattern;
};

// Reusable row-routing metadata for column-granular partitioning. The
// permutation groups source rows by destination; `pattern` initially contains
// send counts/offsets and can be completed with receive counts/offsets once per
// distributed exchange.
struct HashPartitionPlan {
    std::unique_ptr<gamax::column> permutation;
    CommPattern pattern;
};

// Local source-side composition of a forced aligned collective. Segment order
// is main, pre, post. Replication has no pre segment. Bytes count logical table
// payload (fixed values, string lengths, and string characters), before NCCL
// protocol overhead or replication fan-out.
struct AlignmentStats {
    bool applied {false};
    std::array<std::int64_t, 3> rows {{-1, -1, -1}};
    std::array<std::int64_t, 3> bytes {{-1, -1, -1}};
};

// Derive a deterministic destination permutation and send pattern from all key
// columns. This performs no payload gather and no communication.
HashPartitionPlan make_hash_partition_plan(
    gamax::table_view keys,
    const Context& ctx,
    cudaStream_t stream);

// Fill receive counts/offsets with the rank-ordered MPI count exchange. Safe to
// call when the receive side is already complete (a no-op after validation).
void complete_receive_pattern(CommPattern& pattern, const Context& ctx);

// Apply a previously computed row permutation to one column. STRING columns
// use the same Arrow-aware gather path as the legacy full-table partitioner.
gamax::column_ptr apply_partition(
    gamax::column_view input,
    const HashPartitionPlan& plan,
    cudaStream_t stream);

// Exchange one already-partitioned column with a caller-owned row pattern.
// This is the column-granular counterpart of exchange_partitions(), used by
// schedules that intentionally interleave preparation, communication, and
// downstream compute. The pattern is copied because STRING exchange derives a
// second character-byte pattern from the row pattern. Alignment is
// intentionally unsupported here; callers must validate AlignmentPolicy::Off.
gamax::column_ptr exchange_partitioned_column(
    gamax::column_view send,
    CommPattern pattern,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend = Backend::NCCL);

// Locally partition rows by a deterministic hash over all key columns. This
// performs no communication. Equal mixed-type key tuples receive the same
// destination on every rank; STRING keys and payloads are supported.
PartitionedTable hash_partition(
    gamax::table_view in,
    const std::vector<gamax::size_type>& key_ordinals,
    const Context& ctx,
    cudaStream_t stream);

// Partition using a caller-computed INT32 destination rank per input row.
PartitionedTable partition_by_destination(
    gamax::table_view in,
    gamax::column_view destinations,
    const Context& ctx,
    cudaStream_t stream);

// Gather arbitrary rows from a heterogeneous table. STRING columns use the
// Arrow-aware gather path.
std::unique_ptr<gamax::table> gather_rows(
    gamax::table_view in,
    gamax::column_view row_indices,
    cudaStream_t stream);

// Exchange a table that has already been grouped by destination. Fixed-width
// columns use the row pattern directly. STRING columns exchange row lengths
// and character bytes, then rebuild Arrow offsets on the receiver.
std::unique_ptr<gamax::table> exchange_partitions(
    PartitionedTable partitioned,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend = Backend::NCCL,
    AlignmentPolicy alignment = AlignmentPolicy::Off,
    AlignmentStats* alignment_stats = nullptr);

// Replicate every rank's local shard onto every rank.
std::unique_ptr<gamax::table> replicate(
    gamax::table_view in,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend = Backend::NCCL,
    AlignmentPolicy alignment = AlignmentPolicy::Off,
    AlignmentStats* alignment_stats = nullptr);

// Gather every local shard onto `root`. Non-root ranks receive a valid empty
// table with the same schema.
std::unique_ptr<gamax::table> gather_to_root(
    gamax::table_view in,
    int root,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend = Backend::NCCL);

} // namespace alps::collectives
