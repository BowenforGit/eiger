// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Hash shuffle (all-to-all) over GPUs.
//
// Every row is sent to rank = hash(key) reduced into [0, size) (see
// alps::collectives::dest_rank). Because all ranks apply the same hash, rows
// with equal keys land on the same rank — the property a distributed equi-join
// relies on. Works for any `size` (power-of-two or not). Returns a new table
// holding the rows this rank received (any column count / supported types).

#include <memory>

#include "memory/table.hpp"
#include "types/data_type.hpp"
#include "alps/collectives/alltoall.hpp" // Backend
#include "alps/context.hpp"

namespace alps::collectives {

/// Optional per-stage timing (CUDA-event based, milliseconds). Populated only
/// when a non-null pointer is passed to shuffle().
struct ShuffleTiming {
    float partition_ms{0.0f};  // hash + radix partition + gather
    float comm_ms{0.0f};       // NCCL all-to-all transfer
};

/// Shuffle `in` by column `key_idx`. Any `ctx.size` is supported. Work is
/// issued on `stream`; the caller is responsible for synchronizing before
/// reading the result on the host. `backend` selects the all-to-all transport
/// (NCCL grouped Send/Recv vs copy-engine fabric PUSH); both produce the same
/// result and `comm_ms` is measured identically, for A/B comparison.
std::unique_ptr<gamax::table> shuffle(
    gamax::table_view  in,
    gamax::size_type   key_idx,
    const Context&     ctx,
    cudaStream_t       stream,
    ShuffleTiming*     timing = nullptr,
    Backend            backend = Backend::NCCL);

/// Misalignment-resistant shuffle (mirrors the dist_gamax reference) — a SEPARATE
/// implementation alongside shuffle() so the two can be A/B'd. Identical
/// partitioning, but the all-to-all is run via align_comm_buffers()'s
/// pre/main/post split so the bulk (`main`) transfer is 16B-aligned and gets full
/// NCCL bandwidth (see docs/ALPS_COMMUNICATION_STUDIES.md). Same result SET as
/// shuffle(); ROW ORDER WITHIN A RANK IS NOT PRESERVED (recv = [main|pre|post]).
std::unique_ptr<gamax::table> shuffle_aligned(
    gamax::table_view  in,
    gamax::size_type   key_idx,
    const Context&     ctx,
    cudaStream_t       stream,
    ShuffleTiming*     timing = nullptr);

} // namespace alps::collectives
