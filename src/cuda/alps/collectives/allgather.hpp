// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// All-gather over GPUs. Each rank contributes a chunk of count[rank] rows
// (counts differ per rank); after the op every rank holds the full concatenation
// [rank0 | rank1 | ... | rank(g-1)] of size sum(count). Implemented with one
// ncclBroadcast per root (root i sends its chunk into everyone's recv at the
// chunk-i offset) inside a single NCCL group — mirrors the dist_gamax reference,
// which uses broadcast precisely because the per-rank counts are unequal.

#include <memory>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "alps/collectives/alltoall.hpp" // CommPattern
#include "alps/context.hpp"
#include "exec/step_profile.hpp"

namespace alps::collectives {

/// Broadcast each rank's chunk into the gathered recv buffer. The CommPattern is
/// reused: send_counts[i] = count of rank i (same on every rank), send_offsets[i]
/// = chunk i's source offset on root i, and recv_offsets[i] = chunk i's element
/// offset in recv. On non-root ranks the corresponding send pointer is ignored.
/// `recv` must hold pat.total_recv() elements. Comm-only (no alloc).
void allgather_column(gamax::column_view send, gamax::mutable_column_view recv,
                      const CommPattern& pat, const Context& ctx, cudaStream_t stream,
                      gamax::exec::MessageTraceDescriptor trace = {});

/// Copy-engine all-gather — drop-in for allgather_column (identical signature).
/// Broadcast-PUSHes this rank's local chunk (`send`, offset 0, count
/// send_counts[rank]) into every peer's fabric recv staging region at
/// pat.dst_offsets[p] (== recv_offsets[rank] on every peer), fences with a
/// stream-ordered NCCL barrier, then copies the gathered recv out into `recv`.
/// Requires finalize_dst_offsets(pat) to have been called.
void allgather_column_ce(gamax::column_view send, gamax::mutable_column_view recv,
                         const CommPattern& pat, const Context& ctx, cudaStream_t stream);

/// The transfer core of allgather_column_ce: the broadcast-PUSH of this rank's
/// chunk into every peer's fabric staging + the stream-ordered NCCL completion
/// barrier, with NO copy-out and NO trailing reuse fence. After it returns (once
/// `stream` drains) the gathered result lives in ctx.fabric_scratch->recv, NOT in
/// any caller column. See alltoall_column_ce_push for the rationale and the
/// back-to-back-only safety note; used by the bench's --backend ce_nocopy.
void allgather_column_ce_push(gamax::column_view send, const CommPattern& pat,
                              const Context& ctx, cudaStream_t stream);

/// Real operator entry point: MPI_Allgather the per-rank row count, build the
/// pattern, allocate the gathered recv table (rows = sum(count)), broadcast every
/// column. Returns a new table holding the full gather on this rank.
///
/// If `comm_ms` is non-null it receives the CUDA-event time of the selected
/// backend's device sweep (excluding the MPI count/schema exchange and output
/// allocation). For NCCL this is the broadcast sweep; for CE_PULL it includes
/// sender staging, remote pulls, and STRING offset reconstruction.
std::unique_ptr<gamax::table> allgather(gamax::table_view send,
                                        const Context& ctx, cudaStream_t stream,
                                        float* comm_ms = nullptr,
                                        Backend backend = Backend::NCCL);

/// Misalignment-resistant all-gather (mirrors the dist_gamax reference) — a
/// SEPARATE implementation kept alongside allgather() so the two can be A/B'd.
/// Splits each rank's chunk into an aligned `main` bulk (source offset 0, count
/// rounded down to item_alignment) plus a ragged `post` tail of (count %
/// item_alignment) elements. The gathered buffer is laid out [ all mains | all
/// posts ], so every big `main` broadcast lands 16B-aligned on both ends (full
/// NCCL bandwidth) and only the tiny tails stay unaligned. Same result SET as
/// allgather(), but ROW ORDER IS NOT PRESERVED across the main/post boundary
/// (fine for joins / set semantics).
std::unique_ptr<gamax::table> allgather_aligned(gamax::table_view send,
                                                const Context& ctx, cudaStream_t stream);

} // namespace alps::collectives
