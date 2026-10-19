// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// Reusable NCCL all-to-all (the communication core of the shuffle).
//
// A CommPattern fully describes who-sends-what: for each peer p (p in [0,size)),
// how many elements this rank sends to p and from where in the send buffer, and
// how many it receives from p and where they land in the recv buffer. Counts are
// in ELEMENTS (not bytes); offsets are element offsets within the column.
//
// `alltoall_column` does one grouped NCCL Send/Recv sweep into a PRE-ALLOCATED
// recv column (no allocation — so callers can keep allocation out of a timing
// loop). `alltoall` is a convenience that allocates the recv table and loops
// columns. `shuffle()` and the shuffle microbenchmark both build a CommPattern
// and call these.

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "alps/context.hpp"
#include "exec/step_profile.hpp"

namespace alps::collectives {

/// NCCL all-to-all bandwidth collapses (here ~0.4-0.65x, see
/// docs/ALPS_COMMUNICATION_STUDIES.md) when a per-peer chunk's address/count is
/// not 16-byte aligned, because NCCL's user<->staging copy is 128-bit vectorized.
/// 16 bytes is the alignment NCCL wants for full bandwidth.
constexpr size_t kOptimalBufferAlignment = 16;

/// Selects the transport for a collective. NCCL = grouped Send/Recv (or
/// Broadcast); CE = the legacy copy-engine PUSH path with fabric staging;
/// CE_PULL = sender-arena/receiver-pull transport. ColumnOverlap uses its
/// full-staging schedule; global/Legacy collectives use a bounded slot ring.
enum class Backend { NCCL, CE, CE_PULL };

// Forced alignment policy for production table collectives. Off preserves the
// rank/peer row order. On splits transfers into aligned bulk and ragged
// segments; it preserves the row set and cross-column row association but may
// reorder rows across segment boundaries.
enum class AlignmentPolicy { Off, On };

struct CommPattern {
    std::vector<gamax::size_type> send_counts;   // [size] elements sent to peer p
    std::vector<gamax::size_type> send_offsets;  // [size] element offset in send buf
    std::vector<gamax::size_type> recv_counts;   // [size] elements received from peer p
    std::vector<gamax::size_type> recv_offsets;  // [size] element offset in recv buf

    /// [size] PUSH destination: element offset in peer p's recv buffer where THIS
    /// rank's chunk to p must land (== peer p's recv_offsets[rank]). Empty for the
    /// NCCL path; populated once per pattern by finalize_dst_offsets() before the
    /// CE path runs, so the exchange is amortized across all columns of a table.
    std::vector<gamax::size_type> dst_offsets;

    gamax::size_type total_recv() const {
        gamax::size_type s = 0;
        for (auto c : recv_counts) s += c;
        return s;
    }
};

/// Fill `pat.dst_offsets` via one MPI_Alltoall of `recv_offsets`: after it, each
/// rank knows where its chunk lands in every peer's recv. COLLECTIVE; call once
/// per pattern before alltoall_column_ce / allgather_column_ce. (Correct for the
/// all-gather pattern too — it yields recv_offsets[rank] for every peer.)
void finalize_dst_offsets(CommPattern& pat, const Context& ctx);

/// One grouped NCCL all-to-all of a single column into a pre-allocated recv
/// column (recv must hold at least pat.total_recv() elements). Issues on `stream`.
void alltoall_column(gamax::column_view send, gamax::mutable_column_view recv,
                     const CommPattern& pat, const Context& ctx, cudaStream_t stream,
                     gamax::exec::MessageTraceDescriptor trace = {});

/// Copy-engine all-to-all — drop-in for alltoall_column (identical signature).
/// PUSHes each peer's chunk straight from `send` into that peer's fabric recv
/// staging region (ctx.fabric_scratch) at pat.dst_offsets[p], fences cross-rank
/// completion with a stream-ordered NCCL barrier, then copies the gathered recv
/// out into `recv`. Requires finalize_dst_offsets(pat) to have been called.
/// Fully stream-ordered: correct after the caller synchronizes `stream`.
void alltoall_column_ce(gamax::column_view send, gamax::mutable_column_view recv,
                        const CommPattern& pat, const Context& ctx, cudaStream_t stream);

/// The transfer core of alltoall_column_ce: the per-peer PUSH fan-out + the
/// stream-ordered NCCL completion barrier, with NO copy-out and NO trailing reuse
/// fence. After it returns (once `stream` drains) the gathered result lives in
/// ctx.fabric_scratch->recv, laid out exactly per recv_offsets — it is NOT
/// delivered to any caller column. This is the part NCCL also pays for (cross-GPU
/// transfer + completion); alltoall_column_ce wraps it with the CE-specific
/// copy-out into `recv` plus the reuse fence that protects the staging across
/// columns. Exposed so the bench can time the transfer in isolation
/// (--backend ce_nocopy), separating it from the copy-out tax. Requires
/// finalize_dst_offsets(pat). NOTE: with no trailing fence, the staging is only
/// safe for back-to-back calls because each call's pushes are stream-ordered after
/// the prior call's completion barrier (no reader of the staging exists here).
void alltoall_column_ce_push(gamax::column_view send, const CommPattern& pat,
                             const Context& ctx, cudaStream_t stream);

/// Convenience: allocate a recv table (rows = pat.total_recv(), same schema as
/// `send`) and all-to-all every column into it.
std::unique_ptr<gamax::table> alltoall(gamax::table_view send, const CommPattern& pat,
                                       const Context& ctx, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Misalignment-resistant all-to-all (mirrors the dist_gamax reference).
// ---------------------------------------------------------------------------

/// Element alignment for a column whose smallest element type is `min_type_size`
/// bytes: the number of elements spanning kOptimalBufferAlignment (16) bytes —
/// e.g. int32 -> 4, int64 -> 2. Counts/offsets that are multiples of this value
/// are 16B-aligned.
size_t compute_item_alignment(size_t min_type_size);

/// Split an arbitrary (unaligned) all-to-all CommPattern into three patterns —
/// returned as {pre, main, post} — so the bulk transfer (`main`) is 16B-aligned
/// on BOTH the send and recv side, while only the tiny ragged head/tail
/// (`pre`/`post`, each < item_alignment elements per peer) stay unaligned. The
/// recv buffer is laid out [ all main | all pre | all post ], so ROW ORDER IS
/// NOT PRESERVED (fine for joins / set-semantics). Exchanges each segment's recv
/// counts via MPI internally; only reads `unaligned.send_counts`/`send_offsets`.
/// `min_type_size` is the smallest column element size of the table being moved.
std::array<CommPattern, 3> align_comm_buffers(const CommPattern& unaligned,
                                              size_t min_type_size, const Context& ctx);

/// Variant used by heterogeneous table exchange after it computes the least
/// common row alignment required by every fixed-width representation in the
/// table (including STRING length rows).
std::array<CommPattern, 3> align_comm_buffers_by_item_alignment(
    const CommPattern& unaligned, size_t item_alignment, const Context& ctx);

} // namespace alps::collectives
