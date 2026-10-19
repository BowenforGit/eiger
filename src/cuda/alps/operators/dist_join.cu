// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/operators/dist_join.hpp"

#include <algorithm>

#include <cuda_runtime.h>
#include <mpi.h>

#include "operators/join/inner_join.hpp"
#include "alps/collectives/allgather.hpp"
#include "alps/collectives/shuffle.hpp"
#include "alps/operators/local_map_join.hpp"
#include "alps/error.hpp"

namespace alps::join {

namespace {

// Per-rank local join: the opt-in static_map path (unique-left PK-FK -- mirrors
// shared_map_join; narrow 3-col or wide P_L+1+P_R output) or gamax's general multiset
// inner_eq_join. `build` is the unique side, `probe` the other (callers pass them so).
std::unique_ptr<gamax::table> local_join(gamax::table_view build, gamax::table_view probe,
                                         const DistJoinHint& hint, cudaStream_t stream) {
    // Route to local_map_join for WIDE (multiset-wide is out of scope) or for the
    // opt-in narrow static_map path; hint.wide selects inline-value vs index+gather.
    // Both require a unique build side (PK-FK): wide assumes it (the multiset fallback
    // is bypassed), narrow gates on hint.local.unique_left.
    if (hint.wide || (hint.local_static_map && hint.local.unique_left))
        return local_map_join(build, probe, stream,
                              {hint.left_key_idx, hint.right_key_idx, hint.local_map_cg, hint.local_map_lf, hint.wide});
    return gamax::join::inner_eq_join(build, probe, hint.local, stream);
}

// Measured crossover (align_bench, GB200 np=4/8/12): below ~2^22 int32 rows/rank
// the aligned all-to-all's 3-sweep overhead outweighs its bandwidth win.
constexpr gamax::size_type kAlignRowsPerRank = gamax::size_type{1} << 22;

// --- shuffle strategy: hash-shuffle both sides, local join per rank. ---
std::unique_ptr<gamax::table> inner_eq_join_shuffle(
    gamax::table_view   left,
    gamax::table_view   right,
    const DistJoinHint& hint,
    const Context&      ctx,
    cudaStream_t        stream,
    DistJoinTiming*     timing) {

    collectives::ShuffleTiming lt{}, rt{};
    collectives::ShuffleTiming* lt_p = timing ? &lt : nullptr;
    collectives::ShuffleTiming* rt_p = timing ? &rt : nullptr;

    // Decide, per side, whether to use the misalignment-resistant shuffle. The
    // gate is on the GLOBAL per-rank average row count (one MPI_Allreduce), so
    // every rank picks the same variant for each collective.
    long long local_rows[2]  = { static_cast<long long>(left.num_rows()),
                                 static_cast<long long>(right.num_rows()) };
    long long global_rows[2] = { 0, 0 };
    ALPS_MPICHECK(MPI_Allreduce(local_rows, global_rows, 2, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));

    const int world = ctx.size > 0 ? ctx.size : 1;
    auto use_aligned = [&](long long total_rows) {
        switch (hint.shuffle_align) {
            case DistJoinHint::ShuffleAlign::on:  return true;
            case DistJoinHint::ShuffleAlign::off: return false;
            case DistJoinHint::ShuffleAlign::autosel:
            default:
                return (total_rows / world) >= static_cast<long long>(kAlignRowsPerRank);
        }
    };
    const bool left_aligned  = use_aligned(global_rows[0]);
    const bool right_aligned = use_aligned(global_rows[1]);

    // Shuffle both inputs on the join key; equal keys co-locate.
    auto left_s  = left_aligned
        ? collectives::shuffle_aligned(left, hint.left_key_idx, ctx, stream, lt_p)
        : collectives::shuffle(left, hint.left_key_idx, ctx, stream, lt_p);
    auto right_s = right_aligned
        ? collectives::shuffle_aligned(right, hint.right_key_idx, ctx, stream, rt_p)
        : collectives::shuffle(right, hint.right_key_idx, ctx, stream, rt_p);

    // Local inner join on this rank's shard.
    cudaEvent_t ev0{}, ev1{};
    if (timing) {
        ALPS_CUDACHECK(cudaEventCreate(&ev0));
        ALPS_CUDACHECK(cudaEventCreate(&ev1));
        ALPS_CUDACHECK(cudaEventRecord(ev0, stream));
    }

    auto out = local_join(left_s->view(), right_s->view(), hint, stream);

    if (timing) {
        ALPS_CUDACHECK(cudaEventRecord(ev1, stream));
        ALPS_CUDACHECK(cudaEventSynchronize(ev1));
        ALPS_CUDACHECK(cudaEventElapsedTime(&timing->join_ms, ev0, ev1));
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        timing->partition_ms = lt.partition_ms + rt.partition_ms;
        timing->comm_ms      = lt.comm_ms + rt.comm_ms;
    }

    return out;
}

// --- broadcast strategy: all-gather the smaller side, join vs each rank's shard
//     of the larger side. Output schema stays [left cols | right cols]. ---
std::unique_ptr<gamax::table> inner_eq_join_broadcast(
    gamax::table_view   left,
    gamax::table_view   right,
    const DistJoinHint& hint,
    const Context&      ctx,
    cudaStream_t        stream,
    DistJoinTiming*     timing) {

    // Choose the side to gather from GLOBAL row counts so all ranks agree.
    long long local_rows[2]  = { static_cast<long long>(left.num_rows()),
                                 static_cast<long long>(right.num_rows()) };
    long long global_rows[2] = { 0, 0 };
    ALPS_MPICHECK(MPI_Allreduce(local_rows, global_rows, 2, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));

    bool gather_left;
    switch (hint.broadcast_side) {
        case DistJoinHint::BcastSide::left:  gather_left = true;  break;
        case DistJoinHint::BcastSide::right: gather_left = false; break;
        case DistJoinHint::BcastSide::autosel:
        default:                             gather_left = (global_rows[0] <= global_rows[1]); break;
    }

    // comm: replicate the small side to every rank (NCCL broadcast all-gather;
    // the aligned variant is counterproductive for broadcast -- see align_bench).
    // allgather reports the pure NCCL-transfer time (same semantics as the
    // shuffle's comm_ms), excluding its MPI count exchange and recv alloc.
    float comm_ms = 0.0f;
    auto gathered = collectives::allgather(gather_left ? left : right, ctx, stream,
                                           timing ? &comm_ms : nullptr);

    // Local join. Swap args by which side travelled so the output is [left|right].
    cudaEvent_t j0{}, j1{};
    if (timing) {
        ALPS_CUDACHECK(cudaEventCreate(&j0));
        ALPS_CUDACHECK(cudaEventCreate(&j1));
        ALPS_CUDACHECK(cudaEventRecord(j0, stream));
    }
    auto out = gather_left
        ? local_join(gathered->view(), right, hint, stream)
        : local_join(left, gathered->view(), hint, stream);

    if (timing) {
        ALPS_CUDACHECK(cudaEventRecord(j1, stream));
        ALPS_CUDACHECK(cudaEventSynchronize(j1));
        ALPS_CUDACHECK(cudaEventElapsedTime(&timing->join_ms, j0, j1));
        timing->comm_ms      = comm_ms;
        timing->partition_ms = 0.0f; // broadcast has no partition stage
        cudaEventDestroy(j0);
        cudaEventDestroy(j1);
    }

    return out;
}

} // namespace

std::unique_ptr<gamax::table> inner_eq_join(
    gamax::table_view   left,
    gamax::table_view   right,
    const DistJoinHint& hint,
    const Context&      ctx,
    cudaStream_t        stream,
    DistJoinTiming*     timing) {

    switch (hint.strategy) {
        case DistJoinHint::Strategy::broadcast:
            return inner_eq_join_broadcast(left, right, hint, ctx, stream, timing);
        case DistJoinHint::Strategy::autoselect:
            // TODO(bench): choose shuffle vs broadcast by the measured size
            // crossover. Until then, run the robust shuffle path.
            return inner_eq_join_shuffle(left, right, hint, ctx, stream, timing);
        case DistJoinHint::Strategy::shuffle:
        default:
            return inner_eq_join_shuffle(left, right, hint, ctx, stream, timing);
    }
}

} // namespace alps::join
