// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/operators/shared_map_join.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cuda/std/tuple>

#include "alps/containers/shared_static_map.cuh"
#include "alps/operators/map_join_detail.cuh"   // detail::kEmptySentinel + build_compacted_output
#include "alps/error.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

namespace alps::join {

namespace {

using gamax::size_type;

template <int CG, typename T>
std::unique_ptr<gamax::table> run(gamax::table_view left, gamax::table_view right,
                                  containers::FabricStripedRegion& region,
                                  const Context& ctx, cudaStream_t stream,
                                  const SharedJoinHint& hint, SharedJoinTiming* timing) {
    const size_type lkey = hint.left_key_idx,  lval = size_type{1} - hint.left_key_idx;
    const size_type rkey = hint.right_key_idx, rval = size_type{1} - hint.right_key_idx;
    const size_type left_n  = left.num_rows();
    const size_type right_n = right.num_rows();

    const T* left_key_ptr  = left.get_column(lkey).data<T>();
    const T* left_val_ptr  = left.get_column(lval).data<T>();
    const T* right_key_ptr = right.get_column(rkey).data<T>();
    const T* right_val_ptr = right.get_column(rval).data<T>();

    const T empty = static_cast<T>(detail::kEmptySentinel);

    // Global map capacity = sum of every rank's local left rows (one shared table).
    long long local_left = static_cast<long long>(left_n), global_left = 0;
    ALPS_MPICHECK(MPI_Allreduce(&local_left, &global_left, 1, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));
    const std::size_t capacity = static_cast<std::size_t>(global_left > 0 ? global_left : 1);

    cudaEvent_t e0{}, e1{};
    ALPS_CUDACHECK(cudaEventCreate(&e0));
    ALPS_CUDACHECK(cudaEventCreate(&e1));
    auto tic = [&] { if (timing) ALPS_CUDACHECK(cudaEventRecord(e0, stream)); };
    auto toc = [&](float* dst) {
        if (!timing) return;
        ALPS_CUDACHECK(cudaEventRecord(e1, stream));
        ALPS_CUDACHECK(cudaEventSynchronize(e1));
        ALPS_CUDACHECK(cudaEventElapsedTime(dst, e0, e1));
    };

    // ---- construct: allocate + zero the global slot array (fabric-striped) ----
    tic();
    containers::FabricAllocator<cuco::pair<T, T>> alloc(&region);
    containers::SharedStaticMap<CG, T, T> map(
        capacity, hint.lf,
        cuco::empty_key{empty}, cuco::empty_value{empty},
        {}, {}, {}, {}, alloc, stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(stream)); // local slot init done
    // CRITICAL: cuco's ctor clears the WHOLE striped slot array to the sentinel
    // from EVERY rank (incl. peers' segments over the fabric). Without a cross-rank
    // barrier here, a fast rank's inserts race a slow rank's zeroing of the same
    // (remote) slots and get clobbered back to empty -> lost build rows -> missed
    // probes. Barrier so all ranks finish zeroing before ANY insert begins.
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
    toc(timing ? &timing->construct_ms : nullptr);

    // ---- build: insert this rank's local left (key -> value), then barrier ----
    // insert_async is async and inserts may target remote GPUs' slots, so EVERY
    // rank must finish ALL inserts before ANY rank probes (else a probe races a
    // peer's still-in-flight insert over the fabric).
    tic();
    auto pairs = thrust::make_zip_iterator(cuda::std::make_tuple(left_key_ptr, left_val_ptr));
    if (left_n > 0) map.insert_async(pairs, pairs + left_n, stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(stream)); // local inserts done + system-visible
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));       // all ranks done before any probe
    toc(timing ? &timing->build_ms : nullptr);

#ifndef NDEBUG
    // Precondition guard: globally-unique, non-sentinel left keys => the map's
    // occupancy equals the global left row count. map.size() scans the whole
    // striped table (over the fabric) so every rank computes the same value.
    {
        const long long occ = static_cast<long long>(map.size(stream));
        if (occ != global_left)
            std::fprintf(stderr,
                "[shared_map_join] WARNING: map.size()=%lld != global_left_rows=%lld -- "
                "left keys are not globally unique (or a key/value hit the -1 sentinel); "
                "result will be short.\n", occ, global_left);
    }
#endif

    // ---- probe (this rank's local right shard) -> found = value or -1 sentinel ----
    const size_type cap = right_n > 0 ? right_n : 1;
    auto found = gamax::make_column<T>(cap, stream);
    T* found_ptr = found->template data<T>();
    if (right_n > 0) {
        tic();
        map.find_async(right_key_ptr, right_key_ptr + right_n, found_ptr, stream);
        toc(timing ? &timing->probe_ms : nullptr);
    }
    // ---- compact matches into [found_left_value, right_key, right_value] ----
    auto out = detail::build_compacted_output<T>(
        right_n > 0 ? found_ptr : nullptr, right_key_ptr, right_val_ptr, right_n, stream,
        timing ? &timing->compact_ms : nullptr);

    // Trailing cross-rank fence. This rank's find/compact has finished reading the
    // shared striped slots (build_compacted_output synced the stream). A peer must
    // not start its NEXT shared_map_join -- whose map ctor zeroes the whole striped
    // table, including this rank's segment over the fabric -- until every rank has
    // finished probing here, else back-to-back joins reusing the region race a
    // straggler's probe against a fast rank's slot-zeroing (cf. shared_map_bench.cu).
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));

    ALPS_CUDACHECK(cudaEventDestroy(e0));
    ALPS_CUDACHECK(cudaEventDestroy(e1));
    return out;
}

// WIDE (int64-only, multi-payload). The slot stores key -> PADDED global left index
// (rank*max_rows + local); the P_L left payloads live in P_L caller-owned fabric
// regions (one per column), and the compaction GATHERS them by that index -- remote
// reads over the fabric. Mirrors run<> plus the payload copy/gather. See design in
// shared_map_join.hpp and the CRITICAL ordering note on the build barrier.
template <int CG>
std::unique_ptr<gamax::table> run_wide(
    gamax::table_view left, gamax::table_view right,
    containers::FabricStripedRegion& region,
    const std::vector<containers::FabricStripedRegion*>& payload_regions,
    const Context& ctx, cudaStream_t stream,
    const SharedJoinHint& hint, SharedJoinTiming* timing) {
    using T = int64_t;
    const size_type lkey = hint.left_key_idx, rkey = hint.right_key_idx;
    const size_type left_n  = left.num_rows();
    const size_type right_n = right.num_rows();
    const T* left_key_ptr  = left.get_column(lkey).data<T>();
    const T* right_key_ptr = right.get_column(rkey).data<T>();
    const T  empty = static_cast<T>(detail::kEmptySentinel);

    // capacity = SUM of local left rows; max_rows = MAX (the per-rank padded-id stride).
    long long ll = static_cast<long long>(left_n), g_sum = 0, g_max = 0;
    ALPS_MPICHECK(MPI_Allreduce(&ll, &g_sum, 1, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Allreduce(&ll, &g_max, 1, MPI_LONG_LONG, MPI_MAX, ctx.mpi_comm));
    const std::size_t capacity  = static_cast<std::size_t>(g_sum > 0 ? g_sum : 1);
    const long long   max_rows  = g_max > 0 ? g_max : 1;
    // base_id / idx_space are derived AFTER acquiring the payload regions, from the
    // region's ACTUAL slab stride (not max_rows) -- see the payload-copy phase. The
    // region rounds each rank's slab up to the alloc granularity, so addressing with
    // max_rows would put a rank's payload in a PEER's physical slab; using the real
    // slab stride keeps each rank's write/self-gather local regardless of rounding.

    // left payload columns = all non-key left columns, in column order.
    std::vector<size_type> lpay;
    for (size_type c = 0; c < left.num_columns(); ++c) if (c != lkey) lpay.push_back(c);
    const size_type P_L = static_cast<size_type>(lpay.size());

    cudaEvent_t e0{}, e1{};
    ALPS_CUDACHECK(cudaEventCreate(&e0));
    ALPS_CUDACHECK(cudaEventCreate(&e1));
    auto tic = [&] { if (timing) ALPS_CUDACHECK(cudaEventRecord(e0, stream)); };
    auto toc = [&](float* dst) {
        if (!timing) return;
        ALPS_CUDACHECK(cudaEventRecord(e1, stream));
        ALPS_CUDACHECK(cudaEventSynchronize(e1));
        ALPS_CUDACHECK(cudaEventElapsedTime(dst, e0, e1));
    };

    // ---- construct: allocate + zero the striped slot array (int64 key/index) ----
    tic();
    containers::FabricAllocator<cuco::pair<T, T>> alloc(&region);
    containers::SharedStaticMap<CG, T, T> map(
        capacity, hint.lf, cuco::empty_key{empty}, cuco::empty_value{empty},
        {}, {}, {}, {}, alloc, stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));   // construct->insert barrier (see run<>)
    toc(timing ? &timing->construct_ms : nullptr);

    // ---- payload copy: acquire P_L payload regions, memcpy this rank's payloads into
    //      its slab. CRITICAL: this is fenced by the build barrier below (issued before
    //      it), so all payload writes are system-visible before any peer's gather. ----
    // Request >= world*max_rows*8 per region; each region rounds its per-rank slab up
    // to the alloc granularity. Then address payloads with the region's ACTUAL slab
    // stride so rank p writes at p*slab (== its own physical segment, local write) and
    // the padded index p*slab_elems+local self-consistently gathers from p's slab.
    const std::size_t req_bytes = static_cast<std::size_t>(static_cast<long long>(ctx.size) * max_rows)
                                  * sizeof(T);
    std::vector<void*> pay_base(P_L, nullptr);
    tic();
    for (size_type c = 0; c < P_L; ++c)
        pay_base[c] = payload_regions[c]->acquire(req_bytes, stream);
    // All P_L regions share the same slab (same request), so region 0's is canonical.
    // P_L==0 (key-only left): no payload buffers -> stride is irrelevant (no left gather);
    // the stored index is then just a matched/non-matched marker, so max_rows is fine.
    const long long slab_elems = P_L > 0
        ? static_cast<long long>(payload_regions[0]->slab_bytes() / sizeof(T))
        : max_rows;
    const long long base_id    = static_cast<long long>(ctx.rank) * slab_elems;
    const size_type idx_space  = static_cast<size_type>(static_cast<long long>(ctx.size) * slab_elems);
    if (left_n > 0)
        for (size_type c = 0; c < P_L; ++c) {
            const T* src = left.get_column(lpay[c]).data<T>();
            void* dst = static_cast<char*>(pay_base[c]) + static_cast<std::size_t>(base_id) * sizeof(T);
            ALPS_CUDACHECK(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(left_n) * sizeof(T),
                                           cudaMemcpyDeviceToDevice, stream));
        }
    toc(timing ? &timing->payload_copy_ms : nullptr);

    // ---- build: insert (key -> padded id), then barrier (fences inserts + copies) ----
    tic();
    auto ids = gamax::make_column(gamax::data_type{gamax::type_id::INT64},
                                  left_n > 0 ? left_n : 1, stream);
    T* ids_ptr = ids->data<T>();
    if (left_n > 0) {
        gamax::rmm_exec_policy_handle policy{stream};
        thrust::sequence(
            policy.get(), ids_ptr, ids_ptr + left_n, static_cast<T>(base_id));
        auto pairs = thrust::make_zip_iterator(cuda::std::make_tuple(left_key_ptr, ids_ptr));
        map.insert_async(pairs, pairs + left_n, stream);
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));   // inserts + payload copies done + system-visible
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));         // all ranks done writing before any probe/gather
    toc(timing ? &timing->build_ms : nullptr);

    // ---- probe (local right) -> found = padded left index or -1 ----
    const size_type cap = right_n > 0 ? right_n : 1;
    auto found = gamax::make_column(gamax::data_type{gamax::type_id::INT64}, cap, stream);
    T* found_ptr = found->data<T>();
    if (right_n > 0) {
        tic();
        map.find_async(right_key_ptr, right_key_ptr + right_n, found_ptr, stream);
        toc(timing ? &timing->probe_ms : nullptr);
    }

    // ---- compact + gather: left payloads (fabric reads) + [right_key, right payloads] ----
    std::vector<gamax::column_view> left_views, right_views;
    for (size_type c = 0; c < P_L; ++c)
        left_views.emplace_back(pay_base[c], idx_space, gamax::data_type{gamax::type_id::INT64});
    right_views.push_back(right.get_column(rkey));
    for (size_type c = 0; c < right.num_columns(); ++c) if (c != rkey) right_views.push_back(right.get_column(c));
    auto out = detail::build_compacted_output_wide(
        right_n > 0 ? found_ptr : nullptr, right_n,
        left_views, right_views, stream,
        timing ? &timing->compact_ms : nullptr, timing ? &timing->gather_ms : nullptr);

    // ---- trailing fence: peers must finish gathering (reading fabric payloads/slots)
    //      before any rank re-acquires/rewrites in its NEXT join. Then release. ----
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
    for (size_type c = 0; c < P_L; ++c)
        payload_regions[c]->release(pay_base[c], req_bytes, stream);

    ALPS_CUDACHECK(cudaEventDestroy(e0));
    ALPS_CUDACHECK(cudaEventDestroy(e1));
    return out;
}

} // namespace

std::unique_ptr<gamax::table> shared_map_join(
    gamax::table_view              left,
    gamax::table_view              right,
    containers::FabricStripedRegion& region,
    const Context&                 ctx,
    cudaStream_t                   stream,
    const SharedJoinHint&          hint,
    SharedJoinTiming*              timing,
    const std::vector<containers::FabricStripedRegion*>& left_payload_regions) {

    if (hint.wide) {   // int64-only, multi-payload index+gather path
        // Collective validation (MAX-reduce fail code so all ranks throw together).
        static const char* kMsgW[] = {"",
            "wide: left and right need >=1 column (the key; payloads optional, P>=0)",
            "wide: key_idx out of range",
            "wide: all columns must be INT64",
            "wide: cg must be 1, 2, or 4",
            "wide: need >= P_L left_payload_regions (one per left payload column)"};
        int fail = 0;
        if (left.num_columns() < 1 || right.num_columns() < 1) fail = 1;
        else if (hint.left_key_idx  >= static_cast<size_type>(left.num_columns()) ||
                 hint.right_key_idx >= static_cast<size_type>(right.num_columns())) fail = 2;
        else {
            auto all_i64 = [](gamax::table_view t) {
                for (size_type c = 0; c < t.num_columns(); ++c)
                    if (t.get_column(c).type().id() != gamax::type_id::INT64) return false;
                return true;
            };
            if (!all_i64(left) || !all_i64(right)) fail = 3;
            else if (hint.cg != 1 && hint.cg != 2 && hint.cg != 4) fail = 4;
            else if (static_cast<size_type>(left_payload_regions.size()) <
                     static_cast<size_type>(left.num_columns()) - 1) fail = 5;
        }
        int gfail = 0;
        ALPS_MPICHECK(MPI_Allreduce(&fail, &gfail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
        if (gfail) {
            std::string msg = std::string("shared_map_join: ") + kMsgW[gfail];
            if (fail != gfail) msg += " (precondition failed on another rank)";
            throw std::runtime_error(msg);
        }
        switch (hint.cg) {
            case 1: return run_wide<1>(left, right, region, left_payload_regions, ctx, stream, hint, timing);
            case 2: return run_wide<2>(left, right, region, left_payload_regions, ctx, stream, hint, timing);
            case 4: return run_wide<4>(left, right, region, left_payload_regions, ctx, stream, hint, timing);
            default: throw std::runtime_error("shared_map_join: hint.cg must be 1, 2, or 4");
        }
    }

    // Validate preconditions COLLECTIVELY: each rank computes a failure code, we
    // MPI_Allreduce(MAX) it, and every rank throws together on any failure. A
    // per-rank throw here (before run()'s first collective at the MPI_Allreduce
    // below) would deadlock the ranks that passed -- they would block forever in
    // run()'s capacity Allreduce / barriers waiting for the rank that bailed.
    static const char* kMsg[] = {"",
        "left and right must have exactly 2 columns",
        "key_idx must be 0 or 1",
        "all key/value columns must share one type",
        "only INT32 or INT64 columns are supported",
        "cg must be 1, 2, or 4"};
    int fail = 0;
    gamax::type_id id = gamax::type_id::EMPTY;
    if (left.num_columns() != 2 || right.num_columns() != 2) fail = 1;
    else if (hint.left_key_idx < 0 || hint.left_key_idx > 1 ||
             hint.right_key_idx < 0 || hint.right_key_idx > 1) fail = 2;
    else {
        // Safe now: 2 columns each, key_idx in [0,1]. All four key/value columns
        // must share one type, INT32 or INT64.
        id = left.get_column(hint.left_key_idx).type().id();
        auto same = [&](const gamax::column_view& c) { return c.type().id() == id; };
        if (!same(left.get_column(size_type{1} - hint.left_key_idx)) ||
            !same(right.get_column(hint.right_key_idx)) ||
            !same(right.get_column(size_type{1} - hint.right_key_idx)))
            fail = 3;
        else if (id != gamax::type_id::INT32 && id != gamax::type_id::INT64) fail = 4;
        else if (hint.cg != 1 && hint.cg != 2 && hint.cg != 4) fail = 5;
    }
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(&fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (global_fail) {
        std::string msg = std::string("shared_map_join: ") + kMsg[global_fail];
        if (fail != global_fail) msg += " (precondition failed on another rank)";
        throw std::runtime_error(msg);
    }

    const bool i64 = (id == gamax::type_id::INT64);
    switch (hint.cg) {
        case 1: return i64 ? run<1, int64_t>(left, right, region, ctx, stream, hint, timing)
                           : run<1, int32_t>(left, right, region, ctx, stream, hint, timing);
        case 2: return i64 ? run<2, int64_t>(left, right, region, ctx, stream, hint, timing)
                           : run<2, int32_t>(left, right, region, ctx, stream, hint, timing);
        case 4: return i64 ? run<4, int64_t>(left, right, region, ctx, stream, hint, timing)
                           : run<4, int32_t>(left, right, region, ctx, stream, hint, timing);
        default: throw std::runtime_error("shared_map_join: hint.cg must be 1, 2, or 4");
    }
}

} // namespace alps::join
