// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/shuffle.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/partition/partition.hpp"
#include "utils/transform.hpp"

#include "alps/collectives/alltoall.hpp"
#include "alps/collectives/hash.cuh"
#include "alps/convert.hpp"
#include "alps/error.hpp"

namespace alps::collectives {

namespace {

using gamax::size_type;

// Per-row destination rank in [0, world). We partition by the LOW
// ceil(log2(world)) bits of this small value rather than the high bits of the
// raw hash: CUB's radix sort twiddles the sign bit of signed keys, which would
// desync a high-bit partition from compute_partition_offsets' raw-bit histogram.
// The destination is in [0, world) <= [0, 2^rank_bits), so its low `rank_bits`
// bits ARE the value, and bit-31 twiddling is irrelevant. For a non-power-of-two
// world the partition still uses 2^rank_bits buckets; buckets [world, 2^rank_bits)
// simply receive no rows.
template <typename KeyT>
__global__ void dest_kernel(const KeyT* __restrict__ keys, int32_t* __restrict__ out, size_type n, int world) {
    size_type i = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = alps::collectives::dest_rank<KeyT>(keys[i], world);
}

template <typename KeyT>
void launch_dest(gamax::column_view key, gamax::mutable_column_view out, size_type n, int world, cudaStream_t stream) {
    if (n == 0) return;
    constexpr int threads = 256;
    const size_type blocks = (n + threads - 1) / threads;
    dest_kernel<KeyT><<<blocks, threads, 0, stream>>>(key.data<KeyT>(), out.data<int32_t>(), n, world);
}

std::unique_ptr<gamax::column> make_dest_column(gamax::column_view key, size_type n, int world, cudaStream_t stream) {
    auto d = gamax::make_column<int32_t>(n, stream);
    switch (key.type().id()) {
        case gamax::type_id::INT32:  launch_dest<int32_t>(key, d->mutable_view(), n, world, stream);  break;
        case gamax::type_id::INT64:  launch_dest<int64_t>(key, d->mutable_view(), n, world, stream);  break;
        case gamax::type_id::UINT32: launch_dest<uint32_t>(key, d->mutable_view(), n, world, stream); break;
        case gamax::type_id::UINT64: launch_dest<uint64_t>(key, d->mutable_view(), n, world, stream); break;
        default:
            throw std::runtime_error("alps::shuffle: key column must be a 32/64-bit integer");
    }
    return d;
}

} // namespace

std::unique_ptr<gamax::table> shuffle(
    gamax::table_view in,
    size_type         key_idx,
    const Context&    ctx,
    cudaStream_t      stream,
    ShuffleTiming*    timing,
    Backend           backend) {

    if (backend == Backend::CE_PULL) {
        throw std::invalid_argument(
            "shuffle: CE_PULL is only supported by the column-overlap "
            "equality-join transport");
    }

    const int       world = ctx.size;
    const size_type ncols = in.num_columns();
    const size_type nrows = in.num_rows();

    // size == 1: nothing to move, return a deep copy.
    if (world == 1) {
        std::vector<std::unique_ptr<gamax::column>> cols;
        cols.reserve(ncols);
        for (size_type c = 0; c < ncols; ++c) {
            cols.push_back(gamax::make_column_from_view(in.get_column(c), stream));
        }
        ALPS_CUDACHECK(cudaStreamSynchronize(stream));
        return gamax::make_table(std::move(cols));
    }

    // Bits needed to address all `world` ranks: ceil(log2(world)). For a
    // power-of-two world this is exactly log2(world); otherwise the radix
    // partition uses 2^rank_bits buckets and the high ones stay empty.
    const int rank_bits = ctx.rank_bits();

    // Set ALPS_SHUFFLE_PROFILE to break the partition stage into its sub-steps.
    static const bool kProfile = std::getenv("ALPS_SHUFFLE_PROFILE") != nullptr;
    const bool prof = timing && kProfile;

    cudaEvent_t ev_p0{}, ev_p1{}, ev_c0{}, ev_c1{};
    cudaEvent_t ev_dest{}, ev_sort{}, ev_gath{};
    if (timing) {
        ALPS_CUDACHECK(cudaEventCreate(&ev_p0));
        ALPS_CUDACHECK(cudaEventCreate(&ev_p1));
        ALPS_CUDACHECK(cudaEventCreate(&ev_c0));
        ALPS_CUDACHECK(cudaEventCreate(&ev_c1));
        ALPS_CUDACHECK(cudaEventRecord(ev_p0, stream));
    }
    if (prof) {
        ALPS_CUDACHECK(cudaEventCreate(&ev_dest));
        ALPS_CUDACHECK(cudaEventCreate(&ev_sort));
        ALPS_CUDACHECK(cudaEventCreate(&ev_gath));
    }

    // (1) destination rank per row, in [0, world).
    auto dest_col = make_dest_column(in.get_column(key_idx), nrows, world, stream);
    if (prof) ALPS_CUDACHECK(cudaEventRecord(ev_dest, stream));

    // (2) radix-partition rows by the low rank_bits bits of the destination.
    //     `perm` is the row permutation; `dest_sorted` is the destination in
    //     sorted order.
    auto parted = gamax::partition::arg_partition(
        dest_col->view(), 0, rank_bits, stream, nullptr, 0);
    auto& dest_sorted = parted.first;
    auto& perm        = parted.second;
    if (prof) ALPS_CUDACHECK(cudaEventRecord(ev_sort, stream));

    // (3) reorder every column into destination-rank order.
    std::vector<std::unique_ptr<gamax::column>> send_cols(ncols);
    for (size_type c = 0; c < ncols; ++c) {
        send_cols[c] = gamax::util::device::gather(perm->view(), in.get_column(c), stream);
    }
    if (prof) ALPS_CUDACHECK(cudaEventRecord(ev_gath, stream));

    // (4) per-bucket boundaries (inclusive prefix sum over the 2^rank_bits
    //     buckets; we read only the first `world` of them below).
    auto offsets = gamax::partition::compute_partition_offsets(dest_sorted->view(), 0, rank_bits, stream);
    if (timing) {
        ALPS_CUDACHECK(cudaEventRecord(ev_p1, stream));
    }

    // excl[r] = first row destined for rank r; excl[world] = nrows.
    std::vector<size_type> excl(world + 1);
    excl[0] = 0;
    ALPS_CUDACHECK(cudaMemcpyAsync(excl.data() + 1, offsets->data<size_type>(),
                                   sizeof(size_type) * world, cudaMemcpyDeviceToHost, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream)); // counts needed on host; also drains partition work

    CommPattern pat;
    pat.send_counts.resize(world);
    pat.send_offsets.resize(world);
    for (int r = 0; r < world; ++r) {
        pat.send_offsets[r] = excl[r];
        pat.send_counts[r]  = excl[r + 1] - excl[r];
    }

    // (5) exchange row counts so each rank knows how much it will receive.
    pat.recv_counts.resize(world);
    ALPS_MPICHECK(MPI_Alltoall(pat.send_counts.data(), 1, alps::convert::mpi_size_type(),
                               pat.recv_counts.data(), 1, alps::convert::mpi_size_type(), ctx.mpi_comm));

    pat.recv_offsets.resize(world);
    size_type recv_rows = 0;
    for (int r = 0; r < world; ++r) {
        pat.recv_offsets[r] = recv_rows;
        recv_rows += pat.recv_counts[r];
    }

    // (5b) CE backend: derive each rank's PUSH destination offsets once per
    //      pattern (one MPI_Alltoall, reused across columns). Untimed, like the
    //      count exchange above — kept out of comm_ms.
    if (backend == Backend::CE) finalize_dst_offsets(pat, ctx);

    // (6) allocate the receive table (same schema, recv_rows rows) — untimed,
    //     so comm_ms below stays pure communication.
    std::vector<std::unique_ptr<gamax::column>> recv_cols(ncols);
    for (size_type c = 0; c < ncols; ++c) {
        recv_cols[c] = gamax::make_column(in.get_column(c).type(), recv_rows, stream);
    }

    // (7) NCCL all-to-all via the shared alltoall_column (one grouped sweep/col).
    if (timing) {
        ALPS_CUDACHECK(cudaEventRecord(ev_c0, stream));
    }
    for (size_type c = 0; c < ncols; ++c) {
        if (backend == Backend::CE)
            alltoall_column_ce(send_cols[c]->view(), recv_cols[c]->mutable_view(), pat, ctx, stream);
        else
            alltoall_column(send_cols[c]->view(), recv_cols[c]->mutable_view(), pat, ctx, stream);
    }

    if (timing) {
        ALPS_CUDACHECK(cudaEventRecord(ev_c1, stream));
        ALPS_CUDACHECK(cudaEventSynchronize(ev_c1));
        ALPS_CUDACHECK(cudaEventElapsedTime(&timing->partition_ms, ev_p0, ev_p1));
        ALPS_CUDACHECK(cudaEventElapsedTime(&timing->comm_ms, ev_c0, ev_c1));
        if (prof) {
            float d=0, s=0, g=0, o=0;
            ALPS_CUDACHECK(cudaEventElapsedTime(&d, ev_p0,   ev_dest));
            ALPS_CUDACHECK(cudaEventElapsedTime(&s, ev_dest, ev_sort));
            ALPS_CUDACHECK(cudaEventElapsedTime(&g, ev_sort, ev_gath));
            ALPS_CUDACHECK(cudaEventElapsedTime(&o, ev_gath, ev_p1));
            std::fprintf(stderr, "[shuffle_profile] rank=%d nrows=%lld ncols=%lld  dest %.3f  sort %.3f  "
                         "gather %.3f  offsets %.3f  | partition %.3f ms\n",
                         ctx.rank, (long long)nrows, (long long)ncols, d, s, g, o, timing->partition_ms);
            cudaEventDestroy(ev_dest);
            cudaEventDestroy(ev_sort);
            cudaEventDestroy(ev_gath);
        }
        cudaEventDestroy(ev_p0);
        cudaEventDestroy(ev_p1);
        cudaEventDestroy(ev_c0);
        cudaEventDestroy(ev_c1);
    }

    return gamax::make_table(std::move(recv_cols));
}

std::unique_ptr<gamax::table> shuffle_aligned(
    gamax::table_view in,
    size_type         key_idx,
    const Context&    ctx,
    cudaStream_t      stream,
    ShuffleTiming*    timing) {

    const int       world = ctx.size;
    const size_type ncols = in.num_columns();
    const size_type nrows = in.num_rows();

    if (world == 1) {
        std::vector<std::unique_ptr<gamax::column>> cols;
        cols.reserve(ncols);
        for (size_type c = 0; c < ncols; ++c)
            cols.push_back(gamax::make_column_from_view(in.get_column(c), stream));
        ALPS_CUDACHECK(cudaStreamSynchronize(stream));
        return gamax::make_table(std::move(cols));
    }

    const int rank_bits = ctx.rank_bits();

    cudaEvent_t ev_p0{}, ev_p1{}, ev_c0{}, ev_c1{};
    if (timing) {
        ALPS_CUDACHECK(cudaEventCreate(&ev_p0));
        ALPS_CUDACHECK(cudaEventCreate(&ev_p1));
        ALPS_CUDACHECK(cudaEventCreate(&ev_c0));
        ALPS_CUDACHECK(cudaEventCreate(&ev_c1));
        ALPS_CUDACHECK(cudaEventRecord(ev_p0, stream));
    }

    // (1)-(4) identical partition pipeline to shuffle().
    auto dest_col = make_dest_column(in.get_column(key_idx), nrows, world, stream);
    auto parted   = gamax::partition::arg_partition(
        dest_col->view(), 0, rank_bits, stream, nullptr, 0);
    auto& dest_sorted = parted.first;
    auto& perm        = parted.second;

    std::vector<std::unique_ptr<gamax::column>> send_cols(ncols);
    for (size_type c = 0; c < ncols; ++c)
        send_cols[c] = gamax::util::device::gather(perm->view(), in.get_column(c), stream);

    auto offsets = gamax::partition::compute_partition_offsets(dest_sorted->view(), 0, rank_bits, stream);
    if (timing) ALPS_CUDACHECK(cudaEventRecord(ev_p1, stream));

    std::vector<size_type> excl(world + 1);
    excl[0] = 0;
    ALPS_CUDACHECK(cudaMemcpyAsync(excl.data() + 1, offsets->data<size_type>(),
                                   sizeof(size_type) * world, cudaMemcpyDeviceToHost, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream)); // counts needed on host; drains partition work

    // (5) unaligned SEND pattern; align_comm_buffers exchanges/derives the recv side.
    CommPattern un;
    un.send_counts.resize(world);
    un.send_offsets.resize(world);
    for (int r = 0; r < world; ++r) {
        un.send_offsets[r] = excl[r];
        un.send_counts[r]  = excl[r + 1] - excl[r];
    }

    size_t min_type_size = kOptimalBufferAlignment; // upper bound for the min-reduce
    for (size_type c = 0; c < ncols; ++c)
        min_type_size = std::min(min_type_size, gamax::size_of(in.get_column(c).type()));

    auto seg = align_comm_buffers(un, min_type_size, ctx);
    const CommPattern& pre  = seg[0];
    const CommPattern& main = seg[1];
    const CommPattern& post = seg[2];

    const size_type recv_rows = main.total_recv() + pre.total_recv() + post.total_recv();

    // (6) allocate recv table (untimed) — all three sweeps write disjoint ranges.
    std::vector<std::unique_ptr<gamax::column>> recv_cols(ncols);
    for (size_type c = 0; c < ncols; ++c)
        recv_cols[c] = gamax::make_column(in.get_column(c).type(), recv_rows, stream);

    // (7) three NCCL all-to-all sweeps: aligned bulk first, then the ragged edges.
    if (timing) ALPS_CUDACHECK(cudaEventRecord(ev_c0, stream));
    for (const CommPattern* pat : {&main, &pre, &post})
        for (size_type c = 0; c < ncols; ++c)
            alltoall_column(send_cols[c]->view(), recv_cols[c]->mutable_view(), *pat, ctx, stream);

    if (timing) {
        ALPS_CUDACHECK(cudaEventRecord(ev_c1, stream));
        ALPS_CUDACHECK(cudaEventSynchronize(ev_c1));
        ALPS_CUDACHECK(cudaEventElapsedTime(&timing->partition_ms, ev_p0, ev_p1));
        ALPS_CUDACHECK(cudaEventElapsedTime(&timing->comm_ms, ev_c0, ev_c1));
        cudaEventDestroy(ev_p0);
        cudaEventDestroy(ev_p1);
        cudaEventDestroy(ev_c0);
        cudaEventDestroy(ev_c1);
    }

    return gamax::make_table(std::move(recv_cols));
}

} // namespace alps::collectives
