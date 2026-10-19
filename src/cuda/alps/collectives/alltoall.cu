// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/alltoall.hpp"

#include <algorithm>
#include <array>
#include <numeric>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "alps/convert.hpp"
#include "alps/error.hpp"

namespace alps::collectives {

using gamax::size_type;

void alltoall_column(gamax::column_view send, gamax::mutable_column_view recv,
                     const CommPattern& pat, const Context& ctx, cudaStream_t stream,
                     gamax::exec::MessageTraceDescriptor trace) {
    const int            world = ctx.size;
    const gamax::data_type dt   = send.type();
    const ncclDataType_t   ncclt = alps::convert::to_nccl_type(dt);
    const size_t           esz   = gamax::size_of(dt);

    const char* sbuf = send.data<char>();
    char*       rbuf = recv.data<char>();

    ALPS_NCCLCHECK(ncclGroupStart());
    for (int p = 0; p < world; ++p) {
        if (pat.send_counts[p] > 0) {
            if (ctx.message_trace_sink) {
                ctx.message_trace_sink->record_message(
                    gamax::exec::MessageApi::NcclSend, p, -1, dt,
                    pat.send_counts[p], static_cast<std::int64_t>(esz), trace);
            }
            ALPS_NCCLCHECK(ncclSend(sbuf + pat.send_offsets[p] * esz, pat.send_counts[p],
                                    ncclt, p, ctx.nccl_comm, stream));
        }
        if (pat.recv_counts[p] > 0) {
            if (ctx.message_trace_sink) {
                ctx.message_trace_sink->record_message(
                    gamax::exec::MessageApi::NcclRecv, p, -1, dt,
                    pat.recv_counts[p], static_cast<std::int64_t>(esz), trace);
            }
            ALPS_NCCLCHECK(ncclRecv(rbuf + pat.recv_offsets[p] * esz, pat.recv_counts[p],
                                    ncclt, p, ctx.nccl_comm, stream));
        }
    }
    ALPS_NCCLCHECK(ncclGroupEnd());
}

void finalize_dst_offsets(CommPattern& pat, const Context& ctx) {
    const int world = ctx.size;
    pat.dst_offsets.resize(world);
    // MPI_Alltoall: after this, dst_offsets[p] on rank r holds rank p's
    // recv_offsets[r] — i.e. where r's chunk to p lands in p's recv buffer.
    const auto mpi_t = alps::convert::mpi_size_type();
    ALPS_MPICHECK(MPI_Alltoall(pat.recv_offsets.data(), 1, mpi_t,
                               pat.dst_offsets.data(), 1, mpi_t, ctx.mpi_comm));
}

std::unique_ptr<gamax::table> alltoall(gamax::table_view send, const CommPattern& pat,
                                       const Context& ctx, cudaStream_t stream) {
    const gamax::size_type ncols     = send.num_columns();
    const gamax::size_type recv_rows = pat.total_recv();

    std::vector<std::unique_ptr<gamax::column>> recv_cols(ncols);
    for (gamax::size_type c = 0; c < ncols; ++c) {
        recv_cols[c] = gamax::make_column(send.get_column(c).type(), recv_rows, stream);
    }
    for (gamax::size_type c = 0; c < ncols; ++c) {
        alltoall_column(send.get_column(c), recv_cols[c]->mutable_view(), pat, ctx, stream);
    }
    return gamax::make_table(std::move(recv_cols));
}

size_t compute_item_alignment(size_t min_type_size) {
    if (min_type_size == 0) return 1;
    const size_t ia = kOptimalBufferAlignment / min_type_size;
    if (ia * min_type_size == kOptimalBufferAlignment) return ia;
    // 16 not a multiple of the type size (exotic): smallest #elems spanning a
    // whole number of 16B units.
    return std::lcm(kOptimalBufferAlignment, min_type_size) / min_type_size;
}

std::array<CommPattern, 3> align_comm_buffers_by_item_alignment(
    const CommPattern& unaligned, size_t item_alignment, const Context& ctx) {
    const int       size = ctx.size;
    const size_type ia = static_cast<size_type>(std::max<size_t>(
        1, item_alignment));

    CommPattern pre, main, post;
    for (CommPattern* p : {&pre, &main, &post}) {
        p->send_counts.assign(size, 0);
        p->send_offsets.assign(size, 0);
        p->recv_counts.assign(size, 0);
        p->recv_offsets.assign(size, 0);
    }

    // Per peer, carve [offset, offset+cnt) into: pre = leading bytes up to the
    // next aligned offset; main = the aligned-offset, aligned-length bulk; post =
    // the trailing bytes after the last aligned offset. pre/post each hold
    // < ia elements; a nonempty main's offset and count are multiples of ia.
    for (int i = 0; i < size; ++i) {
        const size_type offset = unaligned.send_offsets[i];
        const size_type cnt    = unaligned.send_counts[i];

        const size_type start_correction =
            (ia - (offset % ia)) % ia; // -> next aligned offset
        const size_type pre_count = std::min(cnt, start_correction);
        const size_type main_start = offset + pre_count;
        const size_type remaining = cnt - pre_count;
        const size_type post_count = remaining % ia;
        const size_type main_count = remaining - post_count;
        const size_type post_start = main_start + main_count;

        pre.send_offsets[i] = offset;
        pre.send_counts[i] = pre_count;

        // When the chunk ends before the next aligned row, main is empty and
        // starts at `end`. Keeping that empty offset inside the source matters
        // for STRING exchange, which reads the Arrow offset at every segment
        // boundary even when the segment contains no rows.
        main.send_offsets[i] = main_start;
        main.send_counts[i] = main_count;

        post.send_offsets[i] = post_start;
        post.send_counts[i] = post_count;
    }

    // Exchange each segment's per-peer recv counts.
    const auto mpi_t = alps::convert::mpi_size_type();
    ALPS_MPICHECK(MPI_Alltoall(pre.send_counts.data(),  1, mpi_t, pre.recv_counts.data(),  1, mpi_t, ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Alltoall(main.send_counts.data(), 1, mpi_t, main.recv_counts.data(), 1, mpi_t, ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Alltoall(post.send_counts.data(), 1, mpi_t, post.recv_counts.data(), 1, mpi_t, ctx.mpi_comm));

    // Recv layout: [ all main | all pre | all post ]. main's recv offsets are
    // prefix sums of (aligned) main recv counts starting at 0 -> all aligned.
    size_type curr = 0;
    for (int i = 0; i < size; ++i) { main.recv_offsets[i] = curr; curr += main.recv_counts[i]; }
    for (int i = 0; i < size; ++i) { pre.recv_offsets[i]  = curr; curr += pre.recv_counts[i];  }
    for (int i = 0; i < size; ++i) { post.recv_offsets[i] = curr; curr += post.recv_counts[i]; }

    return {std::move(pre), std::move(main), std::move(post)};
}

std::array<CommPattern, 3> align_comm_buffers(
    const CommPattern& unaligned, size_t min_type_size, const Context& ctx)
{
    return align_comm_buffers_by_item_alignment(
        unaligned, compute_item_alignment(min_type_size), ctx);
}

} // namespace alps::collectives
