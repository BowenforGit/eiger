// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/allgather.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include "memory/column.hpp"
#include "memory/table.hpp"

#include "alps/collectives/alltoall.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/convert.hpp"
#include "alps/error.hpp"

namespace alps::collectives {

using gamax::size_type;

void allgather_column(gamax::column_view send, gamax::mutable_column_view recv,
                      const CommPattern& pat, const Context& ctx, cudaStream_t stream,
                      gamax::exec::MessageTraceDescriptor trace) {
    const int            world = ctx.size;
    const ncclDataType_t dt    = alps::convert::to_nccl_type(send.type());
    const size_t         esz   = gamax::size_of(send.type());

    const char* sbuf = send.data<char>();
    char*       rbuf = recv.data<char>();

    // One broadcast per root: root i sends its chunk (count[i] elems) to every
    // rank's recv at chunk i's offset. Use a safe base pointer on non-roots:
    // their send pointer is ignored by NCCL, and their local input can be
    // shorter than root i's source offset.
    ALPS_NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < world; ++i) {
        if (pat.send_counts[i] <= 0) continue;
        const auto source_offset = ctx.rank == i ? pat.send_offsets[i] : 0;
        if (ctx.message_trace_sink) {
            ctx.message_trace_sink->record_message(
                gamax::exec::MessageApi::NcclBroadcast, -1, i, send.type(),
                pat.send_counts[i], static_cast<std::int64_t>(esz), trace);
        }
        ALPS_NCCLCHECK(ncclBroadcast(sbuf + source_offset * esz,
                                     rbuf + pat.recv_offsets[i] * esz,
                                     pat.send_counts[i], dt, i, ctx.nccl_comm, stream));
    }
    ALPS_NCCLCHECK(ncclGroupEnd());
}

std::unique_ptr<gamax::table> allgather(gamax::table_view send,
                                        const Context& ctx, cudaStream_t stream,
                                        float* comm_ms, Backend backend) {
    if (backend == Backend::CE_PULL) {
        return ce_pull_allgather(
            send, ctx, stream, ctx.ce_pull_options, comm_ms);
    }
    // Precondition: every rank passes the same schema (same num_columns and
    // matching column types) — a distributed table is uniform by construction.
    // The per-column ncclBroadcast group below would deadlock if ranks disagreed
    // on the column count. Only the per-rank ROW count may differ (exchanged
    // below); that is the whole point of the broadcast-based all-gather.
    const int       world = ctx.size;
    const size_type ncols = send.num_columns();
    const size_type nrows = send.num_rows();

    // Exchange per-rank row counts; build counts + recv offsets (exclusive prefix).
    CommPattern pat;
    pat.send_counts.resize(world);
    ALPS_MPICHECK(MPI_Allgather(&nrows, 1, alps::convert::mpi_size_type(),
                                pat.send_counts.data(), 1, alps::convert::mpi_size_type(), ctx.mpi_comm));
    pat.recv_counts  = pat.send_counts;
    pat.send_offsets.assign(world, 0); // local chunk is always at offset 0
    pat.recv_offsets.resize(world);
    size_type total = 0;
    for (int i = 0; i < world; ++i) { pat.recv_offsets[i] = total; total += pat.send_counts[i]; }

    // CE backend: derive PUSH destination offsets once (untimed, like the count
    // exchange above). For the gather these equal recv_offsets[rank] everywhere.
    if (backend == Backend::CE) finalize_dst_offsets(pat, ctx);

    // Allocate the gathered recv table (same schema, `total` rows) and gather
    // every column.
    std::vector<std::unique_ptr<gamax::column>> cols(ncols);
    for (size_type c = 0; c < ncols; ++c)
        cols[c] = gamax::make_column(send.get_column(c).type(), total, stream);
    auto out = gamax::make_table(std::move(cols));

    cudaEvent_t e0{}, e1{};
    if (comm_ms) {
        ALPS_CUDACHECK(cudaEventCreate(&e0));
        ALPS_CUDACHECK(cudaEventCreate(&e1));
        ALPS_CUDACHECK(cudaEventRecord(e0, stream));
    }
    for (size_type c = 0; c < ncols; ++c) {
        if (backend == Backend::CE)
            allgather_column_ce(send.get_column(c), out->get_column(c).mutable_view(), pat, ctx, stream);
        else
            allgather_column(send.get_column(c), out->get_column(c).mutable_view(), pat, ctx, stream);
    }
    if (comm_ms) ALPS_CUDACHECK(cudaEventRecord(e1, stream));

    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    if (comm_ms) {
        ALPS_CUDACHECK(cudaEventElapsedTime(comm_ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    return out;
}

std::unique_ptr<gamax::table> allgather_aligned(gamax::table_view send,
                                                const Context& ctx, cudaStream_t stream) {
    const int       world = ctx.size;
    const size_type ncols = send.num_columns();
    const size_type nrows = send.num_rows();

    // Item alignment from the smallest column element size (int32 -> 4 elems/16B).
    size_t min_type_size = kOptimalBufferAlignment; // upper bound for the min-reduce
    for (size_type c = 0; c < ncols; ++c)
        min_type_size = std::min(min_type_size, gamax::size_of(send.get_column(c).type()));
    const size_type ia = static_cast<size_type>(compute_item_alignment(min_type_size));

    // Per-rank row counts (same on every rank).
    std::vector<size_type> counts(world);
    ALPS_MPICHECK(MPI_Allgather(&nrows, 1, alps::convert::mpi_size_type(),
                                counts.data(), 1, alps::convert::mpi_size_type(), ctx.mpi_comm));

    // main = aligned bulk (source offset 0), post = the (count % ia) tail.
    CommPattern main, post;
    for (CommPattern* p : {&main, &post}) {
        p->send_counts.resize(world); p->send_offsets.resize(world);
        p->recv_counts.resize(world); p->recv_offsets.resize(world);
    }
    for (int i = 0; i < world; ++i) {
        const size_type post_cnt    = counts[i] % ia;
        const size_type aligned_cnt = counts[i] - post_cnt;
        main.send_counts[i] = aligned_cnt; main.send_offsets[i] = 0;           main.recv_counts[i] = aligned_cnt;
        post.send_counts[i] = post_cnt;    post.send_offsets[i] = aligned_cnt; post.recv_counts[i] = post_cnt;
    }
    // Recv layout [ all mains | all posts ]; main offsets prefix-sum aligned counts -> aligned.
    size_type curr = 0;
    for (int i = 0; i < world; ++i) { main.recv_offsets[i] = curr; curr += main.recv_counts[i]; }
    for (int i = 0; i < world; ++i) { post.recv_offsets[i] = curr; curr += post.recv_counts[i]; }
    const size_type total = curr;

    std::vector<std::unique_ptr<gamax::column>> cols(ncols);
    for (size_type c = 0; c < ncols; ++c)
        cols[c] = gamax::make_column(send.get_column(c).type(), total, stream);
    auto out = gamax::make_table(std::move(cols));

    for (size_type c = 0; c < ncols; ++c) {
        allgather_column(send.get_column(c), out->get_column(c).mutable_view(), main, ctx, stream);
        allgather_column(send.get_column(c), out->get_column(c).mutable_view(), post, ctx, stream);
    }

    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return out;
}

} // namespace alps::collectives
