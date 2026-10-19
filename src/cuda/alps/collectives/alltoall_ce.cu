// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/alltoall.hpp"

#include <cstdint>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvtx3/nvToolsExt.h>

#include "alps/collectives/fabric_scratch.hpp"
#include "alps/error.hpp"

namespace alps::collectives {

using gamax::size_type;

namespace {
inline CUdeviceptr as_cuptr(const void* p) {
    return static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(p));
}
} // namespace

void alltoall_column_ce_push(gamax::column_view send, const CommPattern& pat,
                             const Context& ctx, cudaStream_t stream) {
    const int    world = ctx.size;
    const size_t esz   = gamax::size_of(send.type());

    if (pat.dst_offsets.empty())
        throw std::runtime_error("alltoall_column_ce_push: pat.dst_offsets not populated "
                                 "(call finalize_dst_offsets first)");
    if (!ctx.fabric_scratch)
        throw std::runtime_error("alltoall_column_ce_push: ctx.fabric_scratch is null");

    FabricCollectiveScratch& sc = *ctx.fabric_scratch;
    const size_t my_recv_bytes = static_cast<size_t>(pat.total_recv()) * esz;

    // Grow-only fabric recv staging (sized to the global-max total_recv) + the
    // persistent fork/join sub-streams. Reused across columns and calls.
    sc.recv.ensure(my_recv_bytes, ctx);
    sc.ensure_streams(world, stream);

    const char* sbuf = send.data<char>();

    // (1) PUSH: fan each peer's chunk across a sub-stream straight from `send`
    //     into peer p's fabric recv at dst_offsets[p]. Self (p == rank) is just a
    //     local D2D into our own recv — handled uniformly.
    nvtxRangePushA("a2a_ce.push");
    ALPS_CUDACHECK(cudaEventRecord(sc.fork_event, stream));
    for (int i = 0; i < world; ++i) {
        int p = ctx.rank + i; if (p >= world) p -= world;   // (rank + i) % world
        const size_t cnt = static_cast<size_t>(pat.send_counts[p]);
        if (cnt == 0) continue;
        cudaStream_t s = sc.sub_streams[i];
        ALPS_CUDACHECK(cudaStreamWaitEvent(s, sc.fork_event, 0));
        CUdeviceptr dst = sc.recv.peer(p) + static_cast<size_t>(pat.dst_offsets[p]) * esz;
        CUdeviceptr src = as_cuptr(sbuf) + static_cast<size_t>(pat.send_offsets[p]) * esz;
        ALPS_CUCHECK(cuMemcpyAsync(dst, src, cnt * esz, reinterpret_cast<CUstream>(s)));
        ALPS_CUDACHECK(cudaEventRecord(sc.join_events[i], s));
        ALPS_CUDACHECK(cudaStreamWaitEvent(stream, sc.join_events[i], 0));
    }

    nvtxRangePop(); // a2a_ce.push

    // (2) Stream-ordered cross-rank completion fence: a 1-element in-place NCCL
    //     all-reduce won't complete until every rank's pushes (ordered before it
    //     on each rank's stream) have finished — and CE writes are system-visible
    //     on completion. Keeps the whole op async on `stream`.
    nvtxRangePushA("a2a_ce.barrier");
    ALPS_NCCLCHECK(ncclAllReduce(sc.barrier_buf.data(), sc.barrier_buf.data(), 1, ncclInt32, ncclSum,
                                 ctx.nccl_comm, stream));
    nvtxRangePop(); // a2a_ce.barrier
}

void alltoall_column_ce(gamax::column_view send, gamax::mutable_column_view recv,
                        const CommPattern& pat, const Context& ctx, cudaStream_t stream) {
    nvtxRangePushA("a2a_ce");
    // (1)+(2) Transfer + completion (the part NCCL also pays for); leaves the
    //         gathered result in ctx.fabric_scratch->recv laid out per recv_offsets.
    alltoall_column_ce_push(send, pat, ctx, stream);

    FabricCollectiveScratch& sc = *ctx.fabric_scratch;
    const size_t esz           = gamax::size_of(send.type());
    const size_t my_recv_bytes = static_cast<size_t>(pat.total_recv()) * esz;

    // (3) Copy-out: our fabric recv is laid out exactly per recv_offsets, so one
    //     contiguous D2D into the caller's recv column.
    nvtxRangePushA("a2a_ce.copyout");
    if (my_recv_bytes > 0) {
        ALPS_CUDACHECK(cudaMemcpyAsync(recv.data<char>(),
                                       reinterpret_cast<void*>(static_cast<uintptr_t>(sc.recv.own())),
                                       my_recv_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    nvtxRangePop(); // a2a_ce.copyout

    // (4) Trailing fence. The recv staging region is reused by the next column /
    //     the next call, so no peer may PUSH into it until THIS rank has finished
    //     reading it in the copy-out above. A second stream-ordered NCCL barrier
    //     makes every rank's *next* op's pushes wait for *all* ranks' copy-outs,
    //     closing the cross-rank WAR on the shared staging buffer (multi-column
    //     shuffle/allgather, or back-to-back calls). Collective: run on every rank
    //     unconditionally (even when my_recv_bytes == 0). NOTE: this fence belongs
    //     with the copy-out — alltoall_column_ce_push omits it (no reader there).
    nvtxRangePushA("a2a_ce.fence");
    ALPS_NCCLCHECK(ncclAllReduce(sc.barrier_buf.data(), sc.barrier_buf.data(), 1, ncclInt32, ncclSum,
                                 ctx.nccl_comm, stream));
    nvtxRangePop(); // a2a_ce.fence
    nvtxRangePop(); // a2a_ce
}

} // namespace alps::collectives
