// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/allgather.hpp"

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

void allgather_column_ce_push(gamax::column_view send, const CommPattern& pat,
                              const Context& ctx, cudaStream_t stream) {
    const int    world = ctx.size;
    const size_t esz   = gamax::size_of(send.type());

    if (pat.dst_offsets.empty())
        throw std::runtime_error("allgather_column_ce_push: pat.dst_offsets not populated "
                                 "(call finalize_dst_offsets first)");
    if (!ctx.fabric_scratch)
        throw std::runtime_error("allgather_column_ce_push: ctx.fabric_scratch is null");

    FabricCollectiveScratch& sc = *ctx.fabric_scratch;
    const size_t my_recv_bytes = static_cast<size_t>(pat.total_recv()) * esz;

    sc.recv.ensure(my_recv_bytes, ctx);
    sc.ensure_streams(world, stream);

    // This rank contributes ONE chunk to every peer's gather: [send_offsets[rank],
    // +send_counts[rank]) lands at dst_offsets[p] (== recv_offsets[rank]) in every
    // peer p's recv. (For the standard all-gather send_offsets[rank] == 0.)
    const size_t my_count   = static_cast<size_t>(pat.send_counts[ctx.rank]);
    const char*  sbuf       = send.data<char>();
    CUdeviceptr  src        = as_cuptr(sbuf) + static_cast<size_t>(pat.send_offsets[ctx.rank]) * esz;

    // (1) Broadcast-PUSH our chunk into every peer's recv (self included).
    nvtxRangePushA("ag_ce.push");
    ALPS_CUDACHECK(cudaEventRecord(sc.fork_event, stream));
    if (my_count > 0) {
        for (int i = 0; i < world; ++i) {
            int p = ctx.rank + i; if (p >= world) p -= world;   // (rank + i) % world
            cudaStream_t s = sc.sub_streams[i];
            ALPS_CUDACHECK(cudaStreamWaitEvent(s, sc.fork_event, 0));
            CUdeviceptr dst = sc.recv.peer(p) + static_cast<size_t>(pat.dst_offsets[p]) * esz;
            ALPS_CUCHECK(cuMemcpyAsync(dst, src, my_count * esz, reinterpret_cast<CUstream>(s)));
            ALPS_CUDACHECK(cudaEventRecord(sc.join_events[i], s));
            ALPS_CUDACHECK(cudaStreamWaitEvent(stream, sc.join_events[i], 0));
        }
    }

    nvtxRangePop(); // ag_ce.push

    // (2) Stream-ordered cross-rank completion fence (see alltoall_ce.cu).
    nvtxRangePushA("ag_ce.barrier");
    ALPS_NCCLCHECK(ncclAllReduce(sc.barrier_buf.data(), sc.barrier_buf.data(), 1, ncclInt32, ncclSum,
                                 ctx.nccl_comm, stream));
    nvtxRangePop(); // ag_ce.barrier
}

void allgather_column_ce(gamax::column_view send, gamax::mutable_column_view recv,
                         const CommPattern& pat, const Context& ctx, cudaStream_t stream) {
    nvtxRangePushA("ag_ce");
    // (1)+(2) Broadcast-PUSH + completion (the part NCCL also pays for); leaves the
    //         gathered result in ctx.fabric_scratch->recv.
    allgather_column_ce_push(send, pat, ctx, stream);

    FabricCollectiveScratch& sc = *ctx.fabric_scratch;
    const size_t esz           = gamax::size_of(send.type());
    const size_t my_recv_bytes = static_cast<size_t>(pat.total_recv()) * esz;

    // (3) Copy the gathered recv out into the caller's recv column.
    nvtxRangePushA("ag_ce.copyout");
    if (my_recv_bytes > 0) {
        ALPS_CUDACHECK(cudaMemcpyAsync(recv.data<char>(),
                                       reinterpret_cast<void*>(static_cast<uintptr_t>(sc.recv.own())),
                                       my_recv_bytes, cudaMemcpyDeviceToDevice, stream));
    }
    nvtxRangePop(); // ag_ce.copyout

    // (4) Trailing fence so no peer reuses (PUSHes into) this rank's recv staging
    //     before this rank's copy-out has read it — closes the cross-rank WAR on
    //     the shared staging across columns / back-to-back calls. See alltoall_ce.
    //     Belongs with the copy-out; allgather_column_ce_push omits it.
    nvtxRangePushA("ag_ce.fence");
    ALPS_NCCLCHECK(ncclAllReduce(sc.barrier_buf.data(), sc.barrier_buf.data(), 1, ncclInt32, ncclSum,
                                 ctx.nccl_comm, stream));
    nvtxRangePop(); // ag_ce.fence
    nvtxRangePop(); // ag_ce
}

} // namespace alps::collectives
