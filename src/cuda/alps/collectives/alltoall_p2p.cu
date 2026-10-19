// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/alltoall_p2p.hpp"

#include <cstdint>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/error.hpp"

namespace alps::collectives {

namespace {

using gamax::size_type;

constexpr int kElemBytes = 4; // int32

// PUSH: local read + remote store. dst = peer p's recv[rank*M], src = send[p*M].
__global__ void p2p_push_sm(const uint4* __restrict__ local_send,
                            const unsigned long long* __restrict__ peer_recv,
                            int rank, int world, size_t m_u4) {
    int slot = blockIdx.y;
    int p = rank + slot; if (p >= world) p -= world;          // (rank + slot) % world
    uint4*       dst = reinterpret_cast<uint4*>(static_cast<uintptr_t>(peer_recv[p]))
                       + static_cast<size_t>(rank) * m_u4;
    const uint4* src = local_send + static_cast<size_t>(p) * m_u4;
    size_t idx    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < m_u4; i += stride) __stcg(&dst[i], __ldcg(&src[i]));
    __threadfence_system(); // make the remote stores visible system-wide
}

// PULL: remote read + local store. src = peer p's send[rank*M], dst = recv[p*M].
__global__ void p2p_pull_sm(uint4* __restrict__ local_recv,
                            const unsigned long long* __restrict__ peer_send,
                            int rank, int world, size_t m_u4) {
    int slot = blockIdx.y;
    int p = rank + slot; if (p >= world) p -= world;          // (rank + slot) % world
    const uint4* src = reinterpret_cast<const uint4*>(static_cast<uintptr_t>(peer_send[p]))
                       + static_cast<size_t>(rank) * m_u4;
    uint4*       dst = local_recv + static_cast<size_t>(p) * m_u4;
    size_t idx    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < m_u4; i += stride) __stcg(&dst[i], __ldcg(&src[i]));
    // No system fence: writes are to local recv (visible after stream sync); the
    // remote ops are reads, ordered by the caller's pre-op barrier.
}

} // namespace

P2pBuffers p2p_alloc(size_type send_elems, size_type recv_elems, const Context& ctx) {
    P2pBuffers b;
    b.world      = ctx.size;
    b.rank       = ctx.rank;
    b.send_elems = send_elems;
    b.recv_elems = recv_elems;

    // Fabric send + recv, each mapped on every rank (FabricPeerBuffer does the
    // create/export/exchange/import/map dance). Both directions can address them.
    b.send.ensure(static_cast<size_t>(send_elems) * kElemBytes, ctx);
    b.recv.ensure(static_cast<size_t>(recv_elems) * kElemBytes, ctx);

    // One CE stream preserves the cyclic peer order. Multiple independent
    // streams were observed to serialize anyway while allowing CUDA to reorder
    // the transfers into a poorly scaling 20/24-GPU fabric schedule.
    ALPS_CUDACHECK(cudaStreamCreateWithFlags(&b.ce_stream, cudaStreamNonBlocking));
    ALPS_CUDACHECK(cudaEventCreateWithFlags(&b.ce_done_event, cudaEventDisableTiming));
    ALPS_CUDACHECK(cudaEventCreateWithFlags(&b.fork_event, cudaEventDisableTiming));

    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); // everyone mapped before anyone uses
    return b;
}

void alltoall_p2p(const P2pBuffers& b, size_type per_peer_count,
                  P2pCopy copy, P2pDir dir, const Context& ctx, cudaStream_t stream) {
    const int    world = b.world;
    const size_t M     = static_cast<size_t>(per_peer_count);

    if (copy == P2pCopy::SM) {
        const size_t m_u4 = M / 4; // 16B uint4 units; M is a power of two >= 4
        if (m_u4 == 0) return;
        constexpr int threads = 256;
        int bpp = static_cast<int>((m_u4 + threads - 1) / threads);
        if (bpp < 1)   bpp = 1;
        if (bpp > 256) bpp = 256;            // grid-stride covers the rest
        dim3 grid(static_cast<unsigned>(bpp), static_cast<unsigned>(world));
        if (dir == P2pDir::Push) {
            p2p_push_sm<<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(static_cast<uintptr_t>(b.send.own())),
                b.recv.d_peers(), b.rank, world, m_u4);
        } else { // Pull
            p2p_pull_sm<<<grid, threads, 0, stream>>>(
                reinterpret_cast<uint4*>(static_cast<uintptr_t>(b.recv.own())),
                b.send.d_peers(), b.rank, world, m_u4);
        }
        return;
    }

    // CE: push only (write into peer recv). Fork once, enqueue the rotated peer
    // order on one CE stream, then join once. For every slot i, r -> (r+i)%g is
    // a permutation, so all ranks collectively target every receiver exactly
    // once rather than concentrating traffic on one receiver.
    if (dir != P2pDir::Push)
        throw std::runtime_error("alltoall_p2p: CE supports push only");
    const size_t mbytes = M * kElemBytes;
    ALPS_CUDACHECK(cudaEventRecord(b.fork_event, stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(b.ce_stream, b.fork_event, 0));
    for (int i = 0; i < world; ++i) {
        int p = b.rank + i; if (p >= world) p -= world;   // (rank + i) % world
        CUdeviceptr dst = b.recv.peer(p) + static_cast<size_t>(b.rank) * mbytes;
        CUdeviceptr src = b.send.own()   + static_cast<size_t>(p)      * mbytes;
        ALPS_CUCHECK(cuMemcpyAsync(
            dst, src, mbytes, reinterpret_cast<CUstream>(b.ce_stream)));
    }
    ALPS_CUDACHECK(cudaEventRecord(b.ce_done_event, b.ce_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(stream, b.ce_done_event, 0));
}

void p2p_free(P2pBuffers& b, const Context& ctx) {
    b.send.free(ctx); // collective (MPI_Barrier inside)
    b.recv.free(ctx);
    if (b.ce_stream)     cudaStreamDestroy(b.ce_stream);
    if (b.ce_done_event) cudaEventDestroy(b.ce_done_event);
    if (b.fork_event) cudaEventDestroy(b.fork_event);
    b.ce_stream     = nullptr;
    b.ce_done_event = nullptr;
    b.fork_event    = nullptr;
}

} // namespace alps::collectives
