// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/allgather_p2p.hpp"

#include <cstdint>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "alps/error.hpp"

namespace alps::collectives {

namespace {

using gamax::size_type;
constexpr int kElemBytes = 4; // int32

// PUSH broadcast: every peer slot writes the SAME local send chunk (offset 0)
// into peer p's recv[rank*M]. (src offset 0 is the all-gather difference.)
__global__ void allgather_push_sm(const uint4* __restrict__ local_send,
                                  const unsigned long long* __restrict__ peer_recv,
                                  int rank, int world, size_t m_u4) {
    int slot = blockIdx.y;
    int p = rank + slot; if (p >= world) p -= world;          // (rank + slot) % world
    uint4*       dst = reinterpret_cast<uint4*>(static_cast<uintptr_t>(peer_recv[p]))
                       + static_cast<size_t>(rank) * m_u4;
    const uint4* src = local_send;                            // broadcast: same chunk to every peer
    size_t idx    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < m_u4; i += stride) __stcg(&dst[i], __ldcg(&src[i]));
    __threadfence_system(); // remote stores visible system-wide
}

// PULL gather: every peer slot reads peer p's whole send chunk (offset 0) into
// own recv[p*M].
__global__ void allgather_pull_sm(uint4* __restrict__ local_recv,
                                  const unsigned long long* __restrict__ peer_send,
                                  int rank, int world, size_t m_u4) {
    int slot = blockIdx.y;
    int p = rank + slot; if (p >= world) p -= world;          // (rank + slot) % world
    const uint4* src = reinterpret_cast<const uint4*>(static_cast<uintptr_t>(peer_send[p]));
    uint4*       dst = local_recv + static_cast<size_t>(p) * m_u4;
    size_t idx    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < m_u4; i += stride) __stcg(&dst[i], __ldcg(&src[i]));
    // local recv writes: visible after stream sync; remote ops are reads.
}

} // namespace

void allgather_p2p(const P2pBuffers& b, size_type per_peer_count,
                   P2pCopy copy, P2pDir dir, const Context& ctx, cudaStream_t stream) {
    const int    world = b.world;
    const size_t M     = static_cast<size_t>(per_peer_count);

    if (copy == P2pCopy::SM) {
        const size_t m_u4 = M / 4; // 16B uint4 units
        if (m_u4 == 0) return;
        constexpr int threads = 256;
        int bpp = static_cast<int>((m_u4 + threads - 1) / threads);
        if (bpp < 1)   bpp = 1;
        if (bpp > 256) bpp = 256;
        dim3 grid(static_cast<unsigned>(bpp), static_cast<unsigned>(world));
        if (dir == P2pDir::Push) {
            allgather_push_sm<<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint4*>(static_cast<uintptr_t>(b.send.own())),
                b.recv.d_peers(), b.rank, world, m_u4);
        } else { // Pull
            allgather_pull_sm<<<grid, threads, 0, stream>>>(
                reinterpret_cast<uint4*>(static_cast<uintptr_t>(b.recv.own())),
                b.send.d_peers(), b.rank, world, m_u4);
        }
        return;
    }

    // CE: push only (broadcast local send into every peer's recv[rank*M]).
    // Preserve the same receiver-balanced cyclic order as alltoall_p2p.
    if (dir != P2pDir::Push)
        throw std::runtime_error("allgather_p2p: CE supports push only");
    const size_t mbytes = M * kElemBytes;
    ALPS_CUDACHECK(cudaEventRecord(b.fork_event, stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(b.ce_stream, b.fork_event, 0));
    for (int i = 0; i < world; ++i) {
        int p = b.rank + i; if (p >= world) p -= world;       // (rank + i) % world
        CUdeviceptr dst = b.recv.peer(p) + static_cast<size_t>(b.rank) * mbytes;
        CUdeviceptr src = b.send.own();                        // offset 0 (broadcast)
        ALPS_CUCHECK(cuMemcpyAsync(
            dst, src, mbytes, reinterpret_cast<CUstream>(b.ce_stream)));
    }
    ALPS_CUDACHECK(cudaEventRecord(b.ce_done_event, b.ce_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(stream, b.ce_done_event, 0));
}

} // namespace alps::collectives
