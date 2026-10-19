// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// A per-rank device buffer over the CUDA VMM + fabric-handle API
// (cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC) -> export -> MPI_Allgather -> import +
// map) that EVERY rank maps RW into its own address space. Unlike
// FabricStripedRegion (one contiguous VA whose segments belong to different
// ranks), here each rank owns ONE buffer and reserves a SEPARATE VA per peer:
// after ensure(), this rank can address peer p's whole buffer via peer(p) and a
// copy engine on this device can read/write it across NVLink / the MNNVL fabric.
//
// This is the per-peer fabric primitive shared by the copy-engine collectives:
//   - the CE all-to-all / all-gather (collectives/fabric_scratch.hpp owns one as
//     a persistent, reused recv staging region), and
//   - the direct-P2P microbench (collectives/alltoall_p2p.cu).
//
// SIZING: ensure(bytes) is COLLECTIVE and sizes every rank's buffer to the
// GLOBAL maximum of `bytes` (MPI_Allreduce MAX), so all ranks' buffers — and
// hence all peer mappings — are the same size. That keeps the import/map step
// simple (one size for every peer) and matches the homogeneous-GPU assumption;
// a caller that only needs its own (smaller) prefix just ignores the tail. It is
// GROW-ONLY: a request that fits the current capacity is a no-op, so a size
// sweep reallocates a few times and then reuses the same backing. free() is the
// (collective) teardown; call it once before MPI_Finalize.

#include <cstddef>
#include <vector>

#include <cuda.h>

#include "alps/context.hpp"
#include "memory/device_buffer.hpp"

namespace alps::containers {

class FabricPeerBuffer {
public:
    FabricPeerBuffer() = default;
    ~FabricPeerBuffer();

    FabricPeerBuffer(const FabricPeerBuffer&)            = delete;
    FabricPeerBuffer& operator=(const FabricPeerBuffer&) = delete;

    // Movable so a factory (e.g. p2p_alloc) can return a struct holding one. The
    // moved-from buffer is left empty (its destructor becomes a no-op).
    FabricPeerBuffer(FabricPeerBuffer&& o) noexcept;
    FabricPeerBuffer& operator=(FabricPeerBuffer&& o) noexcept;

    /// Ensure every rank's buffer holds at least `bytes` (sized to the global
    /// MAX across ranks). COLLECTIVE; grow-only — a no-op when already large
    /// enough. Rank-local allocation/mapping failures are coordinated before
    /// any later collective phase, so every rank exits through an exception.
    /// After it returns, own()/peer()/d_peers() are valid.
    void ensure(std::size_t bytes, const Context& ctx);

    /// COLLECTIVE teardown (MPI_Barrier + unmap/release/free). Idempotent.
    void free(const Context& ctx);

    CUdeviceptr own() const noexcept { return own_ptr_; }           // this rank's base (== peer(rank))
    CUdeviceptr peer(int p) const noexcept { return peer_[p]; }     // peer p's base, mapped here
    /// Device array of `world` peer base pointers (as unsigned long long), for
    /// kernels that index peers by slot. Null until the first ensure().
    const unsigned long long* d_peers() const noexcept {
        return static_cast<const unsigned long long*>(d_peer_.data());
    }

    std::size_t capacity_bytes() const noexcept { return padded_bytes_; }
    bool        allocated()      const noexcept { return own_handle_ != 0; }

private:
    void teardown_local() noexcept; // unmap/release/free VA + handles (no MPI)

    int          world_{0};
    int          rank_{0};
    int          dev_{0};
    std::size_t  gran_{0};
    std::size_t  padded_bytes_{0};                 // per-rank (== global max, gran-rounded)
    CUmemGenericAllocationHandle own_handle_{};     // 0 when unallocated
    CUdeviceptr  own_ptr_{0};
    std::vector<CUdeviceptr>     peer_;             // [world]; own slot == own_ptr_
    std::vector<CUmemGenericAllocationHandle> imported_; // [world]; own slot empty
    gamax::device_buffer         d_peer_{};         // device array of peer_ (ull)
};

} // namespace alps::containers
