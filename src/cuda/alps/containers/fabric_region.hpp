// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// A single contiguous virtual address range whose physical backing is striped,
// block-wise, across every rank's GPU via the CUDA VMM + fabric-handle API
// (cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC) -> export -> MPI_Allgather -> import +
// map). This is the memory primitive under a *shared* hash table: rank p owns
// the VA segment [p*slab, (p+1)*slab), and every rank maps every peer's slab RW
// to its own device, so a kernel on any GPU can touch any byte of the range --
// bytes in another rank's segment traverse NVLink / the MNNVL fabric.
//
// Modeled on collectives/alltoall_p2p.cu's cuMem* idioms, but with the key
// difference that there is ONE contiguous VA reservation with peers mapped into
// adjacent segments (alltoall reserves a separate VA per peer). That single-VA
// layout is what lets cuco::static_map treat the striped storage as one slot
// array addressed by a single base pointer.
//
// USAGE CONTRACT: exactly ONE owning allocation (a static_map's slot storage) is
// live at a time. The region is created ONCE (lazily, on the first acquire so
// cuco drives the size) and reused across many acquire/release cycles -- release
// does NOT free physical memory, it only marks the region free, so a benchmark
// can rebuild the table each iteration over the same fabric backing. free()
// performs the (collective) teardown; call it once, after all maps are destroyed
// and before MPI_Finalize.

#include <cstddef>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "alps/context.hpp"
#include "memory/mem_manager.hpp"

namespace alps::containers {

class FabricStripedRegion {
public:
    explicit FabricStripedRegion(const Context& ctx);
    ~FabricStripedRegion();

    FabricStripedRegion(const FabricStripedRegion&)            = delete;
    FabricStripedRegion& operator=(const FabricStripedRegion&) = delete;

    /// cuco allocator entry point. The FIRST live allocation per cycle is the
    /// owning slot storage and is served from the striped fabric region:
    /// COLLECTIVE on the very first call ever (every rank must call in lockstep --
    /// it creates a region sized to hold >= `bytes`, striped across all ranks),
    /// and a plain reuse of the same base on later cycles. Allocations made WHILE
    /// the owning storage is live are cuco temporaries (size() counters, CUB
    /// scratch -- all small) and go through the process RMM resource. A second
    /// LARGE allocation while a map is live means a second concurrent map over
    /// this region, which is unsupported and THROWS (instead of silently handing
    /// back private, non-shared memory). Returns this rank's mapping of the base.
    void* acquire(std::size_t bytes, cudaStream_t stream);

    /// Release a pointer from acquire(). The owning base only marks the region
    /// free (physical mapping kept for reuse); a temporary returns to RMM.
    void release(void* p, std::size_t bytes, cudaStream_t stream);

    std::size_t capacity_bytes() const noexcept { return total_bytes_; }
    std::size_t slab_bytes()     const noexcept { return slab_bytes_; }
    int         world()          const noexcept { return world_; }

    /// Count of times the local cudaMalloc fallback was taken -- i.e. cuco asked
    /// for a temporary while the fabric region was already live. Lets a caller
    /// observe cuco's private-alloc behavior (see the shared-map design notes).
    long long fallback_allocs() const noexcept { return fallback_allocs_; }

    /// COLLECTIVE teardown (MPI_Barrier + unmap/release/free). Idempotent. Aborts
    /// if a map still owns the storage (tearing down live storage is a UAF).
    void free();

private:
    void* allocate_fabric(std::size_t bytes); // create-or-reuse the owning region
    bool  is_fabric_base(const void* p) const noexcept {
        return base_ != 0 && p == reinterpret_cast<const void*>(base_);
    }
    void  teardown_local() noexcept; // unmap/release/free VA + handles (no MPI)

    const Context& ctx_;
    int            dev_{0};
    int            world_{1};
    int            rank_{0};
    std::size_t    gran_{0};
    std::size_t    slab_bytes_{0};
    std::size_t    total_bytes_{0};
    CUdeviceptr    base_{0};
    bool           created_{false};
    bool           in_use_{false};
    long long      fallback_allocs_{0};
    gamax::rmm_cuco_allocator<std::byte> fallback_allocator_;

    CUmemGenericAllocationHandle              own_handle_{};
    std::vector<CUmemGenericAllocationHandle> imported_; // [world]; own slot unused
};

} // namespace alps::containers
