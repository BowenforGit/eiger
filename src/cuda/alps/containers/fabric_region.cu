// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/containers/fabric_region.hpp"

#include <cstdio>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/error.hpp"

namespace alps::containers {

namespace {
std::size_t round_up(std::size_t n, std::size_t mult) {
    return ((n + mult - 1) / mult) * mult;
}
} // namespace

FabricStripedRegion::FabricStripedRegion(const Context& ctx)
    : ctx_(ctx)
    , dev_(ctx.local_device_id)
    , world_(ctx.size)
    , rank_(ctx.rank)
    , fallback_allocator_(gamax::get_device_memory_resource())
{
}

FabricStripedRegion::~FabricStripedRegion() {
    // Local-only cleanup as a safety net. The collective teardown (free()) must
    // be called explicitly before MPI_Finalize; running an MPI_Barrier from a
    // destructor risks deadlock / use-after-finalize, so we do not barrier here.
    if (!created_) return;
    if (in_use_)
        std::fprintf(stderr,
            "[rank %d] FabricStripedRegion destroyed while a map still owns the storage "
            "(use-after-free risk); destroy the map and call free() first\n", rank_);
    teardown_local();
}

void* FabricStripedRegion::acquire(std::size_t bytes, cudaStream_t stream) {
    // First live allocation per cycle == the static_map's owning slot storage.
    if (!in_use_) return allocate_fabric(bytes);

    // The owning storage is already live, so this is a cuco temporary (size()
    // counter, CUB scratch -- all far smaller than one VMM granularity). A request
    // larger than the granularity while a map is live can only be a second
    // concurrent map's slot storage, which this region does not support; reject it
    // loudly instead of silently returning private (non-shared) memory.
    if (bytes > gran_)
        throw std::runtime_error(
            "FabricStripedRegion: second concurrent large allocation -- only one shared "
            "map per region is supported (build and destroy maps serially)");

    void* p = fallback_allocator_.allocate(bytes, cuda::stream_ref{stream});
    ++fallback_allocs_;
    return p;
}

void FabricStripedRegion::release(void* p, std::size_t bytes, cudaStream_t stream) {
    if (is_fabric_base(p)) {
        in_use_ = false; // keep the physical mapping for the next cycle
        return;
    }
    fallback_allocator_.deallocate(
        static_cast<std::byte*>(p), bytes, cuda::stream_ref{stream});
}

void* FabricStripedRegion::allocate_fabric(std::size_t bytes) {
    if (!created_) {
        if (bytes == 0) throw std::runtime_error("FabricStripedRegion: zero-byte allocation");

        CUmemAllocationProp prop = {};
        prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id          = dev_;
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

        ALPS_CUCHECK(cuMemGetAllocationGranularity(&gran_, &prop,
                                                   CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

        // Block striping: one slab per rank, each a multiple of the granularity so
        // segment boundaries never split a VMM page (and, since slab is a multiple
        // of 2 MiB and the slot/bucket size is a small power of two, never split a
        // hash-table bucket either).
        const std::size_t per = (bytes + static_cast<std::size_t>(world_) - 1) / world_;
        slab_bytes_  = round_up(per, gran_);
        total_bytes_ = slab_bytes_ * static_cast<std::size_t>(world_);

        // COLLECTIVE free-HBM check: every rank reports whether its slab fits, and
        // all ranks agree via MPI_Allreduce. Aborting only the failing rank here
        // (before the cuMemCreate/MPI_Allgather collective below) would deadlock
        // the survivors; instead all ranks throw together. (Inputs on
        // The RMM arena reserve must leave this much headroom; see the design notes.)
        size_t freeb = 0, totb = 0;
        ALPS_CUDACHECK(cudaMemGetInfo(&freeb, &totb));
        int local_oom = (freeb < slab_bytes_) ? 1 : 0, global_oom = 0;
        ALPS_MPICHECK(MPI_Allreduce(&local_oom, &global_oom, 1, MPI_INT, MPI_MAX, ctx_.mpi_comm));
        if (global_oom) {
            if (local_oom)
                std::fprintf(stderr,
                    "[rank %d] FabricStripedRegion: need %.2f GiB free for the slab but only "
                    "%.2f GiB available (total %.2f GiB). Lower the input footprint (cudaMM / "
                    "GAMAX_MEM_RESERVE_BYTES).\n",
                    rank_, slab_bytes_ / (1024.0 * 1024.0 * 1024.0),
                    freeb / (1024.0 * 1024.0 * 1024.0), totb / (1024.0 * 1024.0 * 1024.0));
            throw std::runtime_error(
                "FabricStripedRegion: insufficient free HBM for the striped slab on >=1 rank");
        }

        CUmemAccessDesc access = {};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id   = dev_; // this rank's device reads/writes every segment
        access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // Create this rank's slab and export its fabric handle.
        ALPS_CUCHECK(cuMemCreate(&own_handle_, slab_bytes_, &prop, 0));
        CUmemFabricHandle own_fh;
        ALPS_CUCHECK(cuMemExportToShareableHandle(&own_fh, own_handle_,
                                                  CU_MEM_HANDLE_TYPE_FABRIC, 0));

        // Exchange every rank's handle.
        std::vector<CUmemFabricHandle> all_fh(world_);
        ALPS_MPICHECK(MPI_Allgather(&own_fh, sizeof(CUmemFabricHandle), MPI_BYTE,
                                    all_fh.data(), sizeof(CUmemFabricHandle), MPI_BYTE,
                                    ctx_.mpi_comm));

        // Reserve ONE contiguous VA covering all segments, then map peer p's slab
        // into segment p (self uses its own handle), RW to this rank's device.
        ALPS_CUCHECK(cuMemAddressReserve(&base_, total_bytes_, gran_, 0, 0));
        imported_.assign(world_, CUmemGenericAllocationHandle{});
        for (int p = 0; p < world_; ++p) {
            CUdeviceptr                  seg = base_ + static_cast<CUdeviceptr>(p) * slab_bytes_;
            CUmemGenericAllocationHandle h;
            if (p == rank_) {
                h = own_handle_;
            } else {
                ALPS_CUCHECK(cuMemImportFromShareableHandle(&h, &all_fh[p],
                                                            CU_MEM_HANDLE_TYPE_FABRIC));
                imported_[p] = h;
            }
            ALPS_CUCHECK(cuMemMap(seg, slab_bytes_, 0, h, 0));
            ALPS_CUCHECK(cuMemSetAccess(seg, slab_bytes_, &access, 1));
        }

        created_ = true;
        ALPS_MPICHECK(MPI_Barrier(ctx_.mpi_comm)); // everyone mapped before any use
    } else if (bytes > total_bytes_) {
        throw std::runtime_error("FabricStripedRegion: request exceeds the created region");
    }

    in_use_ = true;
    return reinterpret_cast<void*>(base_);
}

void FabricStripedRegion::teardown_local() noexcept {
    // Best-effort local teardown (no MPI, no aborting): cleanup errors are not
    // actionable and the destructor must not abort during stack unwinding.
    for (int p = 0; p < world_; ++p)
        cuMemUnmap(base_ + static_cast<CUdeviceptr>(p) * slab_bytes_, slab_bytes_);
    for (int p = 0; p < world_; ++p)
        if (p != rank_ && imported_[p]) cuMemRelease(imported_[p]);
    cuMemRelease(own_handle_);
    cuMemAddressFree(base_, total_bytes_);
    base_    = 0;
    created_ = false;
    in_use_  = false;
    imported_.clear();
}

void FabricStripedRegion::free() {
    if (!created_) return;
    if (in_use_) {
        std::fprintf(stderr,
            "[rank %d] FabricStripedRegion::free() called while a map still owns the storage; "
            "destroy the map first\n", rank_);
        std::abort();
    }
    ALPS_MPICHECK(MPI_Barrier(ctx_.mpi_comm)); // no peer still touching the maps
    teardown_local();
}

} // namespace alps::containers
