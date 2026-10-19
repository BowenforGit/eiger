// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/containers/fabric_peer_buffer.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/error.hpp"

namespace alps::containers {

namespace {
std::size_t round_up(std::size_t n, std::size_t mult) {
    return mult == 0 ? n : ((n + mult - 1) / mult) * mult;
}

void check_driver(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(
        std::string("FabricPeerBuffer::ensure: ") + operation + " failed: " +
        (message ? message : "unknown CUDA driver error"));
}

void check_runtime(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return;
    throw std::runtime_error(
        std::string("FabricPeerBuffer::ensure: ") + operation + " failed: " +
        cudaGetErrorString(result));
}

bool rank_wide_success(
    const std::exception_ptr& local_error, const Context& ctx) {
    const int local_ok = local_error ? 0 : 1;
    int global_ok = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
    return global_ok != 0;
}

[[noreturn]] void throw_rank_wide_failure(
    const std::exception_ptr& local_error, const char* phase) {
    if (local_error) std::rethrow_exception(local_error);
    throw std::runtime_error(
        std::string("FabricPeerBuffer::ensure: another rank failed during ") +
        phase);
}

bool inject_failure_after_local_export(const Context& ctx) {
    // Deterministic failure injection for the MPI regression test. It runs
    // after a real export so the partial-allocation cleanup path is covered.
    const auto* stage = std::getenv("ALPS_TEST_FABRIC_PEER_BUFFER_FAIL_STAGE");
    return ctx.rank == 0 && stage &&
        std::strcmp(stage, "after_local_export") == 0;
}
} // namespace

FabricPeerBuffer::~FabricPeerBuffer() { teardown_local(); }

FabricPeerBuffer::FabricPeerBuffer(FabricPeerBuffer&& o) noexcept { *this = std::move(o); }

FabricPeerBuffer& FabricPeerBuffer::operator=(FabricPeerBuffer&& o) noexcept {
    if (this == &o) return *this;
    teardown_local(); // release whatever we currently hold
    world_        = o.world_;
    rank_         = o.rank_;
    dev_          = o.dev_;
    gran_         = o.gran_;
    padded_bytes_ = o.padded_bytes_;
    own_handle_   = o.own_handle_;
    own_ptr_      = o.own_ptr_;
    peer_         = std::move(o.peer_);
    imported_     = std::move(o.imported_);
    d_peer_       = std::move(o.d_peer_);
    // Leave the source empty so its destructor is a no-op.
    o.own_handle_   = 0;
    o.own_ptr_      = 0;
    o.padded_bytes_ = 0;
    o.peer_.clear();
    o.imported_.clear();
    return *this;
}

void FabricPeerBuffer::ensure(std::size_t bytes, const Context& ctx) {
    world_ = ctx.size;
    rank_  = ctx.rank;
    dev_   = ctx.local_device_id;

    // Size every rank's buffer to the GLOBAL max so all peer mappings share one
    // size (homogeneous granularity -> identical padded_bytes on every rank).
    unsigned long long want = bytes, max_want = 0;
    ALPS_MPICHECK(MPI_Allreduce(&want, &max_want, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ctx.mpi_comm));
    if (max_want == 0) return; // nothing to back yet

    CUmemAllocationProp prop = {};
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = dev_;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

    if (gran_ == 0)
        ALPS_CUCHECK(cuMemGetAllocationGranularity(&gran_, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    const std::size_t padded_need = round_up(static_cast<std::size_t>(max_want), gran_);
    if (own_handle_ != 0 && padded_need <= padded_bytes_) return; // grow-only: fits

    // Grow: we are about to unmap/release the current backing. Before that, drain
    // ALL in-flight GPU work that may still reference it — this rank's copy engines
    // reading/writing the region, and (paired with the barrier below, after every
    // rank has likewise drained) peers' copies INTO this rank's region. Without the
    // device sync, a regrow triggered by a later/larger column would cuMemUnmap the
    // staging while a prior column's CE pushes / copy-out are still enqueued (UAF).
    // Only needed when there is an existing mapping to tear down.
    if (own_handle_ != 0) ALPS_CUDACHECK(cudaDeviceSynchronize());
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); // no peer still using the old maps
    teardown_local();
    padded_bytes_ = padded_need;

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id   = dev_;                 // THIS rank's device accesses every mapping
    access.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Create + map + export this rank's buffer. Do not let a rank-local
    // allocation failure escape while peers enter the handle exchange.
    CUmemFabricHandle own_fh{};
    std::vector<CUmemFabricHandle> all_fh;
    std::exception_ptr local_allocation_error;
    try {
        check_driver(
            cuMemCreate(&own_handle_, padded_bytes_, &prop, 0),
            "cuMemCreate");
        check_driver(
            cuMemAddressReserve(
                &own_ptr_, padded_bytes_, gran_, 0, 0),
            "cuMemAddressReserve(local)");
        check_driver(
            cuMemMap(own_ptr_, padded_bytes_, 0, own_handle_, 0),
            "cuMemMap(local)");
        check_driver(
            cuMemSetAccess(own_ptr_, padded_bytes_, &access, 1),
            "cuMemSetAccess(local)");
        check_driver(
            cuMemExportToShareableHandle(
                &own_fh, own_handle_, CU_MEM_HANDLE_TYPE_FABRIC, 0),
            "cuMemExportToShareableHandle");
        all_fh.resize(static_cast<std::size_t>(world_));
        if (inject_failure_after_local_export(ctx)) {
            throw std::bad_alloc();
        }
    } catch (...) {
        local_allocation_error = std::current_exception();
    }
    if (!rank_wide_success(local_allocation_error, ctx)) {
        teardown_local();
        padded_bytes_ = 0;
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
        throw_rank_wide_failure(
            local_allocation_error, "local allocation/export");
    }

    // Exchange handles only after every rank has a valid exported handle.
    ALPS_MPICHECK(MPI_Allgather(&own_fh, sizeof(CUmemFabricHandle), MPI_BYTE,
                                all_fh.data(), sizeof(CUmemFabricHandle), MPI_BYTE, ctx.mpi_comm));

    // Import/map peers and construct the device pointer table locally, then
    // coordinate again before returning to a caller that may enter another
    // collective. This also covers rank-local RMM/device allocation failures.
    std::exception_ptr local_mapping_error;
    try {
        peer_.assign(static_cast<std::size_t>(world_), 0);
        imported_.assign(
            static_cast<std::size_t>(world_),
            CUmemGenericAllocationHandle{});
        for (int p = 0; p < world_; ++p) {
            if (p == rank_) {
                peer_[p] = own_ptr_;
                continue;
            }
            check_driver(
                cuMemImportFromShareableHandle(
                    &imported_[p], &all_fh[p],
                    CU_MEM_HANDLE_TYPE_FABRIC),
                "cuMemImportFromShareableHandle");
            CUdeviceptr pptr = 0;
            check_driver(
                cuMemAddressReserve(
                    &pptr, padded_bytes_, gran_, 0, 0),
                "cuMemAddressReserve(peer)");
            // Record the reserved VA immediately so partial-failure cleanup
            // can release it even if map/access setup fails.
            peer_[p] = pptr;
            check_driver(
                cuMemMap(pptr, padded_bytes_, 0, imported_[p], 0),
                "cuMemMap(peer)");
            check_driver(
                cuMemSetAccess(pptr, padded_bytes_, &access, 1),
                "cuMemSetAccess(peer)");
        }

        // Device array of peer base pointers (for kernels that index peers by
        // slot).
        std::vector<unsigned long long> h(static_cast<std::size_t>(world_));
        for (int p = 0; p < world_; ++p) {
            h[p] = static_cast<unsigned long long>(peer_[p]);
        }
        d_peer_ = gamax::device_buffer(
            sizeof(unsigned long long) * static_cast<std::size_t>(world_),
            ctx.stream);
        check_runtime(
            cudaMemcpyAsync(
                d_peer_.data(), h.data(),
                sizeof(unsigned long long) * static_cast<std::size_t>(world_),
                cudaMemcpyHostToDevice, ctx.stream),
            "cudaMemcpyAsync(peer table)");
        check_runtime(
            cudaStreamSynchronize(ctx.stream),
            "cudaStreamSynchronize(peer table)");
    } catch (...) {
        local_mapping_error = std::current_exception();
    }
    if (!rank_wide_success(local_mapping_error, ctx)) {
        teardown_local();
        padded_bytes_ = 0;
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));
        throw_rank_wide_failure(local_mapping_error, "peer mapping");
    }
}

void FabricPeerBuffer::teardown_local() noexcept {
    for (int p = 0; p < world_; ++p) {
        if (p == rank_) continue;
        const auto index = static_cast<std::size_t>(p);
        if (index < peer_.size() && peer_[index] != 0) {
            cuMemUnmap(peer_[index], padded_bytes_);
        }
        if (index < imported_.size() && imported_[index] != 0) {
            cuMemRelease(imported_[index]);
        }
        if (index < peer_.size() && peer_[index] != 0) {
            cuMemAddressFree(peer_[index], padded_bytes_);
        }
    }
    if (own_ptr_) {
        cuMemUnmap(own_ptr_, padded_bytes_);
    }
    if (own_handle_) {
        cuMemRelease(own_handle_);
    }
    if (own_ptr_) {
        cuMemAddressFree(own_ptr_, padded_bytes_);
    }
    d_peer_ = gamax::device_buffer{};
    own_ptr_    = 0;
    own_handle_ = 0;
    peer_.clear();
    imported_.clear();
    // keep gran_/dev_/world_/rank_ for reuse; padded_bytes_ reset on next alloc
}

void FabricPeerBuffer::free(const Context& ctx) {
    ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm)); // no peer still using the maps
    teardown_local();
    padded_bytes_ = 0;
}

} // namespace alps::containers
