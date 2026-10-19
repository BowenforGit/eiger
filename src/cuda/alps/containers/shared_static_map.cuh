// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

// FabricAllocator + a SharedStaticMap alias: a cuco::static_map whose slot
// storage lives in a FabricStripedRegion (striped across every GPU's memory),
// so all ranks insert/probe one logical table and remote slots are reached over
// NVLink / the MNNVL fabric.
//
// Design (see the shared-map notes): the slots are reached by concurrent
// cross-GPU atomics, so the map uses cuda::thread_scope_system. int32 key +
// int32 value packs into 8 B, so each insert is a single 64-bit .sys CAS.
// linear_probing keeps a key's probe chain on one GPU (block striping), so
// remote traffic per key targets a single peer instead of scattering.

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime_api.h>
#include <cuda/std/atomic>
#include <cuda/std/functional>
#include <cuda/stream_ref>

#include "cuco/static_map.cuh"

#include "alps/containers/fabric_region.hpp"
#include "alps/error.hpp"

namespace alps::containers {

/// cuco-compatible, stream-ordered allocator -- a lightweight handle that
/// delegates all policy to the FabricStripedRegion it points at (cuco copies and
/// rebinds the allocator by value, so no real state may live in the handle). The
/// region routes the owning slot storage to fabric and cuco temporaries to local
/// RMM; see FabricStripedRegion::acquire.
template <typename T>
class FabricAllocator {
public:
    using value_type = T;

    explicit FabricAllocator(FabricStripedRegion* region) noexcept : region_(region) {}

    template <typename U>
    FabricAllocator(FabricAllocator<U> const& other) noexcept : region_(other.region()) {}

    FabricStripedRegion* region() const noexcept { return region_; }

    value_type* allocate(std::size_t n, cuda::stream_ref stream) {
        if (region_ == nullptr)
            throw std::runtime_error("FabricAllocator: null region (default-constructed?)");
        return reinterpret_cast<value_type*>(region_->acquire(n * sizeof(value_type), stream.get()));
    }

    void deallocate(value_type* p, std::size_t n, cuda::stream_ref stream) {
        if (region_ != nullptr) region_->release(p, n * sizeof(value_type), stream.get());
    }

private:
    FabricStripedRegion* region_{nullptr};
};

template <typename T, typename U>
bool operator==(FabricAllocator<T> const& a, FabricAllocator<U> const& b) noexcept {
    return a.region() == b.region();
}
template <typename T, typename U>
bool operator!=(FabricAllocator<T> const& a, FabricAllocator<U> const& b) noexcept {
    return !(a == b);
}

/// The prototype's shared hash table: int32/int32 packed, system scope, linear
/// probing with a caller-chosen cooperative-group size, fabric-backed storage.
template <int CGSize, typename Key = int32_t, typename Value = int32_t>
using SharedStaticMap = cuco::static_map<
    Key,
    Value,
    cuco::extent<std::size_t>,
    cuda::thread_scope_system,
    cuda::std::equal_to<Key>,
    cuco::linear_probing<CGSize, cuco::default_hash_function<Key>>,
    FabricAllocator<cuco::pair<Key, Value>>,
    cuco::storage<1>>;

} // namespace alps::containers
