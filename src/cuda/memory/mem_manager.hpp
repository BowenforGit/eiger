// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/thrust_allocator_adaptor.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace gamax {

enum class buffer_init { uninitialized, zero };

enum class memory_resource_mode { arena, cuda };
enum class memory_tracking_mode { off, on, stacks };

struct memory_resource_options {
    memory_resource_mode mode{memory_resource_mode::arena};
    memory_tracking_mode tracking{memory_tracking_mode::off};
    bool statistics{false};

    // Test/diagnostic override for a deliberately small arena. Production
    // initialization leaves this empty and derives the size from cudaMemGetInfo.
    std::optional<std::size_t> arena_size_bytes{};
};

struct arena_sizing {
    std::size_t available_bytes{};
    std::size_t reserved_bytes{};
    std::size_t arena_bytes{};
};

// Pure sizing/parser entry point used by the allocator unit tests. A null
// reserve selects the 75% policy. A non-null reserve must be a complete,
// positive, unsigned decimal value smaller than available_bytes.
arena_sizing compute_arena_sizing(std::size_t available_bytes, const char* reserve_value);

memory_resource_options memory_resource_options_from_environment();

namespace detail {
struct resource_generation;
struct lease_slot;
}

class device_buffer;
class rmm_exec_policy_handle;
class rmm_sync_exec_policy_handle;
class scoped_memory_statistics;
template <typename T>
class rmm_cuco_allocator;

// An owning resource snapshot paired with a lease on its resource generation.
// Project code must keep this handle adjacent to every RMM object/container
// constructed from resource().
class resource_handle {
public:
    resource_handle() = default;
    resource_handle(resource_handle const& other) noexcept;
    resource_handle& operator=(resource_handle const& other) noexcept;
    resource_handle(resource_handle&& other) noexcept;
    resource_handle& operator=(resource_handle&& other) noexcept;
    ~resource_handle() noexcept;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] int device_id() const;
    [[nodiscard]] std::uint64_t generation_id() const;
    [[nodiscard]] memory_resource_options const& options() const;

private:
    [[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible> resource() const;
    [[nodiscard]] cuda::mr::any_resource<cuda::mr::device_accessible>
        acquire_for_buffer(int& device_id);
    explicit resource_handle(
        detail::resource_generation* generation, detail::lease_slot* lease) noexcept;
    void acquire() noexcept;
    void release() noexcept;
    detail::resource_generation* generation_{};
    detail::lease_slot* lease_{};
    bool owner_thread_lease_{false};

    friend resource_handle get_device_memory_resource();
    friend class device_buffer;
    friend class rmm_exec_policy_handle;
    friend class rmm_sync_exec_policy_handle;
    friend class scoped_memory_statistics;
    template <typename T>
    friend class rmm_cuco_allocator;
};

[[nodiscard]] resource_handle get_device_memory_resource();

// Short-lived policy owner used at Thrust call sites. The generation lease is
// declared before the policy so it outlives the policy's owning resource.
class rmm_exec_policy_handle {
public:
    explicit rmm_exec_policy_handle(cudaStream_t stream)
        : resource_{get_device_memory_resource()}
        , policy_{rmm::cuda_stream_view{stream}, resource_.resource()}
    {
    }

    [[nodiscard]] rmm::exec_policy_nosync& get() noexcept { return policy_; }

private:
    resource_handle resource_;
    rmm::exec_policy_nosync policy_;
};

class rmm_sync_exec_policy_handle {
public:
    explicit rmm_sync_exec_policy_handle(cudaStream_t stream)
        : resource_{get_device_memory_resource()}
        , policy_{rmm::cuda_stream_view{stream}, resource_.resource()}
    {
    }

    [[nodiscard]] rmm::exec_policy& get() noexcept { return policy_; }

private:
    resource_handle resource_;
    rmm::exec_policy policy_;
};

template <typename T>
class rmm_cuco_allocator {
public:
    using value_type = T;

    template <typename U>
    struct rebind { using other = rmm_cuco_allocator<U>; };

    explicit rmm_cuco_allocator(resource_handle resource)
        : generation_{std::move(resource)}
        , generation_id_{generation_.generation_id()}
        , resource_{generation_.resource()}
    {
    }

    template <typename U>
    rmm_cuco_allocator(rmm_cuco_allocator<U> const& other)
        : generation_{other.generation_}
        , generation_id_{other.generation_id_}
        , resource_{other.resource_}
    {
    }

    [[nodiscard]] T* allocate(std::size_t count, cuda::stream_ref stream)
    {
        return static_cast<T*>(resource_.allocate(stream, count * sizeof(T), alignof(T)));
    }

    void deallocate(T* pointer, std::size_t count, cuda::stream_ref stream) noexcept
    {
        resource_.deallocate(stream, pointer, count * sizeof(T), alignof(T));
    }

    template <typename U>
    [[nodiscard]] bool operator==(rmm_cuco_allocator<U> const& other) const noexcept
    {
        return generation_id_ == other.generation_id_;
    }

private:
    template <typename>
    friend class rmm_cuco_allocator;
    resource_handle generation_;
    std::uint64_t generation_id_{};
    cuda::mr::any_resource<cuda::mr::device_accessible> resource_;
};

template <typename T>
class rmm_allocator_handle {
public:
    explicit rmm_allocator_handle(cudaStream_t)
        : resource_{get_device_memory_resource()}, allocator_{resource_}
    {
    }

    [[nodiscard]] rmm_cuco_allocator<T>& get() noexcept { return allocator_; }

private:
    resource_handle resource_;
    rmm_cuco_allocator<T> allocator_;
};

struct memory_statistics_counter {
    std::int64_t current{};
    std::int64_t peak{};
    std::int64_t total{};
};

struct memory_statistics_sample {
    memory_statistics_counter bytes{};
    memory_statistics_counter allocations{};
};

// Serialized scoped counters for diagnostic/axis probes. Statistics must be
// enabled when the generation is created; scopes cannot overlap across host
// threads because RMM's counters form a single nested stack.
class scoped_memory_statistics {
public:
    scoped_memory_statistics();
    ~scoped_memory_statistics() noexcept;
    scoped_memory_statistics(scoped_memory_statistics const&) = delete;
    scoped_memory_statistics& operator=(scoped_memory_statistics const&) = delete;
    scoped_memory_statistics(scoped_memory_statistics&&) = delete;
    scoped_memory_statistics& operator=(scoped_memory_statistics&&) = delete;

    [[nodiscard]] memory_statistics_sample finish();

private:
    static std::mutex& probe_mutex();

    std::unique_lock<std::mutex> probe_lock_;
    resource_handle generation_;
    bool active_{false};
};

// Prewarm the rank's process-wide resource after selecting its one CUDA device.
// The overload with options is primarily for tests and diagnostic executables.
void initialize_device_memory_resource(int device_id);
void initialize_device_memory_resource(int device_id, memory_resource_options const& options);
void prewarm_device_memory_resource();

// Lazily initializes from the environment on the current device.
// Hard reset is only legal at a quiescent boundary after all generation leases
// have been destroyed: no other thread may allocate, submit work, or acquire a
// resource while reset/shutdown runs. It synchronizes the device again to drain
// deferred frees, validates tracking/lease state, expires the old generation,
// and installs a new generation using either the previous options or the
// supplied replacement.
void reset_device_memory_resource(
    int device_id,
    std::optional<memory_resource_options> replacement_options = std::nullopt);

// Explicit teardown for executables that own process-static GPU caches. No
// replacement generation is created.
void shutdown_device_memory_resource(int device_id);

[[nodiscard]] std::size_t outstanding_resource_leases();
[[nodiscard]] std::size_t outstanding_tracked_allocations();
[[nodiscard]] std::string outstanding_allocation_diagnostics();
[[nodiscard]] std::uint64_t current_resource_generation_id();

} // namespace gamax
