// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "memory/mem_manager.hpp"
#include "stats/stats_storage.hpp"

#include <cuda.h>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <charconv>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace gamax {
namespace {

using any_device_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;
constexpr std::size_t resource_alignment = 256;

[[noreturn]] void throw_cuda_error(cudaError_t error, std::string_view operation)
{
    throw std::runtime_error(
        std::string(operation) + " failed: " + cudaGetErrorName(error) + " (" +
        cudaGetErrorString(error) + ")");
}

void check_cuda(cudaError_t error, std::string_view operation)
{
    if (error != cudaSuccess) { throw_cuda_error(error, operation); }
}

int current_device()
{
    int device = -1;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    return device;
}

int current_context_device()
{
    CUdevice device = -1;
    auto const error = cuCtxGetDevice(&device);
    if (error != CUDA_SUCCESS) {
        char const* name = nullptr;
        char const* description = nullptr;
        (void)cuGetErrorName(error, &name);
        (void)cuGetErrorString(error, &description);
        throw std::runtime_error(
            std::string("cuCtxGetDevice failed: ") +
            (name ? name : "unknown driver error") + " (" +
            (description ? description : "no description") + ")");
    }
    return static_cast<int>(device);
}

void require_current_device(int expected, std::string_view boundary)
{
    auto const actual = current_device();
    if (actual != expected) {
        throw std::runtime_error(
            std::string(boundary) + ": CUDA device mismatch; rank device is " +
            std::to_string(expected) + ", current thread selected device " +
            std::to_string(actual));
    }
}

std::size_t align_down(std::size_t value)
{
    return value & ~(resource_alignment - 1);
}

bool parse_switch(char const* value, char const* name)
{
    if (value == nullptr || std::string_view{value} == "off") { return false; }
    if (std::string_view{value} == "on") { return true; }
    throw std::invalid_argument(std::string(name) + " must be 'off' or 'on'");
}

enum class context_state { uninitialized, initializing, ready, resetting, failed };

struct process_resource_context {
    std::mutex mutex;
    std::condition_variable changed;
    context_state state{context_state::uninitialized};
    int device_id{-1};
    std::uint64_t next_generation_id{1};
    std::shared_ptr<detail::resource_generation> generation{};
    memory_resource_options options{};
    std::exception_ptr failure{};
    // Zero is unavailable, an odd value is resetting, and an even nonzero
    // value identifies a ready generation. This permits a thread-local weak
    // cache without allowing a lease to race past reset validation.
    std::atomic<std::uint64_t> access_epoch{0};
};

// Intentionally process-lifetime storage. Resource ownership is explicitly
// removed by shutdown/reset; the context object's destructor never calls CUDA.
process_resource_context& context()
{
    static auto* value = new process_resource_context;
    return *value;
}

void initialize_cuda_context()
{
    // cudaSetDevice/cudaFree(nullptr) need not instantiate a cold primary
    // context. A tiny checked allocation/free reliably initializes it before
    // cudaMemGetInfo is used for arena sizing.
    void* probe = nullptr;
    check_cuda(cudaMalloc(&probe, 1), "cold-context cudaMalloc");
    check_cuda(cudaFree(probe), "cold-context cudaFree");
}

std::shared_ptr<detail::resource_generation> make_generation(
    int device_id,
    std::uint64_t generation_id,
    memory_resource_options const& requested_options);

void install_generation(std::shared_ptr<detail::resource_generation> const& generation);

} // namespace

namespace detail {

struct alignas(64) lease_slot {
    std::thread::id owner{std::this_thread::get_id()};
    // Only `owner` writes owner_count, so its hot path needs an atomic
    // load/store rather than a locked read-modify-write. Handles used on other
    // threads update the separate signed adjustment.
    std::atomic<std::size_t> owner_count{0};
    std::atomic<std::ptrdiff_t> cross_thread_adjustment{0};
};

struct resource_generation {
    resource_generation(std::uint64_t id, int device, memory_resource_options snapshot)
        : generation_id{id}, device_id{device}, options{std::move(snapshot)}
    {
    }

    std::uint64_t generation_id;
    int device_id;
    memory_resource_options options;
    mutable std::mutex lease_mutex;
    std::vector<std::unique_ptr<lease_slot>> lease_slots;
    arena_sizing sizing{};

    // Declaration order is deliberate: destruction runs from outer_resource
    // back toward the CUDA upstream.
    std::optional<any_device_resource> upstream_resource;
    std::optional<rmm::mr::arena_memory_resource> arena_resource;
    std::optional<rmm::mr::tracking_resource_adaptor> tracking_resource;
    std::optional<rmm::mr::statistics_resource_adaptor> statistics_resource;
    std::optional<any_device_resource> outer_resource;
};

} // namespace detail

namespace {

struct thread_resource_cache {
    std::uint64_t epoch{};
    detail::resource_generation* generation{};
    detail::lease_slot* lease{};
};

thread_resource_cache& local_resource_cache()
{
    thread_local thread_resource_cache cache;
    return cache;
}

std::uint64_t ready_epoch(std::uint64_t generation_id) { return generation_id << 1; }

detail::lease_slot* make_lease_slot(detail::resource_generation& generation)
{
    auto slot = std::make_unique<detail::lease_slot>();
    auto* result = slot.get();
    std::lock_guard lock{generation.lease_mutex};
    generation.lease_slots.push_back(std::move(slot));
    return result;
}

std::size_t lease_count(detail::resource_generation const& generation)
{
    std::lock_guard lock{generation.lease_mutex};
    std::ptrdiff_t result = 0;
    for (auto const& slot : generation.lease_slots) {
        // Read the cross-thread publication first. An acquire that observes a
        // transferred token also observes the owner's preceding count store.
        auto const adjustment =
            slot->cross_thread_adjustment.load(std::memory_order_acquire);
        auto const owned = slot->owner_count.load(std::memory_order_acquire);
        result += static_cast<std::ptrdiff_t>(owned) + adjustment;
    }
    if (result < 0) { throw std::logic_error("negative allocator generation lease count"); }
    return static_cast<std::size_t>(result);
}

} // namespace

arena_sizing compute_arena_sizing(std::size_t available_bytes, const char* reserve_value)
{
    if (available_bytes < resource_alignment) {
        throw std::invalid_argument("available device memory is too small for an RMM arena");
    }

    std::size_t arena_bytes = 0;
    std::size_t reserve_bytes = 0;

    if (reserve_value == nullptr) {
        // Compute floor(available * 3 / 4) without overflowing size_t.
        arena_bytes = (available_bytes / 4) * 3 + ((available_bytes % 4) * 3) / 4;
        arena_bytes = align_down(arena_bytes);
        reserve_bytes = available_bytes - arena_bytes;
    } else {
        std::string_view text{reserve_value};
        if (text.empty()) {
            throw std::invalid_argument("GAMAX_MEM_RESERVE_BYTES must not be empty");
        }
        for (char c : text) {
            if (c < '0' || c > '9') {
                throw std::invalid_argument(
                    "GAMAX_MEM_RESERVE_BYTES must be an unsigned decimal integer");
            }
        }

        auto const* begin = text.data();
        auto const* end = begin + text.size();
        auto result = std::from_chars(begin, end, reserve_bytes, 10);
        if (result.ec == std::errc::result_out_of_range) {
            throw std::invalid_argument("GAMAX_MEM_RESERVE_BYTES overflows size_t");
        }
        if (result.ec != std::errc{} || result.ptr != end || reserve_bytes == 0) {
            throw std::invalid_argument(
                "GAMAX_MEM_RESERVE_BYTES must be a positive unsigned decimal integer");
        }
        if (reserve_bytes >= available_bytes) {
            throw std::invalid_argument(
                "GAMAX_MEM_RESERVE_BYTES must be smaller than available device memory");
        }
        arena_bytes = align_down(available_bytes - reserve_bytes);
    }

    if (arena_bytes == 0) {
        throw std::invalid_argument("computed RMM arena size is zero after alignment");
    }
    return {available_bytes, reserve_bytes, arena_bytes};
}

memory_resource_options memory_resource_options_from_environment()
{
    memory_resource_options options;
    if (auto const* resource = std::getenv("GAMAX_RMM_RESOURCE")) {
        if (std::string_view{resource} == "arena") {
            options.mode = memory_resource_mode::arena;
        } else if (std::string_view{resource} == "cuda") {
            options.mode = memory_resource_mode::cuda;
        } else {
            throw std::invalid_argument("GAMAX_RMM_RESOURCE must be 'arena' or 'cuda'");
        }
    }

    auto const* tracking = std::getenv("GAMAX_RMM_TRACKING");
    if (tracking == nullptr) {
        options.tracking = options.mode == memory_resource_mode::cuda
            ? memory_tracking_mode::on
            : memory_tracking_mode::off;
    } else if (std::string_view{tracking} == "off") {
        options.tracking = memory_tracking_mode::off;
    } else if (std::string_view{tracking} == "on") {
        options.tracking = memory_tracking_mode::on;
    } else if (std::string_view{tracking} == "stacks") {
        options.tracking = memory_tracking_mode::stacks;
    } else {
        throw std::invalid_argument("GAMAX_RMM_TRACKING must be 'off', 'on', or 'stacks'");
    }

    options.statistics = parse_switch(
        std::getenv("GAMAX_RMM_STATISTICS"), "GAMAX_RMM_STATISTICS");
    return options;
}

namespace {

std::shared_ptr<detail::resource_generation> make_generation(
    int device_id,
    std::uint64_t generation_id,
    memory_resource_options const& requested_options)
{
    require_current_device(device_id, "allocator initialization");
    initialize_cuda_context();

    auto generation = std::make_shared<detail::resource_generation>(
        generation_id, device_id, requested_options);
    generation->upstream_resource.emplace(rmm::mr::cuda_memory_resource{});
    any_device_resource outer = *generation->upstream_resource;

    if (requested_options.mode == memory_resource_mode::arena) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");

        if (requested_options.arena_size_bytes) {
            auto const size = align_down(*requested_options.arena_size_bytes);
            if (size == 0 || size >= free_bytes) {
                throw std::invalid_argument(
                    "explicit RMM arena size must be positive and smaller than available memory");
            }
            generation->sizing = {free_bytes, free_bytes - size, size};
        } else {
            generation->sizing = compute_arena_sizing(
                free_bytes, std::getenv("GAMAX_MEM_RESERVE_BYTES"));
        }

        generation->arena_resource.emplace(outer, generation->sizing.arena_bytes);
        outer = any_device_resource{*generation->arena_resource};
    } else if (requested_options.arena_size_bytes) {
        throw std::invalid_argument("arena_size_bytes is invalid for the direct CUDA resource");
    }

    if (requested_options.tracking != memory_tracking_mode::off) {
        generation->tracking_resource.emplace(
            outer, requested_options.tracking == memory_tracking_mode::stacks);
        outer = any_device_resource{*generation->tracking_resource};
    }

    if (requested_options.statistics) {
        generation->statistics_resource.emplace(outer);
        outer = any_device_resource{*generation->statistics_resource};
    }
    generation->outer_resource.emplace(std::move(outer));

    std::clog << "GAMAX RMM generation " << generation_id
              << ": device=" << device_id
              << " mode=" << (requested_options.mode == memory_resource_mode::arena ? "arena" : "cuda")
              << " tracking="
              << (requested_options.tracking == memory_tracking_mode::off
                      ? "off"
                      : requested_options.tracking == memory_tracking_mode::on ? "on" : "stacks")
              << " statistics=" << (requested_options.statistics ? "on" : "off");
    if (requested_options.mode == memory_resource_mode::arena) {
        std::clog << " available_bytes=" << generation->sizing.available_bytes
                  << " reserved_bytes=" << generation->sizing.reserved_bytes
                  << " arena_bytes=" << generation->sizing.arena_bytes;
    }
    std::clog << '\n';
    return generation;
}

void install_generation(std::shared_ptr<detail::resource_generation> const& generation)
{
    require_current_device(generation->device_id, "allocator installation");
    auto old = rmm::mr::set_per_device_resource(
        rmm::cuda_device_id{generation->device_id}, *generation->outer_resource);
    // The only supported predecessor is RMM's initial cuda_memory_resource or
    // a generation removed by our validated reset path. Releasing this owning
    // value immediately avoids retaining an old resource generation.
    old = any_device_resource{rmm::mr::cuda_memory_resource{}};
}

void initialize_impl(int device_id, memory_resource_options const& options)
{
    require_current_device(device_id, "allocator initialization");
    auto& ctx = context();
    std::unique_lock lock{ctx.mutex};

    while (ctx.state == context_state::initializing || ctx.state == context_state::resetting) {
        ctx.changed.wait(lock);
    }
    if (ctx.device_id != -1 && ctx.device_id != device_id) {
        throw std::runtime_error(
            "the process memory resource is already bound to CUDA device " +
            std::to_string(ctx.device_id));
    }
    if (ctx.state == context_state::ready) {
        if (ctx.options.mode != options.mode || ctx.options.tracking != options.tracking ||
            ctx.options.statistics != options.statistics ||
            ctx.options.arena_size_bytes != options.arena_size_bytes) {
            throw std::runtime_error(
                "memory-resource options cannot change while a generation is installed; use hard reset");
        }
        return;
    }
    if (ctx.state == context_state::failed && ctx.failure) {
        std::rethrow_exception(ctx.failure);
    }

    ctx.device_id = device_id;
    ctx.state = context_state::initializing;
    auto const generation_id = ctx.next_generation_id++;
    lock.unlock();

    try {
        auto generation = make_generation(device_id, generation_id, options);
        install_generation(generation);
        lock.lock();
        ctx.options = options;
        ctx.generation = std::move(generation);
        ctx.failure = {};
        ctx.state = context_state::ready;
        ctx.access_epoch.store(
            ready_epoch(ctx.generation->generation_id), std::memory_order_release);
        lock.unlock();
        ctx.changed.notify_all();
    } catch (...) {
        lock.lock();
        ctx.failure = std::current_exception();
        ctx.state = context_state::failed;
        ctx.access_epoch.store(0, std::memory_order_release);
        lock.unlock();
        ctx.changed.notify_all();
        throw;
    }
}

} // namespace

resource_handle::resource_handle(
    detail::resource_generation* generation, detail::lease_slot* lease) noexcept
    : generation_{generation}, lease_{lease}
{
    if (lease_) {
        owner_thread_lease_ = true;
        auto const count = lease_->owner_count.load(std::memory_order_relaxed);
        lease_->owner_count.store(count + 1, std::memory_order_relaxed);
    }
}

resource_handle::resource_handle(resource_handle const& other) noexcept
    : generation_{other.generation_}, lease_{other.lease_}
{
    acquire();
}

resource_handle& resource_handle::operator=(resource_handle const& other) noexcept
{
    if (this != &other) {
        release();
        generation_ = other.generation_;
        lease_ = other.lease_;
        acquire();
    }
    return *this;
}

resource_handle::resource_handle(resource_handle&& other) noexcept
    : generation_{std::exchange(other.generation_, nullptr)}
    , lease_{std::exchange(other.lease_, nullptr)}
    , owner_thread_lease_{std::exchange(other.owner_thread_lease_, false)}
{
}

resource_handle& resource_handle::operator=(resource_handle&& other) noexcept
{
    if (this != &other) {
        release();
        generation_ = std::exchange(other.generation_, nullptr);
        lease_ = std::exchange(other.lease_, nullptr);
        owner_thread_lease_ = std::exchange(other.owner_thread_lease_, false);
    }
    return *this;
}

resource_handle::~resource_handle() noexcept { release(); }

void resource_handle::acquire() noexcept
{
    if (!lease_) { return; }
    owner_thread_lease_ = std::this_thread::get_id() == lease_->owner;
    if (owner_thread_lease_) {
        auto const count = lease_->owner_count.load(std::memory_order_relaxed);
            lease_->owner_count.store(count + 1, std::memory_order_relaxed);
    } else {
        lease_->cross_thread_adjustment.fetch_add(1, std::memory_order_acq_rel);
    }
}

void resource_handle::release() noexcept
{
    if (lease_) {
        if (owner_thread_lease_ && std::this_thread::get_id() == lease_->owner) {
            auto const count = lease_->owner_count.load(std::memory_order_relaxed);
            assert(count != 0 && "allocator generation lease underflow");
            lease_->owner_count.store(count - 1, std::memory_order_relaxed);
        } else {
            lease_->cross_thread_adjustment.fetch_sub(1, std::memory_order_acq_rel);
        }
        lease_ = nullptr;
        owner_thread_lease_ = false;
    }
    if (generation_) {
        generation_ = nullptr;
    }
}

resource_handle::operator bool() const noexcept { return generation_ != nullptr; }

int resource_handle::device_id() const
{
    if (!generation_) { throw std::logic_error("empty resource_handle"); }
    return generation_->device_id;
}

std::uint64_t resource_handle::generation_id() const
{
    if (!generation_) { throw std::logic_error("empty resource_handle"); }
    return generation_->generation_id;
}

memory_resource_options const& resource_handle::options() const
{
    if (!generation_) { throw std::logic_error("empty resource_handle"); }
    return generation_->options;
}

any_device_resource resource_handle::resource() const
{
    if (!generation_) { throw std::logic_error("empty resource_handle"); }
    return *generation_->outer_resource;
}

void initialize_device_memory_resource(int device_id)
{
    initialize_impl(device_id, memory_resource_options_from_environment());
}

void initialize_device_memory_resource(int device_id, memory_resource_options const& options)
{
    initialize_impl(device_id, options);
}

void prewarm_device_memory_resource()
{
    initialize_device_memory_resource(current_device());
}

resource_handle get_device_memory_resource()
{
    auto const device = current_device();
    auto& ctx = context();

    // Hot path: a raw TLS cache plus the generation epoch avoids taking the
    // process-context mutex on every small allocation. Hard reset is only
    // legal at the documented quiescent boundary and publishes an odd epoch
    // before validating the exact atomic lease count.
    auto& cache = local_resource_cache();
    auto const before = ctx.access_epoch.load(std::memory_order_acquire);
    if (before != 0 && (before & 1u) == 0 && cache.epoch == before &&
        cache.generation && cache.lease) {
        auto* generation = cache.generation;
        auto const after = ctx.access_epoch.load(std::memory_order_acquire);
        if (before == after) {
            if (generation->device_id != device) {
                throw std::runtime_error(
                    "allocator access on CUDA device " + std::to_string(device) +
                    " but this process is bound to device " +
                    std::to_string(generation->device_id));
            }
            return resource_handle{generation, cache.lease};
        }
    }

    for (;;) {
        std::unique_lock lock{ctx.mutex};
        while (ctx.state == context_state::initializing || ctx.state == context_state::resetting) {
            ctx.changed.wait(lock);
        }
        if (ctx.device_id != -1 && ctx.device_id != device) {
            throw std::runtime_error(
                "allocator access on CUDA device " + std::to_string(device) +
                " but this process is bound to device " + std::to_string(ctx.device_id));
        }
        if (ctx.state == context_state::ready) {
            auto* generation = ctx.generation.get();
            cache.epoch = ready_epoch(generation->generation_id);
            cache.generation = generation;
            cache.lease = make_lease_slot(*generation);
            return resource_handle{generation, cache.lease};
        }
        if (ctx.state == context_state::failed && ctx.failure) {
            std::rethrow_exception(ctx.failure);
        }
        lock.unlock();
        initialize_device_memory_resource(device);
        // Reacquire through the normal state-machine path. A concurrent hard
        // shutdown may legally begin immediately after initialization; never
        // return a null generation from that race window.
    }
}

any_device_resource resource_handle::acquire_for_buffer(int& device_id)
{
    if (generation_ != nullptr || lease_ != nullptr) {
        throw std::logic_error("device_buffer resource installation requires an empty handle");
    }

    auto& ctx = context();
    auto& cache = local_resource_cache();
    auto const epoch = ctx.access_epoch.load(std::memory_order_acquire);
    if (epoch != 0 && (epoch & 1u) == 0 && cache.epoch == epoch &&
        cache.generation && cache.lease) {
        // cudaSetDevice is thread-local, so every allocation boundary must
        // query the active context's device even when the generation cache is
        // warm. The driver query avoids the heavier runtime API hot path.
        auto const device = current_context_device();
        if (cache.generation->device_id != device) {
            throw std::runtime_error(
                "allocator access on CUDA device " +
                std::to_string(device) +
                " but this process is bound to device " +
                std::to_string(cache.generation->device_id));
        }
        // device_buffer passes a fresh empty member. Install its generation
        // lease in place to avoid constructing and move-assigning a temporary
        // on every allocation.
        generation_ = cache.generation;
        lease_ = cache.lease;
        owner_thread_lease_ = true;
        auto const count = lease_->owner_count.load(std::memory_order_relaxed);
        lease_->owner_count.store(count + 1, std::memory_order_relaxed);
        device_id = cache.generation->device_id;
        return *cache.generation->outer_resource;
    }
    // Cold thread, initialization, or a concurrent reset: use the fully
    // validating state-machine path and seed this thread's weak cache.
    auto validated = get_device_memory_resource();
    device_id = validated.device_id();
    auto resource = validated.resource();
    *this = std::move(validated);
    return resource;
}

namespace {

void remove_generation(int device_id, bool create_replacement,
                       std::optional<memory_resource_options> replacement)
{
    require_current_device(device_id, create_replacement ? "allocator reset" : "allocator shutdown");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before allocator reset");

    // Statistics can own histogram/sketch columns. They are a process cache,
    // not cell-owned state, so hard reset must explicitly destroy them after
    // draining work and before validating generation leases.
    stats::StatsStorage::instance().clear_all();
    check_cuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize after allocator cache teardown");

    auto& ctx = context();
    std::unique_lock lock{ctx.mutex};
    while (ctx.state == context_state::initializing || ctx.state == context_state::resetting) {
        ctx.changed.wait(lock);
    }
    if (ctx.device_id == -1 || ctx.state == context_state::uninitialized) {
        if (create_replacement) {
            lock.unlock();
            initialize_impl(device_id, replacement.value_or(memory_resource_options_from_environment()));
        }
        return;
    }
    if (ctx.device_id != device_id) {
        throw std::runtime_error("allocator reset requested for a device other than the rank device");
    }

    if (ctx.generation) {
        auto const generation_epoch = ready_epoch(ctx.generation->generation_id);
        ctx.state = context_state::resetting;
        ctx.access_epoch.store(generation_epoch | 1u, std::memory_order_release);
        auto const leases = lease_count(*ctx.generation);
        auto const tracked = ctx.generation->tracking_resource
            ? ctx.generation->tracking_resource->get_outstanding_allocations().size()
            : 0;
        if (leases != 0 || tracked != 0) {
            std::string message =
                "allocator reset rejected: generation " +
                std::to_string(ctx.generation->generation_id) + " still has " +
                std::to_string(leases) + " external lease(s)";
            if (ctx.generation->tracking_resource) {
                message += "; outstanding allocations:\n" +
                    ctx.generation->tracking_resource->get_outstanding_allocations_str();
            } else {
                message += "; tracking disabled, pointer-level diagnostics unavailable";
            }
            ctx.state = context_state::ready;
            ctx.access_epoch.store(generation_epoch, std::memory_order_release);
            lock.unlock();
            ctx.changed.notify_all();
            throw std::runtime_error(std::move(message));
        }
    }

    auto next_options = replacement.value_or(ctx.options);
    if (!ctx.generation) {
        ctx.state = context_state::resetting;
        ctx.access_epoch.store(1, std::memory_order_release);
    }
    auto old_generation = std::move(ctx.generation);
    std::weak_ptr<detail::resource_generation> old_weak = old_generation;
    lock.unlock();

    try {
        auto old_map_resource = rmm::mr::reset_per_device_resource(
            rmm::cuda_device_id{device_id});
        old_map_resource = any_device_resource{rmm::mr::cuda_memory_resource{}};
        old_generation.reset();
        if (!old_weak.expired()) {
            throw std::runtime_error("old allocator generation did not expire during hard reset");
        }

        std::shared_ptr<detail::resource_generation> next_generation;
        if (create_replacement) {
            std::uint64_t id;
            lock.lock();
            id = ctx.next_generation_id++;
            lock.unlock();
            next_generation = make_generation(device_id, id, next_options);
            install_generation(next_generation);
        }

        lock.lock();
        ctx.generation = std::move(next_generation);
        ctx.options = next_options;
        ctx.failure = {};
        ctx.state = create_replacement ? context_state::ready : context_state::uninitialized;
        ctx.access_epoch.store(
            create_replacement ? ready_epoch(ctx.generation->generation_id) : 0,
            std::memory_order_release);
        if (!create_replacement) { ctx.device_id = -1; }
        lock.unlock();
        ctx.changed.notify_all();
    } catch (...) {
        lock.lock();
        ctx.failure = std::current_exception();
        ctx.state = context_state::failed;
        ctx.access_epoch.store(0, std::memory_order_release);
        lock.unlock();
        ctx.changed.notify_all();
        throw;
    }
}

} // namespace

void reset_device_memory_resource(
    int device_id, std::optional<memory_resource_options> replacement_options)
{
    remove_generation(device_id, true, std::move(replacement_options));
}

void shutdown_device_memory_resource(int device_id)
{
    remove_generation(device_id, false, std::nullopt);
}

std::size_t outstanding_resource_leases()
{
    auto& ctx = context();
    std::lock_guard lock{ctx.mutex};
    return ctx.generation
        ? lease_count(*ctx.generation)
        : 0;
}

std::size_t outstanding_tracked_allocations()
{
    auto& ctx = context();
    std::lock_guard lock{ctx.mutex};
    if (!ctx.generation || !ctx.generation->tracking_resource) { return 0; }
    return ctx.generation->tracking_resource->get_outstanding_allocations().size();
}

std::string outstanding_allocation_diagnostics()
{
    auto& ctx = context();
    std::lock_guard lock{ctx.mutex};
    if (!ctx.generation) { return "allocator is not initialized"; }
    if (!ctx.generation->tracking_resource) {
        return "tracking disabled; generation " +
            std::to_string(ctx.generation->generation_id) + " has " +
            std::to_string(lease_count(*ctx.generation)) + " external lease(s)";
    }
    return ctx.generation->tracking_resource->get_outstanding_allocations_str();
}

std::uint64_t current_resource_generation_id()
{
    auto& ctx = context();
    std::lock_guard lock{ctx.mutex};
    if (!ctx.generation) { throw std::logic_error("allocator is not initialized"); }
    return ctx.generation->generation_id;
}

std::mutex& scoped_memory_statistics::probe_mutex()
{
    static auto* mutex = new std::mutex;
    return *mutex;
}

scoped_memory_statistics::scoped_memory_statistics()
    : probe_lock_{probe_mutex()}, generation_{get_device_memory_resource()}
{
    auto const& generation = generation_.generation_;
    if (!generation->statistics_resource) {
        throw std::logic_error(
            "scoped memory statistics require GAMAX_RMM_STATISTICS=on at generation initialization");
    }
    (void)generation->statistics_resource->push_counters();
    active_ = true;
}

scoped_memory_statistics::~scoped_memory_statistics() noexcept
{
    if (active_) {
        try {
            (void)generation_.generation_->statistics_resource->pop_counters();
        } catch (...) {
            std::terminate();
        }
    }
}

memory_statistics_sample scoped_memory_statistics::finish()
{
    if (!active_) { throw std::logic_error("memory statistics scope already finished"); }
    auto const [bytes, allocations] =
        generation_.generation_->statistics_resource->pop_counters();
    active_ = false;
    generation_ = {};
    probe_lock_.unlock();
    return {{bytes.value, bytes.peak, bytes.total},
            {allocations.value, allocations.peak, allocations.total}};
}

} // namespace gamax
