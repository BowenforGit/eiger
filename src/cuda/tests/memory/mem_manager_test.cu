// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <gtest/gtest.h>

#include "memory/device_buffer.hpp"
#include "memory/mem_manager.hpp"
#include "memory/column.hpp"
#include "stats/stats_storage.hpp"
#include "string/string_column.hpp"
#include "containers/map.cuh"
#include "containers/multimap.cuh"
#include "containers/multiset.cuh"
#include "containers/set.cuh"

#include <rmm/error.hpp>
#include <cuco/static_set.cuh>

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

namespace {

__global__ void delayed_store(int* output, unsigned long long cycles)
{
    auto const start = clock64();
    while (clock64() - start < cycles) {}
    *output = 17;
}

gamax::memory_resource_options direct_tracked_options()
{
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::cuda;
    options.tracking = gamax::memory_tracking_mode::on;
    return options;
}

TEST(AllocatorSizing, DefaultUsesAlignedSeventyFivePercent)
{
    auto const sizing = gamax::compute_arena_sizing(10'001, nullptr);
    EXPECT_EQ(sizing.arena_bytes, 7'424);
    EXPECT_EQ(sizing.arena_bytes % 256, 0);
    EXPECT_EQ(sizing.reserved_bytes, 10'001 - 7'424);
}

TEST(AllocatorSizing, PositiveReserveOverridesDefault)
{
    auto const sizing = gamax::compute_arena_sizing(10'000, "1000");
    EXPECT_EQ(sizing.reserved_bytes, 1'000);
    EXPECT_EQ(sizing.arena_bytes, 8'960);
}

TEST(AllocatorSizing, RejectsEveryInvalidReserveForm)
{
    constexpr std::array invalid{
        "", "0", "+1", "-1", " 1", "1 ", "1x", "0x10",
        "10000", "10001", "184467440737095516160000"};
    for (auto const* value : invalid) {
        EXPECT_THROW(gamax::compute_arena_sizing(10'000, value), std::invalid_argument)
            << "value='" << value << "'";
    }
}

TEST(AllocatorOptions, ParsesModesTrackingAndStatisticsStrictly)
{
    setenv("GAMAX_RMM_RESOURCE", "cuda", 1);
    unsetenv("GAMAX_RMM_TRACKING");
    setenv("GAMAX_RMM_STATISTICS", "on", 1);
    auto options = gamax::memory_resource_options_from_environment();
    EXPECT_EQ(options.mode, gamax::memory_resource_mode::cuda);
    EXPECT_EQ(options.tracking, gamax::memory_tracking_mode::on);
    EXPECT_TRUE(options.statistics);

    setenv("GAMAX_RMM_TRACKING", "stacks", 1);
    options = gamax::memory_resource_options_from_environment();
    EXPECT_EQ(options.tracking, gamax::memory_tracking_mode::stacks);

    setenv("GAMAX_RMM_RESOURCE", "invalid", 1);
    EXPECT_THROW(gamax::memory_resource_options_from_environment(), std::invalid_argument);
    unsetenv("GAMAX_RMM_RESOURCE");
    unsetenv("GAMAX_RMM_TRACKING");
    unsetenv("GAMAX_RMM_STATISTICS");
}

// Registered before the other CUDA tests in this executable. cudaSetDevice is
// intentionally the only preceding runtime call; allocator initialization must
// make a cold primary context usable before cudaMemGetInfo/allocation.
TEST(AllocatorResource, ColdContextExplicitPrewarmAndShutdown)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    EXPECT_GT(gamax::current_resource_generation_id(), 0);
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, ConcurrentLazyInitializationProducesOneGeneration)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    setenv("GAMAX_RMM_RESOURCE", "cuda", 1);
    setenv("GAMAX_RMM_TRACKING", "on", 1);

    constexpr int worker_count = 16;
    std::array<std::uint64_t, worker_count> generations{};
    std::array<std::exception_ptr, worker_count> failures{};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            try {
                if (auto error = cudaSetDevice(0); error != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(error));
                }
                auto resource = gamax::get_device_memory_resource();
                generations[worker] = resource.generation_id();
            } catch (...) {
                failures[worker] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) { worker.join(); }
    for (auto const& failure : failures) { EXPECT_EQ(failure, nullptr); }
    for (auto generation : generations) { EXPECT_EQ(generation, generations.front()); }
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);

    gamax::shutdown_device_memory_resource(0);
    unsetenv("GAMAX_RMM_RESOURCE");
    unsetenv("GAMAX_RMM_TRACKING");
}

TEST(AllocatorResource, LiveAllocationAndMovedLeaseRejectReset)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    auto const old_generation = gamax::current_resource_generation_id();

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    {
        gamax::device_buffer allocation{
            4'096, stream, gamax::buffer_init::zero};
        gamax::device_buffer moved{std::move(allocation)};
        EXPECT_EQ(gamax::outstanding_resource_leases(), 1);
        EXPECT_EQ(gamax::outstanding_tracked_allocations(), 1);
        try {
            gamax::reset_device_memory_resource(0);
            FAIL() << "reset unexpectedly accepted a live allocation";
        } catch (std::runtime_error const& error) {
            auto const message = std::string{error.what()};
            EXPECT_NE(message.find("external lease"), std::string::npos);
            EXPECT_NE(message.find("4096"), std::string::npos);
        }
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }

    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);
    gamax::reset_device_memory_resource(0);
    EXPECT_GT(gamax::current_resource_generation_id(), old_generation);
    gamax::shutdown_device_memory_resource(0);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(AllocatorResource, ProjectEmptyBufferDoesNotLeaseGeneration)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    {
        gamax::device_buffer empty{0, cudaStreamDefault, gamax::buffer_init::zero};
        EXPECT_EQ(empty.data(), nullptr);
        EXPECT_EQ(empty.size(), 0);
        EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
        gamax::reset_device_memory_resource(0);
    }
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, EmptyInfrastructureOwnerBlocksReset)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    {
        // The allocator has no allocation, but it owns an RMM resource
        // snapshot and therefore must keep the generation alive.
        gamax::rmm_allocator_handle<int> allocator{cudaStreamDefault};
        EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);
        EXPECT_GT(gamax::outstanding_resource_leases(), 0);
        EXPECT_THROW(gamax::reset_device_memory_resource(0), std::runtime_error);
    }
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, EveryCuCollectionsWrapperRoutesThroughTrackedResource)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        auto keys = gamax::make_column<int>(4, stream);
        gamax::containers::set<int> set{16, 0.5, -1, stream};

        gamax::rmm_allocator_handle<cuco::pair<int, int>> map_allocator{stream};
        gamax::containers::map<int, int> map{
            16, 0.5, -1, -1, map_allocator.get(), stream};

        gamax::rmm_allocator_handle<cuco::pair<int, gamax::size_type>> multimap_allocator{stream};
        gamax::containers::multimap<int, gamax::size_type> multimap{
            16, 0.5, -1, -1, multimap_allocator.get(), stream};

        gamax::rmm_allocator_handle<gamax::size_type> multiset_allocator{stream};
        gamax::containers::multiset<int> multiset{
            keys->view(), 0.5, -1, multiset_allocator.get(), stream};

        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_GE(gamax::outstanding_tracked_allocations(), 5);
        EXPECT_GT(gamax::outstanding_resource_leases(), 0);
        EXPECT_THROW(gamax::reset_device_memory_resource(0), std::runtime_error);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, CuCollectionsDefaultAllocatorIsDetectablyOutsideTracking)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        // Negative control: this is deliberately cuco's default cuda_allocator.
        // It proves why project paths must always provide rmm_cuco_allocator.
        cuco::static_set<int> untracked{
            16, cuco::empty_key{-1}, {}, {}, {}, {}, {}, cuda::stream_ref{stream}};
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);
        EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, StatisticsCacheGpuOwnersAreClearedBeforeReset)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    auto& cache = gamax::stats::StatsStorage::instance();
    cache.clear_all();
    {
        auto source = gamax::make_column<int32_t>(4, cudaStreamDefault);
        auto histogram = std::make_shared<gamax::stats::Histogram>();
        histogram->boundaries = gamax::make_column<int32_t>(3, cudaStreamDefault);
        histogram->counts = gamax::make_column<gamax::size_type>(2, cudaStreamDefault);
        histogram->num_buckets = 2;
        cache.set_stat(
            source->view(), gamax::stats::StatsType::EQUI_WIDTH_HISTOGRAM,
            histogram);
    }
    EXPECT_EQ(cache.num_columns(), 1);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 2);
    gamax::reset_device_memory_resource(0);
    EXPECT_EQ(cache.num_columns(), 0);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, MultiThreadMultiStreamAllocationStress)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::arena;
    options.tracking = gamax::memory_tracking_mode::on;
    options.arena_size_bytes = 64u << 20;
    gamax::initialize_device_memory_resource(0, options);

    constexpr int worker_count = 12;
    constexpr int iterations = 128;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::array<std::exception_ptr, worker_count> failures{};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            cudaStream_t stream{};
            try {
                if (auto error = cudaSetDevice(0); error != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(error));
                }
                if (auto error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                    error != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(error));
                }
                ++ready;
                while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    auto const bytes = static_cast<std::size_t>(
                        256 * (1 + ((worker * 37 + iteration * 17) % 64)));
                    gamax::device_buffer buffer{bytes, stream};
                    auto const value = (worker + iteration) & 0xff;
                    if (auto error = cudaMemsetAsync(buffer.data(), value, bytes, stream);
                        error != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(error));
                    }
                }
                if (auto error = cudaStreamSynchronize(stream); error != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(error));
                }
            } catch (...) {
                failures[worker] = std::current_exception();
            }
            if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
        });
    }
    while (ready.load(std::memory_order_acquire) != worker_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) { worker.join(); }
    for (auto const& failure : failures) { EXPECT_EQ(failure, nullptr); }
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, ConcurrentLeaseDiagnosticsAreRaceFree)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::cuda;
    options.tracking = gamax::memory_tracking_mode::off;
    gamax::initialize_device_memory_resource(0, options);

    constexpr int worker_count = 8;
    constexpr int iterations = 256;
    std::atomic<int> ready{0};
    std::atomic<int> finished{0};
    std::atomic<bool> start{false};
    std::array<std::exception_ptr, worker_count> failures{};
    std::exception_ptr diagnostic_failure;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker] {
            cudaStream_t stream{};
            try {
                if (auto error = cudaSetDevice(0); error != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(error));
                }
                if (auto error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                    error != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(error));
                }
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    auto handle = gamax::get_device_memory_resource();
                    auto copy = handle;
                    gamax::device_buffer buffer{1'024, stream};
                    if (auto error = cudaMemsetAsync(
                            buffer.data(), worker + iteration, buffer.size(), stream);
                        error != cudaSuccess) {
                        throw std::runtime_error(cudaGetErrorString(error));
                    }
                }
                if (auto error = cudaStreamSynchronize(stream); error != cudaSuccess) {
                    throw std::runtime_error(cudaGetErrorString(error));
                }
            } catch (...) {
                failures[worker] = std::current_exception();
            }
            if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
            finished.fetch_add(1, std::memory_order_release);
        });
    }

    std::thread diagnostic([&] {
        try {
            if (auto error = cudaSetDevice(0); error != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(error));
            }
            while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
            do {
                (void)gamax::outstanding_resource_leases();
                auto const text = gamax::outstanding_allocation_diagnostics();
                if (text.find("tracking disabled") == std::string::npos) {
                    throw std::runtime_error("nontracking lease diagnostics changed format");
                }
            } while (finished.load(std::memory_order_acquire) != worker_count);
        } catch (...) {
            diagnostic_failure = std::current_exception();
        }
    });

    while (ready.load(std::memory_order_acquire) != worker_count) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) { worker.join(); }
    diagnostic.join();

    for (auto const& failure : failures) { EXPECT_EQ(failure, nullptr); }
    EXPECT_EQ(diagnostic_failure, nullptr);
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    EXPECT_NE(
        gamax::outstanding_allocation_diagnostics().find("0 external lease(s)"),
        std::string::npos);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, NontrackingDiagnosticsUseLeaseAccounting)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::cuda;
    options.tracking = gamax::memory_tracking_mode::off;
    gamax::initialize_device_memory_resource(0, options);
    {
        gamax::device_buffer buffer{4'096, cudaStreamDefault};
        EXPECT_EQ(gamax::outstanding_resource_leases(), 1);
        EXPECT_NE(
            gamax::outstanding_allocation_diagnostics().find("1 external lease(s)"),
            std::string::npos);
    }
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, ConcurrentStringViewsAreAllocationFree)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    {
        gamax::device_buffer data{64, cudaStreamDefault};
        gamax::device_buffer offsets{9 * sizeof(gamax::size_type), cudaStreamDefault};
        gamax::string_column strings{
            std::move(data), std::move(offsets), 8, gamax::ARROW_STRING};
        auto const allocation_count = gamax::outstanding_tracked_allocations();
        auto const lease_count = gamax::outstanding_resource_leases();
        auto const identity = strings.identity();

        std::array<std::exception_ptr, 16> failures{};
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 16; ++worker) {
            workers.emplace_back([&, worker] {
                try {
                    for (int iteration = 0; iteration < 10'000; ++iteration) {
                        auto view = strings.view();
                        if (view.identity() != identity || view.auxiliary_data() == nullptr) {
                            throw std::runtime_error("string view metadata changed");
                        }
                    }
                } catch (...) {
                    failures[worker] = std::current_exception();
                }
            });
        }
        for (auto& worker : workers) { worker.join(); }
        for (auto const& failure : failures) { EXPECT_EQ(failure, nullptr); }
        EXPECT_EQ(gamax::outstanding_tracked_allocations(), allocation_count);
        EXPECT_EQ(gamax::outstanding_resource_leases(), lease_count);
    }
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, SmallArenaInitializesAndAllocates)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::arena;
    options.tracking = gamax::memory_tracking_mode::on;
    options.arena_size_bytes = 32u << 20;
    gamax::initialize_device_memory_resource(0, options);
    {
        gamax::device_buffer buffer{1u << 20, cudaStreamDefault};
        EXPECT_NE(buffer.data(), nullptr);
    }
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, SmallArenaOutOfMemoryRecoversOnlyAfterDrainAndUnwind)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::arena;
    options.tracking = gamax::memory_tracking_mode::on;
    options.arena_size_bytes = 1u << 20;
    gamax::initialize_device_memory_resource(0, options);
    auto const failed_generation = gamax::current_resource_generation_id();

    bool saw_confirmed_oom = false;
    {
        gamax::device_buffer live_owner{256u << 10, cudaStreamDefault};
        try {
            gamax::device_buffer too_large{2u << 20, cudaStreamDefault};
            (void)too_large;
            FAIL() << "small arena unexpectedly satisfied an oversized request";
        } catch (rmm::out_of_memory const&) {
            saw_confirmed_oom = true;
            // Recovery synchronizes while all failed-cell owners are still
            // alive, then lets normal scope unwinding destroy them.
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            EXPECT_EQ(gamax::outstanding_tracked_allocations(), 1);
        }
    }
    ASSERT_TRUE(saw_confirmed_oom);
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);

    gamax::reset_device_memory_resource(0);
    EXPECT_GT(gamax::current_resource_generation_id(), failed_generation);
    {
        gamax::device_buffer recovered{128u << 10, cudaStreamDefault};
        EXPECT_NE(recovered.data(), nullptr);
    }
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, StackCaptureTrackingProvidesLiveAllocationDiagnostics)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto options = direct_tracked_options();
    options.tracking = gamax::memory_tracking_mode::stacks;
    gamax::initialize_device_memory_resource(0, options);
    {
        gamax::device_buffer buffer{12'288, cudaStreamDefault};
        auto const diagnostics = gamax::outstanding_allocation_diagnostics();
        EXPECT_NE(diagnostics.find("12288"), std::string::npos);
    }
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, AllocationAndFirstUseOnSameStreamAreOrdered)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    std::array<unsigned char, 256> host{};
    {
        gamax::device_buffer buffer{host.size(), stream};
        ASSERT_EQ(cudaMemsetAsync(buffer.data(), 0x5a, buffer.size(), stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
            host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost, stream),
            cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    }
    for (auto value : host) { EXPECT_EQ(value, 0x5a); }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, EventHandoffOrdersFirstUseOnAnotherStream)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t allocation_stream{}, use_stream{};
    cudaEvent_t ready{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&allocation_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&use_stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
    std::array<unsigned char, 512> host{};
    {
        gamax::device_buffer buffer{host.size(), allocation_stream};
        ASSERT_EQ(cudaMemsetAsync(
            buffer.data(), 0x33, buffer.size(), allocation_stream), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(ready, allocation_stream), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(use_stream, ready), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
            host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost, use_stream),
            cudaSuccess);
        buffer.set_stream(use_stream);
        ASSERT_EQ(cudaStreamSynchronize(use_stream), cudaSuccess);
    }
    for (auto value : host) { EXPECT_EQ(value, 0x33); }
    ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(use_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(allocation_stream), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, MultiConsumerEventsJoinBeforeRetirement)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t first{}, second{}, retirement{};
    cudaEvent_t first_done{}, second_done{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&retirement, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&first_done, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&second_done, cudaEventDisableTiming), cudaSuccess);
    std::array<unsigned char, 1'024> host{};
    {
        gamax::device_buffer buffer{host.size(), first};
        auto* bytes = static_cast<unsigned char*>(buffer.data());
        ASSERT_EQ(cudaMemsetAsync(bytes, 0x11, host.size() / 2, first), cudaSuccess);
        ASSERT_EQ(cudaMemsetAsync(
            bytes + host.size() / 2, 0x22, host.size() / 2, second), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(first_done, first), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(second_done, second), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(retirement, first_done), cudaSuccess);
        ASSERT_EQ(cudaStreamWaitEvent(retirement, second_done), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
            host.data(), buffer.data(), buffer.size(), cudaMemcpyDeviceToHost, retirement),
            cudaSuccess);
        buffer.set_stream(retirement);
    }
    ASSERT_EQ(cudaStreamSynchronize(retirement), cudaSuccess);
    for (std::size_t index = 0; index < host.size(); ++index) {
        EXPECT_EQ(host[index], index < host.size() / 2 ? 0x11 : 0x22);
    }
    ASSERT_EQ(cudaEventDestroy(second_done), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(first_done), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(retirement), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(second), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(first), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, ExceptionAfterSubmissionJoinsStreamsBeforeOwnersUnwind)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::arena;
    options.tracking = gamax::memory_tracking_mode::on;
    options.arena_size_bytes = 16u << 20;
    gamax::initialize_device_memory_resource(0, options);

    cudaStream_t first{}, second{}, retirement{};
    cudaEvent_t ready{}, first_done{}, second_done{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&first, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&second, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&retirement, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&first_done, cudaEventDisableTiming), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&second_done, cudaEventDisableTiming), cudaSuccess);

    std::array<unsigned char, 1'024> host{};
    EXPECT_THROW(
        ([&] {
            gamax::device_buffer owner{host.size(), first};
            ASSERT_EQ(cudaEventRecord(ready, first), cudaSuccess);
            ASSERT_EQ(cudaStreamWaitEvent(second, ready), cudaSuccess);
            try {
                auto* bytes = static_cast<unsigned char*>(owner.data());
                ASSERT_EQ(cudaMemsetAsync(bytes, 0x11, host.size() / 2, first), cudaSuccess);
                ASSERT_EQ(cudaMemsetAsync(
                    bytes + host.size() / 2, 0x22, host.size() / 2, second),
                    cudaSuccess);
                throw std::runtime_error("injected failure after GPU submission");
            } catch (...) {
                // Establish the complete join while owner is still alive. Its
                // destructor can then enqueue deallocation after the joined
                // copy on the retirement stream during exception unwinding.
                ASSERT_EQ(cudaEventRecord(first_done, first), cudaSuccess);
                ASSERT_EQ(cudaEventRecord(second_done, second), cudaSuccess);
                ASSERT_EQ(cudaStreamWaitEvent(retirement, first_done), cudaSuccess);
                ASSERT_EQ(cudaStreamWaitEvent(retirement, second_done), cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(
                    host.data(), owner.data(), owner.size(),
                    cudaMemcpyDeviceToHost, retirement), cudaSuccess);
                owner.set_stream(retirement);
                throw;
            }
        }()),
        std::runtime_error);

    ASSERT_EQ(cudaStreamSynchronize(retirement), cudaSuccess);
    for (std::size_t index = 0; index < host.size(); ++index) {
        EXPECT_EQ(host[index], index < host.size() / 2 ? 0x11 : 0x22);
    }
    EXPECT_EQ(gamax::outstanding_resource_leases(), 0);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);

    ASSERT_EQ(cudaEventDestroy(second_done), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(first_done), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(ready), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(retirement), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(second), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(first), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, ConcurrentLiveAllocationsRetainIndependentPatterns)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::arena;
    options.tracking = gamax::memory_tracking_mode::on;
    options.arena_size_bytes = 32u << 20;
    gamax::initialize_device_memory_resource(0, options);

    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    constexpr std::size_t allocation_count = 64;
    constexpr std::size_t bytes = 4'096;
    std::vector<gamax::device_buffer> owners;
    owners.reserve(allocation_count);
    std::vector<std::array<unsigned char, bytes>> host(allocation_count);
    for (std::size_t index = 0; index < allocation_count; ++index) {
        owners.emplace_back(bytes, stream);
        ASSERT_EQ(cudaMemsetAsync(
            owners.back().data(), static_cast<int>(index + 1), bytes, stream), cudaSuccess);
    }
    for (std::size_t index = 0; index < allocation_count; ++index) {
        ASSERT_EQ(cudaMemcpyAsync(
            host[index].data(), owners[index].data(), bytes,
            cudaMemcpyDeviceToHost, stream), cudaSuccess);
    }
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (std::size_t index = 0; index < allocation_count; ++index) {
        for (auto value : host[index]) {
            EXPECT_EQ(value, static_cast<unsigned char>(index + 1));
        }
    }
    owners.clear();
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    EXPECT_EQ(gamax::outstanding_tracked_allocations(), 0);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, SetStreamDoesNotOrderEarlierWork)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    cudaStream_t producer{}, retirement{};
    cudaEvent_t producer_done{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&retirement, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaEventCreateWithFlags(&producer_done, cudaEventDisableTiming), cudaSuccess);
    {
        gamax::device_buffer buffer{sizeof(int), producer};
        delayed_store<<<1, 1, 0, producer>>>(static_cast<int*>(buffer.data()), 50'000'000ULL);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaEventRecord(producer_done, producer), cudaSuccess);
        buffer.set_stream(retirement);
        EXPECT_EQ(buffer.stream(), retirement);
        // Merely changing the deallocation stream does not insert a wait.
        EXPECT_EQ(cudaEventQuery(producer_done), cudaErrorNotReady);
        ASSERT_EQ(cudaStreamSynchronize(producer), cudaSuccess);
    }
    ASSERT_EQ(cudaEventDestroy(producer_done), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(retirement), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(producer), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, OptionalStatisticsReportsScopedPeak)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    auto options = direct_tracked_options();
    options.statistics = true;
    gamax::initialize_device_memory_resource(0, options);

    gamax::scoped_memory_statistics probe;
    {
        gamax::device_buffer buffer{8'192, cudaStreamDefault};
        ASSERT_NE(buffer.data(), nullptr);
    }
    auto const sample = probe.finish();
    EXPECT_EQ(sample.bytes.current, 0);
    EXPECT_GE(sample.bytes.peak, 8'192);
    EXPECT_GE(sample.bytes.total, 8'192);
    EXPECT_EQ(sample.allocations.current, 0);
    EXPECT_GE(sample.allocations.peak, 1);
    EXPECT_GE(sample.allocations.total, 1);
    gamax::shutdown_device_memory_resource(0);
}

TEST(AllocatorResource, WrongCurrentDeviceFailsDeterministically)
{
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count < 2) { GTEST_SKIP() << "requires at least two visible CUDA devices"; }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::initialize_device_memory_resource(0, direct_tracked_options());
    {
        // Seed this host thread's device-buffer TLS fast path on device 0.
        gamax::device_buffer warm{256, cudaStreamDefault};
        ASSERT_NE(warm.data(), nullptr);
    }
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    EXPECT_THROW(
        ([&] { gamax::device_buffer wrong_device{256, cudaStreamDefault}; }()),
        std::runtime_error);
    EXPECT_THROW(
        ([&] { static_cast<void>(gamax::get_device_memory_resource()); }()),
        std::runtime_error);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    gamax::shutdown_device_memory_resource(0);
}

} // namespace
