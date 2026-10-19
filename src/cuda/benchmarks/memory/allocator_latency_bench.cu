// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "memory/device_buffer.hpp"
#include "memory/mem_manager.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/mr/arena_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using any_resource = cuda::mr::any_resource<cuda::mr::device_accessible>;

std::vector<int> allowed_cpus()
{
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        throw std::runtime_error("sched_getaffinity failed in allocator benchmark");
    }
    std::vector<int> result;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) { result.push_back(cpu); }
    }
    if (result.empty()) {
        throw std::runtime_error("allocator benchmark has no allowed host CPUs");
    }
    return result;
}

double run_threads(
    int threads,
    int iterations,
    std::function<void(cudaStream_t)> const& operation)
{
    auto const cpus = allowed_cpus();
    if (static_cast<std::size_t>(threads) > cpus.size()) {
        throw std::runtime_error("allocator benchmark requested more threads than allowed CPUs");
    }
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> failures(static_cast<std::size_t>(threads));
    workers.reserve(threads);
    auto begin = clock_type::time_point{};
    auto end = clock_type::time_point{};

    for (int worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&, worker] {
            cudaStream_t stream{};
            bool announced = false;
            try {
                cpu_set_t affinity;
                CPU_ZERO(&affinity);
                // Spread workers across the allowed CPU set instead of using
                // adjacent cores from one cache/NUMA cluster. The same mapping
                // is used for the wrapper and bare-RMM samples.
                auto const cpu_index =
                    static_cast<std::size_t>(worker) * cpus.size() /
                    static_cast<std::size_t>(threads);
                CPU_SET(cpus[cpu_index], &affinity);
                if (pthread_setaffinity_np(
                        pthread_self(), sizeof(affinity), &affinity) != 0) {
                    throw std::runtime_error(
                        "pthread_setaffinity_np failed in allocator worker");
                }
                if (cudaSetDevice(0) != cudaSuccess) {
                    throw std::runtime_error("cudaSetDevice failed in allocator worker");
                }
                if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
                    throw std::runtime_error("cudaStreamCreate failed in allocator worker");
                }
                ++ready;
                announced = true;
                while (!start.load(std::memory_order_acquire)) { std::this_thread::yield(); }
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    operation(stream);
                }
                if (cudaStreamSynchronize(stream) != cudaSuccess) {
                    throw std::runtime_error("cudaStreamSynchronize failed in allocator worker");
                }
            } catch (...) {
                failures[static_cast<std::size_t>(worker)] = std::current_exception();
                if (!announced) { ++ready; }
            }
            if (stream != nullptr) { (void)cudaStreamDestroy(stream); }
        });
    }
    while (ready.load(std::memory_order_acquire) != threads) { std::this_thread::yield(); }
    begin = clock_type::now();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) { worker.join(); }
    end = clock_type::now();
    for (auto const& failure : failures) {
        if (failure) { std::rethrow_exception(failure); }
    }
    auto const elapsed = std::chrono::duration<double, std::nano>(end - begin).count();
    return elapsed / static_cast<double>(threads * iterations);
}

} // namespace

int main(int argc, char** argv)
try {
    int const iterations = argc > 1 ? std::atoi(argv[1]) : 20'000;
    std::string const output = argc > 2 ? argv[2] : std::string{};
    if (iterations <= 0) { throw std::invalid_argument("iterations must be positive"); }

    if (cudaSetDevice(0) != cudaSuccess) { throw std::runtime_error("cudaSetDevice failed"); }
    gamax::memory_resource_options options;
    options.mode = gamax::memory_resource_mode::arena;
    options.tracking = gamax::memory_tracking_mode::off;
    options.statistics = false;
    options.arena_size_bytes = 256u << 20;
    gamax::initialize_device_memory_resource(0, options);

    rmm::mr::cuda_memory_resource bare_upstream;
    any_resource upstream{bare_upstream};
    rmm::mr::arena_memory_resource bare_arena{upstream, 256u << 20};
    any_resource bare_resource{bare_arena};

    std::ofstream csv;
    if (!output.empty()) {
        csv.open(output);
        if (!csv) { throw std::runtime_error("cannot open output CSV"); }
        csv << "implementation,threads,round,ns_per_allocation\n";
    }

    constexpr int rounds = 10;
    constexpr std::size_t bytes = 4'096;
    for (int threads : {1, 2, 4}) {
        std::vector<double> wrapper_samples;
        std::vector<double> bare_samples;
        auto wrapper_operation = [](cudaStream_t stream) {
            gamax::device_buffer buffer{bytes, stream};
        };
        auto bare_operation = [&](cudaStream_t stream) {
            rmm::device_buffer buffer{
                bytes, rmm::CUDA_ALLOCATION_ALIGNMENT,
                rmm::cuda_stream_view{stream}, bare_resource};
        };
        for (int round = 0; round < rounds; ++round) {
            // Alternate order so CPU frequency warm-up or thermal drift does
            // not consistently favor one implementation.
            double wrapper = 0;
            double bare = 0;
            if ((round & 1) == 0) {
                wrapper = run_threads(threads, iterations, wrapper_operation);
                bare = run_threads(threads, iterations, bare_operation);
            } else {
                bare = run_threads(threads, iterations, bare_operation);
                wrapper = run_threads(threads, iterations, wrapper_operation);
            }
            wrapper_samples.push_back(wrapper);
            bare_samples.push_back(bare);
            if (csv) {
                csv << "eiger," << threads << ',' << round << ',' << wrapper << '\n';
                csv << "bare_rmm," << threads << ',' << round << ',' << bare << '\n';
            }
        }
        std::sort(wrapper_samples.begin(), wrapper_samples.end());
        std::sort(bare_samples.begin(), bare_samples.end());
        auto const wrapper_median =
            (wrapper_samples[rounds / 2 - 1] + wrapper_samples[rounds / 2]) / 2;
        auto const bare_median =
            (bare_samples[rounds / 2 - 1] + bare_samples[rounds / 2]) / 2;
        std::cout << "threads=" << threads
                  << " eiger_median_ns=" << wrapper_median
                  << " bare_rmm_median_ns=" << bare_median
                  << " ratio=" << wrapper_median / bare_median << '\n';
    }

    gamax::shutdown_device_memory_resource(0);
    return 0;
} catch (std::exception const& error) {
    std::cerr << "allocator latency benchmark failed: " << error.what() << '\n';
    return 1;
}
