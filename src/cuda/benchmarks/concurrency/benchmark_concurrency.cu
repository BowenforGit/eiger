// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "memory/mem_manager.hpp"
#include "operation.hpp"
#include "reporting.hpp"
#include "timing.hpp"

namespace fs = std::filesystem;

namespace gamax::benchmark::concurrency {
namespace {

void reset_for_sample(
    SampleSpec sample, PreparedOperation& a, PreparedOperation& b, cudaStream_t setup_stream)
{
    switch (sample.mode) {
        case Mode::a_alone: a.reset(setup_stream); break;
        case Mode::b_alone: b.reset(setup_stream); break;
        case Mode::sequential_ab:
        case Mode::sequential_ba:
        case Mode::concurrent:
            a.reset(setup_stream);
            b.reset(setup_stream);
            break;
    }
    cuda_check(cudaStreamSynchronize(setup_stream), "operation reset synchronization");
}

void validate_result(
    PreparedOperation& operation, gamax::column_ptr& result, cudaStream_t stream)
{
    operation.validate(result, stream);
    result.reset();
    cuda_check(cudaStreamSynchronize(stream), "validation result retirement");
}

void prime_chunked(
    PreparedOperation& operation, std::uint64_t chunk_size, cudaStream_t stream)
{
    operation.reset(stream);
    cuda_check(cudaStreamSynchronize(stream), "chunked prime reset");
    auto result = operation.enqueue_chunked(chunk_size, stream);
    cuda_check(cudaStreamSynchronize(stream), "chunked prime execution");
    operation.validate(result, stream);
    result.reset();
    operation.retire_completed();
    cuda_check(cudaStreamSynchronize(stream), "chunked prime retirement");
}

struct CellRunResult {
    std::vector<Observation> observations;
    CellSummary summary;
};

CellRunResult run_cell(
    GenericConfig ac, GenericConfig bc, OperationDescriptor const& desc_a,
    OperationDescriptor const& desc_b, TimingCore& timing, L2Purger& purger,
    Options const& options, OutputWriter& writer, std::uint64_t cell_seed)
{
    Stream setup;
    auto a = desc_a.prepare(ac, setup.value);
    auto b = desc_b.prepare(bc, setup.value);
    cuda_check(cudaStreamSynchronize(setup.value), "cell preparation synchronization");

    auto [isolated_a, isolated_b] = timing.prime_whole(*a, *b);
    validate_result(*a, isolated_a, setup.value);
    validate_result(*b, isolated_b, setup.value);

    if (options.chunking_enabled()) {
        prime_chunked(*a, options.a_chunk_size, setup.value);
        prime_chunked(*b, options.b_chunk_size, setup.value);
    }

    auto validate_concurrent = [&](ExecutionVariant variant) {
        SampleSpec sample{Mode::concurrent, variant};
        reset_for_sample(sample, *a, *b, setup.value);
        purger.purge();
        auto result = timing.run(sample, *a, *b, true,
                                 options.a_chunk_size, options.b_chunk_size);
        a->validate(result.a_result, setup.value);
        b->validate(result.b_result, setup.value);
        result.a_result.reset();
        result.b_result.reset();
        cuda_check(cudaDeviceSynchronize(), "concurrent validation retirement");
    };
    validate_concurrent(ExecutionVariant::whole);
    if (options.chunking_enabled()) { validate_concurrent(ExecutionVariant::chunked); }

    std::mt19937_64 sample_rng(cell_seed);
    std::vector<Observation> observations;
    int whole_concurrent_repetition = 0;
    int chunked_concurrent_repetition = 0;

    auto run_round = [&](int repetition, bool warmup) {
        auto samples = all_samples(options.chunking_enabled());
        std::shuffle(samples.begin(), samples.end(), sample_rng);
        for (SampleSpec sample : samples) {
            reset_for_sample(sample, *a, *b, setup.value);
            purger.purge();
            bool a_first = true;
            if (sample.mode == Mode::concurrent) {
                int& counter = sample.variant == ExecutionVariant::whole
                    ? whole_concurrent_repetition : chunked_concurrent_repetition;
                a_first = (counter++ % 2) == 0;
            }
            auto sample_timing = [&] {
                ScopedSampleNvtxRange sample_range(sample);
                return timing.run(sample, *a, *b, a_first,
                                  options.a_chunk_size, options.b_chunk_size);
            }();
            Observation observation{sample, repetition, warmup, std::move(sample_timing)};
            observation.timing.a_result.reset();
            observation.timing.b_result.reset();
            writer.raw_observation(ac, bc, observation, cell_seed);
            observations.push_back(std::move(observation));
        }
    };

    for (int warmup = 0; warmup < options.warmups; ++warmup) { run_round(warmup, true); }

    CellSummary summary;
    for (int repetition = 0; repetition < options.max_samples; ++repetition) {
        run_round(repetition, false);
        int const count = repetition + 1;
        if (count >= options.min_samples &&
            (count == options.max_samples || count % 5 == 0)) {
            summary = make_summary(observations, options, cell_seed);
            if (summary.ci_converged) { break; }
        }
    }
    summary = make_summary(observations, options, cell_seed);
    return {std::move(observations), summary};
}

void reset_after_oom(int device, L2Purger& purger)
{
    purger.destroy();
    cuda_check(cudaDeviceSynchronize(), "OOM quiescence synchronization");
    if (gamax::outstanding_resource_leases() != 0) {
        fail("recoverable OOM left outstanding allocator generation leases");
    }
    if (gamax::outstanding_tracked_allocations() != 0) {
        fail("recoverable OOM left tracked allocations: " +
             gamax::outstanding_allocation_diagnostics());
    }
    gamax::reset_device_memory_resource(device);
    purger.recreate();
}

bool is_authoritative(Options const& options)
{
    bool const multiset_only =
        (options.op_a == k_multiset_insert_name || options.op_a == k_multiset_retrieve_name) &&
        (options.op_b == k_multiset_insert_name || options.op_b == k_multiset_retrieve_name);
    int const maximum = multiset_only ? 28 : 29;
    return options.a_min == 20 && options.a_max == maximum &&
           options.b_min == 20 && options.b_max == maximum && options.warmups >= 5 &&
           options.min_samples >= 30 && options.max_samples >= 100 &&
           options.ci_threshold <= 0.02;
}

} // namespace

int benchmark_main(int argc, char** argv)
{
    Options const options = parse_options(argc, argv);
    OperationRegistry registry;
    if (options.list_operations) { registry.list(std::cout); return 0; }
    if (options.self_test_gate) {
        return gate_self_test(options.device, options.gate_timeout_seconds);
    }

    cuda_check(cudaSetDevice(options.device), "cudaSetDevice");
    Metadata const metadata = collect_metadata(options.device);
    auto const& desc_a = registry.lookup(options.op_a);
    auto const& desc_b = registry.lookup(options.op_b);
    for (int exponent = options.a_min; exponent <= options.a_max; ++exponent) {
        desc_a.validate_config({exponent});
    }
    for (int exponent = options.b_min; exponent <= options.b_max; ++exponent) {
        desc_b.validate_config({exponent});
    }

    fs::path prefix(options.output_prefix);
    if (prefix.has_parent_path()) { fs::create_directories(prefix.parent_path()); }
    std::string const raw_path = options.output_prefix + "_raw.csv";
    std::string const summary_path = options.output_prefix + "_summary.csv";
    std::string const metadata_path = options.output_prefix + "_metadata.json";
    write_metadata(metadata_path, metadata, options, is_authoritative(options),
                   raw_path, summary_path);

    std::cout << "run_id=" << metadata.run_id << " gpu=" << metadata.gpu_name
              << " commit=" << metadata.git_commit
              << (metadata.git_dirty ? " (dirty)" : "") << '\n';
    std::cout << "device snapshot: " << metadata.nvidia_smi_snapshot << '\n';
    if (metadata.nvidia_smi_snapshot.find(" 0 %") == std::string::npos &&
        metadata.nvidia_smi_snapshot.find(", 0 %") == std::string::npos) {
        std::cerr << "warning: device may be busy; inspect metadata snapshot\n";
    }

    gamax::prewarm_device_memory_resource();
    int clock_rate_khz = 0;
    cuda_check(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, options.device),
               "query device clock rate");
    TimingCore timing(clock_rate_khz, options.gate_timeout_seconds);
    L2Purger purger(options.device);
    OutputWriter writer(raw_path, summary_path, metadata, desc_a, desc_b, options);

    std::size_t const preflight_limit = static_cast<std::size_t>(
        static_cast<long double>(metadata.initial_free_bytes) * 0.75L * 0.90L);
    std::size_t completed = 0;
    std::size_t const total = static_cast<std::size_t>(options.a_max - options.a_min + 1) *
                              static_cast<std::size_t>(options.b_max - options.b_min + 1);

    for (int a_exp = options.a_min; a_exp <= options.a_max; ++a_exp) {
        for (int b_exp = options.b_min; b_exp <= options.b_max; ++b_exp) {
            GenericConfig ac{a_exp};
            GenericConfig bc{b_exp};
            std::uint64_t const cell_seed = options.random_seed ^
                (static_cast<std::uint64_t>(a_exp) << 32) ^ static_cast<std::uint64_t>(b_exp);
            std::size_t const estimate = desc_a.estimate_memory(ac) +
                                         desc_b.estimate_memory(bc) + purger.bytes();
            std::cout << '[' << (completed + 1) << '/' << total << "] A=2^" << a_exp
                      << " B=2^" << b_exp << " estimate="
                      << (estimate / (1024.0 * 1024.0 * 1024.0)) << " GiB";
            if (options.chunking_enabled()) {
                std::cout << " chunks="
                          << effective_chunk_size(ac.size(), options.a_chunk_size) << '/'
                          << effective_chunk_size(bc.size(), options.b_chunk_size);
            }
            std::cout << '\n';

            if (estimate > preflight_limit) {
                std::string const reason =
                    "estimated cell memory exceeds conservative arena capacity";
                writer.raw_failure(ac, bc, "skipped_insufficient_memory", reason, cell_seed);
                writer.summary_failure(ac, bc, "skipped_insufficient_memory", reason);
                ++completed;
                continue;
            }

            try {
                auto result = run_cell(ac, bc, desc_a, desc_b, timing, purger,
                                       options, writer, cell_seed);
                writer.summary_success(ac, bc, result.summary);
                std::cout << "  H_ideal=" << result.summary.h_ideal
                          << " H_sequential=" << result.summary.h_sequential
                          << " S_A=" << result.summary.slowdown_a
                          << " S_B=" << result.summary.slowdown_b;
                if (options.chunking_enabled()) {
                    std::cout << " chunk_iso_A=" << result.summary.chunked_isolated_slowdown_a
                              << " chunk_iso_B=" << result.summary.chunked_isolated_slowdown_b
                              << " chunk_speedup=" << result.summary.chunked_concurrent_speedup;
                }
                std::cout << " samples=" << result.summary.a_alone.n
                          << (result.summary.stable ? " stable" : " unstable") << '\n';
            } catch (std::bad_alloc const& error) {
                std::string const reason =
                    std::string("recoverable allocator OOM: ") + error.what();
                writer.raw_failure(ac, bc, "failed_oom", reason, cell_seed);
                writer.summary_failure(ac, bc, "failed_oom", reason);
                reset_after_oom(options.device, purger);
            }
            ++completed;
        }
    }

    cuda_check(cudaDeviceSynchronize(), "final device synchronization");
    std::cout << "wrote " << raw_path << '\n'
              << "wrote " << summary_path << '\n'
              << "wrote " << metadata_path << '\n';
    return 0;
}

} // namespace gamax::benchmark::concurrency

int main(int argc, char** argv)
{
    try {
        return gamax::benchmark::concurrency::benchmark_main(argc, argv);
    } catch (std::exception const& error) {
        std::cerr << "benchmark_concurrency: " << error.what() << '\n';
        return 1;
    }
}
