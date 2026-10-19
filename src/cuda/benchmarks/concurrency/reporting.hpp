// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "operation.hpp"
#include "timing.hpp"

namespace gamax::benchmark::concurrency {

struct Options {
    std::string op_a{k_expr_name};
    std::string op_b{k_gather_name};
    int a_min{20};
    int a_max{29};
    int b_min{20};
    int b_max{29};
    std::uint64_t a_chunk_size{};
    std::uint64_t b_chunk_size{};
    int warmups{5};
    int min_samples{30};
    int max_samples{100};
    int bootstrap_iterations{2000};
    double ci_threshold{0.02};
    std::uint64_t random_seed{42};
    double gate_timeout_seconds{5.0};
    int device{0};
    std::string output_prefix;
    bool self_test_gate{false};
    bool list_operations{false};

    bool chunking_enabled() const { return a_chunk_size != 0; }
};

Options parse_options(int argc, char** argv);

struct Metadata {
    std::string run_id;
    std::string timestamp;
    std::string git_commit;
    bool git_dirty{};
    std::string gpu_name;
    int device{};
    int driver_version{};
    int runtime_version{};
    int toolkit_version{};
    std::size_t initial_free_bytes{};
    std::size_t total_bytes{};
    int l2_bytes{};
    std::string nvidia_smi_snapshot;
};

Metadata collect_metadata(int device);
void write_metadata(
    std::string const& path, Metadata const& metadata, Options const& options,
    bool authoritative, std::string const& raw_path, std::string const& summary_path);

struct Observation {
    SampleSpec sample;
    int repetition{};
    bool warmup{};
    Timing timing;
};

struct Distribution {
    int n{};
    double median{};
    double mean{};
    double stddev{};
    double p5{};
    double p95{};
};

struct ConfidenceInterval {
    double low{};
    double high{};
};

struct CellSummary {
    Distribution a_alone;
    Distribution b_alone;
    Distribution seq_ab;
    Distribution seq_ba;
    Distribution concurrent_makespan;
    Distribution concurrent_a;
    Distribution concurrent_b;
    Distribution start_skew;
    Distribution chunked_a_alone;
    Distribution chunked_b_alone;
    Distribution chunked_concurrent_makespan;
    Distribution chunked_concurrent_a;
    Distribution chunked_concurrent_b;
    Distribution chunked_start_skew;

    double h_ideal{};
    double h_sequential{};
    double slowdown_a{};
    double slowdown_b{};
    double sequential_ab_ratio{};
    double sequential_ba_ratio{};
    int material_start_skew_count{};
    double max_start_skew_fraction{};
    ConfidenceInterval h_ideal_ci;
    ConfidenceInterval h_sequential_ci;
    ConfidenceInterval slowdown_a_ci;
    ConfidenceInterval slowdown_b_ci;

    double chunked_isolated_slowdown_a{};
    double chunked_isolated_slowdown_b{};
    double chunked_concurrent_speedup{};
    double chunked_interference_slowdown_a{};
    double chunked_interference_slowdown_b{};
    double chunked_concurrent_a_vs_whole{};
    double chunked_concurrent_b_vs_whole{};
    ConfidenceInterval chunked_isolated_slowdown_a_ci;
    ConfidenceInterval chunked_isolated_slowdown_b_ci;
    ConfidenceInterval chunked_concurrent_speedup_ci;
    int chunked_material_start_skew_count{};
    double chunked_max_start_skew_fraction{};

    bool ci_converged{};
    bool stable{};
};

CellSummary make_summary(
    std::vector<Observation> const& observations, Options const& options,
    std::uint64_t cell_seed);

class OutputWriter {
public:
    OutputWriter(
        std::string raw_path, std::string summary_path, Metadata const& metadata,
        OperationDescriptor const& a, OperationDescriptor const& b, Options const& options);
    ~OutputWriter();
    OutputWriter(OutputWriter const&) = delete;
    OutputWriter& operator=(OutputWriter const&) = delete;

    void raw_observation(
        GenericConfig ac, GenericConfig bc, Observation const& observation,
        std::uint64_t seed);
    void raw_failure(
        GenericConfig ac, GenericConfig bc, std::string const& status,
        std::string const& reason, std::uint64_t seed);
    void summary_success(GenericConfig ac, GenericConfig bc, CellSummary const& summary);
    void summary_failure(
        GenericConfig ac, GenericConfig bc, std::string const& status,
        std::string const& reason);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gamax::benchmark::concurrency
