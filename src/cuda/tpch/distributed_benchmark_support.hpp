// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <mpi.h>

namespace gamax::tpch::benchmark {

struct WarmupPolicy {
    int minimum_runs {3};
    int window {5};
    double relative_range_threshold {0.05};
    int maximum_runs {20};
};

enum class WarmupStatus {
    Continue,
    Stable,
    MaximumReached,
};

void validate_warmup_policy(const WarmupPolicy& policy);
WarmupStatus evaluate_warmup(
    const std::vector<double>& maximum_rank_samples_ms,
    const WarmupPolicy& policy);

struct SummaryStats {
    double minimum {0.0};
    double mean {0.0};
    double median {0.0};
    double maximum {0.0};
    double stddev {0.0};
};

SummaryStats summarize(const std::vector<double>& samples);

struct RankTimingSample {
    double maximum_ms {0.0};
    std::vector<double> rank_ms;
};

struct ResidualTiming {
    std::vector<double> rank_ms;
    bool negative_beyond_tolerance {false};
};

ResidualTiming derive_residual_timing(
    const std::vector<double>& outer_rank_ms,
    const std::vector<double>& direct_phase_rank_ms,
    double absolute_tolerance_ms = 0.01,
    double relative_tolerance = 0.005);

// The maximum is available on every rank. Raw rank values are populated only
// on root, in MPI rank order.
RankTimingSample collect_rank_timing(
    double local_ms, MPI_Comm communicator, int root = 0);

std::string_view e2e_csv_header();
std::string_view steps_csv_header();
std::string_view phases_csv_header();
std::string_view memory_csv_header();
std::string_view messages_csv_header();
// Missing and empty files need a header. A nonempty file must have exactly the
// expected header; otherwise this throws rather than appending shifted fields.
bool csv_needs_header(
    const std::string& path, std::string_view expected_header);
std::string csv_escape(const std::string& value);
std::uint64_t stable_fingerprint(const std::string& value);
std::string fingerprint_hex(std::uint64_t value);
std::string utc_timestamp();

} // namespace gamax::tpch::benchmark
