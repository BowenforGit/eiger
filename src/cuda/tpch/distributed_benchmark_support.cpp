// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "tpch/distributed_benchmark_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace gamax::tpch::benchmark {
namespace {

double median_of_sorted(const std::vector<double>& sorted) {
    if (sorted.empty()) return 0.0;
    const auto middle = sorted.size() / 2;
    if (sorted.size() % 2 == 1) return sorted[middle];
    return (sorted[middle - 1] + sorted[middle]) / 2.0;
}

} // namespace

void validate_warmup_policy(const WarmupPolicy& policy) {
    if (policy.minimum_runs < 0) {
        throw std::invalid_argument("warmup minimum must be nonnegative");
    }
    if (policy.window < 1) {
        throw std::invalid_argument("warmup window must be positive");
    }
    if (!(policy.relative_range_threshold >= 0.0)) {
        throw std::invalid_argument("warmup threshold must be nonnegative");
    }
    if (policy.maximum_runs < policy.minimum_runs ||
        policy.maximum_runs < policy.window) {
        throw std::invalid_argument(
            "warmup maximum must cover the minimum and stability window");
    }
}

WarmupStatus evaluate_warmup(
    const std::vector<double>& samples,
    const WarmupPolicy& policy)
{
    validate_warmup_policy(policy);
    const auto required = static_cast<std::size_t>(
        std::max(policy.minimum_runs, policy.window));
    if (samples.size() >= required) {
        const auto begin = samples.end() - policy.window;
        std::vector<double> window(begin, samples.end());
        std::sort(window.begin(), window.end());
        const double median = median_of_sorted(window);
        const double relative_range = median == 0.0
            ? (window.back() == window.front() ? 0.0 : INFINITY)
            : (window.back() - window.front()) / std::abs(median);
        if (relative_range <= policy.relative_range_threshold) {
            return WarmupStatus::Stable;
        }
    }
    return samples.size() >= static_cast<std::size_t>(policy.maximum_runs)
        ? WarmupStatus::MaximumReached
        : WarmupStatus::Continue;
}

SummaryStats summarize(const std::vector<double>& samples) {
    SummaryStats result;
    if (samples.empty()) return result;
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    result.minimum = sorted.front();
    result.maximum = sorted.back();
    result.median = median_of_sorted(sorted);
    result.mean = std::accumulate(
        samples.begin(), samples.end(), 0.0) /
        static_cast<double>(samples.size());
    double sum_sq = 0.0;
    for (const auto sample : samples) {
        const double delta = sample - result.mean;
        sum_sq += delta * delta;
    }
    result.stddev = std::sqrt(sum_sq / static_cast<double>(samples.size()));
    return result;
}

ResidualTiming derive_residual_timing(
    const std::vector<double>& outer_rank_ms,
    const std::vector<double>& direct_phase_rank_ms,
    double absolute_tolerance_ms,
    double relative_tolerance)
{
    if (outer_rank_ms.size() != direct_phase_rank_ms.size()) {
        throw std::invalid_argument(
            "outer and direct phase rank vectors must have equal size");
    }
    if (absolute_tolerance_ms < 0.0 || relative_tolerance < 0.0) {
        throw std::invalid_argument("residual tolerances must be nonnegative");
    }
    ResidualTiming result;
    result.rank_ms.resize(outer_rank_ms.size());
    for (std::size_t rank = 0; rank < outer_rank_ms.size(); ++rank) {
        const double raw = outer_rank_ms[rank] - direct_phase_rank_ms[rank];
        const double tolerance = std::max(
            absolute_tolerance_ms,
            std::abs(outer_rank_ms[rank]) * relative_tolerance);
        if (raw < -tolerance) result.negative_beyond_tolerance = true;
        result.rank_ms[rank] = std::max(0.0, raw);
    }
    return result;
}

RankTimingSample collect_rank_timing(
    double local_ms, MPI_Comm communicator, int root)
{
    int rank = 0;
    int world = 0;
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &world);
    if (root < 0 || root >= world) {
        throw std::invalid_argument("timing root is outside the communicator");
    }

    RankTimingSample result;
    MPI_Allreduce(
        &local_ms, &result.maximum_ms, 1, MPI_DOUBLE, MPI_MAX,
        communicator);
    if (rank == root) {
        result.rank_ms.resize(static_cast<std::size_t>(world));
    }
    MPI_Gather(
        &local_ms, 1, MPI_DOUBLE,
        result.rank_ms.empty() ? nullptr : result.rank_ms.data(),
        1, MPI_DOUBLE, root, communicator);
    return result;
}

std::string_view e2e_csv_header() {
    return
        "timestamp_utc,query,scale,ranks,nodes,iteration,warmup_runs,"
        "warmup_status,warmup_min,warmup_window,warmup_threshold,warmup_max,"
        "rank_min_ms,rank_mean_ms,rank_max_ms,rank_samples_ms,"
        "plan_fingerprint,git_commit,hostname,gpu_name,rmm_resource,"
        "reserve_bytes,physical_plan_preset,physical_plan_config";
}

std::string_view steps_csv_header() {
    return
        "timestamp_utc,query,scale,ranks,nodes,iteration,warmup_runs,"
        "record_type,rank,step_index,step_kind,step_label,"
        "requested_implementation,used_implementation,fallback_reason,gpu_ms,"
        "rank_min_ms,rank_mean_ms,rank_max_ms,plan_fingerprint,"
        "git_commit,hostname,gpu_name,rmm_resource,reserve_bytes,"
        "physical_plan_preset,physical_plan_config";
}

std::string_view phases_csv_header() {
    return
        "timestamp_utc,query,scale,ranks,nodes,iteration,warmup_runs,"
        "record_type,rank,step_index,step_kind,step_label,phase_index,"
        "phase_name,column_side,column_ordinal,category,"
        "communication_pattern,collective_primitive,"
        "gpu_ms,start_ms,end_ms,bytes_sent,bytes_received,rows_sent,"
        "rows_received,peer_count,alignment_main_rows,alignment_pre_rows,"
        "alignment_post_rows,alignment_main_bytes,alignment_pre_bytes,"
        "alignment_post_bytes,"
        "rank_min_ms,rank_mean_ms,rank_max_ms,quality,plan_fingerprint,"
        "git_commit,hostname,gpu_name,rmm_resource,reserve_bytes,"
        "physical_plan_preset,physical_plan_config";
}

std::string_view memory_csv_header() {
    return
        "timestamp_utc,query,scale,ranks,nodes,record_type,rank,"
        "resident_current_bytes,resident_peak_bytes,"
        "query_current_bytes,query_peak_bytes,total_live_peak_bytes,"
        "resident_max_rank,query_max_rank,total_max_rank,"
        "plan_fingerprint,git_commit,hostname,gpu_name,rmm_resource,"
        "reserve_bytes,physical_plan_preset,physical_plan_config";
}

std::string_view messages_csv_header() {
    return
        "timestamp_utc,query,scale,ranks,nodes,iteration,rank,call_index,"
        "step_index,step_kind,step_label,phase_index,phase_name,"
        "communication_pattern,collective_primitive,api,direction,peer,root,"
        "column_ordinal,column_type,element_type,component,segment,"
        "element_count,element_size,payload_bytes,is_self,plan_fingerprint,"
        "git_commit,hostname,gpu_name,rmm_resource,reserve_bytes,"
        "physical_plan_preset,physical_plan_config";
}

bool csv_needs_header(
    const std::string& path, std::string_view expected_header)
{
    if (!std::filesystem::exists(path) ||
        std::filesystem::file_size(path) == 0) {
        return true;
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot inspect CSV header: " + path);
    }
    std::string actual_header;
    if (!std::getline(input, actual_header)) {
        throw std::runtime_error("cannot read CSV header: " + path);
    }
    if (!actual_header.empty() && actual_header.back() == '\r') {
        actual_header.pop_back();
    }
    if (actual_header != expected_header) {
        throw std::runtime_error(
            "CSV schema mismatch for '" + path +
            "': refusing to append; use a new output path or rotate the "
            "existing file");
    }
    return false;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const char c : value) {
        if (c == '"') result.push_back('"');
        result.push_back(c);
    }
    result.push_back('"');
    return result;
}

std::uint64_t stable_fingerprint(const std::string& value) {
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char byte : value) {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

std::string fingerprint_hex(std::uint64_t value) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << value;
    return output.str();
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw = std::chrono::system_clock::to_time_t(now);
    std::tm utc {};
    gmtime_r(&raw, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

} // namespace gamax::tpch::benchmark
