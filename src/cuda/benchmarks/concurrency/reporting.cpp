// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "reporting.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <unistd.h>

namespace gamax::benchmark::concurrency {
namespace {

constexpr std::array<char const*, 14> kSummaryDistributionPrefixes{
    "a_alone",
    "b_alone",
    "sequential_ab",
    "sequential_ba",
    "concurrent_makespan",
    "concurrent_a",
    "concurrent_b",
    "start_skew",
    "chunked_a_alone",
    "chunked_b_alone",
    "chunked_concurrent_makespan",
    "chunked_concurrent_a",
    "chunked_concurrent_b",
    "chunked_start_skew",
};

constexpr std::array<char const*, 6> kSummaryDistributionSuffixes{
    "_n", "_median_ms", "_mean_ms", "_stddev_ms", "_p5_ms", "_p95_ms",
};

constexpr std::array<char const*, 31> kSummaryMetricFields{
    "h_ideal",
    "h_ideal_ci_low",
    "h_ideal_ci_high",
    "h_sequential",
    "h_sequential_ci_low",
    "h_sequential_ci_high",
    "slowdown_a",
    "slowdown_a_ci_low",
    "slowdown_a_ci_high",
    "slowdown_b",
    "slowdown_b_ci_low",
    "slowdown_b_ci_high",
    "sequential_ab_vs_isolated_sum",
    "sequential_ba_vs_isolated_sum",
    "material_start_skew_count",
    "max_start_skew_fraction",
    "chunked_isolated_slowdown_a",
    "chunked_isolated_slowdown_a_ci_low",
    "chunked_isolated_slowdown_a_ci_high",
    "chunked_isolated_slowdown_b",
    "chunked_isolated_slowdown_b_ci_low",
    "chunked_isolated_slowdown_b_ci_high",
    "chunked_concurrent_speedup",
    "chunked_concurrent_speedup_ci_low",
    "chunked_concurrent_speedup_ci_high",
    "chunked_interference_slowdown_a",
    "chunked_interference_slowdown_b",
    "chunked_concurrent_a_vs_whole",
    "chunked_concurrent_b_vs_whole",
    "chunked_material_start_skew_count",
    "chunked_max_start_skew_fraction",
};

std::string csv_escape(std::string const& value)
{
    if (value.find_first_of(",\"\n\r") == std::string::npos) { return value; }
    std::string out = "\"";
    for (char c : value) {
        out += c;
        if (c == '"') { out += c; }
    }
    out += '"';
    return out;
}

std::string json_escape(std::string const& value)
{
    std::string out;
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

std::string shell_capture(char const* command)
{
    std::array<char, 512> buffer{};
    std::string output;
    std::unique_ptr<FILE, int (*)(FILE*)> pipe(popen(command, "r"), pclose);
    if (!pipe) { return {}; }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get())) {
        output += buffer.data();
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output;
}

std::string timestamp(char const* format)
{
    auto const now = std::chrono::system_clock::now();
    std::time_t const value = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    gmtime_r(&value, &tm);
    std::ostringstream out;
    out << std::put_time(&tm, format);
    return out.str();
}

void print_help(char const* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "  --op-a NAME                  operation A (default " << k_expr_name << ")\n"
        << "  --op-b NAME                  operation B (default " << k_gather_name << ")\n"
        << "  --a-exp-min N --a-exp-max N  A size exponent range (default 20..29)\n"
        << "  --b-exp-min N --b-exp-max N  B size exponent range (default 20..29)\n"
        << "  --a-chunk-size N             enable chunked A samples with N items/chunk\n"
        << "  --b-chunk-size N             enable chunked B samples with N items/chunk\n"
        << "  --warmups N                  warmups per sample type (default 5)\n"
        << "  --min-samples N              samples before CI test (default 30)\n"
        << "  --max-samples N              measured sample cap (default 100)\n"
        << "  --ci-threshold F             relative 95% CI half-width target (default 0.02)\n"
        << "  --bootstrap-iterations N     bootstrap resamples (default 2000)\n"
        << "  --seed N                     sample-order/bootstrap seed (default 42)\n"
        << "  --gate-timeout-seconds F     hidden-sync timeout (default 5)\n"
        << "  --device N                   CUDA device (default 0)\n"
        << "  --output-prefix PATH         output basename\n"
        << "  --list-operations            print registered stable names\n"
        << "  --self-test-gate             run synthetic timing test only\n";
}

template <typename T>
T parse_number(std::string const& text, std::string const& option)
{
    std::istringstream in(text);
    T value{};
    in >> value;
    if (!in || !in.eof()) { fail("invalid value for " + option + ": " + text); }
    return value;
}

double quantile(std::vector<double> values, double q)
{
    if (values.empty()) { return std::numeric_limits<double>::quiet_NaN(); }
    std::sort(values.begin(), values.end());
    double const position = q * static_cast<double>(values.size() - 1);
    auto const lower = static_cast<std::size_t>(std::floor(position));
    auto const upper = static_cast<std::size_t>(std::ceil(position));
    double const fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double median(std::vector<double> const& values) { return quantile(values, 0.5); }

Distribution summarize(std::vector<double> const& values)
{
    Distribution result;
    result.n = static_cast<int>(values.size());
    if (values.empty()) {
        result.median = result.mean = result.stddev = result.p5 = result.p95 =
            std::numeric_limits<double>::quiet_NaN();
        return result;
    }
    result.median = median(values);
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sum_sq = 0.0;
    for (double value : values) {
        double const delta = value - result.mean;
        sum_sq += delta * delta;
    }
    result.stddev = values.size() > 1 ? std::sqrt(sum_sq / (values.size() - 1)) : 0.0;
    result.p5 = quantile(values, 0.05);
    result.p95 = quantile(values, 0.95);
    return result;
}

template <typename Ratio>
ConfidenceInterval bootstrap_ratio(
    std::vector<double> const& a, std::vector<double> const& b,
    std::vector<double> const& c, int iterations, std::uint64_t seed, Ratio ratio)
{
    if (a.empty() || b.empty() || c.empty()) {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick_a(0, a.size() - 1);
    std::uniform_int_distribution<std::size_t> pick_b(0, b.size() - 1);
    std::uniform_int_distribution<std::size_t> pick_c(0, c.size() - 1);
    std::vector<double> aa(a.size()), bb(b.size()), cc(c.size()), estimates;
    estimates.reserve(iterations);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (double& value : aa) value = a[pick_a(rng)];
        for (double& value : bb) value = b[pick_b(rng)];
        for (double& value : cc) value = c[pick_c(rng)];
        estimates.push_back(ratio(median(aa), median(bb), median(cc)));
    }
    return {quantile(estimates, 0.025), quantile(estimates, 0.975)};
}

std::vector<double> values_for(
    std::vector<Observation> const& observations, SampleSpec sample,
    std::function<double(Timing const&)> const& get)
{
    std::vector<double> values;
    for (auto const& observation : observations) {
        if (!observation.warmup && observation.sample == sample) {
            values.push_back(get(observation.timing));
        }
    }
    return values;
}

double relative_half_width(double value, ConfidenceInterval ci)
{
    if (!std::isfinite(value) || value == 0.0 || !std::isfinite(ci.low) ||
        !std::isfinite(ci.high)) {
        return std::numeric_limits<double>::infinity();
    }
    return (ci.high - ci.low) / (2.0 * std::fabs(value));
}

void append_distribution(std::ostream& out, Distribution const& d)
{
    out << ',' << d.n << ',' << d.median << ',' << d.mean << ',' << d.stddev
        << ',' << d.p5 << ',' << d.p95;
}

std::string number(double value)
{
    if (std::isnan(value)) { return {}; }
    std::ostringstream out;
    out << std::setprecision(9) << value;
    return out.str();
}

} // namespace

Options parse_options(int argc, char** argv)
{
    Options options;
    auto take = [&](int& i, std::string const& name) -> std::string {
        if (++i >= argc) { fail("missing value for " + name); }
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { print_help(argv[0]); std::exit(0); }
        else if (arg == "--op-a") options.op_a = take(i, arg);
        else if (arg == "--op-b") options.op_b = take(i, arg);
        else if (arg == "--a-exp-min") options.a_min = parse_number<int>(take(i, arg), arg);
        else if (arg == "--a-exp-max") options.a_max = parse_number<int>(take(i, arg), arg);
        else if (arg == "--b-exp-min") options.b_min = parse_number<int>(take(i, arg), arg);
        else if (arg == "--b-exp-max") options.b_max = parse_number<int>(take(i, arg), arg);
        else if (arg == "--a-chunk-size") options.a_chunk_size =
            parse_number<std::uint64_t>(take(i, arg), arg);
        else if (arg == "--b-chunk-size") options.b_chunk_size =
            parse_number<std::uint64_t>(take(i, arg), arg);
        else if (arg == "--warmups") options.warmups = parse_number<int>(take(i, arg), arg);
        else if (arg == "--min-samples") options.min_samples =
            parse_number<int>(take(i, arg), arg);
        else if (arg == "--max-samples") options.max_samples =
            parse_number<int>(take(i, arg), arg);
        else if (arg == "--ci-threshold") options.ci_threshold =
            parse_number<double>(take(i, arg), arg);
        else if (arg == "--bootstrap-iterations") options.bootstrap_iterations =
            parse_number<int>(take(i, arg), arg);
        else if (arg == "--seed") options.random_seed =
            parse_number<std::uint64_t>(take(i, arg), arg);
        else if (arg == "--gate-timeout-seconds") options.gate_timeout_seconds =
            parse_number<double>(take(i, arg), arg);
        else if (arg == "--device") options.device = parse_number<int>(take(i, arg), arg);
        else if (arg == "--output-prefix") options.output_prefix = take(i, arg);
        else if (arg == "--self-test-gate") options.self_test_gate = true;
        else if (arg == "--list-operations") options.list_operations = true;
        else fail("unknown option: " + arg);
    }
    if (options.a_min > options.a_max || options.b_min > options.b_max) {
        fail("size exponent minimum exceeds maximum");
    }
    if ((options.a_chunk_size == 0) != (options.b_chunk_size == 0)) {
        fail("--a-chunk-size and --b-chunk-size must be specified together");
    }
    if (options.warmups < 0 || options.min_samples < 1 ||
        options.max_samples < options.min_samples) {
        fail("invalid warmup/sample counts");
    }
    if (options.bootstrap_iterations < 100 || options.ci_threshold <= 0.0 ||
        options.gate_timeout_seconds <= 0.0) {
        fail("bootstrap iterations, CI threshold, and gate timeout must be positive");
    }
    if (options.output_prefix.empty()) {
        options.output_prefix = "concurrency_" + timestamp("%Y%m%dT%H%M%SZ");
    }
    return options;
}

Metadata collect_metadata(int device)
{
    Metadata metadata;
    metadata.run_id = timestamp("%Y%m%dT%H%M%SZ") + "-" + std::to_string(::getpid());
    metadata.timestamp = timestamp("%Y-%m-%dT%H:%M:%SZ");
    metadata.git_commit = shell_capture("git rev-parse HEAD 2>/dev/null");
    metadata.git_dirty = !shell_capture("git status --porcelain 2>/dev/null").empty();
    metadata.device = device;
    metadata.toolkit_version = CUDART_VERSION;
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    metadata.gpu_name = prop.name;
    cuda_check(cudaDriverGetVersion(&metadata.driver_version), "cudaDriverGetVersion");
    cuda_check(cudaRuntimeGetVersion(&metadata.runtime_version), "cudaRuntimeGetVersion");
    cuda_check(cudaMemGetInfo(&metadata.initial_free_bytes, &metadata.total_bytes),
               "cudaMemGetInfo");
    cuda_check(cudaDeviceGetAttribute(&metadata.l2_bytes, cudaDevAttrL2CacheSize, device),
               "query L2 size");
    metadata.nvidia_smi_snapshot = shell_capture(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,temperature.gpu,clocks.sm,power.draw "
        "--format=csv,noheader 2>/dev/null");
    return metadata;
}

void write_metadata(
    std::string const& path, Metadata const& m, Options const& o,
    bool authoritative, std::string const& raw_path, std::string const& summary_path)
{
    std::ofstream out(path);
    if (!out) { fail("cannot open metadata output: " + path); }
    out << "{\n"
        << "  \"run_id\": \"" << json_escape(m.run_id) << "\",\n"
        << "  \"timestamp_utc\": \"" << m.timestamp << "\",\n"
        << "  \"git_commit\": \"" << json_escape(m.git_commit) << "\",\n"
        << "  \"git_dirty\": " << (m.git_dirty ? "true" : "false") << ",\n"
        << "  \"gpu_name\": \"" << json_escape(m.gpu_name) << "\",\n"
        << "  \"device_id\": " << m.device << ",\n"
        << "  \"driver_version\": " << m.driver_version << ",\n"
        << "  \"cuda_runtime_version\": " << m.runtime_version << ",\n"
        << "  \"cuda_toolkit_version\": " << m.toolkit_version << ",\n"
        << "  \"initial_free_bytes\": " << m.initial_free_bytes << ",\n"
        << "  \"total_device_bytes\": " << m.total_bytes << ",\n"
        << "  \"l2_bytes\": " << m.l2_bytes << ",\n"
        << "  \"nvidia_smi_snapshot\": \"" << json_escape(m.nvidia_smi_snapshot) << "\",\n"
        << "  \"op_a\": \"" << json_escape(o.op_a) << "\",\n"
        << "  \"op_b\": \"" << json_escape(o.op_b) << "\",\n"
        << "  \"a_exponents\": [" << o.a_min << ", " << o.a_max << "],\n"
        << "  \"b_exponents\": [" << o.b_min << ", " << o.b_max << "],\n"
        << "  \"a_chunk_size\": " << o.a_chunk_size << ",\n"
        << "  \"b_chunk_size\": " << o.b_chunk_size << ",\n"
        << "  \"chunking_enabled\": " << (o.chunking_enabled() ? "true" : "false") << ",\n"
        << "  \"chunk_schedule\": \"natural_alternating_enqueue\",\n"
        << "  \"warmups_per_sample_type\": " << o.warmups << ",\n"
        << "  \"minimum_samples_per_type\": " << o.min_samples << ",\n"
        << "  \"maximum_samples_per_type\": " << o.max_samples << ",\n"
        << "  \"bootstrap_iterations\": " << o.bootstrap_iterations << ",\n"
        << "  \"ci_relative_half_width_threshold\": " << o.ci_threshold << ",\n"
        << "  \"sample_order_random_seed\": " << o.random_seed << ",\n"
        << "  \"data_seed\": " << k_data_seed << ",\n"
        << "  \"authoritative_configuration\": " << (authoritative ? "true" : "false") << ",\n"
        << "  \"raw_csv\": \"" << json_escape(raw_path) << "\",\n"
        << "  \"summary_csv\": \"" << json_escape(summary_path) << "\"\n"
        << "}\n";
}

CellSummary make_summary(
    std::vector<Observation> const& observations, Options const& options,
    std::uint64_t cell_seed)
{
    auto makespan = [](Timing const& t) { return t.makespan_ms; };
    auto a = values_for(observations, {Mode::a_alone, ExecutionVariant::whole}, makespan);
    auto b = values_for(observations, {Mode::b_alone, ExecutionVariant::whole}, makespan);
    auto seq_ab = values_for(observations, {Mode::sequential_ab, ExecutionVariant::whole}, makespan);
    auto seq_ba = values_for(observations, {Mode::sequential_ba, ExecutionVariant::whole}, makespan);
    auto concurrent = values_for(observations, {Mode::concurrent, ExecutionVariant::whole}, makespan);
    auto concurrent_a = values_for(observations, {Mode::concurrent, ExecutionVariant::whole},
                                   [](Timing const& t) { return t.a_ms; });
    auto concurrent_b = values_for(observations, {Mode::concurrent, ExecutionVariant::whole},
                                   [](Timing const& t) { return t.b_ms; });
    auto skew = values_for(observations, {Mode::concurrent, ExecutionVariant::whole},
                           [](Timing const& t) { return t.start_skew_ms; });

    CellSummary s;
    s.a_alone = summarize(a);
    s.b_alone = summarize(b);
    s.seq_ab = summarize(seq_ab);
    s.seq_ba = summarize(seq_ba);
    s.concurrent_makespan = summarize(concurrent);
    s.concurrent_a = summarize(concurrent_a);
    s.concurrent_b = summarize(concurrent_b);
    s.start_skew = summarize(skew);
    s.h_ideal = s.concurrent_makespan.median / std::max(s.a_alone.median, s.b_alone.median);
    s.h_sequential = s.concurrent_makespan.median / (s.a_alone.median + s.b_alone.median);
    s.slowdown_a = s.concurrent_a.median / s.a_alone.median;
    s.slowdown_b = s.concurrent_b.median / s.b_alone.median;
    double const isolated_sum = s.a_alone.median + s.b_alone.median;
    s.sequential_ab_ratio = s.seq_ab.median / isolated_sum;
    s.sequential_ba_ratio = s.seq_ba.median / isolated_sum;

    auto accumulate_skew = [&](ExecutionVariant variant, int& count, double& maximum) {
        for (auto const& observation : observations) {
            if (observation.warmup || observation.sample != SampleSpec{Mode::concurrent, variant}) {
                continue;
            }
            double const denominator = std::min(observation.timing.a_ms, observation.timing.b_ms);
            double const fraction = observation.timing.start_skew_ms / denominator;
            maximum = std::max(maximum, fraction);
            if (fraction > 0.05) { ++count; }
        }
    };
    accumulate_skew(ExecutionVariant::whole, s.material_start_skew_count,
                    s.max_start_skew_fraction);

    s.h_ideal_ci = bootstrap_ratio(a, b, concurrent, options.bootstrap_iterations,
        cell_seed + 11, [](double x, double y, double z) { return z / std::max(x, y); });
    s.h_sequential_ci = bootstrap_ratio(a, b, concurrent, options.bootstrap_iterations,
        cell_seed + 23, [](double x, double y, double z) { return z / (x + y); });
    std::vector<double> dummy{1.0};
    s.slowdown_a_ci = bootstrap_ratio(a, dummy, concurrent_a, options.bootstrap_iterations,
        cell_seed + 37, [](double x, double, double z) { return z / x; });
    s.slowdown_b_ci = bootstrap_ratio(b, dummy, concurrent_b, options.bootstrap_iterations,
        cell_seed + 53, [](double x, double, double z) { return z / x; });

    bool chunk_converged = true;
    if (options.chunking_enabled()) {
        auto chunk_a = values_for(observations, {Mode::a_alone, ExecutionVariant::chunked}, makespan);
        auto chunk_b = values_for(observations, {Mode::b_alone, ExecutionVariant::chunked}, makespan);
        auto chunk_concurrent = values_for(
            observations, {Mode::concurrent, ExecutionVariant::chunked}, makespan);
        auto chunk_concurrent_a = values_for(
            observations, {Mode::concurrent, ExecutionVariant::chunked},
            [](Timing const& t) { return t.a_ms; });
        auto chunk_concurrent_b = values_for(
            observations, {Mode::concurrent, ExecutionVariant::chunked},
            [](Timing const& t) { return t.b_ms; });
        auto chunk_skew = values_for(
            observations, {Mode::concurrent, ExecutionVariant::chunked},
            [](Timing const& t) { return t.start_skew_ms; });
        s.chunked_a_alone = summarize(chunk_a);
        s.chunked_b_alone = summarize(chunk_b);
        s.chunked_concurrent_makespan = summarize(chunk_concurrent);
        s.chunked_concurrent_a = summarize(chunk_concurrent_a);
        s.chunked_concurrent_b = summarize(chunk_concurrent_b);
        s.chunked_start_skew = summarize(chunk_skew);
        s.chunked_isolated_slowdown_a = s.chunked_a_alone.median / s.a_alone.median;
        s.chunked_isolated_slowdown_b = s.chunked_b_alone.median / s.b_alone.median;
        s.chunked_concurrent_speedup =
            s.concurrent_makespan.median / s.chunked_concurrent_makespan.median;
        s.chunked_interference_slowdown_a =
            s.chunked_concurrent_a.median / s.chunked_a_alone.median;
        s.chunked_interference_slowdown_b =
            s.chunked_concurrent_b.median / s.chunked_b_alone.median;
        s.chunked_concurrent_a_vs_whole =
            s.chunked_concurrent_a.median / s.concurrent_a.median;
        s.chunked_concurrent_b_vs_whole =
            s.chunked_concurrent_b.median / s.concurrent_b.median;
        s.chunked_isolated_slowdown_a_ci = bootstrap_ratio(
            a, dummy, chunk_a, options.bootstrap_iterations, cell_seed + 67,
            [](double x, double, double z) { return z / x; });
        s.chunked_isolated_slowdown_b_ci = bootstrap_ratio(
            b, dummy, chunk_b, options.bootstrap_iterations, cell_seed + 79,
            [](double x, double, double z) { return z / x; });
        s.chunked_concurrent_speedup_ci = bootstrap_ratio(
            concurrent, dummy, chunk_concurrent, options.bootstrap_iterations, cell_seed + 97,
            [](double x, double, double z) { return x / z; });
        accumulate_skew(ExecutionVariant::chunked, s.chunked_material_start_skew_count,
                        s.chunked_max_start_skew_fraction);
        chunk_converged =
            relative_half_width(s.chunked_isolated_slowdown_a,
                                s.chunked_isolated_slowdown_a_ci) <= options.ci_threshold &&
            relative_half_width(s.chunked_isolated_slowdown_b,
                                s.chunked_isolated_slowdown_b_ci) <= options.ci_threshold &&
            relative_half_width(s.chunked_concurrent_speedup,
                                s.chunked_concurrent_speedup_ci) <= options.ci_threshold;
    } else {
        auto nan = std::numeric_limits<double>::quiet_NaN();
        s.chunked_a_alone = s.chunked_b_alone = s.chunked_concurrent_makespan =
            s.chunked_concurrent_a = s.chunked_concurrent_b = s.chunked_start_skew =
                summarize({});
        s.chunked_isolated_slowdown_a = s.chunked_isolated_slowdown_b =
            s.chunked_concurrent_speedup = s.chunked_interference_slowdown_a =
            s.chunked_interference_slowdown_b = s.chunked_concurrent_a_vs_whole =
            s.chunked_concurrent_b_vs_whole = nan;
        s.chunked_isolated_slowdown_a_ci = s.chunked_isolated_slowdown_b_ci =
            s.chunked_concurrent_speedup_ci = {nan, nan};
    }

    s.ci_converged =
        relative_half_width(s.h_ideal, s.h_ideal_ci) <= options.ci_threshold &&
        relative_half_width(s.h_sequential, s.h_sequential_ci) <= options.ci_threshold &&
        chunk_converged;
    s.stable = s.ci_converged && s.material_start_skew_count == 0 &&
        s.chunked_material_start_skew_count == 0;
    return s;
}

class OutputWriter::Impl {
public:
    Impl(
        std::string raw_path, std::string summary_path, Metadata const& metadata,
        OperationDescriptor const& a, OperationDescriptor const& b, Options const& options)
        : metadata_{metadata}, a_{a}, b_{b}, options_{options},
          raw_{std::move(raw_path)}, summary_{std::move(summary_path)}
    {
        if (!raw_) { fail("cannot open raw CSV"); }
        if (!summary_) { fail("cannot open summary CSV"); }
        raw_ << "run_id,timestamp_utc,git_commit,git_dirty,gpu_name,device_id,driver_version,"
             << "cuda_runtime_version,cuda_toolkit_version,device_snapshot,op_a,op_b,config_a,config_b,"
             << "a_exponent,b_exponent,a_size,b_size,a_chunk_size,b_chunk_size,a_chunk_count,"
             << "b_chunk_count,mode,execution_variant,enqueue_order,repetition,random_seed,warmup,success,reason,"
             << "a_ms,b_ms,pair_makespan_ms,start_skew_ms,start_skew_material\n";
        summary_ << "run_id,timestamp_utc,git_commit,git_dirty,gpu_name,device_id,op_a,op_b,"
                 << "config_a,config_b,a_exponent,b_exponent,a_size,b_size,a_chunk_size,b_chunk_size,"
                 << "a_chunk_count,b_chunk_count,status,reason,stable";
        for (auto const* prefix : kSummaryDistributionPrefixes) {
            for (auto const* suffix : kSummaryDistributionSuffixes) {
                summary_ << ',' << prefix << suffix;
            }
        }
        for (auto const* field : kSummaryMetricFields) { summary_ << ',' << field; }
        summary_ << '\n';
    }

    std::uint64_t effective_a(GenericConfig ac) const
    {
        return options_.chunking_enabled() ? effective_chunk_size(ac.size(), options_.a_chunk_size)
                                           : ac.size();
    }
    std::uint64_t effective_b(GenericConfig bc) const
    {
        return options_.chunking_enabled() ? effective_chunk_size(bc.size(), options_.b_chunk_size)
                                           : bc.size();
    }

    void prefix(std::ostream& out, GenericConfig ac, GenericConfig bc) const
    {
        auto const a_chunk = effective_a(ac);
        auto const b_chunk = effective_b(bc);
        out << csv_escape(metadata_.run_id) << ',' << metadata_.timestamp << ','
            << metadata_.git_commit << ',' << (metadata_.git_dirty ? 1 : 0) << ','
            << csv_escape(metadata_.gpu_name) << ',' << metadata_.device << ',';
        if (&out == &raw_) {
            out << metadata_.driver_version << ',' << metadata_.runtime_version << ','
                << metadata_.toolkit_version << ',' << csv_escape(metadata_.nvidia_smi_snapshot)
                << ',';
        }
        out << csv_escape(a_.stable_name()) << ',' << csv_escape(b_.stable_name()) << ','
            << csv_escape(a_.config_string(ac)) << ',' << csv_escape(b_.config_string(bc)) << ','
            << ac.size_exponent << ',' << bc.size_exponent << ',' << ac.size() << ',' << bc.size()
            << ',' << a_chunk << ',' << b_chunk << ',' << chunk_count_for(ac.size(), a_chunk)
            << ',' << chunk_count_for(bc.size(), b_chunk);
    }

    Metadata const& metadata_;
    OperationDescriptor const& a_;
    OperationDescriptor const& b_;
    Options const& options_;
    std::ofstream raw_;
    std::ofstream summary_;
};

OutputWriter::OutputWriter(
    std::string raw_path, std::string summary_path, Metadata const& metadata,
    OperationDescriptor const& a, OperationDescriptor const& b, Options const& options)
    : impl_{std::make_unique<Impl>(
          std::move(raw_path), std::move(summary_path), metadata, a, b, options)} {}
OutputWriter::~OutputWriter() = default;

void OutputWriter::raw_observation(
    GenericConfig ac, GenericConfig bc, Observation const& observation, std::uint64_t seed)
{
    auto& out = impl_->raw_;
    impl_->prefix(out, ac, bc);
    out << ',' << mode_name(observation.sample.mode) << ','
        << variant_name(observation.sample.variant) << ',' << observation.timing.order << ','
        << observation.repetition << ',' << seed << ',' << (observation.warmup ? 1 : 0)
        << ",1,," << number(observation.timing.a_ms) << ',' << number(observation.timing.b_ms)
        << ',' << number(observation.timing.makespan_ms) << ','
        << number(observation.timing.start_skew_ms) << ',';
    bool const material = observation.sample.mode == Mode::concurrent &&
        observation.timing.start_skew_ms >
            0.05 * std::min(observation.timing.a_ms, observation.timing.b_ms);
    out << (material ? 1 : 0) << '\n';
    out.flush();
}

void OutputWriter::raw_failure(
    GenericConfig ac, GenericConfig bc, std::string const& status,
    std::string const& reason, std::uint64_t seed)
{
    auto& out = impl_->raw_;
    impl_->prefix(out, ac, bc);
    out << ",cell,whole," << status << ",-1," << seed << ",0,0,"
        << csv_escape(reason) << ",,,,,\n";
    out.flush();
}

void OutputWriter::summary_success(GenericConfig ac, GenericConfig bc, CellSummary const& s)
{
    auto& out = impl_->summary_;
    impl_->prefix(out, ac, bc);
    out << ",success,," << (s.stable ? 1 : 0);
    for (auto const* distribution : {
             &s.a_alone, &s.b_alone, &s.seq_ab, &s.seq_ba, &s.concurrent_makespan,
             &s.concurrent_a, &s.concurrent_b, &s.start_skew, &s.chunked_a_alone,
             &s.chunked_b_alone, &s.chunked_concurrent_makespan, &s.chunked_concurrent_a,
             &s.chunked_concurrent_b, &s.chunked_start_skew}) {
        append_distribution(out, *distribution);
    }
    out << ',' << s.h_ideal << ',' << s.h_ideal_ci.low << ',' << s.h_ideal_ci.high
        << ',' << s.h_sequential << ',' << s.h_sequential_ci.low << ','
        << s.h_sequential_ci.high << ',' << s.slowdown_a << ',' << s.slowdown_a_ci.low
        << ',' << s.slowdown_a_ci.high << ',' << s.slowdown_b << ',' << s.slowdown_b_ci.low
        << ',' << s.slowdown_b_ci.high << ',' << s.sequential_ab_ratio << ','
        << s.sequential_ba_ratio << ',' << s.material_start_skew_count << ','
        << s.max_start_skew_fraction << ',' << s.chunked_isolated_slowdown_a << ','
        << s.chunked_isolated_slowdown_a_ci.low << ','
        << s.chunked_isolated_slowdown_a_ci.high << ',' << s.chunked_isolated_slowdown_b
        << ',' << s.chunked_isolated_slowdown_b_ci.low << ','
        << s.chunked_isolated_slowdown_b_ci.high << ',' << s.chunked_concurrent_speedup
        << ',' << s.chunked_concurrent_speedup_ci.low << ','
        << s.chunked_concurrent_speedup_ci.high << ',' << s.chunked_interference_slowdown_a
        << ',' << s.chunked_interference_slowdown_b << ',' << s.chunked_concurrent_a_vs_whole
        << ',' << s.chunked_concurrent_b_vs_whole << ','
        << s.chunked_material_start_skew_count << ',' << s.chunked_max_start_skew_fraction
        << '\n';
    out.flush();
}

void OutputWriter::summary_failure(
    GenericConfig ac, GenericConfig bc, std::string const& status, std::string const& reason)
{
    auto& out = impl_->summary_;
    impl_->prefix(out, ac, bc);
    out << ',' << status << ',' << csv_escape(reason) << ",0";
    constexpr auto empty_tail_fields =
        kSummaryDistributionPrefixes.size() * kSummaryDistributionSuffixes.size() +
        kSummaryMetricFields.size();
    for (std::size_t i = 0; i < empty_tail_fields; ++i) { out << ','; }
    out << '\n';
    out.flush();
}

} // namespace gamax::benchmark::concurrency
