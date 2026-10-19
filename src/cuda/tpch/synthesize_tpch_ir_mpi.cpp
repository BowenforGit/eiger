// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/context.hpp"
#include "alps/exec/distributed_executor.hpp"
#include "exec/executor.hpp"
#include "exec/print.hpp"
#include "exec/step_profile.hpp"
#include "phys/phys.hpp"
#include "operators/groupby/groupby_partition.hpp"
#include "tpch/distributed_benchmark_support.hpp"
#include "tpch/distributed_tpch.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/physical_plan_config.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"
#include "tpch/verify.hpp"

namespace {

namespace bench = gamax::tpch::benchmark;
namespace physical = gamax::tpch::physical_plan;

constexpr int kTerminalSynthesisExitCode = 64;

struct Args {
    int query_id {0};
    int scale_factor {1};
    int initial_samples {5};
    int tie_samples {10};
    double commit_threshold {0.03};
    double tie_threshold {0.01};
    int max_passes {3};
    std::uint64_t seed {20260728};
    bench::WarmupPolicy warmup;
    std::string output_config;
    std::string checkpoint;
    std::string candidate_csv;
    std::string rmm_resource;
    gamax::tpch::FileShardStrategy file_shard_strategy {
        gamax::tpch::FileShardStrategy::RoundRobin};
    int parquet_decode_threads {8};
    bool resume {false};
    bool skip_dataset_preflight {false};
    bool continue_on_verification_failure {false};
};

[[noreturn]] void usage(
    const char* program,
    const std::string& error = {},
    int exit_code = 2)
{
    auto& output = exit_code == 0 ? std::cout : std::cerr;
    if (!error.empty()) output << "error: " << error << "\n";
    output
        << "usage: " << program << " QUERY_ID [options]\n"
        << "  --scale N                 scale factor\n"
        << "  --output-config PATH      final qNN physical-plan JSON\n"
        << "  --checkpoint PATH         resumable checkpoint JSON\n"
        << "  --candidate-csv PATH      append raw candidate samples\n"
        << "  --resume                  resume an existing checkpoint\n"
        << "  --seed N                  randomized interleave seed\n"
        << "  --initial-samples N       initial samples per candidate (default 5)\n"
        << "  --tie-samples N           extra tie samples (default 10)\n"
        << "  --commit-threshold X      direct commit fraction (default 0.03)\n"
        << "  --tie-threshold X         tie commit fraction (default 0.01)\n"
        << "  --max-passes N            coordinate-descent cap (default 3)\n"
        << "  --warmup-min N            minimum candidate warmups\n"
        << "  --warmup-window N         stability window\n"
        << "  --warmup-threshold X      relative stability range\n"
        << "  --warmup-max N            maximum candidate warmups\n"
        << "  --rmm-resource MODE       arena or cuda\n"
        << "  --file-shard-strategy MODE\n"
        << "  --parquet-decode-threads N\n"
        << "  --continue-on-verification-failure\n"
        << "  --skip-dataset-preflight\n";
    std::exit(exit_code);
}

int parse_int(const char* value, const char* option, const char* program) {
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || *end != '\0') {
        usage(program, std::string(option) + " expects an integer");
    }
    return static_cast<int>(parsed);
}

std::uint64_t parse_u64(
    const char* value,
    const char* option,
    const char* program)
{
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0') {
        usage(program, std::string(option) + " expects an unsigned integer");
    }
    return parsed;
}

double parse_double(const char* value, const char* option, const char* program) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (!end || *end != '\0') {
        usage(program, std::string(option) + " expects a number");
    }
    return parsed;
}

Args parse_args(int argc, char** argv) {
    if (argc < 2) usage(argv[0]);
    if (std::strcmp(argv[1], "--help") == 0 ||
        std::strcmp(argv[1], "-h") == 0) {
        usage(argv[0], {}, 0);
    }
    Args args;
    args.query_id = parse_int(argv[1], "QUERY_ID", argv[0]);
    auto next = [&](int& index, const char* option) -> const char* {
        if (index + 1 >= argc) {
            usage(argv[0], std::string(option) + " needs a value");
        }
        return argv[++index];
    };
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--scale") {
            args.scale_factor = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--output-config") {
            args.output_config = next(index, option.c_str());
        } else if (option == "--checkpoint") {
            args.checkpoint = next(index, option.c_str());
        } else if (option == "--candidate-csv") {
            args.candidate_csv = next(index, option.c_str());
        } else if (option == "--resume") {
            args.resume = true;
        } else if (option == "--skip-dataset-preflight") {
            args.skip_dataset_preflight = true;
        } else if (option == "--continue-on-verification-failure") {
            args.continue_on_verification_failure = true;
        } else if (option == "--seed") {
            args.seed = parse_u64(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--initial-samples") {
            args.initial_samples = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--tie-samples") {
            args.tie_samples = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--commit-threshold") {
            args.commit_threshold = parse_double(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--tie-threshold") {
            args.tie_threshold = parse_double(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--max-passes") {
            args.max_passes = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-min") {
            args.warmup.minimum_runs = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-window") {
            args.warmup.window = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-threshold") {
            args.warmup.relative_range_threshold = parse_double(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-max") {
            args.warmup.maximum_runs = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--rmm-resource") {
            args.rmm_resource = next(index, option.c_str());
        } else if (option == "--file-shard-strategy") {
            try {
                args.file_shard_strategy =
                    gamax::tpch::parse_file_shard_strategy(
                        next(index, option.c_str()));
            } catch (const std::exception& error) {
                usage(argv[0], error.what());
            }
        } else if (option == "--parquet-decode-threads") {
            args.parquet_decode_threads = parse_int(
                next(index, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--help" || option == "-h") {
            usage(argv[0], {}, 0);
        } else {
            usage(argv[0], "unknown option: " + option);
        }
    }
    if (args.query_id < 1 || args.query_id > 22) {
        usage(argv[0], "QUERY_ID must be in [1, 22]");
    }
    if (args.scale_factor <= 0 || args.initial_samples <= 0 ||
        args.tie_samples <= 0 || args.max_passes <= 0 ||
        args.parquet_decode_threads <= 0) {
        usage(argv[0], "scale, sample counts, pass count, and threads must be positive");
    }
    if (args.commit_threshold < 0.0 || args.tie_threshold < 0.0) {
        usage(argv[0], "commit thresholds must be nonnegative");
    }
    if (args.output_config.empty() || args.checkpoint.empty() ||
        args.candidate_csv.empty()) {
        usage(
            argv[0],
            "--output-config, --checkpoint, and --candidate-csv are required");
    }
    if (!args.rmm_resource.empty() && args.rmm_resource != "arena" &&
        args.rmm_resource != "cuda") {
        usage(argv[0], "--rmm-resource must be arena or cuda");
    }
    bench::validate_warmup_policy(args.warmup);
    return args;
}

std::string golden_path_for(int query_id, int scale_factor) {
    const char* data_path = std::getenv("TPCH_DATA_PATH");
    if (!data_path || !*data_path) {
        throw std::runtime_error("verification requires TPCH_DATA_PATH");
    }
    std::ostringstream path;
    path << data_path << "/sf" << scale_factor << "/golden/q"
         << (query_id < 10 ? "0" : "") << query_id << ".parquet";
    return path.str();
}

void ensure_parent_directory(const std::string& path) {
    const std::filesystem::path file(path);
    if (!file.parent_path().empty()) {
        std::filesystem::create_directories(file.parent_path());
    }
}

enum class RunFailureKind : int {
    None = 0,
    PartitionGroupByCapacity = 1,
    Other = 2,
};

struct RunResult {
    bool ok {false};
    double maximum_rank_ms {0.0};
    std::unique_ptr<gamax::table> result;
    std::string detail;
    RunFailureKind failure_kind {RunFailureKind::None};
};

RunResult run_once(
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::BenchExecState& preload,
    const alps::Context& context)
{
    MPI_Barrier(context.mpi_comm);
    const auto start = std::chrono::steady_clock::now();
    std::unique_ptr<gamax::table> result;
    int local_ok = 1;
    int local_failure_kind = static_cast<int>(RunFailureKind::None);
    std::string detail;
    try {
        gamax::phys::Context physical_context{context.stream, nullptr};
        alps::exec::DistributedExecOptions options;
        options.eq_join_implementation =
            alps::exec::DistributedEqJoinImplementation::Legacy;
        result = alps::exec::benchmark_run_distributed_steps(
            plan, preload, physical_context, context, options);
        const auto status = cudaStreamSynchronize(context.stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    } catch (const gamax::groupby::partition::capacity_error& error) {
        local_ok = 0;
        local_failure_kind = static_cast<int>(
            RunFailureKind::PartitionGroupByCapacity);
        detail = error.what();
    } catch (const std::exception& error) {
        local_ok = 0;
        local_failure_kind = static_cast<int>(RunFailureKind::Other);
        detail = error.what();
    }
    const auto finish = std::chrono::steady_clock::now();
    int global_ok = 0;
    MPI_Allreduce(
        &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, context.mpi_comm);
    if (!global_ok) {
        int global_failure_kind = 0;
        MPI_Allreduce(
            &local_failure_kind, &global_failure_kind, 1, MPI_INT,
            MPI_MAX, context.mpi_comm);
        const auto failure_kind =
            static_cast<RunFailureKind>(global_failure_kind);
        if (detail.empty() &&
            failure_kind == RunFailureKind::PartitionGroupByCapacity) {
            detail = "partition GroupBy capacity exceeded on another rank";
        }
        return RunResult{
            false, 0.0, {}, std::move(detail), failure_kind};
    }
    const double local_ms =
        std::chrono::duration<double, std::milli>(finish - start).count();
    double maximum_ms = 0.0;
    MPI_Allreduce(
        &local_ms, &maximum_ms, 1, MPI_DOUBLE, MPI_MAX, context.mpi_comm);
    return RunResult{
        true, maximum_ms, std::move(result), {}, RunFailureKind::None};
}

bool verify(
    RunResult& run,
    const Args& args,
    const alps::Context& context,
    std::string& detail)
{
    int verified = 1;
    if (context.rank == 0) {
        if (!run.result) {
            verified = 0;
            detail = "root did not receive the gathered result";
        } else {
            const auto verification = gamax::tpch::verify_result(
                run.result->view(),
                golden_path_for(args.query_id, args.scale_factor));
            if (!verification.ok) {
                verified = 0;
                detail = verification.diagnostic;
            }
        }
    }
    MPI_Bcast(&verified, 1, MPI_INT, 0, context.mpi_comm);
    return verified != 0;
}

void checkpoint_active(
    physical::Config& config,
    int step_index,
    const std::string& implementation,
    const Args& args,
    const alps::Context& context)
{
    config.search.active_step_index = step_index;
    config.search.active_implementation = implementation;
    if (context.rank == 0) {
        physical::write_json_atomic(config, args.checkpoint);
    }
    MPI_Barrier(context.mpi_comm);
}

void checkpoint_clear(
    physical::Config& config,
    const Args& args,
    const alps::Context& context)
{
    config.search.active_step_index.reset();
    config.search.active_implementation.clear();
    if (context.rank == 0) {
        physical::write_json_atomic(config, args.checkpoint);
    }
    MPI_Barrier(context.mpi_comm);
}

void append_sample(
    const Args& args,
    const physical::Config& config,
    int step_index,
    const std::string& implementation,
    const std::string& phase,
    int sample,
    const RunResult& run,
    const alps::Context& context)
{
    if (context.rank != 0) return;
    ensure_parent_directory(args.candidate_csv);
    const bool header = !std::filesystem::exists(args.candidate_csv) ||
        std::filesystem::file_size(args.candidate_csv) == 0;
    std::ofstream output(args.candidate_csv, std::ios::app);
    if (!output) {
        throw std::runtime_error(
            "cannot append candidate CSV: " + args.candidate_csv);
    }
    if (header) {
        output
            << "timestamp_utc,query,scale,ranks,pass,coordinate_position,"
               "step_index,implementation,phase,sample,status,"
               "maximum_rank_ms,detail\n";
    }
    output << bench::utc_timestamp() << ",Q" << args.query_id << ","
           << args.scale_factor << "," << config.focus.ranks << ","
           << config.search.pass << ","
           << config.search.coordinate_position << "," << step_index << ","
           << implementation << "," << phase << "," << sample << ","
           << (run.ok ? "ok" : "invalid") << "," << std::fixed
           << std::setprecision(6) << run.maximum_rank_ms << ","
           << bench::csv_escape(run.detail) << "\n";
}

void set_join_variant(
    physical::Config& config,
    int step_index,
    const std::string& implementation)
{
    auto found = std::find_if(
        config.join_overrides.begin(), config.join_overrides.end(),
        [&](const auto& value) { return value.step_index == step_index; });
    if (found == config.join_overrides.end()) {
        throw std::runtime_error(
            "join coordinate is absent from physical config");
    }
    found->implementation = implementation;
}

std::string get_join_variant(
    const physical::Config& config,
    int step_index)
{
    const auto found = std::find_if(
        config.join_overrides.begin(), config.join_overrides.end(),
        [&](const auto& value) { return value.step_index == step_index; });
    if (found == config.join_overrides.end()) {
        throw std::runtime_error(
            "join coordinate is absent from physical config");
    }
    return found->implementation;
}

gamax::exec::ExecPlan build_plan(
    const Args& args,
    const physical::Config& config,
    gamax::tpch::SourceReadPoliciesPtr policies)
{
    auto local = gamax::tpch::dist::build_local_exec_plan(args.query_id);
    physical::apply(local, config);
    return gamax::tpch::dist::distribute_exec_plan(
        args.query_id, std::move(policies), std::move(local));
}

std::vector<int> initial_coordinate_order(
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::BenchExecState& preload,
    const std::vector<physical::JoinCoordinate>& coordinates,
    const alps::Context& context)
{
    MPI_Barrier(context.mpi_comm);
    gamax::exec::StepProfileSink sink;
    gamax::phys::Context physical_context{context.stream, &sink};
    alps::exec::DistributedExecOptions options;
    options.eq_join_implementation =
        alps::exec::DistributedEqJoinImplementation::Legacy;
    auto result = alps::exec::benchmark_run_distributed_steps(
        plan, preload, physical_context, context, options);
    sink.finalize(context.stream);
    std::vector<double> distributed_join_times;
    for (const auto& row : sink.rows()) {
        if (row.kind != gamax::exec::StepKind::DistributedEqJoin) continue;
        double maximum = 0.0;
        MPI_Allreduce(
            &row.gpu_ms, &maximum, 1, MPI_DOUBLE, MPI_MAX,
            context.mpi_comm);
        distributed_join_times.push_back(maximum);
    }
    if (distributed_join_times.size() != coordinates.size()) {
        throw std::runtime_error(
            "local/distributed inner-join coordinate count differs");
    }
    std::vector<std::pair<double, int>> ranked;
    for (std::size_t index = 0; index < coordinates.size(); ++index) {
        ranked.emplace_back(
            distributed_join_times[index], coordinates[index].step_index);
    }
    std::stable_sort(
        ranked.begin(), ranked.end(),
        [](const auto& left, const auto& right) {
            return left.first > right.first;
        });
    std::vector<int> order;
    for (const auto& [time, step] : ranked) {
        (void)time;
        order.push_back(step);
    }
    return order;
}

struct Candidate {
    std::string implementation;
    gamax::exec::ExecPlan plan;
    bool valid {true};
    std::string detail;
    std::vector<double> samples;
    int warmup_runs {0};
    bool verification_mismatch_ignored {false};
};

double median(const std::vector<double>& values) {
    return bench::summarize(values).median;
}

std::uint64_t round_seed(
    const physical::Config& config,
    int step_index,
    int round,
    std::uint64_t salt)
{
    return config.search.seed ^
        (static_cast<std::uint64_t>(config.search.pass + 1) << 48) ^
        (static_cast<std::uint64_t>(step_index + 1) << 24) ^
        static_cast<std::uint64_t>(round + 1) ^ salt;
}

RunResult execute_candidate(
    Candidate& candidate,
    physical::Config& config,
    int step_index,
    const std::string& phase,
    int sample,
    const Args& args,
    const gamax::exec::BenchExecState& preload,
    const alps::Context& context)
{
    checkpoint_active(
        config, step_index, candidate.implementation, args, context);
    auto run = run_once(candidate.plan, preload, context);
    append_sample(
        args, config, step_index, candidate.implementation,
        phase, sample, run, context);
    if (!run.ok) {
        candidate.valid = false;
        candidate.detail = run.detail.empty()
            ? "candidate failed on at least one rank"
            : run.detail;
    }
    return run;
}

void warm_candidate(
    Candidate& candidate,
    physical::Config& config,
    int step_index,
    const Args& args,
    const gamax::exec::BenchExecState& preload,
    const alps::Context& context)
{
    std::vector<double> samples;
    auto status = bench::WarmupStatus::Continue;
    while (candidate.valid && status == bench::WarmupStatus::Continue) {
        auto run = execute_candidate(
            candidate, config, step_index, "warmup",
            candidate.warmup_runs, args, preload, context);
        ++candidate.warmup_runs;
        if (!run.ok) break;
        samples.push_back(run.maximum_rank_ms);
        status = bench::evaluate_warmup(samples, args.warmup);
    }
}

void measure_interleaved(
    std::vector<Candidate>& candidates,
    const std::vector<int>& selected,
    int rounds,
    const std::string& phase,
    physical::Config& config,
    int step_index,
    const Args& args,
    const gamax::exec::BenchExecState& preload,
    const alps::Context& context)
{
    for (int round = 0; round < rounds; ++round) {
        auto order = selected;
        std::mt19937_64 random(round_seed(
            config, step_index, round,
            phase == "tie" ? UINT64_C(0x9e3779b97f4a7c15) : 0));
        std::shuffle(order.begin(), order.end(), random);
        for (const int index : order) {
            auto& candidate = candidates[static_cast<std::size_t>(index)];
            if (!candidate.valid) continue;
            auto run = execute_candidate(
                candidate, config, step_index, phase, round,
                args, preload, context);
            if (!run.ok) continue;
            candidate.samples.push_back(run.maximum_rank_ms);
        }
    }
}

void add_record(
    physical::Config& config,
    const Candidate& candidate,
    int step_index)
{
    config.search.records.push_back(physical::CandidateRecord{
        config.search.pass,
        config.search.coordinate_position,
        step_index,
        candidate.implementation,
        candidate.valid
            ? (candidate.verification_mismatch_ignored
                ? "verification_mismatch_ignored" : "ok")
            : "invalid",
        candidate.samples.empty() ? 0.0 : median(candidate.samples),
        static_cast<int>(candidate.samples.size()),
        candidate.detail});
}

int run_synthesis(const Args& args, const alps::Context& context) {
    if (!args.skip_dataset_preflight) {
        gamax::tpch::dist::preflight_dataset(
            args.scale_factor, context.mpi_comm);
    }
    const auto policies =
        gamax::tpch::make_distributed_tpch_source_read_policies();
    auto local = gamax::tpch::dist::build_local_exec_plan(args.query_id);
    cudaDeviceProp properties {};
    cudaGetDeviceProperties(&properties, context.local_device_id);
    const physical::Focus focus{
        args.query_id,
        args.scale_factor,
        context.size,
        properties.name,
        "legacy"};

    physical::Config config;
    if (args.resume && std::filesystem::exists(args.checkpoint)) {
        config = physical::read_json(args.checkpoint);
        physical::validate_focus(config, focus);
        if (config.structure_fingerprint !=
            physical::structure_fingerprint(local)) {
            throw std::runtime_error("resume checkpoint is stale");
        }
        if (config.search.active_step_index) {
            config.search.records.push_back(physical::CandidateRecord{
                config.search.pass,
                config.search.coordinate_position,
                *config.search.active_step_index,
                config.search.active_implementation,
                "crashed",
                0.0,
                0,
                "previous process ended during this candidate"});
            if (context.rank == 0) {
                std::cerr
                    << "RESUME: marking prior active candidate invalid: step "
                    << *config.search.active_step_index << " "
                    << config.search.active_implementation << "\n";
            }
            checkpoint_clear(config, args, context);
        }
    } else {
        config = physical::make_synthesis_seed_config(focus, local);
        config.search.seed = args.seed;
        config.search.max_passes = args.max_passes;
        checkpoint_clear(config, args, context);
    }

    if (config.search.completed) {
        if (context.rank == 0) {
            physical::write_json_atomic(config, args.output_config);
            std::cout << "SYNTHESIS: checkpoint already complete\n";
        }
        return 0;
    }

    const auto coordinates = physical::inner_join_coordinates(local);
    if (coordinates.empty()) {
        config.preset = "synthesized";
        config.search.completed = true;
        config.search.converged = true;
        checkpoint_clear(config, args, context);
        if (context.rank == 0) {
            physical::write_json_atomic(config, args.output_config);
            std::cout << "SYNTHESIS: no inner equijoins; emitted fixed plan\n";
        }
        return 0;
    }

    auto seed_plan = build_plan(args, config, policies);
    auto catalog = gamax::ir::tpch::make_tpch_catalog();
    gamax::tpch::tpch_params params(args.scale_factor);
    params.stream = context.stream;
    params.set_row_group_shard(context.rank, context.size);
    params.set_source_read_policies(policies);
    params.set_file_shard_strategy(args.file_shard_strategy);
    params.set_parquet_decode_threads(args.parquet_decode_threads);
    gamax::tpch::TpchTableReader reader(params);
    gamax::tpch::IrSourceResolver resolver(reader, catalog);
    gamax::exec::SourceResolver source = [&resolver](
        const std::string& name,
        const std::vector<std::string>& columns) {
        return resolver(name, columns);
    };
    auto preload = gamax::exec::bench_resolve_sources(
        seed_plan, source, context.stream);
    MPI_Barrier(context.mpi_comm);
    if (context.rank == 0) {
        std::cout << "LOAD: all source columns resident for Q"
                  << args.query_id << "\n";
    }

    if (config.search.coordinate_order.empty()) {
        while (true) {
            std::vector<double> seed_warmups;
            auto warmup_status = bench::WarmupStatus::Continue;
            RunResult seed_run;
            while (warmup_status == bench::WarmupStatus::Continue) {
                seed_run = run_once(seed_plan, preload, context);
                if (!seed_run.ok) break;
                seed_warmups.push_back(seed_run.maximum_rank_ms);
                warmup_status =
                    bench::evaluate_warmup(seed_warmups, args.warmup);
            }

            if (!seed_run.ok) {
                const auto prior_fallbacks = config.seed_fallbacks.size();
                const bool fell_back =
                    seed_run.failure_kind ==
                        RunFailureKind::PartitionGroupByCapacity &&
                    physical::fallback_partition_groupbys_to_hash(
                        config, "partition_groupby_capacity");
                if (!fell_back) {
                    if (context.rank == 0) {
                        std::cerr
                            << "TERMINAL: synthesis seed failed during "
                               "warmup: "
                            << seed_run.detail << "\n";
                    }
                    return kTerminalSynthesisExitCode;
                }
                checkpoint_clear(config, args, context);
                seed_plan = build_plan(args, config, policies);
                if (context.rank == 0) {
                    for (std::size_t index = prior_fallbacks;
                         index < config.seed_fallbacks.size(); ++index) {
                        const auto& fallback = config.seed_fallbacks[index];
                        std::cerr
                            << "SEED FALLBACK: step "
                            << fallback.step_index << " "
                            << fallback.from_implementation << " -> "
                            << fallback.to_implementation << " reason="
                            << fallback.reason << "\n";
                    }
                }
                continue;
            }

            std::string verification_detail;
            if (!verify(seed_run, args, context, verification_detail)) {
                if (args.continue_on_verification_failure) {
                    if (context.rank == 0) {
                        std::cerr
                            << "WARNING: ignoring synthesis seed verification "
                               "failure: "
                            << verification_detail << "\n";
                    }
                } else if (context.rank == 0) {
                    std::cerr
                        << "TERMINAL: synthesis seed verification failed: "
                        << verification_detail << "\n";
                }
                if (!args.continue_on_verification_failure) {
                    return kTerminalSynthesisExitCode;
                }
            }
            break;
        }
        config.search.coordinate_order = initial_coordinate_order(
            seed_plan, preload, coordinates, context);
        checkpoint_clear(config, args, context);
    }

    bool resumed_mid_pass = config.search.coordinate_position > 0;
    for (int pass = config.search.pass;
         pass < config.search.max_passes; ++pass) {
        config.search.pass = pass;
        bool pass_changed = resumed_mid_pass;
        const int begin_position = resumed_mid_pass
            ? config.search.coordinate_position : 0;
        resumed_mid_pass = false;
        for (int position = begin_position;
             position < static_cast<int>(
                 config.search.coordinate_order.size());
             ++position) {
            config.search.coordinate_position = position;
            const int step_index = config.search.coordinate_order[
                static_cast<std::size_t>(position)];
            const std::string incumbent =
                get_join_variant(config, step_index);

            std::unordered_set<std::string> prior_failures;
            for (const auto& record : config.search.records) {
                if (record.step_index == step_index &&
                    (record.status == "crashed" ||
                     record.status == "invalid")) {
                    prior_failures.insert(record.implementation);
                }
            }

            std::vector<Candidate> candidates;
            for (const auto& implementation : physical::join_candidates()) {
                auto candidate_config = config;
                set_join_variant(
                    candidate_config, step_index, implementation);
                Candidate candidate{
                    implementation,
                    build_plan(args, candidate_config, policies)};
                if (prior_failures.count(implementation)) {
                    candidate.valid = false;
                    candidate.detail =
                        "skipped after an earlier crash or invalid result";
                }
                candidates.push_back(std::move(candidate));
            }

            if (context.rank == 0) {
                std::cout << "PASS " << (pass + 1) << "/"
                          << config.search.max_passes << " coordinate "
                          << (position + 1) << "/"
                          << config.search.coordinate_order.size()
                          << " local-step=" << step_index
                          << " incumbent=" << incumbent << "\n";
            }
            for (auto& candidate : candidates) {
                if (candidate.valid) {
                    warm_candidate(
                        candidate, config, step_index, args, preload,
                        context);
                }
            }
            std::vector<int> all_indices;
            for (std::size_t index = 0; index < candidates.size(); ++index) {
                if (candidates[index].valid) {
                    all_indices.push_back(static_cast<int>(index));
                }
            }
            measure_interleaved(
                candidates, all_indices, args.initial_samples, "initial",
                config, step_index, args, preload, context);

            for (auto& candidate : candidates) {
                if (candidate.valid && !candidate.samples.empty()) {
                    RunResult verification_run;
                    checkpoint_active(
                        config, step_index, candidate.implementation,
                        args, context);
                    verification_run = run_once(
                        candidate.plan, preload, context);
                    std::string detail;
                    if (!verification_run.ok) {
                        candidate.valid = false;
                        candidate.detail = verification_run.detail;
                    } else if (!verify(
                                   verification_run, args, context, detail)) {
                        if (args.continue_on_verification_failure) {
                            candidate.verification_mismatch_ignored = true;
                            candidate.detail =
                                "golden mismatch ignored: " + detail;
                            if (context.rank == 0) {
                                std::cerr
                                    << "WARNING: ignoring verification failure "
                                       "for candidate step "
                                    << step_index << " "
                                    << candidate.implementation << ": "
                                    << detail << "\n";
                            }
                        } else {
                            candidate.valid = false;
                            candidate.detail = "golden mismatch: " + detail;
                        }
                    }
                }
                add_record(config, candidate, step_index);
            }

            auto find_candidate = [&](const std::string& implementation) {
                const auto found = std::find_if(
                    candidates.begin(), candidates.end(),
                    [&](const Candidate& candidate) {
                        return candidate.implementation == implementation;
                    });
                if (found == candidates.end()) {
                    throw std::runtime_error("missing candidate");
                }
                return static_cast<int>(
                    std::distance(candidates.begin(), found));
            };
            const int incumbent_index = find_candidate(incumbent);
            int best_index = -1;
            double best_median = INFINITY;
            for (std::size_t index = 0; index < candidates.size(); ++index) {
                const auto& candidate = candidates[index];
                if (!candidate.valid || candidate.samples.empty()) continue;
                const double value = median(candidate.samples);
                if (value < best_median) {
                    best_median = value;
                    best_index = static_cast<int>(index);
                }
            }
            if (best_index < 0) {
                throw std::runtime_error(
                    "all join candidates are invalid at local step " +
                    std::to_string(step_index));
            }

            int chosen_index = incumbent_index;
            if (!candidates[static_cast<std::size_t>(incumbent_index)].valid) {
                chosen_index = best_index;
            } else if (best_index != incumbent_index) {
                const double incumbent_median = median(
                    candidates[static_cast<std::size_t>(
                        incumbent_index)].samples);
                const auto action = physical::initial_candidate_action(
                    incumbent_median, best_median,
                    args.commit_threshold);
                if (action ==
                    physical::CandidateAction::CommitChallenger) {
                    chosen_index = best_index;
                } else if (
                    action == physical::CandidateAction::RunTieBreak) {
                    measure_interleaved(
                        candidates, {incumbent_index, best_index},
                        args.tie_samples, "tie", config, step_index,
                        args, preload, context);
                    const double tied_incumbent = median(
                        candidates[static_cast<std::size_t>(
                            incumbent_index)].samples);
                    const double tied_challenger = median(
                        candidates[static_cast<std::size_t>(
                            best_index)].samples);
                    if (physical::commit_after_tie_break(
                            tied_incumbent, tied_challenger,
                            args.tie_threshold)) {
                        chosen_index = best_index;
                    }
                }
            }

            const auto& chosen = candidates[
                static_cast<std::size_t>(chosen_index)].implementation;
            if (chosen != incumbent) {
                set_join_variant(config, step_index, chosen);
                pass_changed = true;
            }
            config.search.coordinate_position = position + 1;
            checkpoint_clear(config, args, context);
            if (context.rank == 0) {
                std::cout << "CHOOSE: step " << step_index << " "
                          << incumbent << " -> " << chosen << "\n";
            }
        }

        config.search.pass = pass + 1;
        config.search.coordinate_position = 0;
        if (!pass_changed) {
            config.search.converged = true;
            break;
        }
        checkpoint_clear(config, args, context);
    }

    config.preset = "synthesized";
    config.search.completed = true;
    checkpoint_clear(config, args, context);
    if (context.rank == 0) {
        physical::write_json_atomic(config, args.output_config);
        std::cout << "SYNTHESIS: completed Q" << args.query_id
                  << " converged=" << config.search.converged
                  << " passes=" << config.search.pass << "\n";
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& error) {
        usage(argv[0], error.what());
    }

    alps::configure_process();
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    alps::Context context;
    try {
        if (!args.rmm_resource.empty()) {
            setenv("GAMAX_RMM_RESOURCE", args.rmm_resource.c_str(), 1);
        }
        context = alps::make_context(MPI_COMM_WORLD);
        const int result = run_synthesis(args, context);
        alps::destroy_context(context);
        MPI_Finalize();
        return result;
    } catch (const std::exception& error) {
        std::cerr << "[rank " << rank << "] error: " << error.what() << "\n";
        std::cerr.flush();
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
