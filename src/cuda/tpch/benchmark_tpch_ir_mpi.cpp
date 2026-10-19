// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvtx3/nvToolsExt.h>
#include <unistd.h>

#include "alps/context.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/collectives/hash_partition.hpp"
#include "alps/exec/distributed_executor.hpp"
#include "exec/executor.hpp"
#include "exec/print.hpp"
#include "exec/step_profile.hpp"
#include "memory/mem_manager.hpp"
#include "phys/phys.hpp"
#include "tpch/distributed_benchmark_support.hpp"
#include "tpch/distributed_tpch.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/physical_plan_config.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"
#include "tpch/verify.hpp"
#include "types/type_dispatcher.hpp"

#include "gamax_git_revision.hpp"

namespace {

namespace bench = gamax::tpch::benchmark;

struct Args {
    int query_id {6};
    int scale_factor {1};
    int iterations {10};
    int step_iterations {5};
    bench::WarmupPolicy warmup;
    bool collect_timing {true};
    bool collect_steps {true};
    bool collect_phases {true};
    bool profile {false};
    bool memory_profile {false};
    bool collect_messages {false};
    int message_iterations {1};
    bool verify {true};
    bool collect_on_verify_failure {false};
    bool skip_dataset_preflight {false};
    bool print_plan {false};
    std::string e2e_csv {"distributed_tpch_e2e.csv"};
    std::string steps_csv {"distributed_tpch_steps.csv"};
    std::string phases_csv {"distributed_tpch_phases.csv"};
    std::string plan_file {"distributed_tpch_plans.txt"};
    std::string memory_csv {"distributed_tpch_memory.csv"};
    std::string messages_csv {"distributed_tpch_messages.csv"};
    std::string verification_json;
    std::string physical_plan_config;
    std::string rmm_resource;
    gamax::tpch::FileShardStrategy file_shard_strategy {
        gamax::tpch::FileShardStrategy::RoundRobin};
    int parquet_decode_threads {8};
    alps::exec::DistributedEqJoinImplementation eq_join_implementation {
        alps::exec::DistributedEqJoinImplementation::Legacy};
    alps::exec::DistributedGroupByImplementation group_by_implementation {
        alps::exec::DistributedGroupByImplementation::Legacy};
    alps::exec::DistributedSemiJoinImplementation semi_join_implementation {
        alps::exec::DistributedSemiJoinImplementation::Legacy};
    alps::collectives::AlignmentPolicy shuffle_alignment {
        alps::collectives::AlignmentPolicy::Off};
    alps::collectives::AlignmentPolicy replication_alignment {
        alps::collectives::AlignmentPolicy::Off};
    alps::operators::ColumnSchedule column_overlap_schedule {
        alps::operators::ColumnSchedule::Interleave};
    bool column_overlap_high_priority {true};
    alps::collectives::Backend communication_backend {
        alps::collectives::Backend::NCCL};
    std::optional<alps::collectives::Backend> column_exchange_backend;
    alps::collectives::CePullSchedule ce_pull_schedule {
        alps::collectives::CePullSchedule::RoundRobinWaves};
    int ce_pull_slots {2};
    std::string ce_pull_csv {"distributed_tpch_ce_pull.csv"};
};

const char* eq_join_implementation_name(const Args& args) {
    return args.eq_join_implementation ==
            alps::exec::DistributedEqJoinImplementation::ColumnOverlap
        ? "column-overlap" : "legacy";
}

const char* implementation_name(
    alps::exec::DistributedOperatorImplementation implementation)
{
    return implementation ==
            alps::exec::DistributedOperatorImplementation::ColumnOverlap
        ? "column-overlap" : "legacy";
}

const char* column_schedule_name(const Args& args) {
    switch (args.column_overlap_schedule) {
        case alps::operators::ColumnSchedule::ProbeFirst: return "probe-first";
        case alps::operators::ColumnSchedule::BuildFirst: return "build-first";
        case alps::operators::ColumnSchedule::Interleave: return "interleave";
    }
    return "unknown";
}

const char* column_exchange_backend_name(const Args& args) {
    if (!args.column_exchange_backend) return "inherit";
    return *args.column_exchange_backend == alps::collectives::Backend::CE_PULL
        ? "ce-pull" : "nccl";
}

const char* communication_backend_name(const Args& args) {
    return args.communication_backend == alps::collectives::Backend::CE_PULL
        ? "ce-pull" : "nccl";
}

alps::collectives::Backend resolved_column_exchange_backend(const Args& args) {
    return args.column_exchange_backend.value_or(args.communication_backend);
}

const char* ce_pull_schedule_name(const Args& args) {
    switch (args.ce_pull_schedule) {
        case alps::collectives::CePullSchedule::Unordered:
            return "unordered";
        case alps::collectives::CePullSchedule::RotatedDescriptors:
            return "rotated";
        case alps::collectives::CePullSchedule::RoundRobinWaves:
            return "round-robin-waves";
    }
    return "unknown";
}

const char* alignment_policy_name(
    alps::collectives::AlignmentPolicy policy)
{
    return policy == alps::collectives::AlignmentPolicy::On ? "on" : "off";
}

std::string execution_configuration(
    const Args& args,
    const std::string& physical_plan_preset)
{
    std::ostringstream output;
    output << "eq_join=" << eq_join_implementation_name(args)
           << ";group_by=" << implementation_name(
                  args.group_by_implementation)
           << ";semi_join=" << implementation_name(
                  args.semi_join_implementation)
           << ";shuffle_alignment="
           << alignment_policy_name(args.shuffle_alignment)
           << ";replication_alignment="
           << alignment_policy_name(args.replication_alignment)
           << ";column_schedule=" << column_schedule_name(args)
           << ";column_comm_priority="
           << (args.column_overlap_high_priority ? "high" : "normal")
           << ";communication_backend="
           << communication_backend_name(args)
           << ";column_exchange_backend="
           << column_exchange_backend_name(args)
           << ";ce_pull_schedule=" << ce_pull_schedule_name(args)
           << ";ce_pull_slots=" << args.ce_pull_slots
           << ";shuffle_hash=" << alps::collectives::kShuffleHashName
           << ";file_shard_strategy="
           << gamax::tpch::file_shard_strategy_name(args.file_shard_strategy)
           << ";parquet_decode_threads=" << args.parquet_decode_threads
           << ";physical_plan=" << physical_plan_preset;
    return output.str();
}

[[noreturn]] void usage(
    const char* program,
    const std::string& error = {},
    int exit_code = 2)
{
    auto& output = exit_code == 0 ? std::cout : std::cerr;
    if (!error.empty()) output << "error: " << error << "\n";
    output
        << "usage: " << program << " QUERY_ID [options]\n"
        << "  --scale N               scale factor (default 1)\n"
        << "  --iterations N          end-to-end samples (default 10)\n"
        << "  --step-iterations N     per-step samples (default 5)\n"
        << "  --warmup-min N          minimum warmups (default 3)\n"
        << "  --warmup-window N       stability window (default 5)\n"
        << "  --warmup-threshold X    relative range threshold (default 0.05)\n"
        << "  --warmup-max N          maximum warmups (default 20)\n"
        << "  --timing-only           skip per-step timing pass\n"
        << "  --steps-only            skip clean end-to-end pass\n"
        << "  --profile               warm up, then run one tpch.profile range\n"
        << "  --memory-profile        run one untimed RMM peak measurement\n"
        << "  --messages-only         run exact NCCL API payload tracing only\n"
        << "  --message-iterations N  message-trace samples (default 1)\n"
        << "  --memory-csv PATH       append RMM peak rows\n"
        << "  --messages-csv PATH     append NCCL API payload rows\n"
        << "  --physical-plan-config PATH\n"
        << "                          apply a guarded synthesized qNN JSON plan\n"
        << "  --no-verify             skip golden verification\n"
        << "  --skip-dataset-preflight  skip the rank-0 full dataset scan\n"
        << "  --collect-on-verify-failure\n"
        << "                          finish writing measurements before failing verification\n"
        << "  --verification-json PATH  write the verification verdict and diagnostic\n"
        << "  --e2e-csv PATH          append end-to-end rows\n"
        << "  --steps-csv PATH        append per-step rows\n"
        << "  --phases-csv PATH       append nested compute/communication rows\n"
        << "  --no-phases             omit phase events and phases.csv\n"
        << "  --plan-file PATH        append plan/fingerprint records\n"
        << "  --rmm-resource MODE     arena or cuda\n"
        << "  --file-shard-strategy MODE  round_robin or contiguous_range\n"
        << "  --parquet-decode-threads N  Arrow CPU threads per rank (default 8)\n"
        << "  --eq-join MODE          legacy or column-overlap\n"
        << "  --group-by MODE         legacy or column-overlap\n"
        << "  --semi-join MODE        legacy or column-overlap\n"
        << "  --shuffle-alignment MODE     off or on (default off)\n"
        << "  --replication-alignment MODE off or on (default off)\n"
        << "  --column-schedule MODE  probe-first, build-first, or interleave\n"
        << "  --column-comm-priority MODE  normal or high\n"
        << "  --communication-backend MODE  global bulk-communication backend: nccl or ce-pull\n"
        << "  --column-exchange-backend MODE  ColumnOverlap-only override: inherit, nccl, or ce-pull\n"
        << "  --ce-pull-schedule MODE unordered, rotated, or round-robin-waves\n"
        << "  --ce-pull-slots N       persistent sender-arena slots (default 2)\n"
        << "  --ce-pull-csv PATH      append CE-pull arena/traffic rows\n"
        << "  --print-plan            print final distributed plan on rank 0\n";
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
        if (index + 1 >= argc) usage(argv[0], std::string(option) + " needs a value");
        return argv[++index];
    };
    bool timing_only = false;
    bool steps_only = false;
    for (int i = 2; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--scale") {
            args.scale_factor = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--iterations") {
            args.iterations = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--step-iterations") {
            args.step_iterations = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-min") {
            args.warmup.minimum_runs = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-window") {
            args.warmup.window = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-threshold") {
            args.warmup.relative_range_threshold =
                parse_double(next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--warmup-max") {
            args.warmup.maximum_runs = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--timing-only") {
            timing_only = true;
        } else if (option == "--steps-only") {
            steps_only = true;
        } else if (option == "--profile") {
            args.profile = true;
        } else if (option == "--memory-profile") {
            args.memory_profile = true;
        } else if (option == "--messages-only") {
            args.collect_messages = true;
        } else if (option == "--message-iterations") {
            args.message_iterations = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--no-verify") {
            args.verify = false;
        } else if (option == "--verify") {
            args.verify = true;
        } else if (option == "--collect-on-verify-failure") {
            args.collect_on_verify_failure = true;
        } else if (option == "--skip-dataset-preflight") {
            args.skip_dataset_preflight = true;
        } else if (option == "--verification-json") {
            args.verification_json = next(i, option.c_str());
        } else if (option == "--print-plan") {
            args.print_plan = true;
        } else if (option == "--e2e-csv") {
            args.e2e_csv = next(i, option.c_str());
        } else if (option == "--steps-csv") {
            args.steps_csv = next(i, option.c_str());
        } else if (option == "--phases-csv") {
            args.phases_csv = next(i, option.c_str());
        } else if (option == "--no-phases") {
            args.collect_phases = false;
        } else if (option == "--plan-file") {
            args.plan_file = next(i, option.c_str());
        } else if (option == "--memory-csv") {
            args.memory_csv = next(i, option.c_str());
        } else if (option == "--messages-csv") {
            args.messages_csv = next(i, option.c_str());
        } else if (option == "--physical-plan-config") {
            args.physical_plan_config = next(i, option.c_str());
        } else if (option == "--rmm-resource") {
            args.rmm_resource = next(i, option.c_str());
        } else if (option == "--file-shard-strategy") {
            try {
                args.file_shard_strategy =
                    gamax::tpch::parse_file_shard_strategy(
                        next(i, option.c_str()));
            } catch (const std::exception& error) {
                usage(argv[0], error.what());
            }
        } else if (option == "--parquet-decode-threads") {
            args.parquet_decode_threads = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--eq-join") {
            const std::string value = next(i, option.c_str());
            if (value == "legacy") {
                args.eq_join_implementation =
                    alps::exec::DistributedEqJoinImplementation::Legacy;
            } else if (value == "column-overlap") {
                args.eq_join_implementation =
                    alps::exec::DistributedEqJoinImplementation::ColumnOverlap;
            } else {
                usage(argv[0], "--eq-join must be legacy or column-overlap");
            }
        } else if (option == "--group-by" || option == "--semi-join") {
            const std::string value = next(i, option.c_str());
            alps::exec::DistributedOperatorImplementation implementation;
            if (value == "legacy") {
                implementation =
                    alps::exec::DistributedOperatorImplementation::Legacy;
            } else if (value == "column-overlap") {
                implementation = alps::exec::
                    DistributedOperatorImplementation::ColumnOverlap;
            } else {
                usage(
                    argv[0],
                    option + " must be legacy or column-overlap");
            }
            if (option == "--group-by") {
                args.group_by_implementation = implementation;
            } else {
                args.semi_join_implementation = implementation;
            }
        } else if (
            option == "--shuffle-alignment" ||
            option == "--replication-alignment") {
            const std::string value = next(i, option.c_str());
            alps::collectives::AlignmentPolicy policy;
            if (value == "off") {
                policy = alps::collectives::AlignmentPolicy::Off;
            } else if (value == "on") {
                policy = alps::collectives::AlignmentPolicy::On;
            } else {
                usage(
                    argv[0],
                    option + " must be off or on");
            }
            if (option == "--shuffle-alignment") {
                args.shuffle_alignment = policy;
            } else {
                args.replication_alignment = policy;
            }
        } else if (option == "--column-schedule") {
            const std::string value = next(i, option.c_str());
            if (value == "probe-first") {
                args.column_overlap_schedule =
                    alps::operators::ColumnSchedule::ProbeFirst;
            } else if (value == "build-first") {
                args.column_overlap_schedule =
                    alps::operators::ColumnSchedule::BuildFirst;
            } else if (value == "interleave") {
                args.column_overlap_schedule =
                    alps::operators::ColumnSchedule::Interleave;
            } else {
                usage(argv[0],
                      "--column-schedule must be probe-first, build-first, or interleave");
            }
        } else if (option == "--column-comm-priority") {
            const std::string value = next(i, option.c_str());
            if (value == "normal") args.column_overlap_high_priority = false;
            else if (value == "high") args.column_overlap_high_priority = true;
            else usage(argv[0],
                       "--column-comm-priority must be normal or high");
        } else if (option == "--communication-backend") {
            const std::string value = next(i, option.c_str());
            if (value == "nccl") {
                args.communication_backend =
                    alps::collectives::Backend::NCCL;
            } else if (value == "ce-pull") {
                args.communication_backend =
                    alps::collectives::Backend::CE_PULL;
            } else {
                usage(
                    argv[0],
                    "--communication-backend must be nccl or ce-pull");
            }
        } else if (option == "--column-exchange-backend") {
            const std::string value = next(i, option.c_str());
            if (value == "inherit") {
                args.column_exchange_backend.reset();
            } else if (value == "nccl") {
                args.column_exchange_backend =
                    alps::collectives::Backend::NCCL;
            } else if (value == "ce-pull") {
                args.column_exchange_backend =
                    alps::collectives::Backend::CE_PULL;
            } else {
                usage(
                    argv[0],
                    "--column-exchange-backend must be inherit, nccl, or ce-pull");
            }
        } else if (option == "--ce-pull-schedule") {
            const std::string value = next(i, option.c_str());
            if (value == "unordered") {
                args.ce_pull_schedule =
                    alps::collectives::CePullSchedule::Unordered;
            } else if (value == "rotated") {
                args.ce_pull_schedule =
                    alps::collectives::CePullSchedule::RotatedDescriptors;
            } else if (value == "round-robin-waves") {
                args.ce_pull_schedule =
                    alps::collectives::CePullSchedule::RoundRobinWaves;
            } else {
                usage(
                    argv[0],
                    "--ce-pull-schedule must be unordered, rotated, or "
                    "round-robin-waves");
            }
        } else if (option == "--ce-pull-slots") {
            args.ce_pull_slots = parse_int(
                next(i, option.c_str()), option.c_str(), argv[0]);
        } else if (option == "--ce-pull-csv") {
            args.ce_pull_csv = next(i, option.c_str());
        } else if (option == "--help" || option == "-h") {
            usage(argv[0], {}, 0);
        } else {
            usage(argv[0], "unknown option: " + option);
        }
    }

    if (args.query_id < 1 || args.query_id > 22) {
        usage(argv[0], "QUERY_ID must be in [1, 22]");
    }
    if (args.scale_factor <= 0) usage(argv[0], "--scale must be positive");
    if (args.parquet_decode_threads <= 0) {
        usage(argv[0], "--parquet-decode-threads must be positive");
    }
    if (args.ce_pull_slots < 1) {
        usage(argv[0], "--ce-pull-slots must be positive");
    }
    if (args.iterations < 1 || args.step_iterations < 1 ||
        args.message_iterations < 1) {
        usage(argv[0], "sample counts must be positive");
    }
    if (timing_only && steps_only) {
        usage(argv[0], "--timing-only and --steps-only are mutually exclusive");
    }
    if (args.profile && (timing_only || steps_only)) {
        usage(argv[0], "--profile cannot be combined with timing-pass selectors");
    }
    if (args.memory_profile && (args.profile || timing_only || steps_only)) {
        usage(
            argv[0],
            "--memory-profile cannot be combined with profile/timing selectors");
    }
    if (args.collect_messages &&
        (args.profile || args.memory_profile || timing_only || steps_only)) {
        usage(
            argv[0],
            "--messages-only cannot be combined with other collection modes");
    }
    if (!args.rmm_resource.empty() && args.rmm_resource != "arena" &&
        args.rmm_resource != "cuda") {
        usage(argv[0], "--rmm-resource must be arena or cuda");
    }
    const bool any_column_overlap =
        args.eq_join_implementation ==
            alps::exec::DistributedEqJoinImplementation::ColumnOverlap ||
        args.group_by_implementation ==
            alps::exec::DistributedGroupByImplementation::ColumnOverlap ||
        args.semi_join_implementation ==
            alps::exec::DistributedSemiJoinImplementation::ColumnOverlap;
    if (args.collect_messages &&
        (any_column_overlap ||
         args.communication_backend != alps::collectives::Backend::NCCL ||
         args.column_exchange_backend ||
         args.shuffle_alignment != alps::collectives::AlignmentPolicy::Off ||
         args.replication_alignment != alps::collectives::AlignmentPolicy::Off)) {
        usage(
            argv[0],
            "--messages-only currently requires Legacy operators, the NCCL "
            "backend, and alignment off");
    }
    if (any_column_overlap &&
        args.shuffle_alignment == alps::collectives::AlignmentPolicy::On) {
        usage(
            argv[0],
            "--shuffle-alignment on requires all operator implementations "
            "to be legacy");
    }
    if (args.communication_backend ==
            alps::collectives::Backend::CE_PULL &&
        (args.shuffle_alignment == alps::collectives::AlignmentPolicy::On ||
         args.replication_alignment ==
             alps::collectives::AlignmentPolicy::On)) {
        usage(
            argv[0],
            "--communication-backend ce-pull requires shuffle and "
            "replication alignment to be off");
    }
    if (args.column_exchange_backend &&
        *args.column_exchange_backend ==
            alps::collectives::Backend::CE_PULL &&
        !any_column_overlap) {
        usage(
            argv[0],
            "--column-exchange-backend ce-pull requires "
            "at least one column-overlap operator");
    }
    const bool uses_ce_pull =
        args.communication_backend == alps::collectives::Backend::CE_PULL ||
        resolved_column_exchange_backend(args) ==
            alps::collectives::Backend::CE_PULL;
    if (!uses_ce_pull &&
        args.ce_pull_schedule !=
            alps::collectives::CePullSchedule::RoundRobinWaves) {
        usage(
            argv[0],
            "a non-default --ce-pull-schedule requires "
            "--communication-backend ce-pull or "
            "--column-exchange-backend ce-pull");
    }
    if (!uses_ce_pull &&
        args.ce_pull_slots != 2) {
        usage(
            argv[0],
            "a non-default --ce-pull-slots requires "
            "--communication-backend ce-pull or "
            "--column-exchange-backend ce-pull");
    }
    bench::validate_warmup_policy(args.warmup);
    args.collect_timing =
        !steps_only && !args.profile && !args.memory_profile &&
        !args.collect_messages;
    args.collect_steps =
        !timing_only && !args.profile && !args.memory_profile &&
        !args.collect_messages;
    args.collect_phases = args.collect_steps && args.collect_phases;
    return args;
}

class NvtxRange {
public:
    explicit NvtxRange(const char* label) { nvtxRangePushA(label); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

class NvtxTpchRange {
public:
    NvtxTpchRange() {
        nvtxEventAttributes_t attributes {};
        attributes.version = NVTX_VERSION;
        attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attributes.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        attributes.message.registered = registered_label();
        nvtxDomainRangePushEx(domain(), &attributes);
    }
    ~NvtxTpchRange() { nvtxDomainRangePop(domain()); }
    NvtxTpchRange(const NvtxTpchRange&) = delete;
    NvtxTpchRange& operator=(const NvtxTpchRange&) = delete;

private:
    static nvtxDomainHandle_t domain() {
        static nvtxDomainHandle_t handle =
            nvtxDomainCreateA("gamax.tpch");
        return handle;
    }

    static nvtxStringHandle_t registered_label() {
        static nvtxStringHandle_t handle =
            nvtxDomainRegisterStringA(domain(), "tpch.profile");
        return handle;
    }
};

class MessageTraceBinding {
public:
    MessageTraceBinding(
        const alps::Context& context,
        gamax::exec::StepProfileSink& sink)
        : context_(context)
    {
        if (context_.message_trace_sink) {
            throw std::runtime_error("an NCCL message trace sink is already active");
        }
        context_.message_trace_sink = &sink;
    }
    ~MessageTraceBinding() { context_.message_trace_sink = nullptr; }
    MessageTraceBinding(const MessageTraceBinding&) = delete;
    MessageTraceBinding& operator=(const MessageTraceBinding&) = delete;

private:
    const alps::Context& context_;
};

struct TimedRun {
    std::unique_ptr<gamax::table> result;
    double maximum_rank_ms {0.0};
    std::vector<double> rank_ms;
};

struct E2eSample {
    int iteration {0};
    std::vector<double> rank_ms;
};

struct StepSample {
    int iteration {0};
    int step_index {-1};
    std::string step_kind;
    std::string step_label;
    std::string requested_implementation;
    std::string used_implementation;
    std::string fallback_reason;
    std::vector<double> rank_ms;
    std::vector<double> rank_phase_union_ms;
};

struct PhaseSample {
    int iteration {0};
    int step_index {-1};
    std::string step_kind;
    std::string step_label;
    int phase_index {-1};
    std::string phase_name;
    std::string column_side;
    std::int64_t column_ordinal {-1};
    std::string category;
    std::string communication_pattern;
    std::string collective_primitive;
    std::vector<double> rank_ms;
    std::vector<double> rank_start_ms;
    std::vector<double> rank_end_ms;
    std::vector<std::int64_t> rank_bytes_sent;
    std::vector<std::int64_t> rank_bytes_received;
    std::vector<std::int64_t> rank_rows_sent;
    std::vector<std::int64_t> rank_rows_received;
    std::vector<std::int64_t> rank_peer_count;
    std::vector<std::int64_t> rank_alignment_main_rows;
    std::vector<std::int64_t> rank_alignment_pre_rows;
    std::vector<std::int64_t> rank_alignment_post_rows;
    std::vector<std::int64_t> rank_alignment_main_bytes;
    std::vector<std::int64_t> rank_alignment_pre_bytes;
    std::vector<std::int64_t> rank_alignment_post_bytes;
    std::string quality {"direct"};
};

struct InstrumentedSamples {
    std::vector<StepSample> steps;
    std::vector<PhaseSample> phases;
};

void ensure_parent_directory(const std::string& path);

struct Metadata {
    std::string timestamp;
    std::string plan_fingerprint;
    std::string hostname;
    std::string gpu_name;
    std::string rmm_resource;
    std::string reserve_bytes;
    std::string physical_plan_preset;
    std::string physical_plan_config;
    int nodes {1};
    int warmup_runs {0};
    std::string warmup_status;
};

std::string serialize_message_rows(
    const std::vector<gamax::exec::StepProfileSink::MessageRow>& rows,
    int rank)
{
    std::ostringstream output;
    for (const auto& row : rows) {
        const bool is_self =
            (row.api == gamax::exec::MessageApi::NcclSend ||
             row.api == gamax::exec::MessageApi::NcclRecv) &&
            row.peer == rank;
        output
            << row.iteration << "," << rank << "," << row.call_index << ","
            << row.step_index << "," << row.step_kind_name << ","
            << bench::csv_escape(row.step_label) << ","
            << row.phase_index << ","
            << bench::csv_escape(row.phase_name) << ","
            << gamax::exec::communication_pattern_name(
                   row.communication_pattern) << ","
            << bench::csv_escape(row.collective_primitive) << ","
            << gamax::exec::message_api_name(row.api) << ","
            << gamax::exec::message_direction_name(row.api) << ","
            << row.peer << "," << row.root << ","
            << row.column_ordinal << ","
            << gamax::to_string(row.logical_type) << ","
            << gamax::to_string(row.element_type) << ","
            << gamax::exec::message_component_name(row.component) << ","
            << gamax::exec::message_segment_name(row.segment) << ","
            << row.element_count << "," << row.element_size << ","
            << row.payload_bytes << "," << (is_self ? 1 : 0) << "\n";
    }
    return output.str();
}

std::vector<std::string> gather_rank_text(
    const std::string& local,
    const alps::Context& context)
{
    if (local.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("rank-local message trace exceeds MPI int count");
    }
    const int local_size = static_cast<int>(local.size());
    std::vector<int> sizes;
    if (context.rank == 0) {
        sizes.resize(static_cast<std::size_t>(context.size));
    }
    MPI_Gather(
        &local_size, 1, MPI_INT,
        sizes.empty() ? nullptr : sizes.data(), 1, MPI_INT, 0,
        context.mpi_comm);

    std::vector<int> offsets;
    std::vector<char> gathered;
    if (context.rank == 0) {
        offsets.resize(static_cast<std::size_t>(context.size));
        std::int64_t total = 0;
        for (int rank = 0; rank < context.size; ++rank) {
            if (total > std::numeric_limits<int>::max()) {
                throw std::overflow_error("message trace exceeds MPI_Gatherv limit");
            }
            offsets[static_cast<std::size_t>(rank)] = static_cast<int>(total);
            total += sizes[static_cast<std::size_t>(rank)];
        }
        if (total > std::numeric_limits<int>::max()) {
            throw std::overflow_error("message trace exceeds MPI_Gatherv limit");
        }
        gathered.resize(static_cast<std::size_t>(total));
    }
    MPI_Gatherv(
        local.empty() ? nullptr : local.data(), local_size, MPI_CHAR,
        gathered.empty() ? nullptr : gathered.data(),
        sizes.empty() ? nullptr : sizes.data(),
        offsets.empty() ? nullptr : offsets.data(), MPI_CHAR, 0,
        context.mpi_comm);

    std::vector<std::string> result;
    if (context.rank == 0) {
        result.reserve(static_cast<std::size_t>(context.size));
        for (int rank = 0; rank < context.size; ++rank) {
            const auto index = static_cast<std::size_t>(rank);
            if (sizes[index] == 0) {
                result.emplace_back();
            } else {
                result.emplace_back(
                    gathered.data() + offsets[index],
                    static_cast<std::size_t>(sizes[index]));
            }
        }
    }
    return result;
}

void append_messages_csv(
    const std::string& path,
    const Metadata& metadata,
    const Args& args,
    int world,
    const std::vector<std::string>& rank_rows)
{
    ensure_parent_directory(path);
    const bool header =
        bench::csv_needs_header(path, bench::messages_csv_header());
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot append messages CSV: " + path);
    if (header) output << bench::messages_csv_header() << "\n";

    for (const auto& blob : rank_rows) {
        std::istringstream input(blob);
        std::string row;
        while (std::getline(input, row)) {
            if (row.empty()) continue;
            output
                << metadata.timestamp << ",Q" << args.query_id << ","
                << args.scale_factor << "," << world << "," << metadata.nodes
                << "," << row << "," << metadata.plan_fingerprint << ","
                << GAMAX_GIT_COMMIT << ","
                << bench::csv_escape(metadata.hostname) << ","
                << bench::csv_escape(metadata.gpu_name) << ","
                << bench::csv_escape(metadata.rmm_resource) << ","
                << bench::csv_escape(metadata.reserve_bytes) << ","
                << bench::csv_escape(metadata.physical_plan_preset) << ","
                << bench::csv_escape(metadata.physical_plan_config) << "\n";
        }
    }
}

struct CePullMeasurement {
    std::uint64_t slot_capacity_bytes {0};
    std::uint64_t arena_capacity_bytes {0};
    std::uint64_t control_capacity_bytes {0};
    std::uint64_t peak_requested_slot_bytes {0};
    std::uint64_t growth_count_total {0};
    std::uint64_t growth_count_measured {0};
    std::uint64_t overflow_count_total {0};
    std::uint64_t overflow_count_measured {0};
    std::uint64_t enqueued_columns_total {0};
    std::uint64_t enqueued_columns_measured {0};
    std::uint64_t remote_bytes_total {0};
    std::uint64_t remote_bytes_measured {0};
    std::uint64_t slot_reuse_count_total {0};
    std::uint64_t slot_reuse_count_measured {0};
    double arena_growth_ms_total {0.0};
    double arena_growth_ms_measured {0.0};
    std::uint64_t growth_frozen {0};
};

CePullMeasurement make_ce_pull_measurement(
    const alps::collectives::CePullStats& before,
    const alps::collectives::CePullStats& after)
{
    auto delta = [](std::uint64_t first, std::uint64_t last) {
        return last >= first ? last - first : std::uint64_t{0};
    };
    const auto before_growth_ms = before.arena_growth_ms;
    return CePullMeasurement{
        static_cast<std::uint64_t>(after.slot_capacity_bytes),
        static_cast<std::uint64_t>(after.arena_capacity_bytes),
        static_cast<std::uint64_t>(after.control_capacity_bytes),
        static_cast<std::uint64_t>(after.peak_requested_slot_bytes),
        after.arena_growth_count,
        delta(before.arena_growth_count, after.arena_growth_count),
        after.arena_overflow_count,
        delta(before.arena_overflow_count, after.arena_overflow_count),
        after.enqueued_columns,
        delta(before.enqueued_columns, after.enqueued_columns),
        after.remote_bytes,
        delta(before.remote_bytes, after.remote_bytes),
        after.slot_reuse_count,
        delta(before.slot_reuse_count, after.slot_reuse_count),
        after.arena_growth_ms,
        std::max(0.0, after.arena_growth_ms - before_growth_ms),
        after.growth_frozen ? std::uint64_t{1} : std::uint64_t{0}};
}

std::string getenv_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
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

int count_nodes(MPI_Comm comm, int world) {
    char local[MPI_MAX_PROCESSOR_NAME] {};
    int length = 0;
    MPI_Get_processor_name(local, &length);
    std::vector<char> all(
        static_cast<std::size_t>(world) * MPI_MAX_PROCESSOR_NAME);
    MPI_Allgather(
        local, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
        all.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm);
    std::set<std::string> unique;
    for (int rank = 0; rank < world; ++rank) {
        unique.emplace(all.data() +
            static_cast<std::size_t>(rank) * MPI_MAX_PROCESSOR_NAME);
    }
    return static_cast<int>(unique.size());
}

void ensure_parent_directory(const std::string& path) {
    const std::filesystem::path file(path);
    if (!file.parent_path().empty()) {
        std::filesystem::create_directories(file.parent_path());
    }
}

std::string join_rank_samples(const std::vector<double>& samples) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i != 0) output << ";";
        output << samples[i];
    }
    return output.str();
}

void append_plan(
    const std::string& path,
    const Metadata& metadata,
    const Args& args,
    int world,
    const std::string& plan_text)
{
    ensure_parent_directory(path);
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot append plan file: " + path);
    output << "=== " << metadata.timestamp << " q=" << args.query_id
           << " sf=" << args.scale_factor << " ranks=" << world
           << " fingerprint=" << metadata.plan_fingerprint << " "
           << execution_configuration(args, metadata.physical_plan_preset)
           << " config=" << metadata.physical_plan_config << " ===\n"
           << plan_text << "\n";
}

void append_e2e_csv(
    const std::string& path,
    const Metadata& metadata,
    const Args& args,
    int world,
    const std::vector<E2eSample>& samples)
{
    ensure_parent_directory(path);
    const bool header =
        bench::csv_needs_header(path, bench::e2e_csv_header());
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot append end-to-end CSV: " + path);
    if (header) output << bench::e2e_csv_header() << "\n";
    output << std::fixed << std::setprecision(6);
    for (const auto& sample : samples) {
        const auto stats = bench::summarize(sample.rank_ms);
        output
            << metadata.timestamp << ",Q" << args.query_id << ","
            << args.scale_factor << "," << world << "," << metadata.nodes << ","
            << sample.iteration << "," << metadata.warmup_runs << ","
            << metadata.warmup_status << "," << args.warmup.minimum_runs << ","
            << args.warmup.window << ","
            << args.warmup.relative_range_threshold << ","
            << args.warmup.maximum_runs << ","
            << stats.minimum << "," << stats.mean << "," << stats.maximum << ","
            << bench::csv_escape(join_rank_samples(sample.rank_ms)) << ","
            << metadata.plan_fingerprint << "," << GAMAX_GIT_COMMIT << ","
            << bench::csv_escape(metadata.hostname) << ","
            << bench::csv_escape(metadata.gpu_name) << ","
            << bench::csv_escape(metadata.rmm_resource) << ","
            << bench::csv_escape(metadata.reserve_bytes) << ","
            << bench::csv_escape(metadata.physical_plan_preset) << ","
            << bench::csv_escape(metadata.physical_plan_config) << "\n";
    }
}

void append_steps_csv(
    const std::string& path,
    const Metadata& metadata,
    const Args& args,
    int world,
    const std::vector<StepSample>& samples)
{
    ensure_parent_directory(path);
    const bool header =
        bench::csv_needs_header(path, bench::steps_csv_header());
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot append step CSV: " + path);
    if (header) output << bench::steps_csv_header() << "\n";
    output << std::fixed << std::setprecision(6);
    auto prefix = [&](const StepSample& sample) {
        output
            << metadata.timestamp << ",Q" << args.query_id << ","
            << args.scale_factor << "," << world << "," << metadata.nodes << ","
            << sample.iteration << "," << metadata.warmup_runs << ",";
    };
    auto suffix = [&](const StepSample& sample) {
        output
            << "," << metadata.plan_fingerprint << "," << GAMAX_GIT_COMMIT << ","
            << bench::csv_escape(metadata.hostname) << ","
            << bench::csv_escape(metadata.gpu_name) << ","
            << bench::csv_escape(metadata.rmm_resource) << ","
            << bench::csv_escape(metadata.reserve_bytes) << ","
            << bench::csv_escape(metadata.physical_plan_preset) << ","
            << bench::csv_escape(metadata.physical_plan_config) << "\n";
    };

    for (const auto& sample : samples) {
        for (int rank = 0; rank < world; ++rank) {
            prefix(sample);
            output << "rank," << rank << "," << sample.step_index << ","
                   << sample.step_kind << ","
                   << bench::csv_escape(sample.step_label) << ","
                   << bench::csv_escape(sample.requested_implementation) << ","
                   << bench::csv_escape(sample.used_implementation) << ","
                   << bench::csv_escape(sample.fallback_reason) << ","
                   << sample.rank_ms[static_cast<std::size_t>(rank)] << ",,,";
            suffix(sample);
        }
        const auto stats = bench::summarize(sample.rank_ms);
        prefix(sample);
        output << "aggregate,-1," << sample.step_index << ","
               << sample.step_kind << ","
               << bench::csv_escape(sample.step_label) << ","
               << bench::csv_escape(sample.requested_implementation) << ","
               << bench::csv_escape(sample.used_implementation) << ","
               << bench::csv_escape(sample.fallback_reason) << ",,"
               << stats.minimum << "," << stats.mean << "," << stats.maximum;
        suffix(sample);
    }
}

std::int64_t sum_known(const std::vector<std::int64_t>& values) {
    if (std::any_of(values.begin(), values.end(), [](std::int64_t value) {
            return value < 0;
        })) {
        return -1;
    }
    return std::accumulate(
        values.begin(), values.end(), std::int64_t{0});
}

std::int64_t max_known(const std::vector<std::int64_t>& values) {
    if (values.empty() || std::any_of(
            values.begin(), values.end(), [](std::int64_t value) {
                return value < 0;
            })) {
        return -1;
    }
    return *std::max_element(values.begin(), values.end());
}

void append_phases_csv(
    const std::string& path,
    const Metadata& metadata,
    const Args& args,
    int world,
    const std::vector<PhaseSample>& samples)
{
    ensure_parent_directory(path);
    const bool header =
        bench::csv_needs_header(path, bench::phases_csv_header());
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot append phase CSV: " + path);
    if (header) output << bench::phases_csv_header() << "\n";
    output << std::fixed << std::setprecision(6);
    auto prefix = [&](const PhaseSample& sample) {
        output
            << metadata.timestamp << ",Q" << args.query_id << ","
            << args.scale_factor << "," << world << "," << metadata.nodes << ","
            << sample.iteration << "," << metadata.warmup_runs << ",";
    };
    auto identity = [&](const PhaseSample& sample) {
        output
            << sample.step_index << "," << sample.step_kind << ","
            << bench::csv_escape(sample.step_label) << ","
            << sample.phase_index << ","
            << bench::csv_escape(sample.phase_name) << ","
            << bench::csv_escape(sample.column_side) << ","
            << sample.column_ordinal << ","
            << sample.category << "," << sample.communication_pattern << ","
            << bench::csv_escape(sample.collective_primitive) << ",";
    };
    auto suffix = [&] {
        output
            << "," << metadata.plan_fingerprint << "," << GAMAX_GIT_COMMIT << ","
            << bench::csv_escape(metadata.hostname) << ","
            << bench::csv_escape(metadata.gpu_name) << ","
            << bench::csv_escape(metadata.rmm_resource) << ","
            << bench::csv_escape(metadata.reserve_bytes) << ","
            << bench::csv_escape(metadata.physical_plan_preset) << ","
            << bench::csv_escape(metadata.physical_plan_config) << "\n";
    };

    for (const auto& sample : samples) {
        for (int rank = 0; rank < world; ++rank) {
            const auto index = static_cast<std::size_t>(rank);
            prefix(sample);
            output << "rank," << rank << ",";
            identity(sample);
            output
                << sample.rank_ms[index] << ","
                << sample.rank_start_ms[index] << ","
                << sample.rank_end_ms[index] << ","
                << sample.rank_bytes_sent[index] << ","
                << sample.rank_bytes_received[index] << ","
                << sample.rank_rows_sent[index] << ","
                << sample.rank_rows_received[index] << ","
                << sample.rank_peer_count[index] << ","
                << sample.rank_alignment_main_rows[index] << ","
                << sample.rank_alignment_pre_rows[index] << ","
                << sample.rank_alignment_post_rows[index] << ","
                << sample.rank_alignment_main_bytes[index] << ","
                << sample.rank_alignment_pre_bytes[index] << ","
                << sample.rank_alignment_post_bytes[index] << ",,,,"
                << sample.quality;
            suffix();
        }
        const auto stats = bench::summarize(sample.rank_ms);
        prefix(sample);
        output << "aggregate,-1,";
        identity(sample);
        output
            << ",,," << sum_known(sample.rank_bytes_sent)
            << "," << sum_known(sample.rank_bytes_received)
            << "," << sum_known(sample.rank_rows_sent)
            << "," << sum_known(sample.rank_rows_received)
            << "," << max_known(sample.rank_peer_count)
            << "," << sum_known(sample.rank_alignment_main_rows)
            << "," << sum_known(sample.rank_alignment_pre_rows)
            << "," << sum_known(sample.rank_alignment_post_rows)
            << "," << sum_known(sample.rank_alignment_main_bytes)
            << "," << sum_known(sample.rank_alignment_pre_bytes)
            << "," << sum_known(sample.rank_alignment_post_bytes)
            << "," << stats.minimum << "," << stats.mean << ","
            << stats.maximum << "," << sample.quality;
        suffix();
    }
}

void append_memory_csv(
    const std::string& path,
    const Metadata& metadata,
    const Args& args,
    int world,
    const std::vector<std::int64_t>& resident_current,
    const std::vector<std::int64_t>& resident_peak,
    const std::vector<std::int64_t>& query_current,
    const std::vector<std::int64_t>& query_peak)
{
    ensure_parent_directory(path);
    const bool header =
        bench::csv_needs_header(path, bench::memory_csv_header());
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot append memory CSV: " + path);
    if (header) {
        output << bench::memory_csv_header() << "\n";
    }
    for (int rank = 0; rank < world; ++rank) {
        const auto index = static_cast<std::size_t>(rank);
        output << metadata.timestamp << ",Q" << args.query_id << ","
               << args.scale_factor << "," << world << "," << metadata.nodes
               << ",rank," << rank << "," << resident_current[index] << ","
               << resident_peak[index] << "," << query_current[index] << ","
               << query_peak[index] << ","
               // Three empty max-rank fields require four delimiters between
               // the adjacent total-live-peak and fingerprint values.
               << resident_current[index] + query_peak[index] << ",,,,"
               << metadata.plan_fingerprint << "," << GAMAX_GIT_COMMIT << ","
               << bench::csv_escape(metadata.hostname) << ","
               << bench::csv_escape(metadata.gpu_name) << ","
               << bench::csv_escape(metadata.rmm_resource) << ","
               << bench::csv_escape(metadata.reserve_bytes) << ","
               << bench::csv_escape(metadata.physical_plan_preset) << ","
               << bench::csv_escape(metadata.physical_plan_config) << "\n";
    }
    auto max_index = [](const std::vector<std::int64_t>& values) {
        return static_cast<int>(std::distance(
            values.begin(), std::max_element(values.begin(), values.end())));
    };
    std::vector<std::int64_t> totals(static_cast<std::size_t>(world));
    for (int rank = 0; rank < world; ++rank) {
        const auto index = static_cast<std::size_t>(rank);
        totals[index] = resident_current[index] + query_peak[index];
    }
    const int resident_rank = max_index(resident_current);
    const int query_rank = max_index(query_peak);
    const int total_rank = max_index(totals);
    output << metadata.timestamp << ",Q" << args.query_id << ","
           << args.scale_factor << "," << world << "," << metadata.nodes
           << ",aggregate,-1,"
           << resident_current[static_cast<std::size_t>(resident_rank)] << ","
           << resident_peak[static_cast<std::size_t>(
                  max_index(resident_peak))] << ","
           << query_current[static_cast<std::size_t>(
                  max_index(query_current))] << ","
           << query_peak[static_cast<std::size_t>(query_rank)] << ","
           << totals[static_cast<std::size_t>(total_rank)] << ","
           << resident_rank << "," << query_rank << "," << total_rank << ","
           << metadata.plan_fingerprint << "," << GAMAX_GIT_COMMIT << ","
           << bench::csv_escape(metadata.hostname) << ","
           << bench::csv_escape(metadata.gpu_name) << ","
           << bench::csv_escape(metadata.rmm_resource) << ","
           << bench::csv_escape(metadata.reserve_bytes) << ","
           << bench::csv_escape(metadata.physical_plan_preset) << ","
           << bench::csv_escape(metadata.physical_plan_config) << "\n";
}

void append_ce_pull_csv(
    const std::string& path,
    const Metadata& metadata,
    const Args& args,
    int world,
    const char* measurement_scope,
    const std::vector<CePullMeasurement>& samples)
{
    static const std::string header =
        "timestamp,query,scale_factor,ranks,nodes,scope,rank,"
        "slot_capacity_bytes,arena_capacity_bytes,control_capacity_bytes,"
        "peak_requested_slot_bytes,growth_count_total,"
        "growth_count_measured,overflow_count_total,"
        "overflow_count_measured,enqueued_columns_total,"
        "enqueued_columns_measured,remote_bytes_total,"
        "remote_bytes_measured,slot_reuse_count_total,"
        "slot_reuse_count_measured,arena_growth_ms_total,"
        "arena_growth_ms_measured,growth_frozen,plan_fingerprint,git_commit,"
        "hostname,gpu_name,rmm_resource,reserve_bytes,physical_plan_preset,"
        "physical_plan_config";
    if (samples.size() != static_cast<std::size_t>(world)) {
        throw std::runtime_error("CE-pull sample count does not match ranks");
    }
    ensure_parent_directory(path);
    const bool write_header = bench::csv_needs_header(path, header);
    std::ofstream output(path, std::ios::app);
    if (!output) {
        throw std::runtime_error("cannot append CE-pull CSV: " + path);
    }
    if (write_header) output << header << "\n";
    output << std::fixed << std::setprecision(6);
    for (int rank = 0; rank < world; ++rank) {
        const auto& sample = samples[static_cast<std::size_t>(rank)];
        output
            << metadata.timestamp << ",Q" << args.query_id << ","
            << args.scale_factor << "," << world << "," << metadata.nodes
            << "," << measurement_scope << "," << rank << ","
            << sample.slot_capacity_bytes << ","
            << sample.arena_capacity_bytes << ","
            << sample.control_capacity_bytes << ","
            << sample.peak_requested_slot_bytes << ","
            << sample.growth_count_total << ","
            << sample.growth_count_measured << ","
            << sample.overflow_count_total << ","
            << sample.overflow_count_measured << ","
            << sample.enqueued_columns_total << ","
            << sample.enqueued_columns_measured << ","
            << sample.remote_bytes_total << ","
            << sample.remote_bytes_measured << ","
            << sample.slot_reuse_count_total << ","
            << sample.slot_reuse_count_measured << ","
            << sample.arena_growth_ms_total << ","
            << sample.arena_growth_ms_measured << ","
            << sample.growth_frozen << ","
            << metadata.plan_fingerprint << "," << GAMAX_GIT_COMMIT << ","
            << bench::csv_escape(metadata.hostname) << ","
            << bench::csv_escape(metadata.gpu_name) << ","
            << bench::csv_escape(metadata.rmm_resource) << ","
            << bench::csv_escape(metadata.reserve_bytes) << ","
            << bench::csv_escape(metadata.physical_plan_preset) << ","
            << bench::csv_escape(metadata.physical_plan_config) << "\n";
    }
}

TimedRun run_timed(
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::BenchExecState& preload,
    const alps::Context& distributed_context,
    const alps::exec::DistributedExecOptions& exec_options,
    bool gather_rank_times)
{
    MPI_Barrier(distributed_context.mpi_comm);
    const auto start = std::chrono::steady_clock::now();
    gamax::phys::Context physical_context{distributed_context.stream, nullptr};
    auto result = alps::exec::benchmark_run_distributed_steps(
        plan, preload, physical_context, distributed_context, exec_options);
    cudaStreamSynchronize(distributed_context.stream);
    const auto finish = std::chrono::steady_clock::now();
    const double local_ms =
        std::chrono::duration<double, std::milli>(finish - start).count();

    if (gather_rank_times) {
        auto timing = bench::collect_rank_timing(
            local_ms, distributed_context.mpi_comm);
        return TimedRun{
            std::move(result), timing.maximum_ms, std::move(timing.rank_ms)};
    }
    double maximum_ms = 0.0;
    MPI_Allreduce(
        &local_ms, &maximum_ms, 1, MPI_DOUBLE, MPI_MAX,
        distributed_context.mpi_comm);
    return TimedRun{std::move(result), maximum_ms, {}};
}

std::vector<std::int64_t> collect_rank_counter(
    std::int64_t local_value,
    const alps::Context& distributed_context)
{
    std::vector<std::int64_t> values;
    if (distributed_context.rank == 0) {
        values.resize(static_cast<std::size_t>(distributed_context.size));
    }
    MPI_Gather(
        &local_value, 1, MPI_INT64_T,
        values.empty() ? nullptr : values.data(), 1, MPI_INT64_T, 0,
        distributed_context.mpi_comm);
    return values;
}

std::vector<CePullMeasurement> collect_ce_pull_measurements(
    const alps::collectives::CePullStats& before,
    const alps::collectives::CePullStats& after,
    const alps::Context& distributed_context)
{
    const auto local = make_ce_pull_measurement(before, after);
    std::vector<CePullMeasurement> values;
    if (distributed_context.rank == 0) {
        values.resize(static_cast<std::size_t>(distributed_context.size));
    }
    MPI_Gather(
        &local, static_cast<int>(sizeof(local)), MPI_BYTE,
        values.empty() ? nullptr : values.data(),
        static_cast<int>(sizeof(local)), MPI_BYTE, 0,
        distributed_context.mpi_comm);
    return values;
}

void print_ce_pull_summary(const std::vector<CePullMeasurement>& samples) {
    std::uint64_t max_arena = 0;
    std::uint64_t max_slot = 0;
    std::uint64_t max_control = 0;
    std::uint64_t measured_bytes = 0;
    std::uint64_t measured_columns = 0;
    std::uint64_t measured_growth = 0;
    std::uint64_t measured_overflow = 0;
    for (const auto& sample : samples) {
        max_arena = std::max(max_arena, sample.arena_capacity_bytes);
        max_slot = std::max(max_slot, sample.slot_capacity_bytes);
        max_control = std::max(
            max_control, sample.control_capacity_bytes);
        measured_bytes += sample.remote_bytes_measured;
        measured_columns += sample.enqueued_columns_measured;
        measured_growth = std::max(
            measured_growth, sample.growth_count_measured);
        measured_overflow = std::max(
            measured_overflow, sample.overflow_count_measured);
    }
    std::cout << "CE_PULL: max_rank_arena=" << max_arena
              << " bytes max_rank_slot=" << max_slot
              << " bytes max_rank_control=" << max_control
              << " bytes measured_remote_bytes_sum=" << measured_bytes
              << " measured_columns_sum=" << measured_columns
              << " measured_growth_max=" << measured_growth
              << " measured_overflow_max=" << measured_overflow << "\n";
}

void validate_phase_identity(
    const gamax::exec::StepProfileSink::PhaseRow& row,
    const alps::Context& distributed_context)
{
    std::ostringstream identity;
    identity << row.step_index << ":" << row.step_kind_name << ":"
             << row.phase_index << ":" << row.metadata.name << ":"
             << row.metadata.column_side << ":"
             << row.metadata.column_ordinal << ":"
             << gamax::exec::phase_category_name(row.metadata.category) << ":"
             << gamax::exec::communication_pattern_name(
                    row.metadata.communication_pattern) << ":"
             << row.metadata.collective_primitive;
    const std::uint64_t local = bench::stable_fingerprint(identity.str());
    std::uint64_t minimum = 0;
    std::uint64_t maximum = 0;
    MPI_Allreduce(
        &local, &minimum, 1, MPI_UINT64_T, MPI_MIN,
        distributed_context.mpi_comm);
    MPI_Allreduce(
        &local, &maximum, 1, MPI_UINT64_T, MPI_MAX,
        distributed_context.mpi_comm);
    if (minimum != maximum) {
        throw std::runtime_error(
            "nested phase identity differs across MPI ranks");
    }
}

InstrumentedSamples collect_step_samples(
    const Args& args,
    const gamax::exec::ExecPlan& plan,
    const gamax::exec::BenchExecState& preload,
    const alps::Context& distributed_context,
    const alps::exec::DistributedExecOptions& exec_options,
    std::unique_ptr<gamax::table>& final_result)
{
    InstrumentedSamples collected;
    for (int iteration = 0; iteration < args.step_iterations; ++iteration) {
        const auto step_begin = collected.steps.size();
        const auto phase_begin = collected.phases.size();
        final_result.reset();
        MPI_Barrier(distributed_context.mpi_comm);
        gamax::exec::StepProfileSink sink(args.collect_phases);
        gamax::phys::Context physical_context{
            distributed_context.stream, &sink};
        final_result = alps::exec::benchmark_run_distributed_steps(
            plan, preload, physical_context, distributed_context, exec_options);
        sink.finalize(distributed_context.stream);

        const int local_count = static_cast<int>(sink.rows().size());
        int minimum_count = 0;
        int maximum_count = 0;
        MPI_Allreduce(
            &local_count, &minimum_count, 1, MPI_INT, MPI_MIN,
            distributed_context.mpi_comm);
        MPI_Allreduce(
            &local_count, &maximum_count, 1, MPI_INT, MPI_MAX,
            distributed_context.mpi_comm);
        if (minimum_count != maximum_count) {
            throw std::runtime_error(
                "per-step profile row count differs across ranks");
        }

        for (int row_index = 0; row_index < local_count; ++row_index) {
            const auto& row = sink.rows()[static_cast<std::size_t>(row_index)];
            const double local_ms = row.gpu_ms;
            auto timing = bench::collect_rank_timing(
                local_ms, distributed_context.mpi_comm);
            auto phase_union = bench::collect_rank_timing(
                row.phase_union_ms, distributed_context.mpi_comm);
            if (distributed_context.rank == 0) {
                collected.steps.push_back(StepSample{
                    iteration, row.step_index, row.kind_name, row.label,
                    row.requested_implementation,
                    row.used_implementation,
                    row.fallback_reason,
                    std::move(timing.rank_ms),
                    std::move(phase_union.rank_ms)});
            }
        }

        if (!args.collect_phases) continue;
        const int local_phase_count = static_cast<int>(sink.phase_rows().size());
        int minimum_phase_count = 0;
        int maximum_phase_count = 0;
        MPI_Allreduce(
            &local_phase_count, &minimum_phase_count, 1, MPI_INT, MPI_MIN,
            distributed_context.mpi_comm);
        MPI_Allreduce(
            &local_phase_count, &maximum_phase_count, 1, MPI_INT, MPI_MAX,
            distributed_context.mpi_comm);
        if (minimum_phase_count != maximum_phase_count) {
            throw std::runtime_error(
                "nested phase profile row count differs across ranks");
        }

        for (int row_index = 0; row_index < local_phase_count; ++row_index) {
            const auto& row = sink.phase_rows()[
                static_cast<std::size_t>(row_index)];
            validate_phase_identity(row, distributed_context);
            auto timing = bench::collect_rank_timing(
                row.gpu_ms, distributed_context.mpi_comm);
            auto start_timing = bench::collect_rank_timing(
                row.start_ms, distributed_context.mpi_comm);
            auto end_timing = bench::collect_rank_timing(
                row.end_ms, distributed_context.mpi_comm);
            auto bytes_sent = collect_rank_counter(
                row.metadata.bytes_sent, distributed_context);
            auto bytes_received = collect_rank_counter(
                row.metadata.bytes_received, distributed_context);
            auto rows_sent = collect_rank_counter(
                row.metadata.rows_sent, distributed_context);
            auto rows_received = collect_rank_counter(
                row.metadata.rows_received, distributed_context);
            auto peer_count = collect_rank_counter(
                row.metadata.peer_count, distributed_context);
            auto alignment_main_rows = collect_rank_counter(
                row.metadata.alignment_main_rows, distributed_context);
            auto alignment_pre_rows = collect_rank_counter(
                row.metadata.alignment_pre_rows, distributed_context);
            auto alignment_post_rows = collect_rank_counter(
                row.metadata.alignment_post_rows, distributed_context);
            auto alignment_main_bytes = collect_rank_counter(
                row.metadata.alignment_main_bytes, distributed_context);
            auto alignment_pre_bytes = collect_rank_counter(
                row.metadata.alignment_pre_bytes, distributed_context);
            auto alignment_post_bytes = collect_rank_counter(
                row.metadata.alignment_post_bytes, distributed_context);
            if (distributed_context.rank == 0) {
                collected.phases.push_back(PhaseSample{
                    iteration,
                    row.step_index,
                    row.step_kind_name,
                    row.step_label,
                    row.phase_index,
                    row.metadata.name,
                    row.metadata.column_side,
                    row.metadata.column_ordinal,
                    gamax::exec::phase_category_name(row.metadata.category),
                    gamax::exec::communication_pattern_name(
                        row.metadata.communication_pattern),
                    row.metadata.collective_primitive,
                    std::move(timing.rank_ms),
                    std::move(start_timing.rank_ms),
                    std::move(end_timing.rank_ms),
                    std::move(bytes_sent),
                    std::move(bytes_received),
                    std::move(rows_sent),
                    std::move(rows_received),
                    std::move(peer_count),
                    std::move(alignment_main_rows),
                    std::move(alignment_pre_rows),
                    std::move(alignment_post_rows),
                    std::move(alignment_main_bytes),
                    std::move(alignment_pre_bytes),
                    std::move(alignment_post_bytes),
                    "direct"});
            }
        }

        if (distributed_context.rank == 0) {
            std::map<int, int> next_phase_index;
            for (std::size_t index = phase_begin;
                 index < collected.phases.size(); ++index) {
                const auto& phase = collected.phases[index];
                next_phase_index[phase.step_index] = std::max(
                    next_phase_index[phase.step_index], phase.phase_index + 1);
            }

            for (std::size_t index = step_begin;
                 index < collected.steps.size(); ++index) {
                const auto& step = collected.steps[index];
                auto residual = bench::derive_residual_timing(
                    step.rank_ms, step.rank_phase_union_ms);
                const std::vector<std::int64_t> unknown(
                    static_cast<std::size_t>(distributed_context.size), -1);
                const std::vector<double> unknown_interval(
                    static_cast<std::size_t>(distributed_context.size), -1.0);
                collected.phases.push_back(PhaseSample{
                    iteration,
                    step.step_index,
                    step.step_kind,
                    step.step_label,
                    next_phase_index[step.step_index],
                    "unattributed",
                    "",
                    -1,
                    "overhead",
                    "none",
                    "",
                    std::move(residual.rank_ms),
                    unknown_interval, unknown_interval,
                    unknown, unknown, unknown, unknown, unknown,
                    unknown, unknown, unknown, unknown, unknown, unknown,
                    residual.negative_beyond_tolerance
                        ? "negative_residual_clamped" : "derived"});
            }
        }
    }
    return collected;
}

int run_benchmark(
    const Args& args,
    const alps::Context& distributed_context)
{
    const int rank = distributed_context.rank;
    const int world = distributed_context.size;
    bool verification_failed = false;
    auto json_escape = [](const std::string& value) {
        std::ostringstream escaped;
        escaped << std::hex << std::setfill('0');
        for (const unsigned char ch : value) {
            switch (ch) {
                case '\"': escaped << "\\\""; break;
                case '\\': escaped << "\\\\"; break;
                case '\b': escaped << "\\b"; break;
                case '\f': escaped << "\\f"; break;
                case '\n': escaped << "\\n"; break;
                case '\r': escaped << "\\r"; break;
                case '\t': escaped << "\\t"; break;
                default:
                    if (ch < 0x20) {
                        escaped << "\\u" << std::setw(4)
                                << static_cast<unsigned int>(ch);
                    } else {
                        escaped << static_cast<char>(ch);
                    }
            }
        }
        return escaped.str();
    };
    auto write_verification = [&](bool ok, const std::string& diagnostic) {
        if (rank != 0 || args.verification_json.empty()) return;
        std::ofstream output(args.verification_json, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "could not open verification JSON: " + args.verification_json);
        }
        output << "{\n"
               << "  \"status\": \"" << (ok ? "pass" : "mismatch") << "\",\n"
               << "  \"diagnostic\": \"" << json_escape(diagnostic) << "\"\n"
               << "}\n";
        if (!output) {
            throw std::runtime_error(
                "could not write verification JSON: " + args.verification_json);
        }
    };
    if (!args.skip_dataset_preflight) {
        gamax::tpch::dist::preflight_dataset(
            args.scale_factor, distributed_context.mpi_comm);
    }
    const auto policies =
        gamax::tpch::make_distributed_tpch_source_read_policies();
    cudaDeviceProp gpu_properties {};
    cudaGetDeviceProperties(
        &gpu_properties, distributed_context.local_device_id);
    std::string local_signatures;
    auto local_plan = gamax::tpch::dist::build_local_exec_plan(
        args.query_id,
        rank == 0 && args.print_plan ? &local_signatures : nullptr);
    const gamax::tpch::physical_plan::Focus focus{
        args.query_id,
        args.scale_factor,
        world,
        gpu_properties.name,
        eq_join_implementation_name(args)};
    auto physical_config = args.physical_plan_config.empty()
        ? gamax::tpch::physical_plan::make_stock_config(focus, local_plan)
        : gamax::tpch::physical_plan::read_json(args.physical_plan_config);
    gamax::tpch::physical_plan::validate_focus(physical_config, focus);
    gamax::tpch::physical_plan::apply(local_plan, physical_config);
    auto plan = gamax::tpch::dist::distribute_exec_plan(
        args.query_id, policies, std::move(local_plan));
    const std::string plan_text = gamax::exec::format_exec_plan(plan);
    const std::string execution =
        execution_configuration(args, physical_config.preset);
    const std::string plan_fingerprint = bench::fingerprint_hex(
        bench::stable_fingerprint(plan_text + "\n" + execution));

    if (rank == 0) {
        std::cout << "== Distributed TPC-H benchmark Q" << args.query_id
                  << " SF" << args.scale_factor << " ranks=" << world
                  << " plan=" << plan_fingerprint << " " << execution
                  << " ==\n";
        if (args.print_plan) {
            std::cout << "-- Local Join Signatures --\n" << local_signatures
                      << "-- Distributed ExecPlan --\n" << plan_text << "\n";
        }
    }

    auto catalog = gamax::ir::tpch::make_tpch_catalog();
    gamax::tpch::tpch_params params(args.scale_factor);
    params.stream = distributed_context.stream;
    params.set_row_group_shard(rank, world);
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

    gamax::memory_statistics_sample source_memory;
    std::unique_ptr<gamax::scoped_memory_statistics> source_memory_probe;
    if (args.memory_profile) {
        source_memory_probe =
            std::make_unique<gamax::scoped_memory_statistics>();
    }
    gamax::exec::BenchExecState preload;
    {
        NvtxRange load_range("tpch.load");
        preload = gamax::exec::bench_resolve_sources(
            plan, source, distributed_context.stream);
    }
    if (source_memory_probe) {
        source_memory = source_memory_probe->finish();
        source_memory_probe.reset();
    }
    MPI_Barrier(distributed_context.mpi_comm);
    if (rank == 0) std::cout << "LOAD: all source columns resident\n";

    std::unique_ptr<gamax::table> final_result;
    alps::exec::DistributedExecOptions exec_options;
    exec_options.backend = args.communication_backend;
    exec_options.shuffle_alignment = args.shuffle_alignment;
    exec_options.replication_alignment = args.replication_alignment;
    exec_options.eq_join_implementation = args.eq_join_implementation;
    exec_options.group_by_implementation = args.group_by_implementation;
    exec_options.semi_join_implementation = args.semi_join_implementation;
    exec_options.column_overlap_schedule = args.column_overlap_schedule;
    exec_options.column_overlap_high_priority =
        args.column_overlap_high_priority;
    exec_options.column_exchange_backend = args.column_exchange_backend;
    exec_options.ce_pull_schedule = args.ce_pull_schedule;
    exec_options.ce_pull_slots = args.ce_pull_slots;
    const bool use_ce_pull =
        args.communication_backend == alps::collectives::Backend::CE_PULL ||
        resolved_column_exchange_backend(args) ==
            alps::collectives::Backend::CE_PULL;

    if (args.memory_profile) {
        const auto ce_before = use_ce_pull
            ? alps::collectives::ce_pull_stats(distributed_context)
            : alps::collectives::CePullStats{};
        gamax::scoped_memory_statistics query_probe;
        auto sample = run_timed(
            plan, preload, distributed_context, exec_options, false);
        final_result = std::move(sample.result);
        const auto query_memory = query_probe.finish();
        const auto ce_after = use_ce_pull
            ? alps::collectives::ce_pull_stats(distributed_context)
            : alps::collectives::CePullStats{};
        auto ce_measurements = use_ce_pull
            ? collect_ce_pull_measurements(
                  ce_before, ce_after, distributed_context)
            : std::vector<CePullMeasurement>{};

        int verified = 1;
        if (args.verify && rank == 0) {
            const auto verification = gamax::tpch::verify_result(
                final_result->view(),
                golden_path_for(args.query_id, args.scale_factor));
            if (!verification.ok) {
                verified = 0;
                std::cerr << "VERIFY: MISMATCH\n"
                          << verification.diagnostic;
            } else {
                std::cout << "VERIFY: OK\n";
            }
            write_verification(
                verification.ok, verification.diagnostic);
        }
        MPI_Bcast(&verified, 1, MPI_INT, 0, distributed_context.mpi_comm);
        verification_failed = !verified;
        if (verification_failed && !args.collect_on_verify_failure) return 1;
        if (verification_failed && rank == 0) {
            std::cerr << "VERIFY: continuing memory data collection after mismatch\n";
        }

        auto resident_current = collect_rank_counter(
            source_memory.bytes.current, distributed_context);
        auto resident_peak = collect_rank_counter(
            source_memory.bytes.peak, distributed_context);
        auto query_current = collect_rank_counter(
            query_memory.bytes.current, distributed_context);
        auto query_peak = collect_rank_counter(
            query_memory.bytes.peak, distributed_context);
        const int nodes = count_nodes(distributed_context.mpi_comm, world);
        if (rank == 0) {
            Metadata metadata;
            metadata.timestamp = bench::utc_timestamp();
            metadata.plan_fingerprint = plan_fingerprint;
            char hostname[256] {};
            gethostname(hostname, sizeof(hostname) - 1);
            metadata.hostname = hostname;
            metadata.gpu_name = gpu_properties.name;
            metadata.rmm_resource = args.rmm_resource.empty()
                ? getenv_or("GAMAX_RMM_RESOURCE", "default")
                : args.rmm_resource;
            metadata.reserve_bytes =
                getenv_or("GAMAX_MEM_RESERVE_BYTES", "default");
            metadata.physical_plan_preset = physical_config.preset;
            metadata.physical_plan_config =
                args.physical_plan_config.empty()
                    ? "<implicit-stock>"
                    : args.physical_plan_config;
            metadata.nodes = nodes;
            append_plan(args.plan_file, metadata, args, world, plan_text);
            append_memory_csv(
                args.memory_csv, metadata, args, world,
                resident_current, resident_peak, query_current, query_peak);
            if (use_ce_pull) {
                append_ce_pull_csv(
                    args.ce_pull_csv, metadata, args, world,
                    "memory_profile", ce_measurements);
                print_ce_pull_summary(ce_measurements);
            }
            const auto max_total = [&] {
                std::int64_t value = 0;
                for (int index = 0; index < world; ++index) {
                    value = std::max(
                        value,
                        resident_current[static_cast<std::size_t>(index)] +
                        query_peak[static_cast<std::size_t>(index)]);
                }
                return value;
            }();
            std::cout << "MEMORY: max total live peak=" << max_total
                      << " bytes\n";
        }
        return verification_failed ? 1 : 0;
    }

    std::vector<double> warmup_samples;
    bench::WarmupStatus warmup_status = bench::WarmupStatus::Continue;
    while (warmup_status == bench::WarmupStatus::Continue) {
        final_result.reset();
        auto sample = run_timed(
            plan, preload, distributed_context, exec_options, false);
        warmup_samples.push_back(sample.maximum_rank_ms);
        final_result = std::move(sample.result);
        warmup_status = bench::evaluate_warmup(
            warmup_samples, args.warmup);
    }
    const bool warmup_stable = warmup_status == bench::WarmupStatus::Stable;
    if (rank == 0) {
        std::cout << "WARMUP: " << warmup_samples.size() << " runs, "
                  << (warmup_stable ? "stable" : "maximum reached") << "\n";
        if (!warmup_stable) {
            std::cerr << "WARMUP WARNING: stability threshold was not reached; "
                         "continuing after the configured cap\n";
        }
    }

    alps::collectives::CePullStats ce_frozen_stats;
    if (use_ce_pull) {
        alps::collectives::freeze_ce_pull_arena(distributed_context);
        ce_frozen_stats =
            alps::collectives::ce_pull_stats(distributed_context);
        // A requested CE backend can legitimately see no traffic: a query may
        // have no equality join, or all of its equality joins may use a
        // broadcast strategy. Freezing an empty arena is still a valid steady
        // state and is reported as zero traffic in the CE CSV.
        const int local_ready =
            (ce_frozen_stats.growth_frozen ||
             ce_frozen_stats.enqueued_columns == 0)
            ? 1 : 0;
        int all_ready = 0;
        MPI_Allreduce(
            &local_ready, &all_ready, 1, MPI_INT, MPI_MIN,
            distributed_context.mpi_comm);
        if (!all_ready) {
            throw std::runtime_error(
                "CE-pull was requested but warmup did not freeze its arena "
                "on every rank");
        }
    }

    std::vector<E2eSample> e2e_samples;
    std::vector<StepSample> step_samples;
    std::vector<PhaseSample> phase_samples;
    std::vector<gamax::exec::StepProfileSink::MessageRow> message_rows;
    if (args.collect_messages) {
        for (int iteration = 0; iteration < args.message_iterations; ++iteration) {
            final_result.reset();
            MPI_Barrier(distributed_context.mpi_comm);
            gamax::exec::StepProfileSink sink(
                /*collect_phases=*/true,
                /*collect_messages=*/true,
                iteration);
            gamax::phys::Context physical_context{
                distributed_context.stream, &sink};
            {
                MessageTraceBinding binding(distributed_context, sink);
                final_result = alps::exec::benchmark_run_distributed_steps(
                    plan, preload, physical_context, distributed_context,
                    exec_options);
            }
            sink.finalize(distributed_context.stream);
            message_rows.insert(
                message_rows.end(), sink.message_rows().begin(),
                sink.message_rows().end());
        }
    } else if (args.profile) {
        final_result.reset();
        MPI_Barrier(distributed_context.mpi_comm);
        {
            NvtxTpchRange profile_range;
            gamax::phys::Context physical_context{
                distributed_context.stream, nullptr,
                /*emit_phase_nvtx=*/true};
            final_result = alps::exec::benchmark_run_distributed_steps(
                plan, preload, physical_context, distributed_context,
                exec_options);
            cudaStreamSynchronize(distributed_context.stream);
        }
        if (rank == 0) std::cout << "PROFILE: captured one tpch.profile run\n";
    } else {
        if (args.collect_timing) {
            for (int iteration = 0; iteration < args.iterations; ++iteration) {
                final_result.reset();
                auto sample = run_timed(
                    plan, preload, distributed_context, exec_options, true);
                if (rank == 0) {
                    e2e_samples.push_back(E2eSample{
                        iteration, std::move(sample.rank_ms)});
                }
                final_result = std::move(sample.result);
            }
        }
        if (args.collect_steps) {
            auto instrumented = collect_step_samples(
                args, plan, preload, distributed_context, exec_options,
                final_result);
            step_samples = std::move(instrumented.steps);
            phase_samples = std::move(instrumented.phases);
        }
    }

    int verified = 1;
    if (args.verify && rank == 0) {
        const auto golden = golden_path_for(
            args.query_id, args.scale_factor);
        const auto verification = gamax::tpch::verify_result(
            final_result->view(), golden);
        if (!verification.ok) {
            verified = 0;
            std::cerr << "VERIFY: MISMATCH\n" << verification.diagnostic;
        } else {
            std::cout << "VERIFY: OK\n";
        }
        write_verification(verification.ok, verification.diagnostic);
    }
    MPI_Bcast(&verified, 1, MPI_INT, 0, distributed_context.mpi_comm);
    verification_failed = !verified;
    if (verification_failed && !args.collect_on_verify_failure) return 1;
    if (verification_failed && rank == 0) {
        std::cerr << "VERIFY: continuing data collection after mismatch\n";
    }

    std::vector<std::string> message_rank_rows;
    if (args.collect_messages) {
        message_rank_rows = gather_rank_text(
            serialize_message_rows(message_rows, rank), distributed_context);
    }

    std::vector<CePullMeasurement> ce_measurements;
    if (use_ce_pull) {
        const auto ce_final_stats =
            alps::collectives::ce_pull_stats(distributed_context);
        const bool no_ce_activity =
            ce_frozen_stats.enqueued_columns == 0 &&
            ce_final_stats.enqueued_columns == 0;
        const int local_steady_state =
            (no_ce_activity ||
             (ce_final_stats.growth_frozen &&
              ce_final_stats.arena_growth_count ==
                  ce_frozen_stats.arena_growth_count &&
              ce_final_stats.arena_overflow_count == 0 &&
              ce_final_stats.slot_capacity_bytes ==
                  ce_frozen_stats.slot_capacity_bytes &&
              ce_final_stats.arena_capacity_bytes ==
                  ce_frozen_stats.arena_capacity_bytes &&
              ce_final_stats.control_capacity_bytes ==
                  ce_frozen_stats.control_capacity_bytes))
            ? 1 : 0;
        int all_steady_state = 0;
        MPI_Allreduce(
            &local_steady_state, &all_steady_state, 1, MPI_INT, MPI_MIN,
            distributed_context.mpi_comm);
        ce_measurements = collect_ce_pull_measurements(
            ce_frozen_stats, ce_final_stats, distributed_context);
        if (rank == 0) print_ce_pull_summary(ce_measurements);
        if (!all_steady_state) {
            throw std::runtime_error(
                "CE-pull arena grew, overflowed, or changed capacity after "
                "the warmup watermark was frozen");
        }
    }

    if (!args.profile && rank == 0) {
        Metadata metadata;
        metadata.timestamp = bench::utc_timestamp();
        metadata.plan_fingerprint = plan_fingerprint;
        char hostname[256] {};
        gethostname(hostname, sizeof(hostname) - 1);
        metadata.hostname = hostname;
        metadata.gpu_name = gpu_properties.name;
        metadata.rmm_resource = args.rmm_resource.empty()
            ? getenv_or("GAMAX_RMM_RESOURCE", "default")
            : args.rmm_resource;
        metadata.reserve_bytes = getenv_or("GAMAX_MEM_RESERVE_BYTES", "default");
        metadata.physical_plan_preset = physical_config.preset;
        metadata.physical_plan_config = args.physical_plan_config.empty()
            ? "<implicit-stock>"
            : args.physical_plan_config;
        metadata.nodes = count_nodes(distributed_context.mpi_comm, world);
        metadata.warmup_runs = static_cast<int>(warmup_samples.size());
        metadata.warmup_status = warmup_stable ? "stable" : "maximum_reached";

        append_plan(args.plan_file, metadata, args, world, plan_text);
        if (args.collect_messages) {
            append_messages_csv(
                args.messages_csv, metadata, args, world, message_rank_rows);
            std::size_t records = 0;
            for (const auto& rows : message_rank_rows) {
                records += static_cast<std::size_t>(
                    std::count(rows.begin(), rows.end(), '\n'));
            }
            std::cout << "MESSAGES: " << args.message_iterations
                      << " samples, " << records
                      << " NCCL API calls\n";
        }
        if (use_ce_pull) {
            append_ce_pull_csv(
                args.ce_pull_csv, metadata, args, world,
                "steady_state", ce_measurements);
        }
        if (args.collect_timing) {
            append_e2e_csv(args.e2e_csv, metadata, args, world, e2e_samples);
            std::vector<double> maximums;
            maximums.reserve(e2e_samples.size());
            for (const auto& sample : e2e_samples) {
                maximums.push_back(*std::max_element(
                    sample.rank_ms.begin(), sample.rank_ms.end()));
            }
            const auto stats = bench::summarize(maximums);
            std::cout << std::fixed << std::setprecision(3)
                      << "E2E: n=" << maximums.size()
                      << " median=" << stats.median << " ms"
                      << " min=" << stats.minimum << " ms"
                      << " max=" << stats.maximum << " ms"
                      << " stddev=" << stats.stddev << " ms\n";
        }
        if (args.collect_steps) {
            append_steps_csv(
                args.steps_csv, metadata, args, world, step_samples);
            std::cout << "STEPS: " << args.step_iterations
                      << " samples, " << step_samples.size()
                      << " aggregate step observations\n";
            if (args.collect_phases) {
                append_phases_csv(
                    args.phases_csv, metadata, args, world, phase_samples);
                std::cout << "PHASES: " << phase_samples.size()
                          << " direct and derived observations\n";
            }
        }
    } else if (!args.profile) {
        // count_nodes is collective and rank 0 calls it while constructing
        // metadata, so non-root ranks must participate as well.
        (void)count_nodes(distributed_context.mpi_comm, world);
    }
    return verification_failed ? 1 : 0;
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
    alps::Context distributed_context;
    try {
        if (!args.rmm_resource.empty()) {
            setenv("GAMAX_RMM_RESOURCE", args.rmm_resource.c_str(), 1);
        }
        if (args.memory_profile) {
            setenv("GAMAX_RMM_STATISTICS", "on", 1);
        }
        distributed_context = alps::make_context(MPI_COMM_WORLD);
        const int result = run_benchmark(args, distributed_context);
        alps::destroy_context(distributed_context);
        MPI_Finalize();
        return result;
    } catch (const std::exception& error) {
        std::cerr << "[rank " << rank << "] error: " << error.what() << "\n";
        std::cerr.flush();
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
