// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/context.hpp"
#include "alps/exec/distributed_executor.hpp"
#include "alps/exec/distributed_planner.hpp"
#include "exec/print.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/rules.hpp"
#include "memory/mem_manager.hpp"
#include "phys/phys.hpp"
#include "tpch/ir_source_resolver.hpp"
#include "tpch/distributed_tpch.hpp"
#include "tpch/tpch.hpp"
#include "tpch/tpch_ir.hpp"
#include "tpch/verify.hpp"

namespace {

struct Args {
    int query_id {6};
    int scale_factor {1};
    bool verify {false};
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
    alps::operators::ColumnSchedule column_overlap_schedule {
        alps::operators::ColumnSchedule::Interleave};
    bool column_overlap_high_priority {true};
    alps::collectives::Backend communication_backend {
        alps::collectives::Backend::NCCL};
    std::optional<alps::collectives::Backend> column_exchange_backend;
    alps::collectives::CePullSchedule ce_pull_schedule {
        alps::collectives::CePullSchedule::RoundRobinWaves};
    int ce_pull_slots {2};
};

Args parse_args(int argc, char** argv) {
    Args args;
    if (argc >= 2) {
        args.query_id = std::atoi(argv[1]);
    }
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc) {
            args.scale_factor = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            args.verify = true;
        } else if (std::strcmp(argv[i], "--rmm-resource") == 0 && i + 1 < argc) {
            args.rmm_resource = argv[++i];
            if (args.rmm_resource != "arena" && args.rmm_resource != "cuda") {
                throw std::runtime_error("--rmm-resource must be arena or cuda");
            }
        } else if (std::strcmp(argv[i], "--file-shard-strategy") == 0 &&
                   i + 1 < argc) {
            args.file_shard_strategy =
                gamax::tpch::parse_file_shard_strategy(argv[++i]);
        } else if (std::strcmp(argv[i], "--parquet-decode-threads") == 0 &&
                   i + 1 < argc) {
            args.parquet_decode_threads = std::atoi(argv[++i]);
            if (args.parquet_decode_threads <= 0) {
                throw std::runtime_error(
                    "--parquet-decode-threads must be positive");
            }
        } else if (std::strcmp(argv[i], "--eq-join") == 0 && i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "legacy") {
                args.eq_join_implementation =
                    alps::exec::DistributedEqJoinImplementation::Legacy;
            } else if (value == "column-overlap") {
                args.eq_join_implementation =
                    alps::exec::DistributedEqJoinImplementation::ColumnOverlap;
            } else {
                throw std::runtime_error(
                    "--eq-join must be legacy or column-overlap");
            }
        } else if ((std::strcmp(argv[i], "--group-by") == 0 ||
                    std::strcmp(argv[i], "--semi-join") == 0) &&
                   i + 1 < argc) {
            const bool group_by = std::strcmp(argv[i], "--group-by") == 0;
            const std::string option = argv[i];
            const std::string value = argv[++i];
            alps::exec::DistributedOperatorImplementation implementation;
            if (value == "legacy") {
                implementation =
                    alps::exec::DistributedOperatorImplementation::Legacy;
            } else if (value == "column-overlap") {
                implementation = alps::exec::
                    DistributedOperatorImplementation::ColumnOverlap;
            } else {
                throw std::runtime_error(
                    option + " must be legacy or column-overlap");
            }
            if (group_by) {
                args.group_by_implementation = implementation;
            } else {
                args.semi_join_implementation = implementation;
            }
        } else if (std::strcmp(argv[i], "--column-schedule") == 0 &&
                   i + 1 < argc) {
            const std::string value = argv[++i];
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
                throw std::runtime_error(
                    "--column-schedule must be probe-first, build-first, or interleave");
            }
        } else if (std::strcmp(argv[i], "--column-comm-priority") == 0 &&
                   i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "normal") args.column_overlap_high_priority = false;
            else if (value == "high") args.column_overlap_high_priority = true;
            else throw std::runtime_error(
                "--column-comm-priority must be normal or high");
        } else if (std::strcmp(argv[i], "--communication-backend") == 0 &&
                   i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "nccl") {
                args.communication_backend =
                    alps::collectives::Backend::NCCL;
            } else if (value == "ce-pull") {
                args.communication_backend =
                    alps::collectives::Backend::CE_PULL;
            } else {
                throw std::runtime_error(
                    "--communication-backend must be nccl or ce-pull");
            }
        } else if (std::strcmp(argv[i], "--column-exchange-backend") == 0 &&
                   i + 1 < argc) {
            const std::string value = argv[++i];
            if (value == "inherit") {
                args.column_exchange_backend.reset();
            } else if (value == "nccl") {
                args.column_exchange_backend =
                    alps::collectives::Backend::NCCL;
            } else if (value == "ce-pull") {
                args.column_exchange_backend =
                    alps::collectives::Backend::CE_PULL;
            } else {
                throw std::runtime_error(
                    "--column-exchange-backend must be inherit, nccl, or ce-pull");
            }
        } else if (std::strcmp(argv[i], "--ce-pull-schedule") == 0 &&
                   i + 1 < argc) {
            const std::string value = argv[++i];
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
                throw std::runtime_error(
                    "--ce-pull-schedule must be unordered, rotated, or "
                    "round-robin-waves");
            }
        } else if (std::strcmp(argv[i], "--ce-pull-slots") == 0 &&
                   i + 1 < argc) {
            args.ce_pull_slots = std::atoi(argv[++i]);
            if (args.ce_pull_slots < 1) {
                throw std::runtime_error(
                    "--ce-pull-slots must be positive");
            }
        } else {
            throw std::runtime_error("Unknown arg: " + std::string(argv[i]));
        }
    }
    if (args.query_id < 1 || args.query_id > 22) {
        throw std::runtime_error(
            "run_tpch_ir_mpi supports TPC-H query IDs 1 through 22");
    }
    const bool any_column_overlap =
        args.eq_join_implementation ==
            alps::exec::DistributedEqJoinImplementation::ColumnOverlap ||
        args.group_by_implementation ==
            alps::exec::DistributedGroupByImplementation::ColumnOverlap ||
        args.semi_join_implementation ==
            alps::exec::DistributedSemiJoinImplementation::ColumnOverlap;
    if (args.column_exchange_backend &&
        *args.column_exchange_backend ==
            alps::collectives::Backend::CE_PULL &&
        !any_column_overlap) {
        throw std::runtime_error(
            "--column-exchange-backend ce-pull requires "
            "at least one column-overlap operator");
    }
    const auto resolved_column_exchange_backend =
        args.column_exchange_backend.value_or(args.communication_backend);
    const bool uses_ce_pull =
        args.communication_backend == alps::collectives::Backend::CE_PULL ||
        resolved_column_exchange_backend ==
            alps::collectives::Backend::CE_PULL;
    if (!uses_ce_pull &&
        args.ce_pull_schedule !=
            alps::collectives::CePullSchedule::RoundRobinWaves) {
        throw std::runtime_error(
            "a non-default --ce-pull-schedule requires "
            "--communication-backend ce-pull or "
            "--column-exchange-backend ce-pull");
    }
    if (!uses_ce_pull &&
        args.ce_pull_slots != 2) {
        throw std::runtime_error(
            "a non-default --ce-pull-slots requires "
            "--communication-backend ce-pull or "
            "--column-exchange-backend ce-pull");
    }
    return args;
}

std::string golden_path_for(int query_id, int scale_factor) {
    const char* data_path = std::getenv("TPCH_DATA_PATH");
    if (data_path == nullptr || *data_path == '\0') {
        throw std::runtime_error("--verify requires TPCH_DATA_PATH to be set");
    }
    std::ostringstream path;
    path << data_path << "/sf" << scale_factor << "/golden/q"
         << (query_id < 10 ? "0" : "") << query_id << ".parquet";
    return path.str();
}

std::unique_ptr<gamax::table> execute_distributed_plan(
    const Args& args,
    int rank,
    int world,
    const alps::Context& distributed_context,
    cudaStream_t stream)
{
    auto catalog = gamax::ir::tpch::make_tpch_catalog();
    auto source_read_policies =
        gamax::tpch::make_distributed_tpch_source_read_policies();
    std::string local_join_signatures;
    const bool debug = rank == 0 &&
        std::getenv("GAMAX_DEBUG_DISTRIBUTED_PLAN") != nullptr;
    auto distributed_plan = gamax::tpch::dist::build_exec_plan(
        args.query_id, source_read_policies,
        debug ? &local_join_signatures : nullptr);

    if (debug) {
        std::cout << "-- Local Join Signatures --\n"
                  << local_join_signatures
                  << "-- Distributed ExecPlan --\n"
                  << gamax::exec::format_exec_plan(distributed_plan) << "\n";
    }

    gamax::tpch::tpch_params params(args.scale_factor);
    params.stream = stream;
    params.set_row_group_shard(rank, world);
    params.set_source_read_policies(std::move(source_read_policies));
    params.set_file_shard_strategy(args.file_shard_strategy);
    params.set_parquet_decode_threads(args.parquet_decode_threads);
    gamax::tpch::TpchTableReader reader(params);
    gamax::tpch::IrSourceResolver resolver(reader, catalog);
    gamax::exec::SourceResolver source = [&] (
        const std::string& name,
        const std::vector<std::string>& columns) {
        return resolver(name, columns);
    };

    gamax::phys::Context physical_context{stream};
    alps::exec::DistributedExecOptions exec_options;
    exec_options.backend = args.communication_backend;
    exec_options.eq_join_implementation = args.eq_join_implementation;
    exec_options.group_by_implementation = args.group_by_implementation;
    exec_options.semi_join_implementation = args.semi_join_implementation;
    exec_options.column_overlap_schedule = args.column_overlap_schedule;
    exec_options.column_overlap_high_priority =
        args.column_overlap_high_priority;
    exec_options.column_exchange_backend = args.column_exchange_backend;
    exec_options.ce_pull_schedule = args.ce_pull_schedule;
    exec_options.ce_pull_slots = args.ce_pull_slots;
    auto result = alps::exec::execute_distributed(
        distributed_plan, source, physical_context, distributed_context,
        exec_options);
    cudaStreamSynchronize(stream);
    return result;
}

void print_root_summary(gamax::table_view result, int query_id) {
    std::cout << "Distributed Q" << query_id << " result: "
              << result.num_rows() << " rows x "
              << result.num_columns() << " columns\n";
    if (query_id == 6 && result.num_rows() > 0) {
        auto values = gamax::copy_column_to_host<double>(
            result[0], cudaStreamDefault);
        std::cout << "Q6 total=" << std::setprecision(17) << values[0] << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    alps::Context distributed_context;
    try {
        auto args = parse_args(argc, argv);
        if (!args.rmm_resource.empty()) {
            setenv("GAMAX_RMM_RESOURCE", args.rmm_resource.c_str(), 1);
        }

        distributed_context = alps::make_context(MPI_COMM_WORLD);
        rank = distributed_context.rank;
        world = distributed_context.size;
        cudaStream_t stream = distributed_context.stream;

        if (rank == 0) {
            std::cout << "== Distributed TPC-H Q" << args.query_id
                      << " SF" << args.scale_factor
                      << " ranks=" << world << " ==\n";
        }
        std::cout << "[rank " << rank << "] cuda device "
                  << distributed_context.local_device_id << "\n";

        gamax::tpch::dist::preflight_dataset(
            args.scale_factor, distributed_context.mpi_comm);
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::steady_clock::now();
        auto result = execute_distributed_plan(
            args, rank, world, distributed_context, stream);
        cudaStreamSynchronize(stream);
        MPI_Barrier(MPI_COMM_WORLD);
        auto finish = std::chrono::steady_clock::now();

        int ok = 1;
        if (rank == 0) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                finish - start).count();
            std::cout << "Distributed execution completed in "
                      << elapsed_ms << " ms\n";
            print_root_summary(result->view(), args.query_id);

            if (args.verify) {
                const auto golden = golden_path_for(
                    args.query_id, args.scale_factor);
                std::cout << "VERIFY: comparing against " << golden << "\n";
                auto verification = gamax::tpch::verify_result(
                    result->view(), golden);
                if (verification.ok) {
                    std::cout << "VERIFY: OK\n";
                } else {
                    std::cout << "VERIFY: MISMATCH\n"
                              << verification.diagnostic;
                    ok = 0;
                }
            }
        }

        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        alps::destroy_context(distributed_context);
        MPI_Finalize();
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[rank " << rank << "] error: " << error.what() << "\n";
        std::cerr.flush();
        // A peer may already be blocked in a different MPI or NCCL collective.
        // Entering another cooperative collective here can deadlock the job.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
