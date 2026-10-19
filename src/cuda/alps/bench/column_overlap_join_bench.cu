// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/operators/column_overlap_eq_join.hpp"
#include "alps/operators/distributed_relational.hpp"
#include "exec/exec_plan.hpp"
#include "memory/column.hpp"
#include "memory/mem_manager.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "phys/phys.hpp"

namespace {

using gamax::size_type;

struct Shape {
    const char* name;
    int64_t left_rows_per_rank;
    int64_t right_rows_per_rank;
    int left_columns;
    int right_columns;
};

constexpr Shape kShapes[] = {
    {"q5", 17066854, 449997273, 4, 4},
    {"q8", 6836383, 449997273, 8, 5},
    {"q9", 815767, 449997273, 1, 6},
    {"narrow", 17066854, 449997273, 1, 1},
};

struct Args {
    std::string shape {"q5"};
    std::string variant {"all"};
    int64_t divisor {64};
    int warmups {1};
    int iterations {3};
    bool memory_statistics {false};
    bool stage_timing {false};
    bool communication_high_priority {false};
    bool early_reclaim {false};
};

int64_t parse_positive(const char* text, const char* option) {
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value <= 0) {
        throw std::invalid_argument(std::string("invalid ") + option);
    }
    return value;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument("missing value for " + option);
            }
            return argv[index];
        };
        if (option == "--shape") args.shape = value();
        else if (option == "--variant") args.variant = value();
        else if (option == "--divisor") {
            args.divisor = parse_positive(value(), "--divisor");
        } else if (option == "--warmups") {
            args.warmups = static_cast<int>(parse_positive(value(), "--warmups"));
        } else if (option == "--iterations") {
            args.iterations = static_cast<int>(parse_positive(value(), "--iterations"));
        } else if (option == "--memory-statistics") {
            args.memory_statistics = true;
        } else if (option == "--stage-timing") {
            args.stage_timing = true;
        } else if (option == "--comm-priority") {
            const std::string priority = value();
            if (priority == "normal") args.communication_high_priority = false;
            else if (priority == "high") args.communication_high_priority = true;
            else throw std::invalid_argument(
                "--comm-priority must be normal or high");
        } else if (option == "--early-reclaim") {
            args.early_reclaim = true;
        } else if (option == "--help") {
            std::printf(
                "Usage: column_overlap_join_bench [--shape q5|q8|q9|narrow] "
                "[--variant all|legacy|probe_first|build_first|interleave] "
                "[--divisor N] [--warmups N] [--iterations N] "
                "[--memory-statistics] [--stage-timing] "
                "[--comm-priority normal|high] "
                "[--early-reclaim]\n");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return args;
}

Shape resolve_shape(const Args& args) {
    for (auto shape : kShapes) {
        if (args.shape == shape.name) {
            shape.left_rows_per_rank =
                std::max<int64_t>(1, shape.left_rows_per_rank / args.divisor);
            shape.right_rows_per_rank =
                std::max<int64_t>(1, shape.right_rows_per_rank / args.divisor);
            return shape;
        }
    }
    throw std::invalid_argument("unknown shape: " + args.shape);
}

__global__ void fill_build(int64_t* output, int64_t rows, int64_t base) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows) output[row] = base + row;
}

__global__ void fill_probe(
    int64_t* output,
    int64_t rows,
    int64_t global_offset,
    int64_t global_build_rows)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows) {
        const auto position = static_cast<uint64_t>(global_offset + row);
        output[row] = static_cast<int64_t>(
            (position * 11400714819323198485ull + 7046029254386353131ull) %
            static_cast<uint64_t>(global_build_rows));
    }
}

__global__ void compare_column(
    const int64_t* key,
    const int64_t* payload,
    int64_t rows,
    int* mismatch)
{
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows && key[row] != payload[row]) atomicExch(mismatch, 1);
}

std::unique_ptr<gamax::table> make_input(
    int columns,
    int64_t rows,
    int rank,
    int world,
    bool build,
    int64_t build_rows_per_rank,
    int64_t probe_rows_per_rank,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> result;
    result.reserve(static_cast<std::size_t>(columns));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    for (int column = 0; column < columns; ++column) {
        auto values = gamax::make_column<int64_t>(rows, stream);
        if (rows > 0) {
            if (build) {
                fill_build<<<blocks, threads, 0, stream>>>(
                    values->data<int64_t>(), rows,
                    static_cast<int64_t>(rank) * build_rows_per_rank);
            } else {
                fill_probe<<<blocks, threads, 0, stream>>>(
                    values->data<int64_t>(), rows,
                    static_cast<int64_t>(rank) * probe_rows_per_rank,
                    static_cast<int64_t>(world) * build_rows_per_rank);
            }
            ALPS_CUDACHECK(cudaGetLastError());
        }
        result.push_back(std::move(values));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(result));
}

bool validate_output(gamax::table_view output, cudaStream_t stream) {
    if (output.num_columns() == 0) return false;
    auto mismatch = gamax::make_column<int32_t>(1, stream, gamax::buffer_init::zero);
    const auto rows = output.num_rows();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((rows + threads - 1) / threads);
    for (size_type column = 1; column < output.num_columns(); ++column) {
        if (rows > 0) {
            compare_column<<<blocks, threads, 0, stream>>>(
                output.get_column(0).data<int64_t>(),
                output.get_column(column).data<int64_t>(),
                rows, mismatch->data<int32_t>());
            ALPS_CUDACHECK(cudaGetLastError());
        }
    }
    int host_mismatch = 0;
    ALPS_CUDACHECK(cudaMemcpyAsync(
        &host_mismatch, mismatch->data<int32_t>(), sizeof(host_mismatch),
        cudaMemcpyDeviceToHost, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return host_mismatch == 0;
}

struct Variant {
    const char* name;
    bool legacy;
    alps::operators::ColumnSchedule schedule;
};

std::vector<Variant> variants(const Args& args) {
    using Schedule = alps::operators::ColumnSchedule;
    const std::vector<Variant> all = {
        {"legacy", true, Schedule::ProbeFirst},
        {"probe_first", false, Schedule::ProbeFirst},
        {"build_first", false, Schedule::BuildFirst},
        {"interleave", false, Schedule::Interleave},
    };
    if (args.variant == "all") return all;
    for (const auto& variant : all) {
        if (args.variant == variant.name) return {variant};
    }
    throw std::invalid_argument("unknown variant: " + args.variant);
}

std::unique_ptr<gamax::table> run_variant(
    const Variant& variant,
    gamax::table_view left,
    gamax::table_view right,
    const gamax::join::join_hint& hint,
    const alps::Context& context,
    alps::operators::ColumnOverlapTiming* timing,
    const Args& args)
{
    if (!variant.legacy) {
        return alps::operators::column_overlap_eq_join(
            left, 0, right, 0, hint, context, context.stream,
            true, true, variant.schedule, timing, nullptr,
            args.communication_high_priority, args.early_reclaim);
    }
    const gamax::phys::Context physical{context.stream};
    return alps::operators::eq_join(
        left, {0}, right, {0},
        gamax::exec::JoinDistributionStrategy::Shuffle,
        true, true, false, false, hint, context, physical,
        alps::collectives::Backend::NCCL);
}

void print_header(int rank) {
    if (rank == 0) {
        std::printf(
            "shape,ranks,left_rows_per_rank,right_rows_per_rank,left_columns,"
            "right_columns,variant,comm_priority,early_reclaim,iteration,"
            "wall_ms,total_gpu_ms,"
            "key_prepare_ms,payload_prepare_ms,key_exchange_ms,"
            "payload_exchange_ms,match_ms,materialize_ms,peak_bytes,"
            "global_output_rows\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
        if (args.memory_statistics) {
            if (setenv("GAMAX_RMM_STATISTICS", "on", 1) != 0) {
                throw std::runtime_error("failed to enable allocator statistics");
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    alps::configure_process();
    MPI_Init(&argc, &argv);
    auto context = alps::make_context();

    int result = 0;
    try {
        const auto shape = resolve_shape(args);
        auto left = make_input(
            shape.left_columns, shape.left_rows_per_rank,
            context.rank, context.size, true,
            shape.left_rows_per_rank, shape.right_rows_per_rank,
            context.stream);
        auto right = make_input(
            shape.right_columns, shape.right_rows_per_rank,
            context.rank, context.size, false,
            shape.left_rows_per_rank, shape.right_rows_per_rank,
            context.stream);

        gamax::join::join_hint hint;
        hint.join_algo = gamax::join::HASH;
        hint.mat_algo = gamax::join::GFUR;
        hint.unique_left = true;
        const auto selected = variants(args);
        print_header(context.rank);

        for (const auto& variant : selected) {
            for (int warmup = 0; warmup < args.warmups; ++warmup) {
                ALPS_MPICHECK(MPI_Barrier(context.mpi_comm));
                auto output = run_variant(
                    variant, left->view(), right->view(), hint,
                    context, nullptr, args);
                ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
            }

            for (int iteration = 0; iteration < args.iterations; ++iteration) {
                alps::operators::ColumnOverlapTiming timing;
                ALPS_MPICHECK(MPI_Barrier(context.mpi_comm));
                std::unique_ptr<gamax::scoped_memory_statistics> memory_probe;
                if (args.memory_statistics) {
                    memory_probe =
                        std::make_unique<gamax::scoped_memory_statistics>();
                }
                const double start = MPI_Wtime();
                auto output = run_variant(
                    variant, left->view(), right->view(), hint,
                    context,
                    !variant.legacy && args.stage_timing ? &timing : nullptr,
                    args);
                ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
                const double local_wall_ms = (MPI_Wtime() - start) * 1000.0;
                const int64_t local_peak_bytes = memory_probe
                    ? memory_probe->finish().bytes.peak
                    : -1;
                double wall_ms = 0.0;
                ALPS_MPICHECK(MPI_Reduce(
                    &local_wall_ms, &wall_ms, 1, MPI_DOUBLE, MPI_MAX,
                    0, context.mpi_comm));
                int64_t peak_bytes = -1;
                ALPS_MPICHECK(MPI_Reduce(
                    &local_peak_bytes, &peak_bytes, 1, MPI_LONG_LONG, MPI_MAX,
                    0, context.mpi_comm));

                std::array<float, 7> local_stages = {
                    timing.total_ms,
                    timing.key_prepare_ms,
                    timing.payload_prepare_ms,
                    timing.key_exchange_ms,
                    timing.payload_exchange_ms,
                    timing.match_ms,
                    timing.materialize_ms};
                std::array<float, 7> stages{};
                ALPS_MPICHECK(MPI_Reduce(
                    local_stages.data(), stages.data(),
                    static_cast<int>(stages.size()), MPI_FLOAT, MPI_MAX,
                    0, context.mpi_comm));

                const int64_t local_rows = output->num_rows();
                int64_t global_rows = 0;
                ALPS_MPICHECK(MPI_Reduce(
                    &local_rows, &global_rows, 1, MPI_LONG_LONG, MPI_SUM,
                    0, context.mpi_comm));
                int local_valid = validate_output(output->view(), context.stream) ? 1 : 0;
                int global_valid = 0;
                ALPS_MPICHECK(MPI_Allreduce(
                    &local_valid, &global_valid, 1, MPI_INT, MPI_MIN,
                    context.mpi_comm));
                const int64_t expected_rows =
                    shape.right_rows_per_rank * context.size;
                if (!global_valid || (context.rank == 0 && global_rows != expected_rows)) {
                    throw std::runtime_error(
                        std::string("output validation failed for ") + variant.name);
                }

                if (context.rank == 0) {
                    std::printf(
                        "%s,%d,%lld,%lld,%d,%d,%s,%s,%d,%d,%.6f,%.6f,%.6f,"
                        "%.6f,%.6f,%.6f,%.6f,%.6f,%lld,%lld\n",
                        shape.name, context.size,
                        static_cast<long long>(shape.left_rows_per_rank),
                        static_cast<long long>(shape.right_rows_per_rank),
                        shape.left_columns, shape.right_columns,
                        variant.name,
                        args.communication_high_priority ? "high" : "normal",
                        args.early_reclaim ? 1 : 0,
                        iteration, wall_ms,
                        stages[0], stages[1], stages[2], stages[3],
                        stages[4], stages[5], stages[6],
                        static_cast<long long>(peak_bytes),
                        static_cast<long long>(global_rows));
                    std::fflush(stdout);
                }
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[rank %d] %s\n", context.rank, error.what());
        result = 1;
    }

    int global_result = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &result, &global_result, 1, MPI_INT, MPI_MAX, context.mpi_comm));
    alps::destroy_context(context);
    MPI_Finalize();
    return global_result;
}
