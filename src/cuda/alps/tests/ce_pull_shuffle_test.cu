// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/collectives/table_exchange.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"

namespace {

struct Inputs {
    std::vector<gamax::column_ptr> columns;
    std::unique_ptr<gamax::table> keys;
};

gamax::column_ptr make_strings(
    const std::vector<std::string>& values, cudaStream_t stream) {
    std::vector<std::int32_t> lengths(values.size());
    std::string chars;
    for (std::size_t index = 0; index < values.size(); ++index) {
        lengths[index] = static_cast<std::int32_t>(values[index].size());
        chars += values[index];
    }
    static char empty_char = '\0';
    static std::int32_t empty_length = 0;
    auto result = gamax::make_string_column(
        chars.empty() ? &empty_char : chars.data(),
        lengths.empty() ? &empty_length : lengths.data(),
        static_cast<gamax::size_type>(values.size()), stream,
        gamax::ARROW_STRING);
    return gamax::column_ptr(result.release());
}

Inputs make_inputs(const alps::Context& ctx) {
    // Keep one empty rank when possible; empty send segments and non-empty
    // receives are an important edge case for a receiver-pull protocol.
    const auto rows = static_cast<gamax::size_type>(
        ctx.size > 2 && ctx.rank == ctx.size - 1 ? 0 : 769 + 37 * ctx.rank);
    std::vector<std::int32_t> key(static_cast<std::size_t>(rows));
    std::vector<std::int8_t> i8(static_cast<std::size_t>(rows));
    std::vector<std::uint8_t> u8(static_cast<std::size_t>(rows));
    std::vector<std::int32_t> i32(static_cast<std::size_t>(rows));
    std::vector<std::int64_t> i64(static_cast<std::size_t>(rows));
    std::vector<float> f32(static_cast<std::size_t>(rows));
    std::vector<double> f64(static_cast<std::size_t>(rows));
    std::vector<std::string> short_strings(static_cast<std::size_t>(rows));
    std::vector<std::string> variable_strings(static_cast<std::size_t>(rows));
    for (gamax::size_type row = 0; row < rows; ++row) {
        const auto index = static_cast<std::size_t>(row);
        key[index] = static_cast<std::int32_t>((row * 29 + ctx.rank * 11) % 127);
        i8[index] = static_cast<std::int8_t>((row + ctx.rank) % 113);
        u8[index] = static_cast<std::uint8_t>((row * 7 + ctx.rank) % 251);
        i32[index] = ctx.rank * 1000000 + static_cast<std::int32_t>(row);
        i64[index] = (static_cast<std::int64_t>(ctx.rank) << 48) | row;
        f32[index] = static_cast<float>(ctx.rank * 1000) +
                     static_cast<float>(row) * 0.25F;
        f64[index] = static_cast<double>(ctx.rank * 100000) +
                     static_cast<double>(row) * 0.125;
        short_strings[index] = row % 11 == 0
            ? std::string{}
            : "rank-" + std::to_string(ctx.rank) + "-row-" +
                std::to_string(row);
        variable_strings[index] = std::string(
            static_cast<std::size_t>((row * 7 + ctx.rank) % 67),
            static_cast<char>('a' + (row + ctx.rank) % 26));
    }

    Inputs result;
    result.columns.push_back(gamax::make_column_from_host(
        i8.data(), rows, ctx.stream));
    result.columns.push_back(gamax::make_column_from_host(
        u8.data(), rows, ctx.stream));
    result.columns.push_back(gamax::make_column_from_host(
        i32.data(), rows, ctx.stream));
    result.columns.push_back(make_strings(short_strings, ctx.stream));
    result.columns.push_back(gamax::make_column_from_host(
        i64.data(), rows, ctx.stream));
    result.columns.push_back(gamax::make_column_from_host(
        f32.data(), rows, ctx.stream));
    result.columns.push_back(gamax::make_column_from_host(
        f64.data(), rows, ctx.stream));
    result.columns.push_back(make_strings(variable_strings, ctx.stream));

    std::vector<gamax::column_ptr> keys;
    keys.push_back(gamax::make_column_from_host(key.data(), rows, ctx.stream));
    result.keys = gamax::make_table(std::move(keys));
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    return result;
}

std::vector<gamax::column_view> views(
    const std::vector<gamax::column_ptr>& columns) {
    std::vector<gamax::column_view> result;
    result.reserve(columns.size());
    for (const auto& column : columns) result.push_back(column->view());
    return result;
}

std::vector<gamax::column_ptr> run_ce_pull(
    const alps::collectives::HashPartitionPlan& plan,
    const std::vector<gamax::column_ptr>& inputs,
    const alps::Context& ctx,
    cudaStream_t preparation_stream,
    alps::collectives::CePullSchedule schedule,
    int slots) {
    alps::collectives::CePullOptions options;
    options.schedule = schedule;
    options.slots = slots;
    auto session = alps::collectives::begin_ce_pull_shuffle(
        plan, views(inputs), ctx, preparation_stream, options);
    std::vector<gamax::column_ptr> outputs(inputs.size());
    for (std::size_t ordinal = 0; ordinal < inputs.size(); ++ordinal) {
        session->enqueue_column(ordinal);
        session->wait_on(ordinal, ctx.stream);
        outputs[ordinal] = session->release_output(ordinal, ctx.stream);
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    return outputs;
}

std::vector<gamax::column_ptr> run_nccl_reference(
    const alps::collectives::HashPartitionPlan& plan,
    const std::vector<gamax::column_ptr>& inputs,
    const alps::Context& ctx) {
    std::vector<gamax::column_ptr> outputs;
    outputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        auto partitioned = alps::collectives::apply_partition(
            input->view(), plan, ctx.stream);
        outputs.push_back(alps::collectives::exchange_partitioned_column(
            partitioned->view(), plan.pattern, ctx, ctx.stream,
            alps::collectives::Backend::NCCL));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    return outputs;
}

int compare_columns(gamax::column_view got, gamax::column_view expected,
                    int rank, std::size_t ordinal) {
    if (got.type() != expected.type() || got.size() != expected.size()) {
        std::fprintf(stderr,
            "[rank %d] CE-pull shape mismatch column=%zu got_rows=%lld "
            "expected_rows=%lld\n", rank, ordinal,
            static_cast<long long>(got.size()),
            static_cast<long long>(expected.size()));
        return 1;
    }
    if (got.type().id() == gamax::type_id::STRING) {
        if (got.auxiliary_tag() != expected.auxiliary_tag() ||
            got.auxiliary_bytes() != expected.auxiliary_bytes()) {
            std::fprintf(stderr,
                "[rank %d] CE-pull STRING extent mismatch column=%zu "
                "got_chars=%lld expected_chars=%lld\n", rank, ordinal,
                static_cast<long long>(got.auxiliary_bytes()),
                static_cast<long long>(expected.auxiliary_bytes()));
            return 1;
        }
        const auto offset_bytes = static_cast<std::size_t>(got.size() + 1) *
                                  sizeof(gamax::size_type);
        const auto char_bytes = static_cast<std::size_t>(got.auxiliary_bytes());
        std::vector<unsigned char> got_offsets(offset_bytes);
        std::vector<unsigned char> expected_offsets(offset_bytes);
        std::vector<unsigned char> got_chars(char_bytes);
        std::vector<unsigned char> expected_chars(char_bytes);
        ALPS_CUDACHECK(cudaMemcpy(
            got_offsets.data(), got.auxiliary_data(), offset_bytes,
            cudaMemcpyDeviceToHost));
        ALPS_CUDACHECK(cudaMemcpy(
            expected_offsets.data(), expected.auxiliary_data(), offset_bytes,
            cudaMemcpyDeviceToHost));
        if (char_bytes != 0) {
            ALPS_CUDACHECK(cudaMemcpy(
                got_chars.data(), got.data<char>(), char_bytes,
                cudaMemcpyDeviceToHost));
            ALPS_CUDACHECK(cudaMemcpy(
                expected_chars.data(), expected.data<char>(), char_bytes,
                cudaMemcpyDeviceToHost));
        }
        if (got_offsets != expected_offsets || got_chars != expected_chars) {
            std::fprintf(stderr,
                "[rank %d] CE-pull STRING payload mismatch column=%zu\n",
                rank, ordinal);
            return 1;
        }
        return 0;
    }
    const auto bytes = static_cast<std::size_t>(got.size()) *
                       gamax::size_of(got.type());
    std::vector<unsigned char> got_host(bytes);
    std::vector<unsigned char> expected_host(bytes);
    if (bytes != 0) {
        ALPS_CUDACHECK(cudaMemcpy(
            got_host.data(), got.data(), bytes, cudaMemcpyDeviceToHost));
        ALPS_CUDACHECK(cudaMemcpy(
            expected_host.data(), expected.data(), bytes,
            cudaMemcpyDeviceToHost));
    }
    if (got_host != expected_host) {
        std::size_t first = 0;
        while (first < bytes && got_host[first] == expected_host[first]) ++first;
        std::fprintf(stderr,
            "[rank %d] CE-pull payload mismatch column=%zu byte=%zu "
            "got=%u expected=%u\n", rank, ordinal, first,
            first < bytes ? static_cast<unsigned>(got_host[first]) : 0U,
            first < bytes ? static_cast<unsigned>(expected_host[first]) : 0U);
        return 1;
    }
    return 0;
}

int run_test(
    const alps::Context& ctx,
    alps::collectives::CePullSchedule schedule,
    int slots) {
    auto input = make_inputs(ctx);
    auto plan = alps::collectives::make_hash_partition_plan(
        input.keys->view(), ctx, ctx.stream);

    cudaStream_t preparation_stream{};
    ALPS_CUDACHECK(cudaStreamCreateWithFlags(
        &preparation_stream, cudaStreamNonBlocking));

    int failed = 0;
    auto mismatched_schema = views(input.columns);
    if (ctx.rank == 0) {
        std::swap(mismatched_schema[0], mismatched_schema[1]);
    }
    alps::collectives::CePullOptions preflight_options;
    preflight_options.schedule = schedule;
    preflight_options.slots = slots;
    bool preflight_rejected = false;
    try {
        auto unexpected = alps::collectives::begin_ce_pull_shuffle(
            plan, std::move(mismatched_schema), ctx, preparation_stream,
            preflight_options);
        (void)unexpected;
    } catch (const std::invalid_argument&) {
        preflight_rejected = true;
    }
    int local_rejected = preflight_rejected ? 1 : 0;
    int every_rank_rejected = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_rejected, &every_rank_rejected, 1, MPI_INT, MPI_MIN,
        ctx.mpi_comm));
    failed |= !every_rank_rejected;

    // An all-empty STRING is a valid shuffle. It must work before the source
    // arena has ever grown: there is no remote payload and therefore no peer
    // slot mapping to look up.
    std::int32_t empty_value = 0;
    std::vector<gamax::column_ptr> empty_key_columns;
    empty_key_columns.push_back(gamax::make_column_from_host(
        &empty_value, 0, ctx.stream));
    auto empty_keys = gamax::make_table(std::move(empty_key_columns));
    std::vector<gamax::column_ptr> empty_columns;
    empty_columns.push_back(make_strings({}, ctx.stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    auto empty_plan = alps::collectives::make_hash_partition_plan(
        empty_keys->view(), ctx, ctx.stream);
    auto empty_result = run_ce_pull(
        empty_plan, empty_columns, ctx, preparation_stream, schedule, slots);
    auto empty_reference = run_nccl_reference(
        empty_plan, empty_columns, ctx);
    failed |= compare_columns(
        empty_result[0]->view(), empty_reference[0]->view(), ctx.rank, 0);
    const auto empty_stats = alps::collectives::ce_pull_stats(ctx);
    failed |= empty_stats.arena_growth_count != 0;
    failed |= empty_stats.arena_capacity_bytes != 0;

    // Simulate a rank-local allocation failure after device/string planning
    // but before STRING metadata exchange. Every rank must reject the session
    // at the coordinated boundary instead of entering mismatched collectives.
    ALPS_CUDACHECK(cudaStreamSynchronize(preparation_stream));
    setenv(
        "ALPS_TEST_CE_PULL_FAIL_STAGE", "before_string_metadata", 1);
    bool planning_failure_rejected = false;
    try {
        auto unexpected = alps::collectives::begin_ce_pull_shuffle(
            plan, views(input.columns), ctx, preparation_stream,
            preflight_options);
        (void)unexpected;
    } catch (const std::exception&) {
        planning_failure_rejected = true;
    }
    unsetenv("ALPS_TEST_CE_PULL_FAIL_STAGE");
    local_rejected = planning_failure_rejected ? 1 : 0;
    every_rank_rejected = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_rejected, &every_rank_rejected, 1, MPI_INT, MPI_MIN,
        ctx.mpi_comm));
    failed |= !every_rank_rejected;

    auto first = run_ce_pull(
        plan, input.columns, ctx, preparation_stream, schedule, slots);
    const auto warm_stats = alps::collectives::ce_pull_stats(ctx);
    failed |= warm_stats.arena_growth_count !=
        empty_stats.arena_growth_count + 1;
    failed |= warm_stats.enqueued_columns !=
        empty_stats.enqueued_columns + input.columns.size();
    const auto expected_reuse = input.columns.size() >
            static_cast<std::size_t>(slots)
        ? input.columns.size() - static_cast<std::size_t>(slots)
        : 0;
    failed |= warm_stats.slot_reuse_count < expected_reuse;
    failed |= warm_stats.slot_capacity_bytes == 0;

    auto reference = run_nccl_reference(plan, input.columns, ctx);
    for (std::size_t ordinal = 0; ordinal < first.size(); ++ordinal) {
        failed |= compare_columns(
            first[ordinal]->view(), reference[ordinal]->view(),
            ctx.rank, ordinal);
    }

    // Benchmark warm-up freezes the watermark. A same-sized steady-state
    // repetition must neither remap nor grow the VMM arena.
    alps::collectives::freeze_ce_pull_arena(ctx);
    auto second = run_ce_pull(
        plan, input.columns, ctx, preparation_stream, schedule, slots);
    const auto frozen_stats = alps::collectives::ce_pull_stats(ctx);
    failed |= !frozen_stats.growth_frozen;
    failed |= frozen_stats.arena_growth_count != warm_stats.arena_growth_count;
    failed |= frozen_stats.arena_capacity_bytes != warm_stats.arena_capacity_bytes;
    for (std::size_t ordinal = 0; ordinal < second.size(); ++ordinal) {
        failed |= compare_columns(
            second[ordinal]->view(), reference[ordinal]->view(),
            ctx.rank, ordinal);
    }

    ALPS_CUDACHECK(cudaStreamDestroy(preparation_stream));
    int global_failed = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &failed, &global_failed, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    return global_failed;
}

} // namespace

int main(int argc, char** argv) {
    alps::collectives::CePullSchedule schedule =
        alps::collectives::CePullSchedule::Unordered;
    const char* schedule_name = "unordered";
    int slots = 2;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--schedule") == 0 && index + 1 < argc) {
            const char* value = argv[++index];
            if (std::strcmp(value, "rotated") == 0) {
                schedule =
                    alps::collectives::CePullSchedule::RotatedDescriptors;
                schedule_name = "rotated";
            } else if (std::strcmp(value, "round-robin-waves") == 0) {
                schedule =
                    alps::collectives::CePullSchedule::RoundRobinWaves;
                schedule_name = "round-robin-waves";
            } else if (std::strcmp(value, "unordered") != 0) {
                std::fprintf(stderr, "unknown --schedule value: %s\n", value);
                return 2;
            }
        } else if (std::strcmp(argv[index], "--slots") == 0 &&
                   index + 1 < argc) {
            slots = std::atoi(argv[++index]);
            if (slots < 1) {
                std::fprintf(stderr, "--slots must be positive\n");
                return 2;
            }
        } else {
            std::fprintf(
                stderr,
                "usage: %s [--schedule unordered|rotated|round-robin-waves] "
                "[--slots N]\n",
                argv[0]);
            return 2;
        }
    }
    alps::configure_process();
    MPI_Init(&argc, &argv);
    auto ctx = alps::make_context();

    const int failed = run_test(ctx, schedule, slots);
    if (ctx.rank == 0) {
        std::printf(
            "[ce_pull_shuffle_test] np=%d preflight+empty-string+"
            "asymmetric-planning+fixed_width+strings+slots=%d+freeze "
            "schedule=%s -> %s\n",
            ctx.size, slots, schedule_name, failed ? "FAIL" : "PASS");
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return failed;
}
