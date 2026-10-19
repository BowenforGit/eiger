// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/operators/column_overlap_eq_join.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "string/string_column.hpp"

namespace {

gamax::column_ptr make_strings(
    const std::vector<std::string>& values, cudaStream_t stream) {
    std::vector<std::int32_t> lengths(values.size());
    std::string chars;
    for (std::size_t index = 0; index < values.size(); ++index) {
        lengths[index] = static_cast<std::int32_t>(values[index].size());
        chars += values[index];
    }
    static const char empty_char[1] = {'\0'};
    static const std::int32_t empty_length[1] = {0};
    auto result = gamax::make_string_column(
        chars.empty() ? empty_char : chars.data(),
        lengths.empty() ? empty_length : lengths.data(),
        static_cast<gamax::size_type>(values.size()), stream,
        gamax::ARROW_STRING);
    return gamax::column_ptr(result.release());
}

std::unique_ptr<gamax::table> make_side(
    const alps::Context& ctx, bool left) {
    const auto rows = static_cast<gamax::size_type>(
        (left && ctx.rank == ctx.size - 1) || (!left && ctx.rank == 0)
            ? 0 : (left ? 127 + ctx.rank * 3 : 191 + ctx.rank * 5));
    std::vector<std::int64_t> ids(static_cast<std::size_t>(rows));
    std::vector<std::int32_t> keys(static_cast<std::size_t>(rows));
    std::vector<std::string> text(static_cast<std::size_t>(rows));
    std::vector<std::int64_t> values(static_cast<std::size_t>(rows));
    for (gamax::size_type row = 0; row < rows; ++row) {
        const auto index = static_cast<std::size_t>(row);
        ids[index] = (left ? INT64_C(1000000000) : INT64_C(2000000000)) +
                     static_cast<std::int64_t>(ctx.rank) * 1000000 + row;
        keys[index] = static_cast<std::int32_t>(
            (row * (left ? 13 : 17) + ctx.rank * 7) % 31);
        text[index] = row % (left ? 9 : 7) == 0
            ? std::string{}
            : (left ? "left-" : "right-") + std::to_string(ids[index]) +
                std::string(static_cast<std::size_t>((row * 11) % 97),
                            left ? 'l' : 'r');
        values[index] = ids[index] * (left ? 3 : 5) + 1;
    }
    std::vector<gamax::column_ptr> columns;
    columns.push_back(make_strings(text, ctx.stream));
    columns.push_back(gamax::make_column_from_host(
        keys.data(), rows, ctx.stream));
    columns.push_back(gamax::make_column_from_host(
        ids.data(), rows, ctx.stream));
    columns.push_back(gamax::make_column_from_host(
        values.data(), rows, ctx.stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    return gamax::make_table(std::move(columns));
}

template <typename T>
std::vector<T> fixed_to_host(gamax::column_view column) {
    std::vector<T> result(static_cast<std::size_t>(column.size()));
    if (!result.empty()) {
        ALPS_CUDACHECK(cudaMemcpy(
            result.data(), column.data<T>(), result.size() * sizeof(T),
            cudaMemcpyDeviceToHost));
    }
    return result;
}

std::vector<std::string> strings_to_host(gamax::column_view column) {
    auto offsets = fixed_to_host<gamax::size_type>(gamax::column_view(
        column.auxiliary_data(), column.size() + 1,
        gamax::data_type{gamax::type_id::INT64}));
    std::vector<char> chars(static_cast<std::size_t>(offsets.back()));
    if (!chars.empty()) {
        ALPS_CUDACHECK(cudaMemcpy(
            chars.data(), column.data<char>(), chars.size(),
            cudaMemcpyDeviceToHost));
    }
    std::vector<std::string> result(static_cast<std::size_t>(column.size()));
    for (std::size_t row = 0; row < result.size(); ++row) {
        const auto length =
            static_cast<std::size_t>(offsets[row + 1] - offsets[row]);
        if (length != 0) {
            result[row].assign(chars.data() + offsets[row], length);
        }
    }
    return result;
}

struct JoinRow {
    std::int32_t key;
    std::string left_text;
    std::int64_t left_id;
    std::int64_t left_value;
    std::string right_text;
    std::int64_t right_id;
    std::int64_t right_value;

    auto tuple() const {
        return std::tie(
            key, left_text, left_id, left_value,
            right_text, right_id, right_value);
    }
    bool operator<(const JoinRow& other) const { return tuple() < other.tuple(); }
    bool operator==(const JoinRow& other) const { return tuple() == other.tuple(); }
};

std::vector<JoinRow> rows_to_host(gamax::table_view table) {
    auto keys = fixed_to_host<std::int32_t>(table.get_column(0));
    auto left_text = strings_to_host(table.get_column(1));
    auto left_ids = fixed_to_host<std::int64_t>(table.get_column(2));
    auto left_values = fixed_to_host<std::int64_t>(table.get_column(3));
    auto right_text = strings_to_host(table.get_column(4));
    auto right_ids = fixed_to_host<std::int64_t>(table.get_column(5));
    auto right_values = fixed_to_host<std::int64_t>(table.get_column(6));
    std::vector<JoinRow> result(keys.size());
    for (std::size_t row = 0; row < result.size(); ++row) {
        result[row] = JoinRow{
            keys[row], std::move(left_text[row]), left_ids[row],
            left_values[row], std::move(right_text[row]), right_ids[row],
            right_values[row]};
    }
    std::sort(result.begin(), result.end());
    return result;
}

int compare_tables(gamax::table_view got, gamax::table_view expected, int rank) {
    if (got.num_columns() != expected.num_columns() ||
        got.num_rows() != expected.num_rows()) return 1;
    auto got_rows = rows_to_host(got);
    auto expected_rows = rows_to_host(expected);
    if (got_rows != expected_rows) {
        std::fprintf(stderr,
            "[rank %d] CE-pull join row-set mismatch got=%zu expected=%zu\n",
            rank, got_rows.size(), expected_rows.size());
        return 1;
    }
    return 0;
}

int run(const alps::Context& ctx) {
    auto left = make_side(ctx, true);
    auto right = make_side(ctx, false);
    gamax::join::join_hint hint;
    hint.join_algo = gamax::join::HASH;
    hint.mat_algo = gamax::join::GFUR;

    auto reference = alps::operators::column_overlap_eq_join(
        left->view(), 1, right->view(), 1, hint, ctx, ctx.stream,
        true, true, alps::operators::ColumnSchedule::Interleave,
        nullptr, nullptr, true, false,
        alps::collectives::Backend::NCCL);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    auto first = alps::operators::column_overlap_eq_join(
        left->view(), 1, right->view(), 1, hint, ctx, ctx.stream,
        true, true, alps::operators::ColumnSchedule::Interleave,
        nullptr, nullptr, true, false,
        alps::collectives::Backend::CE_PULL);
    int failed = 0;
    const auto warm_stats = alps::collectives::ce_pull_stats(ctx);
    if (warm_stats.arena_growth_count == 0 ||
        warm_stats.enqueued_columns != 8) {
        std::fprintf(stderr,
            "[rank %d] CE-pull warm stats growth=%llu enqueues=%llu\n",
            ctx.rank,
            static_cast<unsigned long long>(warm_stats.arena_growth_count),
            static_cast<unsigned long long>(warm_stats.enqueued_columns));
        failed = 1;
    }

    alps::collectives::freeze_ce_pull_arena(ctx);
    auto second = alps::operators::column_overlap_eq_join(
        left->view(), 1, right->view(), 1, hint, ctx, ctx.stream,
        true, true, alps::operators::ColumnSchedule::Interleave,
        nullptr, nullptr, true, false,
        alps::collectives::Backend::CE_PULL);
    // Deliberately synchronize only after the second invocation. This makes
    // the second full-staging shuffle enqueue arena reuse while the first
    // shuffle may still be in flight, exercising the whole-shuffle completion
    // handshake rather than relying on a host barrier between shuffles.
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    failed |= compare_tables(first->view(), reference->view(), ctx.rank);
    failed |= compare_tables(second->view(), reference->view(), ctx.rank);
    const auto frozen_stats = alps::collectives::ce_pull_stats(ctx);
    if (frozen_stats.arena_growth_count != warm_stats.arena_growth_count ||
        frozen_stats.arena_capacity_bytes != warm_stats.arena_capacity_bytes ||
        frozen_stats.arena_overflow_count != 0) {
        std::fprintf(stderr,
            "[rank %d] CE-pull frozen stats growth=%llu/%llu capacity=%zu/%zu "
            "overflow=%llu\n", ctx.rank,
            static_cast<unsigned long long>(frozen_stats.arena_growth_count),
            static_cast<unsigned long long>(warm_stats.arena_growth_count),
            frozen_stats.arena_capacity_bytes, warm_stats.arena_capacity_bytes,
            static_cast<unsigned long long>(frozen_stats.arena_overflow_count));
        failed = 1;
    }

    int global_failed = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &failed, &global_failed, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    return global_failed;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    auto ctx = alps::make_context();
    const int failed = run(ctx);
    if (ctx.rank == 0) {
        std::printf(
            "[ce_pull_join_test] np=%d NCCL_oracle+strings+freeze -> %s\n",
            ctx.size, failed ? "FAIL" : "PASS");
    }
    alps::destroy_context(ctx);
    MPI_Finalize();
    return failed;
}
