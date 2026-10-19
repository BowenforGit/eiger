// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/context.hpp"
#include "alps/collectives/table_exchange.hpp"
#include "alps/error.hpp"
#include "alps/operators/column_overlap_eq_join.hpp"
#include "alps/tests/dist_test_util.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/join/join.hpp"
#include "string/string_column.hpp"

namespace {

using Row = std::array<int64_t, 4>;
using EdgeRow = std::array<int32_t, 3>;
using StringRow = std::array<int64_t, 3>;

gamax::column_ptr make_strings(
    const std::vector<std::string>& values, cudaStream_t stream)
{
    std::vector<int32_t> lengths(values.size());
    std::string flat;
    for (std::size_t index = 0; index < values.size(); ++index) {
        lengths[index] = static_cast<int32_t>(values[index].size());
        flat += values[index];
    }
    static const char empty_data[1] = {'\0'};
    static const int32_t empty_length[1] = {0};
    auto result = gamax::make_string_column(
        flat.empty() ? empty_data : flat.data(),
        lengths.empty() ? empty_length : lengths.data(),
        static_cast<gamax::size_type>(values.size()), stream,
        gamax::ARROW_STRING);
    return gamax::column_ptr(result.release());
}

std::vector<std::string> strings_to_host(gamax::column_view column) {
    std::vector<gamax::size_type> offsets(
        static_cast<std::size_t>(column.size()) + 1);
    ALPS_CUDACHECK(cudaMemcpy(
        offsets.data(), column.auxiliary_data(),
        offsets.size() * sizeof(gamax::size_type), cudaMemcpyDeviceToHost));
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
        if (length > 0) {
            result[row].assign(chars.data() + offsets[row], length);
        }
    }
    return result;
}

std::string left_text(int64_t id, bool secondary) {
    const auto row = id % 1000000;
    if ((!secondary && row % 4 == 0) || (secondary && row % 6 == 0)) {
        return {};
    }
    const std::string prefix = secondary ? "left-secondary-" : "left-";
    return prefix + std::to_string(id) +
        (row % 5 == 1 ? std::string(257, secondary ? 'y' : 'x') : "");
}

std::string right_text(int64_t id, bool secondary) {
    const auto encoded = id - 1000000000LL;
    const auto row = encoded % 1000000;
    if ((!secondary && row % 3 == 0) || (secondary && row % 7 == 0)) {
        return {};
    }
    const std::string prefix = secondary ? "right-secondary-" : "right-";
    return prefix + std::to_string(id) +
        (row % 5 == 2 ? std::string(513, secondary ? 'q' : 'z') : "");
}

std::unique_ptr<gamax::table> make_columns(
    const std::vector<std::vector<int64_t>>& host,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> columns;
    columns.reserve(host.size());
    for (const auto& values : host) {
        columns.push_back(gamax::make_column_from_host(
            const_cast<int64_t*>(values.data()),
            static_cast<gamax::size_type>(values.size()), stream));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(columns));
}

std::unique_ptr<gamax::table> make_columns32(
    const std::vector<std::vector<int32_t>>& host,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> columns;
    columns.reserve(host.size());
    for (const auto& values : host) {
        columns.push_back(gamax::make_column_from_host(
            const_cast<int32_t*>(values.data()),
            static_cast<gamax::size_type>(values.size()), stream));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(columns));
}

const char* schedule_name(alps::operators::ColumnSchedule schedule) {
    using Schedule = alps::operators::ColumnSchedule;
    switch (schedule) {
        case Schedule::ProbeFirst: return "probe_first";
        case Schedule::BuildFirst: return "build_first";
        case Schedule::Interleave: return "interleave";
    }
    return "unknown";
}

std::unique_ptr<gamax::table> make_string_table(
    const std::vector<int32_t>& keys,
    const std::vector<int64_t>& ids,
    const std::vector<std::string>& first,
    const std::vector<std::string>& second,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> columns;
    columns.push_back(make_strings(first, stream));
    columns.push_back(gamax::make_column_from_host(
        const_cast<int32_t*>(keys.data()),
        static_cast<gamax::size_type>(keys.size()), stream));
    columns.push_back(gamax::make_column_from_host(
        const_cast<int64_t*>(ids.data()),
        static_cast<gamax::size_type>(ids.size()), stream));
    columns.push_back(make_strings(second, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(columns));
}

int run_string_payload_cases(const alps::Context& ctx) {
    const int64_t left_rows = ctx.rank == ctx.size - 1 ? 0 : 19 + ctx.rank;
    const int64_t right_rows = ctx.rank == 0 ? 0 : 23 + ctx.rank;
    std::vector<int32_t> left_keys(static_cast<std::size_t>(left_rows));
    std::vector<int64_t> left_ids(static_cast<std::size_t>(left_rows));
    std::vector<std::string> left_first(static_cast<std::size_t>(left_rows));
    std::vector<std::string> left_second(static_cast<std::size_t>(left_rows));
    for (int64_t row = 0; row < left_rows; ++row) {
        left_keys[row] = static_cast<int32_t>(row % 5);
        left_ids[row] = static_cast<int64_t>(ctx.rank) * 1000000 + row;
        left_first[row] = left_text(left_ids[row], false);
        left_second[row] = left_text(left_ids[row], true);
    }
    std::vector<int32_t> right_keys(static_cast<std::size_t>(right_rows));
    std::vector<int64_t> right_ids(static_cast<std::size_t>(right_rows));
    std::vector<std::string> right_first(static_cast<std::size_t>(right_rows));
    std::vector<std::string> right_second(static_cast<std::size_t>(right_rows));
    for (int64_t row = 0; row < right_rows; ++row) {
        right_keys[row] = static_cast<int32_t>(row % 7);
        right_ids[row] = 1000000000LL +
            static_cast<int64_t>(ctx.rank) * 1000000 + row;
        right_first[row] = right_text(right_ids[row], false);
        right_second[row] = right_text(right_ids[row], true);
    }

    auto left = make_string_table(
        left_keys, left_ids, left_first, left_second, ctx.stream);
    auto right = make_string_table(
        right_keys, right_ids, right_first, right_second, ctx.stream);
    auto distributed_left = alps::collectives::exchange_partitions(
        alps::collectives::hash_partition(left->view(), {1}, ctx, ctx.stream),
        ctx, ctx.stream);
    auto distributed_right = alps::collectives::exchange_partitions(
        alps::collectives::hash_partition(right->view(), {1}, ctx, ctx.stream),
        ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    const auto all_left_keys =
        alps::testutil::gather_to_root(left_keys, ctx);
    const auto all_left_ids = alps::testutil::gather_to_root(left_ids, ctx);
    const auto all_right_keys =
        alps::testutil::gather_to_root(right_keys, ctx);
    const auto all_right_ids = alps::testutil::gather_to_root(right_ids, ctx);
    std::vector<StringRow> expected;
    if (ctx.rank == 0) {
        for (std::size_t left_row = 0;
             left_row < all_left_keys.size(); ++left_row) {
            for (std::size_t right_row = 0;
                 right_row < all_right_keys.size(); ++right_row) {
                if (all_left_keys[left_row] == all_right_keys[right_row]) {
                    expected.push_back({
                        all_left_keys[left_row], all_left_ids[left_row],
                        all_right_ids[right_row]});
                }
            }
        }
        std::sort(expected.begin(), expected.end());
    }

    struct DistributionCase {
        const char* name;
        gamax::table_view left;
        gamax::table_view right;
        bool partition_left;
        bool partition_right;
    };
    const std::array<DistributionCase, 4> distributions = {{
        {"string_shuffle_both", left->view(), right->view(), true, true},
        {"string_left_ready", distributed_left->view(), right->view(), false, true},
        {"string_right_ready", left->view(), distributed_right->view(), true, false},
        {"string_both_ready", distributed_left->view(), distributed_right->view(), false, false},
    }};

    gamax::join::join_hint hint;
    hint.join_algo = gamax::join::HASH;
    hint.mat_algo = gamax::join::GFUR;
    int failed = 0;
    for (const auto& distribution : distributions) {
        for (const auto schedule : {
                 alps::operators::ColumnSchedule::ProbeFirst,
                 alps::operators::ColumnSchedule::BuildFirst,
                 alps::operators::ColumnSchedule::Interleave}) {
            auto result = alps::operators::column_overlap_eq_join(
                distribution.left, 1, distribution.right, 1, hint, ctx,
                ctx.stream, distribution.partition_left,
                distribution.partition_right, schedule);
            ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
            if (result->num_columns() != 7) {
                std::fprintf(
                    stderr, "[rank %d] %s/%s: expected 7 columns, got %lld\n",
                    ctx.rank, distribution.name, schedule_name(schedule),
                    static_cast<long long>(result->num_columns()));
                failed = 1;
                continue;
            }

            const auto local_keys = alps::testutil::col_to_host<int32_t>(
                result->get_column(0), ctx.stream);
            const auto local_left_first =
                strings_to_host(result->get_column(1).view());
            const auto local_left_ids = alps::testutil::col_to_host<int64_t>(
                result->get_column(2), ctx.stream);
            const auto local_left_second =
                strings_to_host(result->get_column(3).view());
            const auto local_right_first =
                strings_to_host(result->get_column(4).view());
            const auto local_right_ids = alps::testutil::col_to_host<int64_t>(
                result->get_column(5), ctx.stream);
            const auto local_right_second =
                strings_to_host(result->get_column(6).view());
            bool local_values_match =
                local_keys.size() == local_left_first.size() &&
                local_keys.size() == local_left_ids.size() &&
                local_keys.size() == local_left_second.size() &&
                local_keys.size() == local_right_first.size() &&
                local_keys.size() == local_right_ids.size() &&
                local_keys.size() == local_right_second.size();
            for (std::size_t row = 0;
                 local_values_match && row < local_keys.size(); ++row) {
                const auto left_row = local_left_ids[row] % 1000000;
                const auto right_row =
                    (local_right_ids[row] - 1000000000LL) % 1000000;
                local_values_match = local_values_match &&
                    local_keys[row] == left_row % 5 &&
                    local_keys[row] == right_row % 7 &&
                    local_left_first[row] == left_text(local_left_ids[row], false) &&
                    local_left_second[row] == left_text(local_left_ids[row], true) &&
                    local_right_first[row] == right_text(local_right_ids[row], false) &&
                    local_right_second[row] == right_text(local_right_ids[row], true);
            }
            const auto all_keys =
                alps::testutil::gather_to_root(local_keys, ctx);
            const auto all_left =
                alps::testutil::gather_to_root(local_left_ids, ctx);
            const auto all_right =
                alps::testutil::gather_to_root(local_right_ids, ctx);
            if (ctx.rank == 0) {
                std::vector<StringRow> actual(all_keys.size());
                for (std::size_t row = 0; row < actual.size(); ++row) {
                    actual[row] = {all_keys[row], all_left[row], all_right[row]};
                }
                std::sort(actual.begin(), actual.end());
                local_values_match = local_values_match && actual == expected;
            }
            int local_ok = local_values_match ? 1 : 0;
            int global_ok = 0;
            ALPS_MPICHECK(MPI_Allreduce(
                &local_ok, &global_ok, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
            if (!global_ok) {
                std::fprintf(
                    stderr, "[rank %d] %s/%s STRING mismatch\n", ctx.rank,
                    distribution.name, schedule_name(schedule));
                failed = 1;
            } else if (ctx.rank == 0) {
                std::printf(
                    "[column_overlap_join_test] np=%d %s/%s rows=%zu -> PASS\n",
                    ctx.size, distribution.name, schedule_name(schedule),
                    expected.size());
            }
        }
    }

    // A planning allocation can fail on only one rank under skew. Verify that
    // both coordination boundaries turn that local exception into a
    // rank-wide failure, and that draining the private streams leaves the
    // communicator/runtime reusable by the next join.
    for (const char* stage : {"string_offsets", "string_output_plan"}) {
        if (ctx.rank == 0) {
            setenv("ALPS_TEST_COLUMN_OVERLAP_FAIL_STAGE", stage, 1);
        } else {
            unsetenv("ALPS_TEST_COLUMN_OVERLAP_FAIL_STAGE");
        }
        ALPS_MPICHECK(MPI_Barrier(ctx.mpi_comm));

        bool threw = false;
        try {
            auto result = alps::operators::column_overlap_eq_join(
                left->view(), 1, right->view(), 1, hint, ctx, ctx.stream,
                true, true, alps::operators::ColumnSchedule::Interleave);
            ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        } catch (const std::exception&) {
            threw = true;
        }
        unsetenv("ALPS_TEST_COLUMN_OVERLAP_FAIL_STAGE");

        int local_threw = threw ? 1 : 0;
        int all_threw = 0;
        ALPS_MPICHECK(MPI_Allreduce(
            &local_threw, &all_threw, 1, MPI_INT, MPI_MIN, ctx.mpi_comm));
        if (!all_threw) {
            std::fprintf(
                stderr,
                "[rank %d] %s fault was not propagated to every rank\n",
                ctx.rank, stage);
            failed = 1;
        }

        auto recovery = alps::operators::column_overlap_eq_join(
            left->view(), 1, right->view(), 1, hint, ctx, ctx.stream,
            true, true, alps::operators::ColumnSchedule::Interleave);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        if (recovery->num_columns() != 7) {
            std::fprintf(
                stderr, "[rank %d] %s recovery join failed\n",
                ctx.rank, stage);
            failed = 1;
        } else if (ctx.rank == 0 && all_threw) {
            std::printf(
                "[column_overlap_join_test] np=%d %s rank-wide failure/recovery"
                " -> PASS\n",
                ctx.size, stage);
        }
    }
    return failed;
}

int run_edge_case(const alps::Context& ctx) {
    // Rank-local empties plus globally duplicated and missing keys exercise
    // empty receive partitions, zero-count peers, skew, and multiset output.
    // With two ranks, opposite sides are empty on opposite ranks; larger runs
    // preserve that case while adding more duplicated rows.
    std::vector<int32_t> left_key;
    std::vector<int32_t> left_payload;
    if (ctx.rank != 0) {
        left_key = {7, 7};
        left_payload = {
            static_cast<int32_t>(ctx.rank * 100 + 1),
            static_cast<int32_t>(ctx.rank * 100 + 2)};
    }
    std::vector<int32_t> right_key;
    std::vector<int32_t> right_payload;
    if (ctx.rank != 1) {
        right_key = {7, 999, 7};
        right_payload = {
            static_cast<int32_t>(ctx.rank * 1000 + 1),
            static_cast<int32_t>(ctx.rank * 1000 + 2),
            static_cast<int32_t>(ctx.rank * 1000 + 3)};
    }

    auto left = make_columns32({left_key, left_payload}, ctx.stream);
    auto right = make_columns32({right_key, right_payload}, ctx.stream);
    const auto all_left_key = alps::testutil::gather_to_root(left_key, ctx);
    const auto all_left_payload =
        alps::testutil::gather_to_root(left_payload, ctx);
    const auto all_right_key = alps::testutil::gather_to_root(right_key, ctx);
    const auto all_right_payload =
        alps::testutil::gather_to_root(right_payload, ctx);

    std::vector<EdgeRow> expected;
    if (ctx.rank == 0) {
        for (std::size_t left_row = 0;
             left_row < all_left_key.size(); ++left_row) {
            for (std::size_t right_row = 0;
                 right_row < all_right_key.size(); ++right_row) {
                if (all_left_key[left_row] == all_right_key[right_row]) {
                    expected.push_back({
                        all_left_key[left_row],
                        all_left_payload[left_row],
                        all_right_payload[right_row]});
                }
            }
        }
        std::sort(expected.begin(), expected.end());
    }

    gamax::join::join_hint hint;
    hint.join_algo = gamax::join::HASH;
    hint.mat_algo = gamax::join::GFUR;
    auto result = alps::operators::column_overlap_eq_join(
        left->view(), 0, right->view(), 0, hint, ctx, ctx.stream,
        true, true, alps::operators::ColumnSchedule::Interleave);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    if (result->num_columns() != 3) return 1;
    auto local_key =
        alps::testutil::col_to_host<int32_t>(result->get_column(0), ctx.stream);
    auto local_left =
        alps::testutil::col_to_host<int32_t>(result->get_column(1), ctx.stream);
    auto local_right =
        alps::testutil::col_to_host<int32_t>(result->get_column(2), ctx.stream);
    auto all_key = alps::testutil::gather_to_root(local_key, ctx);
    auto all_left = alps::testutil::gather_to_root(local_left, ctx);
    auto all_right = alps::testutil::gather_to_root(local_right, ctx);

    if (ctx.rank != 0) return 0;
    std::vector<EdgeRow> actual(all_key.size());
    for (std::size_t row = 0; row < actual.size(); ++row) {
        actual[row] = {all_key[row], all_left[row], all_right[row]};
    }
    std::sort(actual.begin(), actual.end());
    if (actual != expected) {
        std::fprintf(
            stderr,
            "[column_overlap_join_test] np=%d int32_edge mismatch: expected=%zu actual=%zu\n",
            ctx.size, expected.size(), actual.size());
        return 1;
    }
    std::printf(
        "[column_overlap_join_test] np=%d int32_edge rows=%zu -> PASS\n",
        ctx.size, actual.size());
    return 0;
}

int run(const alps::Context& ctx) {
    const int64_t left_rows = 257 + ctx.rank * 3;
    const int64_t right_rows = 1021 + ctx.rank * 5;
    int64_t left_prefix = 0;
    ALPS_MPICHECK(MPI_Exscan(
        &left_rows, &left_prefix, 1, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));
    if (ctx.rank == 0) left_prefix = 0;
    int64_t global_left_rows = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &left_rows, &global_left_rows, 1, MPI_LONG_LONG, MPI_SUM,
        ctx.mpi_comm));

    // Nonzero key ordinals exercise output-schema reconstruction:
    // left=[payload,key], right=[payload0,key,payload1].
    std::vector<int64_t> left_key(left_rows), left_payload(left_rows);
    for (int64_t row = 0; row < left_rows; ++row) {
        const auto key = left_prefix + row;
        left_key[static_cast<std::size_t>(row)] = key;
        left_payload[static_cast<std::size_t>(row)] = key * 17 + 3;
    }
    std::vector<int64_t> right_key(right_rows);
    std::vector<int64_t> right_payload0(right_rows);
    std::vector<int64_t> right_payload1(right_rows);
    for (int64_t row = 0; row < right_rows; ++row) {
        const auto key =
            (ctx.rank * 131 + row * 37 + 11) % global_left_rows;
        right_key[static_cast<std::size_t>(row)] = key;
        right_payload0[static_cast<std::size_t>(row)] = key * 19 + 5;
        right_payload1[static_cast<std::size_t>(row)] =
            static_cast<int64_t>(ctx.rank) * 1000000 + row;
    }

    auto left = make_columns({left_payload, left_key}, ctx.stream);
    auto right = make_columns(
        {right_payload0, right_key, right_payload1}, ctx.stream);

    const auto all_left_key = alps::testutil::gather_to_root(left_key, ctx);
    const auto all_left_payload =
        alps::testutil::gather_to_root(left_payload, ctx);
    const auto all_right_key = alps::testutil::gather_to_root(right_key, ctx);
    const auto all_right_payload0 =
        alps::testutil::gather_to_root(right_payload0, ctx);
    const auto all_right_payload1 =
        alps::testutil::gather_to_root(right_payload1, ctx);

    std::vector<Row> expected;
    if (ctx.rank == 0) {
        std::unordered_map<int64_t, int64_t> build;
        for (std::size_t row = 0; row < all_left_key.size(); ++row) {
            build.emplace(all_left_key[row], all_left_payload[row]);
        }
        expected.reserve(all_right_key.size());
        for (std::size_t row = 0; row < all_right_key.size(); ++row) {
            const auto found = build.find(all_right_key[row]);
            if (found == build.end()) continue;
            expected.push_back({
                all_right_key[row], found->second,
                all_right_payload0[row], all_right_payload1[row]});
        }
        std::sort(expected.begin(), expected.end());
    }

    gamax::join::join_hint hint;
    hint.join_algo = gamax::join::HASH;
    hint.mat_algo = gamax::join::GFUR;
    hint.unique_left = true;

    auto distributed_left = alps::collectives::exchange_partitions(
        alps::collectives::hash_partition(
            left->view(), {1}, ctx, ctx.stream),
        ctx, ctx.stream);
    auto distributed_right = alps::collectives::exchange_partitions(
        alps::collectives::hash_partition(
            right->view(), {1}, ctx, ctx.stream),
        ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    struct DistributionCase {
        const char* name;
        gamax::table_view left;
        gamax::table_view right;
        bool partition_left;
        bool partition_right;
    };
    const std::array<DistributionCase, 4> distribution_cases = {{
        {"shuffle_both", left->view(), right->view(), true, true},
        {"left_ready", distributed_left->view(), right->view(), false, true},
        {"right_ready", left->view(), distributed_right->view(), true, false},
        {"both_ready", distributed_left->view(), distributed_right->view(), false, false},
    }};

    int failed = 0;
    for (const auto& distribution : distribution_cases) {
      for (const auto schedule : {
               alps::operators::ColumnSchedule::ProbeFirst,
               alps::operators::ColumnSchedule::BuildFirst,
               alps::operators::ColumnSchedule::Interleave}) {
        alps::operators::ColumnOverlapTiming timing;
        auto result = alps::operators::column_overlap_eq_join(
            distribution.left, 1, distribution.right, 1, hint, ctx, ctx.stream,
            distribution.partition_left, distribution.partition_right,
            schedule, &timing);
        ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
        if (result->num_columns() != 4) {
            std::fprintf(
                stderr, "[rank %d] %s/%s: expected 4 columns, got %lld\n",
                ctx.rank, distribution.name, schedule_name(schedule),
                static_cast<long long>(result->num_columns()));
            failed = 1;
            continue;
        }

        auto local_key =
            alps::testutil::col_to_host<int64_t>(result->get_column(0), ctx.stream);
        auto local_left =
            alps::testutil::col_to_host<int64_t>(result->get_column(1), ctx.stream);
        auto local_right0 =
            alps::testutil::col_to_host<int64_t>(result->get_column(2), ctx.stream);
        auto local_right1 =
            alps::testutil::col_to_host<int64_t>(result->get_column(3), ctx.stream);
        auto all_key = alps::testutil::gather_to_root(local_key, ctx);
        auto all_left = alps::testutil::gather_to_root(local_left, ctx);
        auto all_right0 = alps::testutil::gather_to_root(local_right0, ctx);
        auto all_right1 = alps::testutil::gather_to_root(local_right1, ctx);

        if (ctx.rank == 0) {
            std::vector<Row> actual(all_key.size());
            for (std::size_t row = 0; row < actual.size(); ++row) {
                actual[row] = {
                    all_key[row], all_left[row],
                    all_right0[row], all_right1[row]};
            }
            std::sort(actual.begin(), actual.end());
            if (actual != expected) {
                std::fprintf(
                    stderr,
                    "[column_overlap_join_test] np=%d %s/%s mismatch: expected=%zu actual=%zu\n",
                    ctx.size, distribution.name, schedule_name(schedule),
                    expected.size(), actual.size());
                failed = 1;
            } else {
                std::printf(
                    "[column_overlap_join_test] np=%d %s/%s rows=%zu total_ms=%.3f -> PASS\n",
                    ctx.size, distribution.name, schedule_name(schedule),
                    actual.size(), timing.total_ms);
            }
        }
      }
    }
    return std::max({failed, run_edge_case(ctx), run_string_payload_cases(ctx)});
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    auto ctx = alps::make_context();
    int local_failed = run(ctx);
    int global_failed = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_failed, &global_failed, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    alps::destroy_context(ctx);
    MPI_Finalize();
    return global_failed;
}
