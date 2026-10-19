// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cuco/hash_functions.cuh>
#include <mpi.h>

#include "alps/collectives/table_exchange.hpp"
#include "alps/collectives/hash_partition.hpp"
#include "alps/context.hpp"
#include "alps/error.hpp"
#include "exec/step_profile.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"

namespace {

using gamax::size_type;

std::size_t int64_row_hash(int64_t key) {
    std::size_t hash = 0;
    const std::size_t column_hash =
        cuco::default_hash_function<int64_t>{}(key);
    hash ^= column_hash + 0x9e3779b9U + (hash << 6) + (hash >> 2);
    return hash;
}

int legacy_hash_band(std::size_t row_hash, int world) {
    const auto folded = static_cast<uint32_t>(row_hash) ^
                        static_cast<uint32_t>(row_hash >> 32);
    return static_cast<int>(
        (static_cast<uint64_t>(folded) * static_cast<uint64_t>(world)) >> 32);
}

uint64_t reduced_width_linear_probe_cost(bool independent_destination) {
    constexpr int world = 16;
    constexpr uint32_t hash_bits = 12;
    constexpr uint32_t hash_domain = uint32_t{1} << hash_bits;
    constexpr uint32_t keys = 4500;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> shards(world);
    for (uint32_t key = 0; key < keys; ++key) {
        const uint32_t hash = key % hash_domain;
        const int destination = independent_destination
            ? alps::collectives::destination_from_row_hash(hash, world)
            : static_cast<int>(
                  (static_cast<uint64_t>(hash) * world) >> hash_bits);
        shards[destination].emplace_back(key, hash);
    }

    uint64_t probes = 0;
    for (const auto& shard : shards) {
        const std::size_t capacity = std::max<std::size_t>(1, 4 * shard.size());
        std::vector<uint32_t> slots(
            capacity, std::numeric_limits<uint32_t>::max());
        for (const auto& [key, hash] : shard) {
            std::size_t slot = hash % capacity;
            for (;;) {
                ++probes;
                if (slots[slot] == std::numeric_limits<uint32_t>::max()) {
                    slots[slot] = key;
                    break;
                }
                slot = (slot + 1) % capacity;
            }
        }
    }
    return probes;
}

int test_shuffle_hash_contract(const alps::Context& ctx) {
    if (ctx.rank != 0) return 0;
    static_assert(alps::collectives::kShuffleHashVersion == 2);
    static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));
    int fail = 0;
    constexpr int64_t samples = int64_t{1} << 20;

    // The dependency-free public helper must remain bit-for-bit compatible
    // with the seeded CUCO hash used to validate the Q18 fix.
    constexpr std::array<std::pair<std::size_t, std::uint32_t>, 5>
        compatibility_vectors{{
            {std::size_t{0}, 0x82b55a58U},
            {std::size_t{1}, 0x450eaf2dU},
            {std::size_t{0xffffffffU}, 0x500af6dbU},
            {std::size_t{0x0123456789abcdefULL}, 0x586a1812U},
            {std::size_t{0xffffffffffffffffULL}, 0x82b55a58U},
        }};
    for (const auto& [row_hash, expected] : compatibility_vectors) {
        const auto folded = static_cast<std::uint32_t>(row_hash) ^
                            static_cast<std::uint32_t>(row_hash >> 32);
        fail |= alps::collectives::shuffle_partition_hash(row_hash) != expected;
        fail |= cuco::xxhash_32<std::uint32_t>{
                    alps::collectives::kShuffleHashSeed}(folded) != expected;
    }

    // Equal hashes must route identically, while arbitrary world sizes stay
    // balanced under multiply-high reduction.
    for (const int world : {3, 4, 7, 16}) {
        std::vector<int64_t> counts(static_cast<std::size_t>(world), 0);
        for (int64_t key = 0; key < samples; ++key) {
            const auto hash = int64_row_hash(key);
            const int first =
                alps::collectives::destination_from_row_hash(hash, world);
            const int second =
                alps::collectives::destination_from_row_hash(hash, world);
            fail |= first != second || first < 0 || first >= world;
            ++counts[static_cast<std::size_t>(first)];
        }
        const double expected = static_cast<double>(samples) / world;
        for (const auto count : counts) {
            fail |= std::abs(static_cast<double>(count) / expected - 1.0) > 0.02;
        }
    }

    // Conditioning on a v2 destination must not condition the original row
    // hash to the corresponding legacy 4-bit band. Every destination/band
    // cell should receive a substantial sample.
    constexpr int world = 16;
    std::array<std::array<int64_t, world>, world> matrix{};
    for (int64_t key = 0; key < samples; ++key) {
        const auto hash = int64_row_hash(key);
        const int destination =
            alps::collectives::destination_from_row_hash(hash, world);
        const int original_band = legacy_hash_band(hash, world);
        ++matrix[static_cast<std::size_t>(destination)]
                [static_cast<std::size_t>(original_band)];
    }
    const double cell_expected =
        static_cast<double>(samples) / (world * world);
    for (const auto& row : matrix) {
        for (const auto count : row) {
            fail |= static_cast<double>(count) < 0.80 * cell_expected;
            fail |= static_cast<double>(count) > 1.20 * cell_expected;
        }
    }

    // Reduced-width model of the Q18 threshold: more unique keys per rank
    // than the legacy destination's remaining hash domain. Independent rank
    // selection must eliminate the linear-probing explosion.
    const auto legacy_probes = reduced_width_linear_probe_cost(false);
    const auto independent_probes = reduced_width_linear_probe_cost(true);
    fail |= legacy_probes < 10 * independent_probes;
    return fail;
}

std::string key_string(int64_t row) {
    if (row % 7 == 0) return {};
    return "group-" + std::to_string(row % 5);
}

std::string payload_string(int rank, int64_t row) {
    return "payload-" + std::to_string(rank) + "-" + std::to_string(row);
}

gamax::column_ptr make_strings(
    const std::vector<std::string>& values, cudaStream_t stream)
{
    std::vector<int32_t> lengths(values.size());
    std::string flat;
    for (std::size_t i = 0; i < values.size(); ++i) {
        lengths[i] = static_cast<int32_t>(values[i].size());
        flat += values[i];
    }
    static const char empty_data[1] = {'\0'};
    static const int32_t empty_length[1] = {0};
    auto result = gamax::make_string_column(
        flat.empty() ? empty_data : flat.data(),
        lengths.empty() ? empty_length : lengths.data(),
        static_cast<size_type>(values.size()),
        stream, gamax::ARROW_STRING);
    return gamax::column_ptr(result.release());
}

std::unique_ptr<gamax::table> make_input(int rank, int world, cudaStream_t stream) {
    const int64_t rows = rank == world - 1 ? 0 : 257 + rank;
    std::vector<int32_t> numeric_key(static_cast<std::size_t>(rows));
    std::vector<int64_t> payload(static_cast<std::size_t>(rows));
    std::vector<std::string> string_key(static_cast<std::size_t>(rows));
    std::vector<std::string> string_payload(static_cast<std::size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
        numeric_key[row] = static_cast<int32_t>(row % 17);
        string_key[row] = key_string(row);
        payload[row] = static_cast<int64_t>(rank) * 1000000 + row;
        string_payload[row] = payload_string(rank, row);
    }

    std::vector<gamax::column_ptr> columns;
    columns.push_back(gamax::make_column_from_host(
        numeric_key.data(), rows, stream));
    columns.push_back(make_strings(string_key, stream));
    columns.push_back(gamax::make_column_from_host(payload.data(), rows, stream));
    columns.push_back(make_strings(string_payload, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
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
    std::vector<size_type> offsets(
        static_cast<std::size_t>(column.size()) + 1);
    ALPS_CUDACHECK(cudaMemcpy(
        offsets.data(), column.auxiliary_data(),
        offsets.size() * sizeof(size_type), cudaMemcpyDeviceToHost));
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
        if (length == 0) continue;
        result[row].assign(
            chars.data() + offsets[row],
            length);
    }
    return result;
}

int validate_rows(gamax::table_view table, int world) {
    if (table.num_columns() != 4) return 1;
    auto keys = fixed_to_host<int32_t>(table.get_column(0));
    auto string_keys = strings_to_host(table.get_column(1));
    auto payloads = fixed_to_host<int64_t>(table.get_column(2));
    auto string_payloads = strings_to_host(table.get_column(3));
    for (std::size_t row = 0; row < payloads.size(); ++row) {
        const int origin = static_cast<int>(payloads[row] / 1000000);
        const int64_t origin_row = payloads[row] % 1000000;
        if (origin < 0 || origin >= world ||
            keys[row] != origin_row % 17 ||
            string_keys[row] != key_string(origin_row) ||
            string_payloads[row] != payload_string(origin, origin_row)) {
            return 1;
        }
    }
    return 0;
}

std::vector<int64_t> sorted_payloads(gamax::table_view table) {
    auto payloads = fixed_to_host<int64_t>(table.get_column(2));
    std::sort(payloads.begin(), payloads.end());
    return payloads;
}

std::vector<int64_t> expected_replicated_payloads(int world) {
    std::vector<int64_t> result;
    for (int rank = 0; rank < world; ++rank) {
        const int64_t rows = rank == world - 1 ? 0 : 257 + rank;
        for (int64_t row = 0; row < rows; ++row) {
            result.push_back(static_cast<int64_t>(rank) * 1000000 + row);
        }
    }
    return result;
}

int test_short_aligned_string_segments(const alps::Context& ctx) {
    const int destinations = std::min(ctx.size, 2);
    std::vector<int32_t> values(static_cast<std::size_t>(destinations));
    std::vector<std::string> strings(static_cast<std::size_t>(destinations));
    for (int row = 0; row < destinations; ++row) {
        values[static_cast<std::size_t>(row)] = ctx.rank * 10 + row;
        strings[static_cast<std::size_t>(row)] =
            "short-" + std::to_string(ctx.rank) + "-" + std::to_string(row);
    }

    std::vector<gamax::column_ptr> columns;
    columns.push_back(gamax::make_column_from_host(
        values.data(), destinations, ctx.stream));
    columns.push_back(make_strings(strings, ctx.stream));

    alps::collectives::CommPattern pattern;
    pattern.send_counts.assign(ctx.size, 0);
    pattern.send_offsets.assign(ctx.size, destinations);
    for (int peer = 0; peer < destinations; ++peer) {
        pattern.send_counts[peer] = 1;
        pattern.send_offsets[peer] = peer;
    }

    // INT32 makes the table row alignment four rows. The peer-1 chunk
    // [1, 2) is shorter than the three-row correction to the next aligned
    // row, and every later empty peer starts at row 2. All three split
    // offsets must remain inside those original peer ranges because the
    // STRING path reads their Arrow boundaries even for empty segments.
    const auto split =
        alps::collectives::align_comm_buffers_by_item_alignment(
            pattern, 4, ctx);
    int fail = 0;
    for (int peer = 0; peer < ctx.size; ++peer) {
        const auto begin = pattern.send_offsets[peer];
        const auto end = begin + pattern.send_counts[peer];
        size_type split_rows = 0;
        for (const auto& segment : split) {
            const auto segment_begin = segment.send_offsets[peer];
            const auto segment_count = segment.send_counts[peer];
            fail |= segment_begin < begin || segment_begin > end;
            fail |= segment_count < 0 ||
                segment_begin + segment_count > end;
            split_rows += segment_count;
        }
        fail |= split_rows != pattern.send_counts[peer];
    }

    alps::collectives::PartitionedTable partitioned;
    partitioned.table = gamax::make_table(std::move(columns));
    partitioned.pattern = std::move(pattern);
    auto exchanged = alps::collectives::exchange_partitions(
        std::move(partitioned), ctx, ctx.stream,
        alps::collectives::Backend::NCCL,
        alps::collectives::AlignmentPolicy::On);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));

    std::vector<int32_t> expected_values;
    std::vector<std::string> expected_strings;
    if (ctx.rank < destinations) {
        for (int origin = 0; origin < ctx.size; ++origin) {
            expected_values.push_back(origin * 10 + ctx.rank);
            expected_strings.push_back(
                "short-" + std::to_string(origin) + "-" +
                std::to_string(ctx.rank));
        }
    }
    fail |= fixed_to_host<int32_t>(
                exchanged->get_column(0).view()) != expected_values;
    fail |= strings_to_host(
                exchanged->get_column(1).view()) != expected_strings;
    return fail;
}

int run_tests(const alps::Context& ctx) {
    int fail = test_shuffle_hash_contract(ctx);
    fail |= test_short_aligned_string_segments(ctx);
    auto input = make_input(ctx.rank, ctx.size, ctx.stream);
    const int64_t local_rows = input->num_rows();
    int64_t global_rows = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_rows, &global_rows, 1, MPI_LONG_LONG, MPI_SUM, ctx.mpi_comm));

    auto partitioned = alps::collectives::hash_partition(
        input->view(), {0, 1}, ctx, ctx.stream);
    gamax::exec::StepProfileSink message_sink(
        /*collect_phases=*/true, /*collect_messages=*/true);
    message_sink.begin_step(
        0, gamax::exec::StepKind::ExchangePartitions,
        "message-trace exchange", ctx.stream);
    std::unique_ptr<gamax::table> shuffled;
    {
        gamax::exec::PhaseRange phase(
            &message_sink, ctx.stream,
            {"shuffle.exchange",
             gamax::exec::PhaseCategory::Communication,
             gamax::exec::CommunicationPattern::Shuffle,
             "nccl_send_recv"});
        ctx.message_trace_sink = &message_sink;
        shuffled = alps::collectives::exchange_partitions(
            std::move(partitioned), ctx, ctx.stream);
        ctx.message_trace_sink = nullptr;
    }
    message_sink.end_step(ctx.stream);
    message_sink.finalize(ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    fail |= validate_rows(shuffled->view(), ctx.size);

    std::int64_t local_send_bytes = 0;
    std::int64_t local_recv_bytes = 0;
    bool saw_string_lengths = false;
    bool saw_string_chars = false;
    for (const auto& row : message_sink.message_rows()) {
        fail |= row.payload_bytes != row.element_count * row.element_size;
        fail |= row.phase_name != "shuffle.exchange";
        fail |= row.column_ordinal < 0 || row.column_ordinal >= 4;
        if (row.api == gamax::exec::MessageApi::NcclSend) {
            local_send_bytes += row.payload_bytes;
        } else if (row.api == gamax::exec::MessageApi::NcclRecv) {
            local_recv_bytes += row.payload_bytes;
        }
        if (row.component == gamax::exec::MessageComponent::StringLength) {
            saw_string_lengths = true;
            fail |= row.logical_type.id() != gamax::type_id::STRING;
            fail |= row.element_type.id() != gamax::type_id::INT64;
            fail |= row.element_size != sizeof(size_type);
        } else if (row.component == gamax::exec::MessageComponent::StringChars) {
            saw_string_chars = true;
            fail |= row.logical_type.id() != gamax::type_id::STRING;
            fail |= row.element_type.id() != gamax::type_id::INT8;
            fail |= row.element_size != 1;
        }
    }
    fail |= !saw_string_lengths || !saw_string_chars;
    const auto expected_local_send =
        local_rows * static_cast<std::int64_t>(
            sizeof(std::int32_t) + sizeof(std::int64_t) +
            2 * sizeof(size_type)) +
        static_cast<std::int64_t>(input->get_column(1).view().auxiliary_bytes()) +
        static_cast<std::int64_t>(input->get_column(3).view().auxiliary_bytes());
    fail |= local_send_bytes != expected_local_send;
    std::int64_t global_send_bytes = 0;
    std::int64_t global_recv_bytes = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_send_bytes, &global_send_bytes, 1, MPI_INT64_T, MPI_SUM,
        ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Allreduce(
        &local_recv_bytes, &global_recv_bytes, 1, MPI_INT64_T, MPI_SUM,
        ctx.mpi_comm));
    fail |= global_send_bytes != global_recv_bytes;

    int64_t shuffled_rows = shuffled->num_rows();
    int64_t all_shuffled_rows = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &shuffled_rows, &all_shuffled_rows, 1, MPI_LONG_LONG, MPI_SUM,
        ctx.mpi_comm));
    fail |= all_shuffled_rows != global_rows;

    auto aligned_partitioned = alps::collectives::hash_partition(
        input->view(), {0, 1}, ctx, ctx.stream);
    alps::collectives::AlignmentStats shuffle_alignment_stats;
    auto aligned_shuffled = alps::collectives::exchange_partitions(
        std::move(aligned_partitioned), ctx, ctx.stream,
        alps::collectives::Backend::NCCL,
        alps::collectives::AlignmentPolicy::On,
        &shuffle_alignment_stats);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    fail |= validate_rows(aligned_shuffled->view(), ctx.size);
    fail |= aligned_shuffled->num_rows() != shuffled->num_rows();
    fail |= sorted_payloads(aligned_shuffled->view()) !=
            sorted_payloads(shuffled->view());
    fail |= !shuffle_alignment_stats.applied;
    fail |= std::accumulate(
                shuffle_alignment_stats.rows.begin(),
                shuffle_alignment_stats.rows.end(), std::int64_t{0}) !=
            local_rows;

    // The column-granular plan/apply path must reproduce the legacy
    // full-table partition and exchange exactly, including STRING columns.
    auto plan = alps::collectives::make_hash_partition_plan(
        input->view().select_columns({0, 1}), ctx, ctx.stream);
    alps::collectives::complete_receive_pattern(plan.pattern, ctx);
    std::vector<gamax::column_ptr> planned_columns;
    planned_columns.reserve(
        static_cast<std::size_t>(input->num_columns()));
    for (size_type column = 0; column < input->num_columns(); ++column) {
        planned_columns.push_back(alps::collectives::apply_partition(
            input->get_column(column).view(), plan, ctx.stream));
    }
    alps::collectives::PartitionedTable planned_partition;
    planned_partition.table = gamax::make_table(std::move(planned_columns));
    planned_partition.pattern = plan.pattern;
    auto planned_shuffle = alps::collectives::exchange_partitions(
        std::move(planned_partition), ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    fail |= validate_rows(planned_shuffle->view(), ctx.size);
    fail |= planned_shuffle->num_rows() != shuffled->num_rows();
    fail |= fixed_to_host<int32_t>(planned_shuffle->get_column(0).view()) !=
            fixed_to_host<int32_t>(shuffled->get_column(0).view());
    fail |= strings_to_host(planned_shuffle->get_column(1).view()) !=
            strings_to_host(shuffled->get_column(1).view());
    fail |= fixed_to_host<int64_t>(planned_shuffle->get_column(2).view()) !=
            fixed_to_host<int64_t>(shuffled->get_column(2).view());
    fail |= strings_to_host(planned_shuffle->get_column(3).view()) !=
            strings_to_host(shuffled->get_column(3).view());

    // Every equal composite key must occur on only one rank.
    constexpr int key_slots = 17 * 6;
    std::vector<int> local_min(key_slots, ctx.size);
    std::vector<int> local_max(key_slots, -1);
    const auto shuffled_keys = fixed_to_host<int32_t>(shuffled->get_column(0).view());
    const auto shuffled_strings = strings_to_host(shuffled->get_column(1).view());
    for (std::size_t row = 0; row < shuffled_keys.size(); ++row) {
        int string_slot = shuffled_strings[row].empty()
            ? 5
            : std::stoi(shuffled_strings[row].substr(6));
        const int slot = shuffled_keys[row] * 6 + string_slot;
        local_min[slot] = ctx.rank;
        local_max[slot] = ctx.rank;
    }
    std::vector<int> global_min(key_slots), global_max(key_slots);
    ALPS_MPICHECK(MPI_Allreduce(
        local_min.data(), global_min.data(), key_slots, MPI_INT, MPI_MIN,
        ctx.mpi_comm));
    ALPS_MPICHECK(MPI_Allreduce(
        local_max.data(), global_max.data(), key_slots, MPI_INT, MPI_MAX,
        ctx.mpi_comm));
    for (int slot = 0; slot < key_slots; ++slot) {
        if (global_max[slot] >= 0 && global_min[slot] != global_max[slot]) fail = 1;
    }

    auto replicated = alps::collectives::replicate(
        input->view(), ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    fail |= replicated->num_rows() != global_rows;
    fail |= validate_rows(replicated->view(), ctx.size);
    fail |= fixed_to_host<int64_t>(replicated->get_column(2).view()) !=
            expected_replicated_payloads(ctx.size);

    alps::collectives::AlignmentStats replication_alignment_stats;
    auto aligned_replicated = alps::collectives::replicate(
        input->view(), ctx, ctx.stream,
        alps::collectives::Backend::NCCL,
        alps::collectives::AlignmentPolicy::On,
        &replication_alignment_stats);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    fail |= aligned_replicated->num_rows() != global_rows;
    fail |= validate_rows(aligned_replicated->view(), ctx.size);
    fail |= sorted_payloads(aligned_replicated->view()) !=
            sorted_payloads(replicated->view());
    fail |= !replication_alignment_stats.applied;
    fail |= std::accumulate(
                replication_alignment_stats.rows.begin(),
                replication_alignment_stats.rows.end(), std::int64_t{0}) !=
            local_rows;

    auto gathered = alps::collectives::gather_to_root(
        input->view(), 0, ctx, ctx.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(ctx.stream));
    fail |= gathered->num_rows() != (ctx.rank == 0 ? global_rows : 0);
    fail |= validate_rows(gathered->view(), ctx.size);

    return fail;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    auto ctx = alps::make_context();

    int local_fail = run_tests(ctx);
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_fail, &global_fail, 1, MPI_INT, MPI_MAX, ctx.mpi_comm));
    if (ctx.rank == 0) {
        std::printf(
            "[table_exchange_test] np=%d off/on shuffle+replicate strings+gather -> %s\n",
            ctx.size, global_fail ? "FAIL" : "PASS");
    }

    alps::destroy_context(ctx);
    MPI_Finalize();
    return global_fail;
}
