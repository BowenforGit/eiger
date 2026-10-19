// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/collectives/table_exchange.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/collectives/allgather.hpp"
#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/collectives/hash_partition.hpp"
#include "alps/convert.hpp"
#include "alps/error.hpp"
#include "memory/row_utils.cuh"
#include "operators/partition/partition.hpp"
#include "string/string_column.hpp"
#include "string/string.hpp"
#include "utils/transform.hpp"

namespace alps::collectives {
namespace {

using gamax::size_type;

__global__ void hash_destinations_kernel(
    gamax::experimental::table_device_view keys,
    int32_t* destinations,
    size_type rows,
    int world)
{
    const size_type row =
        static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    destinations[row] = alps::collectives::destination_from_row_hash(
        gamax::row_hasher{keys}(row), world);
}

__global__ void string_lengths_kernel(
    const size_type* offsets, size_type* lengths, size_type rows)
{
    const size_type row =
        static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows) lengths[row] = offsets[row + 1] - offsets[row];
}

void finish_receive_pattern(CommPattern& pattern, const Context& ctx) {
    const int world = ctx.size;
    pattern.recv_counts.resize(world);
    ALPS_MPICHECK(MPI_Alltoall(
        pattern.send_counts.data(), 1, alps::convert::mpi_size_type(),
        pattern.recv_counts.data(), 1, alps::convert::mpi_size_type(),
        ctx.mpi_comm));

    pattern.recv_offsets.resize(world);
    size_type offset = 0;
    for (int rank = 0; rank < world; ++rank) {
        pattern.recv_offsets[rank] = offset;
        offset += pattern.recv_counts[rank];
    }
}

void exchange_column(
    gamax::column_view send,
    gamax::mutable_column_view recv,
    CommPattern& pattern,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend,
    gamax::exec::MessageTraceDescriptor trace)
{
    if (backend == Backend::CE) {
        if (pattern.dst_offsets.empty()) finalize_dst_offsets(pattern, ctx);
        alltoall_column_ce(send, recv, pattern, ctx, stream);
    } else if (backend == Backend::NCCL) {
        alltoall_column(send, recv, pattern, ctx, stream, trace);
    } else {
        throw std::invalid_argument(
            "exchange_column: CE_PULL requires a CePullShuffleSession");
    }
}

gamax::column_ptr gather_column(
    gamax::column_view map,
    gamax::column_view input,
    cudaStream_t stream)
{
    if (input.type().id() == gamax::type_id::STRING) {
        gamax::string_column_view strings(input, gamax::ARROW_STRING);
        auto result = gamax::gather_strings(strings, map, stream);
        return gamax::column_ptr(result.release());
    }
    return gamax::util::device::gather(map, input, stream);
}

size_type receive_extent(const std::vector<CommPattern>& patterns) {
    size_type extent = 0;
    for (const auto& pattern : patterns) {
        for (std::size_t peer = 0; peer < pattern.recv_counts.size(); ++peer) {
            extent = std::max(
                extent,
                pattern.recv_offsets[peer] + pattern.recv_counts[peer]);
        }
    }
    return extent;
}

size_t table_row_alignment(gamax::table_view input) {
    size_t alignment = 1;
    for (size_type column = 0; column < input.num_columns(); ++column) {
        const auto view = input.get_column(column);
        const size_t element_size =
            view.type().id() == gamax::type_id::STRING
                ? sizeof(size_type)
                : gamax::size_of(view.type());
        alignment = std::lcm(
            alignment, compute_item_alignment(element_size));
    }
    return alignment;
}

std::unique_ptr<gamax::column> exchange_string_column_patterns(
    gamax::column_view send,
    std::vector<CommPattern>& row_patterns,
    const std::vector<int>& stat_slots,
    std::int64_t column_ordinal,
    const std::vector<gamax::exec::MessageSegment>& trace_segments,
    AlignmentStats* alignment_stats,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend)
{
    if (send.auxiliary_tag() != static_cast<int32_t>(gamax::ARROW_STRING)) {
        throw std::runtime_error(
            "exchange_partitions: STRING input must use Arrow offsets");
    }

    const size_type send_rows = send.size();
    const size_type recv_rows = receive_extent(row_patterns);
    const auto* send_offsets =
        static_cast<const size_type*>(send.auxiliary_data());

    auto send_lengths = gamax::make_column<size_type>(send_rows, stream);
    if (send_rows > 0) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((send_rows + threads - 1) / threads);
        string_lengths_kernel<<<blocks, threads, 0, stream>>>(
            send_offsets, send_lengths->data<size_type>(), send_rows);
        ALPS_CUDACHECK(cudaGetLastError());
    }

    auto recv_lengths = gamax::make_column<size_type>(recv_rows, stream);
    for (std::size_t segment = 0; segment < row_patterns.size(); ++segment) {
        auto& row_pattern = row_patterns[segment];
        exchange_column(
            send_lengths->view(), recv_lengths->mutable_view(), row_pattern,
            ctx, stream, backend,
            {column_ordinal, send.type(),
             gamax::exec::MessageComponent::StringLength,
             trace_segments[segment]});
    }

    std::vector<CommPattern> char_patterns(row_patterns.size());
    std::vector<std::vector<size_type>> begin_offsets(
        row_patterns.size(), std::vector<size_type>(ctx.size));
    std::vector<std::vector<size_type>> end_offsets(
        row_patterns.size(), std::vector<size_type>(ctx.size));
    for (std::size_t segment = 0; segment < row_patterns.size(); ++segment) {
        const auto& row_pattern = row_patterns[segment];
        for (int peer = 0; peer < ctx.size; ++peer) {
            const size_type begin_row = row_pattern.send_offsets[peer];
            const size_type end_row =
                begin_row + row_pattern.send_counts[peer];
            ALPS_CUDACHECK(cudaMemcpyAsync(
                &begin_offsets[segment][peer], send_offsets + begin_row,
                sizeof(size_type), cudaMemcpyDeviceToHost, stream));
            ALPS_CUDACHECK(cudaMemcpyAsync(
                &end_offsets[segment][peer], send_offsets + end_row,
                sizeof(size_type), cudaMemcpyDeviceToHost, stream));
        }
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));

    size_type total_chars = 0;
    for (std::size_t segment = 0; segment < row_patterns.size(); ++segment) {
        auto& char_pattern = char_patterns[segment];
        char_pattern.send_counts.resize(ctx.size);
        char_pattern.send_offsets.resize(ctx.size);
        for (int peer = 0; peer < ctx.size; ++peer) {
            char_pattern.send_offsets[peer] = begin_offsets[segment][peer];
            char_pattern.send_counts[peer] =
                end_offsets[segment][peer] - begin_offsets[segment][peer];
        }
        char_pattern.recv_counts.resize(ctx.size);
        ALPS_MPICHECK(MPI_Alltoall(
            char_pattern.send_counts.data(), 1,
            alps::convert::mpi_size_type(),
            char_pattern.recv_counts.data(), 1,
            alps::convert::mpi_size_type(), ctx.mpi_comm));
        char_pattern.recv_offsets.resize(ctx.size);
        for (int peer = 0; peer < ctx.size; ++peer) {
            char_pattern.recv_offsets[peer] = total_chars;
            total_chars += char_pattern.recv_counts[peer];
        }
        if (alignment_stats) {
            alignment_stats->bytes[stat_slots[segment]] +=
                std::accumulate(
                    char_pattern.send_counts.begin(),
                    char_pattern.send_counts.end(),
                    std::int64_t{0});
        }
    }

    gamax::device_buffer recv_chars(
        static_cast<std::size_t>(total_chars), stream);
    gamax::column_view send_chars(
        send.data<char>(), static_cast<size_type>(send.auxiliary_bytes()),
        gamax::data_type{gamax::type_id::INT8});
    gamax::mutable_column_view recv_chars_view(
        recv_chars.data(), total_chars,
        gamax::data_type{gamax::type_id::INT8});
    for (std::size_t segment = 0; segment < char_patterns.size(); ++segment) {
        auto& char_pattern = char_patterns[segment];
        exchange_column(
            send_chars, recv_chars_view, char_pattern, ctx, stream, backend,
            {column_ordinal, send.type(),
             gamax::exec::MessageComponent::StringChars,
             trace_segments[segment]});
    }

    auto recv_offsets = gamax::util::device::prefix_sum(
        recv_lengths->view(), gamax::prefix_sum_options::INCLUSIVE_PAD_ZERO,
        gamax::data_type{gamax::type_to_id<size_type>()}, stream);
    auto result = gamax::make_string_column(
        std::move(recv_chars), std::move(recv_offsets->buffer()), recv_rows,
        stream, gamax::ARROW_STRING);
    return std::unique_ptr<gamax::column>(result.release());
}

CommPattern make_allgather_pattern(
    size_type local_rows, const Context& ctx)
{
    CommPattern pattern;
    pattern.send_counts.resize(ctx.size);
    ALPS_MPICHECK(MPI_Allgather(
        &local_rows, 1, alps::convert::mpi_size_type(),
        pattern.send_counts.data(), 1, alps::convert::mpi_size_type(),
        ctx.mpi_comm));
    pattern.recv_counts = pattern.send_counts;
    pattern.send_offsets.assign(ctx.size, 0);
    pattern.recv_offsets.resize(ctx.size);
    size_type offset = 0;
    for (int root = 0; root < ctx.size; ++root) {
        pattern.recv_offsets[root] = offset;
        offset += pattern.recv_counts[root];
    }
    return pattern;
}

std::vector<CommPattern> make_allgather_patterns(
    size_type local_rows,
    size_t item_alignment,
    const Context& ctx,
    AlignmentPolicy alignment)
{
    auto plain = make_allgather_pattern(local_rows, ctx);
    if (alignment == AlignmentPolicy::Off) {
        return {std::move(plain)};
    }

    const auto ia = static_cast<size_type>(std::max<size_t>(
        1, item_alignment));
    CommPattern main;
    CommPattern post;
    for (auto* pattern : {&main, &post}) {
        pattern->send_counts.resize(ctx.size);
        pattern->send_offsets.resize(ctx.size);
        pattern->recv_counts.resize(ctx.size);
        pattern->recv_offsets.resize(ctx.size);
    }
    for (int root = 0; root < ctx.size; ++root) {
        const auto count = plain.send_counts[root];
        const auto post_count = count % ia;
        const auto main_count = count - post_count;
        main.send_counts[root] = main_count;
        main.send_offsets[root] = 0;
        main.recv_counts[root] = main_count;
        post.send_counts[root] = post_count;
        post.send_offsets[root] = main_count;
        post.recv_counts[root] = post_count;
    }
    size_type offset = 0;
    for (int root = 0; root < ctx.size; ++root) {
        main.recv_offsets[root] = offset;
        offset += main.recv_counts[root];
    }
    for (int root = 0; root < ctx.size; ++root) {
        post.recv_offsets[root] = offset;
        offset += post.recv_counts[root];
    }
    return {std::move(main), std::move(post)};
}

void broadcast_column(
    gamax::column_view send,
    gamax::mutable_column_view recv,
    CommPattern& pattern,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend,
    gamax::exec::MessageTraceDescriptor trace)
{
    if (backend == Backend::CE) {
        if (pattern.dst_offsets.empty()) finalize_dst_offsets(pattern, ctx);
        allgather_column_ce(send, recv, pattern, ctx, stream);
    } else if (backend == Backend::NCCL) {
        allgather_column(send, recv, pattern, ctx, stream, trace);
    } else {
        throw std::invalid_argument(
            "broadcast_column: CE_PULL is not an all-gather backend");
    }
}

std::unique_ptr<gamax::column> broadcast_string_column_patterns(
    gamax::column_view send,
    std::vector<CommPattern>& row_patterns,
    const std::vector<int>& stat_slots,
    std::int64_t column_ordinal,
    const std::vector<gamax::exec::MessageSegment>& trace_segments,
    AlignmentStats* alignment_stats,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend)
{
    if (send.auxiliary_tag() != static_cast<int32_t>(gamax::ARROW_STRING)) {
        throw std::runtime_error(
            "replicate: STRING input must use Arrow offsets");
    }
    const auto* send_offsets =
        static_cast<const size_type*>(send.auxiliary_data());
    auto send_lengths = gamax::make_column<size_type>(send.size(), stream);
    if (send.size() > 0) {
        constexpr int threads = 256;
        const int blocks =
            static_cast<int>((send.size() + threads - 1) / threads);
        string_lengths_kernel<<<blocks, threads, 0, stream>>>(
            send_offsets, send_lengths->data<size_type>(), send.size());
        ALPS_CUDACHECK(cudaGetLastError());
    }

    const size_type recv_rows = receive_extent(row_patterns);
    auto recv_lengths = gamax::make_column<size_type>(recv_rows, stream);
    for (std::size_t segment = 0; segment < row_patterns.size(); ++segment) {
        auto& row_pattern = row_patterns[segment];
        broadcast_column(
            send_lengths->view(), recv_lengths->mutable_view(), row_pattern,
            ctx, stream, backend,
            {column_ordinal, send.type(),
             gamax::exec::MessageComponent::StringLength,
             trace_segments[segment]});
    }

    std::vector<size_type> local_begin(row_patterns.size());
    std::vector<size_type> local_end(row_patterns.size());
    for (std::size_t segment = 0; segment < row_patterns.size(); ++segment) {
        const auto& pattern = row_patterns[segment];
        const auto begin_row = pattern.send_offsets[ctx.rank];
        const auto end_row =
            begin_row + pattern.send_counts[ctx.rank];
        ALPS_CUDACHECK(cudaMemcpyAsync(
            &local_begin[segment], send_offsets + begin_row,
            sizeof(size_type), cudaMemcpyDeviceToHost, stream));
        ALPS_CUDACHECK(cudaMemcpyAsync(
            &local_end[segment], send_offsets + end_row,
            sizeof(size_type), cudaMemcpyDeviceToHost, stream));
    }
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));

    std::vector<CommPattern> char_patterns(row_patterns.size());
    size_type total_chars = 0;
    for (std::size_t segment = 0; segment < row_patterns.size(); ++segment) {
        const size_type local_count =
            local_end[segment] - local_begin[segment];
        auto& pattern = char_patterns[segment];
        pattern.send_counts.resize(ctx.size);
        pattern.send_offsets.resize(ctx.size);
        ALPS_MPICHECK(MPI_Allgather(
            &local_count, 1, alps::convert::mpi_size_type(),
            pattern.send_counts.data(), 1,
            alps::convert::mpi_size_type(), ctx.mpi_comm));
        ALPS_MPICHECK(MPI_Allgather(
            &local_begin[segment], 1, alps::convert::mpi_size_type(),
            pattern.send_offsets.data(), 1,
            alps::convert::mpi_size_type(), ctx.mpi_comm));
        pattern.recv_counts = pattern.send_counts;
        pattern.recv_offsets.resize(ctx.size);
        for (int root = 0; root < ctx.size; ++root) {
            pattern.recv_offsets[root] = total_chars;
            total_chars += pattern.recv_counts[root];
        }
        if (alignment_stats) {
            alignment_stats->bytes[stat_slots[segment]] +=
                pattern.send_counts[ctx.rank];
        }
    }

    gamax::device_buffer recv_chars(
        static_cast<std::size_t>(total_chars), stream);
    gamax::column_view send_chars(
        send.data<char>(), static_cast<size_type>(send.auxiliary_bytes()),
        gamax::data_type{gamax::type_id::INT8});
    gamax::mutable_column_view recv_chars_view(
        recv_chars.data(), total_chars,
        gamax::data_type{gamax::type_id::INT8});
    for (std::size_t segment = 0; segment < char_patterns.size(); ++segment) {
        auto& char_pattern = char_patterns[segment];
        broadcast_column(
            send_chars, recv_chars_view, char_pattern, ctx, stream, backend,
            {column_ordinal, send.type(),
             gamax::exec::MessageComponent::StringChars,
             trace_segments[segment]});
    }

    auto recv_offsets = gamax::util::device::prefix_sum(
        recv_lengths->view(), gamax::prefix_sum_options::INCLUSIVE_PAD_ZERO,
        gamax::data_type{gamax::type_to_id<size_type>()}, stream);
    auto result = gamax::make_string_column(
        std::move(recv_chars), std::move(recv_offsets->buffer()), recv_rows,
        stream, gamax::ARROW_STRING);
    return std::unique_ptr<gamax::column>(result.release());
}

PartitionedTable make_gather_partition(
    gamax::table_view in, const Context& ctx, int root)
{
    PartitionedTable result;
    result.borrowed = in;
    result.pattern.send_counts.assign(ctx.size, 0);
    result.pattern.send_counts[root] = in.num_rows();
    result.pattern.send_offsets.resize(ctx.size);
    size_type next = 0;
    for (int peer = 0; peer < ctx.size; ++peer) {
        result.pattern.send_offsets[peer] = next;
        next += result.pattern.send_counts[peer];
    }
    return result;
}

} // namespace

void complete_receive_pattern(CommPattern& pattern, const Context& ctx) {
    const auto world = static_cast<std::size_t>(ctx.size);
    if (pattern.send_counts.size() != world ||
        pattern.send_offsets.size() != world) {
        throw std::invalid_argument(
            "complete_receive_pattern: send pattern does not match world size");
    }
    const bool have_recv =
        pattern.recv_counts.size() == world &&
        pattern.recv_offsets.size() == world;
    if (have_recv) return;
    if (!pattern.recv_counts.empty() || !pattern.recv_offsets.empty()) {
        throw std::invalid_argument(
            "complete_receive_pattern: incomplete receive pattern");
    }
    finish_receive_pattern(pattern, ctx);
}

HashPartitionPlan make_hash_partition_plan(
    gamax::table_view keys,
    const Context& ctx,
    cudaStream_t stream)
{
    if (keys.num_columns() == 0) {
        throw std::invalid_argument(
            "make_hash_partition_plan: key list must not be empty");
    }

    const size_type rows = keys.num_rows();
    auto device_keys = keys.device_view(stream);
    auto destinations = gamax::make_column<int32_t>(rows, stream);
    if (rows > 0) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((rows + threads - 1) / threads);
        hash_destinations_kernel<<<blocks, threads, 0, stream>>>(
            *device_keys, destinations->data<int32_t>(), rows, ctx.size);
        ALPS_CUDACHECK(cudaGetLastError());
    }

    const int rank_bits = ctx.rank_bits();
    auto [destinations_sorted, permutation] = gamax::partition::arg_partition(
        destinations->view(), 0, rank_bits, stream, nullptr, 0);
    auto offsets = gamax::partition::compute_partition_offsets(
        destinations_sorted->view(), 0, rank_bits, stream);

    std::vector<size_type> exclusive(static_cast<std::size_t>(ctx.size) + 1, 0);
    ALPS_CUDACHECK(cudaMemcpyAsync(
        exclusive.data() + 1, offsets->data<size_type>(),
        sizeof(size_type) * static_cast<std::size_t>(ctx.size),
        cudaMemcpyDeviceToHost, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));

    HashPartitionPlan result;
    result.permutation = std::move(permutation);
    result.pattern.send_counts.resize(ctx.size);
    result.pattern.send_offsets.resize(ctx.size);
    for (int peer = 0; peer < ctx.size; ++peer) {
        result.pattern.send_offsets[peer] = exclusive[peer];
        result.pattern.send_counts[peer] =
            exclusive[peer + 1] - exclusive[peer];
    }
    return result;
}

gamax::column_ptr apply_partition(
    gamax::column_view input,
    const HashPartitionPlan& plan,
    cudaStream_t stream)
{
    if (!plan.permutation || plan.permutation->size() != input.size()) {
        throw std::invalid_argument(
            "apply_partition: permutation does not match input column");
    }
    return gather_column(plan.permutation->view(), input, stream);
}

gamax::column_ptr exchange_partitioned_column(
    gamax::column_view send,
    CommPattern pattern,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend)
{
    if (backend == Backend::CE_PULL) {
        throw std::invalid_argument(
            "exchange_partitioned_column: CE_PULL requires a "
            "CePullShuffleSession");
    }
    complete_receive_pattern(pattern, ctx);
    std::vector<CommPattern> patterns;
    patterns.push_back(std::move(pattern));
    if (send.type().id() == gamax::type_id::STRING) {
        return exchange_string_column_patterns(
            send, patterns, {}, -1,
            {gamax::exec::MessageSegment::Whole},
            nullptr, ctx, stream, backend);
    }

    const size_type recv_rows = receive_extent(patterns);
    auto recv = gamax::make_column(send.type(), recv_rows, stream);
    exchange_column(
        send, recv->mutable_view(), patterns.front(), ctx, stream, backend,
        {-1, send.type(), gamax::exec::MessageComponent::FixedValue,
         gamax::exec::MessageSegment::Whole});
    return recv;
}

PartitionedTable hash_partition(
    gamax::table_view in,
    const std::vector<size_type>& key_ordinals,
    const Context& ctx,
    cudaStream_t stream)
{
    if (key_ordinals.empty()) {
        throw std::invalid_argument("hash_partition: key list must not be empty");
    }
    for (const auto ordinal : key_ordinals) {
        if (ordinal < 0 || ordinal >= in.num_columns()) {
            throw std::out_of_range("hash_partition: key ordinal out of range");
        }
    }

    auto plan = make_hash_partition_plan(
        in.select_columns(key_ordinals), ctx, stream);
    std::vector<gamax::column_ptr> columns;
    columns.reserve(static_cast<std::size_t>(in.num_columns()));
    for (size_type column = 0; column < in.num_columns(); ++column) {
        columns.push_back(apply_partition(
            in.get_column(column), plan, stream));
    }
    PartitionedTable result;
    result.table = gamax::make_table(std::move(columns));
    result.pattern = std::move(plan.pattern);
    return result;
}

std::unique_ptr<gamax::table> gather_rows(
    gamax::table_view in,
    gamax::column_view row_indices,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> columns;
    columns.reserve(static_cast<std::size_t>(in.num_columns()));
    for (size_type column = 0; column < in.num_columns(); ++column) {
        columns.push_back(gather_column(
            row_indices, in.get_column(column), stream));
    }
    return gamax::make_table(std::move(columns));
}

PartitionedTable partition_by_destination(
    gamax::table_view in,
    gamax::column_view destinations,
    const Context& ctx,
    cudaStream_t stream)
{
    if (destinations.type().id() != gamax::type_id::INT32 ||
        destinations.size() != in.num_rows()) {
        throw std::invalid_argument(
            "partition_by_destination: expected one INT32 destination per row");
    }
    const int rank_bits = ctx.rank_bits();
    auto [destinations_sorted, permutation] = gamax::partition::arg_partition(
        destinations, 0, rank_bits, stream, nullptr, 0);

    auto gathered = gather_rows(in, permutation->view(), stream);

    auto offsets = gamax::partition::compute_partition_offsets(
        destinations_sorted->view(), 0, rank_bits, stream);
    std::vector<size_type> exclusive(static_cast<std::size_t>(ctx.size) + 1, 0);
    ALPS_CUDACHECK(cudaMemcpyAsync(
        exclusive.data() + 1, offsets->data<size_type>(),
        sizeof(size_type) * static_cast<std::size_t>(ctx.size),
        cudaMemcpyDeviceToHost, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));

    PartitionedTable result;
    result.table = std::move(gathered);
    result.pattern.send_counts.resize(ctx.size);
    result.pattern.send_offsets.resize(ctx.size);
    for (int peer = 0; peer < ctx.size; ++peer) {
        result.pattern.send_offsets[peer] = exclusive[peer];
        result.pattern.send_counts[peer] = exclusive[peer + 1] - exclusive[peer];
    }
    return result;
}

std::unique_ptr<gamax::table> exchange_partitions(
    PartitionedTable partitioned,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend,
    AlignmentPolicy alignment,
    AlignmentStats* alignment_stats)
{
    if (!partitioned.table && partitioned.borrowed.num_columns() == 0) {
        throw std::invalid_argument("exchange_partitions: missing table view");
    }
    auto pattern = std::move(partitioned.pattern);
    if (pattern.send_counts.size() != static_cast<std::size_t>(ctx.size) ||
        pattern.send_offsets.size() != static_cast<std::size_t>(ctx.size)) {
        throw std::invalid_argument(
            "exchange_partitions: send pattern does not match world size");
    }
    const auto input = partitioned.table
        ? partitioned.table->view()
        : partitioned.borrowed;
    if (backend == Backend::CE_PULL) {
        if (alignment != AlignmentPolicy::Off) {
            throw std::invalid_argument(
                "exchange_partitions: alignment requires the NCCL backend");
        }
        if (alignment_stats) *alignment_stats = AlignmentStats{};
        return ce_pull_exchange_partitioned(
            input, std::move(pattern), ctx, stream, ctx.ce_pull_options);
    }
    std::vector<CommPattern> patterns;
    std::vector<int> stat_slots;
    std::vector<gamax::exec::MessageSegment> trace_segments;
    if (alignment == AlignmentPolicy::On) {
        if (backend != Backend::NCCL) {
            throw std::invalid_argument(
                "exchange_partitions: alignment requires the NCCL backend");
        }
        auto split = align_comm_buffers_by_item_alignment(
            pattern, table_row_alignment(input), ctx);
        // The aligned helper returns {pre, main, post}, with receive offsets
        // already arranged as [all main | all pre | all post].
        patterns.push_back(std::move(split[1]));
        patterns.push_back(std::move(split[0]));
        patterns.push_back(std::move(split[2]));
        stat_slots = {0, 1, 2};
        trace_segments = {
            gamax::exec::MessageSegment::Main,
            gamax::exec::MessageSegment::Pre,
            gamax::exec::MessageSegment::Post};
    } else {
        complete_receive_pattern(pattern, ctx);
        patterns.push_back(std::move(pattern));
        trace_segments = {gamax::exec::MessageSegment::Whole};
    }

    if (alignment_stats) {
        *alignment_stats = AlignmentStats{};
        if (alignment == AlignmentPolicy::On) {
            alignment_stats->applied = true;
            alignment_stats->rows = {{0, 0, 0}};
            alignment_stats->bytes = {{0, 0, 0}};
            for (std::size_t segment = 0; segment < patterns.size(); ++segment) {
                alignment_stats->rows[stat_slots[segment]] =
                    std::accumulate(
                        patterns[segment].send_counts.begin(),
                        patterns[segment].send_counts.end(),
                        std::int64_t{0});
            }
        }
    }

    const size_type recv_rows = receive_extent(patterns);
    std::vector<gamax::column_ptr> columns;
    columns.reserve(static_cast<std::size_t>(input.num_columns()));
    for (size_type column = 0; column < input.num_columns(); ++column) {
        const auto send = input.get_column(column);
        if (alignment_stats && alignment_stats->applied) {
            const auto row_bytes =
                send.type().id() == gamax::type_id::STRING
                    ? sizeof(size_type)
                    : gamax::size_of(send.type());
            for (int slot = 0; slot < 3; ++slot) {
                alignment_stats->bytes[slot] +=
                    alignment_stats->rows[slot] * row_bytes;
            }
        }
        if (send.type().id() == gamax::type_id::STRING) {
            columns.push_back(exchange_string_column_patterns(
                send, patterns, stat_slots, column, trace_segments,
                alignment_stats && alignment_stats->applied
                    ? alignment_stats : nullptr,
                ctx, stream, backend));
            continue;
        }
        auto recv = gamax::make_column(send.type(), recv_rows, stream);
        for (std::size_t segment = 0; segment < patterns.size(); ++segment) {
            auto& fixed_pattern = patterns[segment];
            exchange_column(
                send, recv->mutable_view(), fixed_pattern,
                ctx, stream, backend,
                {column, send.type(),
                 gamax::exec::MessageComponent::FixedValue,
                 trace_segments[segment]});
        }
        columns.push_back(std::move(recv));
    }
    return gamax::make_table(std::move(columns));
}

std::unique_ptr<gamax::table> replicate(
    gamax::table_view in,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend,
    AlignmentPolicy alignment,
    AlignmentStats* alignment_stats)
{
    if (alignment == AlignmentPolicy::On && backend != Backend::NCCL) {
        throw std::invalid_argument(
            "replicate: alignment requires the NCCL backend");
    }
    if (backend == Backend::CE_PULL) {
        if (alignment_stats) *alignment_stats = AlignmentStats{};
        return ce_pull_allgather(in, ctx, stream, ctx.ce_pull_options);
    }
    auto patterns = make_allgather_patterns(
        in.num_rows(), table_row_alignment(in), ctx, alignment);
    std::vector<int> stat_slots;
    std::vector<gamax::exec::MessageSegment> trace_segments;
    if (alignment == AlignmentPolicy::On) {
        stat_slots = {0, 2};
        trace_segments = {
            gamax::exec::MessageSegment::Main,
            gamax::exec::MessageSegment::Post};
    } else {
        trace_segments = {gamax::exec::MessageSegment::Whole};
    }
    if (alignment_stats) {
        *alignment_stats = AlignmentStats{};
        if (alignment == AlignmentPolicy::On) {
            alignment_stats->applied = true;
            alignment_stats->rows = {{0, 0, 0}};
            alignment_stats->bytes = {{0, 0, 0}};
            for (std::size_t segment = 0; segment < patterns.size(); ++segment) {
                alignment_stats->rows[stat_slots[segment]] =
                    patterns[segment].send_counts[ctx.rank];
            }
        }
    }
    const size_type recv_rows = receive_extent(patterns);
    std::vector<gamax::column_ptr> columns;
    columns.reserve(static_cast<std::size_t>(in.num_columns()));
    for (size_type column = 0; column < in.num_columns(); ++column) {
        const auto send = in.get_column(column);
        if (alignment_stats && alignment_stats->applied) {
            const auto row_bytes =
                send.type().id() == gamax::type_id::STRING
                    ? sizeof(size_type)
                    : gamax::size_of(send.type());
            for (int slot = 0; slot < 3; ++slot) {
                alignment_stats->bytes[slot] +=
                    alignment_stats->rows[slot] * row_bytes;
            }
        }
        if (send.type().id() == gamax::type_id::STRING) {
            columns.push_back(broadcast_string_column_patterns(
                send, patterns, stat_slots, column, trace_segments,
                alignment_stats && alignment_stats->applied
                    ? alignment_stats : nullptr,
                ctx, stream, backend));
            continue;
        }
        auto recv = gamax::make_column(send.type(), recv_rows, stream);
        for (std::size_t segment = 0; segment < patterns.size(); ++segment) {
            auto& pattern = patterns[segment];
            broadcast_column(
                send, recv->mutable_view(), pattern, ctx, stream, backend,
                {column, send.type(),
                 gamax::exec::MessageComponent::FixedValue,
                 trace_segments[segment]});
        }
        columns.push_back(std::move(recv));
    }
    return gamax::make_table(std::move(columns));
}

std::unique_ptr<gamax::table> gather_to_root(
    gamax::table_view in,
    int root,
    const Context& ctx,
    cudaStream_t stream,
    Backend backend)
{
    if (root < 0 || root >= ctx.size) {
        throw std::out_of_range("gather_to_root: root rank out of range");
    }
    return exchange_partitions(
        make_gather_partition(in, ctx, root),
        ctx, stream, backend);
}

} // namespace alps::collectives
