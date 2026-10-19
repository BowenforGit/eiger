// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/operators/distributed_relational.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <nccl.h>
#include <cuda/std/type_traits>
#include <mpi.h>

#include "alps/collectives/ce_pull_shuffle.hpp"
#include "alps/collectives/table_exchange.hpp"
#include "alps/convert.hpp"
#include "alps/error.hpp"
#include "alps/operators/column_overlap_eq_join.hpp"
#include "exec/step_profile.hpp"
#include "operators/groupby/groupby_hash.hpp"
#include "operators/join/materialize.hpp"
#include "operators/join/semi_join.hpp"
#include "string/string_column.hpp"
#include "types/type_dispatcher.hpp"
#include "utils/sequence.cuh"
#include "utils/transform.hpp"

namespace alps::operators {
namespace {

using AggId = gamax::groupby::aggregation_id;
using gamax::size_type;

ColumnJoinRuntime& column_overlap_runtime(
    const alps::Context& context,
    bool communication_high_priority)
{
    if (!context.column_join_runtime) {
        context.column_join_runtime = std::make_shared<ColumnJoinRuntime>(
            communication_high_priority);
    } else if (context.column_join_runtime->communication_high_priority !=
               communication_high_priority) {
        throw std::invalid_argument(
            "ColumnOverlap communication priority cannot change within one Context");
    }
    return *context.column_join_runtime;
}

std::int64_t logical_column_bytes(gamax::column_view column) noexcept
{
    if (column.type().id() == gamax::type_id::STRING) {
        return static_cast<std::int64_t>(column.auxiliary_bytes()) +
            static_cast<std::int64_t>(column.size() + 1) *
                static_cast<std::int64_t>(sizeof(gamax::size_type));
    }
    return static_cast<std::int64_t>(column.size()) *
        static_cast<std::int64_t>(gamax::size_of(column.type()));
}

gamax::exec::PhaseMetadata column_exchange_metadata(
    const char* name,
    const char* side,
    size_type ordinal,
    alps::collectives::Backend backend =
        alps::collectives::Backend::NCCL)
{
    gamax::exec::PhaseMetadata metadata{
        name,
        gamax::exec::PhaseCategory::Communication,
        gamax::exec::CommunicationPattern::Shuffle,
        backend == alps::collectives::Backend::CE_PULL
            ? "cuda_memcpy_batch_pull" : "nccl_send_recv"};
    metadata.column_side = side;
    metadata.column_ordinal = static_cast<std::int64_t>(ordinal);
    return metadata;
}

std::vector<gamax::column_view> column_views(gamax::table_view table)
{
    std::vector<gamax::column_view> result;
    result.reserve(static_cast<std::size_t>(table.num_columns()));
    for (size_type ordinal = 0; ordinal < table.num_columns(); ++ordinal) {
        result.push_back(table.get_column(ordinal));
    }
    return result;
}

bool rank_wide_stage_succeeded(
    bool local_success, const alps::Context& context)
{
    int local = local_success ? 1 : 0;
    int global = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local, &global, 1, MPI_INT, MPI_MIN, context.mpi_comm));
    return global != 0;
}

[[noreturn]] void throw_rank_wide_stage_failure(
    std::exception_ptr local_error,
    const char* operator_name,
    const char* operation)
{
    if (local_error) std::rethrow_exception(local_error);
    throw std::runtime_error(
        std::string(operator_name) + ": another rank failed during " +
        operation);
}

alps::collectives::CePullOptions full_staging_options(
    std::size_t operations,
    bool communication_high_priority,
    alps::collectives::CePullSchedule schedule,
    int requested_slots,
    const char* operator_name)
{
    if (requested_slots < 1) {
        throw std::invalid_argument(
            std::string(operator_name) + ": slot count must be positive");
    }
    if (operations > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::string(operator_name) +
            ": too many full-staging operations");
    }
    alps::collectives::CePullOptions options;
    options.slots = std::max(requested_slots, static_cast<int>(operations));
    options.high_priority_transfer_stream = communication_high_priority;
    options.schedule = schedule;
    options.full_staging = true;
    return options;
}

void set_column_lifetime_stream(gamax::column& column, cudaStream_t stream)
{
    column.buffer().set_stream(stream);
    if (column.type().id() == gamax::type_id::STRING) {
        static_cast<gamax::string_column&>(column).meta_buffer().set_stream(
            stream);
    }
}

void set_table_lifetime_stream(gamax::table& table, cudaStream_t stream)
{
    for (size_type column = 0; column < table.num_columns(); ++column) {
        set_column_lifetime_stream(table.get_column(column), stream);
    }
}

std::vector<size_type> key_first_ordinals(
    gamax::table_view input, size_type key)
{
    std::vector<size_type> result;
    result.reserve(static_cast<std::size_t>(input.num_columns()));
    result.push_back(key);
    for (size_type column = 0; column < input.num_columns(); ++column) {
        if (column != key) result.push_back(column);
    }
    return result;
}

const char* exchange_primitive(alps::collectives::Backend backend) {
    if (backend == alps::collectives::Backend::CE) {
        return "fabric_copy_engine_push";
    }
    if (backend == alps::collectives::Backend::CE_PULL) {
        return "fabric_copy_engine_pull";
    }
    return "nccl_send_recv";
}

const char* shuffle_primitive(
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy alignment)
{
    if (backend == alps::collectives::Backend::CE) {
        return "fabric_copy_engine_push";
    }
    if (backend == alps::collectives::Backend::CE_PULL) {
        return "fabric_copy_engine_pull";
    }
    return alignment == alps::collectives::AlignmentPolicy::On
        ? "nccl_send_recv_aligned_split" : "nccl_send_recv";
}

const char* replication_primitive(
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy alignment)
{
    if (backend == alps::collectives::Backend::CE) {
        return "fabric_copy_engine_push";
    }
    if (backend == alps::collectives::Backend::CE_PULL) {
        return "fabric_copy_engine_pull";
    }
    return alignment == alps::collectives::AlignmentPolicy::On
        ? "nccl_broadcast_aligned_split" : "nccl_broadcast";
}

void set_alignment_stats(
    gamax::exec::PhaseRange& phase,
    const alps::collectives::AlignmentStats& stats)
{
    if (!stats.applied) return;
    phase.set_alignment(
        stats.rows[0], stats.rows[1], stats.rows[2],
        stats.bytes[0], stats.bytes[1], stats.bytes[2]);
}

struct OriginalAggState {
    bool average {false};
    size_type state_count {1};
};

struct ExpandedAggs {
    std::vector<gamax::phys::AggSpec> partial_specs;
    std::vector<gamax::phys::AggSpec> merge_specs;
    std::vector<OriginalAggState> originals;
};

ExpandedAggs expand_aggregates(
    const std::vector<gamax::phys::AggSpec>& specs)
{
    ExpandedAggs result;
    result.originals.reserve(specs.size());
    for (const auto& spec : specs) {
        switch (spec.agg) {
            case AggId::SUM:
                result.partial_specs.push_back(spec);
                result.merge_specs.push_back({
                    static_cast<size_type>(result.merge_specs.size()), AggId::SUM});
                result.originals.push_back({false, 1});
                break;
            case AggId::MIN:
                result.partial_specs.push_back(spec);
                result.merge_specs.push_back({
                    static_cast<size_type>(result.merge_specs.size()), AggId::MIN});
                result.originals.push_back({false, 1});
                break;
            case AggId::MAX:
                result.partial_specs.push_back(spec);
                result.merge_specs.push_back({
                    static_cast<size_type>(result.merge_specs.size()), AggId::MAX});
                result.originals.push_back({false, 1});
                break;
            case AggId::COUNT_ALL:
                result.partial_specs.push_back({-1, AggId::COUNT_ALL});
                result.merge_specs.push_back({
                    static_cast<size_type>(result.merge_specs.size()), AggId::SUM});
                result.originals.push_back({false, 1});
                break;
            case AggId::AVG:
                result.partial_specs.push_back({spec.column, AggId::SUM});
                result.merge_specs.push_back({
                    static_cast<size_type>(result.merge_specs.size()), AggId::SUM});
                result.partial_specs.push_back({-1, AggId::COUNT_ALL});
                result.merge_specs.push_back({
                    static_cast<size_type>(result.merge_specs.size()), AggId::SUM});
                result.originals.push_back({true, 2});
                break;
            default:
                throw std::runtime_error(
                    "distributed aggregate does not support ANY/unknown state merging");
        }
    }
    return result;
}

struct MakeIdentityColumn {
    template <typename T>
    gamax::column_ptr operator()(
        AggId operation, cudaStream_t stream) const
    {
        T value {};
        if (operation == AggId::MIN) {
            value = std::numeric_limits<T>::max();
        } else if (operation == AggId::MAX) {
            value = std::numeric_limits<T>::lowest();
        }
        return gamax::make_column_from_host(&value, 1, stream);
    }
};

gamax::column_ptr make_identity_column(
    gamax::data_type type, AggId operation, cudaStream_t stream)
{
    return gamax::type_dispatcher(
        type, MakeIdentityColumn{}, operation, stream);
}

std::unique_ptr<gamax::table> empty_scalar_partials(
    gamax::table_view input,
    const std::vector<gamax::phys::AggSpec>& specs,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> columns;
    columns.reserve(specs.size());
    for (const auto& spec : specs) {
        if (spec.agg == AggId::COUNT_ALL) {
            columns.push_back(make_identity_column(
                gamax::data_type{gamax::type_id::INT64}, AggId::SUM, stream));
            continue;
        }
        const auto output_type = gamax::groupby::agg_output_data_type(
            spec.agg, input.get_column(spec.column).type());
        columns.push_back(make_identity_column(output_type, spec.agg, stream));
    }
    return gamax::make_table(std::move(columns));
}

ncclRedOp_t reduction_op(AggId operation) {
    switch (operation) {
        case AggId::SUM: return ncclSum;
        case AggId::MIN: return ncclMin;
        case AggId::MAX: return ncclMax;
        default:
            throw std::runtime_error("invalid distributed reduction operation");
    }
}

std::unique_ptr<gamax::table> allreduce_states(
    gamax::table_view local,
    const std::vector<gamax::phys::AggSpec>& merge_specs,
    const alps::Context& context,
    cudaStream_t stream)
{
    if (local.num_columns() != static_cast<size_type>(merge_specs.size()) ||
        local.num_rows() != 1) {
        throw std::runtime_error("allreduce_states expects one scalar row per state");
    }
    std::vector<gamax::column_ptr> columns;
    columns.reserve(merge_specs.size());
    for (size_type state = 0; state < local.num_columns(); ++state) {
        columns.push_back(gamax::make_column(
            local.get_column(state).type(), 1, stream));
    }
    ALPS_NCCLCHECK(ncclGroupStart());
    for (size_type state = 0; state < local.num_columns(); ++state) {
        const auto input = local.get_column(state);
        if (context.message_trace_sink) {
            context.message_trace_sink->record_message(
                gamax::exec::MessageApi::NcclAllReduce, -1, -1,
                input.type(), 1,
                static_cast<std::int64_t>(gamax::size_of(input.type())),
                {state, input.type(),
                 gamax::exec::MessageComponent::ScalarState,
                 gamax::exec::MessageSegment::Whole});
        }
        ALPS_NCCLCHECK(ncclAllReduce(
            input.data(), columns[state]->data(), 1,
            alps::convert::to_nccl_type(input.type()),
            reduction_op(merge_specs[state].agg), context.nccl_comm, stream));
    }
    ALPS_NCCLCHECK(ncclGroupEnd());
    return gamax::make_table(std::move(columns));
}

template <typename SumT>
__global__ void finalize_average_kernel(
    const SumT* sums, const int64_t* counts, double* output, size_type rows)
{
    const size_type row =
        static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    output[row] = counts[row] == 0
        ? 0.0
        : static_cast<double>(sums[row]) / static_cast<double>(counts[row]);
}

struct FinalizeAverage {
    template <typename SumT>
    gamax::column_ptr operator()(
        gamax::column_view sums,
        gamax::column_view counts,
        cudaStream_t stream) const
    {
        if (counts.type().id() != gamax::type_id::INT64 ||
            counts.size() != sums.size()) {
            throw std::runtime_error("invalid distributed AVG state");
        }
        auto output = gamax::make_column<double>(sums.size(), stream);
        if (sums.size() > 0) {
            constexpr int threads = 256;
            const int blocks = static_cast<int>((sums.size() + threads - 1) / threads);
            finalize_average_kernel<<<blocks, threads, 0, stream>>>(
                sums.data<SumT>(), counts.data<int64_t>(),
                output->data<double>(), sums.size());
            ALPS_CUDACHECK(cudaGetLastError());
        }
        return output;
    }
};

gamax::column_ptr finalize_average(
    gamax::column_view sums,
    gamax::column_view counts,
    cudaStream_t stream)
{
    return gamax::type_dispatcher(
        sums.type(), FinalizeAverage{}, sums, counts, stream);
}

std::unique_ptr<gamax::table> finalize_states(
    std::unique_ptr<gamax::table> states,
    size_type key_count,
    const ExpandedAggs& layout,
    cudaStream_t stream)
{
    std::vector<gamax::column_ptr> output;
    output.reserve(static_cast<std::size_t>(key_count) + layout.originals.size());
    for (size_type key = 0; key < key_count; ++key) {
        output.push_back(states->remove_column(0));
    }
    for (const auto& original : layout.originals) {
        if (!original.average) {
            output.push_back(states->remove_column(0));
            continue;
        }
        auto sum = states->remove_column(0);
        auto count = states->remove_column(0);
        output.push_back(finalize_average(sum->view(), count->view(), stream));
    }
    return gamax::make_table(std::move(output));
}

std::vector<size_type> sequence(size_type count, size_type begin = 0) {
    std::vector<size_type> result(static_cast<std::size_t>(count));
    for (size_type i = 0; i < count; ++i) result[i] = begin + i;
    return result;
}

constexpr int kMaxRangeSortKeys = 8;

struct RangeOrder {
    bool ascending[kMaxRangeSortKeys] {};
};

__device__ int compare_string_values(
    gamax::column_view left,
    gamax::column_view right,
    size_type left_row,
    size_type right_row)
{
    const auto* left_offsets =
        static_cast<const size_type*>(left.auxiliary_data());
    const auto* right_offsets =
        static_cast<const size_type*>(right.auxiliary_data());
    const size_type left_begin = left_offsets[left_row];
    const size_type right_begin = right_offsets[right_row];
    const size_type left_size = left_offsets[left_row + 1] - left_begin;
    const size_type right_size = right_offsets[right_row + 1] - right_begin;
    const size_type common = left_size < right_size ? left_size : right_size;
    const char* left_data = left.data<char>() + left_begin;
    const char* right_data = right.data<char>() + right_begin;
    for (size_type i = 0; i < common; ++i) {
        if (left_data[i] < right_data[i]) return -1;
        if (left_data[i] > right_data[i]) return 1;
    }
    if (left_size < right_size) return -1;
    if (left_size > right_size) return 1;
    return 0;
}

struct CompareAcrossColumns {
    template <typename T>
    __device__ int operator()(
        gamax::column_view left,
        gamax::column_view right,
        size_type left_row,
        size_type right_row) const
    {
        if constexpr (cuda::std::is_same_v<T, gamax::string_type_tag>) {
            return compare_string_values(left, right, left_row, right_row);
        } else {
            const T lhs = left.data<T>()[left_row];
            const T rhs = right.data<T>()[right_row];
            if (lhs < rhs) return -1;
            if (lhs > rhs) return 1;
            return 0;
        }
    }
};

__device__ int compare_rows_in_order(
    gamax::experimental::table_device_view rows,
    gamax::experimental::table_device_view splitters,
    const bool* ascending,
    size_type row,
    size_type splitter)
{
    for (size_type key = 0; key < rows.num_columns(); ++key) {
        auto left = rows.get_column(key);
        auto right = splitters.get_column(key);
        int comparison = gamax::type_dispatcher_all(
            left.type(), CompareAcrossColumns{},
            left, right, row, splitter);
        if (comparison != 0) return ascending[key] ? comparison : -comparison;
    }
    return 0;
}

__global__ void range_destinations_kernel(
    gamax::experimental::table_device_view rows,
    gamax::experimental::table_device_view splitters,
    RangeOrder order,
    int32_t* destinations,
    size_type row_count,
    int splitter_count)
{
    const size_type row =
        static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    int low = 0;
    int high = splitter_count;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (compare_rows_in_order(
                rows, splitters, order.ascending, row, middle) <= 0) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    destinations[row] = low;
}

std::unique_ptr<gamax::column> index_column(
    const std::vector<size_type>& indices, cudaStream_t stream)
{
    static size_type empty = 0;
    return gamax::make_column_from_host(
        indices.empty() ? &empty : const_cast<size_type*>(indices.data()),
        static_cast<size_type>(indices.size()), stream);
}

std::unique_ptr<gamax::column> make_range_destinations(
    gamax::table_view rows,
    gamax::table_view splitters,
    const std::vector<gamax::phys::SortSpec>& keys,
    cudaStream_t stream)
{
    if (keys.empty() || keys.size() > kMaxRangeSortKeys) {
        throw std::invalid_argument(
            "distributed sort requires between 1 and 8 sort keys");
    }
    std::vector<size_type> ordinals;
    ordinals.reserve(keys.size());
    RangeOrder order;
    for (std::size_t key = 0; key < keys.size(); ++key) {
        ordinals.push_back(keys[key].column);
        order.ascending[key] =
            keys[key].order == gamax::sort_order::ASCENDING;
    }
    auto row_keys = rows.select_columns(ordinals);
    auto splitter_keys = splitters.select_columns(ordinals);
    auto row_device = row_keys.device_view(stream);
    auto splitter_device = splitter_keys.device_view(stream);
    auto destinations = gamax::make_column<int32_t>(rows.num_rows(), stream);
    if (rows.num_rows() > 0) {
        constexpr int threads = 256;
        const int blocks = static_cast<int>((rows.num_rows() + threads - 1) / threads);
        range_destinations_kernel<<<blocks, threads, 0, stream>>>(
            *row_device, *splitter_device, order,
            destinations->data<int32_t>(), rows.num_rows(),
            static_cast<int>(splitters.num_rows()));
        ALPS_CUDACHECK(cudaGetLastError());
    }
    return destinations;
}

} // namespace

std::unique_ptr<gamax::table> scalar_aggregate(
    gamax::table_view input,
    const std::vector<gamax::phys::AggSpec>& specs,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context)
{
    const auto layout = expand_aggregates(specs);
    std::unique_ptr<gamax::table> local;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"aggregate.local_partial", gamax::exec::PhaseCategory::Compute});
        local = input.num_rows() == 0
            ? empty_scalar_partials(
                input, layout.partial_specs, physical_context.stream)
            : gamax::phys::scalar_aggregate(
                input, layout.partial_specs, physical_context);
    }
    std::unique_ptr<gamax::table> global;
    {
        const auto bytes = gamax::exec::table_data_bytes(local->view());
        const auto rows = local->num_rows();
        gamax::exec::PhaseRange phase(
            physical_context,
            {"reduce.allreduce_states",
             gamax::exec::PhaseCategory::Communication,
             gamax::exec::CommunicationPattern::Reduce,
             "nccl_allreduce"});
        global = allreduce_states(
            local->view(), layout.merge_specs, distributed_context,
            physical_context.stream);
        phase.set_transfer(
            bytes, gamax::exec::table_data_bytes(global->view()),
            rows, global->num_rows(), distributed_context.size - 1);
    }
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"aggregate.finalize", gamax::exec::PhaseCategory::Compute});
        return finalize_states(
            std::move(global), 0, layout, physical_context.stream);
    }
}

std::unique_ptr<gamax::table> group_by(
    gamax::table_view input,
    size_type n_keys,
    const std::vector<gamax::phys::AggSpec>& specs,
    const gamax::groupby::groupby_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy shuffle_alignment)
{
    if (n_keys <= 0 || n_keys > input.num_columns()) {
        throw std::invalid_argument("distributed group_by: invalid key count");
    }
    const auto layout = expand_aggregates(specs);
    std::unique_ptr<gamax::table> local;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"group_by.local_partial", gamax::exec::PhaseCategory::Compute});
        local = gamax::phys::group_by(
            input.select_columns(sequence(n_keys)),
            input.select_columns(sequence(input.num_columns() - n_keys, n_keys)),
            layout.partial_specs, physical_context, hint);
    }

    alps::collectives::PartitionedTable partitioned;
    {
        const auto bytes = gamax::exec::table_data_bytes(local->view());
        gamax::exec::PhaseRange phase(
            physical_context,
            {"shuffle.hash_partition",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "hash_partition"});
        partitioned = alps::collectives::hash_partition(
            local->view(), sequence(n_keys), distributed_context,
            physical_context.stream);
        phase.set_transfer(
            bytes, gamax::exec::table_data_bytes(partitioned.table->view()),
            local->num_rows(), partitioned.table->num_rows(),
            distributed_context.size - 1);
    }
    std::unique_ptr<gamax::table> exchanged;
    {
        const auto bytes = gamax::exec::table_data_bytes(partitioned.table->view());
        const auto rows = partitioned.table->num_rows();
        gamax::exec::PhaseRange phase(
            physical_context,
            {"shuffle.exchange",
             gamax::exec::PhaseCategory::Communication,
             gamax::exec::CommunicationPattern::Shuffle,
             shuffle_primitive(backend, shuffle_alignment)});
        alps::collectives::AlignmentStats alignment_stats;
        exchanged = alps::collectives::exchange_partitions(
            std::move(partitioned), distributed_context,
            physical_context.stream, backend, shuffle_alignment,
            &alignment_stats);
        set_alignment_stats(phase, alignment_stats);
        phase.set_transfer(
            bytes, gamax::exec::table_data_bytes(exchanged->view()),
            rows, exchanged->num_rows(), distributed_context.size - 1);
    }

    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"group_by.merge_finalize", gamax::exec::PhaseCategory::Compute});
        auto final_states = gamax::phys::group_by(
            exchanged->view().select_columns(sequence(n_keys)),
            exchanged->view().select_columns(sequence(
                exchanged->num_columns() - n_keys, n_keys)),
            layout.merge_specs, physical_context, hint);
        return finalize_states(
            std::move(final_states), n_keys, layout, physical_context.stream);
    }
}

std::unique_ptr<gamax::table> column_overlap_group_by_ce_pull(
    gamax::table& local,
    size_type n_keys,
    const ExpandedAggs& layout,
    const gamax::groupby::groupby_hint& hint,
    const alps::collectives::HashPartitionPlan& plan,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    bool communication_high_priority,
    alps::collectives::CePullSchedule ce_pull_schedule,
    int ce_pull_slots)
{
    auto& runtime = column_overlap_runtime(
        distributed_context, communication_high_priority);
    runtime.ensure_dependency_events(4);
    const auto entry = runtime.dependency_events[0];
    const auto final = runtime.dependency_events[1];
    const auto all_states_prepared = runtime.dependency_events[2];
    const auto release_states = runtime.dependency_events[3];

    ALPS_CUDACHECK(cudaEventRecord(entry, physical_context.stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.preparation_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(runtime.join_stream, entry, 0));

    const auto operations = static_cast<std::size_t>(local.num_columns());
    const auto ce_options = full_staging_options(
        operations, communication_high_priority, ce_pull_schedule,
        ce_pull_slots, "column_overlap_group_by");
    std::unique_ptr<alps::collectives::CePullShuffleSession> session;
    std::exception_ptr local_error;
    try {
        alps::collectives::begin_ce_pull_full_staging_batch(
            distributed_context, operations, runtime.preparation_stream,
            ce_options);
        session = alps::collectives::begin_ce_pull_shuffle(
            plan, column_views(local.view()), distributed_context,
            runtime.preparation_stream, ce_options);
        alps::collectives::seal_ce_pull_full_staging_batch(
            distributed_context);
    } catch (...) {
        local_error = std::current_exception();
    }
    if (!rank_wide_stage_succeeded(!local_error, distributed_context)) {
        alps::collectives::cancel_ce_pull_full_staging_batch(
            distributed_context);
        throw_rank_wide_stage_failure(
            local_error, "column_overlap_group_by",
            "CE-pull full-staging session planning");
    }

    const auto transfer_stream =
        alps::collectives::ce_pull_transfer_stream(distributed_context);
    ALPS_CUDACHECK(cudaStreamWaitEvent(transfer_stream, entry, 0));

    for (size_type column = 0; column < n_keys; ++column) {
        gamax::exec::PhaseMetadata prepare_metadata{
            "group_by.column_overlap.key_partition_apply",
            gamax::exec::PhaseCategory::Compute,
            gamax::exec::CommunicationPattern::Shuffle,
            "split_gather"};
        prepare_metadata.column_side = "key";
        prepare_metadata.column_ordinal = column;
        {
            gamax::exec::PhaseRange phase(
                physical_context.profile_sink, runtime.preparation_stream,
                std::move(prepare_metadata));
            session->prepare_column(static_cast<std::size_t>(column));
        }

        const auto& output = session->output(
            static_cast<std::size_t>(column));
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, transfer_stream,
            column_exchange_metadata(
                "group_by.column_overlap.key_exchange", "key", column,
                alps::collectives::Backend::CE_PULL));
        session->enqueue_pull(static_cast<std::size_t>(column));
        phase.set_transfer(
            logical_column_bytes(local.get_column(column).view()),
            logical_column_bytes(output.view()),
            local.get_column(column).size(), output.size(),
            distributed_context.size - 1);
    }

    for (size_type column = n_keys;
         column < local.num_columns(); ++column) {
        gamax::exec::PhaseMetadata prepare_metadata{
            "group_by.column_overlap.state_partition_apply",
            gamax::exec::PhaseCategory::Compute,
            gamax::exec::CommunicationPattern::Shuffle,
            "split_gather"};
        prepare_metadata.column_side = "state";
        prepare_metadata.column_ordinal = column;
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, runtime.preparation_stream,
            std::move(prepare_metadata));
        session->prepare_column(static_cast<std::size_t>(column));
    }
    ALPS_CUDACHECK(cudaEventRecord(
        all_states_prepared, runtime.preparation_stream));

    std::vector<gamax::column_ptr> exchanged(operations);
    for (size_type column = 0; column < n_keys; ++column) {
        const auto index = static_cast<std::size_t>(column);
        exchanged[index] = session->release_output(
            index, runtime.join_stream);
        session->wait_on(index, runtime.join_stream);
    }

    // State split/gathers must be submitted before group-ID construction.
    // Releasing the state pulls from the join stream lets the copy engine run
    // beside compute_group_ids while keeping every state lease alive.
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, all_states_prepared, 0));
    ALPS_CUDACHECK(cudaEventRecord(release_states, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        transfer_stream, release_states, 0));
    for (size_type column = n_keys;
         column < local.num_columns(); ++column) {
        const auto index = static_cast<std::size_t>(column);
        const auto& output = session->output(index);
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, transfer_stream,
            column_exchange_metadata(
                "group_by.column_overlap.state_exchange", "state", column,
                alps::collectives::Backend::CE_PULL));
        session->enqueue_pull(index);
        phase.set_transfer(
            logical_column_bytes(local.get_column(column).view()),
            logical_column_bytes(output.view()),
            local.get_column(column).size(), output.size(),
            distributed_context.size - 1);
    }
    alps::collectives::finish_ce_pull_full_staging_batch(
        distributed_context);
    for (size_type column = n_keys;
         column < local.num_columns(); ++column) {
        const auto index = static_cast<std::size_t>(column);
        exchanged[index] = session->release_output(
            index, runtime.join_stream);
    }

    std::vector<gamax::column_view> key_columns;
    key_columns.reserve(static_cast<std::size_t>(n_keys));
    for (size_type column = 0; column < n_keys; ++column) {
        key_columns.push_back(
            exchanged[static_cast<std::size_t>(column)]->view());
    }
    gamax::table_view exchanged_keys(key_columns);

    gamax::groupby::hash::group_id_plan group_ids;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, runtime.join_stream,
            {"group_by.column_overlap.group_ids",
             gamax::exec::PhaseCategory::Compute});
        if (exchanged_keys.num_rows() > 0) {
            auto group_hint = hint;
            group_ids = gamax::groupby::hash::compute_group_ids(
                exchanged_keys, group_hint, runtime.join_stream);
        }
    }

    std::vector<gamax::column_view> state_columns;
    state_columns.reserve(operations - static_cast<std::size_t>(n_keys));
    for (size_type column = n_keys;
         column < local.num_columns(); ++column) {
        const auto index = static_cast<std::size_t>(column);
        session->wait_on(index, runtime.join_stream);
        state_columns.push_back(exchanged[index]->view());
    }
    gamax::table_view exchanged_states(state_columns);

    std::unique_ptr<gamax::table> final_states;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, runtime.join_stream,
            {"group_by.column_overlap.merge_finalize",
             gamax::exec::PhaseCategory::Compute});
        if (exchanged_keys.num_rows() == 0) {
            gamax::phys::Context worker_context{
                runtime.join_stream, physical_context.profile_sink};
            final_states = gamax::phys::group_by(
                exchanged_keys, exchanged_states, layout.merge_specs,
                worker_context, hint);
        } else {
            std::vector<gamax::groupby::aggregation_request> requests;
            requests.reserve(layout.merge_specs.size());
            for (const auto& spec : layout.merge_specs) {
                requests.emplace_back(spec.column, spec.agg);
            }
            final_states = gamax::groupby::hash::groupby_with_plan(
                exchanged_keys, exchanged_states, requests, group_ids,
                runtime.join_stream);
        }
        final_states = finalize_states(
            std::move(final_states), n_keys, layout, runtime.join_stream);
    }

    ALPS_CUDACHECK(cudaEventRecord(final, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(physical_context.stream, final, 0));
    set_table_lifetime_stream(*final_states, physical_context.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(physical_context.stream));
    return final_states;
}

std::unique_ptr<gamax::table> column_overlap_group_by(
    gamax::table_view input,
    size_type n_keys,
    const std::vector<gamax::phys::AggSpec>& specs,
    const gamax::groupby::groupby_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    bool communication_high_priority,
    alps::collectives::Backend exchange_backend,
    alps::collectives::CePullSchedule ce_pull_schedule,
    int ce_pull_slots)
{
    if (n_keys <= 0 || n_keys > input.num_columns()) {
        throw std::invalid_argument(
            "column_overlap_group_by: invalid key count");
    }
    if (hint.groupby_algo != gamax::groupby::HASH || hint.enable_skf) {
        throw std::invalid_argument(
            "column_overlap_group_by: requires HASH without SKF");
    }
    if (exchange_backend != alps::collectives::Backend::NCCL &&
        exchange_backend != alps::collectives::Backend::CE_PULL) {
        throw std::invalid_argument(
            "column_overlap_group_by: exchange backend must be NCCL or "
            "CE_PULL");
    }

    const auto layout = expand_aggregates(specs);
    std::unique_ptr<gamax::table> local;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"group_by.local_partial", gamax::exec::PhaseCategory::Compute});
        local = gamax::phys::group_by(
            input.select_columns(sequence(n_keys)),
            input.select_columns(sequence(input.num_columns() - n_keys, n_keys)),
            layout.partial_specs, physical_context, hint);
    }

    alps::collectives::HashPartitionPlan plan;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"group_by.column_overlap.partition_plan",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "hash_partition_plan"});
        plan = alps::collectives::make_hash_partition_plan(
            local->view().select_columns(sequence(n_keys)),
            distributed_context, physical_context.stream);
        alps::collectives::complete_receive_pattern(
            plan.pattern, distributed_context);
    }

    if (exchange_backend == alps::collectives::Backend::CE_PULL) {
        return column_overlap_group_by_ce_pull(
            *local, n_keys, layout, hint, plan, distributed_context,
            physical_context, communication_high_priority,
            ce_pull_schedule, ce_pull_slots);
    }

    auto& runtime = column_overlap_runtime(
        distributed_context, communication_high_priority);
    const auto column_count =
        static_cast<std::size_t>(local->num_columns());
    runtime.ensure_dependency_events(4 + 2 * column_count);
    std::size_t event_index = 0;
    const auto entry = runtime.dependency_events[event_index++];
    const auto final = runtime.dependency_events[event_index++];
    const auto keys_received = runtime.dependency_events[event_index++];
    const auto all_received = runtime.dependency_events[event_index++];
    std::vector<cudaEvent_t> prepared(column_count);
    std::vector<cudaEvent_t> received_events(column_count);
    for (auto& event : prepared) {
        event = runtime.dependency_events[event_index++];
    }
    for (auto& event : received_events) {
        event = runtime.dependency_events[event_index++];
    }

    ALPS_CUDACHECK(cudaEventRecord(entry, physical_context.stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.preparation_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.communication_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(runtime.join_stream, entry, 0));

    std::vector<gamax::column_ptr> partitioned(column_count);
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.preparation_stream,
            {"group_by.column_overlap.key_partition_apply",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "apply_partition"});
        for (size_type column = 0; column < n_keys; ++column) {
            const auto index = static_cast<std::size_t>(column);
            partitioned[index] = alps::collectives::apply_partition(
                local->get_column(column).view(), plan,
                runtime.preparation_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                prepared[index], runtime.preparation_stream));
        }
    }
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.preparation_stream,
            {"group_by.column_overlap.state_partition_apply",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "apply_partition"});
        for (size_type column = n_keys;
             column < local->num_columns(); ++column) {
            const auto index = static_cast<std::size_t>(column);
            partitioned[index] = alps::collectives::apply_partition(
                local->get_column(column).view(), plan,
                runtime.preparation_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                prepared[index], runtime.preparation_stream));
        }
    }

    std::vector<gamax::column_ptr> exchanged(column_count);
    for (size_type column = 0; column < n_keys; ++column) {
        const auto index = static_cast<std::size_t>(column);
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.communication_stream,
            column_exchange_metadata(
                "group_by.column_overlap.key_exchange", "key", column));
        ALPS_CUDACHECK(cudaStreamWaitEvent(
            runtime.communication_stream, prepared[index], 0));
        const auto bytes_sent = logical_column_bytes(
            partitioned[index]->view());
        const auto rows_sent = partitioned[index]->size();
        exchanged[index] = alps::collectives::exchange_partitioned_column(
            partitioned[index]->view(), plan.pattern, distributed_context,
            runtime.communication_stream, alps::collectives::Backend::NCCL);
        phase.set_transfer(
            bytes_sent, logical_column_bytes(exchanged[index]->view()),
            rows_sent, exchanged[index]->size(), distributed_context.size - 1);
        ALPS_CUDACHECK(cudaEventRecord(
            received_events[index], runtime.communication_stream));
    }
    ALPS_CUDACHECK(cudaEventRecord(
        keys_received, runtime.communication_stream));
    for (size_type column = n_keys;
         column < local->num_columns(); ++column) {
        const auto index = static_cast<std::size_t>(column);
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.communication_stream,
            column_exchange_metadata(
                "group_by.column_overlap.state_exchange", "state", column));
        ALPS_CUDACHECK(cudaStreamWaitEvent(
            runtime.communication_stream, prepared[index], 0));
        const auto bytes_sent = logical_column_bytes(
            partitioned[index]->view());
        const auto rows_sent = partitioned[index]->size();
        exchanged[index] = alps::collectives::exchange_partitioned_column(
            partitioned[index]->view(), plan.pattern, distributed_context,
            runtime.communication_stream, alps::collectives::Backend::NCCL);
        phase.set_transfer(
            bytes_sent, logical_column_bytes(exchanged[index]->view()),
            rows_sent, exchanged[index]->size(), distributed_context.size - 1);
        ALPS_CUDACHECK(cudaEventRecord(
            received_events[index], runtime.communication_stream));
    }
    ALPS_CUDACHECK(cudaEventRecord(
        all_received, runtime.communication_stream));

    std::vector<gamax::column_view> key_columns;
    key_columns.reserve(static_cast<std::size_t>(n_keys));
    for (size_type column = 0; column < n_keys; ++column) {
        key_columns.push_back(
            exchanged[static_cast<std::size_t>(column)]->view());
    }
    gamax::table_view exchanged_keys(key_columns);

    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, keys_received, 0));
    gamax::groupby::hash::group_id_plan group_ids;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.join_stream,
            {"group_by.column_overlap.group_ids",
             gamax::exec::PhaseCategory::Compute});
        if (exchanged_keys.num_rows() > 0) {
            auto group_hint = hint;
            group_ids = gamax::groupby::hash::compute_group_ids(
                exchanged_keys, group_hint, runtime.join_stream);
        }
    }

    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, all_received, 0));
    std::vector<gamax::column_view> state_columns;
    state_columns.reserve(column_count - static_cast<std::size_t>(n_keys));
    for (size_type column = n_keys;
         column < local->num_columns(); ++column) {
        state_columns.push_back(
            exchanged[static_cast<std::size_t>(column)]->view());
    }
    gamax::table_view exchanged_states(state_columns);

    std::unique_ptr<gamax::table> final_states;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.join_stream,
            {"group_by.column_overlap.merge_finalize",
             gamax::exec::PhaseCategory::Compute});
        if (exchanged_keys.num_rows() == 0) {
            gamax::phys::Context worker_context{
                runtime.join_stream, physical_context.profile_sink};
            final_states = gamax::phys::group_by(
                exchanged_keys, exchanged_states, layout.merge_specs,
                worker_context, hint);
        } else {
            std::vector<gamax::groupby::aggregation_request> requests;
            requests.reserve(layout.merge_specs.size());
            for (const auto& spec : layout.merge_specs) {
                requests.emplace_back(spec.column, spec.agg);
            }
            final_states = gamax::groupby::hash::groupby_with_plan(
                exchanged_keys, exchanged_states, requests, group_ids,
                runtime.join_stream);
        }
        final_states = finalize_states(
            std::move(final_states), n_keys, layout, runtime.join_stream);
    }

    ALPS_CUDACHECK(cudaEventRecord(final, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(physical_context.stream, final, 0));
    set_table_lifetime_stream(*final_states, physical_context.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(physical_context.stream));
    return final_states;
}

std::unique_ptr<gamax::table> distinct(
    gamax::table_view input,
    const std::vector<size_type>& key_ordinals,
    const gamax::groupby::groupby_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy shuffle_alignment)
{
    if (key_ordinals.empty()) {
        throw std::invalid_argument("distributed distinct: empty key list");
    }
    std::unique_ptr<gamax::table> local;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"distinct.local", gamax::exec::PhaseCategory::Compute});
        local = gamax::phys::distinct(
            input.select_columns(key_ordinals), physical_context, hint);
    }
    const auto all_keys = sequence(local->num_columns());
    alps::collectives::PartitionedTable partitioned;
    {
        const auto bytes = gamax::exec::table_data_bytes(local->view());
        gamax::exec::PhaseRange phase(
            physical_context,
            {"shuffle.hash_partition",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "hash_partition"});
        partitioned = alps::collectives::hash_partition(
            local->view(), all_keys, distributed_context, physical_context.stream);
        phase.set_transfer(
            bytes, gamax::exec::table_data_bytes(partitioned.table->view()),
            local->num_rows(), partitioned.table->num_rows(),
            distributed_context.size - 1);
    }
    std::unique_ptr<gamax::table> exchanged;
    {
        const auto bytes = gamax::exec::table_data_bytes(partitioned.table->view());
        const auto rows = partitioned.table->num_rows();
        gamax::exec::PhaseRange phase(
            physical_context,
            {"shuffle.exchange",
             gamax::exec::PhaseCategory::Communication,
             gamax::exec::CommunicationPattern::Shuffle,
             shuffle_primitive(backend, shuffle_alignment)});
        alps::collectives::AlignmentStats alignment_stats;
        exchanged = alps::collectives::exchange_partitions(
            std::move(partitioned), distributed_context,
            physical_context.stream, backend, shuffle_alignment,
            &alignment_stats);
        set_alignment_stats(phase, alignment_stats);
        phase.set_transfer(
            bytes, gamax::exec::table_data_bytes(exchanged->view()),
            rows, exchanged->num_rows(), distributed_context.size - 1);
    }
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"distinct.final", gamax::exec::PhaseCategory::Compute});
        return gamax::phys::distinct(
            exchanged->view(), physical_context, hint);
    }
}

std::unique_ptr<gamax::table> top_k(
    gamax::table_view input,
    const std::vector<gamax::phys::SortSpec>& keys,
    size_type k,
    int root,
    const gamax::sort::sort_table_options& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend)
{
    std::unique_ptr<gamax::table> local;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"top_k.local", gamax::exec::PhaseCategory::Compute});
        local = gamax::phys::top_k(input, keys, k, physical_context, hint);
    }
    std::unique_ptr<gamax::table> candidates;
    {
        const auto bytes = gamax::exec::table_data_bytes(local->view());
        const auto rows = local->num_rows();
        gamax::exec::PhaseRange phase(
            physical_context,
            {"gather.candidates",
             gamax::exec::PhaseCategory::Communication,
             gamax::exec::CommunicationPattern::Gather,
             exchange_primitive(backend)});
        candidates = alps::collectives::gather_to_root(
            local->view(), root, distributed_context,
            physical_context.stream, backend);
        const bool is_root = distributed_context.rank == root;
        phase.set_transfer(
            is_root ? 0 : bytes,
            is_root ? gamax::exec::table_data_bytes(candidates->view()) - bytes : 0,
            is_root ? 0 : rows,
            is_root ? candidates->num_rows() - rows : 0,
            distributed_context.size > 1
                ? (is_root ? distributed_context.size - 1 : 1) : 0);
    }
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"top_k.final", gamax::exec::PhaseCategory::Compute});
        if (distributed_context.rank != root) return candidates;
        return gamax::phys::top_k(
            candidates->view(), keys, k, physical_context, hint);
    }
}

std::unique_ptr<gamax::table> sort(
    gamax::table_view input,
    const std::vector<gamax::phys::SortSpec>& keys,
    int root,
    const gamax::sort::sort_table_options& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy shuffle_alignment,
    alps::collectives::AlignmentPolicy replication_alignment)
{
    if (keys.empty()) {
        throw std::invalid_argument("distributed sort: empty sort key list");
    }
    std::unique_ptr<gamax::table> local_sorted;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"sort.local", gamax::exec::PhaseCategory::Compute});
        local_sorted = gamax::phys::sort(
            input, keys, physical_context, hint);
    }

    if (distributed_context.size > 1) {
        std::unique_ptr<gamax::table> local_samples;
        {
            gamax::exec::PhaseRange phase(
                physical_context,
                {"sampling.local_samples", gamax::exec::PhaseCategory::Compute});
            const size_type sample_count = std::min<size_type>(
                distributed_context.size - 1, local_sorted->num_rows());
            std::vector<size_type> sample_indices(
                static_cast<std::size_t>(sample_count));
            for (size_type sample = 0; sample < sample_count; ++sample) {
                sample_indices[sample] =
                    ((sample + 1) * local_sorted->num_rows()) /
                    (sample_count + 1);
            }
            auto sample_map = index_column(
                sample_indices, physical_context.stream);
            local_samples = alps::collectives::gather_rows(
                local_sorted->view(), sample_map->view(),
                physical_context.stream);
        }
        std::unique_ptr<gamax::table> global_samples;
        {
            const auto bytes = gamax::exec::table_data_bytes(local_samples->view());
            const auto rows = local_samples->num_rows();
            gamax::exec::PhaseRange phase(
                physical_context,
                {"sampling.replicate",
                 gamax::exec::PhaseCategory::Communication,
                 gamax::exec::CommunicationPattern::Sampling,
                 replication_primitive(backend, replication_alignment)});
            alps::collectives::AlignmentStats alignment_stats;
            global_samples = alps::collectives::replicate(
                local_samples->view(), distributed_context,
                physical_context.stream, backend, replication_alignment,
                &alignment_stats);
            set_alignment_stats(phase, alignment_stats);
            phase.set_transfer(
                bytes * (distributed_context.size - 1),
                gamax::exec::table_data_bytes(global_samples->view()) - bytes,
                static_cast<std::int64_t>(rows) *
                    (distributed_context.size - 1),
                global_samples->num_rows() - rows,
                distributed_context.size - 1);
        }

        if (global_samples->num_rows() > 0) {
            std::unique_ptr<gamax::column> destinations;
            {
                gamax::exec::PhaseRange phase(
                    physical_context,
                    {"sampling.choose_splitters",
                     gamax::exec::PhaseCategory::Compute});
                auto sorted_samples = gamax::phys::sort(
                    global_samples->view(), keys, physical_context, hint);
                std::vector<size_type> splitter_indices(
                    static_cast<std::size_t>(distributed_context.size - 1));
                for (int splitter = 0;
                     splitter < distributed_context.size - 1; ++splitter) {
                    splitter_indices[splitter] = std::min<size_type>(
                        sorted_samples->num_rows() - 1,
                        (static_cast<size_type>(splitter + 1) *
                         sorted_samples->num_rows()) /
                            distributed_context.size);
                }
                auto splitter_map = index_column(
                    splitter_indices, physical_context.stream);
                auto splitters = alps::collectives::gather_rows(
                    sorted_samples->view(), splitter_map->view(),
                    physical_context.stream);
                destinations = make_range_destinations(
                    local_sorted->view(), splitters->view(), keys,
                    physical_context.stream);
            }
            alps::collectives::PartitionedTable partitioned;
            {
                const auto bytes = gamax::exec::table_data_bytes(
                    local_sorted->view());
                gamax::exec::PhaseRange phase(
                    physical_context,
                    {"shuffle.range_partition",
                     gamax::exec::PhaseCategory::Communication,
                     gamax::exec::CommunicationPattern::Shuffle,
                     "range_partition"});
                partitioned = alps::collectives::partition_by_destination(
                    local_sorted->view(), destinations->view(),
                    distributed_context, physical_context.stream);
                phase.set_transfer(
                    bytes,
                    gamax::exec::table_data_bytes(partitioned.table->view()),
                    local_sorted->num_rows(), partitioned.table->num_rows(),
                    distributed_context.size - 1);
            }
            std::unique_ptr<gamax::table> exchanged;
            {
                const auto bytes = gamax::exec::table_data_bytes(
                    partitioned.table->view());
                const auto rows = partitioned.table->num_rows();
                gamax::exec::PhaseRange phase(
                    physical_context,
                    {"shuffle.exchange",
                     gamax::exec::PhaseCategory::Communication,
                     gamax::exec::CommunicationPattern::Shuffle,
                     shuffle_primitive(backend, shuffle_alignment)});
                alps::collectives::AlignmentStats alignment_stats;
                exchanged = alps::collectives::exchange_partitions(
                    std::move(partitioned), distributed_context,
                    physical_context.stream, backend, shuffle_alignment,
                    &alignment_stats);
                set_alignment_stats(phase, alignment_stats);
                phase.set_transfer(
                    bytes, gamax::exec::table_data_bytes(exchanged->view()),
                    rows, exchanged->num_rows(), distributed_context.size - 1);
            }
            {
                gamax::exec::PhaseRange phase(
                    physical_context,
                    {"sort.partition", gamax::exec::PhaseCategory::Compute});
                local_sorted = gamax::phys::sort(
                    exchanged->view(), keys, physical_context, hint);
            }
        }
    }

    {
        const auto bytes = gamax::exec::table_data_bytes(local_sorted->view());
        const auto rows = local_sorted->num_rows();
        gamax::exec::PhaseRange phase(
            physical_context,
            {"gather.sorted_partitions",
             gamax::exec::PhaseCategory::Communication,
             gamax::exec::CommunicationPattern::Gather,
             exchange_primitive(backend)});
        auto result = alps::collectives::gather_to_root(
            local_sorted->view(), root, distributed_context,
            physical_context.stream, backend);
        const bool is_root = distributed_context.rank == root;
        phase.set_transfer(
            is_root ? 0 : bytes,
            is_root ? gamax::exec::table_data_bytes(result->view()) - bytes : 0,
            is_root ? 0 : rows,
            is_root ? result->num_rows() - rows : 0,
            distributed_context.size > 1
                ? (is_root ? distributed_context.size - 1 : 1) : 0);
        return result;
    }
}

namespace {

std::unique_ptr<gamax::table> hash_exchange(
    gamax::table_view input,
    const std::vector<size_type>& keys,
    bool enabled,
    const char* side,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy alignment)
{
    if (!enabled) return nullptr;
    alps::collectives::PartitionedTable partitioned;
    {
        const auto bytes = gamax::exec::table_data_bytes(input);
        gamax::exec::PhaseRange phase(
            physical_context,
            {std::string("shuffle.") + side + ".hash_partition",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "hash_partition"});
        partitioned = alps::collectives::hash_partition(
            input, keys, distributed_context, physical_context.stream);
        phase.set_transfer(
            bytes, gamax::exec::table_data_bytes(partitioned.table->view()),
            input.num_rows(), partitioned.table->num_rows(),
            distributed_context.size - 1);
    }
    const auto bytes = gamax::exec::table_data_bytes(partitioned.table->view());
    const auto rows = partitioned.table->num_rows();
    gamax::exec::PhaseRange phase(
        physical_context,
        {std::string("shuffle.") + side + ".exchange",
         gamax::exec::PhaseCategory::Communication,
         gamax::exec::CommunicationPattern::Shuffle,
         shuffle_primitive(backend, alignment)});
    alps::collectives::AlignmentStats alignment_stats;
    auto result = alps::collectives::exchange_partitions(
        std::move(partitioned), distributed_context,
        physical_context.stream, backend, alignment, &alignment_stats);
    set_alignment_stats(phase, alignment_stats);
    phase.set_transfer(
        bytes, gamax::exec::table_data_bytes(result->view()),
        rows, result->num_rows(), distributed_context.size - 1);
    return result;
}

std::unique_ptr<gamax::table> replicate_input(
    gamax::table_view input,
    bool enabled,
    const char* side,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy alignment)
{
    if (!enabled) return nullptr;
    const auto bytes = gamax::exec::table_data_bytes(input);
    const auto rows = input.num_rows();
    gamax::exec::PhaseRange phase(
        physical_context,
        {std::string("broadcast.") + side,
         gamax::exec::PhaseCategory::Communication,
         gamax::exec::CommunicationPattern::Broadcast,
         replication_primitive(backend, alignment)});
    alps::collectives::AlignmentStats alignment_stats;
    auto result = alps::collectives::replicate(
        input, distributed_context, physical_context.stream, backend,
        alignment, &alignment_stats);
    set_alignment_stats(phase, alignment_stats);
    phase.set_transfer(
        bytes * (distributed_context.size - 1),
        gamax::exec::table_data_bytes(result->view()) - bytes,
        static_cast<std::int64_t>(rows) * (distributed_context.size - 1),
        result->num_rows() - rows, distributed_context.size - 1);
    return result;
}

gamax::table_view resolved_view(
    gamax::table_view original,
    const std::unique_ptr<gamax::table>& redistributed)
{
    return redistributed ? redistributed->view() : original;
}

void validate_join_controls(
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right)
{
    using Strategy = gamax::exec::JoinDistributionStrategy;
    if (strategy == Strategy::Shuffle) {
        if (replicate_left || replicate_right) {
            throw std::invalid_argument(
                "distributed join: shuffle strategy cannot replicate inputs");
        }
        return;
    }
    if (partition_left || partition_right) {
        throw std::invalid_argument(
            "distributed join: broadcast strategy cannot partition inputs");
    }
    if (strategy == Strategy::BroadcastLeft && replicate_right) {
        throw std::invalid_argument(
            "distributed join: BroadcastLeft cannot replicate the right input");
    }
    if (strategy == Strategy::BroadcastRight && replicate_left) {
        throw std::invalid_argument(
            "distributed join: BroadcastRight cannot replicate the left input");
    }
}

struct JoinInputs {
    std::unique_ptr<gamax::table> left;
    std::unique_ptr<gamax::table> right;
};

JoinInputs prepare_join_inputs(
    gamax::table_view left,
    const std::vector<size_type>& left_keys,
    gamax::table_view right,
    const std::vector<size_type>& right_keys,
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy shuffle_alignment,
    alps::collectives::AlignmentPolicy replication_alignment)
{
    validate_join_controls(
        strategy, partition_left, partition_right,
        replicate_left, replicate_right);
    JoinInputs result;
    if (strategy == gamax::exec::JoinDistributionStrategy::Shuffle) {
        result.left = hash_exchange(
            left, left_keys, partition_left, "left", distributed_context,
            physical_context, backend, shuffle_alignment);
        result.right = hash_exchange(
            right, right_keys, partition_right, "right", distributed_context,
            physical_context, backend, shuffle_alignment);
    } else {
        result.left = replicate_input(
            left, replicate_left, "left", distributed_context,
            physical_context, backend, replication_alignment);
        result.right = replicate_input(
            right, replicate_right, "right", distributed_context,
            physical_context, backend, replication_alignment);
    }
    return result;
}

} // namespace

std::unique_ptr<gamax::table> eq_join(
    gamax::table_view left,
    const std::vector<size_type>& left_keys,
    gamax::table_view right,
    const std::vector<size_type>& right_keys,
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy shuffle_alignment,
    alps::collectives::AlignmentPolicy replication_alignment)
{
    auto inputs = prepare_join_inputs(
        left, left_keys, right, right_keys, strategy,
        partition_left, partition_right, replicate_left, replicate_right,
        distributed_context, physical_context, backend,
        shuffle_alignment, replication_alignment);
    gamax::exec::PhaseRange phase(
        physical_context,
        {"join.local_eq", gamax::exec::PhaseCategory::Compute});
    return gamax::phys::eq_join(
        resolved_view(left, inputs.left), left_keys,
        resolved_view(right, inputs.right), right_keys,
        physical_context, hint);
}

std::unique_ptr<gamax::table> semi_join(
    gamax::table_view left,
    const std::vector<size_type>& left_keys,
    gamax::table_view right,
    const std::vector<size_type>& right_keys,
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy shuffle_alignment,
    alps::collectives::AlignmentPolicy replication_alignment)
{
    if (strategy == gamax::exec::JoinDistributionStrategy::BroadcastLeft) {
        throw std::invalid_argument(
            "distributed semi join only supports shuffle or BroadcastRight");
    }
    auto inputs = prepare_join_inputs(
        left, left_keys, right, right_keys, strategy,
        partition_left, partition_right, replicate_left, replicate_right,
        distributed_context, physical_context, backend,
        shuffle_alignment, replication_alignment);
    gamax::exec::PhaseRange phase(
        physical_context,
        {"join.local_semi", gamax::exec::PhaseCategory::Compute});
    return gamax::phys::semi_join(
        resolved_view(left, inputs.left), left_keys,
        resolved_view(right, inputs.right), right_keys,
        physical_context, hint);
}

std::unique_ptr<gamax::table> column_overlap_semi_join_ce_pull(
    gamax::table_view left,
    size_type left_key,
    gamax::table_view right,
    size_type right_key,
    bool partition_left,
    bool partition_right,
    const alps::collectives::HashPartitionPlan& left_plan,
    const alps::collectives::HashPartitionPlan& right_plan,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    bool communication_high_priority,
    alps::collectives::CePullSchedule ce_pull_schedule,
    int ce_pull_slots)
{
    const auto left_order = key_first_ordinals(left, left_key);
    const auto left_columns = left_order.size();
    auto& runtime = column_overlap_runtime(
        distributed_context, communication_high_priority);
    runtime.ensure_dependency_events(4);
    const auto entry = runtime.dependency_events[0];
    const auto final = runtime.dependency_events[1];
    const auto all_probe_columns_prepared = runtime.dependency_events[2];
    const auto release_probe_pulls = runtime.dependency_events[3];

    ALPS_CUDACHECK(cudaEventRecord(entry, physical_context.stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.preparation_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(runtime.join_stream, entry, 0));

    const auto operations =
        (partition_right ? std::size_t{1} : std::size_t{0}) +
        (partition_left ? left_columns : std::size_t{0});
    const auto ce_options = full_staging_options(
        operations, communication_high_priority, ce_pull_schedule,
        ce_pull_slots, "column_overlap_semi_join");
    std::unique_ptr<alps::collectives::CePullShuffleSession> right_session;
    std::unique_ptr<alps::collectives::CePullShuffleSession> left_session;
    if (operations > 0) {
        std::exception_ptr local_error;
        try {
            alps::collectives::begin_ce_pull_full_staging_batch(
                distributed_context, operations, runtime.preparation_stream,
                ce_options);
            if (partition_right) {
                right_session = alps::collectives::begin_ce_pull_shuffle(
                    right_plan, {right.get_column(right_key)},
                    distributed_context, runtime.preparation_stream,
                    ce_options);
            }
            if (partition_left) {
                left_session = alps::collectives::begin_ce_pull_shuffle(
                    left_plan, column_views(left), distributed_context,
                    runtime.preparation_stream, ce_options);
            }
            alps::collectives::seal_ce_pull_full_staging_batch(
                distributed_context);
        } catch (...) {
            local_error = std::current_exception();
        }
        if (!rank_wide_stage_succeeded(!local_error, distributed_context)) {
            alps::collectives::cancel_ce_pull_full_staging_batch(
                distributed_context);
            throw_rank_wide_stage_failure(
                local_error, "column_overlap_semi_join",
                "CE-pull full-staging session planning");
        }
    }

    const auto transfer_stream = operations > 0
        ? alps::collectives::ce_pull_transfer_stream(distributed_context)
        : runtime.communication_stream;
    ALPS_CUDACHECK(cudaStreamWaitEvent(transfer_stream, entry, 0));

    gamax::column_ptr exchanged_right;
    gamax::column_view build_key = right.get_column(right_key);
    if (right_session) {
        gamax::exec::PhaseMetadata prepare_metadata{
            "semi_join.column_overlap.build_key_partition_apply",
            gamax::exec::PhaseCategory::Compute,
            gamax::exec::CommunicationPattern::Shuffle,
            "split_gather"};
        prepare_metadata.column_side = "right";
        prepare_metadata.column_ordinal = right_key;
        {
            gamax::exec::PhaseRange phase(
                physical_context.profile_sink, runtime.preparation_stream,
                std::move(prepare_metadata));
            right_session->prepare_column(0);
        }
        const auto& output = right_session->output(0);
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, transfer_stream,
            column_exchange_metadata(
                "semi_join.column_overlap.build_key_exchange", "right",
                right_key, alps::collectives::Backend::CE_PULL));
        right_session->enqueue_pull(0);
        phase.set_transfer(
            logical_column_bytes(right.get_column(right_key)),
            logical_column_bytes(output.view()),
            right.get_column(right_key).size(), output.size(),
            distributed_context.size - 1);
    }

    if (left_session) {
        for (const auto ordinal : left_order) {
            gamax::exec::PhaseMetadata prepare_metadata{
                "semi_join.column_overlap.probe_partition_apply",
                gamax::exec::PhaseCategory::Compute,
                gamax::exec::CommunicationPattern::Shuffle,
                "split_gather"};
            prepare_metadata.column_side = "left";
            prepare_metadata.column_ordinal = ordinal;
            gamax::exec::PhaseRange phase(
                physical_context.profile_sink, runtime.preparation_stream,
                std::move(prepare_metadata));
            left_session->prepare_column(
                static_cast<std::size_t>(ordinal));
        }
    }
    ALPS_CUDACHECK(cudaEventRecord(
        all_probe_columns_prepared, runtime.preparation_stream));

    if (right_session) {
        exchanged_right = right_session->release_output(
            0, runtime.join_stream);
        build_key = exchanged_right->view();
        right_session->wait_on(0, runtime.join_stream);
    }

    // Hold the hash build until every probe split/gather has been submitted.
    // Probe pulls are released immediately before the build so CE traffic can
    // overlap set construction without starving preparation kernels.
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, all_probe_columns_prepared, 0));
    ALPS_CUDACHECK(cudaEventRecord(
        release_probe_pulls, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        transfer_stream, release_probe_pulls, 0));

    if (left_session) {
        for (const auto ordinal : left_order) {
            const auto index = static_cast<std::size_t>(ordinal);
            const auto& output = left_session->output(index);
            gamax::exec::PhaseRange phase(
                physical_context.profile_sink, transfer_stream,
                column_exchange_metadata(
                    "semi_join.column_overlap.probe_exchange", "left",
                    ordinal, alps::collectives::Backend::CE_PULL));
            left_session->enqueue_pull(index);
            phase.set_transfer(
                logical_column_bytes(left.get_column(ordinal)),
                logical_column_bytes(output.view()),
                left.get_column(ordinal).size(), output.size(),
                distributed_context.size - 1);
        }
    }
    if (operations > 0) {
        alps::collectives::finish_ce_pull_full_staging_batch(
            distributed_context);
    }

    std::vector<gamax::column_ptr> exchanged_left(left_columns);
    std::vector<gamax::column_view> probe_columns(left_columns);
    for (std::size_t index = 0; index < left_columns; ++index) {
        const auto ordinal = left_order[index];
        if (left_session) {
            exchanged_left[index] = left_session->release_output(
                static_cast<std::size_t>(ordinal), runtime.join_stream);
            probe_columns[index] = exchanged_left[index]->view();
        } else {
            probe_columns[index] = left.get_column(ordinal);
        }
    }

    std::unique_ptr<gamax::join::detail::semi_hash_set> build_set;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, runtime.join_stream,
            {"semi_join.column_overlap.build",
             gamax::exec::PhaseCategory::Compute});
        if (build_key.size() > 0) {
            build_set = gamax::join::detail::build_semi_hash_set(
                build_key, runtime.join_stream);
        }
    }

    if (left_session) {
        left_session->wait_on(
            static_cast<std::size_t>(left_key), runtime.join_stream);
    }
    std::unique_ptr<gamax::column> survivor_map;
    std::unique_ptr<gamax::column> mask;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, runtime.join_stream,
            {"semi_join.column_overlap.probe",
             gamax::exec::PhaseCategory::Compute});
        if (build_set && probe_columns.front().size() > 0) {
            mask = build_set->contains(
                probe_columns.front(), runtime.join_stream);
        }
    }
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, runtime.join_stream,
            {"semi_join.column_overlap.survivor_map",
             gamax::exec::PhaseCategory::Compute});
        if (mask) {
            auto tids = gamax::make_sequence<size_type>(
                probe_columns.front().size(), runtime.join_stream, 0, 1);
            survivor_map = gamax::util::device::compact(
                tids->view(), mask->view(), runtime.join_stream);
        }
    }
    if (!survivor_map) {
        survivor_map = gamax::make_idx_column(0, runtime.join_stream);
    }

    std::vector<gamax::column_ptr> output_columns(left_columns);
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink, runtime.join_stream,
            {"semi_join.column_overlap.materialize",
             gamax::exec::PhaseCategory::Compute});
        for (std::size_t index = 0; index < left_columns; ++index) {
            const auto ordinal = left_order[index];
            if (left_session) {
                left_session->wait_on(
                    static_cast<std::size_t>(ordinal), runtime.join_stream);
            }
            output_columns[static_cast<std::size_t>(ordinal)] =
                gamax::join::detail::simple_materialize(
                    probe_columns[index], survivor_map->view(),
                    runtime.join_stream);
        }
    }
    auto result = std::make_unique<gamax::table>(
        std::move(output_columns));
    ALPS_CUDACHECK(cudaEventRecord(final, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(physical_context.stream, final, 0));
    set_table_lifetime_stream(*result, physical_context.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(physical_context.stream));
    return result;
}

std::unique_ptr<gamax::table> column_overlap_semi_join(
    gamax::table_view left,
    size_type left_key,
    gamax::table_view right,
    size_type right_key,
    bool partition_left,
    bool partition_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    bool communication_high_priority,
    alps::collectives::Backend exchange_backend,
    alps::collectives::CePullSchedule ce_pull_schedule,
    int ce_pull_slots)
{
    if (left_key < 0 || left_key >= left.num_columns() ||
        right_key < 0 || right_key >= right.num_columns()) {
        throw std::out_of_range(
            "column_overlap_semi_join: key ordinal out of range");
    }
    if (left.get_column(left_key).type() !=
        right.get_column(right_key).type()) {
        throw std::invalid_argument(
            "column_overlap_semi_join: key types differ");
    }
    if (hint.join_algo != gamax::join::HASH ||
        hint.mat_algo == gamax::join::AUTO_MAT ||
        hint.mat_algo == gamax::join::INDEX_CLUSTER) {
        throw std::invalid_argument(
            "column_overlap_semi_join: requires HASH and a supported materializer");
    }
    if (exchange_backend != alps::collectives::Backend::NCCL &&
        exchange_backend != alps::collectives::Backend::CE_PULL) {
        throw std::invalid_argument(
            "column_overlap_semi_join: exchange backend must be NCCL or "
            "CE_PULL");
    }

    alps::collectives::HashPartitionPlan left_plan;
    alps::collectives::HashPartitionPlan right_plan;
    {
        gamax::exec::PhaseRange phase(
            physical_context,
            {"semi_join.column_overlap.partition_plans",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "hash_partition_plan"});
        if (partition_left) {
            left_plan = alps::collectives::make_hash_partition_plan(
                left.select_columns({left_key}), distributed_context,
                physical_context.stream);
            alps::collectives::complete_receive_pattern(
                left_plan.pattern, distributed_context);
        }
        if (partition_right) {
            right_plan = alps::collectives::make_hash_partition_plan(
                right.select_columns({right_key}), distributed_context,
                physical_context.stream);
            alps::collectives::complete_receive_pattern(
                right_plan.pattern, distributed_context);
        }
    }

    if (exchange_backend == alps::collectives::Backend::CE_PULL) {
        return column_overlap_semi_join_ce_pull(
            left, left_key, right, right_key, partition_left,
            partition_right, left_plan, right_plan, hint,
            distributed_context, physical_context,
            communication_high_priority, ce_pull_schedule, ce_pull_slots);
    }

    const auto left_order = key_first_ordinals(left, left_key);
    const auto left_columns = left_order.size();
    auto& runtime = column_overlap_runtime(
        distributed_context, communication_high_priority);
    runtime.ensure_dependency_events(5 + 2 * left_columns);
    std::size_t event_index = 0;
    const auto entry = runtime.dependency_events[event_index++];
    const auto final = runtime.dependency_events[event_index++];
    const auto right_prepared = runtime.dependency_events[event_index++];
    const auto right_ready = runtime.dependency_events[event_index++];
    const auto all_received = runtime.dependency_events[event_index++];
    std::vector<cudaEvent_t> left_prepared(left_columns);
    std::vector<cudaEvent_t> left_ready(left_columns);
    for (auto& event : left_prepared) {
        event = runtime.dependency_events[event_index++];
    }
    for (auto& event : left_ready) {
        event = runtime.dependency_events[event_index++];
    }

    ALPS_CUDACHECK(cudaEventRecord(entry, physical_context.stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.preparation_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.communication_stream, entry, 0));
    ALPS_CUDACHECK(cudaStreamWaitEvent(runtime.join_stream, entry, 0));

    gamax::column_ptr partitioned_right;
    std::vector<gamax::column_ptr> partitioned_left(left_columns);
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.preparation_stream,
            {"semi_join.column_overlap.build_key_partition_apply",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "apply_partition"});
        if (partition_right) {
            partitioned_right = alps::collectives::apply_partition(
                right.get_column(right_key), right_plan,
                runtime.preparation_stream);
            ALPS_CUDACHECK(cudaEventRecord(
                right_prepared, runtime.preparation_stream));
        } else {
            ALPS_CUDACHECK(cudaEventRecord(
                right_prepared, runtime.preparation_stream));
        }
    }
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.preparation_stream,
            {"semi_join.column_overlap.probe_partition_apply",
             gamax::exec::PhaseCategory::Compute,
             gamax::exec::CommunicationPattern::Shuffle,
             "apply_partition"});
        for (std::size_t index = 0; index < left_columns; ++index) {
            if (partition_left) {
                partitioned_left[index] =
                    alps::collectives::apply_partition(
                        left.get_column(left_order[index]), left_plan,
                        runtime.preparation_stream);
            }
            ALPS_CUDACHECK(cudaEventRecord(
                left_prepared[index], runtime.preparation_stream));
        }
    }

    gamax::column_ptr exchanged_right;
    gamax::column_view build_key =
        right.get_column(right_key);
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.communication_stream,
            column_exchange_metadata(
                "semi_join.column_overlap.build_key_exchange",
                "right", right_key));
        ALPS_CUDACHECK(cudaStreamWaitEvent(
            runtime.communication_stream, right_prepared, 0));
        if (partition_right) {
            const auto bytes_sent = logical_column_bytes(
                partitioned_right->view());
            const auto rows_sent = partitioned_right->size();
            exchanged_right =
                alps::collectives::exchange_partitioned_column(
                    partitioned_right->view(), right_plan.pattern,
                    distributed_context, runtime.communication_stream,
                    alps::collectives::Backend::NCCL);
            build_key = exchanged_right->view();
            phase.set_transfer(
                bytes_sent, logical_column_bytes(build_key),
                rows_sent, build_key.size(), distributed_context.size - 1);
        } else {
            phase.set_transfer(0, 0, 0, 0, 0);
        }
        ALPS_CUDACHECK(cudaEventRecord(
            right_ready, runtime.communication_stream));
    }

    std::vector<gamax::column_ptr> exchanged_left(left_columns);
    std::vector<gamax::column_view> probe_columns(left_columns);
    for (std::size_t index = 0; index < left_columns; ++index) {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.communication_stream,
            column_exchange_metadata(
                "semi_join.column_overlap.probe_exchange",
                "left", left_order[index]));
        ALPS_CUDACHECK(cudaStreamWaitEvent(
            runtime.communication_stream, left_prepared[index], 0));
        if (partition_left) {
            const auto bytes_sent = logical_column_bytes(
                partitioned_left[index]->view());
            const auto rows_sent = partitioned_left[index]->size();
            exchanged_left[index] =
                alps::collectives::exchange_partitioned_column(
                    partitioned_left[index]->view(), left_plan.pattern,
                    distributed_context, runtime.communication_stream,
                    alps::collectives::Backend::NCCL);
            probe_columns[index] = exchanged_left[index]->view();
            phase.set_transfer(
                bytes_sent, logical_column_bytes(probe_columns[index]),
                rows_sent, probe_columns[index].size(),
                distributed_context.size - 1);
        } else {
            probe_columns[index] = left.get_column(left_order[index]);
            phase.set_transfer(0, 0, 0, 0, 0);
        }
        ALPS_CUDACHECK(cudaEventRecord(
            left_ready[index], runtime.communication_stream));
    }
    ALPS_CUDACHECK(cudaEventRecord(
        all_received, runtime.communication_stream));

    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, right_ready, 0));
    std::unique_ptr<gamax::join::detail::semi_hash_set> build_set;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.join_stream,
            {"semi_join.column_overlap.build",
             gamax::exec::PhaseCategory::Compute});
        if (build_key.size() > 0) {
            build_set = gamax::join::detail::build_semi_hash_set(
                build_key, runtime.join_stream);
        }
    }

    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, left_ready.front(), 0));
    std::unique_ptr<gamax::column> survivor_map;
    std::unique_ptr<gamax::column> mask;
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.join_stream,
            {"semi_join.column_overlap.probe",
             gamax::exec::PhaseCategory::Compute});
        if (build_set && probe_columns.front().size() > 0) {
            mask = build_set->contains(
                probe_columns.front(), runtime.join_stream);
        }
    }
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.join_stream,
            {"semi_join.column_overlap.survivor_map",
             gamax::exec::PhaseCategory::Compute});
        if (mask) {
            auto tids = gamax::make_sequence<size_type>(
                probe_columns.front().size(), runtime.join_stream, 0, 1);
            survivor_map = gamax::util::device::compact(
                tids->view(), mask->view(), runtime.join_stream);
        }
    }
    if (!survivor_map) {
        survivor_map = gamax::make_idx_column(0, runtime.join_stream);
    }

    std::vector<gamax::column_ptr> output_columns(left_columns);
    {
        gamax::exec::PhaseRange phase(
            physical_context.profile_sink,
            runtime.join_stream,
            {"semi_join.column_overlap.materialize",
             gamax::exec::PhaseCategory::Compute});
        for (std::size_t index = 0; index < left_columns; ++index) {
            ALPS_CUDACHECK(cudaStreamWaitEvent(
                runtime.join_stream, left_ready[index], 0));
            const auto output_index =
                static_cast<std::size_t>(left_order[index]);
            output_columns[output_index] =
                gamax::join::detail::simple_materialize(
                    probe_columns[index], survivor_map->view(),
                    runtime.join_stream);
        }
    }
    ALPS_CUDACHECK(cudaStreamWaitEvent(
        runtime.join_stream, all_received, 0));
    auto result = std::make_unique<gamax::table>(
        std::move(output_columns));
    ALPS_CUDACHECK(cudaEventRecord(final, runtime.join_stream));
    ALPS_CUDACHECK(cudaStreamWaitEvent(physical_context.stream, final, 0));
    set_table_lifetime_stream(*result, physical_context.stream);
    ALPS_CUDACHECK(cudaStreamSynchronize(physical_context.stream));
    return result;
}

std::unique_ptr<gamax::table> anti_join(
    gamax::table_view left,
    const std::vector<size_type>& left_keys,
    gamax::table_view right,
    const std::vector<size_type>& right_keys,
    gamax::exec::JoinDistributionStrategy strategy,
    bool partition_left,
    bool partition_right,
    bool replicate_left,
    bool replicate_right,
    const gamax::join::join_hint& hint,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy shuffle_alignment,
    alps::collectives::AlignmentPolicy replication_alignment)
{
    if (strategy == gamax::exec::JoinDistributionStrategy::BroadcastLeft) {
        throw std::invalid_argument(
            "distributed anti join only supports shuffle or BroadcastRight");
    }
    auto inputs = prepare_join_inputs(
        left, left_keys, right, right_keys, strategy,
        partition_left, partition_right, replicate_left, replicate_right,
        distributed_context, physical_context, backend,
        shuffle_alignment, replication_alignment);
    gamax::exec::PhaseRange phase(
        physical_context,
        {"join.local_anti", gamax::exec::PhaseCategory::Compute});
    return gamax::phys::anti_join(
        resolved_view(left, inputs.left), left_keys,
        resolved_view(right, inputs.right), right_keys,
        physical_context, hint);
}

std::unique_ptr<gamax::table> cross_join(
    gamax::table_view left,
    gamax::table_view right,
    gamax::exec::JoinDistributionStrategy strategy,
    bool replicate_left,
    bool replicate_right,
    const alps::Context& distributed_context,
    const gamax::phys::Context& physical_context,
    alps::collectives::Backend backend,
    alps::collectives::AlignmentPolicy replication_alignment)
{
    if (strategy == gamax::exec::JoinDistributionStrategy::Shuffle) {
        throw std::invalid_argument(
            "distributed cross join requires a broadcast strategy");
    }
    validate_join_controls(
        strategy, false, false, replicate_left, replicate_right);
    auto left_copy = replicate_input(
        left, replicate_left, "left", distributed_context,
        physical_context, backend, replication_alignment);
    auto right_copy = replicate_input(
        right, replicate_right, "right", distributed_context,
        physical_context, backend, replication_alignment);
    gamax::exec::PhaseRange phase(
        physical_context,
        {"join.local_cross", gamax::exec::PhaseCategory::Compute});
    return gamax::phys::cross_join(
        resolved_view(left, left_copy), resolved_view(right, right_copy),
        physical_context);
}

} // namespace alps::operators
