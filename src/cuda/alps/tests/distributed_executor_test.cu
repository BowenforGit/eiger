// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "alps/context.hpp"
#include "alps/error.hpp"
#include "alps/exec/distributed_executor.hpp"
#include "alps/exec/distributed_planner.hpp"
#include "exec/exec_plan.hpp"
#include "exec/step_profile.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string_column.hpp"

namespace {

using gamax::size_type;
namespace gexec = gamax::exec;

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
    auto strings = gamax::make_string_column(
        flat.empty() ? empty_data : flat.data(),
        lengths.empty() ? empty_length : lengths.data(),
        static_cast<size_type>(values.size()), stream, gamax::ARROW_STRING);
    return gamax::column_ptr(strings.release());
}

bool has_lifetime_stream(
    const gamax::column& column, cudaStream_t stream)
{
    if (column.buffer().stream() != stream) return false;
    if (column.type().id() != gamax::type_id::STRING) return true;
    return static_cast<const gamax::string_column&>(column)
        .meta_buffer().stream() == stream;
}

bool has_lifetime_stream(
    const gamax::table& table, cudaStream_t stream)
{
    for (size_type column = 0; column < table.num_columns(); ++column) {
        if (!has_lifetime_stream(table.get_column(column), stream)) {
            return false;
        }
    }
    return true;
}

std::unique_ptr<gamax::table> make_input(
    int rank, int world, cudaStream_t stream)
{
    const int64_t rows = rank == world - 1 ? 0 : 101 + rank;
    std::vector<int32_t> keys(static_cast<std::size_t>(rows));
    std::vector<int64_t> row_ids(static_cast<std::size_t>(rows));
    std::vector<std::string> names(static_cast<std::size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
        keys[row] = row % 13;
        names[row] = row % 4 == 0 ? "" : "name-" + std::to_string(row % 7);
        row_ids[row] = static_cast<int64_t>(rank) * 1000000 + row;
    }
    std::vector<gamax::column_ptr> columns;
    columns.push_back(gamax::make_column_from_host(keys.data(), rows, stream));
    columns.push_back(make_strings(names, stream));
    columns.push_back(gamax::make_column_from_host(row_ids.data(), rows, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(columns));
}

std::unique_ptr<gamax::table> make_right_subset(
    int rank, int world, cudaStream_t stream)
{
    const int64_t rows = rank == world - 1 ? 0 : 6;
    std::vector<int32_t> keys(static_cast<std::size_t>(rows));
    std::vector<int64_t> row_ids(static_cast<std::size_t>(rows));
    std::vector<std::string> names(static_cast<std::size_t>(rows));
    for (int64_t row = 0; row < rows; ++row) {
        keys[row] = static_cast<int32_t>(row);
        names[row] = "right-" + std::to_string(rank) + "-" +
            std::to_string(row);
        row_ids[row] = static_cast<int64_t>(rank) * 1000000 + row;
    }
    std::vector<gamax::column_ptr> columns;
    columns.push_back(gamax::make_column_from_host(keys.data(), rows, stream));
    columns.push_back(make_strings(names, stream));
    columns.push_back(gamax::make_column_from_host(row_ids.data(), rows, stream));
    ALPS_CUDACHECK(cudaStreamSynchronize(stream));
    return gamax::make_table(std::move(columns));
}

gexec::Value value(
    gexec::ValueId id,
    size_type last_use,
    gexec::DistributionKind distribution,
    std::vector<size_type> keys = {},
    int root = 0)
{
    gexec::Value result;
    result.id = id;
    result.last_used_step = last_use;
    result.distribution.kind = distribution;
    result.distribution.key_ordinals = std::move(keys);
    result.distribution.root = root;
    return result;
}

gamax::ir::Schema schema(std::initializer_list<const char*> names) {
    gamax::ir::Schema result;
    gamax::size_type id = 0;
    for (const auto* name : names) {
        gamax::ir::Field field;
        field.id = gamax::ir::ColumnId{id++};
        field.type = gamax::data_type{gamax::type_id::INT64};
        field.name = name;
        result.append(std::move(field));
    }
    return result;
}

gexec::ExecPlan shuffle_gather_plan() {
    gexec::ExecPlan plan;
    plan.values.push_back(value(0, 1, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(1, 2, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(
        2, 3, gexec::DistributionKind::HashPartitioned, {0, 1}));
    plan.values.push_back(value(3, -1, gexec::DistributionKind::RootOnly, {}, 0));
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::PartitionRows, {0}, 1,
        gexec::PartitionRowsPayload{
            gexec::PartitionStrategy::Hash, {0, 1}, {}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::ExchangePartitions, {1}, 2,
        gexec::ExchangePartitionsPayload{}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::GatherToRoot, {2}, 3,
        gexec::GatherToRootPayload{0}});
    return plan;
}

gexec::ExecPlan replicate_plan() {
    gexec::ExecPlan plan;
    plan.values.push_back(value(0, 1, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(1, -1, gexec::DistributionKind::Replicated));
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::Replicate, {0}, 1, gexec::ReplicatePayload{}});
    return plan;
}

gexec::ExecPlan replicated_concat_plan() {
    gexec::ExecPlan plan;
    plan.values.push_back(value(0, 1, gexec::DistributionKind::Unknown));
    plan.values.push_back(value(1, 4, gexec::DistributionKind::Unknown));
    plan.values.push_back(value(2, 3, gexec::DistributionKind::Unknown));
    plan.values.push_back(value(3, 4, gexec::DistributionKind::Unknown));
    plan.values.push_back(value(4, -1, gexec::DistributionKind::Unknown));
    plan.values[0].schema = schema({"key", "name", "row_id"});
    plan.values[1].schema = schema({"count"});
    plan.values[2].schema = schema({"key", "name", "row_id"});
    plan.values[3].schema = schema({"count"});
    plan.values[4].schema = schema({"count"});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::ScalarAgg, {0}, 1,
        gexec::ScalarAggPayload{{
            {-1, gamax::groupby::aggregation_id::COUNT_ALL}}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 2,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::ScalarAgg, {2}, 3,
        gexec::ScalarAggPayload{{
            {-1, gamax::groupby::aggregation_id::COUNT_ALL}}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::Concat, {1, 3}, 4, gexec::ConcatPayload{}});
    return plan;
}

gexec::ExecPlan mixed_concat_plan() {
    gexec::ExecPlan plan;
    plan.values.push_back(value(0, 1, gexec::DistributionKind::Unknown));
    plan.values.push_back(value(1, 3, gexec::DistributionKind::Unknown));
    plan.values.push_back(value(2, 3, gexec::DistributionKind::Unknown));
    plan.values.push_back(value(3, -1, gexec::DistributionKind::Unknown));
    plan.values[0].schema = schema({"key", "name", "row_id"});
    plan.values[1].schema = schema({"count"});
    plan.values[2].schema = schema({"key", "name", "row_id"});
    plan.values[3].schema = schema({"count"});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::ScalarAgg, {0}, 1,
        gexec::ScalarAggPayload{{
            {-1, gamax::groupby::aggregation_id::COUNT_ALL}}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 2,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::Concat, {1, 2}, 3, gexec::ConcatPayload{}});
    return plan;
}

gexec::ExecPlan unary_plan(
    gexec::StepKind kind,
    gexec::StepPayload payload,
    gexec::DistributionKind distribution,
    std::vector<size_type> keys = {},
    int root = 0)
{
    gexec::ExecPlan plan;
    plan.values.push_back(value(0, 1, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(1, -1, distribution, std::move(keys), root));
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{kind, {0}, 1, std::move(payload)});
    return plan;
}

gexec::ExecPlan binary_plan(
    gexec::StepKind kind,
    gexec::StepPayload payload,
    gexec::DistributionKind distribution,
    std::string right_source = "input",
    std::vector<size_type> keys = {})
{
    gexec::ExecPlan plan;
    plan.values.push_back(value(0, 2, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(1, 2, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(2, -1, distribution, std::move(keys)));
    plan.values[0].schema = schema({"key", "name", "row_id"});
    plan.values[1].schema = schema({"key", "name", "row_id"});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"input", {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 1,
        gexec::SourceTablePayload{
            std::move(right_source), {"key", "name", "row_id"}}});
    plan.steps.push_back(gexec::Step{kind, {0, 1}, 2, std::move(payload)});
    return plan;
}

gexec::ExecPlan numeric_binary_plan(gexec::DistributedEqJoinPayload payload) {
    gexec::ExecPlan plan;
    plan.values.push_back(value(0, 2, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(1, 2, gexec::DistributionKind::LocalShard));
    plan.values.push_back(value(
        2, -1, gexec::DistributionKind::HashPartitioned, {0}));
    plan.values[0].schema = schema({"key", "row_id"});
    plan.values[1].schema = schema({"key", "row_id"});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 0,
        gexec::SourceTablePayload{"numeric_input", {"key", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::SourceTable, {}, 1,
        gexec::SourceTablePayload{"numeric_input", {"key", "row_id"}}});
    plan.steps.push_back(gexec::Step{
        gexec::StepKind::DistributedEqJoin, {0, 1}, 2, std::move(payload)});
    return plan;
}

int64_t global_output_rows(
    const gamax::table& output, const alps::Context& context)
{
    const int64_t local_rows = output.num_rows();
    int64_t global_rows = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_rows, &global_rows, 1, MPI_LONG_LONG, MPI_SUM,
        context.mpi_comm));
    return global_rows;
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
        const std::size_t length = offsets[row + 1] - offsets[row];
        if (length > 0) {
            result[row].assign(chars.data() + offsets[row], length);
        }
    }
    return result;
}

using InputRow = std::tuple<int32_t, std::string, int64_t>;

std::multiset<InputRow> input_rows_to_host(gamax::table_view table) {
    const auto keys = fixed_to_host<int32_t>(table.get_column(0));
    const auto names = strings_to_host(table.get_column(1));
    const auto row_ids = fixed_to_host<int64_t>(table.get_column(2));
    std::multiset<InputRow> result;
    for (std::size_t row = 0; row < keys.size(); ++row) {
        result.emplace(keys[row], names[row], row_ids[row]);
    }
    return result;
}

using GroupRow = std::tuple<int32_t, std::string, int64_t, int64_t>;

std::multiset<GroupRow> group_rows_to_host(gamax::table_view table) {
    const auto keys = fixed_to_host<int32_t>(table.get_column(0));
    const auto names = strings_to_host(table.get_column(1));
    const auto sums = fixed_to_host<int64_t>(table.get_column(2));
    const auto counts = fixed_to_host<int64_t>(table.get_column(3));
    std::multiset<GroupRow> result;
    for (std::size_t row = 0; row < keys.size(); ++row) {
        result.emplace(keys[row], names[row], sums[row], counts[row]);
    }
    return result;
}

int run(const alps::Context& context) {
    auto input = make_input(context.rank, context.size, context.stream);
    auto right_subset = make_right_subset(
        context.rank, context.size, context.stream);
    const auto numeric_input = input->view().select_columns({0, 2});
    const int64_t local_rows = input->num_rows();
    int64_t global_rows = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_rows, &global_rows, 1, MPI_LONG_LONG, MPI_SUM,
        context.mpi_comm));

    gexec::SourceResolver resolver = [&](
        const std::string& name, const std::vector<std::string>&) {
        if (name == "numeric_input") return numeric_input;
        return name == "right_subset"
            ? right_subset->view()
            : input->view();
    };
    gamax::phys::Context physical{context.stream, nullptr};

    int fail = 0;
    auto require = [&](bool condition, const char* label) {
        if (condition) return;
        std::fprintf(stderr, "[rank %d] failed: %s\n", context.rank, label);
        fail = 1;
    };
    {
        auto local = unary_plan(
            gexec::StepKind::ScalarAgg,
            gexec::ScalarAggPayload{{
                {-1, gamax::groupby::aggregation_id::COUNT_ALL}}},
            gexec::DistributionKind::Unknown);
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local));
        require(
            planned.steps.back().kind ==
                gexec::StepKind::DistributedScalarAgg,
            "planner rewrites scalar aggregate");
        require(
            planned.values.front().distribution.kind ==
                gexec::DistributionKind::LocalShard,
            "planner marks source local shard");
        require(
            planned.values.back().distribution.kind ==
                gexec::DistributionKind::Replicated,
            "planner marks scalar aggregate replicated");
    }
    {
        auto local = binary_plan(
            gexec::StepKind::EqJoin,
            gexec::EqJoinPayload{
                {0}, {0}, gamax::phys::default_eq_join_hint()},
            gexec::DistributionKind::Unknown);
        alps::exec::DistributedPlanOptions options;
        options.gather_final_result = false;
        options.join_strategy_overrides.push_back({
            gexec::StepKind::EqJoin, {"key"}, {"key"},
            gexec::JoinDistributionStrategy::BroadcastRight});
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local), options);
        require(
            planned.steps.back().kind == gexec::StepKind::DistributedEqJoin,
            "planner rewrites EqJoin");
        const auto& payload = std::get<gexec::DistributedEqJoinPayload>(
            planned.steps.back().payload);
        require(
            payload.strategy == gexec::JoinDistributionStrategy::BroadcastRight,
            "planner selects BroadcastRight");
        require(
            !payload.partition_left && !payload.partition_right,
            "broadcast plan does not partition");
        require(
            !payload.replicate_left && payload.replicate_right,
            "broadcast plan replicates only right");
    }
    {
        auto local = binary_plan(
            gexec::StepKind::EqJoin,
            gexec::EqJoinPayload{
                {0}, {0}, gamax::phys::default_eq_join_hint()},
            gexec::DistributionKind::Unknown, "replicated");
        alps::exec::DistributedPlanOptions options;
        options.gather_final_result = false;
        options.source_distribution = [](const std::string& table) {
            return table == "replicated"
                ? gexec::DistributionKind::Replicated
                : gexec::DistributionKind::LocalShard;
        };
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local), options);
        const auto& payload = std::get<gexec::DistributedEqJoinPayload>(
            planned.steps.back().payload);
        require(
            planned.values[0].distribution.kind ==
                    gexec::DistributionKind::LocalShard &&
                planned.values[1].distribution.kind ==
                    gexec::DistributionKind::Replicated,
            "planner applies per-source distribution");
        require(
            payload.strategy == gexec::JoinDistributionStrategy::BroadcastRight &&
                !payload.partition_left && !payload.partition_right &&
                !payload.replicate_left && !payload.replicate_right,
            "replicated right input avoids all join movement");
        require(
            planned.values.back().distribution.kind ==
                gexec::DistributionKind::LocalShard,
            "replicated right join retains left distribution");
    }
    {
        auto local = binary_plan(
            gexec::StepKind::EqJoin,
            gexec::EqJoinPayload{
                {0}, {0}, gamax::phys::default_eq_join_hint()},
            gexec::DistributionKind::Unknown, "right_subset");
        alps::exec::DistributedPlanOptions options;
        options.gather_final_result = false;
        options.source_distribution = [](const std::string& table) {
            return table == "input"
                ? gexec::DistributionKind::Replicated
                : gexec::DistributionKind::LocalShard;
        };
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local), options);
        const auto& payload = std::get<gexec::DistributedEqJoinPayload>(
            planned.steps.back().payload);
        require(
            payload.strategy == gexec::JoinDistributionStrategy::BroadcastLeft &&
                !payload.partition_left && !payload.partition_right &&
                !payload.replicate_left && !payload.replicate_right,
            "replicated left input avoids all join movement");
        require(
            planned.values.back().distribution.kind ==
                gexec::DistributionKind::LocalShard,
            "replicated left join retains right distribution");
    }
    {
        auto local = binary_plan(
            gexec::StepKind::EqJoin,
            gexec::EqJoinPayload{
                {0}, {0}, gamax::phys::default_eq_join_hint()},
            gexec::DistributionKind::Unknown);
        alps::exec::DistributedPlanOptions options;
        options.source_distribution = [](const std::string&) {
            return gexec::DistributionKind::Replicated;
        };
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local), options);
        const auto& payload = std::get<gexec::DistributedEqJoinPayload>(
            planned.steps.back().payload);
        require(
            payload.strategy == gexec::JoinDistributionStrategy::BroadcastRight &&
                !payload.partition_left && !payload.partition_right &&
                !payload.replicate_left && !payload.replicate_right,
            "two replicated join inputs stay local");
        require(
            planned.values.back().distribution.kind ==
                gexec::DistributionKind::Replicated &&
                planned.steps.back().kind == gexec::StepKind::DistributedEqJoin,
            "two replicated join inputs produce replicated output");
    }
    {
        gexec::ExecPlan local;
        local.values.push_back(value(0, 1, gexec::DistributionKind::Unknown));
        local.values.push_back(value(1, 2, gexec::DistributionKind::Unknown));
        local.values.push_back(value(2, -1, gexec::DistributionKind::Unknown));
        local.values[0].schema = schema({"key", "payload"});
        local.values[1].schema = schema({"key", "payload"});
        local.values[2].schema = schema({"key"});
        auto filter_expr = gamax::expression::ExprTree::greater_than(
            gamax::expression::ExprTree::column(
                0, gamax::data_type{gamax::type_id::INT64}),
            gamax::expression::ExprTree::constant(gamax::scalar(int64_t{0})));
        local.steps.push_back(gexec::Step{
            gexec::StepKind::SourceTable, {}, 0,
            gexec::SourceTablePayload{"replicated", {"key", "payload"}}});
        local.steps.push_back(gexec::Step{
            gexec::StepKind::Filter, {0}, 1,
            gexec::FilterPayload{
                gamax::select::Predicate::expr(std::move(filter_expr)), {}, {}}});
        local.steps.push_back(gexec::Step{
            gexec::StepKind::Project, {1}, 2,
            gexec::ProjectPayload{{
                gamax::expression::ExprTree::column(
                    0, gamax::data_type{gamax::type_id::INT64})}, {}}});
        alps::exec::DistributedPlanOptions options;
        options.gather_final_result = false;
        options.source_distribution = [](const std::string&) {
            return gexec::DistributionKind::Replicated;
        };
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local), options);
        require(
            planned.values[0].distribution.kind ==
                    gexec::DistributionKind::Replicated &&
                planned.values[1].distribution.kind ==
                    gexec::DistributionKind::Replicated &&
                planned.values[2].distribution.kind ==
                    gexec::DistributionKind::Replicated,
            "filter and project preserve replicated placement");
    }
    {
        auto local = binary_plan(
            gexec::StepKind::EqJoin,
            gexec::EqJoinPayload{
                {0}, {0}, gamax::phys::default_eq_join_hint()},
            gexec::DistributionKind::Unknown);
        alps::exec::DistributedPlanOptions options;
        options.join_strategy_overrides.push_back({
            gexec::StepKind::EqJoin, {"stale_key"}, {"key"},
            gexec::JoinDistributionStrategy::BroadcastRight});
        bool rejected = false;
        try {
            (void)alps::exec::plan_for_distributed_execution(
                std::move(local), options);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "planner rejects unmatched join override");
    }
    {
        auto local = binary_plan(
            gexec::StepKind::EqJoin,
            gexec::EqJoinPayload{
                {0}, {0}, gamax::phys::default_eq_join_hint()},
            gexec::DistributionKind::Unknown);
        alps::exec::DistributedPlanOptions options;
        options.join_strategy_overrides.push_back({
            gexec::StepKind::EqJoin, {"key"}, {"key"},
            gexec::JoinDistributionStrategy::BroadcastLeft});
        options.join_strategy_overrides.push_back({
            gexec::StepKind::EqJoin, {"key"}, {"key"},
            gexec::JoinDistributionStrategy::BroadcastRight});
        bool rejected = false;
        try {
            (void)alps::exec::plan_for_distributed_execution(
                std::move(local), options);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "planner rejects ambiguous join override");
    }
    {
        auto local = binary_plan(
            gexec::StepKind::EqJoin,
            gexec::EqJoinPayload{
                {0}, {0}, gamax::phys::default_eq_join_hint()},
            gexec::DistributionKind::Unknown);
        local.values.back().schema = schema({"joined_key"});
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local));
        require(
            planned.steps.size() >= 2 &&
                planned.steps[planned.steps.size() - 2].kind ==
                    gexec::StepKind::DistributedEqJoin &&
                planned.steps.back().kind == gexec::StepKind::GatherToRoot,
            "planner gathers final partitioned result");
        require(
            planned.values.back().schema.size() == 1 &&
                planned.values.back().schema[0].name == "joined_key",
            "planner preserves schema through final gather");
    }
    {
        auto planned = alps::exec::plan_for_distributed_execution(
            replicated_concat_plan());
        require(
            planned.steps.back().kind == gexec::StepKind::Concat &&
                planned.values.back().distribution.kind ==
                    gexec::DistributionKind::Replicated,
            "planner preserves replicated concat distribution");
        auto concatenated = alps::exec::execute_distributed(
            planned, resolver, physical, context);
        ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
        require(
            concatenated->num_rows() == 2,
            "replicated concat does not multiply rows");
        if (concatenated->num_rows() == 2) {
            const auto counts = fixed_to_host<int64_t>(
                concatenated->get_column(0).view());
            require(
                counts[0] == global_rows && counts[1] == global_rows,
                "replicated concat values");
        }
    }
    {
        bool rejected = false;
        try {
            (void)alps::exec::plan_for_distributed_execution(
                mixed_concat_plan());
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "planner rejects mixed replicated concat");
    }
    {
        gexec::ExecPlan local;
        local.values.push_back(value(
            0, 1, gexec::DistributionKind::Unknown));
        local.values.push_back(value(
            1, 2, gexec::DistributionKind::Unknown));
        local.values.push_back(value(
            2, -1, gexec::DistributionKind::Unknown));
        local.steps.push_back(gexec::Step{
            gexec::StepKind::SourceTable, {}, 0,
            gexec::SourceTablePayload{
                "input", {"key", "name", "row_id"}}});
        local.steps.push_back(gexec::Step{
            gexec::StepKind::ScalarAgg, {0}, 1,
            gexec::ScalarAggPayload{{
                {-1, gamax::groupby::aggregation_id::COUNT_ALL}}}});
        local.steps.push_back(gexec::Step{
            gexec::StepKind::ScalarAgg, {1}, 2,
            gexec::ScalarAggPayload{{
                {0, gamax::groupby::aggregation_id::SUM}}}});
        auto planned = alps::exec::plan_for_distributed_execution(
            std::move(local));
        require(
            planned.steps[1].kind ==
                gexec::StepKind::DistributedScalarAgg &&
                planned.steps[2].kind == gexec::StepKind::ScalarAgg &&
                planned.values[2].distribution.kind ==
                    gexec::DistributionKind::Replicated,
            "planner keeps aggregate over replicated input local");
    }
    auto gathered = alps::exec::execute_distributed(
        shuffle_gather_plan(), resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        gathered->num_rows() == (context.rank == 0 ? global_rows : 0),
        "shuffle/gather row count");
    require(gathered->num_columns() == 3, "shuffle/gather columns");

    auto replicated = alps::exec::execute_distributed(
        replicate_plan(), resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(replicated->num_rows() == global_rows, "replicate row count");
    require(replicated->num_columns() == 3, "replicate columns");

    // Global CE_PULL keeps the Legacy operator schedule: the complete hash
    // partition/exchange and replication finish before downstream compute.
    // Both blocking collectives must match the NCCL result, including STRING
    // columns, an empty final rank, and gather-to-root.
    alps::exec::DistributedExecOptions legacy_ce_pull_options;
    legacy_ce_pull_options.backend = alps::collectives::Backend::CE_PULL;
    auto gathered_ce_pull = alps::exec::execute_distributed(
        shuffle_gather_plan(), resolver, physical, context,
        legacy_ce_pull_options);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        gathered_ce_pull->num_rows() == gathered->num_rows() &&
            gathered_ce_pull->num_columns() == gathered->num_columns() &&
            input_rows_to_host(gathered_ce_pull->view()) ==
                input_rows_to_host(gathered->view()),
        "Legacy CE-pull shuffle/gather matches NCCL");

    gexec::StepProfileSink replicate_ce_profile(/*collect_phases=*/true);
    gamax::phys::Context replicate_ce_physical{
        context.stream, &replicate_ce_profile};
    auto replicated_ce_pull = alps::exec::execute_distributed(
        replicate_plan(), resolver, replicate_ce_physical, context,
        legacy_ce_pull_options);
    replicate_ce_profile.finalize(context.stream);
    require(
        replicated_ce_pull->num_rows() == replicated->num_rows() &&
            replicated_ce_pull->num_columns() == replicated->num_columns() &&
            input_rows_to_host(replicated_ce_pull->view()) ==
                input_rows_to_host(replicated->view()),
        "Legacy CE-pull allgather matches NCCL");
    bool found_ce_pull_replicate = false;
    for (const auto& phase : replicate_ce_profile.phase_rows()) {
        if (phase.metadata.name == "broadcast.replicate") {
            found_ce_pull_replicate =
                phase.metadata.collective_primitive ==
                    "fabric_copy_engine_pull";
        }
    }
    require(
        found_ce_pull_replicate,
        "Legacy CE-pull allgather exposes the pull primitive");

    int64_t local_id_sum = 0;
    int64_t local_key_sum = 0;
    const int active_rows = context.rank == context.size - 1
        ? 0
        : 101 + context.rank;
    for (int64_t row = 0; row < active_rows; ++row) {
        local_id_sum += static_cast<int64_t>(context.rank) * 1000000 + row;
        local_key_sum += row % 13;
    }
    int64_t global_id_sum = 0;
    int64_t global_key_sum = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_id_sum, &global_id_sum, 1, MPI_LONG_LONG, MPI_SUM,
        context.mpi_comm));
    ALPS_MPICHECK(MPI_Allreduce(
        &local_key_sum, &global_key_sum, 1, MPI_LONG_LONG, MPI_SUM,
        context.mpi_comm));

    auto scalar_plan = unary_plan(
        gexec::StepKind::DistributedScalarAgg,
        gexec::DistributedScalarAggPayload{{
            {2, gamax::groupby::aggregation_id::SUM},
            {0, gamax::groupby::aggregation_id::MIN},
            {0, gamax::groupby::aggregation_id::MAX},
            {-1, gamax::groupby::aggregation_id::COUNT_ALL},
            {0, gamax::groupby::aggregation_id::AVG}}},
        gexec::DistributionKind::Replicated);
    auto scalar = alps::exec::execute_distributed(
        scalar_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        scalar->num_rows() == 1 && scalar->num_columns() == 5,
        "scalar aggregate shape");
    require(
        fixed_to_host<int64_t>(scalar->get_column(0).view())[0] == global_id_sum,
        "scalar SUM");
    require(
        fixed_to_host<int32_t>(scalar->get_column(1).view())[0] == 0,
        "scalar MIN");
    require(
        fixed_to_host<int32_t>(scalar->get_column(2).view())[0] == 12,
        "scalar MAX");
    require(
        fixed_to_host<int64_t>(scalar->get_column(3).view())[0] == global_rows,
        "scalar COUNT");
    const auto average = fixed_to_host<double>(scalar->get_column(4).view())[0];
    require(
        std::abs(
            average - static_cast<double>(global_key_sum) /
                          static_cast<double>(global_rows)) <= 1e-12,
        "scalar AVG");

    {
        int resolver_calls = 0;
        gexec::SourceResolver counted_resolver = [&resolver_calls, &input](
            const std::string&,
            const std::vector<std::string>&) {
            ++resolver_calls;
            return input->view();
        };
        const auto preload = gexec::bench_resolve_sources(
            scalar_plan, counted_resolver, context.stream);
        require(resolver_calls == 1, "benchmark preload resolves source once");

        for (int repetition = 0; repetition < 2; ++repetition) {
            auto repeated = alps::exec::benchmark_run_distributed_steps(
                scalar_plan, preload, physical, context);
            ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
            require(
                repeated->num_rows() == 1 && repeated->num_columns() == 5,
                "benchmark preload supports repeated distributed execution");
        }
        require(
            resolver_calls == 1,
            "benchmark execution does not resolve sources again");

        gexec::StepProfileSink profile_sink(/*collect_phases=*/true);
        gamax::phys::Context profiled_physical{
            context.stream, &profile_sink};
        auto profiled = alps::exec::benchmark_run_distributed_steps(
            scalar_plan, preload, profiled_physical, context);
        profile_sink.finalize(context.stream);
        require(
            profiled->num_rows() == 1,
            "profiled benchmark execution produces a result");
        require(
            profile_sink.rows().size() == 1 &&
                profile_sink.rows().front().step_index == 1 &&
                profile_sink.rows().front().kind ==
                    gexec::StepKind::DistributedScalarAgg,
            "benchmark step profile excludes source loading");
        require(
            profile_sink.phase_rows().size() == 3 &&
                profile_sink.phase_rows()[0].metadata.category ==
                    gexec::PhaseCategory::Compute &&
                profile_sink.phase_rows()[1].metadata.communication_pattern ==
                    gexec::CommunicationPattern::Reduce &&
                profile_sink.phase_rows()[2].metadata.name ==
                    "aggregate.finalize",
            "scalar aggregate exposes compute/reduce/finalize phases");
        const auto& reduce_phase = profile_sink.phase_rows()[1];
        require(
            reduce_phase.metadata.collective_primitive == "nccl_allreduce" &&
                reduce_phase.metadata.bytes_sent > 0 &&
                reduce_phase.metadata.bytes_received > 0 &&
                reduce_phase.metadata.rows_sent == 1 &&
                reduce_phase.metadata.rows_received == 1 &&
                reduce_phase.metadata.peer_count == context.size - 1,
            "reduce phase exposes primitive and transfer metadata");

        gexec::StepProfileSink step_only_sink;
        gamax::phys::Context step_only_physical{
            context.stream, &step_only_sink};
        auto step_only = alps::exec::benchmark_run_distributed_steps(
            scalar_plan, preload, step_only_physical, context);
        step_only_sink.finalize(context.stream);
        require(
            step_only->num_rows() == 1 &&
                step_only_sink.rows().size() == 1 &&
                step_only_sink.phase_rows().empty(),
            "disabled phase profiling emits no nested records");
        bool rejected_unmatched_end = false;
        try {
            step_only_sink.end_step(context.stream);
        } catch (const std::runtime_error&) {
            rejected_unmatched_end = true;
        }
        require(
            rejected_unmatched_end,
            "step profiling rejects an unmatched end_step");
    }

    auto grouped_plan = unary_plan(
        gexec::StepKind::DistributedGroupBy,
        gexec::DistributedGroupByPayload{
            2,
            {{0, gamax::groupby::aggregation_id::SUM},
             {-1, gamax::groupby::aggregation_id::COUNT_ALL}},
            {}},
        gexec::DistributionKind::HashPartitioned, {0, 1});
    std::get<gexec::DistributedGroupByPayload>(
        grouped_plan.steps.back().payload).hint.groupby_algo =
            gamax::groupby::HASH;
    auto grouped = alps::exec::execute_distributed(
        grouped_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    int64_t local_grouped_count = 0;
    for (const auto count : fixed_to_host<int64_t>(grouped->get_column(3).view())) {
        local_grouped_count += count;
    }
    int64_t global_grouped_count = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_grouped_count, &global_grouped_count, 1, MPI_LONG_LONG,
        MPI_SUM, context.mpi_comm));
    require(global_grouped_count == global_rows, "grouped COUNT");

    alps::exec::DistributedExecOptions group_overlap_options;
    group_overlap_options.group_by_implementation =
        alps::exec::DistributedGroupByImplementation::ColumnOverlap;
    gexec::StepProfileSink group_overlap_profile(/*collect_phases=*/true);
    gamax::phys::Context group_overlap_physical{
        context.stream, &group_overlap_profile};
    auto grouped_overlap = alps::exec::execute_distributed(
        grouped_plan, resolver, group_overlap_physical, context,
        group_overlap_options);
    group_overlap_profile.finalize(context.stream);
    int64_t local_overlap_count = 0;
    for (const auto count :
         fixed_to_host<int64_t>(grouped_overlap->get_column(3).view())) {
        local_overlap_count += count;
    }
    int64_t global_overlap_count = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_overlap_count, &global_overlap_count, 1, MPI_LONG_LONG,
        MPI_SUM, context.mpi_comm));
    require(
        global_overlap_count == global_rows,
        "column-overlap grouped COUNT");
    require(
        grouped_overlap->num_columns() == 4 &&
            grouped_overlap->get_column(1).type().id() ==
                gamax::type_id::STRING,
        "column-overlap grouped multi-key STRING shape");
    require(
        has_lifetime_stream(*grouped_overlap, context.stream),
        "column-overlap GroupBy returns caller-stream-owned buffers");
    bool found_group_ids = false;
    bool group_exchange_metadata_valid = true;
    std::set<std::pair<std::string, std::int64_t>> group_exchange_columns;
    for (const auto& phase : group_overlap_profile.phase_rows()) {
        found_group_ids |= phase.metadata.name ==
            "group_by.column_overlap.group_ids";
        if (phase.metadata.name ==
                "group_by.column_overlap.key_exchange" ||
            phase.metadata.name ==
                "group_by.column_overlap.state_exchange") {
            const auto& metadata = phase.metadata;
            group_exchange_metadata_valid &=
                metadata.bytes_sent >= 0 &&
                metadata.bytes_received >= 0 &&
                metadata.rows_sent >= 0 &&
                metadata.rows_received >= 0 &&
                metadata.peer_count == context.size - 1 &&
                !metadata.column_side.empty() &&
                metadata.column_ordinal >= 0;
            group_exchange_columns.emplace(
                metadata.column_side, metadata.column_ordinal);
        }
    }
    require(
        found_group_ids,
        "column-overlap GroupBy exposes group-ID phase");
    const std::set<std::pair<std::string, std::int64_t>>
        expected_group_exchange_columns{
            {"key", 0}, {"key", 1}, {"state", 2}, {"state", 3}};
    require(
        group_exchange_metadata_valid &&
            group_exchange_columns == expected_group_exchange_columns,
        "column-overlap GroupBy exposes per-column transfer metadata");
    bool group_decision_recorded = false;
    for (const auto& row : group_overlap_profile.rows()) {
        if (row.kind == gexec::StepKind::DistributedGroupBy) {
            group_decision_recorded =
                row.requested_implementation == "column_overlap" &&
                row.used_implementation == "column_overlap" &&
                row.fallback_reason.empty();
        }
    }
    require(
        group_decision_recorded,
        "column-overlap GroupBy records implementation decision");

    auto group_ce_options = group_overlap_options;
    group_ce_options.column_exchange_backend =
        alps::collectives::Backend::CE_PULL;
    gexec::StepProfileSink group_ce_profile(/*collect_phases=*/true);
    gamax::phys::Context group_ce_physical{
        context.stream, &group_ce_profile};
    auto grouped_ce = alps::exec::execute_distributed(
        grouped_plan, resolver, group_ce_physical, context,
        group_ce_options);
    group_ce_profile.finalize(context.stream);
    int64_t local_ce_grouped_count = 0;
    for (const auto count :
         fixed_to_host<int64_t>(grouped_ce->get_column(3).view())) {
        local_ce_grouped_count += count;
    }
    int64_t global_ce_grouped_count = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_ce_grouped_count, &global_ce_grouped_count, 1,
        MPI_LONG_LONG, MPI_SUM, context.mpi_comm));
    require(
        global_ce_grouped_count == global_overlap_count &&
            grouped_ce->num_columns() == grouped_overlap->num_columns() &&
            grouped_ce->get_column(1).type().id() ==
                gamax::type_id::STRING &&
            group_rows_to_host(grouped_ce->view()) ==
                group_rows_to_host(grouped_overlap->view()),
        "CE-pull GroupBy matches NCCL rows and STRING shape");
    require(
        has_lifetime_stream(*grouped_ce, context.stream),
        "CE-pull GroupBy returns caller-stream-owned buffers");
    std::size_t group_ce_exchange_columns = 0;
    bool group_ce_primitive_valid = true;
    for (const auto& phase : group_ce_profile.phase_rows()) {
        if (phase.metadata.name ==
                "group_by.column_overlap.key_exchange" ||
            phase.metadata.name ==
                "group_by.column_overlap.state_exchange") {
            ++group_ce_exchange_columns;
            group_ce_primitive_valid &=
                phase.metadata.collective_primitive ==
                    "cuda_memcpy_batch_pull" &&
                phase.metadata.bytes_sent >= 0 &&
                phase.metadata.bytes_received >= 0;
        }
    }
    require(
        group_ce_exchange_columns == 4 && group_ce_primitive_valid,
        "CE-pull GroupBy exposes four copy-engine column exchanges");

    auto invalid_group_options = group_overlap_options;
    invalid_group_options.backend = alps::collectives::Backend::CE;
    bool group_rejected_non_nccl = false;
    try {
        (void)alps::exec::execute_distributed(
            grouped_plan, resolver, physical, context,
            invalid_group_options);
    } catch (const std::invalid_argument&) {
        group_rejected_non_nccl = true;
    }
    require(
        group_rejected_non_nccl,
        "column-overlap GroupBy rejects a non-NCCL backend");

    auto distinct_plan = unary_plan(
        gexec::StepKind::DistributedDistinct,
        gexec::DistributedDistinctPayload{{0, 1}, {}},
        gexec::DistributionKind::HashPartitioned, {0, 1});
    auto distinct = alps::exec::execute_distributed(
        distinct_plan, resolver, physical, context);
    int64_t local_distinct = distinct->num_rows();
    int64_t global_distinct = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_distinct, &global_distinct, 1, MPI_LONG_LONG, MPI_SUM,
        context.mpi_comm));
    std::set<std::pair<int32_t, std::string>> expected_distinct;
    for (int rank = 0; rank < context.size - 1; ++rank) {
        for (int64_t row = 0; row < 101 + rank; ++row) {
            expected_distinct.emplace(
                static_cast<int32_t>(row % 13),
                row % 4 == 0 ? "" : "name-" + std::to_string(row % 7));
        }
    }
    require(
        global_distinct == static_cast<int64_t>(expected_distinct.size()),
        "distributed distinct cardinality");

    auto topk_plan = unary_plan(
        gexec::StepKind::DistributedTopK,
        gexec::DistributedTopKPayload{
            {{2, gamax::sort_order::DESCENDING}}, 10, {}, 0},
        gexec::DistributionKind::RootOnly, {}, 0);
    auto topk = alps::exec::execute_distributed(
        topk_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        topk->num_rows() == (context.rank == 0 ? 10 : 0),
        "distributed TopK row count");
    if (context.rank == 0) {
        const auto row_ids = fixed_to_host<int64_t>(topk->get_column(2).view());
        const int max_rank = context.size - 2;
        const int64_t expected_max =
            static_cast<int64_t>(max_rank) * 1000000 + (101 + max_rank - 1);
        require(
            !row_ids.empty() && row_ids.front() == expected_max,
            "distributed TopK maximum");
    }


    auto sort_plan = unary_plan(
        gexec::StepKind::DistributedSort,
        gexec::DistributedSortPayload{
            {{0, gamax::sort_order::ASCENDING},
             {1, gamax::sort_order::ASCENDING},
             {2, gamax::sort_order::DESCENDING}},
            {}, 0},
        gexec::DistributionKind::RootOnly, {}, 0);
    auto sorted = alps::exec::execute_distributed(
        sort_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        sorted->num_rows() == (context.rank == 0 ? global_rows : 0),
        "distributed sort row count");
    if (context.rank == 0) {
        const auto keys = fixed_to_host<int32_t>(sorted->get_column(0).view());
        const auto names = strings_to_host(sorted->get_column(1).view());
        const auto row_ids = fixed_to_host<int64_t>(sorted->get_column(2).view());
        for (std::size_t row = 1; row < keys.size(); ++row) {
            const bool ordered =
                keys[row - 1] < keys[row] ||
                (keys[row - 1] == keys[row] &&
                 (names[row - 1] < names[row] ||
                  (names[row - 1] == names[row] &&
                   row_ids[row - 1] >= row_ids[row])));
            if (!ordered) {
                require(false, "distributed sort ordering");
                break;
            }
        }
    }

    if (fail) {
        std::fprintf(
            stderr,
            "[rank %d] failure occurred before distributed join checks\n",
            context.rank);
    }

    int64_t local_key_counts[13] {};
    for (int64_t row = 0; row < active_rows; ++row) {
        ++local_key_counts[row % 13];
    }
    int64_t global_key_counts[13] {};
    ALPS_MPICHECK(MPI_Allreduce(
        local_key_counts, global_key_counts, 13, MPI_LONG_LONG, MPI_SUM,
        context.mpi_comm));
    int64_t expected_eq_rows = 0;
    int64_t expected_anti_rows = 0;
    for (int key = 0; key < 13; ++key) {
        expected_eq_rows += global_key_counts[key] * global_key_counts[key];
        if (key >= 6) expected_anti_rows += global_key_counts[key];
    }

    auto eq_shuffle_plan = binary_plan(
        gexec::StepKind::DistributedEqJoin,
        gexec::DistributedEqJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::Shuffle,
            true, true, false, false,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::HashPartitioned, "input", {0});
    auto eq_shuffle = alps::exec::execute_distributed(
        eq_shuffle_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    const auto eq_shuffle_rows = global_output_rows(*eq_shuffle, context);
    if (eq_shuffle_rows != expected_eq_rows) {
        std::fprintf(
            stderr, "[rank %d] eq shuffle rows: actual=%ld expected=%ld\n",
            context.rank, eq_shuffle_rows, expected_eq_rows);
        fail = 1;
    }
    require(eq_shuffle->num_columns() == 5, "eq shuffle output columns");
    if (eq_shuffle->num_columns() == 5) {
        require(
            eq_shuffle->get_column(1).type().id() == gamax::type_id::STRING,
            "eq shuffle left string payload");
        require(
            eq_shuffle->get_column(3).type().id() == gamax::type_id::STRING,
            "eq shuffle right string payload");
    }

    alps::exec::DistributedExecOptions overlap_options;
    overlap_options.eq_join_implementation =
        alps::exec::DistributedEqJoinImplementation::ColumnOverlap;
    auto overlap_hint = gamax::phys::default_eq_join_hint();
    overlap_hint.join_algo = gamax::join::HASH;
    overlap_hint.mat_algo = gamax::join::GFUR;
    auto eq_overlap = alps::exec::execute_distributed(
        numeric_binary_plan(gexec::DistributedEqJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::Shuffle,
            true, true, false, false,
            overlap_hint}),
        resolver, physical, context, overlap_options);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        global_output_rows(*eq_overlap, context) == expected_eq_rows,
        "column-overlap executor row count");
    require(eq_overlap->num_columns() == 3,
            "column-overlap executor output columns");

    gexec::StepProfileSink overlap_profile(/*collect_phases=*/true);
    gamax::phys::Context overlap_profiled_physical{
        context.stream, &overlap_profile};
    auto profiled_overlap = alps::exec::execute_distributed(
        numeric_binary_plan(gexec::DistributedEqJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::Shuffle,
            true, true, false, false,
            overlap_hint}),
        resolver, overlap_profiled_physical, context, overlap_options);
    overlap_profile.finalize(context.stream);
    require(
        profiled_overlap->num_columns() == 3,
        "profiled column-overlap executor output columns");
    bool found_key_exchange = false;
    bool found_payload_exchange = false;
    std::set<std::pair<std::string, std::int64_t>> key_exchange_columns;
    std::set<std::pair<std::string, std::int64_t>> payload_exchange_columns;
    for (const auto& phase : overlap_profile.phase_rows()) {
        const auto& metadata = phase.metadata;
        if (metadata.name == "join.column.key_exchange") {
            found_key_exchange = true;
            key_exchange_columns.emplace(
                metadata.column_side, metadata.column_ordinal);
            require(
                metadata.bytes_sent ==
                        metadata.rows_sent * static_cast<std::int64_t>(sizeof(int32_t)) &&
                    metadata.bytes_received ==
                        metadata.rows_received * static_cast<std::int64_t>(sizeof(int32_t)) &&
                    metadata.rows_sent >= 0 && metadata.rows_received >= 0 &&
                    metadata.peer_count == context.size - 1,
                "column-overlap key exchange exposes transfer metadata");
        } else if (metadata.name == "join.column.payload_exchange") {
            found_payload_exchange = true;
            payload_exchange_columns.emplace(
                metadata.column_side, metadata.column_ordinal);
            require(
                metadata.bytes_sent ==
                        metadata.rows_sent * static_cast<std::int64_t>(sizeof(int64_t)) &&
                    metadata.bytes_received ==
                        metadata.rows_received * static_cast<std::int64_t>(sizeof(int64_t)) &&
                    metadata.rows_sent >= 0 && metadata.rows_received >= 0 &&
                    metadata.peer_count == context.size - 1,
                "column-overlap payload exchange exposes transfer metadata");
        }
    }
    require(found_key_exchange && found_payload_exchange,
            "column-overlap profile exposes both exchange phases");
    require(
        key_exchange_columns ==
            std::set<std::pair<std::string, std::int64_t>>{
                {"left", 0}, {"right", 0}},
        "column-overlap key exchanges expose exact input columns");
    require(
        payload_exchange_columns ==
            std::set<std::pair<std::string, std::int64_t>>{
                {"left", 1}, {"right", 1}},
        "column-overlap payload exchanges expose exact input columns");

    gexec::StepProfileSink string_overlap_profile(/*collect_phases=*/true);
    gamax::phys::Context string_overlap_physical{
        context.stream, &string_overlap_profile};
    auto string_overlap = alps::exec::execute_distributed(
        eq_shuffle_plan, resolver, string_overlap_physical, context,
        overlap_options);
    string_overlap_profile.finalize(context.stream);
    require(
        global_output_rows(*string_overlap, context) == expected_eq_rows,
        "STRING column-overlap executor row count");
    require(string_overlap->num_columns() == 5,
            "STRING column-overlap executor output columns");
    if (string_overlap->num_columns() == 5) {
        require(
            string_overlap->get_column(1).type().id() == gamax::type_id::STRING &&
                string_overlap->get_column(3).type().id() == gamax::type_id::STRING,
            "STRING column-overlap executor preserves payload types");
        const auto keys = fixed_to_host<int32_t>(
            string_overlap->get_column(0).view());
        const auto left_names = strings_to_host(
            string_overlap->get_column(1).view());
        const auto left_ids = fixed_to_host<int64_t>(
            string_overlap->get_column(2).view());
        const auto right_names = strings_to_host(
            string_overlap->get_column(3).view());
        const auto right_ids = fixed_to_host<int64_t>(
            string_overlap->get_column(4).view());
        bool values_match =
            keys.size() == left_names.size() &&
            keys.size() == left_ids.size() &&
            keys.size() == right_names.size() &&
            keys.size() == right_ids.size();
        auto expected_name = [] (int64_t row_id) {
            const auto row = row_id % 1000000;
            return row % 4 == 0
                ? std::string{}
                : "name-" + std::to_string(row % 7);
        };
        for (std::size_t row = 0; values_match && row < keys.size(); ++row) {
            values_match =
                keys[row] == (left_ids[row] % 1000000) % 13 &&
                keys[row] == (right_ids[row] % 1000000) % 13 &&
                left_names[row] == expected_name(left_ids[row]) &&
                right_names[row] == expected_name(right_ids[row]);
        }
        require(values_match,
                "STRING column-overlap executor payload values");
    }
    bool found_lengths_exchange = false;
    bool found_chars_exchange = false;
    bool found_offsets_rebuild = false;
    std::set<std::pair<std::string, std::int64_t>> lengths_columns;
    std::set<std::pair<std::string, std::int64_t>> chars_columns;
    std::set<std::pair<std::string, std::int64_t>> offsets_columns;
    for (const auto& phase : string_overlap_profile.phase_rows()) {
        const auto& metadata = phase.metadata;
        if (metadata.name == "join.column.string_lengths_exchange") {
            found_lengths_exchange = true;
            lengths_columns.emplace(
                metadata.column_side, metadata.column_ordinal);
            require(
                metadata.bytes_sent ==
                        metadata.rows_sent *
                            static_cast<std::int64_t>(sizeof(int32_t)) &&
                    metadata.bytes_received ==
                        metadata.rows_received *
                            static_cast<std::int64_t>(sizeof(int32_t)) &&
                    metadata.peer_count == context.size - 1,
                "STRING lengths exchange exposes transfer metadata");
        } else if (metadata.name == "join.column.string_chars_exchange") {
            found_chars_exchange = true;
            chars_columns.emplace(
                metadata.column_side, metadata.column_ordinal);
            require(
                metadata.bytes_sent >= 0 && metadata.bytes_received >= 0 &&
                    metadata.rows_sent >= 0 && metadata.rows_received >= 0 &&
                    metadata.peer_count == context.size - 1,
                "STRING chars exchange exposes transfer metadata");
        } else if (metadata.name == "join.column.string_offsets_rebuild") {
            found_offsets_rebuild = true;
            offsets_columns.emplace(
                metadata.column_side, metadata.column_ordinal);
        }
    }
    require(
        found_lengths_exchange && found_chars_exchange && found_offsets_rebuild,
        "STRING column-overlap profile exposes staged phases");
    const std::set<std::pair<std::string, std::int64_t>> string_columns{
        {"left", 1}, {"right", 1}};
    require(
        lengths_columns == string_columns &&
            chars_columns == string_columns &&
            offsets_columns == string_columns,
        "STRING column-overlap phases expose exact input columns");

    auto eq_broadcast_plan = binary_plan(
        gexec::StepKind::DistributedEqJoin,
        gexec::DistributedEqJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::BroadcastRight,
            false, false, false, true,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::LocalShard);
    auto eq_broadcast = alps::exec::execute_distributed(
        eq_broadcast_plan, resolver, physical, context, overlap_options);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    const auto eq_broadcast_rows = global_output_rows(*eq_broadcast, context);
    if (eq_broadcast_rows != expected_eq_rows) {
        std::fprintf(
            stderr, "[rank %d] eq broadcast rows: actual=%ld expected=%ld\n",
            context.rank, eq_broadcast_rows, expected_eq_rows);
        fail = 1;
    }
    gexec::StepProfileSink eq_fallback_profile(/*collect_phases=*/false);
    gamax::phys::Context eq_fallback_physical{
        context.stream, &eq_fallback_profile};
    auto eq_fallback_options = overlap_options;
    eq_fallback_options.shuffle_alignment =
        alps::collectives::AlignmentPolicy::On;
    auto eq_fallback = alps::exec::execute_distributed(
        eq_broadcast_plan, resolver, eq_fallback_physical, context,
        eq_fallback_options);
    eq_fallback_profile.finalize(context.stream);
    require(
        global_output_rows(*eq_fallback, context) == expected_eq_rows,
        "column-overlap EqJoin fallback cardinality");
    bool eq_fallback_recorded = false;
    for (const auto& row : eq_fallback_profile.rows()) {
        if (row.kind == gexec::StepKind::DistributedEqJoin) {
            eq_fallback_recorded =
                row.requested_implementation == "column_overlap" &&
                row.used_implementation == "legacy" &&
                row.fallback_reason == "distribution_not_shuffle";
        }
    }
    require(
        eq_fallback_recorded,
        "unsupported EqJoin bypasses overlap guards and records fallback");

    auto semi_plan = binary_plan(
        gexec::StepKind::DistributedSemiJoin,
        gexec::DistributedSemiJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::Shuffle,
            true, true, false, false,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::HashPartitioned, "input", {0});
    auto semi = alps::exec::execute_distributed(
        semi_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    const auto semi_rows = global_output_rows(*semi, context);
    if (semi_rows != global_rows) {
        std::fprintf(
            stderr, "[rank %d] semi rows: actual=%ld expected=%ld\n",
            context.rank, semi_rows, global_rows);
        fail = 1;
    }
    require(semi->num_columns() == 3, "semi output columns");

    alps::exec::DistributedExecOptions semi_overlap_options;
    semi_overlap_options.semi_join_implementation =
        alps::exec::DistributedSemiJoinImplementation::ColumnOverlap;
    gexec::StepProfileSink semi_overlap_profile(/*collect_phases=*/true);
    gamax::phys::Context semi_overlap_physical{
        context.stream, &semi_overlap_profile};
    auto semi_overlap = alps::exec::execute_distributed(
        semi_plan, resolver, semi_overlap_physical, context,
        semi_overlap_options);
    semi_overlap_profile.finalize(context.stream);
    require(
        global_output_rows(*semi_overlap, context) == global_rows,
        "column-overlap SemiJoin cardinality");
    require(
        semi_overlap->num_columns() == 3 &&
            semi_overlap->get_column(1).type().id() ==
                gamax::type_id::STRING,
        "column-overlap SemiJoin preserves STRING payload");
    require(
        has_lifetime_stream(*semi_overlap, context.stream),
        "column-overlap SemiJoin returns caller-stream-owned buffers");
    bool found_semi_build = false;
    bool found_semi_probe_exchange = false;
    bool semi_exchange_metadata_valid = true;
    std::set<std::pair<std::string, std::int64_t>> semi_exchange_columns;
    for (const auto& phase : semi_overlap_profile.phase_rows()) {
        found_semi_build |=
            phase.metadata.name == "semi_join.column_overlap.build";
        found_semi_probe_exchange |=
            phase.metadata.name == "semi_join.column_overlap.probe_exchange";
        if (phase.metadata.name ==
                "semi_join.column_overlap.build_key_exchange" ||
            phase.metadata.name ==
                "semi_join.column_overlap.probe_exchange") {
            const auto& metadata = phase.metadata;
            semi_exchange_metadata_valid &=
                metadata.bytes_sent >= 0 &&
                metadata.bytes_received >= 0 &&
                metadata.rows_sent >= 0 &&
                metadata.rows_received >= 0 &&
                metadata.peer_count == context.size - 1 &&
                !metadata.column_side.empty() &&
                metadata.column_ordinal >= 0;
            semi_exchange_columns.emplace(
                metadata.column_side, metadata.column_ordinal);
        }
    }
    require(
        found_semi_build && found_semi_probe_exchange,
        "column-overlap SemiJoin exposes build and probe-exchange phases");
    const std::set<std::pair<std::string, std::int64_t>>
        expected_semi_exchange_columns{
            {"right", 0}, {"left", 0}, {"left", 1}, {"left", 2}};
    require(
        semi_exchange_metadata_valid &&
            semi_exchange_columns == expected_semi_exchange_columns,
        "column-overlap SemiJoin exposes per-column transfer metadata");
    bool semi_decision_recorded = false;
    for (const auto& row : semi_overlap_profile.rows()) {
        if (row.kind == gexec::StepKind::DistributedSemiJoin) {
            semi_decision_recorded =
                row.requested_implementation == "column_overlap" &&
                row.used_implementation == "column_overlap" &&
                row.fallback_reason.empty();
        }
    }
    require(
        semi_decision_recorded,
        "column-overlap SemiJoin records implementation decision");

    auto semi_ce_options = semi_overlap_options;
    semi_ce_options.column_exchange_backend =
        alps::collectives::Backend::CE_PULL;
    gexec::StepProfileSink semi_ce_profile(/*collect_phases=*/true);
    gamax::phys::Context semi_ce_physical{
        context.stream, &semi_ce_profile};
    auto semi_ce = alps::exec::execute_distributed(
        semi_plan, resolver, semi_ce_physical, context, semi_ce_options);
    semi_ce_profile.finalize(context.stream);
    require(
        global_output_rows(*semi_ce, context) == semi_rows &&
            semi_ce->num_columns() == semi_overlap->num_columns() &&
            semi_ce->get_column(1).type().id() == gamax::type_id::STRING &&
            input_rows_to_host(semi_ce->view()) ==
                input_rows_to_host(semi_overlap->view()),
        "CE-pull SemiJoin matches NCCL rows and STRING shape");
    require(
        has_lifetime_stream(*semi_ce, context.stream),
        "CE-pull SemiJoin returns caller-stream-owned buffers");
    std::size_t semi_ce_exchange_columns = 0;
    bool semi_ce_primitive_valid = true;
    for (const auto& phase : semi_ce_profile.phase_rows()) {
        if (phase.metadata.name ==
                "semi_join.column_overlap.build_key_exchange" ||
            phase.metadata.name ==
                "semi_join.column_overlap.probe_exchange") {
            ++semi_ce_exchange_columns;
            semi_ce_primitive_valid &=
                phase.metadata.collective_primitive ==
                    "cuda_memcpy_batch_pull" &&
                phase.metadata.bytes_sent >= 0 &&
                phase.metadata.bytes_received >= 0;
        }
    }
    require(
        semi_ce_exchange_columns == 4 && semi_ce_primitive_valid,
        "CE-pull SemiJoin exposes four copy-engine column exchanges");

    auto semi_no_partition_plan = binary_plan(
        gexec::StepKind::DistributedSemiJoin,
        gexec::DistributedSemiJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::Shuffle,
            false, false, false, false,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::HashPartitioned, "input", {0});
    const auto stats_before_zero_traffic =
        alps::collectives::ce_pull_stats(context);
    auto semi_ce_zero_traffic = alps::exec::execute_distributed(
        semi_no_partition_plan, resolver, physical, context,
        semi_ce_options);
    const auto stats_after_zero_traffic =
        alps::collectives::ce_pull_stats(context);
    require(
        global_output_rows(*semi_ce_zero_traffic, context) == global_rows,
        "CE-pull SemiJoin supports already-partitioned inputs");
    require(
        stats_after_zero_traffic.enqueued_columns ==
            stats_before_zero_traffic.enqueued_columns &&
            stats_after_zero_traffic.remote_bytes ==
                stats_before_zero_traffic.remote_bytes,
        "already-partitioned CE-pull SemiJoin records zero traffic");

    const auto warm_ce_stats =
        alps::collectives::ce_pull_stats(context);
    require(
        warm_ce_stats.arena_growth_count > 0 &&
            warm_ce_stats.enqueued_columns >= 8 &&
            warm_ce_stats.arena_overflow_count == 0,
        "GroupBy and SemiJoin warm the CE-pull arena");
    alps::collectives::freeze_ce_pull_arena(context);
    auto grouped_ce_frozen = alps::exec::execute_distributed(
        grouped_plan, resolver, physical, context, group_ce_options);
    auto semi_ce_frozen = alps::exec::execute_distributed(
        semi_plan, resolver, physical, context, semi_ce_options);
    int64_t local_frozen_grouped_count = 0;
    for (const auto count :
         fixed_to_host<int64_t>(grouped_ce_frozen->get_column(3).view())) {
        local_frozen_grouped_count += count;
    }
    int64_t global_frozen_grouped_count = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_frozen_grouped_count, &global_frozen_grouped_count, 1,
        MPI_LONG_LONG, MPI_SUM, context.mpi_comm));
    require(
        global_frozen_grouped_count == global_overlap_count &&
            global_output_rows(*semi_ce_frozen, context) == semi_rows &&
            group_rows_to_host(grouped_ce_frozen->view()) ==
                group_rows_to_host(grouped_overlap->view()) &&
            input_rows_to_host(semi_ce_frozen->view()) ==
                input_rows_to_host(semi_overlap->view()),
        "frozen-arena CE-pull relational repeats match NCCL");
    const auto frozen_ce_stats =
        alps::collectives::ce_pull_stats(context);
    require(
        frozen_ce_stats.arena_growth_count ==
            warm_ce_stats.arena_growth_count &&
            frozen_ce_stats.arena_capacity_bytes ==
                warm_ce_stats.arena_capacity_bytes &&
            frozen_ce_stats.slot_capacity_bytes ==
                warm_ce_stats.slot_capacity_bytes &&
            frozen_ce_stats.arena_overflow_count == 0 &&
            frozen_ce_stats.enqueued_columns ==
                warm_ce_stats.enqueued_columns + 8,
        "frozen CE-pull arena remains stable across relational repeats");

    auto invalid_semi_options = semi_overlap_options;
    invalid_semi_options.shuffle_alignment =
        alps::collectives::AlignmentPolicy::On;
    bool semi_rejected_alignment = false;
    try {
        (void)alps::exec::execute_distributed(
            semi_plan, resolver, physical, context,
            invalid_semi_options);
    } catch (const std::invalid_argument&) {
        semi_rejected_alignment = true;
    }
    require(
        semi_rejected_alignment,
        "column-overlap SemiJoin rejects alignment-aware shuffle");

    auto semi_nonzero_key_plan = binary_plan(
        gexec::StepKind::DistributedSemiJoin,
        gexec::DistributedSemiJoinPayload{
            {2}, {2}, gexec::JoinDistributionStrategy::Shuffle,
            true, true, false, false,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::HashPartitioned, "input", {2});
    auto semi_nonzero_key = alps::exec::execute_distributed(
        semi_nonzero_key_plan, resolver, physical, context,
        semi_overlap_options);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        global_output_rows(*semi_nonzero_key, context) == global_rows,
        "column-overlap SemiJoin supports a nonzero key ordinal");
    require(
        semi_nonzero_key->num_columns() == 3 &&
            semi_nonzero_key->get_column(0).type().id() ==
                gamax::type_id::INT32 &&
            semi_nonzero_key->get_column(1).type().id() ==
                gamax::type_id::STRING &&
            semi_nonzero_key->get_column(2).type().id() ==
                gamax::type_id::INT64,
        "column-overlap SemiJoin restores original left-column order");
    if (semi_nonzero_key->num_columns() == 3) {
        const auto keys = fixed_to_host<int32_t>(
            semi_nonzero_key->get_column(0).view());
        const auto names = strings_to_host(
            semi_nonzero_key->get_column(1).view());
        const auto row_ids = fixed_to_host<int64_t>(
            semi_nonzero_key->get_column(2).view());
        bool values_match =
            keys.size() == names.size() && keys.size() == row_ids.size();
        for (std::size_t row = 0;
             values_match && row < row_ids.size(); ++row) {
            const auto source_row = row_ids[row] % 1000000;
            const auto expected_name = source_row % 4 == 0
                ? std::string{}
                : "name-" + std::to_string(source_row % 7);
            values_match =
                keys[row] == source_row % 13 &&
                names[row] == expected_name;
        }
        require(
            values_match,
            "column-overlap SemiJoin preserves non-key payload values");
    }

    auto semi_broadcast_plan = binary_plan(
        gexec::StepKind::DistributedSemiJoin,
        gexec::DistributedSemiJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::BroadcastRight,
            false, false, false, true,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::LocalShard);
    auto semi_broadcast = alps::exec::execute_distributed(
        semi_broadcast_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        global_output_rows(*semi_broadcast, context) == global_rows,
        "semi broadcast cardinality");
    gexec::StepProfileSink semi_fallback_profile(/*collect_phases=*/false);
    gamax::phys::Context semi_fallback_physical{
        context.stream, &semi_fallback_profile};
    auto semi_fallback_options = semi_overlap_options;
    semi_fallback_options.column_exchange_backend =
        alps::collectives::Backend::CE_PULL;
    semi_fallback_options.shuffle_alignment =
        alps::collectives::AlignmentPolicy::On;
    auto semi_fallback = alps::exec::execute_distributed(
        semi_broadcast_plan, resolver, semi_fallback_physical, context,
        semi_fallback_options);
    semi_fallback_profile.finalize(context.stream);
    require(
        global_output_rows(*semi_fallback, context) == global_rows,
        "column-overlap SemiJoin fallback cardinality");
    bool semi_fallback_recorded = false;
    for (const auto& row : semi_fallback_profile.rows()) {
        if (row.kind == gexec::StepKind::DistributedSemiJoin) {
            semi_fallback_recorded =
                row.requested_implementation == "column_overlap" &&
                row.used_implementation == "legacy" &&
                row.fallback_reason == "distribution_not_shuffle";
        }
    }
    require(
        semi_fallback_recorded,
        "unsupported SemiJoin bypasses overlap guards and records fallback");

    auto anti_plan = binary_plan(
        gexec::StepKind::DistributedAntiJoin,
        gexec::DistributedAntiJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::Shuffle,
            true, true, false, false,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::HashPartitioned, "right_subset", {0});
    auto anti = alps::exec::execute_distributed(
        anti_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    const auto anti_rows = global_output_rows(*anti, context);
    if (anti_rows != expected_anti_rows) {
        std::fprintf(
            stderr, "[rank %d] anti rows: actual=%ld expected=%ld\n",
            context.rank, anti_rows, expected_anti_rows);
        fail = 1;
    }
    require(anti->num_columns() == 3, "anti output columns");

    auto anti_broadcast_plan = binary_plan(
        gexec::StepKind::DistributedAntiJoin,
        gexec::DistributedAntiJoinPayload{
            {0}, {0}, gexec::JoinDistributionStrategy::BroadcastRight,
            false, false, false, true,
            gamax::phys::default_eq_join_hint()},
        gexec::DistributionKind::LocalShard, "right_subset");
    auto anti_broadcast = alps::exec::execute_distributed(
        anti_broadcast_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    require(
        global_output_rows(*anti_broadcast, context) == expected_anti_rows,
        "anti broadcast cardinality");

    auto cross_plan = binary_plan(
        gexec::StepKind::DistributedCrossJoin,
        gexec::DistributedCrossJoinPayload{
            gexec::JoinDistributionStrategy::BroadcastRight, false, true},
        gexec::DistributionKind::LocalShard);
    auto crossed = alps::exec::execute_distributed(
        cross_plan, resolver, physical, context);
    ALPS_CUDACHECK(cudaStreamSynchronize(context.stream));
    const auto cross_rows = global_output_rows(*crossed, context);
    if (cross_rows != global_rows * global_rows) {
        std::fprintf(
            stderr, "[rank %d] cross rows: actual=%ld expected=%ld\n",
            context.rank, cross_rows, global_rows * global_rows);
        fail = 1;
    }
    require(crossed->num_columns() == 6, "cross output columns");
    return fail;
}

} // namespace

int main(int argc, char** argv) {
    alps::configure_process();
    MPI_Init(&argc, &argv);
    auto context = alps::make_context();
    int local_fail = run(context);
    int global_fail = 0;
    ALPS_MPICHECK(MPI_Allreduce(
        &local_fail, &global_fail, 1, MPI_INT, MPI_MAX, context.mpi_comm));
    if (context.rank == 0) {
        std::printf(
            "[distributed_executor_test] np=%d physical distribution steps -> %s\n",
            context.size, global_fail ? "FAIL" : "PASS");
    }
    alps::destroy_context(context);
    MPI_Finalize();
    return global_fail;
}
