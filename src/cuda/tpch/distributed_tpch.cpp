// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "tpch/distributed_tpch.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ir/lower_to_exec_plan.hpp"
#include "ir/rules.hpp"
#include "tpch/tpch_layout.hpp"
#include "tpch/tpch_ir.hpp"

namespace gamax::tpch::dist {

void preflight_dataset(int scale_factor, MPI_Comm communicator) {
    int rank = 0;
    MPI_Comm_rank(communicator, &rank);

    int success = 1;
    std::string diagnostic;
    if (rank == 0) {
        try {
            const char* data_path = std::getenv("TPCH_DATA_PATH");
            if (data_path == nullptr || *data_path == '\0') {
                throw std::runtime_error(
                    "TPCH_DATA_PATH is not set for dataset preflight");
            }
            const auto sf_dir =
                std::filesystem::path(data_path) /
                ("sf" + std::to_string(scale_factor));
            preflight_tpch_dataset(sf_dir.string(), scale_factor);
        } catch (const std::exception& error) {
            success = 0;
            diagnostic = error.what();
        }
    }

    MPI_Bcast(&success, 1, MPI_INT, 0, communicator);
    std::uint64_t length =
        rank == 0 ? static_cast<std::uint64_t>(diagnostic.size()) : 0;
    MPI_Bcast(&length, 1, MPI_UINT64_T, 0, communicator);
    if (rank != 0) diagnostic.resize(static_cast<std::size_t>(length));
    if (length > 0) {
        MPI_Bcast(
            diagnostic.data(), static_cast<int>(length), MPI_CHAR, 0,
            communicator);
    }
    if (!success) {
        throw std::runtime_error("TPC-H dataset preflight failed: " + diagnostic);
    }
}

namespace {

gamax::ir::PlanPtr build_query(
    int query_id,
    const gamax::ir::QueryBuilder& query_builder)
{
    using namespace gamax::ir::tpch;
    switch (query_id) {
        case 1: return build_q1(query_builder);
        case 2: return build_q2(query_builder);
        case 3: return build_q3(query_builder);
        case 4: return build_q4(query_builder);
        case 5: return build_q5(query_builder);
        case 6: return build_q6(query_builder);
        case 7: return build_q7(query_builder);
        case 8: return build_q8(query_builder);
        case 9: return build_q9(query_builder);
        case 10: return build_q10(query_builder);
        case 11: return build_q11(query_builder);
        case 12: return build_q12(query_builder);
        case 13: return build_q13(query_builder);
        case 14: return build_q14(query_builder);
        case 15: return build_q15(query_builder);
        case 16: return build_q16(query_builder);
        case 17: return build_q17(query_builder);
        case 18: return build_q18(query_builder);
        case 19: return build_q19(query_builder);
        case 20: return build_q20(query_builder);
        case 21: return build_q21(query_builder);
        case 22: return build_q22(query_builder);
        default:
            throw std::runtime_error(
                "distributed TPC-H supports query IDs 1 through 22");
    }
}

std::string join_signatures(const gamax::exec::ExecPlan& plan) {
    std::ostringstream output;
    auto print_keys = [&](gamax::exec::ValueId value_id,
                          const std::vector<gamax::size_type>& keys) {
        const auto& schema = plan.values.at(
            static_cast<std::size_t>(value_id)).schema;
        output << "[";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i != 0) output << ", ";
            output << schema[static_cast<std::size_t>(keys[i])].name;
        }
        output << "]";
    };

    for (std::size_t index = 0; index < plan.steps.size(); ++index) {
        const auto& step = plan.steps[index];
        const std::vector<gamax::size_type>* left = nullptr;
        const std::vector<gamax::size_type>* right = nullptr;
        if (step.kind == gamax::exec::StepKind::EqJoin) {
            const auto& payload =
                std::get<gamax::exec::EqJoinPayload>(step.payload);
            left = &payload.lkeys;
            right = &payload.rkeys;
        } else if (step.kind == gamax::exec::StepKind::SemiJoin) {
            const auto& payload =
                std::get<gamax::exec::SemiJoinPayload>(step.payload);
            left = &payload.lkeys;
            right = &payload.rkeys;
        } else if (step.kind == gamax::exec::StepKind::AntiJoin) {
            const auto& payload =
                std::get<gamax::exec::AntiJoinPayload>(step.payload);
            left = &payload.lkeys;
            right = &payload.rkeys;
        }
        if (left == nullptr) continue;
        output << "step " << index << " left=";
        print_keys(step.inputs[0], *left);
        output << " right=";
        print_keys(step.inputs[1], *right);
        output << "\n";
    }
    return output.str();
}

} // namespace

alps::exec::DistributedPlanOptions plan_options(
    int query_id,
    SourceReadPoliciesPtr source_read_policies)
{
    if (!source_read_policies) {
        throw std::runtime_error(
            "distributed TPC-H source read policies cannot be null");
    }

    using Strategy = gamax::exec::JoinDistributionStrategy;
    using StepKind = gamax::exec::StepKind;
    alps::exec::DistributedPlanOptions options;
    options.root = 0;
    options.gather_final_result = true;
    options.source_distribution = [policies = std::move(source_read_policies)](
        const std::string& table_name) {
        return policies->for_table(table_name) == SourceReadPolicy::Replicated
            ? gamax::exec::DistributionKind::Replicated
            : gamax::exec::DistributionKind::LocalShard;
    };

    auto broadcast_left = [&] (
        StepKind kind,
        std::vector<std::string> left_keys,
        std::vector<std::string> right_keys) {
        options.join_strategy_overrides.push_back({
            kind, std::move(left_keys), std::move(right_keys),
            Strategy::BroadcastLeft});
    };
    auto broadcast_right = [&] (
        StepKind kind,
        std::vector<std::string> left_keys,
        std::vector<std::string> right_keys) {
        options.join_strategy_overrides.push_back({
            kind, std::move(left_keys), std::move(right_keys),
            Strategy::BroadcastRight});
    };

    // Only non-replicated Spark-guided broadcasts remain here. Nation and
    // region movement is derived generically from source placement.
    switch (query_id) {
        case 2:
            broadcast_left(StepKind::EqJoin, {"s_suppkey"}, {"ps_suppkey"});
            break;
        case 5:
            broadcast_right(StepKind::EqJoin, {"l_suppkey"}, {"s_suppkey"});
            break;
        case 7:
            broadcast_left(StepKind::EqJoin, {"s_suppkey"}, {"l_suppkey"});
            break;
        case 8:
            broadcast_right(StepKind::EqJoin, {"l_partkey"}, {"p_partkey"});
            break;
        case 11:
            broadcast_right(StepKind::EqJoin, {"ps_suppkey"}, {"s_suppkey"});
            break;
        case 15:
            broadcast_right(StepKind::EqJoin, {"s_suppkey"}, {"supplier_no"});
            break;
        case 16:
            broadcast_right(StepKind::AntiJoin, {"ps_suppkey"}, {"s_suppkey"});
            break;
        case 17:
            broadcast_left(StepKind::EqJoin, {"p_partkey"}, {"l_partkey"});
            broadcast_left(StepKind::EqJoin, {"l_partkey"}, {"l_partkey"});
            break;
        case 19:
            broadcast_right(StepKind::EqJoin, {"l_partkey"}, {"p_partkey"});
            break;
        case 21:
            broadcast_right(StepKind::EqJoin, {"l_suppkey"}, {"s_suppkey"});
            break;
        default:
            break;
    }
    return options;
}

gamax::exec::ExecPlan build_local_exec_plan(
    int query_id,
    std::string* local_join_signatures)
{
    auto catalog = gamax::ir::tpch::make_tpch_catalog();
    gamax::ir::IdGenerator generator;
    gamax::ir::QueryBuilder query_builder(catalog, generator);
    auto logical_plan = gamax::ir::apply_rules(
        build_query(query_id, query_builder));

    gamax::ir::LowerOptions lower_options;
    gamax::ir::tpch::apply_query_hints(query_id, lower_options);
    auto local_plan = gamax::ir::lower_to_exec_plan(
        logical_plan, catalog, lower_options);
    if (local_join_signatures) {
        *local_join_signatures = join_signatures(local_plan);
    }
    return local_plan;
}

gamax::exec::ExecPlan distribute_exec_plan(
    int query_id,
    SourceReadPoliciesPtr source_read_policies,
    gamax::exec::ExecPlan local_plan)
{
    return alps::exec::plan_for_distributed_execution(
        std::move(local_plan),
        plan_options(query_id, std::move(source_read_policies)));
}

gamax::exec::ExecPlan build_exec_plan(
    int query_id,
    SourceReadPoliciesPtr source_read_policies,
    std::string* local_join_signatures)
{
    return distribute_exec_plan(
        query_id, std::move(source_read_policies),
        build_local_exec_plan(query_id, local_join_signatures));
}

} // namespace gamax::tpch::dist
