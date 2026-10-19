// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "alps/exec/distributed_planner.hpp"

#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "exec/print.hpp"

namespace alps::exec {
namespace {

using gamax::exec::Distribution;
using gamax::exec::DistributionKind;
using gamax::exec::ExecPlan;
using gamax::exec::Step;
using gamax::exec::StepKind;
using gamax::exec::ValueId;
using gamax::size_type;

std::vector<size_type> sequence(size_type count) {
    std::vector<size_type> result(static_cast<std::size_t>(count));
    for (size_type i = 0; i < count; ++i) result[i] = i;
    return result;
}

Distribution unknown() {
    return Distribution{};
}

Distribution local_shard() {
    Distribution result;
    result.kind = DistributionKind::LocalShard;
    return result;
}

Distribution replicated() {
    Distribution result;
    result.kind = DistributionKind::Replicated;
    return result;
}

Distribution source_distribution(
    const DistributedPlanOptions& options,
    const std::string& table_name)
{
    if (!options.source_distribution) return local_shard();
    const auto kind = options.source_distribution(table_name);
    if (kind == DistributionKind::LocalShard) return local_shard();
    if (kind == DistributionKind::Replicated) return replicated();
    throw std::runtime_error(
        "distributed planner: source distribution for '" + table_name +
        "' must be LocalShard or Replicated");
}

Distribution root_only(int root) {
    Distribution result;
    result.kind = DistributionKind::RootOnly;
    result.root = root;
    return result;
}

Distribution hash_partitioned(std::vector<size_type> keys) {
    Distribution result;
    result.kind = DistributionKind::HashPartitioned;
    result.key_ordinals = std::move(keys);
    return result;
}

bool hash_partitioned_on(
    const Distribution& distribution,
    const std::vector<size_type>& keys)
{
    return distribution.kind == DistributionKind::HashPartitioned &&
        distribution.key_ordinals == keys;
}

bool is_replicated(const Distribution& distribution) {
    return distribution.kind == DistributionKind::Replicated;
}

const Distribution& input_distribution(
    const ExecPlan& plan, const Step& step, std::size_t input);

Distribution concat_distribution(
    const ExecPlan& plan,
    const Step& step)
{
    if (step.inputs.empty()) {
        throw std::runtime_error(
            "distributed planner: concat requires at least one input");
    }

    const auto& first = input_distribution(plan, step, 0);
    bool all_replicated = first.kind == DistributionKind::Replicated;
    bool all_same_root = first.kind == DistributionKind::RootOnly;
    bool all_same_hash = first.kind == DistributionKind::HashPartitioned;
    bool contains_replicated = all_replicated;

    for (std::size_t input = 0; input < step.inputs.size(); ++input) {
        const auto& distribution = input_distribution(plan, step, input);
        if (distribution.kind == DistributionKind::Unknown) {
            throw std::runtime_error(
                "distributed planner: concat input has unknown distribution");
        }
        contains_replicated = contains_replicated ||
            distribution.kind == DistributionKind::Replicated;
        all_replicated = all_replicated &&
            distribution.kind == DistributionKind::Replicated;
        all_same_root = all_same_root &&
            distribution.kind == DistributionKind::RootOnly &&
            distribution.root == first.root;
        all_same_hash = all_same_hash &&
            distribution.kind == DistributionKind::HashPartitioned &&
            distribution.key_ordinals == first.key_ordinals;
    }

    if (all_replicated) return replicated();
    if (contains_replicated) {
        throw std::runtime_error(
            "distributed planner: concat cannot mix replicated and disjoint inputs");
    }
    if (all_same_root) return root_only(first.root);
    if (all_same_hash) return hash_partitioned(first.key_ordinals);

    // Local shards, differently partitioned inputs, and root-only inputs on
    // different ranks remain globally disjoint after local concatenation.
    return local_shard();
}

std::vector<std::string> join_key_names(
    const ExecPlan& plan,
    const Step& step,
    std::size_t input,
    const std::vector<size_type>& keys)
{
    const auto& schema = plan.values.at(
        static_cast<std::size_t>(step.inputs.at(input))).schema;
    std::vector<std::string> names;
    names.reserve(keys.size());
    for (const auto key : keys) {
        if (key < 0 || static_cast<std::size_t>(key) >= schema.size()) {
            throw std::runtime_error(
                "distributed planner: join key ordinal is outside input schema");
        }
        names.push_back(schema[static_cast<std::size_t>(key)].name);
    }
    return names;
}

std::string join_override_description(const JoinStrategyOverride& override) {
    std::ostringstream out;
    auto write_names = [&](const std::vector<std::string>& names) {
        out << "[";
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i != 0) out << ", ";
            out << names[i];
        }
        out << "]";
    };
    out << gamax::exec::step_kind_name(override.kind) << " left=";
    write_names(override.left_key_names);
    out << " right=";
    write_names(override.right_key_names);
    return out.str();
}

bool local_join_keys(
    const Step& step,
    const std::vector<size_type>*& left,
    const std::vector<size_type>*& right)
{
    switch (step.kind) {
        case StepKind::EqJoin: {
            const auto& payload = std::get<gamax::exec::EqJoinPayload>(
                step.payload);
            left = &payload.lkeys;
            right = &payload.rkeys;
            return true;
        }
        case StepKind::SemiJoin: {
            const auto& payload = std::get<gamax::exec::SemiJoinPayload>(
                step.payload);
            left = &payload.lkeys;
            right = &payload.rkeys;
            return true;
        }
        case StepKind::AntiJoin: {
            const auto& payload = std::get<gamax::exec::AntiJoinPayload>(
                step.payload);
            left = &payload.lkeys;
            right = &payload.rkeys;
            return true;
        }
        default:
            return false;
    }
}

using OptionalJoinStrategy = std::optional<
    gamax::exec::JoinDistributionStrategy>;

std::vector<OptionalJoinStrategy> resolve_join_strategy_overrides(
    const DistributedPlanOptions& options,
    const ExecPlan& plan)
{
    std::vector<OptionalJoinStrategy> resolved(plan.steps.size());
    for (const auto& override : options.join_strategy_overrides) {
        std::vector<std::size_t> matches;
        for (std::size_t step_index = 0;
             step_index < plan.steps.size(); ++step_index) {
            const auto& step = plan.steps[step_index];
            if (step.kind != override.kind) continue;
            const std::vector<size_type>* left_keys = nullptr;
            const std::vector<size_type>* right_keys = nullptr;
            if (!local_join_keys(step, left_keys, right_keys)) continue;
            if (join_key_names(plan, step, 0, *left_keys) ==
                    override.left_key_names &&
                join_key_names(plan, step, 1, *right_keys) ==
                    override.right_key_names) {
                matches.push_back(step_index);
            }
        }

        const auto description = join_override_description(override);
        if (matches.empty()) {
            throw std::runtime_error(
                "distributed planner: unmatched join strategy override: " +
                description);
        }
        if (matches.size() != 1 || resolved[matches.front()].has_value()) {
            throw std::runtime_error(
                "distributed planner: ambiguous join strategy override: " +
                description);
        }
        resolved[matches.front()] = override.strategy;
    }
    return resolved;
}

Distribution broadcast_join_output(
    const Distribution& retained,
    const std::vector<size_type>& retained_keys)
{
    switch (retained.kind) {
        case DistributionKind::Replicated:
            return replicated();
        case DistributionKind::RootOnly:
            return root_only(retained.root);
        case DistributionKind::HashPartitioned:
            if (retained.key_ordinals == retained_keys) {
                return hash_partitioned({0});
            }
            return local_shard();
        case DistributionKind::LocalShard:
            return local_shard();
        case DistributionKind::Unknown:
        case DistributionKind::RangePartitioned:
            break;
    }
    return unknown();
}

Distribution remap_distribution(
    const Distribution& input,
    const std::vector<size_type>& output_to_input)
{
    if (input.kind != DistributionKind::HashPartitioned &&
        input.kind != DistributionKind::RangePartitioned) {
        return input;
    }

    Distribution output = input;
    output.key_ordinals.clear();
    for (const auto input_key : input.key_ordinals) {
        const auto found = std::find(
            output_to_input.begin(), output_to_input.end(), input_key);
        if (found == output_to_input.end()) return unknown();
        output.key_ordinals.push_back(
            static_cast<size_type>(found - output_to_input.begin()));
    }
    if (output.kind == DistributionKind::RangePartitioned) {
        for (auto& order : output.ordering) {
            const auto found = std::find(
                output_to_input.begin(), output_to_input.end(), order.column);
            if (found == output_to_input.end()) return unknown();
            order.column = static_cast<size_type>(found - output_to_input.begin());
        }
    }
    return output;
}

Distribution project_distribution(
    const Distribution& input,
    const gamax::exec::ProjectPayload& payload)
{
    std::vector<size_type> output_to_input(payload.items.size(), -1);
    for (std::size_t output = 0; output < payload.items.size(); ++output) {
        if (!std::holds_alternative<gamax::expression::ExprTree>(
                payload.items[output])) {
            continue;
        }
        const auto& expression =
            std::get<gamax::expression::ExprTree>(payload.items[output]);
        if (expression.is_column_ref()) {
            output_to_input[output] = expression.get_column_index();
        }
    }
    return remap_distribution(input, output_to_input);
}

const Distribution& input_distribution(
    const ExecPlan& plan, const Step& step, std::size_t input)
{
    return plan.values.at(static_cast<std::size_t>(step.inputs.at(input))).distribution;
}

void set_output_distribution(
    ExecPlan& plan, const Step& step, Distribution distribution)
{
    plan.values.at(static_cast<std::size_t>(step.output)).distribution =
        std::move(distribution);
}

} // namespace

ExecPlan plan_for_distributed_execution(
    ExecPlan plan,
    const DistributedPlanOptions& options)
{
    const auto join_strategies = resolve_join_strategy_overrides(options, plan);
    for (auto& value : plan.values) value.distribution = unknown();

    for (std::size_t step_index = 0;
         step_index < plan.steps.size(); ++step_index) {
        auto& step = plan.steps[step_index];
        switch (step.kind) {
            case StepKind::SourceTable:
                set_output_distribution(
                    plan, step,
                    source_distribution(
                        options,
                        std::get<gamax::exec::SourceTablePayload>(
                            step.payload).table_name));
                break;
            case StepKind::Filter: {
                const auto& payload =
                    std::get<gamax::exec::FilterPayload>(step.payload);
                set_output_distribution(
                    plan, step,
                    payload.output_ordinals.empty()
                        ? input_distribution(plan, step, 0)
                        : remap_distribution(
                            input_distribution(plan, step, 0),
                            payload.output_ordinals));
                break;
            }
            case StepKind::Reorder: {
                const auto& payload =
                    std::get<gamax::exec::ReorderPayload>(step.payload);
                set_output_distribution(
                    plan, step,
                    remap_distribution(
                        input_distribution(plan, step, 0), payload.ordinals));
                break;
            }
            case StepKind::Project: {
                const auto& payload =
                    std::get<gamax::exec::ProjectPayload>(step.payload);
                set_output_distribution(
                    plan, step,
                    project_distribution(
                        input_distribution(plan, step, 0), payload));
                break;
            }
            case StepKind::ScalarAgg: {
                if (is_replicated(input_distribution(plan, step, 0))) {
                    set_output_distribution(plan, step, replicated());
                    break;
                }
                auto payload =
                    std::get<gamax::exec::ScalarAggPayload>(std::move(step.payload));
                step.kind = StepKind::DistributedScalarAgg;
                step.payload = gamax::exec::DistributedScalarAggPayload{
                    std::move(payload.specs)};
                set_output_distribution(plan, step, replicated());
                break;
            }
            case StepKind::GroupBy: {
                if (is_replicated(input_distribution(plan, step, 0))) {
                    set_output_distribution(plan, step, replicated());
                    break;
                }
                auto payload =
                    std::get<gamax::exec::GroupByPayload>(std::move(step.payload));
                const auto keys = sequence(payload.n_keys);
                step.kind = StepKind::DistributedGroupBy;
                step.payload = gamax::exec::DistributedGroupByPayload{
                    payload.n_keys, std::move(payload.specs), payload.hint};
                set_output_distribution(
                    plan, step, hash_partitioned(keys));
                break;
            }
            case StepKind::Distinct: {
                if (is_replicated(input_distribution(plan, step, 0))) {
                    set_output_distribution(plan, step, replicated());
                    break;
                }
                auto payload =
                    std::get<gamax::exec::DistinctPayload>(std::move(step.payload));
                const auto output_keys = sequence(
                    static_cast<size_type>(payload.key_ordinals.size()));
                step.kind = StepKind::DistributedDistinct;
                step.payload = gamax::exec::DistributedDistinctPayload{
                    std::move(payload.key_ordinals), payload.hint};
                set_output_distribution(
                    plan, step, hash_partitioned(output_keys));
                break;
            }
            case StepKind::TopK: {
                if (is_replicated(input_distribution(plan, step, 0))) {
                    set_output_distribution(plan, step, replicated());
                    break;
                }
                auto payload =
                    std::get<gamax::exec::TopKPayload>(std::move(step.payload));
                step.kind = StepKind::DistributedTopK;
                step.payload = gamax::exec::DistributedTopKPayload{
                    std::move(payload.keys), payload.k, payload.hint, options.root};
                set_output_distribution(plan, step, root_only(options.root));
                break;
            }
            case StepKind::Sort: {
                if (is_replicated(input_distribution(plan, step, 0))) {
                    set_output_distribution(plan, step, replicated());
                    break;
                }
                auto payload =
                    std::get<gamax::exec::SortPayload>(std::move(step.payload));
                step.kind = StepKind::DistributedSort;
                step.payload = gamax::exec::DistributedSortPayload{
                    std::move(payload.keys), payload.hint, options.root};
                set_output_distribution(plan, step, root_only(options.root));
                break;
            }
            case StepKind::EqJoin: {
                auto payload = std::get<gamax::exec::EqJoinPayload>(
                    std::move(step.payload));
                const auto left_distribution = input_distribution(plan, step, 0);
                const auto right_distribution = input_distribution(plan, step, 1);
                auto strategy = join_strategies[step_index].value_or(
                    options.default_join_strategy);
                if (is_replicated(left_distribution) &&
                    !is_replicated(right_distribution)) {
                    strategy = gamax::exec::JoinDistributionStrategy::BroadcastLeft;
                } else if (is_replicated(right_distribution)) {
                    // Also provides a deterministic local strategy when both
                    // inputs are already replicated.
                    strategy = gamax::exec::JoinDistributionStrategy::BroadcastRight;
                }
                const bool shuffle = strategy ==
                    gamax::exec::JoinDistributionStrategy::Shuffle;
                const bool partition_left = shuffle &&
                    !is_replicated(left_distribution) &&
                    !hash_partitioned_on(left_distribution, payload.lkeys);
                const bool partition_right = shuffle &&
                    !is_replicated(right_distribution) &&
                    !hash_partitioned_on(right_distribution, payload.rkeys);
                const bool replicate_left =
                    strategy == gamax::exec::JoinDistributionStrategy::BroadcastLeft &&
                    !is_replicated(left_distribution);
                const bool replicate_right =
                    strategy == gamax::exec::JoinDistributionStrategy::BroadcastRight &&
                    !is_replicated(right_distribution);
                step.kind = StepKind::DistributedEqJoin;
                step.payload = gamax::exec::DistributedEqJoinPayload{
                    std::move(payload.lkeys), std::move(payload.rkeys), strategy,
                    partition_left, partition_right,
                    replicate_left, replicate_right, payload.hint};
                if (shuffle) {
                    set_output_distribution(
                        plan, step,
                        is_replicated(left_distribution) &&
                                is_replicated(right_distribution)
                            ? replicated()
                            : hash_partitioned({0}));
                } else if (
                    strategy == gamax::exec::JoinDistributionStrategy::BroadcastRight) {
                    set_output_distribution(
                        plan, step,
                        broadcast_join_output(
                            left_distribution,
                            std::get<gamax::exec::DistributedEqJoinPayload>(
                                step.payload).lkeys));
                } else {
                    set_output_distribution(
                        plan, step,
                        broadcast_join_output(
                            right_distribution,
                            std::get<gamax::exec::DistributedEqJoinPayload>(
                                step.payload).rkeys));
                }
                break;
            }
            case StepKind::SemiJoin:
            case StepKind::AntiJoin: {
                const bool semi = step.kind == StepKind::SemiJoin;
                std::vector<size_type> left_keys;
                std::vector<size_type> right_keys;
                gamax::join::join_hint hint;
                if (semi) {
                    auto payload = std::get<gamax::exec::SemiJoinPayload>(
                        std::move(step.payload));
                    left_keys = std::move(payload.lkeys);
                    right_keys = std::move(payload.rkeys);
                    hint = payload.hint;
                } else {
                    auto payload = std::get<gamax::exec::AntiJoinPayload>(
                        std::move(step.payload));
                    left_keys = std::move(payload.lkeys);
                    right_keys = std::move(payload.rkeys);
                    hint = payload.hint;
                }
                const auto left_distribution = input_distribution(plan, step, 0);
                const auto right_distribution = input_distribution(plan, step, 1);
                auto strategy = join_strategies[step_index].value_or(
                    options.default_join_strategy);
                if (strategy ==
                    gamax::exec::JoinDistributionStrategy::BroadcastLeft) {
                    throw std::runtime_error(
                        "distributed planner: semi/anti joins cannot broadcast left");
                }
                if (is_replicated(left_distribution) ||
                    is_replicated(right_distribution)) {
                    strategy = gamax::exec::JoinDistributionStrategy::BroadcastRight;
                }
                const bool shuffle = strategy ==
                    gamax::exec::JoinDistributionStrategy::Shuffle;
                const bool partition_left = shuffle &&
                    !hash_partitioned_on(left_distribution, left_keys);
                const bool partition_right = shuffle &&
                    !hash_partitioned_on(right_distribution, right_keys);
                const bool replicate_right = !shuffle &&
                    !is_replicated(right_distribution);
                if (semi) {
                    step.kind = StepKind::DistributedSemiJoin;
                    step.payload = gamax::exec::DistributedSemiJoinPayload{
                        left_keys, right_keys, strategy,
                        partition_left, partition_right,
                        false, replicate_right, hint};
                } else {
                    step.kind = StepKind::DistributedAntiJoin;
                    step.payload = gamax::exec::DistributedAntiJoinPayload{
                        left_keys, right_keys, strategy,
                        partition_left, partition_right,
                        false, replicate_right, hint};
                }
                set_output_distribution(
                    plan, step,
                    shuffle
                        ? hash_partitioned({0})
                        : broadcast_join_output(
                            left_distribution, left_keys));
                break;
            }
            case StepKind::CrossJoin: {
                const auto left_distribution = input_distribution(plan, step, 0);
                const auto right_distribution = input_distribution(plan, step, 1);
                auto strategy = options.cross_join_strategy;
                if (is_replicated(right_distribution)) {
                    strategy = gamax::exec::JoinDistributionStrategy::BroadcastRight;
                } else if (is_replicated(left_distribution)) {
                    strategy = gamax::exec::JoinDistributionStrategy::BroadcastLeft;
                }
                if (strategy == gamax::exec::JoinDistributionStrategy::Shuffle) {
                    throw std::runtime_error(
                        "distributed planner: cross joins require broadcast");
                }
                const bool replicate_left =
                    strategy == gamax::exec::JoinDistributionStrategy::BroadcastLeft &&
                    !is_replicated(left_distribution);
                const bool replicate_right =
                    strategy == gamax::exec::JoinDistributionStrategy::BroadcastRight &&
                    !is_replicated(right_distribution);
                step.kind = StepKind::DistributedCrossJoin;
                step.payload = gamax::exec::DistributedCrossJoinPayload{
                    strategy, replicate_left, replicate_right};
                set_output_distribution(
                    plan, step,
                    strategy == gamax::exec::JoinDistributionStrategy::BroadcastRight
                        ? broadcast_join_output(left_distribution, {})
                        : broadcast_join_output(right_distribution, {}));
                break;
            }
            case StepKind::Concat:
                set_output_distribution(
                    plan, step, concat_distribution(plan, step));
                break;
            case StepKind::PartitionRows:
            case StepKind::ExchangePartitions:
            case StepKind::Replicate:
            case StepKind::GatherToRoot:
            case StepKind::DistributedScalarAgg:
            case StepKind::DistributedGroupBy:
            case StepKind::DistributedDistinct:
            case StepKind::DistributedTopK:
            case StepKind::DistributedSort:
            case StepKind::DistributedEqJoin:
            case StepKind::DistributedSemiJoin:
            case StepKind::DistributedAntiJoin:
            case StepKind::DistributedCrossJoin:
                throw std::runtime_error(
                    "distributed planner: input plan already contains distributed steps");
        }
    }

    if (options.gather_final_result && !plan.steps.empty()) {
        const auto final_value_id = plan.steps.back().output;
        const auto& final_distribution = plan.values.at(
            static_cast<std::size_t>(final_value_id)).distribution;
        if (final_distribution.kind != DistributionKind::RootOnly &&
            final_distribution.kind != DistributionKind::Replicated) {
            const auto gather_step = static_cast<size_type>(plan.steps.size());
            plan.values.at(static_cast<std::size_t>(final_value_id)).last_used_step =
                gather_step;
            const auto gathered_value = static_cast<ValueId>(plan.values.size());
            gamax::exec::Value value;
            value.id = gathered_value;
            value.schema = plan.values.at(
                static_cast<std::size_t>(final_value_id)).schema;
            value.last_used_step = -1;
            value.distribution = root_only(options.root);
            plan.values.push_back(std::move(value));
            plan.steps.push_back(Step{
                StepKind::GatherToRoot,
                {final_value_id},
                gathered_value,
                gamax::exec::GatherToRootPayload{options.root}});
        }
    }
    return plan;
}

} // namespace alps::exec
