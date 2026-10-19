// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include "exec/print.hpp"

#include <sstream>
#include <variant>

namespace gamax::exec {

const char* step_kind_name(StepKind k) {
    switch (k) {
        case StepKind::SourceTable: return "SourceTable";
        case StepKind::Reorder:     return "Reorder";
        case StepKind::Project:     return "Project";
        case StepKind::Filter:      return "Filter";
        case StepKind::EqJoin:      return "EqJoin";
        case StepKind::SemiJoin:    return "SemiJoin";
        case StepKind::AntiJoin:    return "AntiJoin";
        case StepKind::CrossJoin:   return "CrossJoin";
        case StepKind::ScalarAgg:   return "ScalarAgg";
        case StepKind::GroupBy:     return "GroupBy";
        case StepKind::Distinct:    return "Distinct";
        case StepKind::Concat:      return "Concat";
        case StepKind::Sort:        return "Sort";
        case StepKind::TopK:        return "TopK";
        case StepKind::PartitionRows: return "PartitionRows";
        case StepKind::ExchangePartitions: return "ExchangePartitions";
        case StepKind::Replicate: return "Replicate";
        case StepKind::GatherToRoot: return "GatherToRoot";
        case StepKind::DistributedScalarAgg: return "DistributedScalarAgg";
        case StepKind::DistributedGroupBy: return "DistributedGroupBy";
        case StepKind::DistributedDistinct: return "DistributedDistinct";
        case StepKind::DistributedTopK: return "DistributedTopK";
        case StepKind::DistributedSort: return "DistributedSort";
        case StepKind::DistributedEqJoin: return "DistributedEqJoin";
        case StepKind::DistributedSemiJoin: return "DistributedSemiJoin";
        case StepKind::DistributedAntiJoin: return "DistributedAntiJoin";
        case StepKind::DistributedCrossJoin: return "DistributedCrossJoin";
    }
    return "?";
}

namespace {

const char* agg_name(gamax::groupby::aggregation_id a) {
    switch (a) {
        case gamax::groupby::aggregation_id::SUM: return "sum";
        case gamax::groupby::aggregation_id::MIN: return "min";
        case gamax::groupby::aggregation_id::MAX: return "max";
        case gamax::groupby::aggregation_id::AVG: return "avg";
        case gamax::groupby::aggregation_id::ANY: return "any";
        case gamax::groupby::aggregation_id::COUNT_ALL: return "count_all";
        default: return "?";
    }
}

const char* order_name(gamax::sort_order o) {
    return o == gamax::sort_order::ASCENDING ? "ASC" : "DESC";
}

const char* join_distribution_name(JoinDistributionStrategy strategy) {
    switch (strategy) {
        case JoinDistributionStrategy::Shuffle: return "shuffle";
        case JoinDistributionStrategy::BroadcastLeft: return "broadcast-left";
        case JoinDistributionStrategy::BroadcastRight: return "broadcast-right";
    }
    return "unknown";
}

const char* distribution_name(DistributionKind kind) {
    switch (kind) {
        case DistributionKind::Unknown: return "unknown";
        case DistributionKind::LocalShard: return "local-shard";
        case DistributionKind::HashPartitioned: return "hash-partitioned";
        case DistributionKind::RangePartitioned: return "range-partitioned";
        case DistributionKind::Replicated: return "replicated";
        case DistributionKind::RootOnly: return "root-only";
    }
    return "unknown";
}

const char* join_algo_name(gamax::join::join_algo value) {
    switch (value) {
        case gamax::join::AUTO_ALGO: return "auto";
        case gamax::join::HASH: return "hash";
        case gamax::join::SORT: return "sort";
        case gamax::join::PARTITION: return "partition";
        case gamax::join::UNSUPPORTED_JOIN_ALGO: return "unsupported";
    }
    return "unknown";
}

const char* join_materialization_name(
    gamax::join::materialization_algo value)
{
    switch (value) {
        case gamax::join::AUTO_MAT: return "auto";
        case gamax::join::GFUR: return "gfur";
        case gamax::join::GFTR: return "gftr";
        case gamax::join::INDEX_CLUSTER: return "index-cluster";
        case gamax::join::UNSUPPORTED_MATERIALIZATION_ALGO:
            return "unsupported";
    }
    return "unknown";
}

const char* groupby_algo_name(gamax::groupby::groupby_algo value) {
    switch (value) {
        case gamax::groupby::AUTO_ALGO: return "auto";
        case gamax::groupby::HASH: return "hash";
        case gamax::groupby::SORT: return "sort";
        case gamax::groupby::SORT_OPT: return "sort-opt";
        case gamax::groupby::SORT_GFUR: return "sort-gfur";
        case gamax::groupby::PARTITION: return "partition-gftr";
        case gamax::groupby::PARTITION_GFUR: return "partition-gfur";
        case gamax::groupby::UNSUPPORTED_GROUPBY_ALGO: return "unsupported";
    }
    return "unknown";
}

void write_join_hint(
    std::ostream& os,
    const gamax::join::join_hint& hint)
{
    os << ", hint=" << join_algo_name(hint.join_algo)
       << "+" << join_materialization_name(hint.mat_algo);
}

void write_groupby_hint(
    std::ostream& os,
    const gamax::groupby::groupby_hint& hint)
{
    os << ", hint=" << groupby_algo_name(hint.groupby_algo)
       << ", skf=" << hint.enable_skf
       << ", hash_fallback=" << hint.allow_hash_fallback;
    if (hint.groups_upper_bound != gamax::MAXIMUM_SIZE) {
        os << ", groups_upper_bound=" << hint.groups_upper_bound;
    }
}

void write_ordinals(std::ostream& os,
                    const std::vector<gamax::size_type>& xs);

void write_distribution(std::ostream& os, const Distribution& distribution) {
    os << distribution_name(distribution.kind);
    if (!distribution.key_ordinals.empty()) {
        os << " keys=";
        write_ordinals(os, distribution.key_ordinals);
    }
    if (distribution.kind == DistributionKind::RootOnly) {
        os << " root=" << distribution.root;
    }
}

void write_ordinals(std::ostream& os,
                    const std::vector<gamax::size_type>& xs) {
    os << "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ", ";
        os << xs[i];
    }
    os << "]";
}

void write_payload(std::ostream& os, const StepPayload& payload) {
    std::visit([&](auto&& pl) {
        using T = std::decay_t<decltype(pl)>;
        if constexpr (std::is_same_v<T, SourceTablePayload>) {
            os << "\"" << pl.table_name << "\" cols=[";
            for (size_t i = 0; i < pl.columns.size(); ++i) {
                if (i) os << ", ";
                os << pl.columns[i];
            }
            os << "]";
        } else if constexpr (std::is_same_v<T, ReorderPayload>) {
            os << "ordinals=";
            write_ordinals(os, pl.ordinals);
        } else if constexpr (std::is_same_v<T, ProjectPayload>) {
            gamax::size_type n_pred = 0, n_tree = 0, n_substr = 0;
            for (const auto& it : pl.items) {
                if (std::holds_alternative<gamax::select::Predicate>(it)) {
                    ++n_pred;
                } else if (std::holds_alternative<gamax::phys::SubstringItem>(it)) {
                    ++n_substr;
                } else {
                    ++n_tree;
                }
            }
            os << "#items=" << pl.items.size()
               << " (tree=" << n_tree << ", pred=" << n_pred
               << ", substr=" << n_substr << ")"
               << ", hint=" << (pl.hint.fuse_operations ? "fuse" : "nofuse");
        } else if constexpr (std::is_same_v<T, FilterPayload>) {
            os << "predicate=<Predicate>";
            if (!pl.output_ordinals.empty()) {
                os << ", out=";
                write_ordinals(os, pl.output_ordinals);
            }
            os << ", hint=" << (pl.hint.fuse_operations ? "fuse" : "nofuse");
        } else if constexpr (std::is_same_v<T, EqJoinPayload>) {
            os << "lkeys="; write_ordinals(os, pl.lkeys);
            os << ", rkeys="; write_ordinals(os, pl.rkeys);
            write_join_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, SemiJoinPayload>) {
            os << "lkeys="; write_ordinals(os, pl.lkeys);
            os << ", rkeys="; write_ordinals(os, pl.rkeys);
            write_join_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, AntiJoinPayload>) {
            os << "lkeys="; write_ordinals(os, pl.lkeys);
            os << ", rkeys="; write_ordinals(os, pl.rkeys);
            write_join_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, CrossJoinPayload>) {
            // no args
        } else if constexpr (std::is_same_v<T, ScalarAggPayload>) {
            os << "aggs=[";
            for (size_t i = 0; i < pl.specs.size(); ++i) {
                if (i) os << ", ";
                os << agg_name(pl.specs[i].agg) << "(#"
                   << pl.specs[i].column << ")";
            }
            os << "]";
        } else if constexpr (std::is_same_v<T, GroupByPayload>) {
            os << "n_keys=" << pl.n_keys << ", aggs=[";
            for (size_t i = 0; i < pl.specs.size(); ++i) {
                if (i) os << ", ";
                os << agg_name(pl.specs[i].agg) << "(#"
                   << pl.specs[i].column << ")";
            }
            os << "]";
            write_groupby_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, DistinctPayload>) {
            os << "keys="; write_ordinals(os, pl.key_ordinals);
            write_groupby_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, ConcatPayload>) {
            // N-ary inputs already on Step::inputs; no extra payload.
        } else if constexpr (std::is_same_v<T, SortPayload>) {
            os << "keys=[";
            for (size_t i = 0; i < pl.keys.size(); ++i) {
                if (i) os << ", ";
                os << "#" << pl.keys[i].column << " "
                   << order_name(pl.keys[i].order);
            }
            os << "], hint="
               << (pl.hint.smart_keys_fusion ? "skf" : "no-skf");
        } else if constexpr (std::is_same_v<T, TopKPayload>) {
            os << "k=" << pl.k << ", keys=[";
            for (size_t i = 0; i < pl.keys.size(); ++i) {
                if (i) os << ", ";
                os << "#" << pl.keys[i].column << " "
                   << order_name(pl.keys[i].order);
            }
            os << "], hint="
               << (pl.hint.smart_keys_fusion ? "skf" : "no-skf");
        } else if constexpr (std::is_same_v<T, PartitionRowsPayload>) {
            os << "strategy="
               << (pl.strategy == PartitionStrategy::Hash ? "hash" : "range");
            if (!pl.key_ordinals.empty()) {
                os << ", keys=";
                write_ordinals(os, pl.key_ordinals);
            }
            if (!pl.ordering.empty()) {
                os << ", ordering=[";
                for (size_t i = 0; i < pl.ordering.size(); ++i) {
                    if (i) os << ", ";
                    os << "#" << pl.ordering[i].column << " "
                       << order_name(pl.ordering[i].order);
                }
                os << "]";
            }
        } else if constexpr (std::is_same_v<T, ExchangePartitionsPayload>) {
            // Distribution is carried by the output Value.
        } else if constexpr (std::is_same_v<T, ReplicatePayload>) {
            // no args
        } else if constexpr (std::is_same_v<T, GatherToRootPayload>) {
            os << "root=" << pl.root;
        } else if constexpr (std::is_same_v<T, DistributedScalarAggPayload>) {
            os << "aggs=[";
            for (size_t i = 0; i < pl.specs.size(); ++i) {
                if (i) os << ", ";
                os << agg_name(pl.specs[i].agg) << "(#"
                   << pl.specs[i].column << ")";
            }
            os << "]";
        } else if constexpr (std::is_same_v<T, DistributedGroupByPayload>) {
            os << "n_keys=" << pl.n_keys << ", aggs=[";
            for (size_t i = 0; i < pl.specs.size(); ++i) {
                if (i) os << ", ";
                os << agg_name(pl.specs[i].agg) << "(#"
                   << pl.specs[i].column << ")";
            }
            os << "]";
            write_groupby_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, DistributedDistinctPayload>) {
            os << "keys=";
            write_ordinals(os, pl.key_ordinals);
            write_groupby_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, DistributedTopKPayload>) {
            os << "k=" << pl.k << ", root=" << pl.root << ", keys=[";
            for (size_t i = 0; i < pl.keys.size(); ++i) {
                if (i) os << ", ";
                os << "#" << pl.keys[i].column << " "
                   << order_name(pl.keys[i].order);
            }
            os << "], hint="
               << (pl.hint.smart_keys_fusion ? "skf" : "no-skf");
        } else if constexpr (std::is_same_v<T, DistributedSortPayload>) {
            os << "root=" << pl.root << ", keys=[";
            for (size_t i = 0; i < pl.keys.size(); ++i) {
                if (i) os << ", ";
                os << "#" << pl.keys[i].column << " "
                   << order_name(pl.keys[i].order);
            }
            os << "], hint="
               << (pl.hint.smart_keys_fusion ? "skf" : "no-skf");
        } else if constexpr (
            std::is_same_v<T, DistributedEqJoinPayload> ||
            std::is_same_v<T, DistributedSemiJoinPayload> ||
            std::is_same_v<T, DistributedAntiJoinPayload>) {
            os << "strategy=" << join_distribution_name(pl.strategy)
               << ", lkeys=";
            write_ordinals(os, pl.lkeys);
            os << ", rkeys=";
            write_ordinals(os, pl.rkeys);
            os << ", partition=(" << pl.partition_left << ", "
               << pl.partition_right << ")"
               << ", replicate=(" << pl.replicate_left << ", "
               << pl.replicate_right << ")";
            write_join_hint(os, pl.hint);
        } else if constexpr (std::is_same_v<T, DistributedCrossJoinPayload>) {
            os << "strategy=" << join_distribution_name(pl.strategy)
               << ", replicate=(" << pl.replicate_left << ", "
               << pl.replicate_right << ")";
        }
    }, payload);
}

} // namespace

std::string format_exec_plan(const ExecPlan& plan) {
    std::ostringstream os;
    print_exec_plan(os, plan);
    return os.str();
}

void print_exec_plan(std::ostream& os, const ExecPlan& plan) {
    if (!plan.required_sources.empty()) {
        os << "// required sources:";
        for (const auto& s : plan.required_sources) os << " " << s;
        os << "\n";
    }
    for (const auto& s : plan.steps) {
        os << "v" << s.output << " = " << step_kind_name(s.kind) << "(";
        for (size_t i = 0; i < s.inputs.size(); ++i) {
            if (i) os << ", ";
            os << "v" << s.inputs[i];
        }
        if (!s.inputs.empty()) os << "; ";
        write_payload(os, s.payload);
        os << ")";
        const auto& distribution = plan.values.at(
            static_cast<std::size_t>(s.output)).distribution;
        if (distribution.kind != DistributionKind::Unknown) {
            os << "  // distribution=";
            write_distribution(os, distribution);
        }
        os << "\n";
    }
    if (!plan.steps.empty()) {
        os << "return v" << plan.steps.back().output << "\n";
    }
}

std::string format_step_label(
    const ExecPlan& plan,
    gamax::size_type step_index)
{
    if (step_index < 0 ||
        static_cast<std::size_t>(step_index) >= plan.steps.size()) {
        return "InvalidStep";
    }
    const auto& step = plan.steps[static_cast<std::size_t>(step_index)];
    std::ostringstream os;
    os << step_kind_name(step.kind);

    auto write_name = [&](ValueId value_id, gamax::size_type ordinal) {
        const auto& schema = plan.values.at(
            static_cast<std::size_t>(value_id)).schema;
        if (ordinal >= 0 &&
            static_cast<std::size_t>(ordinal) < schema.size()) {
            os << schema[static_cast<std::size_t>(ordinal)].name;
        } else {
            os << "#" << ordinal;
        }
    };
    auto write_names = [&](ValueId value_id,
                           const std::vector<gamax::size_type>& ordinals) {
        os << "[";
        for (std::size_t i = 0; i < ordinals.size(); ++i) {
            if (i != 0) os << ",";
            write_name(value_id, ordinals[i]);
        }
        os << "]";
    };
    auto write_output_names = [&]() {
        const auto& schema = plan.values.at(
            static_cast<std::size_t>(step.output)).schema;
        os << "[";
        for (std::size_t i = 0; i < schema.size(); ++i) {
            if (i != 0) os << ",";
            os << schema[i].name;
        }
        os << "]";
    };
    auto write_sort_keys = [&](ValueId input,
                               const std::vector<gamax::phys::SortSpec>& keys) {
        os << "[";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (i != 0) os << ",";
            write_name(input, keys[i].column);
            os << " " << order_name(keys[i].order);
        }
        os << "]";
    };
    auto write_aggs = [&](ValueId input,
                          const auto& specs,
                          gamax::size_type column_offset = 0) {
        os << "[";
        for (std::size_t i = 0; i < specs.size(); ++i) {
            if (i != 0) os << ",";
            os << agg_name(specs[i].agg) << "(";
            if (specs[i].column < 0) {
                os << "*";
            } else {
                write_name(input, column_offset + specs[i].column);
            }
            os << ")";
        }
        os << "]";
    };
    auto write_join = [&](const auto& payload) {
        using Payload = std::decay_t<decltype(payload)>;
        os << "(";
        write_names(step.inputs.at(0), payload.lkeys);
        os << "=";
        write_names(step.inputs.at(1), payload.rkeys);
        if constexpr (
            std::is_same_v<Payload, DistributedEqJoinPayload> ||
            std::is_same_v<Payload, DistributedSemiJoinPayload> ||
            std::is_same_v<Payload, DistributedAntiJoinPayload>) {
            os << ";" << join_distribution_name(payload.strategy);
            if (payload.partition_left || payload.partition_right) {
                os << ";partition="
                   << (payload.partition_left ? "L" : "")
                   << (payload.partition_right ? "R" : "");
            }
            if (payload.replicate_left || payload.replicate_right) {
                os << ";replicate="
                   << (payload.replicate_left ? "L" : "")
                   << (payload.replicate_right ? "R" : "");
            }
        }
        os << ")";
    };

    switch (step.kind) {
        case StepKind::SourceTable: {
            const auto& payload = std::get<SourceTablePayload>(step.payload);
            os << "(" << payload.table_name << ")";
            break;
        }
        case StepKind::Filter:
            os << "(out=";
            write_output_names();
            os << ")";
            break;
        case StepKind::Project:
        case StepKind::Reorder:
            os << "(out=";
            write_output_names();
            os << ")";
            break;
        case StepKind::EqJoin:
            write_join(std::get<EqJoinPayload>(step.payload));
            break;
        case StepKind::SemiJoin:
            write_join(std::get<SemiJoinPayload>(step.payload));
            break;
        case StepKind::AntiJoin:
            write_join(std::get<AntiJoinPayload>(step.payload));
            break;
        case StepKind::DistributedEqJoin:
            write_join(std::get<DistributedEqJoinPayload>(step.payload));
            break;
        case StepKind::DistributedSemiJoin:
            write_join(std::get<DistributedSemiJoinPayload>(step.payload));
            break;
        case StepKind::DistributedAntiJoin:
            write_join(std::get<DistributedAntiJoinPayload>(step.payload));
            break;
        case StepKind::ScalarAgg: {
            const auto& payload = std::get<ScalarAggPayload>(step.payload);
            os << "(aggs=";
            write_aggs(step.inputs.at(0), payload.specs);
            os << ")";
            break;
        }
        case StepKind::DistributedScalarAgg: {
            const auto& payload =
                std::get<DistributedScalarAggPayload>(step.payload);
            os << "(aggs=";
            write_aggs(step.inputs.at(0), payload.specs);
            os << ")";
            break;
        }
        case StepKind::GroupBy: {
            const auto& payload = std::get<GroupByPayload>(step.payload);
            os << "(keys=";
            std::vector<gamax::size_type> keys(
                static_cast<std::size_t>(payload.n_keys));
            for (gamax::size_type i = 0; i < payload.n_keys; ++i) keys[i] = i;
            write_names(step.inputs.at(0), keys);
            os << ";aggs=";
            write_aggs(step.inputs.at(0), payload.specs, payload.n_keys);
            os << ")";
            break;
        }
        case StepKind::DistributedGroupBy: {
            const auto& payload =
                std::get<DistributedGroupByPayload>(step.payload);
            os << "(keys=";
            std::vector<gamax::size_type> keys(
                static_cast<std::size_t>(payload.n_keys));
            for (gamax::size_type i = 0; i < payload.n_keys; ++i) keys[i] = i;
            write_names(step.inputs.at(0), keys);
            os << ";aggs=";
            write_aggs(step.inputs.at(0), payload.specs, payload.n_keys);
            os << ")";
            break;
        }
        case StepKind::Distinct:
            os << "(keys=";
            write_names(
                step.inputs.at(0),
                std::get<DistinctPayload>(step.payload).key_ordinals);
            os << ")";
            break;
        case StepKind::DistributedDistinct:
            os << "(keys=";
            write_names(
                step.inputs.at(0),
                std::get<DistributedDistinctPayload>(step.payload).key_ordinals);
            os << ")";
            break;
        case StepKind::Sort:
            os << "(keys=";
            write_sort_keys(
                step.inputs.at(0), std::get<SortPayload>(step.payload).keys);
            os << ")";
            break;
        case StepKind::TopK:
            os << "(keys=";
            write_sort_keys(
                step.inputs.at(0), std::get<TopKPayload>(step.payload).keys);
            os << ";k=" << std::get<TopKPayload>(step.payload).k << ")";
            break;
        case StepKind::DistributedSort:
            os << "(keys=";
            write_sort_keys(
                step.inputs.at(0),
                std::get<DistributedSortPayload>(step.payload).keys);
            os << ";root="
               << std::get<DistributedSortPayload>(step.payload).root << ")";
            break;
        case StepKind::DistributedTopK:
            os << "(keys=";
            write_sort_keys(
                step.inputs.at(0),
                std::get<DistributedTopKPayload>(step.payload).keys);
            os << ";k=" << std::get<DistributedTopKPayload>(step.payload).k
               << ";root="
               << std::get<DistributedTopKPayload>(step.payload).root << ")";
            break;
        case StepKind::PartitionRows: {
            const auto& payload =
                std::get<PartitionRowsPayload>(step.payload);
            os << "(strategy="
               << (payload.strategy == PartitionStrategy::Hash
                       ? "hash" : "range");
            if (!payload.key_ordinals.empty()) {
                os << ";keys=";
                write_names(step.inputs.at(0), payload.key_ordinals);
            }
            if (!payload.ordering.empty()) {
                os << ";ordering=";
                write_sort_keys(step.inputs.at(0), payload.ordering);
            }
            os << ")";
            break;
        }
        case StepKind::ExchangePartitions: {
            const auto& distribution = plan.values.at(
                static_cast<std::size_t>(step.output)).distribution;
            os << "(distribution=" << distribution_name(distribution.kind);
            if (!distribution.key_ordinals.empty()) {
                os << ";keys=";
                write_names(
                    step.inputs.at(0), distribution.key_ordinals);
            }
            os << ")";
            break;
        }
        case StepKind::Replicate:
            os << "(to=all-ranks)";
            break;
        case StepKind::GatherToRoot:
            os << "(root="
               << std::get<GatherToRootPayload>(step.payload).root << ")";
            break;
        case StepKind::DistributedCrossJoin: {
            const auto& payload =
                std::get<DistributedCrossJoinPayload>(step.payload);
            os << "(" << join_distribution_name(payload.strategy) << ")";
            break;
        }
        default:
            break;
    }

    auto label = os.str();
    constexpr std::size_t max_label_size = 240;
    if (label.size() > max_label_size) {
        label.resize(max_label_size - 3);
        label += "...";
    }
    return label;
}

} // namespace gamax::exec
