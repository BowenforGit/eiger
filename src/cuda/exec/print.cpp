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
        default: return "?";
    }
}

const char* order_name(gamax::sort_order o) {
    return o == gamax::sort_order::ASCENDING ? "ASC" : "DESC";
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
               << ", substr=" << n_substr << ")";
        } else if constexpr (std::is_same_v<T, FilterPayload>) {
            os << "predicate=<Predicate>";
            if (!pl.output_ordinals.empty()) {
                os << ", out=";
                write_ordinals(os, pl.output_ordinals);
            }
        } else if constexpr (std::is_same_v<T, EqJoinPayload>) {
            os << "lkeys="; write_ordinals(os, pl.lkeys);
            os << ", rkeys="; write_ordinals(os, pl.rkeys);
        } else if constexpr (std::is_same_v<T, SemiJoinPayload>) {
            os << "lkeys="; write_ordinals(os, pl.lkeys);
            os << ", rkeys="; write_ordinals(os, pl.rkeys);
        } else if constexpr (std::is_same_v<T, AntiJoinPayload>) {
            os << "lkeys="; write_ordinals(os, pl.lkeys);
            os << ", rkeys="; write_ordinals(os, pl.rkeys);
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
        } else if constexpr (std::is_same_v<T, DistinctPayload>) {
            os << "keys="; write_ordinals(os, pl.key_ordinals);
        } else if constexpr (std::is_same_v<T, ConcatPayload>) {
            // N-ary inputs already on Step::inputs; no extra payload.
        } else if constexpr (std::is_same_v<T, SortPayload>) {
            os << "keys=[";
            for (size_t i = 0; i < pl.keys.size(); ++i) {
                if (i) os << ", ";
                os << "#" << pl.keys[i].column << " "
                   << order_name(pl.keys[i].order);
            }
            os << "]";
        } else if constexpr (std::is_same_v<T, TopKPayload>) {
            os << "k=" << pl.k << ", keys=[";
            for (size_t i = 0; i < pl.keys.size(); ++i) {
                if (i) os << ", ";
                os << "#" << pl.keys[i].column << " "
                   << order_name(pl.keys[i].order);
            }
            os << "]";
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
        os << ")\n";
    }
    if (!plan.steps.empty()) {
        os << "return v" << plan.steps.back().output << "\n";
    }
}

} // namespace gamax::exec
