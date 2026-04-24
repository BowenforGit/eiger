#include "ir/plan_node.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace gamax::ir {

namespace {

[[noreturn]] void plan_error(const std::string& what) {
    throw std::runtime_error("ir::PlanNode: " + what);
}

void require_refs_in_schema(const Expr& e, const Schema& input) {
    auto refs = collect_column_refs(e);
    for (auto id : refs) {
        if (!input.index_of(id)) {
            plan_error("expression references ColumnId " +
                       std::to_string(id.value) + " not in input schema");
        }
    }
}

gamax::data_type aggregate_output_type(AggKind agg, gamax::data_type in) {
    using gamax::type_id;
    switch (agg) {
        case AggKind::Sum:
            if (is_integer(in.id())) return gamax::data_type{type_id::INT64};
            if (in.id() == type_id::FLOAT32 || in.id() == type_id::FLOAT64) {
                return gamax::data_type{type_id::FLOAT64};
            }
            plan_error("SUM requires numeric input");
        case AggKind::Avg:
            if (!is_numeric(in.id())) plan_error("AVG requires numeric input");
            return gamax::data_type{type_id::FLOAT64};
        case AggKind::Min:
        case AggKind::Max:
            return in;
        case AggKind::Count:
        case AggKind::CountAll:
        case AggKind::CountDistinct:
            return gamax::data_type{type_id::INT64};
    }
    plan_error("unknown AggKind");
}

// Union schema used to validate a join condition: refs may come from
// either side. Ids are globally unique so concatenation cannot collide.
Schema combined_schema(const Schema& l, const Schema& r) {
    Schema s = l;
    for (const auto& f : r.fields()) s.append(f);
    return s;
}

Schema join_output_schema(JoinKind kind, const Schema& l, const Schema& r) {
    switch (kind) {
        case JoinKind::Inner:
        case JoinKind::Cross:
            return combined_schema(l, r);
        case JoinKind::LeftOuter: {
            Schema s = l;
            for (auto f : r.fields()) {
                // Right side becomes NULL-extended on non-match.
                f.nullable = true;
                s.append(std::move(f));
            }
            return s;
        }
        case JoinKind::LeftSemi:
        case JoinKind::LeftAnti:
            return l;
    }
    plan_error("unknown JoinKind");
}

PlanPtr build_join(
    JoinKind kind,
    PlanPtr left,
    PlanPtr right,
    std::optional<Expr> on)
{
    if (!left)  plan_error("join: null left input");
    if (!right) plan_error("join: null right input");

    if (kind == JoinKind::Cross) {
        if (on) plan_error("cross_join: condition must be absent");
    } else {
        if (!on) plan_error("join: condition is required for this JoinKind");
        if (on->type.id() != gamax::type_id::BOOL8) {
            plan_error("join: condition must be BOOL8");
        }
        // Refs may reference either side.
        auto combined = combined_schema(left->output_schema,
                                        right->output_schema);
        require_refs_in_schema(*on, combined);
    }

    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Join;
    node->inputs.push_back(left);
    node->inputs.push_back(right);
    node->join_kind = kind;
    node->join_condition = std::move(on);
    node->output_schema = join_output_schema(
        kind, left->output_schema, right->output_schema);
    return node;
}

} // namespace

PlanPtr scan(
    const Catalog& catalog,
    std::string table_name,
    std::vector<std::string> columns,
    IdGenerator& gen)
{
    const auto* table = catalog.find(table_name);
    if (!table) plan_error("scan: table '" + table_name + "' not in catalog");
    if (columns.empty()) plan_error("scan: must request at least one column");

    Schema out;
    for (const auto& col : columns) {
        auto it = std::find(table->column_names.begin(),
                            table->column_names.end(), col);
        if (it == table->column_names.end()) {
            plan_error("scan: column '" + col + "' not in table '" +
                       table_name + "'");
        }
        auto idx = static_cast<size_t>(
            std::distance(table->column_names.begin(), it));
        Field f;
        f.id = gen.next();
        f.type = table->column_types[idx];
        f.nullable = table->column_nullable[idx];
        f.name = col;
        out.append(std::move(f));
    }

    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Scan;
    node->scan_table = std::move(table_name);
    node->scan_columns = std::move(columns);
    node->output_schema = std::move(out);
    return node;
}

PlanPtr filter(PlanPtr input, Expr predicate) {
    if (!input) plan_error("filter: null input");
    if (predicate.type.id() != gamax::type_id::BOOL8) {
        plan_error("filter: predicate must be BOOL8");
    }
    require_refs_in_schema(predicate, input->output_schema);

    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Filter;
    node->inputs.push_back(input);
    node->output_schema = input->output_schema;
    node->filter_predicate = std::move(predicate);
    return node;
}

PlanPtr project(
    PlanPtr input,
    std::vector<ProjectItem> items,
    IdGenerator& gen)
{
    if (!input) plan_error("project: null input");
    if (items.empty()) plan_error("project: must project at least one column");

    Schema out;
    for (const auto& item : items) {
        require_refs_in_schema(item.expr, input->output_schema);
        Field f;
        // Pure ColRef projections preserve the input's ColumnId so
        // that rename/reorder-style projects do not renumber anything.
        // Computed expressions allocate a fresh id.
        if (item.expr.kind == ExprKind::ColRef) {
            f.id = item.expr.col_ref;
        } else {
            f.id = gen.next();
        }
        f.type = item.expr.type;
        f.nullable = item.expr.nullable;
        f.name = item.name;
        out.append(std::move(f));
    }

    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Project;
    node->inputs.push_back(input);
    node->output_schema = std::move(out);
    node->project_items = std::move(items);
    return node;
}

PlanPtr aggregate(
    PlanPtr input,
    std::vector<ColumnId> keys,
    std::vector<AggregateItem> aggs,
    IdGenerator& gen)
{
    if (!input) plan_error("aggregate: null input");
    if (aggs.empty()) {
        plan_error("aggregate: must have at least one aggregate "
                   "(use distinct() for key-only deduplication)");
    }

    const auto& in = input->output_schema;

    Schema out;
    for (auto kid : keys) {
        auto idx = in.index_of(kid);
        if (!idx) {
            plan_error("aggregate: key ColumnId " +
                       std::to_string(kid.value) + " not in input schema");
        }
        out.append(in[*idx]);  // preserves id, type, nullable, name
    }

    for (const auto& ag : aggs) {
        if (ag.agg != AggKind::CountAll) {
            require_refs_in_schema(ag.body, in);
        }
        Field f;
        f.id = gen.next();
        f.name = ag.name;
        if (ag.agg == AggKind::CountAll
            || ag.agg == AggKind::Count
            || ag.agg == AggKind::CountDistinct)
        {
            f.type = gamax::data_type{gamax::type_id::INT64};
            f.nullable = false;  // COUNT* are never NULL
        } else {
            f.type = aggregate_output_type(ag.agg, ag.body.type);
            // SUM / AVG over an empty group yields NULL; MIN / MAX
            // inherit the body's nullability.
            f.nullable = (ag.agg == AggKind::Sum || ag.agg == AggKind::Avg)
                         || ag.body.nullable;
        }
        out.append(std::move(f));
    }

    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Aggregate;
    node->inputs.push_back(input);
    node->agg_keys = std::move(keys);
    node->agg_items = std::move(aggs);
    node->output_schema = std::move(out);
    return node;
}

PlanPtr sort(PlanPtr input, std::vector<SortKey> keys) {
    if (!input) plan_error("sort: null input");
    if (keys.empty()) plan_error("sort: must have at least one sort key");
    for (const auto& k : keys) {
        if (!input->output_schema.index_of(k.id)) {
            plan_error("sort: key ColumnId " + std::to_string(k.id.value) +
                       " not in input schema");
        }
    }

    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Sort;
    node->inputs.push_back(input);
    node->output_schema = input->output_schema;
    node->sort_keys = std::move(keys);
    return node;
}

PlanPtr limit(PlanPtr input, gamax::size_type n, gamax::size_type offset) {
    if (!input) plan_error("limit: null input");
    if (n < 0)      plan_error("limit: n must be non-negative");
    if (offset < 0) plan_error("limit: offset must be non-negative");

    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Limit;
    node->inputs.push_back(input);
    node->output_schema = input->output_schema;
    node->limit_n      = n;
    node->limit_offset = offset;
    return node;
}

PlanPtr inner_join(PlanPtr left, PlanPtr right, Expr on) {
    return build_join(JoinKind::Inner, std::move(left), std::move(right),
                      std::move(on));
}
PlanPtr left_outer_join(PlanPtr left, PlanPtr right, Expr on) {
    return build_join(JoinKind::LeftOuter, std::move(left), std::move(right),
                      std::move(on));
}
PlanPtr left_semi_join(PlanPtr left, PlanPtr right, Expr on) {
    return build_join(JoinKind::LeftSemi, std::move(left), std::move(right),
                      std::move(on));
}
PlanPtr left_anti_join(PlanPtr left, PlanPtr right, Expr on) {
    return build_join(JoinKind::LeftAnti, std::move(left), std::move(right),
                      std::move(on));
}
PlanPtr cross_join(PlanPtr left, PlanPtr right) {
    return build_join(JoinKind::Cross, std::move(left), std::move(right),
                      std::nullopt);
}

PlanPtr distinct(PlanPtr input, std::vector<ColumnId> keys) {
    if (!input) plan_error("distinct: null input");
    for (auto k : keys) {
        if (!input->output_schema.index_of(k)) {
            plan_error("distinct: key ColumnId " + std::to_string(k.value) +
                       " not in input schema");
        }
    }

    // Output-schema policy:
    //   * Empty keys → "distinct over all columns" (SQL DISTINCT).
    //     Output schema is the input's schema unchanged.
    //   * Non-empty keys that cover *every* input column (by id) → the
    //     degenerate case of above; same result, output == input schema.
    //   * Non-empty keys that are a proper subset → output schema is
    //     just those key columns, in caller order. Matches the
    //     `SELECT DISTINCT k1, k2 FROM t` shape — non-key columns are
    //     dropped rather than getting an arbitrary pick per group.
    //
    // SQL `DISTINCT ON (subset) keep all cols` is NOT expressible under
    // this single-factory scheme; if a caller needs it, a separate
    // entry point will be added rather than parameterising this one.
    auto node = std::make_shared<PlanNode>();
    node->kind = NodeKind::Distinct;
    node->inputs.push_back(input);

    bool covers_all = keys.empty();
    if (!covers_all &&
        keys.size() == static_cast<size_t>(input->output_schema.size()))
    {
        std::unordered_set<int64_t> key_ids;
        for (auto k : keys) key_ids.insert(k.value);
        covers_all = true;
        for (const auto& f : input->output_schema.fields()) {
            if (!key_ids.count(f.id.value)) { covers_all = false; break; }
        }
    }

    if (covers_all) {
        node->output_schema = input->output_schema;
    } else {
        Schema narrow;
        for (auto k : keys) {
            auto idx = input->output_schema.index_of(k);
            narrow.append(input->output_schema[*idx]);
        }
        node->output_schema = std::move(narrow);
    }
    node->distinct_keys = std::move(keys);
    return node;
}

} // namespace gamax::ir
