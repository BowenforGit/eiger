#include "ir/lower_to_exec_plan.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ir/expr.hpp"
#include "ir/lower_expr.hpp"
#include "ir/schema.hpp"
#include "operators/groupby/aggregation.hpp"

namespace gamax::ir {

namespace {

using exec::ExecPlan;
using exec::Step;
using exec::StepKind;
using exec::Value;
using exec::ValueId;

[[noreturn]] void lower_error(const std::string& what) {
    throw std::runtime_error("ir::lower_to_exec_plan: " + what);
}

[[noreturn]] void not_implemented(const std::string& what) {
    throw std::runtime_error(
        "ir::lower_to_exec_plan: not yet implemented: " + what);
}

// Walk state threaded through the recursive lowering. Owns the
// ExecPlan under construction, the Catalog reference, a memo for
// DAG-shared plan nodes, and the monotonic ValueId allocator.
struct Lowering {
    ExecPlan plan;
    const Catalog* cat;
    std::unordered_map<const PlanNode*, ValueId> memo;

    ValueId next_value(Schema schema) {
        ValueId id = static_cast<ValueId>(plan.values.size());
        plan.values.push_back(Value{id, std::move(schema), -1});
        return id;
    }

    gamax::size_type next_step_index() const {
        return static_cast<gamax::size_type>(plan.steps.size());
    }

    // Record that step index `step_i` consumes value `id`. Updates
    // last_used_step to the max of the current value and step_i.
    void record_use(ValueId id, gamax::size_type step_i) {
        auto& v = plan.values[id];
        if (v.last_used_step < step_i) v.last_used_step = step_i;
    }

    // Append a step; return its index. The caller has already
    // allocated `output` via `next_value` and recorded uses for
    // the inputs.
    gamax::size_type push_step(StepKind kind, std::vector<ValueId> inputs,
                               ValueId output, exec::StepPayload payload)
    {
        for (auto in : inputs) record_use(in, next_step_index());
        gamax::size_type idx = next_step_index();
        plan.steps.push_back(
            Step{kind, std::move(inputs), output, std::move(payload)});
        return idx;
    }
};

// ── Per-node lowering ──────────────────────────────────────────────

ValueId lower_node(const PlanNode& node, Lowering& lo);

ValueId lower_scan(const PlanNode& node, Lowering& lo) {
    // Scan → SourceTable (producing a value that aliases the caller's
    // table_view) + Reorder (to pick and re-order the requested
    // columns). If the requested set already matches the source in
    // order, the Reorder is a trivial identity copy — keep it for
    // uniformity.
    const auto* cat_table = lo.cat->find(node.scan_table);
    if (!cat_table) lower_error(
        "scan: table '" + node.scan_table + "' not in catalog");

    // `required_sources` is a dedup'd list for the executor to supply.
    auto& req = lo.plan.required_sources;
    if (std::find(req.begin(), req.end(), node.scan_table) == req.end()) {
        req.push_back(node.scan_table);
    }

    // The SourceTable value's schema is the catalog-declared schema
    // (positional, matching what the source table_view must have).
    gamax::ir::Schema src_schema;
    for (size_t i = 0; i < cat_table->column_names.size(); ++i) {
        src_schema.append({
            ColumnId{},  // retired at exec layer
            cat_table->column_types[i],
            cat_table->column_nullable[i],
            cat_table->column_names[i],
        });
    }
    ValueId src_id = lo.next_value(std::move(src_schema));
    lo.push_step(StepKind::SourceTable, /*inputs=*/{}, src_id,
                 exec::SourceTablePayload{node.scan_table});

    // Then a Reorder to produce the requested subset in order.
    std::vector<gamax::size_type> ordinals;
    ordinals.reserve(node.scan_columns.size());
    for (const auto& name : node.scan_columns) {
        auto it = std::find(cat_table->column_names.begin(),
                            cat_table->column_names.end(), name);
        if (it == cat_table->column_names.end()) {
            lower_error("scan: column '" + name + "' not in table '" +
                        node.scan_table + "'");
        }
        ordinals.push_back(static_cast<gamax::size_type>(
            std::distance(cat_table->column_names.begin(), it)));
    }

    // Schema of the reordered value: straight copy of the IR node's
    // declared output_schema (fresh ColumnIds there; we don't need
    // them at this layer).
    gamax::ir::Schema out_schema = node.output_schema;
    ValueId out_id = lo.next_value(std::move(out_schema));
    lo.push_step(StepKind::Reorder, /*inputs=*/{src_id}, out_id,
                 exec::ReorderPayload{std::move(ordinals)});
    return out_id;
}

ValueId lower_filter(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = node.inputs[0]->output_schema;

    // Build the Predicate now, once, using the default stream. String
    // scalars allocated here live inside the Predicate and persist
    // for the plan's lifetime.
    auto pred = lower_expr_to_predicate(
        node.filter_predicate, in_schema, cudaStreamDefault);

    ValueId out_id = lo.next_value(node.output_schema);
    lo.push_step(StepKind::Filter, {input_id}, out_id,
                 exec::FilterPayload{std::move(pred)});
    return out_id;
}

ValueId lower_project(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = node.inputs[0]->output_schema;

    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(node.project_items.size());
    for (const auto& it : node.project_items) {
        exprs.push_back(lower_expr_to_tree(it.expr, in_schema));
    }

    ValueId out_id = lo.next_value(node.output_schema);
    lo.push_step(StepKind::Project, {input_id}, out_id,
                 exec::ProjectPayload{std::move(exprs)});
    return out_id;
}

// Helper: map ir::AggKind → phys::AggSpec's aggregation_id, matching
// the translation emit.cpp already does today.
gamax::groupby::aggregation_id to_physical_agg(AggKind k) {
    switch (k) {
        case AggKind::Sum: return gamax::groupby::aggregation_id::SUM;
        case AggKind::Min: return gamax::groupby::aggregation_id::MIN;
        case AggKind::Max: return gamax::groupby::aggregation_id::MAX;
        case AggKind::Avg: return gamax::groupby::aggregation_id::AVG;
        case AggKind::Count:
        case AggKind::CountAll:
        case AggKind::CountDistinct:
            not_implemented("COUNT / COUNT(*) / COUNT(DISTINCT) "
                            "(physical backend has no count; express "
                            "via SUM of a 1-literal column — rule-engine "
                            "work)");
    }
    lower_error("unknown AggKind");
}

// Materialize `[keys..., agg-body-columns...]` as an intermediate
// value that `phys::group_by` / `phys::scalar_aggregate` can consume
// via column-index. Matches the shape the current emit.cpp builds
// inline; hoisted into the ExecPlan so it's visible and reusable.
//
// Returns `(value_id, n_keys)`.
std::pair<ValueId, gamax::size_type> materialize_keys_and_values(
    const PlanNode& node, ValueId input_id,
    const Schema& in_schema, Lowering& lo)
{
    // Decide between a pure-ColRef path (Reorder) and a mixed path
    // (Project) based on whether any agg body is computed.
    bool all_col_refs = true;
    for (const auto& ag : node.agg_items) {
        if (ag.agg == AggKind::CountAll) continue;
        if (ag.body.kind != ExprKind::ColRef) { all_col_refs = false; break; }
    }

    // Build the materialized value's output schema: keys first, then
    // one field per agg body (types pulled from the exprs).
    Schema mat_schema;
    for (auto kid : node.agg_keys) {
        auto ordinal = resolve_ordinal(in_schema, kid);
        mat_schema.append(in_schema[ordinal]);
    }
    for (const auto& ag : node.agg_items) {
        if (ag.agg == AggKind::CountAll) {
            not_implemented("COUNT(*)");
        }
        Field f;
        f.id = ColumnId{};
        f.type = ag.body.type;
        f.nullable = ag.body.nullable;
        f.name = ag.name;
        mat_schema.append(std::move(f));
    }

    ValueId mat_id = lo.next_value(mat_schema);
    const auto n_keys =
        static_cast<gamax::size_type>(node.agg_keys.size());

    if (all_col_refs) {
        // Pure-ColRef materialization → Reorder.
        std::vector<gamax::size_type> ordinals;
        ordinals.reserve(node.agg_keys.size() + node.agg_items.size());
        for (auto kid : node.agg_keys) {
            ordinals.push_back(resolve_ordinal(in_schema, kid));
        }
        for (const auto& ag : node.agg_items) {
            ordinals.push_back(resolve_ordinal(in_schema, ag.body.col_ref));
        }
        lo.push_step(StepKind::Reorder, {input_id}, mat_id,
                     exec::ReorderPayload{std::move(ordinals)});
    } else {
        // Mixed / computed → Project.
        std::vector<gamax::expression::ExprTree> exprs;
        exprs.reserve(node.agg_keys.size() + node.agg_items.size());
        for (auto kid : node.agg_keys) {
            auto ordinal = resolve_ordinal(in_schema, kid);
            exprs.push_back(gamax::expression::ExprTree::column(
                ordinal, in_schema[ordinal].type));
        }
        for (const auto& ag : node.agg_items) {
            exprs.push_back(lower_expr_to_tree(ag.body, in_schema));
        }
        lo.push_step(StepKind::Project, {input_id}, mat_id,
                     exec::ProjectPayload{std::move(exprs)});
    }
    return {mat_id, n_keys};
}

ValueId lower_aggregate(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = node.inputs[0]->output_schema;

    auto [mat_id, n_keys] =
        materialize_keys_and_values(node, input_id, in_schema, lo);

    // Build AggSpecs. Specs' column ordinals reference the
    // materialized value's columns: value-only indices run 0..N-1
    // when passed to phys::group_by (keys/values are split at the
    // Executor); they run n_keys..end-1 for phys::scalar_aggregate
    // (the whole table is the "values" view).
    std::vector<gamax::phys::AggSpec> specs;
    specs.reserve(node.agg_items.size());
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(node.agg_items.size()); ++i)
    {
        specs.push_back({
            n_keys + i,  // scalar-aggregate view index
            to_physical_agg(node.agg_items[i].agg),
        });
    }

    ValueId out_id = lo.next_value(node.output_schema);
    if (n_keys == 0) {
        lo.push_step(StepKind::ScalarAgg, {mat_id}, out_id,
                     exec::ScalarAggPayload{std::move(specs)});
    } else {
        // Rebase specs onto the value-only view (0-indexed).
        for (gamax::size_type i = 0;
             i < static_cast<gamax::size_type>(specs.size()); ++i)
        {
            specs[i].column = i;
        }
        lo.push_step(StepKind::GroupBy, {mat_id}, out_id,
                     exec::GroupByPayload{n_keys, std::move(specs)});
    }
    return out_id;
}

ValueId lower_sort(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = node.inputs[0]->output_schema;

    std::vector<gamax::phys::SortSpec> keys;
    keys.reserve(node.sort_keys.size());
    for (const auto& k : node.sort_keys) {
        keys.push_back({resolve_ordinal(in_schema, k.id), k.order});
    }
    ValueId out_id = lo.next_value(node.output_schema);
    lo.push_step(StepKind::Sort, {input_id}, out_id,
                 exec::SortPayload{std::move(keys)});
    return out_id;
}

ValueId lower_limit(const PlanNode& node, Lowering& lo) {
    if (node.limit_offset != 0) {
        not_implemented("LIMIT ... OFFSET");
    }
    if (node.inputs[0]->kind != NodeKind::Sort) {
        not_implemented("Limit without a preceding Sort");
    }
    // Fuse Sort + Limit into a single TopK step. We skip lowering the
    // Sort node itself and emit a TopK keyed on the Sort's keys over
    // the Sort's INPUT.
    const auto& sort_node = *node.inputs[0];
    auto input_id = lower_node(*sort_node.inputs[0], lo);
    const auto& in_schema = sort_node.inputs[0]->output_schema;

    std::vector<gamax::phys::SortSpec> keys;
    keys.reserve(sort_node.sort_keys.size());
    for (const auto& k : sort_node.sort_keys) {
        keys.push_back({resolve_ordinal(in_schema, k.id), k.order});
    }
    ValueId out_id = lo.next_value(node.output_schema);
    lo.push_step(StepKind::TopK, {input_id}, out_id,
                 exec::TopKPayload{std::move(keys), node.limit_n});
    return out_id;
}

// Mirrors the same walk `emit.cpp::collect_equi_keys` does, but
// output-free — the caller reads from `out` after return.
struct EquiKeys { std::vector<ColumnId> left, right; };
void collect_equi_keys(const Expr& e,
                       const Schema& left, const Schema& right,
                       EquiKeys& out)
{
    if (e.kind == ExprKind::BinLogical && e.logical_op == LogicalOp::And) {
        collect_equi_keys(e.children[0], left, right, out);
        collect_equi_keys(e.children[1], left, right, out);
        return;
    }
    if (e.kind == ExprKind::BinCompare && e.compare_op == CompareOp::Eq) {
        const auto& a = e.children[0];
        const auto& b = e.children[1];
        if (a.kind != ExprKind::ColRef || b.kind != ExprKind::ColRef) {
            lower_error("join: equi-join operand must be a column reference");
        }
        bool a_in_l = left.index_of(a.col_ref).has_value();
        bool b_in_r = right.index_of(b.col_ref).has_value();
        bool a_in_r = right.index_of(a.col_ref).has_value();
        bool b_in_l = left.index_of(b.col_ref).has_value();
        if (a_in_l && b_in_r) {
            out.left.push_back(a.col_ref);
            out.right.push_back(b.col_ref);
        } else if (b_in_l && a_in_r) {
            out.left.push_back(b.col_ref);
            out.right.push_back(a.col_ref);
        } else {
            lower_error("join: equality must reference one column on each side");
        }
        return;
    }
    lower_error("join: condition must be a conjunction of column "
                "equi-predicates");
}

// After phys::eq_join, the physical output is
// `[shared_key, L_non_keys..., R_non_keys...]` in the order each side's
// non-key columns appeared. The IR's declared output_schema is
// `[L.all_in_order, R.all_in_order]` with L's key and R's key fields
// both referring to the single shared column. This helper computes a
// reorder-ordinal list that maps from the physical layout back to the
// IR-declared order.
std::vector<gamax::size_type> compute_inner_join_reshape_ordinals(
    const Schema& L, const Schema& R, ColumnId lkey, ColumnId rkey)
{
    auto L_key_idx = L.require_index(lkey);
    auto R_key_idx = R.require_index(rkey);
    auto l_rest_start = gamax::size_type{1};
    auto r_rest_start =
        l_rest_start + (static_cast<gamax::size_type>(L.size()) - 1);

    auto offset_among_non_keys = [](gamax::size_type key_idx,
                                    gamax::size_type i) {
        return i < key_idx ? i : i - 1;
    };
    std::vector<gamax::size_type> out;
    out.reserve(L.size() + R.size());
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(L.size()); ++i)
    {
        out.push_back(i == L_key_idx
            ? 0
            : l_rest_start + offset_among_non_keys(L_key_idx, i));
    }
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(R.size()); ++i)
    {
        out.push_back(i == R_key_idx
            ? 0
            : r_rest_start + offset_among_non_keys(R_key_idx, i));
    }
    return out;
}

std::vector<gamax::size_type> compute_semi_join_reshape_ordinals(
    const Schema& L, ColumnId lkey)
{
    auto L_key_idx = L.require_index(lkey);
    std::vector<gamax::size_type> out;
    out.reserve(L.size());
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(L.size()); ++i)
    {
        out.push_back(i == L_key_idx
            ? 0
            : 1 + (i < L_key_idx ? i : i - 1));
    }
    return out;
}

ValueId lower_join(const PlanNode& node, Lowering& lo) {
    if (node.join_kind == JoinKind::Cross) {
        // Cross join: lower children, then emit a CrossJoin step.
        auto l_id = lower_node(*node.inputs[0], lo);
        auto r_id = lower_node(*node.inputs[1], lo);
        ValueId out_id = lo.next_value(node.output_schema);
        lo.push_step(StepKind::CrossJoin, {l_id, r_id}, out_id,
                     exec::CrossJoinPayload{});
        return out_id;
    }
    if (node.join_kind != JoinKind::Inner
        && node.join_kind != JoinKind::LeftSemi)
    {
        not_implemented("join kind (only Inner / LeftSemi / Cross wired today)");
    }
    if (!node.join_condition) {
        lower_error("join: missing condition");
    }

    const auto& left_schema  = node.inputs[0]->output_schema;
    const auto& right_schema = node.inputs[1]->output_schema;

    EquiKeys keys;
    collect_equi_keys(*node.join_condition, left_schema, right_schema, keys);
    if (keys.left.size() != 1) {
        not_implemented("multi-key equi-joins");
    }

    auto l_id = lower_node(*node.inputs[0], lo);
    auto r_id = lower_node(*node.inputs[1], lo);
    auto l_key_ord = resolve_ordinal(left_schema, keys.left[0]);
    auto r_key_ord = resolve_ordinal(right_schema, keys.right[0]);

    // Intermediate value: raw join output in physical layout
    // `[shared_key, L_rest..., R_rest...]` (for Inner) or
    // `[shared_key, L_rest...]` (for Semi). Schema doesn't matter for
    // execution (positional) but keeping something useful for the
    // printer.
    Schema raw_schema;
    raw_schema.append(Field{ColumnId{}, left_schema[l_key_ord].type,
                            left_schema[l_key_ord].nullable,
                            left_schema[l_key_ord].name});
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(left_schema.size()); ++i)
    {
        if (i == l_key_ord) continue;
        raw_schema.append(left_schema[i]);
    }
    if (node.join_kind == JoinKind::Inner) {
        for (gamax::size_type i = 0;
             i < static_cast<gamax::size_type>(right_schema.size()); ++i)
        {
            if (i == r_key_ord) continue;
            raw_schema.append(right_schema[i]);
        }
    }

    ValueId raw_id = lo.next_value(std::move(raw_schema));
    if (node.join_kind == JoinKind::Inner) {
        lo.push_step(StepKind::EqJoin, {l_id, r_id}, raw_id,
                     exec::EqJoinPayload{{l_key_ord}, {r_key_ord}});
    } else {
        lo.push_step(StepKind::SemiJoin, {l_id, r_id}, raw_id,
                     exec::SemiJoinPayload{{l_key_ord}, {r_key_ord}});
    }

    // Reshape raw → IR-declared order.
    std::vector<gamax::size_type> ordinals =
        (node.join_kind == JoinKind::Inner)
            ? compute_inner_join_reshape_ordinals(
                  left_schema, right_schema, keys.left[0], keys.right[0])
            : compute_semi_join_reshape_ordinals(left_schema, keys.left[0]);

    ValueId out_id = lo.next_value(node.output_schema);
    lo.push_step(StepKind::Reorder, {raw_id}, out_id,
                 exec::ReorderPayload{std::move(ordinals)});
    return out_id;
}

ValueId lower_node(const PlanNode& node, Lowering& lo) {
    // Memoize DAG shares so that a subtree referenced by two parents
    // gets emitted exactly once.
    auto it = lo.memo.find(&node);
    if (it != lo.memo.end()) return it->second;

    ValueId id = exec::kUnsetValueId;
    switch (node.kind) {
        case NodeKind::Scan:      id = lower_scan(node, lo);      break;
        case NodeKind::Filter:    id = lower_filter(node, lo);    break;
        case NodeKind::Project:   id = lower_project(node, lo);   break;
        case NodeKind::Aggregate: id = lower_aggregate(node, lo); break;
        case NodeKind::Sort:      id = lower_sort(node, lo);      break;
        case NodeKind::Limit:     id = lower_limit(node, lo);     break;
        case NodeKind::Join:      id = lower_join(node, lo);      break;
        case NodeKind::Distinct:
            not_implemented("Distinct (no physical op yet)");
    }
    lo.memo[&node] = id;
    return id;
}

} // namespace

exec::ExecPlan lower_to_exec_plan(const PlanPtr& root, const Catalog& cat) {
    if (!root) lower_error("null root");
    Lowering lo;
    lo.cat = &cat;
    (void)lower_node(*root, lo);
    return std::move(lo.plan);
}

} // namespace gamax::ir
