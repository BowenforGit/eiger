#include "ir/emit.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ir/expr.hpp"
#include "ir/lower_expr.hpp"
#include "ir/schema.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/sort/sort.hpp"
#include "utils/transform.hpp"

namespace gamax::ir {

namespace {

[[noreturn]] void emit_error(const std::string& what) {
    throw std::runtime_error("ir::emit: " + what);
}

[[noreturn]] void not_implemented(const std::string& what) {
    throw std::runtime_error("ir::emit: not yet implemented: " + what);
}

// ── Scan ────────────────────────────────────────────────────────────

std::unique_ptr<gamax::table> emit_scan(const PlanNode& node, const EmitContext& ctx) {
    if (!ctx.catalog)       emit_error("scan: EmitContext.catalog is null");
    if (!ctx.resolve_table) emit_error("scan: EmitContext.resolve_table is null");

    const auto* cat_table = ctx.catalog->find(node.scan_table);
    if (!cat_table) {
        emit_error("scan: table '" + node.scan_table + "' not in catalog");
    }
    auto src = ctx.resolve_table(node.scan_table);

    // Build an ExprTree per requested column (pure ColRef), so that
    // project::project produces a table whose columns are exactly the
    // requested ones in the requested order.
    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(node.scan_columns.size());
    for (const auto& col : node.scan_columns) {
        auto it = std::find(cat_table->column_names.begin(),
                            cat_table->column_names.end(), col);
        if (it == cat_table->column_names.end()) {
            emit_error("scan: column '" + col + "' not in table '" +
                       node.scan_table + "'");
        }
        auto ordinal = static_cast<gamax::size_type>(
            std::distance(cat_table->column_names.begin(), it));
        exprs.push_back(gamax::expression::ExprTree::column(
            ordinal, cat_table->column_types[ordinal]));
    }

    return gamax::project::project(src, exprs, ctx.project_opts, ctx.stream);
}

// ── Filter ─────────────────────────────────────────────────────────

std::unique_ptr<gamax::table> emit_filter(
    const PlanNode& node, gamax::table_view input, const EmitContext& ctx)
{
    // Check for string columns — the per-column string compact path is
    // deferred until the next emitter bite (needs string_column_view +
    // gather_strings plumbing).
    for (int i = 0; i < input.num_columns(); ++i) {
        if (input.get_column(i).type().id() == gamax::type_id::STRING) {
            not_implemented("filter over a table containing a STRING column");
        }
    }

    // Lower the IR predicate against the INPUT schema (same as filter's
    // output, since Filter preserves column identity).
    auto& in_schema = node.inputs[0]->output_schema;
    auto pred = lower_expr_to_predicate(
        node.filter_predicate, in_schema, ctx.stream);
    auto mask = gamax::select::select_to_bitmask(input, pred, ctx.stream);
    return gamax::util::device::compact(input, mask->view(), ctx.stream);
}

// ── Project ────────────────────────────────────────────────────────

std::unique_ptr<gamax::table> emit_project(
    const PlanNode& node, gamax::table_view input)
{
    auto& in_schema = node.inputs[0]->output_schema;
    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(node.project_items.size());
    for (const auto& it : node.project_items) {
        exprs.push_back(lower_expr_to_tree(it.expr, in_schema));
    }
    gamax::project::project_options opts;  // default
    return gamax::project::project(input, exprs, opts);
}

// ── Aggregate ──────────────────────────────────────────────────────

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
                            "(physical backend only has SUM/MIN/MAX/AVG; "
                            "express count via SUM of a 1-literal column)");
    }
    emit_error("unknown AggKind");
}

// Emits a project that materializes the key columns followed by the
// aggregate body columns (keys first, values second) from `input`.
// Returns (table, n_keys). Aggregate bodies that aren't pure ColRef
// are computed in the project so the subsequent groupby / aggregate
// sees column-index-only agg specs.
std::pair<std::unique_ptr<gamax::table>, gamax::size_type> materialize_key_and_value_table(
    const PlanNode& node,
    gamax::table_view input,
    const Schema& in_schema,
    const EmitContext& ctx)
{
    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(node.agg_keys.size() + node.agg_items.size());
    for (auto kid : node.agg_keys) {
        auto ordinal = resolve_ordinal(in_schema, kid);
        exprs.push_back(gamax::expression::ExprTree::column(
            ordinal, in_schema[ordinal].type));
    }
    for (const auto& ag : node.agg_items) {
        if (ag.agg == AggKind::CountAll) {
            not_implemented("COUNT(*)");
        }
        exprs.push_back(lower_expr_to_tree(ag.body, in_schema));
    }
    gamax::project::project_options opts;
    auto t = gamax::project::project(input, exprs, opts, ctx.stream);
    return {std::move(t),
            static_cast<gamax::size_type>(node.agg_keys.size())};
}

std::unique_ptr<gamax::table> emit_aggregate(
    const PlanNode& node, gamax::table_view input, const EmitContext& ctx)
{
    auto& in_schema = node.inputs[0]->output_schema;
    // Validate supported AggKinds up front so we fail before touching GPU.
    for (const auto& ag : node.agg_items) {
        (void)to_physical_agg(ag.agg);
    }

    auto [mat, n_keys] =
        materialize_key_and_value_table(node, input, in_schema, ctx);

    // Build aggregation_request entries targeting the value columns of
    // the materialized table (which sit at positions [n_keys, end)).
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.reserve(node.agg_items.size());
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(node.agg_items.size()); ++i)
    {
        requests.emplace_back(
            n_keys + i, to_physical_agg(node.agg_items[i].agg));
    }

    if (n_keys == 0) {
        // Scalar aggregate — value columns only.
        return gamax::groupby::aggregate::aggregate(
            mat->view(), requests, ctx.stream);
    }

    // Grouped aggregate — split the materialized table into a key view
    // and a value view (contiguous column ranges).
    std::vector<gamax::size_type> key_indices(n_keys);
    std::iota(key_indices.begin(), key_indices.end(), gamax::size_type{0});
    std::vector<gamax::size_type> val_indices(node.agg_items.size());
    std::iota(val_indices.begin(), val_indices.end(), n_keys);
    auto key_view = mat->view().select_columns(key_indices);
    auto val_view = mat->view().select_columns(val_indices);
    // The groupby API takes non-const refs to both the requests and
    // the hint; make a local mutable copy of the hint.
    gamax::groupby::groupby_hint hint = ctx.groupby_hint;
    // Rebase request indices onto the value-only view (0-based again).
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(requests.size()); ++i)
    {
        requests[i] = {i, requests[i].agg_kind};
    }
    return gamax::groupby::groupby(
        key_view, val_view, requests, hint, ctx.stream);
}

// ── Sort ───────────────────────────────────────────────────────────

std::unique_ptr<gamax::table> emit_sort(
    const PlanNode& node, gamax::table_view input, const EmitContext& ctx)
{
    auto& in_schema = node.inputs[0]->output_schema;
    std::vector<int> keys;
    std::vector<gamax::sort_order> orders;
    keys.reserve(node.sort_keys.size());
    orders.reserve(node.sort_keys.size());
    for (const auto& k : node.sort_keys) {
        keys.push_back(static_cast<int>(resolve_ordinal(in_schema, k.id)));
        orders.push_back(k.order);
    }
    return gamax::sort::sort_table(input, keys, orders, ctx.stream);
}

// ── Dispatcher ─────────────────────────────────────────────────────

std::unique_ptr<gamax::table> emit_node(
    const PlanNode& node,
    std::unordered_map<const PlanNode*, std::unique_ptr<gamax::table>>& memo,
    const EmitContext& ctx);

gamax::table_view emit_view_of(
    const PlanNode& node,
    std::unordered_map<const PlanNode*, std::unique_ptr<gamax::table>>& memo,
    const EmitContext& ctx)
{
    auto it = memo.find(&node);
    if (it == memo.end()) {
        memo[&node] = emit_node(node, memo, ctx);
        it = memo.find(&node);
    }
    return it->second->view();
}

std::unique_ptr<gamax::table> emit_node(
    const PlanNode& node,
    std::unordered_map<const PlanNode*, std::unique_ptr<gamax::table>>& memo,
    const EmitContext& ctx)
{
    switch (node.kind) {
        case NodeKind::Scan:
            return emit_scan(node, ctx);
        case NodeKind::Filter:
            return emit_filter(node, emit_view_of(*node.inputs[0], memo, ctx), ctx);
        case NodeKind::Project:
            return emit_project(node, emit_view_of(*node.inputs[0], memo, ctx));
        case NodeKind::Aggregate:
            return emit_aggregate(node, emit_view_of(*node.inputs[0], memo, ctx), ctx);
        case NodeKind::Sort:
            return emit_sort(node, emit_view_of(*node.inputs[0], memo, ctx), ctx);
        case NodeKind::Limit:
            not_implemented("Limit (pending join/limit bite)");
        case NodeKind::Join:
            not_implemented("Join (pending join bite)");
        case NodeKind::Distinct:
            not_implemented("Distinct (no direct physical op yet)");
    }
    emit_error("unknown NodeKind");
}

} // namespace

std::unique_ptr<gamax::table> emit(const PlanPtr& root, const EmitContext& ctx) {
    if (!root) emit_error("emit: null root");
    std::unordered_map<const PlanNode*, std::unique_ptr<gamax::table>> memo;
    return emit_node(*root, memo, ctx);
}

} // namespace gamax::ir
