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
#include "operators/join/inner_join.hpp"
#include "operators/join/join.hpp"
#include "operators/join/semi_join.hpp"
#include "operators/sort/sort.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "utils/copying.hpp"
#include "utils/transform.hpp"

namespace gamax::ir {

namespace {

[[noreturn]] void emit_error(const std::string& what) {
    throw std::runtime_error("ir::emit: " + what);
}

[[noreturn]] void not_implemented(const std::string& what) {
    throw std::runtime_error("ir::emit: not yet implemented: " + what);
}

// Forward declarations so the per-kind emitters can recurse into
// children via a shared memoization map. The memo holds an owned
// table_ptr per PlanNode; emit_view_of emits on first visit, then
// returns a view on every subsequent visit (which matters for the DAG
// case where a sub-plan feeds multiple parents).
using Memo =
    std::unordered_map<const PlanNode*, std::unique_ptr<gamax::table>>;
std::unique_ptr<gamax::table> emit_node(
    const PlanNode& node, Memo& memo, const EmitContext& ctx);
gamax::table_view emit_view_of(
    const PlanNode& node, Memo& memo, const EmitContext& ctx);

// ── Scan ────────────────────────────────────────────────────────────

// Deep-copy a single source column regardless of type.
// project::project's numeric-only type dispatcher can't reorder a
// table containing STRING columns, so Scan builds the subset column-
// by-column: `copy_column` for numeric, identity-gather via
// `gather_strings` for STRING.
gamax::column_ptr copy_any_column(
    gamax::column_view col, cudaStream_t stream)
{
    if (col.type().id() == gamax::type_id::STRING) {
        auto ids = gamax::make_sequence<gamax::size_type>(
            col.size(), 0, 1, stream);
        gamax::string_column_view sv(col, gamax::ARROW_STRING, stream);
        auto gathered = gamax::gather_strings(sv, ids->view(), stream);
        return std::unique_ptr<gamax::column>(std::move(gathered));
    }
    return gamax::copy_column(col, stream);
}

std::unique_ptr<gamax::table> emit_scan(const PlanNode& node, const EmitContext& ctx) {
    if (!ctx.catalog)       emit_error("scan: EmitContext.catalog is null");
    if (!ctx.resolve_table) emit_error("scan: EmitContext.resolve_table is null");

    const auto* cat_table = ctx.catalog->find(node.scan_table);
    if (!cat_table) {
        emit_error("scan: table '" + node.scan_table + "' not in catalog");
    }
    auto src = ctx.resolve_table(node.scan_table);

    // Resolve each requested column name to its ordinal in the source,
    // then copy that column into the new owned table.
    std::vector<gamax::column_ptr> cols;
    cols.reserve(node.scan_columns.size());
    for (const auto& name : node.scan_columns) {
        auto it = std::find(cat_table->column_names.begin(),
                            cat_table->column_names.end(), name);
        if (it == cat_table->column_names.end()) {
            emit_error("scan: column '" + name + "' not in table '" +
                       node.scan_table + "'");
        }
        auto ordinal = static_cast<gamax::size_type>(
            std::distance(cat_table->column_names.begin(), it));
        cols.push_back(copy_any_column(src.get_column(ordinal), ctx.stream));
    }
    return gamax::make_table(std::move(cols));
}

// ── Filter ─────────────────────────────────────────────────────────

// Materialize a filtered table from `input` by applying `mask`. Numeric
// columns compact directly; string columns go through a gather via
// compacted tids (the same pattern tpch_ops.cu uses). Kept local to
// emit.cpp because it's the only place in the IR that needs to cross
// the string/non-string boundary.
std::unique_ptr<gamax::table> apply_bitmask_to_heterogeneous_table(
    gamax::table_view input,
    gamax::column_view mask,
    cudaStream_t stream)
{
    bool any_string = false;
    for (int i = 0; i < input.num_columns(); ++i) {
        if (input.get_column(i).type().id() == gamax::type_id::STRING) {
            any_string = true;
            break;
        }
    }
    if (!any_string) {
        // Fast path: all-numeric, the table overload handles it.
        return gamax::util::device::compact(input, mask, stream);
    }

    // Mixed path: gather tids once, then reuse for every string column.
    std::vector<gamax::column_ptr> materialized;
    materialized.reserve(input.num_columns());
    gamax::column_ptr compact_tids;
    for (int i = 0; i < input.num_columns(); ++i) {
        auto col = input.get_column(i);
        if (col.type().id() == gamax::type_id::STRING) {
            if (!compact_tids) {
                auto seq = gamax::make_sequence<gamax::size_type>(
                    input.num_rows(), 0, 1, stream);
                compact_tids = gamax::util::device::compact(
                    seq->view(), mask, stream);
            }
            gamax::string_column_view sv(col, gamax::ARROW_STRING, stream);
            auto gathered = gamax::gather_strings(sv, compact_tids->view(), stream);
            // string_column inherits from column; upcast via unique_ptr
            // conversion.
            materialized.push_back(std::unique_ptr<gamax::column>(std::move(gathered)));
        } else {
            materialized.push_back(
                gamax::util::device::compact(col, mask, stream));
        }
    }
    return std::make_unique<gamax::table>(std::move(materialized));
}

std::unique_ptr<gamax::table> emit_filter(
    const PlanNode& node, gamax::table_view input, const EmitContext& ctx)
{
    // Lower the IR predicate against the INPUT schema (Filter preserves
    // column identity, so the input and output schemas are the same).
    auto& in_schema = node.inputs[0]->output_schema;
    auto pred = lower_expr_to_predicate(
        node.filter_predicate, in_schema, ctx.stream);
    auto mask = gamax::select::select_to_bitmask(input, pred, ctx.stream);
    return apply_bitmask_to_heterogeneous_table(input, mask->view(), ctx.stream);
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

// ── Join (Inner + LeftSemi, single-key equi) ───────────────────────

// Walks a join condition expecting a conjunction of column-vs-column
// equalities. Fills the per-side ColumnId lists so that left_keys[i]
// matches right_keys[i]. Throws for non-equi or non-column conditions.
struct EquiKeys {
    std::vector<ColumnId> left;
    std::vector<ColumnId> right;
};
void collect_equi_keys(
    const Expr& e,
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
            emit_error("join: equi-join operand must be a column reference");
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
            emit_error("join: equality must reference one column on each side");
        }
        return;
    }
    emit_error("join: condition must be a conjunction of column equi-predicates "
               "(no non-equi join support yet)");
}

// Reshape a table so that the ColumnId in `key_id` sits at position 0,
// followed by every other column in its original relative order.
std::unique_ptr<gamax::table> reshape_key_first(
    gamax::table_view input,
    const Schema& in_schema,
    ColumnId key_id,
    const EmitContext& ctx)
{
    auto key_ordinal = resolve_ordinal(in_schema, key_id);
    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(in_schema.size());
    exprs.push_back(gamax::expression::ExprTree::column(
        key_ordinal, in_schema[key_ordinal].type));
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(in_schema.size()); ++i)
    {
        if (i == key_ordinal) continue;
        exprs.push_back(gamax::expression::ExprTree::column(
            i, in_schema[i].type));
    }
    return gamax::project::project(input, exprs, ctx.project_opts, ctx.stream);
}

// After inner_eq_join of reshaped inputs, the physical output has shape
// [shared_key, L_rest..., R_rest...] (the right-side key is deduplicated
// — see ieq_join_test.cu's rcols+scols-1 assertion). This helper
// produces a final project that remaps back to the IR's declared
// output_schema ordering (left.all ++ right.all, with each key field
// referring to the single shared column in the physical output).
std::unique_ptr<gamax::table> reshape_inner_join_back(
    std::unique_ptr<gamax::table> physical,
    const Schema& left_schema,
    const Schema& right_schema,
    ColumnId left_key, ColumnId right_key,
    const EmitContext& ctx)
{
    auto left_key_idx = resolve_ordinal(left_schema, left_key);
    auto right_key_idx = resolve_ordinal(right_schema, right_key);

    // Build a lookup: for each position in left_schema's original
    // order, what's its position in L_rest (i.e. position in the
    // physical output after the shared key)?
    std::vector<gamax::size_type> left_rest_pos(left_schema.size());
    for (gamax::size_type i = 0, rest = 0;
         i < static_cast<gamax::size_type>(left_schema.size()); ++i)
    {
        if (i == left_key_idx) { left_rest_pos[i] = -1; continue; }
        left_rest_pos[i] = rest++;
    }
    std::vector<gamax::size_type> right_rest_pos(right_schema.size());
    for (gamax::size_type i = 0, rest = 0;
         i < static_cast<gamax::size_type>(right_schema.size()); ++i)
    {
        if (i == right_key_idx) { right_rest_pos[i] = -1; continue; }
        right_rest_pos[i] = rest++;
    }

    auto L_rest_count =
        static_cast<gamax::size_type>(left_schema.size()) - 1;

    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(left_schema.size() + right_schema.size());
    // Left-side columns in IR order.
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(left_schema.size()); ++i)
    {
        gamax::size_type ordinal =
            (i == left_key_idx) ? 0 : 1 + left_rest_pos[i];
        exprs.push_back(gamax::expression::ExprTree::column(
            ordinal, left_schema[i].type));
    }
    // Right-side columns in IR order.
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(right_schema.size()); ++i)
    {
        gamax::size_type ordinal =
            (i == right_key_idx)
                ? 0
                : 1 + L_rest_count + right_rest_pos[i];
        exprs.push_back(gamax::expression::ExprTree::column(
            ordinal, right_schema[i].type));
    }
    return gamax::project::project(
        std::move(physical), exprs, ctx.project_opts, ctx.stream);
}

// Semi-join output is just the left table (surviving rows). After
// reshaping left to keys-first, the output is [L_key, L_rest...]; we
// restore the left schema's original ordering.
std::unique_ptr<gamax::table> reshape_semi_join_back(
    std::unique_ptr<gamax::table> physical,
    const Schema& left_schema,
    ColumnId left_key,
    const EmitContext& ctx)
{
    auto left_key_idx = resolve_ordinal(left_schema, left_key);
    std::vector<gamax::size_type> left_rest_pos(left_schema.size());
    for (gamax::size_type i = 0, rest = 0;
         i < static_cast<gamax::size_type>(left_schema.size()); ++i)
    {
        if (i == left_key_idx) { left_rest_pos[i] = -1; continue; }
        left_rest_pos[i] = rest++;
    }

    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(left_schema.size());
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(left_schema.size()); ++i)
    {
        gamax::size_type ordinal =
            (i == left_key_idx) ? 0 : 1 + left_rest_pos[i];
        exprs.push_back(gamax::expression::ExprTree::column(
            ordinal, left_schema[i].type));
    }
    return gamax::project::project(
        std::move(physical), exprs, ctx.project_opts, ctx.stream);
}

gamax::join::join_hint default_join_hint(const EmitContext& ctx) {
    // inner_eq_join rejects AUTO_ALGO and AUTO_MAT, so fill in sensible
    // defaults. HASH+GFTR matches the join_test conventions and works
    // across shapes; users that want a specific algo can set this via
    // ctx.  (Kept local rather than on EmitContext — ctx is const, and
    // the physical op wants the hint by non-const ref.)
    gamax::join::join_hint h;
    if (ctx.select_opts.fuse_operations) { /* just to touch ctx */ }
    h.join_algo = gamax::join::HASH;
    h.mat_algo  = gamax::join::GFTR;
    return h;
}

std::unique_ptr<gamax::table> emit_join(
    const PlanNode& node, Memo& memo, const EmitContext& ctx)
{
    // Only Inner / LeftSemi have physical paths right now. The IR still
    // accepts LeftOuter / LeftAnti / Cross at construction time.
    if (node.join_kind != JoinKind::Inner
        && node.join_kind != JoinKind::LeftSemi)
    {
        not_implemented("join kind (only Inner / LeftSemi wired today)");
    }
    if (!node.join_condition) {
        not_implemented("join without a condition (cross join)");
    }

    const auto& left_schema  = node.inputs[0]->output_schema;
    const auto& right_schema = node.inputs[1]->output_schema;

    EquiKeys keys;
    collect_equi_keys(*node.join_condition, left_schema, right_schema, keys);
    if (keys.left.size() != 1) {
        not_implemented("multi-key equi-joins "
                        "(physical inner_eq_join is single-key)");
    }

    auto lv = emit_view_of(*node.inputs[0], memo, ctx);
    auto rv = emit_view_of(*node.inputs[1], memo, ctx);

    auto left_reshaped  = reshape_key_first(lv, left_schema,  keys.left[0],  ctx);
    auto right_reshaped = reshape_key_first(rv, right_schema, keys.right[0], ctx);

    auto hint = default_join_hint(ctx);

    std::unique_ptr<gamax::table> raw;
    if (node.join_kind == JoinKind::Inner) {
        raw = gamax::join::inner_eq_join(
            left_reshaped->view(), right_reshaped->view(), hint, ctx.stream);
        return reshape_inner_join_back(
            std::move(raw), left_schema, right_schema,
            keys.left[0], keys.right[0], ctx);
    }
    // LeftSemi
    raw = gamax::join::semi_join(
        left_reshaped->view(), right_reshaped->view(), hint, ctx.stream);
    return reshape_semi_join_back(
        std::move(raw), left_schema, keys.left[0], ctx);
}

// ── Limit (fused with a preceding Sort -> topk_table) ──────────────

std::unique_ptr<gamax::table> emit_limit(
    const PlanNode& node, Memo& memo, const EmitContext& ctx)
{
    if (node.limit_offset != 0) {
        not_implemented("LIMIT ... OFFSET (topk_table has no offset)");
    }
    // Fuse only with an immediate Sort parent; the topk_table operator
    // requires sort keys. A plain Limit without any ordering has no
    // direct physical counterpart in this backend.
    if (node.inputs[0]->kind != NodeKind::Sort) {
        not_implemented("Limit without a preceding Sort (needs ORDER BY)");
    }
    const auto& sort_node = *node.inputs[0];
    auto input_view = emit_view_of(*sort_node.inputs[0], memo, ctx);
    const auto& in_schema = sort_node.inputs[0]->output_schema;

    std::vector<int> keys;
    std::vector<gamax::sort_order> orders;
    keys.reserve(sort_node.sort_keys.size());
    orders.reserve(sort_node.sort_keys.size());
    for (const auto& k : sort_node.sort_keys) {
        keys.push_back(static_cast<int>(resolve_ordinal(in_schema, k.id)));
        orders.push_back(k.order);
    }
    return gamax::sort::topk_table(
        input_view, keys, orders, node.limit_n,
        /*result_in_sorted_order=*/true, ctx.stream);
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

gamax::table_view emit_view_of(
    const PlanNode& node, Memo& memo, const EmitContext& ctx)
{
    auto it = memo.find(&node);
    if (it == memo.end()) {
        memo[&node] = emit_node(node, memo, ctx);
        it = memo.find(&node);
    }
    return it->second->view();
}

std::unique_ptr<gamax::table> emit_node(
    const PlanNode& node, Memo& memo, const EmitContext& ctx)
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
            return emit_limit(node, memo, ctx);
        case NodeKind::Join:
            return emit_join(node, memo, ctx);
        case NodeKind::Distinct:
            not_implemented("Distinct (no direct physical op — use groupby "
                            "on the distinct keys with any aggregate)");
    }
    emit_error("unknown NodeKind");
}

} // namespace

std::unique_ptr<gamax::table> emit(const PlanPtr& root, const EmitContext& ctx) {
    if (!root) emit_error("emit: null root");
    Memo memo;
    return emit_node(*root, memo, ctx);
}

} // namespace gamax::ir
