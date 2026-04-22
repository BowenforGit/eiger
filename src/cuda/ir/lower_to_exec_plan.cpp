#include "ir/lower_to_exec_plan.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ir/expr.hpp"
#include "ir/liveness.hpp"
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
// DAG-shared plan nodes, the liveness analysis, and the monotonic
// ValueId allocator.
//
// Invariant: each Value's `schema` reflects the EXEC-time column
// layout (possibly pruned vs. IR's declared `output_schema`). Every
// lowering helper that resolves a ColumnId to an ordinal reads from
// `exec_schema(child_id)`, not from the IR child's `output_schema`,
// so upstream pruning propagates correctly through the DAG.
struct Lowering {
    ExecPlan plan;
    const Catalog* cat;
    LowerOptions opts;
    LivenessResult liveness;
    std::unordered_map<const PlanNode*, ValueId> memo;

    ValueId next_value(Schema schema) {
        ValueId id = static_cast<ValueId>(plan.values.size());
        plan.values.push_back(Value{id, std::move(schema), -1});
        return id;
    }

    const Schema& exec_schema(ValueId id) const {
        return plan.values[id].schema;
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

// Build an EXEC output schema by keeping only those fields of `declared`
// whose ColumnId is live (present in `live_union`). Used by Join: the
// IR-declared output schema is L ++ R, but if either side was pruned
// upstream, some of those fields are no longer produced.
Schema filter_schema_by_live(
    const Schema& declared,
    const std::unordered_set<ColumnId>& live_union)
{
    Schema out;
    for (const auto& f : declared.fields()) {
        if (live_union.count(f.id)) out.append(f);
    }
    return out;
}

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
    const auto& in_schema = lo.exec_schema(input_id);

    // Build the Predicate now, once, using the default stream. String
    // scalars allocated here live inside the Predicate and persist
    // for the plan's lifetime. Note: ordinals are resolved against
    // the EXEC schema — columns referenced by the predicate are
    // guaranteed to be live at the child's output (liveness forces
    // it), so they're present.
    auto pred = lower_expr_to_predicate(
        node.filter_predicate, in_schema, cudaStreamDefault);

    // Column-pruning rule: drop columns the predicate touches but no
    // downstream op consumes. Liveness tells us which ids survive past
    // this Filter; everything else can be discarded during
    // materialization rather than compact/gathered and then thrown
    // away by a follow-up Reorder. Gated by `prune_filter_materialization`
    // so benchmarks / debugging can disable the optimization.
    std::vector<gamax::size_type> output_ordinals;
    Schema pruned_schema;
    bool actually_pruned = false;
    if (lo.opts.prune_filter_materialization) {
        const auto& live = lo.liveness.at(&node);
        output_ordinals.reserve(in_schema.size());
        for (gamax::size_type i = 0;
             i < static_cast<gamax::size_type>(in_schema.size()); ++i)
        {
            const auto& f = in_schema[i];
            if (live.count(f.id)) {
                output_ordinals.push_back(i);
                pruned_schema.append(f);
            }
        }
        actually_pruned = (output_ordinals.size() != in_schema.size());
    }

    // If nothing was dropped, leave output_ordinals empty so the
    // Executor takes the cheap "all-columns" filter path (no change
    // vs. pre-rule behavior).
    Schema out_schema = actually_pruned
        ? std::move(pruned_schema)
        : in_schema;  // pass-through = input's EXEC schema
    if (!actually_pruned) output_ordinals.clear();

    ValueId out_id = lo.next_value(std::move(out_schema));
    lo.push_step(StepKind::Filter, {input_id}, out_id,
                 exec::FilterPayload{
                     std::move(pred), std::move(output_ordinals)});
    return out_id;
}

ValueId lower_project(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = lo.exec_schema(input_id);

    std::vector<gamax::expression::ExprTree> exprs;
    exprs.reserve(node.project_items.size());
    for (const auto& it : node.project_items) {
        exprs.push_back(lower_expr_to_tree(it.expr, in_schema));
    }

    // Project rebuilds its output from scratch (one column per item);
    // it's independent of any upstream pruning, so the EXEC schema
    // equals the IR's declared output_schema.
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
// Returns `(value_id, n_keys)`. `in_schema` must be the EXEC schema of
// the aggregate's input (i.e., the actual column layout of `input_id`).
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
    const auto& in_schema = lo.exec_schema(input_id);

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
    const auto& in_schema = lo.exec_schema(input_id);

    std::vector<gamax::phys::SortSpec> keys;
    keys.reserve(node.sort_keys.size());
    for (const auto& k : node.sort_keys) {
        keys.push_back({resolve_ordinal(in_schema, k.id), k.order});
    }
    // Sort preserves its input layout exactly; EXEC schema passes
    // through (so upstream pruning flows downstream unchanged).
    ValueId out_id = lo.next_value(in_schema);
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
    const auto& in_schema = lo.exec_schema(input_id);

    std::vector<gamax::phys::SortSpec> keys;
    keys.reserve(sort_node.sort_keys.size());
    for (const auto& k : sort_node.sort_keys) {
        keys.push_back({resolve_ordinal(in_schema, k.id), k.order});
    }
    // TopK preserves input layout like Sort — EXEC schema passes
    // through.
    ValueId out_id = lo.next_value(in_schema);
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

// Raw phys::eq_join output is `[shared_key, L_non_keys..., R_non_keys...]`
// over L and R's EXEC schemas (post any upstream column pruning).
// Compute the raw-output ordinal for:
//   * an L-side column at EXEC index `i` (given the key's EXEC index).
//   * an R-side column, using `l_non_key_count` as the R-rest starting
//     offset (the key slot occupies raw ord 0, L's non-keys run from
//     1 upward, then R's non-keys).
gamax::size_type raw_pos_of_L_idx(
    gamax::size_type i, gamax::size_type l_key_idx)
{
    if (i == l_key_idx) return 0;
    gamax::size_type offset = (i < l_key_idx) ? i : i - 1;
    return 1 + offset;
}
gamax::size_type raw_pos_of_R_idx(
    gamax::size_type i, gamax::size_type r_key_idx,
    gamax::size_type l_non_key_count)
{
    if (i == r_key_idx) return 0;
    gamax::size_type offset = (i < r_key_idx) ? i : i - 1;
    return 1 + l_non_key_count + offset;
}

// Build the raw-join EXEC schema: [shared_key_field, L_non_keys_in_L_order,
// R_non_keys_in_R_order]. Uses EXEC L and EXEC R so that upstream
// column pruning (e.g. a filter that dropped payload columns) flows
// naturally through the join.
Schema build_raw_inner_join_schema(
    const Schema& L, const Schema& R, ColumnId lkey, ColumnId rkey)
{
    auto l_key_idx = L.require_index(lkey);
    auto r_key_idx = R.require_index(rkey);
    Schema out;
    out.append(Field{
        ColumnId{}, L[l_key_idx].type, L[l_key_idx].nullable,
        L[l_key_idx].name});
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(L.size()); ++i)
    {
        if (i == l_key_idx) continue;
        out.append(L[i]);
    }
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(R.size()); ++i)
    {
        if (i == r_key_idx) continue;
        out.append(R[i]);
    }
    return out;
}

Schema build_raw_semi_join_schema(const Schema& L, ColumnId lkey) {
    auto l_key_idx = L.require_index(lkey);
    Schema out;
    out.append(Field{
        ColumnId{}, L[l_key_idx].type, L[l_key_idx].nullable,
        L[l_key_idx].name});
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(L.size()); ++i)
    {
        if (i == l_key_idx) continue;
        out.append(L[i]);
    }
    return out;
}

// After phys::eq_join, raw output is `[shared_key, L_non_keys..., R_non_keys...]`.
// The IR declares the join's output as L.output_schema ++ R.output_schema,
// but upstream pruning may have dropped some fields from the EXEC L / R.
// This helper walks IR's declared order, skipping any field whose
// ColumnId is no longer in the EXEC side (pruned upstream), and emits
// raw-ordinal indices in IR order. The result feeds a `phys::reorder`
// that materializes the join output in the IR-declared-and-still-live
// layout.
std::pair<Schema, std::vector<gamax::size_type>>
compute_inner_join_reshape(
    const Schema& declared,
    const Schema& exec_L, const Schema& exec_R,
    ColumnId lkey, ColumnId rkey)
{
    auto l_key_idx = exec_L.require_index(lkey);
    auto r_key_idx = exec_R.require_index(rkey);
    auto l_non_key_count =
        static_cast<gamax::size_type>(exec_L.size()) - 1;

    Schema out_schema;
    std::vector<gamax::size_type> ordinals;
    for (const auto& f : declared.fields()) {
        // `lkey` and `rkey` may or may not equal any declared field's
        // ColumnId — IR assigns fresh ids per Scan, so typically they
        // don't coincide. Detect membership via exec_L / exec_R.
        if (auto li = exec_L.index_of(f.id); li.has_value()) {
            ordinals.push_back(raw_pos_of_L_idx(*li, l_key_idx));
            out_schema.append(f);
        } else if (auto ri = exec_R.index_of(f.id); ri.has_value()) {
            ordinals.push_back(
                raw_pos_of_R_idx(*ri, r_key_idx, l_non_key_count));
            out_schema.append(f);
        }
        // else: pruned upstream — skip.
    }
    return {std::move(out_schema), std::move(ordinals)};
}

std::pair<Schema, std::vector<gamax::size_type>>
compute_semi_join_reshape(
    const Schema& declared, const Schema& exec_L, ColumnId lkey)
{
    auto l_key_idx = exec_L.require_index(lkey);
    Schema out_schema;
    std::vector<gamax::size_type> ordinals;
    for (const auto& f : declared.fields()) {
        if (auto li = exec_L.index_of(f.id); li.has_value()) {
            ordinals.push_back(raw_pos_of_L_idx(*li, l_key_idx));
            out_schema.append(f);
        }
        // Semi-join's declared output is L only, so everything not in
        // exec_L was pruned upstream.
    }
    return {std::move(out_schema), std::move(ordinals)};
}

ValueId lower_join(const PlanNode& node, Lowering& lo) {
    if (node.join_kind == JoinKind::Cross) {
        auto l_id = lower_node(*node.inputs[0], lo);
        auto r_id = lower_node(*node.inputs[1], lo);
        // CrossJoin's EXEC output = EXEC L ++ EXEC R (both sides kept
        // in their EXEC order). Rebuild by filtering node.output_schema
        // to fields that survived upstream pruning.
        std::unordered_set<ColumnId> live_union;
        for (const auto& f : lo.exec_schema(l_id).fields()) live_union.insert(f.id);
        for (const auto& f : lo.exec_schema(r_id).fields()) live_union.insert(f.id);
        Schema out_schema = filter_schema_by_live(node.output_schema, live_union);
        ValueId out_id = lo.next_value(std::move(out_schema));
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

    auto l_id = lower_node(*node.inputs[0], lo);
    auto r_id = lower_node(*node.inputs[1], lo);
    const auto& left_schema  = lo.exec_schema(l_id);
    const auto& right_schema = lo.exec_schema(r_id);

    EquiKeys keys;
    collect_equi_keys(*node.join_condition, left_schema, right_schema, keys);
    if (keys.left.size() != 1) {
        not_implemented("multi-key equi-joins");
    }

    auto l_key_ord = resolve_ordinal(left_schema, keys.left[0]);
    auto r_key_ord = resolve_ordinal(right_schema, keys.right[0]);

    // Intermediate value: raw join output in physical layout (built
    // over EXEC schemas so upstream pruning is honored).
    Schema raw_schema = (node.join_kind == JoinKind::Inner)
        ? build_raw_inner_join_schema(left_schema, right_schema,
                                      keys.left[0], keys.right[0])
        : build_raw_semi_join_schema(left_schema, keys.left[0]);

    ValueId raw_id = lo.next_value(std::move(raw_schema));
    if (node.join_kind == JoinKind::Inner) {
        lo.push_step(StepKind::EqJoin, {l_id, r_id}, raw_id,
                     exec::EqJoinPayload{{l_key_ord}, {r_key_ord}});
    } else {
        lo.push_step(StepKind::SemiJoin, {l_id, r_id}, raw_id,
                     exec::SemiJoinPayload{{l_key_ord}, {r_key_ord}});
    }

    // Reshape raw → IR-declared order, skipping fields that were
    // pruned upstream (their ColumnIds are no longer in EXEC L / R).
    auto [final_schema, ordinals] = (node.join_kind == JoinKind::Inner)
        ? compute_inner_join_reshape(
              node.output_schema, left_schema, right_schema,
              keys.left[0], keys.right[0])
        : compute_semi_join_reshape(
              node.output_schema, left_schema, keys.left[0]);

    ValueId out_id = lo.next_value(std::move(final_schema));
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

exec::ExecPlan lower_to_exec_plan(
    const PlanPtr& root, const Catalog& cat, const LowerOptions& opts)
{
    if (!root) lower_error("null root");
    Lowering lo;
    lo.cat = &cat;
    lo.opts = opts;
    lo.liveness = compute_liveness(*root);
    (void)lower_node(*root, lo);
    return std::move(lo.plan);
}

} // namespace gamax::ir
