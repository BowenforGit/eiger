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

// Engine defaults used when no `LowerOptions` override is supplied.
// Mirrors the legacy `phys.cu::default_join_hint()` so behavior is
// identical for IR queries that don't set hints — the join operator
// rejects `AUTO_ALGO`/`AUTO_MAT`, so a concrete pair is required.
gamax::join::join_hint default_join_hint() {
    gamax::join::join_hint h;
    h.join_algo = gamax::join::HASH;
    h.mat_algo  = gamax::join::GFTR;
    return h;
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

    // Return-by-value: callers often hold onto the result across a
    // subsequent `next_value` call, and next_value can grow the
    // `plan.values` vector and invalidate any reference we returned.
    // Copy protects all callers from that reallocation hazard.
    Schema exec_schema(ValueId id) const {
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
    // Scan → SourceTable directly. The payload carries the requested
    // column names in output order, and the resolver is contractually
    // obliged to surface those columns (no full-catalog load, no
    // subsequent Reorder). Column-order reshuffles happen at the
    // SourceTable boundary, not as a separate step — saves one D2D
    // copy per source and, with a column-aware reader, avoids
    // materialising unused columns altogether.
    const auto* cat_table = lo.cat->find(node.scan_table);
    if (!cat_table) lower_error(
        "scan: table '" + node.scan_table + "' not in catalog");

    // Validate each requested column exists in the catalog so errors
    // surface here rather than deep in the resolver.
    for (const auto& name : node.scan_columns) {
        auto it = std::find(cat_table->column_names.begin(),
                            cat_table->column_names.end(), name);
        if (it == cat_table->column_names.end()) {
            lower_error("scan: column '" + name + "' not in table '" +
                        node.scan_table + "'");
        }
    }

    // `required_sources` is a dedup'd list for the executor to supply.
    auto& req = lo.plan.required_sources;
    if (std::find(req.begin(), req.end(), node.scan_table) == req.end()) {
        req.push_back(node.scan_table);
    }

    // SourceTable's schema is the IR Scan's declared output_schema —
    // one entry per requested column, in order.
    gamax::ir::Schema out_schema = node.output_schema;
    ValueId out_id = lo.next_value(std::move(out_schema));
    lo.push_step(StepKind::SourceTable, /*inputs=*/{}, out_id,
                 exec::SourceTablePayload{
                     node.scan_table,
                     node.scan_columns,
                 });
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
    exec::FilterPayload pl;
    pl.predicate = std::move(pred);
    pl.output_ordinals = std::move(output_ordinals);
    pl.hint = lo.opts.filter.value_or(gamax::select::select_options{});
    lo.push_step(StepKind::Filter, {input_id}, out_id, std::move(pl));
    return out_id;
}

ValueId lower_project(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = lo.exec_schema(input_id);

    // Dispatch per item:
    //   * `substring(col, offset_lit, length_lit)` → SubstringItem
    //     variant (produces a STRING column via gamax::substring).
    //     We lower it before the StringOp/ExprTree branches because
    //     its FunctionCall has no ExprTree form (STRING output).
    //   * items that contain a StringOp anywhere → lower_expr_to_predicate
    //     (Predicate evaluated via select_to_bitmask → BOOL8).
    //   * everything else → lower_expr_to_tree. StringOps nested inside
    //     a numeric ExprTree (e.g. inside a CASE body) are still rejected;
    //     builders must pre-split those.
    std::vector<gamax::phys::ProjectItem> items;
    items.reserve(node.project_items.size());
    for (const auto& it : node.project_items) {
        const auto& e = it.expr;
        if (e.kind == ExprKind::FunctionCall && e.function_name == "substring") {
            if (e.children.size() != 3) {
                lower_error("substring(...) takes exactly three arguments");
            }
            if (e.children[0].kind != ExprKind::ColRef ||
                e.children[0].type.id() != gamax::type_id::STRING) {
                lower_error("substring: first argument must be a STRING ColRef");
            }
            if (e.children[1].kind != ExprKind::Literal ||
                e.children[2].kind != ExprKind::Literal) {
                lower_error("substring: offset and length must be integer literals");
            }
            auto get_int = [](const Expr& lit) -> int64_t {
                return std::visit([](auto&& v) -> int64_t {
                    using V = std::decay_t<decltype(v)>;
                    if constexpr (std::is_same_v<V, int64_t>) return v;
                    else if constexpr (std::is_same_v<V, bool>) return v ? 1 : 0;
                    else lower_error("substring: offset/length literal must be integer");
                }, lit.literal);
            };
            const int64_t sql_offset = get_int(e.children[1]);
            const int64_t length     = get_int(e.children[2]);
            auto ordinal = resolve_ordinal(in_schema, e.children[0].col_ref);
            items.emplace_back(gamax::phys::SubstringItem{
                ordinal,
                static_cast<gamax::size_type>(sql_offset),
                static_cast<gamax::size_type>(length)});
        } else if (expr_contains_string_op(it.expr)) {
            items.emplace_back(lower_expr_to_predicate(
                it.expr, in_schema, cudaStreamDefault));
        } else {
            items.emplace_back(lower_expr_to_tree(it.expr, in_schema));
        }
    }

    // Project rebuilds its output from scratch (one column per item);
    // it's independent of any upstream pruning, so the EXEC schema
    // equals the IR's declared output_schema.
    ValueId out_id = lo.next_value(node.output_schema);
    exec::ProjectPayload pl;
    pl.items = std::move(items);
    pl.hint = lo.opts.project.value_or(gamax::project::project_options{});
    lo.push_step(StepKind::Project, {input_id}, out_id, std::move(pl));
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
            // Both lower to the phys COUNT_ALL kind. Spec-level
            // Count(x) is "count non-null x"; we treat it the same
            // because TPC-H data has no nulls and the engine doesn't
            // propagate null masks anyway. A future null-aware engine
            // would need to split these (Count via mask popcount,
            // CountAll via row-count broadcast). The G4a rewrite
            // (rule_count_to_sum) used to materialize a full-
            // cardinality lit(1) column to feed Sum here — at SF=100
            // that allocation OOMed the slab. Native COUNT_ALL skips
            // the materialization entirely; the agg layer reuses the
            // internal "ones" column already built for AVG's
            // denominator.
            return gamax::groupby::aggregation_id::COUNT_ALL;
        case AggKind::CountDistinct:
            // Only a `rule_count_distinct_to_distinct_plus_count`-
            // rewritten Aggregate lowers today (single CountDistinct as
            // the sole agg, with a ColRef body). Any CountDistinct that
            // survives rules violates that narrow scope.
            not_implemented("COUNT(DISTINCT ...) reached lowering. "
                            "Supported shape: a single CountDistinct as "
                            "the sole aggregate item with a plain column "
                            "reference as its body; enable "
                            "rewrite_count_distinct in RuleConfig. Mixing "
                            "CountDistinct with other aggregates, or "
                            "applying it to a computed expression, is "
                            "not supported today.");
    }
    lower_error("unknown AggKind");
}

// Result of materialize_keys_and_values.
//   * value_id      — the materialized intermediate that GroupBy /
//                     ScalarAgg consumes by column index.
//   * n_keys        — number of group-by keys (== first n_keys
//                     columns of the materialized table).
//   * body_position — for each agg item, its body column's position
//                     in the *value-only* slice of the materialized
//                     table (i.e. 0 maps to the first column AFTER
//                     the keys). -1 for COUNT_ALL/Count items
//                     (which materialize no body column). When two
//                     agg items reference the same source ColRef,
//                     they share a body_position — a deduplication
//                     that drops the redundant materialization for
//                     patterns like Sum(x) + Avg(x) on the same x.
struct AggMaterialization {
    ValueId mat_id;
    gamax::size_type n_keys;
    std::vector<gamax::size_type> body_position;
};

// Walk agg_items, assign each a body-column position. Agg items
// whose body is a ColRef referring to a ColumnId we've already seen
// reuse the prior position (no fresh column emitted). Computed
// bodies always emit a fresh column. CountAll/Count emit nothing
// and get sentinel -1.
struct DedupResult {
    std::vector<gamax::size_type> body_position;     // per agg item
    std::vector<gamax::size_type> emit_for_item;     // indices of items
                                                     // that DRIVE a fresh
                                                     // body emission (in
                                                     // emission order)
};

DedupResult dedup_agg_bodies(const PlanNode& node) {
    DedupResult r;
    r.body_position.assign(node.agg_items.size(), -1);
    std::unordered_map<ColumnId, gamax::size_type> col_ref_pos;
    gamax::size_type next_pos = 0;
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(node.agg_items.size()); ++i)
    {
        const auto& ag = node.agg_items[i];
        if (ag.agg == AggKind::CountAll || ag.agg == AggKind::Count) {
            continue;
        }
        if (ag.body.kind == ExprKind::ColRef) {
            auto it = col_ref_pos.find(ag.body.col_ref);
            if (it != col_ref_pos.end()) {
                r.body_position[i] = it->second;
                continue;
            }
            col_ref_pos[ag.body.col_ref] = next_pos;
        }
        r.body_position[i] = next_pos++;
        r.emit_for_item.push_back(i);
    }
    return r;
}

// Materialize `[keys..., agg-body-columns...]` as an intermediate
// value that `phys::group_by` / `phys::scalar_aggregate` can consume
// via column-index. Matches the shape the current emit.cpp builds
// inline; hoisted into the ExecPlan so it's visible and reusable.
//
// `in_schema` must be the EXEC schema of the aggregate's input
// (i.e., the actual column layout of `input_id`).
AggMaterialization materialize_keys_and_values(
    const PlanNode& node, ValueId input_id,
    const Schema& in_schema, Lowering& lo)
{
    auto dedup = dedup_agg_bodies(node);

    // Decide between a pure-ColRef path (Reorder) and a mixed path
    // (Project) based on whether any item that drives a fresh body
    // emission has a non-ColRef body. Items deduped to an earlier
    // ColRef (or skipped because they're CountAll/Count) don't
    // affect the decision — they emit nothing of their own.
    bool all_col_refs = true;
    for (auto i : dedup.emit_for_item) {
        if (node.agg_items[i].body.kind != ExprKind::ColRef) {
            all_col_refs = false;
            break;
        }
    }

    // Build the materialized value's output schema: keys first, then
    // one field per UNIQUE body column (deduped across agg items by
    // ColRef ColumnId). CountAll/Count items produce an output column
    // at the agg step but consume nothing from the materialized
    // table — the phys layer's COUNT_ALL handler synthesizes its
    // result column from the internal "ones" column it builds for
    // AVG's denominator.
    Schema mat_schema;
    for (auto kid : node.agg_keys) {
        auto ordinal = resolve_ordinal(in_schema, kid);
        mat_schema.append(in_schema[ordinal]);
    }
    for (auto i : dedup.emit_for_item) {
        const auto& ag = node.agg_items[i];
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
        // Pure-ColRef materialization → Reorder. One ordinal per
        // unique body (dedup-driven) plus one per key.
        std::vector<gamax::size_type> ordinals;
        ordinals.reserve(node.agg_keys.size() + dedup.emit_for_item.size());
        for (auto kid : node.agg_keys) {
            ordinals.push_back(resolve_ordinal(in_schema, kid));
        }
        for (auto i : dedup.emit_for_item) {
            const auto& ag = node.agg_items[i];
            ordinals.push_back(resolve_ordinal(in_schema, ag.body.col_ref));
        }
        lo.push_step(StepKind::Reorder, {input_id}, mat_id,
                     exec::ReorderPayload{std::move(ordinals)});
    } else {
        // Mixed / computed → Project. Agg keys are always ColRefs;
        // agg bodies are ColRefs (deduped) or numeric computed — no
        // StringOps flow through here, so every item is an ExprTree
        // wrapped in the ProjectItem variant.
        std::vector<gamax::phys::ProjectItem> items;
        items.reserve(node.agg_keys.size() + dedup.emit_for_item.size());
        for (auto kid : node.agg_keys) {
            auto ordinal = resolve_ordinal(in_schema, kid);
            items.emplace_back(gamax::expression::ExprTree::column(
                ordinal, in_schema[ordinal].type));
        }
        for (auto i : dedup.emit_for_item) {
            items.emplace_back(
                lower_expr_to_tree(node.agg_items[i].body, in_schema));
        }
        exec::ProjectPayload mat_pl;
        mat_pl.items = std::move(items);
        mat_pl.hint = lo.opts.project.value_or(gamax::project::project_options{});
        lo.push_step(StepKind::Project, {input_id}, mat_id, std::move(mat_pl));
    }
    return AggMaterialization{
        mat_id, n_keys, std::move(dedup.body_position)};
}

ValueId lower_aggregate(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = lo.exec_schema(input_id);

    auto mat = materialize_keys_and_values(node, input_id, in_schema, lo);

    // Build AggSpecs. Spec.column is val-relative (val is the part of
    // the materialized table after the n_keys keys for groupby; for
    // scalar aggregate, n_keys is 0 so val == whole table). The
    // dedup pass in materialize_keys_and_values produced
    // `body_position[i]` — the val-relative index of the body column
    // for agg item i, with -1 for COUNT_ALL/Count. Two agg items
    // sharing a ColRef body share a body_position, so the spec
    // points to the single materialized column for both — and
    // groupby_noskf's existing AVG-reuses-SUM dedup catches it from
    // there.
    std::vector<gamax::phys::AggSpec> specs;
    specs.reserve(node.agg_items.size());
    for (gamax::size_type i = 0;
         i < static_cast<gamax::size_type>(node.agg_items.size()); ++i)
    {
        const auto& ag = node.agg_items[i];
        if (ag.agg == AggKind::CountAll || ag.agg == AggKind::Count) {
            specs.push_back({/*column=*/-1,
                             gamax::groupby::aggregation_id::COUNT_ALL});
        } else {
            specs.push_back({mat.body_position[i],
                             to_physical_agg(ag.agg)});
        }
    }

    ValueId out_id = lo.next_value(node.output_schema);
    if (mat.n_keys == 0) {
        lo.push_step(StepKind::ScalarAgg, {mat.mat_id}, out_id,
                     exec::ScalarAggPayload{std::move(specs)});
    } else {
        exec::GroupByPayload pl;
        pl.n_keys = mat.n_keys;
        pl.specs = std::move(specs);
        pl.hint = lo.opts.group_by.value_or(gamax::groupby::groupby_hint{});
        lo.push_step(StepKind::GroupBy, {mat.mat_id}, out_id, std::move(pl));
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
    exec::SortPayload pl;
    pl.keys = std::move(keys);
    pl.hint = lo.opts.sort.value_or(gamax::sort::sort_table_options{});
    lo.push_step(StepKind::Sort, {input_id}, out_id, std::move(pl));
    return out_id;
}

// Lower NodeKind::Concat → StepKind::Concat. All inputs are
// recursively lowered first; their EXEC schemas must all have the
// same column count and positionally-matching types (the IR factory
// enforces this, and we re-check here because post-rules passes
// could conceivably break the invariant).
//
// The output Value's EXEC schema carries the IR's declared output
// schema — which uses fresh ColumnIds per the factory's policy (see
// plan_node.cpp::concat). Every downstream consumer resolves against
// those fresh ids.
ValueId lower_concat(const PlanNode& node, Lowering& lo) {
    std::vector<ValueId> input_ids;
    input_ids.reserve(node.inputs.size());
    for (const auto& in : node.inputs) {
        input_ids.push_back(lower_node(*in, lo));
    }
    // Sanity check on EXEC schemas after any pruning / rewrites.
    const auto& ref = lo.exec_schema(input_ids[0]);
    for (size_t k = 1; k < input_ids.size(); ++k) {
        const auto& s = lo.exec_schema(input_ids[k]);
        if (s.size() != ref.size()) {
            lower_error("concat: input " + std::to_string(k) +
                        " has different column count after lowering");
        }
        for (size_t c = 0; c < s.size(); ++c) {
            if (!(s[c].type == ref[c].type)) {
                lower_error("concat: input " + std::to_string(k) +
                            " column " + std::to_string(c) +
                            " type mismatch after lowering");
            }
        }
    }
    ValueId out_id = lo.next_value(node.output_schema);
    lo.push_step(StepKind::Concat, std::move(input_ids), out_id,
                 exec::ConcatPayload{});
    return out_id;
}

// Lower NodeKind::Distinct → StepKind::Distinct with a sparse
// `key_ordinals` payload. The EXEC output schema is the narrow
// keys-only subset when `distinct_keys` is a proper subset of the
// input's columns, and the full input schema when the keys cover
// every input column (or are empty). This mirrors the IR factory's
// output_schema policy exactly.
ValueId lower_distinct(const PlanNode& node, Lowering& lo) {
    auto input_id = lower_node(*node.inputs[0], lo);
    const auto& in_schema = lo.exec_schema(input_id);

    // Resolve each distinct-key id to an EXEC-side ordinal. Empty keys
    // mean "dedup over all input columns" — we synthesise the
    // identity ordinals so the phys side still gets a complete key
    // set.
    std::vector<gamax::size_type> key_ordinals;
    if (node.distinct_keys.empty()) {
        key_ordinals.resize(in_schema.size());
        std::iota(key_ordinals.begin(), key_ordinals.end(),
                  gamax::size_type{0});
    } else {
        key_ordinals.reserve(node.distinct_keys.size());
        for (auto kid : node.distinct_keys) {
            key_ordinals.push_back(resolve_ordinal(in_schema, kid));
        }
    }

    // Build the Distinct's EXEC output schema by picking the key
    // columns out of the input schema in the order they appear in
    // `key_ordinals`. When all input columns are keys (either empty
    // or covers-all), this naturally reconstructs the input schema.
    Schema out_schema;
    for (auto ord : key_ordinals) out_schema.append(in_schema[ord]);

    ValueId out_id = lo.next_value(std::move(out_schema));
    lo.push_step(StepKind::Distinct, {input_id}, out_id,
                 exec::DistinctPayload{std::move(key_ordinals)});
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
    exec::TopKPayload pl;
    pl.keys = std::move(keys);
    pl.k = node.limit_n;
    pl.hint = lo.opts.top_k.value_or(gamax::sort::sort_table_options{});
    lo.push_step(StepKind::TopK, {input_id}, out_id, std::move(pl));
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
        // to fields that survived upstream pruning. `exec_schema`
        // returns by value (see its declaration); keep the Schemas
        // named so the range-for below doesn't iterate over a
        // destroyed temporary's subobject.
        auto l_schema = lo.exec_schema(l_id);
        auto r_schema = lo.exec_schema(r_id);
        std::unordered_set<ColumnId> live_union;
        for (const auto& f : l_schema.fields()) live_union.insert(f.id);
        for (const auto& f : r_schema.fields()) live_union.insert(f.id);
        Schema out_schema = filter_schema_by_live(node.output_schema, live_union);
        ValueId out_id = lo.next_value(std::move(out_schema));
        lo.push_step(StepKind::CrossJoin, {l_id, r_id}, out_id,
                     exec::CrossJoinPayload{});
        return out_id;
    }
    if (node.join_kind != JoinKind::Inner
        && node.join_kind != JoinKind::LeftSemi
        && node.join_kind != JoinKind::LeftAnti)
    {
        not_implemented("join kind (only Inner / LeftSemi / LeftAnti / Cross wired today)");
    }
    if (!node.join_condition) {
        lower_error("join: missing condition");
    }

    auto l_id = lower_node(*node.inputs[0], lo);
    auto r_id = lower_node(*node.inputs[1], lo);
    auto left_schema  = lo.exec_schema(l_id);
    auto right_schema = lo.exec_schema(r_id);

    EquiKeys keys;
    collect_equi_keys(*node.join_condition, left_schema, right_schema, keys);
    if (keys.left.size() != 1) {
        not_implemented("multi-key equi-joins");
    }

    auto l_key_ord = resolve_ordinal(left_schema, keys.left[0]);
    auto r_key_ord = resolve_ordinal(right_schema, keys.right[0]);

    // LeftSemi and LeftAnti share the same output shape — only L's
    // columns in L's order, with the key first. They differ only in
    // the membership predicate (keep-if-match vs keep-if-no-match),
    // which is the kernel's job. Use the same raw-schema builder and
    // reshape function for both; pick the step kind based on join kind.
    const bool is_semi_or_anti =
        node.join_kind == JoinKind::LeftSemi
     || node.join_kind == JoinKind::LeftAnti;

    // Intermediate value: raw join output in physical layout (built
    // over EXEC schemas so upstream pruning is honored).
    Schema raw_schema = (node.join_kind == JoinKind::Inner)
        ? build_raw_inner_join_schema(left_schema, right_schema,
                                      keys.left[0], keys.right[0])
        : build_raw_semi_join_schema(left_schema, keys.left[0]);

    ValueId raw_id = lo.next_value(std::move(raw_schema));
    if (node.join_kind == JoinKind::Inner) {
        exec::EqJoinPayload pl;
        pl.lkeys = {l_key_ord};
        pl.rkeys = {r_key_ord};
        pl.hint  = lo.opts.eq_join.value_or(default_join_hint());
        lo.push_step(StepKind::EqJoin, {l_id, r_id}, raw_id, std::move(pl));
    } else if (node.join_kind == JoinKind::LeftSemi) {
        exec::SemiJoinPayload pl;
        pl.lkeys = {l_key_ord};
        pl.rkeys = {r_key_ord};
        pl.hint  = lo.opts.semi_join.value_or(default_join_hint());
        lo.push_step(StepKind::SemiJoin, {l_id, r_id}, raw_id, std::move(pl));
    } else {
        exec::AntiJoinPayload pl;
        pl.lkeys = {l_key_ord};
        pl.rkeys = {r_key_ord};
        pl.hint  = lo.opts.anti_join.value_or(default_join_hint());
        lo.push_step(StepKind::AntiJoin, {l_id, r_id}, raw_id, std::move(pl));
    }

    // Reshape raw → IR-declared order, skipping fields that were
    // pruned upstream (their ColumnIds are no longer in EXEC L / R).
    auto [final_schema, ordinals] = (node.join_kind == JoinKind::Inner)
        ? compute_inner_join_reshape(
              node.output_schema, left_schema, right_schema,
              keys.left[0], keys.right[0])
        : compute_semi_join_reshape(
              node.output_schema, left_schema, keys.left[0]);
    (void)is_semi_or_anti;

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
        case NodeKind::Distinct:  id = lower_distinct(node, lo); break;
        case NodeKind::Concat:    id = lower_concat(node, lo);   break;
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
