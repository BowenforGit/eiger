#include "ir/liveness.hpp"

#include <stdexcept>

#include "ir/expr.hpp"
#include "ir/schema.hpp"

namespace gamax::ir {

namespace {

using ColumnSet = std::unordered_set<ColumnId>;

// Add every ColRef-leaf ColumnId in `e` to `out`. Thin wrapper over
// `collect_column_refs` that writes into an unordered_set.
void add_refs(const Expr& e, ColumnSet& out) {
    auto refs = collect_column_refs(e);
    for (auto id : refs) out.insert(id);
}

// Given a Filter / Sort / Limit / Distinct node `N` whose output
// column set equals its input column set (pass-through semantics),
// compute what N needs from its child: its own `live_out` plus every
// ColumnId referenced in its operator-specific expressions (e.g.
// filter predicate, sort keys).
//
// `extra_refs` accumulates those operator-specific references.
ColumnSet needs_from_child_passthrough(
    const ColumnSet& live_out, const ColumnSet& extra_refs)
{
    ColumnSet out = live_out;
    for (auto id : extra_refs) out.insert(id);
    return out;
}

// Look up the entry for `node`, creating an empty set if absent.
ColumnSet& get_or_insert(
    std::unordered_map<const PlanNode*, ColumnSet>& m,
    const PlanNode* node)
{
    return m[node];  // default-constructs to empty set on first access
}

// Merge `src` into `dst`. Returns true if any element was added —
// the DAG walker uses this to decide whether a shared node needs to
// be re-visited (liveness grew since the last visit).
bool merge_into(ColumnSet& dst, const ColumnSet& src) {
    bool changed = false;
    for (auto id : src) {
        if (dst.insert(id).second) changed = true;
    }
    return changed;
}

// For each project item, its output ColumnId is either the first
// input id (pure ColRef — the factory preserves it) or a fresh id
// from the item's Field in the node's output_schema. The canonical
// answer lives in `output_schema[i].id`, so we read it directly.
ColumnId project_item_output_id(const PlanNode& node, size_t i) {
    return node.output_schema[i].id;
}

// Same for aggregate items: keys keep their input ColumnId (stored
// as-is at the head of output_schema), then agg_items[i] maps to
// `output_schema[keys.size() + i].id`.
ColumnId agg_item_output_id(const PlanNode& node, size_t i) {
    return node.output_schema[node.agg_keys.size() + i].id;
}

// Per-node: given the live set at N's output, produce the live sets
// at each of N's child outputs. Returned vector has the same length
// as `node.inputs`.
//
// Intuition:
//   * Filter / Sort / Distinct / Limit are pass-through — their
//     input and output schemas share ColumnIds field-for-field.
//     Their child needs (live_out ∪ cols-referenced-by-this-op).
//   * Project prunes dead items: a child-side column is live only
//     if its id appears in some item-expr whose output is live.
//   * Aggregate fully redefines its output — keys passthrough their
//     ids, agg-bodies get fresh ids. Child needs: agg_keys (always,
//     they control grouping) ∪ referenced-by-live-agg-body-items.
//   * Join needs its side-specific live_out ∪ join-condition refs
//     that belong to that side.
//   * Scan has no children.
std::vector<ColumnSet> child_live_sets(
    const PlanNode& node, const ColumnSet& live_out)
{
    switch (node.kind) {
        case NodeKind::Scan:
            return {};

        case NodeKind::Filter: {
            ColumnSet extra;
            add_refs(node.filter_predicate, extra);
            return { needs_from_child_passthrough(live_out, extra) };
        }

        case NodeKind::Sort: {
            ColumnSet extra;
            for (const auto& k : node.sort_keys) extra.insert(k.id);
            return { needs_from_child_passthrough(live_out, extra) };
        }

        case NodeKind::Distinct: {
            // Distinct with explicit keys: child needs keys ∪ live_out.
            // Distinct with empty keys (SQL DISTINCT default): every
            // output column contributes to deduplication, so child
            // needs the full output schema of this node (= its input
            // schema, since Distinct is pass-through).
            ColumnSet extra;
            if (node.distinct_keys.empty()) {
                for (const auto& f : node.output_schema.fields()) {
                    extra.insert(f.id);
                }
            } else {
                for (auto id : node.distinct_keys) extra.insert(id);
            }
            return { needs_from_child_passthrough(live_out, extra) };
        }

        case NodeKind::Limit:
            // Limit is pure row-count restriction; ColumnId set at its
            // output equals its input. No extra refs.
            return { live_out };

        case NodeKind::Project: {
            ColumnSet child;
            for (size_t i = 0; i < node.project_items.size(); ++i) {
                auto out_id = project_item_output_id(node, i);
                if (live_out.count(out_id) == 0) continue;
                add_refs(node.project_items[i].expr, child);
            }
            return { std::move(child) };
        }

        case NodeKind::Aggregate: {
            ColumnSet child;
            // Grouping keys are always needed from the child — the
            // group assignment logic consumes them even if the output
            // key column happens not to be live (unusual but possible).
            for (auto id : node.agg_keys) child.insert(id);
            for (size_t i = 0; i < node.agg_items.size(); ++i) {
                auto out_id = agg_item_output_id(node, i);
                if (live_out.count(out_id) == 0) continue;
                const auto& ag = node.agg_items[i];
                if (ag.agg == AggKind::CountAll) continue;  // no body
                add_refs(ag.body, child);
            }
            return { std::move(child) };
        }

        case NodeKind::Join: {
            // Split live_out into the parts contributed by each side
            // (IDs from left_schema stay on the left, from right_schema
            // on the right) and add the side-specific join-condition
            // references. Cross joins have no condition.
            const auto& L = node.inputs[0]->output_schema;
            const auto& R = node.inputs[1]->output_schema;

            ColumnSet left_needs, right_needs;
            for (auto id : live_out) {
                if (L.index_of(id).has_value()) left_needs.insert(id);
                if (R.index_of(id).has_value()) right_needs.insert(id);
                // Note: the same ColumnId can't legitimately be on
                // both sides (Scan allocates fresh ids per scan); if
                // it is, both parents get told.
            }
            if (node.join_condition) {
                ColumnSet cond_refs;
                add_refs(*node.join_condition, cond_refs);
                for (auto id : cond_refs) {
                    if (L.index_of(id).has_value()) left_needs.insert(id);
                    if (R.index_of(id).has_value()) right_needs.insert(id);
                }
            }
            return { std::move(left_needs), std::move(right_needs) };
        }
    }
    throw std::runtime_error("liveness: unhandled NodeKind");
}

} // namespace

const std::unordered_set<ColumnId>&
LivenessResult::at(const PlanNode* node) const {
    static const std::unordered_set<ColumnId> empty;
    auto it = live_out.find(node);
    return it == live_out.end() ? empty : it->second;
}

LivenessResult compute_liveness(const PlanNode& root) {
    LivenessResult res;

    // Root's live_out is everything in its output schema.
    {
        auto& s = res.live_out[&root];
        for (const auto& f : root.output_schema.fields()) s.insert(f.id);
    }

    // Worklist-driven propagation. A node may be visited multiple
    // times when it sits beneath a shared node whose live set grew.
    // Each pass updates children's live sets; on a child's change we
    // re-enqueue it. Liveness is a monotonically-growing lattice so
    // this terminates after finite merges.
    std::vector<const PlanNode*> worklist = {&root};
    while (!worklist.empty()) {
        const PlanNode* n = worklist.back();
        worklist.pop_back();

        auto current_live = res.live_out[n];   // copy: we may mutate map below
        auto child_sets = child_live_sets(*n, current_live);

        for (size_t i = 0; i < n->inputs.size(); ++i) {
            const auto* child = n->inputs[i].get();
            auto& child_set = get_or_insert(res.live_out, child);
            bool changed = merge_into(child_set, child_sets[i]);
            if (changed) worklist.push_back(child);
        }
    }
    return res;
}

} // namespace gamax::ir
