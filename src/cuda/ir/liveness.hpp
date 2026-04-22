#pragma once

#include <unordered_map>
#include <unordered_set>

#include "ir/column_id.hpp"
#include "ir/plan_node.hpp"

namespace gamax::ir {

// Column-liveness analysis over an IR DAG.
//
// For each node, `live_out[node]` is the set of ColumnIds from that
// node's `output_schema` that are consumed by at least one ancestor
// (either the final result or some operator that references them via
// ColRef in a predicate / expression / key / sort / join-condition).
//
// Root's `live_out` is the full set of output ColumnIds — a query
// produces everything its IR root declares.
//
// DAG sharing is handled correctly: a node with multiple parents
// receives the union of what each parent needs.
//
// Used by the lowering pass's column-pruning rule: for a Filter whose
// input contains columns not in `live_out[filter]`, the lowering
// passes `output_ordinals` to `phys::filter` so only the live columns
// are materialized after the bitmask is computed.
//
// Future rules (Scan pushdown, Project dead-item elimination, etc.)
// consume the same analysis.
struct LivenessResult {
    std::unordered_map<const PlanNode*, std::unordered_set<ColumnId>> live_out;

    // Convenience: look up the live set for `node`; returns an empty
    // set if the node wasn't reached by the walk (shouldn't happen
    // for nodes in the analyzed DAG, but safe default).
    const std::unordered_set<ColumnId>&
        at(const PlanNode* node) const;
};

// Walk the DAG rooted at `root`. Every node reachable from `root`
// gets an entry in the returned map.
LivenessResult compute_liveness(const PlanNode& root);

} // namespace gamax::ir
