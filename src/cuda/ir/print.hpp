#pragma once

#include <ostream>
#include <string>
#include <unordered_map>

#include "ir/column_id.hpp"
#include "ir/expr.hpp"
#include "ir/plan_node.hpp"
#include "ir/schema.hpp"

namespace gamax::ir {

// Render a plan DAG as indented text for inspection and snapshot tests.
//
// Column references in expressions print by name when the plan's
// schemas supply one; unknown ids fall back to `#<id>`. Rule-based
// rewrites see the same format before/after, so plan diffs are readable.
//
// This is a tree-form renderer: shared sub-plans (multiple parents)
// duplicate on output. Since typical TPC-H plans don't yet share,
// reference-label support (`$tN`) can be added alongside the rule
// engine when it starts producing shared outputs.
std::string format_plan(const PlanNode& root);
void print_plan(std::ostream& os, const PlanNode& root);

// Stand-alone expression formatter, useful for debug prints outside
// of a full plan. ColRef leaves print as `#<id>` (no schema context).
std::string format_expr(const Expr& e);

// Lower-level: format an expression using `names` to render ColRef
// leaves by name when available. Exposed so callers that thread their
// own name map (e.g., a rule-engine traversal) can reuse the rendering.
std::string format_expr(
    const Expr& e,
    const std::unordered_map<ColumnId, std::string>& names);

// Schema summary: "[name:TYPE, name:TYPE, ...]". Useful for debug and
// for printing Scan's requested-column list.
std::string format_schema(const Schema& s);

} // namespace gamax::ir
