#pragma once

#include "exec/exec_plan.hpp"
#include "ir/catalog.hpp"
#include "ir/plan_node.hpp"

namespace gamax::ir {

// Lower an IR plan DAG to an `exec::ExecPlan`. All host-side work
// (ColumnId→ordinal resolution, Predicate/ExprTree construction,
// join-key extraction, reshape-ordinal computation, Sort+Limit fusion
// into TopK) happens here. The resulting ExecPlan is ready-to-fire —
// an Executor or CodeGenerator never touches the IR.
//
// `cat` supplies the catalog columns Scan nodes reference by name; the
// returned `required_sources` lists the table names an Executor or
// CodeGen caller must supply at run time.
exec::ExecPlan lower_to_exec_plan(const PlanPtr& root, const Catalog& cat);

} // namespace gamax::ir
