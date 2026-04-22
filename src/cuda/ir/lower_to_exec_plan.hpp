#pragma once

#include "exec/exec_plan.hpp"
#include "ir/catalog.hpp"
#include "ir/plan_node.hpp"

namespace gamax::ir {

// Codegen-level toggles for the lowering pass.
//
// These are distinct from `RuleConfig`: rules rewrite the IR (semantic
// / structural transformations that are observable in the IR tree
// handed to lowering), whereas `LowerOptions` controls decisions the
// lowering itself makes about how to produce an ExecPlan from a given
// IR tree — no IR rewrite involved. The two are orthogonal knobs that
// compose cleanly.
//
// All options default to the currently-best behavior. Flipping them
// off is useful for benchmarking a given optimization's impact, and
// for debugging when an optimization's effect needs to be ruled out
// as the source of a bug.
struct LowerOptions {
    // When lowering a `Filter` node, drop input columns from the
    // Filter's output if no downstream consumer reads them. Uses
    // liveness analysis to decide what's dead; emits non-empty
    // `FilterPayload::output_ordinals` when any column is pruned.
    //
    // Always correctness-preserving — downstream references resolve
    // by ColumnId against the pruned Value schema, and columns the
    // predicate touches are forced live by liveness so they stay in
    // the input. Off by default turns into emitting the Filter with
    // input-=-output materialization (everything survives; Reorder
    // picks out what parents actually want).
    bool prune_filter_materialization {true};
};

// Lower an IR plan DAG to an `exec::ExecPlan`. All host-side work
// (ColumnId→ordinal resolution, Predicate/ExprTree construction,
// join-key extraction, reshape-ordinal computation, Sort+Limit fusion
// into TopK) happens here. The resulting ExecPlan is ready-to-fire —
// an Executor or CodeGenerator never touches the IR.
//
// `cat` supplies the catalog columns Scan nodes reference by name; the
// returned `required_sources` lists the table names an Executor or
// CodeGen caller must supply at run time.
exec::ExecPlan lower_to_exec_plan(
    const PlanPtr& root, const Catalog& cat,
    const LowerOptions& opts = {});

} // namespace gamax::ir
