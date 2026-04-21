#pragma once

#include <cuda_runtime_api.h>

#include "expr/expr_fast.hpp"
#include "ir/expr.hpp"
#include "ir/schema.hpp"
#include "operators/select/predicate.hpp"

namespace gamax::ir {

// Resolves a ColumnId against `input` to its 0-based positional ordinal,
// or throws. This is the *only* place in the emit path where ColumnIds
// become ordinals; everywhere else in the IR stays ColumnId-keyed.
gamax::size_type resolve_ordinal(const Schema& input, ColumnId id);

// Lower a numeric / logical / Case / Column / Literal Expr into the
// engine's ExprTree. Used by project bodies, aggregate bodies, and the
// nested ExprTree wrapped inside Predicate::expr leaves.
//
// ExprTree does not natively support Not-Equal or Unary-Not, so those
// must appear in a Predicate context. StringOp is likewise rejected —
// strings must be materialized as boolean columns via a preceding
// select rather than appearing mid-computation.
gamax::expression::ExprTree lower_expr_to_tree(
    const Expr& e, const Schema& input);

// Lower a boolean Expr into a Predicate (select's filter format).
//
// Strategy: if the subtree contains no StringOp it's collapsed into a
// single Predicate::expr(ExprTree), which minimizes kernel launches for
// compound numeric filters like Q6's five-way AND. If any StringOp is
// present anywhere in the tree, logical combinators stay at the
// Predicate level and string leaves become Predicate::string_* leaves
// directly, matching what the existing select backend consumes.
//
// Needs a stream because string_scalar materialization involves a
// device copy.
gamax::select::Predicate lower_expr_to_predicate(
    const Expr& e,
    const Schema& input,
    cudaStream_t stream = cudaStreamDefault);

// Exposed for tests and emitter-local use: walks `e` and reports whether
// any node (any depth) is a StringOp. Drives the lowering heuristic
// above.
bool expr_contains_string_op(const Expr& e);

} // namespace gamax::ir
