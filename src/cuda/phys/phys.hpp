#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "expr/expr_fast.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/select/predicate.hpp"
#include "types/data_type.hpp"

namespace gamax::phys {

// Physical operator contract.
//
// This layer is the ONLY place in the stack that knows about the
// incomplete / quirky surfaces of the raw `gamax::*` operators
// (e.g. `project::project` rejecting STRING; `inner_eq_join`
// assuming column 0 is the key). Callers above the line see a
// uniform, complete interface. Gaps in the underlying operators
// are either routed around here (B7: STRING passthrough via
// per-column copy) or fail with a clear "not yet implemented"
// error at call time (B1 / B2 / B3: outer / anti / cross join).
//
// Design notes captured in the plan doc:
//   * One stream per executor / codegen instance (carried on Context).
//   * `phys::*` does NOT rewrite semantics — rewrites (COUNT → SUM(1),
//     StringOp-in-project materialization, cross-join-with-scalar
//     broadcast) live at the rule-engine / lowering layer. Only pure
//     *implementation details* (STRING copy, multi-key key-fusion)
//     belong inside these wrappers.
//   * Payloads passed in are ready-to-fire (ordinals, pre-built
//     Predicate / ExprTree, AggSpec / SortSpec lists). No host-side
//     IR walking happens here; that's the lowering pass's job.
struct Context {
    cudaStream_t stream {cudaStreamDefault};
    // Future: memory resource, tracing hooks, NVTX ranges. One stream
    // per executor today; a stream pool is a later, additive extension.
};

using table_ptr = std::unique_ptr<gamax::table>;

// Aggregation specification: which input column to aggregate and which
// aggregation kind. Uses the existing `gamax::groupby::aggregation_id`
// so this layer is a thin wrapper over the physical enum; callers
// above the line (e.g. the ExecPlan lowering) translate from the
// richer `ir::AggKind` set and skip / rewrite unsupported kinds.
struct AggSpec {
    gamax::size_type column;
    gamax::groupby::aggregation_id agg;
};

struct SortSpec {
    gamax::size_type column;
    gamax::sort_order order;
};

// ── Data reshaping ────────────────────────────────────────────────

// Pure-ColRef reorder / column subset. Produces a new owned table
// whose i-th column is the `ordinals[i]`-th column of `in`. STRING
// columns are deep-copied via identity `gather_strings`; numeric
// columns via `copy_column`.
//
// **Closes gap B7 for every caller above this layer.**
//
// This is the primary primitive used for:
//   * Scan's "pick requested columns from the source table" step.
//   * Pre-join reshape that moves keys to the front.
//   * Post-join reshape that restores IR-declared column order.
//   * Aggregate's key+value materialization.
//
// Ordinals may repeat; each occurrence materializes a fresh copy of
// the referenced column.
table_ptr reorder(gamax::table_view in,
                  std::vector<gamax::size_type> ordinals,
                  const Context& ctx);

// ── Computation ───────────────────────────────────────────────────

// Computed projection via the existing `gamax::project::project`.
// `exprs` must reference columns by ordinal and produce numeric /
// BOOL8 / Case results — StringOp bodies are rejected at the
// `lower_expr_to_tree` layer, so they can't reach here.
table_ptr project(gamax::table_view in,
                  std::vector<gamax::expression::ExprTree>& exprs,
                  const Context& ctx);

// Heterogeneous filter. Uses `select_to_bitmask(Predicate)` then
// materializes surviving rows across every column of `in`: numeric
// via `compact`, STRING via a shared `compact_tids` + `gather_strings`
// pattern — same approach today's `apply_bitmask_to_heterogeneous_table`
// uses in `emit.cpp`, just lifted into the phys layer.
table_ptr filter(gamax::table_view in,
                 gamax::select::Predicate& pred,
                 const Context& ctx);

// Selective-output overload: predicate is evaluated against all of
// `in` (so it may reference any ordinal), but only the columns listed
// in `output_ordinals` are materialized in the result (in that order).
// An empty `output_ordinals` is equivalent to the 3-arg overload and
// produces every input column.
//
// Used by the IR-level column-pruning rule: a column the predicate
// touches but downstream ops never consume is dropped without the
// per-row compact/gather cost of carrying it through.
table_ptr filter(gamax::table_view in,
                 gamax::select::Predicate& pred,
                 std::vector<gamax::size_type> output_ordinals,
                 const Context& ctx);

// ── Joins ─────────────────────────────────────────────────────────

// Inner equi-join. `lkeys[i]` joins to `rkeys[i]`; `lkeys.size()` must
// equal `rkeys.size()`.
//
// Today's raw `inner_eq_join` assumes column 0 of each input is the
// (single) key and deduplicates it on output. This wrapper therefore
// *requires* `lkeys.size() == 1` for now and throws on multi-key —
// the multi-key path via `multikey_compress` lands as a follow-up
// without changing this signature.
//
// Output shape: `[shared_key, L_rest..., R_rest...]` (the raw
// `inner_eq_join` convention). Callers that want IR-declared column
// order follow up with a `phys::reorder`.
table_ptr eq_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                  gamax::table_view R, std::vector<gamax::size_type> rkeys,
                  const Context& ctx);

// Left semi-join. Same key-list semantics as `eq_join`; output is
// `[shared_key, L_rest...]`.
table_ptr semi_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                    gamax::table_view R, std::vector<gamax::size_type> rkeys,
                    const Context& ctx);

// ── Joins declared but not yet implemented at the backend ─────────
//
// Surface is complete so the lowering pass can call them uniformly;
// bodies throw `"phys::<name>: not yet implemented"` until the
// physical operator lands.
table_ptr left_outer_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                          gamax::table_view R, std::vector<gamax::size_type> rkeys,
                          const Context& ctx);
table_ptr anti_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                    gamax::table_view R, std::vector<gamax::size_type> rkeys,
                    const Context& ctx);
table_ptr cross_join(gamax::table_view L, gamax::table_view R,
                     const Context& ctx);

// ── Aggregation ───────────────────────────────────────────────────

// Scalar aggregate over `values` (one-row output).
table_ptr scalar_aggregate(gamax::table_view values,
                           std::vector<AggSpec> specs,
                           const Context& ctx);

// Grouped aggregate. `keys` and `values` are two views over the same
// underlying table (or separate tables with matching row counts).
// The caller is responsible for having materialized them into
// contiguous column ranges — use `phys::reorder` for that.
table_ptr group_by(gamax::table_view keys,
                   gamax::table_view values,
                   std::vector<AggSpec> specs,
                   const Context& ctx);

// ── Ordering ──────────────────────────────────────────────────────

table_ptr sort(gamax::table_view in,
               std::vector<SortSpec> keys,
               const Context& ctx);

// Top-K: fused sort + limit. `result_in_sorted_order=true` matches
// the existing `sort::topk_table` convention.
table_ptr top_k(gamax::table_view in,
                std::vector<SortSpec> keys,
                gamax::size_type k,
                const Context& ctx);

} // namespace gamax::phys
