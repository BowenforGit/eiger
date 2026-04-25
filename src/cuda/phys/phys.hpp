#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <cuda_runtime_api.h>

#include "expr/expr_fast.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/groupby.hpp"
#include "operators/join/join.hpp"
#include "operators/project/project.hpp"
#include "operators/select/predicate.hpp"
#include "operators/select/select.hpp"
#include "operators/sort/sort.hpp"
#include "types/data_type.hpp"

namespace gamax::exec { class StepProfileSink; }

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
    // Optional per-step GPU timing sink. When non-null, the executor
    // wraps each step's phys dispatch in a `cudaEvent` pair on `stream`
    // and pushes the elapsed time into the sink at finalize time. Null
    // by default — zero overhead for ordinary runs.
    gamax::exec::StepProfileSink* profile_sink {nullptr};
    // Future: memory resource; one stream per executor today (a stream
    // pool is a later, additive extension).
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

// Engine defaults for hint structs whose default-constructed values
// don't match what the underlying operators actually accept.
// `gamax::join::join_hint{}` carries `AUTO_ALGO`/`AUTO_MAT`, but
// `inner_eq_join` rejects auto and demands a concrete pair — HASH+GFTR
// matches the existing test/benchmark coverage. Inline so the default
// argument expressions in the eq_join / semi_join / anti_join
// signatures below can call this helper.
inline gamax::join::join_hint default_eq_join_hint() {
    gamax::join::join_hint h;
    h.join_algo = gamax::join::HASH;
    h.mat_algo  = gamax::join::GFTR;
    return h;
}

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

// Substring projection: produces a STRING column by slicing
// `[sql_offset, sql_offset + length)` out of `src_col`. `sql_offset`
// is 1-based (SQL convention); non-positive offsets are clamped to 1.
// Offsets and lengths are constants pinned at plan time — per-row
// dynamic bounds are not wired today (neither TPC-H Q22 nor any
// currently-supported query needs them).
struct SubstringItem {
    gamax::size_type src_col;
    gamax::size_type sql_offset;
    gamax::size_type length;
};

// Per-item projection. Each item is one of:
//   * computed / ColRef `ExprTree` (numeric expression evaluator)
//   * `Predicate`                  (→ BOOL8 column via select_to_bitmask)
//   * `SubstringItem`              (→ STRING column via gamax::substring)
// Output schema matches the `items` vector positionally. The predicate
// path is the vehicle for StringOp-based columns (e.g. `p_type LIKE
// 'PROMO%'`) living at the top of a ProjectItem — StringOps nested
// inside a numeric computational context are still rejected upstream
// at lowering time.
using ProjectItem = std::variant<
    gamax::expression::ExprTree,
    gamax::select::Predicate,
    SubstringItem>;
table_ptr project(gamax::table_view in,
                  std::vector<ProjectItem>& items,
                  const Context& ctx,
                  const gamax::project::project_options& hint = {});

// Heterogeneous filter. Uses `select_to_bitmask(Predicate)` then
// materializes surviving rows across every column of `in`: numeric
// via `compact`, STRING via a shared `compact_tids` + `gather_strings`
// pattern — same approach today's `apply_bitmask_to_heterogeneous_table`
// uses in `emit.cpp`, just lifted into the phys layer.
//
// `hint` chooses the algorithm used by `select_to_bitmask` for any
// numeric `EXPR` leaves of the predicate (fuse vs nofuse). String
// leaves currently bake their own per-leaf `string_op_options` on
// the Predicate; that is a separate, finer-grained concern.
table_ptr filter(gamax::table_view in,
                 gamax::select::Predicate& pred,
                 const Context& ctx,
                 const gamax::select::select_options& hint = {});

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
                 const Context& ctx,
                 const gamax::select::select_options& hint = {});

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
                  const Context& ctx,
                  const gamax::join::join_hint& hint = default_eq_join_hint());

// Left semi-join. Same key-list semantics as `eq_join`; output is
// `[shared_key, L_rest...]`.
table_ptr semi_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                    gamax::table_view R, std::vector<gamax::size_type> rkeys,
                    const Context& ctx,
                    const gamax::join::join_hint& hint = default_eq_join_hint());

// ── Joins declared but not yet implemented at the backend ─────────
//
// Surface is complete so the lowering pass can call them uniformly;
// bodies throw `"phys::<name>: not yet implemented"` until the
// physical operator lands.
table_ptr left_outer_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                          gamax::table_view R, std::vector<gamax::size_type> rkeys,
                          const Context& ctx,
                          const gamax::join::join_hint& hint = default_eq_join_hint());
table_ptr anti_join(gamax::table_view L, std::vector<gamax::size_type> lkeys,
                    gamax::table_view R, std::vector<gamax::size_type> rkeys,
                    const Context& ctx,
                    const gamax::join::join_hint& hint = default_eq_join_hint());
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
                   const Context& ctx,
                   const gamax::groupby::groupby_hint& hint = {});

// Distinct over all of `keys`' columns. One row per unique key
// combination. Numeric, STRING, and multi-key inputs are all handled.
// Output column order matches input. Output row order is insertion
// order (first-encounter) — callers that depend on a specific order
// must sort downstream.
table_ptr distinct(gamax::table_view keys,
                   const Context& ctx);

// Vertical concatenation (UNION ALL). All inputs must have identical
// column counts and positionally matching types (caller ensures at
// lowering time; asserts here). Output row count = sum of input row
// counts; output column order matches inputs[0]. Numeric columns are
// produced by a direct memcpy per input; STRING columns get the byte
// buffer concatenated with a per-input offset shift applied to the
// offsets array.
table_ptr concat(std::vector<gamax::table_view> inputs,
                 const Context& ctx);

// ── Ordering ──────────────────────────────────────────────────────

table_ptr sort(gamax::table_view in,
               std::vector<SortSpec> keys,
               const Context& ctx,
               const gamax::sort::sort_table_options& hint = {});

// Top-K: fused sort + limit. `result_in_sorted_order=true` matches
// the existing `sort::topk_table` convention.
table_ptr top_k(gamax::table_view in,
                std::vector<SortSpec> keys,
                gamax::size_type k,
                const Context& ctx,
                const gamax::sort::sort_table_options& hint = {});

} // namespace gamax::phys
