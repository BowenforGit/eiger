#pragma once

#include "ir/plan_node.hpp"

namespace gamax::ir {

// Pluggable rule engine for IR → IR rewrites.
//
// Rules operate on a plan DAG before it enters `lower_to_exec_plan`.
// They are the designated place for *semantic* rewrites — COUNT → SUM(1),
// StringOp::In → disjunction, non-equi-in-join → equi-prefix + filter,
// etc. — as distinguished from the *implementation-detail* work that
// happens inside `phys::*` (STRING passthrough, multi-key fusion).
//
// Each rule is opt-in via `RuleConfig`. Default-constructing the
// config enables every rule we currently ship; callers that want a
// specific shape (for debugging, for testing a particular IR form,
// or to benchmark rule impact) flip individual fields off.
//
// Why a config struct instead of a rule-list: the rules we care about
// for TPC-H are few and well-known, so a named boolean per rule is
// clearer to read at a call site and at a test than a vector of
// opaque rule objects. When we grow beyond ~10 rules we can promote
// this to a registry.
//
// Each rule is also expected to be **idempotent**: running it twice
// produces the same result as running it once. The engine therefore
// runs all enabled rules in a single bottom-up pass over the DAG
// rather than iterating to a fixpoint. If a future rule requires
// iteration (e.g. constant-folding chains), add a `fixpoint()` wrapper
// rather than changing the single-pass default.
struct RuleConfig {
    // G4a. Rewrite `Aggregate` items whose kind is `Count` or
    // `CountAll` into `Sum(lit(int64_t{1}))`. Now disabled by
    // default — the phys agg layer handles
    // `aggregation_id::COUNT_ALL` natively (via the same internal
    // ones-broadcast it already builds for AVG's denominator), so
    // there's no need to materialize a full-cardinality lit(1)
    // column. Lit-1 materialization at SF=100 was a 4.5 GiB
    // allocation that pushed Q1 over the slab budget. Leaving the
    // rule defined (just unhooked) so that it can still be enabled
    // via RuleConfig if someone wants the old behavior — e.g. for
    // null-aware semantics where `Count(x)` and `CountAll` need to
    // diverge.
    //
    // CountDistinct is still rewritten by `rewrite_count_distinct`
    // below, which emits a `CountAll` on top of a `Distinct` subtree.
    // That CountAll now flows directly through to the phys layer's
    // native COUNT_ALL handler instead of being further rewritten
    // by G4a.
    bool rewrite_count_to_sum {false};

    // Rewrite a single-item Aggregate with `CountDistinct` into
    // Aggregate(keys=K, aggs=[CountAll], input=Distinct(K∪{x})).
    //
    // Scope is deliberately narrow:
    //   * exactly one aggregate item (mixed agg sets with CountDistinct
    //     are not expressible via this rewrite),
    //   * the aggregate kind is CountDistinct,
    //   * the body is a plain ColRef (no computed expressions).
    //
    // Anything outside that scope falls through and the CountDistinct
    // reaches lowering, where it throws with a pointer back at this
    // rule's scope. TPC-H Q16 is the only query this targets today.
    //
    // The output ColumnId of the original CountDistinct is preserved
    // on the outer CountAll's output field so ancestor nodes (Sort,
    // Project) continue to resolve their refs.
    bool rewrite_count_distinct {true};

    // G7. For an `Inner` join whose condition is a conjunction
    // including at least one non-equi clause (e.g. `a == b && c > d`,
    // or a complex predicate over a ColRef pair), split the
    // conjunction into:
    //   * the equi part (ColRef == ColRef, one side each) — stays on
    //     the join's `on` clause.
    //   * the residual — becomes a `Filter` wrapped around the join.
    //
    // Restricted to `Inner` joins because residual-post-filter is
    // only semantics-preserving for inner joins. Semi/Outer/Anti
    // residuals affect match semantics, not the output rows, and
    // need a different treatment (push residual into the match
    // predicate; tracked as a follow-up rule).
    //
    // Does NOT split equi clauses against each other — a join whose
    // condition is `k1==k1' && k2==k2'` stays as a multi-key equi
    // condition. Making multi-key equi joins actually run is a
    // separate concern handled by `split_multi_key_equi_join` below
    // (or eventually by a real multi-key backend via `multikey_compress`
    // at the `phys::eq_join` layer — whichever lands first).
    bool split_non_equi_join_conditions {true};

    // G1-rule. For an `Inner` join whose `on` clause is a conjunction
    // of two or more pure-equi clauses, retain the first clause as
    // the join's `on` and move the remaining equi clauses into a
    // post-join `Filter`. Reduces every join to single-key at the
    // IR layer, so `phys::eq_join`'s current single-key limitation
    // stops being a blocker. The residual Filter re-checks the
    // surplus equalities; under `lower_filter`'s existing column-
    // pruning logic it compiles down to a `select_to_bitmask` over
    // a trivial `k2 == k2'` expression.
    //
    // Runs AFTER `split_non_equi_join_conditions` in the pass, so
    // by the time this rule inspects the condition, any non-equi
    // residual has already been lifted out. A condition that still
    // contains non-equi clauses after that stage is skipped (defensive
    // — shouldn't happen if G7 was enabled).
    //
    // Restricted to Inner, same reasoning as G7: post-join filter on
    // extra keys is only semantics-preserving for Inner.
    //
    // A future real multi-key backend will make this rule redundant
    // but not incorrect; flipping the knob off at that point recovers
    // the native multi-key join.
    bool split_multi_key_equi_join {true};

    // Scan column pushdown. For every `Scan` node, consults liveness
    // over the DAG and removes any `scan_columns` entry whose output
    // ColumnId isn't consumed by any downstream operator. Narrows the
    // Scan's `output_schema` to match.
    //
    // The structural counterpart to the lowering-time
    // `prune_filter_materialization` knob: that one decides what a
    // Filter materializes; this one decides what a Scan loads in the
    // first place. Both use the same `compute_liveness` analysis.
    //
    // Ancestor `output_schema`s become formally stale after a Scan
    // shrinks (they still list columns no longer produced). This is
    // tolerated end-to-end: the lowering resolves every ordinal
    // through Value schemas (i.e., what's actually produced), never
    // through the cached IR schemas; `compute_liveness` itself is
    // ColumnId-keyed and uses `Schema::index_of` which returns nullopt
    // for absent ids, so stale ancestor schemas don't misroute
    // liveness either. Printers that show IR schemas may display
    // extra columns that no runtime step actually produces — worth
    // flagging if it surprises a reader.
    bool scan_column_pushdown {true};
};

// Apply all enabled rules to `root`, producing a new (possibly
// identical) plan DAG. The input is not mutated; nodes that don't
// change are shared by pointer with the output.
//
// DAG sharing is honored: a node reachable from multiple parents is
// rewritten exactly once.
PlanPtr apply_rules(const PlanPtr& root, const RuleConfig& cfg = {});

} // namespace gamax::ir
