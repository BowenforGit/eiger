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
    // `CountAll` into `Sum(lit(int64_t{1}))`. Semantically equivalent
    // under TPC-H's no-NULL assumption; the backend has no native
    // count aggregation, so this rule is what lets Q1/Q4/Q13/Q21/Q22
    // lower at all.
    //
    // Leaves `CountDistinct` alone — that needs a separate
    // `Distinct`-plus-count rewrite that's not yet wired.
    bool rewrite_count_to_sum {true};

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
};

// Apply all enabled rules to `root`, producing a new (possibly
// identical) plan DAG. The input is not mutated; nodes that don't
// change are shared by pointer with the output.
//
// DAG sharing is honored: a node reachable from multiple parents is
// rewritten exactly once.
PlanPtr apply_rules(const PlanPtr& root, const RuleConfig& cfg = {});

} // namespace gamax::ir
