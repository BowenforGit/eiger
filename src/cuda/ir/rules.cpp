#include "ir/rules.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "ir/expr.hpp"

namespace gamax::ir {

namespace {

// ── G4a: CountAll / Count → Sum(lit(int64_t{1})) ──────────────────
//
// The Aggregate node's output_schema already has INT64 for every
// count item (see `aggregate` factory: Count / CountAll /
// CountDistinct all set `f.type = INT64`). `Sum(lit(int64_t{1}))`
// also produces INT64 (`aggregate_output_type(Sum, INT64) = INT64`),
// so the schema survives unchanged across the rewrite.
//
// Nullability: the factory marked Count fields `nullable=false`;
// Sum is normally marked nullable (per factory's "SUM over empty
// group is NULL" comment). We deliberately DON'T touch the schema
// here — downstream consumers read the Value's schema from the
// ExecPlan lowering, and the unrewritten Field's `nullable=false`
// is still correct (Sum of 1 over any group size >= 1 is non-null;
// empty-group handling is a separate semantic concern the backend
// doesn't currently observe anyway).
//
// `CountDistinct` is intentionally NOT rewritten — its proper
// rewrite is `Aggregate(keys=prev_keys ∪ body_col, aggs=[])` then
// `Aggregate(keys=prev_keys, aggs=[Count(*)])`, which requires
// introducing a Distinct-style dedup. Left as a follow-up.
PlanPtr rule_count_to_sum(PlanPtr node) {
    if (node->kind != NodeKind::Aggregate) return node;

    bool any_count = false;
    for (const auto& ag : node->agg_items) {
        if (ag.agg == AggKind::Count || ag.agg == AggKind::CountAll) {
            any_count = true;
            break;
        }
    }
    if (!any_count) return node;

    // Copy-on-write: shallow-copy the node, then rewrite agg_items.
    auto result = std::make_shared<PlanNode>(*node);
    for (auto& ag : result->agg_items) {
        if (ag.agg == AggKind::Count || ag.agg == AggKind::CountAll) {
            ag.agg = AggKind::Sum;
            ag.body = lit(int64_t{1});
        }
    }
    return result;
}

// ── G7: split non-equi clauses out of Inner join conditions ──────

// Walk an AND-chain, appending leaf clauses to `out`. Non-AND roots
// yield a single element (the expression itself). Preserves order.
void split_conjunction(const Expr& e, std::vector<Expr>& out) {
    if (e.kind == ExprKind::BinLogical && e.logical_op == LogicalOp::And) {
        split_conjunction(e.children[0], out);
        split_conjunction(e.children[1], out);
        return;
    }
    out.push_back(e);  // copy
}

// An equi-join clause is `ColRef == ColRef` with one side in L's
// schema and the other in R's. Everything else is residual.
bool is_equi_colref_pair(
    const Expr& e, const Schema& L, const Schema& R)
{
    if (e.kind != ExprKind::BinCompare || e.compare_op != CompareOp::Eq)
        return false;
    const auto& a = e.children[0];
    const auto& b = e.children[1];
    if (a.kind != ExprKind::ColRef || b.kind != ExprKind::ColRef)
        return false;
    const bool a_l = L.index_of(a.col_ref).has_value();
    const bool a_r = R.index_of(a.col_ref).has_value();
    const bool b_l = L.index_of(b.col_ref).has_value();
    const bool b_r = R.index_of(b.col_ref).has_value();
    return (a_l && b_r) || (a_r && b_l);
}

Expr conjoin(std::vector<Expr> parts) {
    Expr acc = std::move(parts.front());
    for (size_t i = 1; i < parts.size(); ++i) {
        acc = std::move(acc) && std::move(parts[i]);
    }
    return acc;
}

PlanPtr rule_split_non_equi_join(PlanPtr node) {
    if (node->kind != NodeKind::Join) return node;
    // Only Inner: residual-as-post-filter preserves semantics only
    // for inner joins. Semi/Outer/Anti would need a different
    // rewrite (push residual INTO the match predicate), which is
    // separate work.
    if (node->join_kind != JoinKind::Inner) return node;
    if (!node->join_condition) return node;  // Cross has no condition

    const auto& L = node->inputs[0]->output_schema;
    const auto& R = node->inputs[1]->output_schema;

    std::vector<Expr> conjuncts;
    split_conjunction(*node->join_condition, conjuncts);

    std::vector<Expr> equi, residual;
    equi.reserve(conjuncts.size());
    residual.reserve(conjuncts.size());
    for (auto& c : conjuncts) {
        if (is_equi_colref_pair(c, L, R)) equi.push_back(std::move(c));
        else                               residual.push_back(std::move(c));
    }
    if (residual.empty()) return node;   // nothing to split
    if (equi.empty())     return node;   // would turn inner join into
                                         // cross join; defer to a
                                         // dedicated rule if needed

    // Clone join with shrunken condition; wrap in a Filter.
    auto new_join = std::make_shared<PlanNode>(*node);
    new_join->join_condition = conjoin(std::move(equi));
    return filter(new_join, conjoin(std::move(residual)));
}

// ── G1-rule: split multi-key equi-join into single-key + filter ──
//
// Strictly weaker than a real multi-key backend — the second (and
// subsequent) equi clauses are enforced via a post-join bitmask
// filter rather than as part of the hash/merge key — but it turns
// every TPC-H multi-key inner join into something `phys::eq_join`
// can already handle.
//
// Composes cleanly with G7: G7 runs first and lifts non-equi
// residuals into a Filter; this rule then inspects the shrunken
// condition. If it's still a conjunction of 2+ equi clauses, the
// rule keeps the first as the join's `on` and lifts the rest into
// their own Filter. The caller ends up with two sibling Filters
// over the Join's output (one for non-equi residual, one for
// surplus equi keys); both are structurally the same and the
// lowering treats them identically.
PlanPtr rule_split_multi_key_equi_join(PlanPtr node) {
    if (node->kind != NodeKind::Join) return node;
    if (node->join_kind != JoinKind::Inner) return node;
    if (!node->join_condition) return node;

    const auto& L = node->inputs[0]->output_schema;
    const auto& R = node->inputs[1]->output_schema;

    std::vector<Expr> conjuncts;
    split_conjunction(*node->join_condition, conjuncts);

    // If any clause isn't pure equi, G7 either didn't run or didn't
    // apply cleanly; stay out of the way rather than invent a split.
    for (const auto& c : conjuncts) {
        if (!is_equi_colref_pair(c, L, R)) return node;
    }
    if (conjuncts.size() < 2) return node;

    // Keep the first clause on the join, conjoin the rest into a Filter.
    // (First-clause selection is deterministic but arbitrary; a real
    // optimizer would pick based on selectivity. Deterministic suffices
    // for TPC-H today.)
    Expr first = std::move(conjuncts.front());
    std::vector<Expr> rest(
        std::make_move_iterator(conjuncts.begin() + 1),
        std::make_move_iterator(conjuncts.end()));

    auto new_join = std::make_shared<PlanNode>(*node);
    new_join->join_condition = std::move(first);
    return filter(new_join, conjoin(std::move(rest)));
}

// ── Engine ────────────────────────────────────────────────────────

// Apply every enabled rule to `node` in sequence. Rules are
// independent (target disjoint NodeKinds), so any order works; we
// fix the order here for determinism across builds.
PlanPtr apply_local_rules(PlanPtr node, const RuleConfig& cfg) {
    if (cfg.rewrite_count_to_sum)            node = rule_count_to_sum(node);
    // Order is significant here: G7 runs BEFORE the multi-key split so
    // that any non-equi residual is lifted out first. The multi-key
    // rule only fires on conditions that are already a pure conjunction
    // of equi clauses; G7's output satisfies that precondition.
    if (cfg.split_non_equi_join_conditions)  node = rule_split_non_equi_join(node);
    if (cfg.split_multi_key_equi_join)       node = rule_split_multi_key_equi_join(node);
    return node;
}

// Shallow-copy the node with a fresh inputs vector. Used when a
// child was rewritten and we therefore need this parent to point
// at the new child without mutating the original.
PlanPtr clone_with_new_inputs(
    const PlanPtr& orig, std::vector<PlanPtr> new_inputs)
{
    auto node = std::make_shared<PlanNode>(*orig);
    node->inputs = std::move(new_inputs);
    return node;
}

class RuleEngine {
public:
    explicit RuleEngine(const RuleConfig& cfg) : _cfg(cfg) {}

    PlanPtr rewrite(const PlanPtr& node) {
        if (auto it = _memo.find(node.get()); it != _memo.end()) {
            return it->second;
        }
        // Rewrite children bottom-up.
        std::vector<PlanPtr> new_inputs;
        new_inputs.reserve(node->inputs.size());
        bool children_changed = false;
        for (const auto& in : node->inputs) {
            auto new_in = rewrite(in);
            if (new_in.get() != in.get()) children_changed = true;
            new_inputs.push_back(std::move(new_in));
        }
        PlanPtr current = children_changed
            ? clone_with_new_inputs(node, std::move(new_inputs))
            : node;
        current = apply_local_rules(std::move(current), _cfg);
        _memo.emplace(node.get(), current);
        return current;
    }

private:
    const RuleConfig& _cfg;
    std::unordered_map<const PlanNode*, PlanPtr> _memo;
};

} // namespace

PlanPtr apply_rules(const PlanPtr& root, const RuleConfig& cfg) {
    if (!root) return root;
    // Fixpoint over single-pass DAG walks.
    //
    // A single pass rewrites each original node once, which is fine
    // for rules whose output kind matches their input kind (like G4a:
    // Aggregate → Aggregate). But G7 and the multi-key-split rule
    // produce a Filter wrapping a freshly-created Join — and that new
    // Join isn't in the engine's memo, so rules targeting Join miss
    // it on this pass. A second pass walks the rewritten tree afresh
    // and catches the new Join.
    //
    // Each pass builds a new engine (fresh memo) over the previous
    // pass's output. Iteration stops when a pass returns a pointer-
    // identical tree (no rule fired anywhere). The bound guards
    // against a malformed rule that keeps producing new shapes; our
    // three rules are all strictly simplifying (residual-count
    // decreasing), so 2–3 passes suffice in practice.
    constexpr int kMaxPasses = 8;
    PlanPtr current = root;
    for (int pass = 0; pass < kMaxPasses; ++pass) {
        RuleEngine engine(cfg);
        PlanPtr next = engine.rewrite(current);
        if (next.get() == current.get()) return next;
        current = next;
    }
    throw std::runtime_error(
        "ir::apply_rules: did not converge within " +
        std::to_string(kMaxPasses) + " passes");
}

} // namespace gamax::ir
