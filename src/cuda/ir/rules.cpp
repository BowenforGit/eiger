#include "ir/rules.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "ir/expr.hpp"
#include "ir/liveness.hpp"

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

// ── CountDistinct → Distinct + CountAll ──────────────────────────
//
// Rewrite `Aggregate(keys=K, aggs=[CountDistinct(x) AS c])` into
//
//   Aggregate(keys=K, aggs=[CountAll AS c],
//             input=Distinct(keys=K∪{x}, input=orig))
//
// Ancestor nodes reference the CountDistinct's output ColumnId; the
// outer Aggregate must emit that same id on its CountAll output or
// lowering's ordinal-resolution breaks. `aggregate(...)`'s factory
// allocates a fresh id per aggregate, so we construct via the factory
// (to reuse its schema / validation machinery) and then overwrite the
// single output id in-place before returning.
//
// Under the narrow scope we fire on (single agg, CountDistinct,
// ColRef body), K∪{x} can always be expressed as a proper-subset
// Distinct — the outer Aggregate's groupby refers to K by id and
// CountAll needs x only to keep scan-pushdown from pruning it. The
// CountAll body is `lit(int64_t{1})` to match G4a's output shape (so
// G4a can pick it up on the same apply_rules pass and rewrite to
// Sum(lit(1)) without an extra fixpoint iteration).
//
// G4a's rewrite IS downstream of this rule in the same
// `apply_local_rules` sequence — we emit `CountAll`, G4a rewrites it
// to `Sum(lit(1))` one call later.
PlanPtr rule_count_distinct_to_distinct_plus_count(PlanPtr node) {
    if (node->kind != NodeKind::Aggregate) return node;
    // Narrow scope: single agg, CountDistinct, ColRef body.
    if (node->agg_items.size() != 1) return node;
    const auto& ag = node->agg_items[0];
    if (ag.agg != AggKind::CountDistinct) return node;
    if (ag.body.kind != ExprKind::ColRef) return node;

    const auto x_id = ag.body.col_ref;
    const auto preserved_id = node->output_schema.back().id;

    // Build K∪{x} — the Distinct dedup keys. Ordering doesn't matter
    // semantically (hash-based dedup); we put K first so the shape
    // reads naturally in --print-plan.
    std::vector<ColumnId> inner_keys = node->agg_keys;
    inner_keys.push_back(x_id);

    auto dedup = distinct(node->inputs[0], inner_keys);

    // Outer: Aggregate(K, [CountAll AS c]). CountAll body is
    // `lit(int64_t{1})` to match G4a's emitted shape.
    std::vector<AggregateItem> outer_aggs;
    outer_aggs.push_back(AggregateItem{
        ag.name, AggKind::CountAll, lit(int64_t{1})});
    auto outer = aggregate(dedup, node->agg_keys, std::move(outer_aggs));

    // Patch the aggregate output id to preserve the original
    // CountDistinct id. The factory already handles the keys (keys
    // carry input ids forward unchanged); only the agg output field
    // gets a fresh id from the IdGenerator, so we overwrite that one.
    outer->output_schema.back().id = preserved_id;
    return outer;
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

// ── Scan column pushdown ─────────────────────────────────────────
//
// Removes scan_columns entries whose output ColumnId is not live at
// this Scan's output (i.e., no ancestor references them). Narrows the
// Scan's output_schema to match. Does NOT rebuild ancestor
// output_schemas — callers tolerate stale upstream schemas (see the
// note on RuleConfig::scan_column_pushdown).
PlanPtr rule_scan_column_pushdown(
    PlanPtr node, const LivenessResult& liveness)
{
    if (node->kind != NodeKind::Scan) return node;

    // Liveness is keyed by the pass-start pointer. For a Scan that
    // hasn't been rewritten yet in this pass, this lookup hits. For
    // a Scan produced by an earlier rule in this pass the lookup
    // misses — return an empty set defensively, and we bail below.
    const auto& live = liveness.at(node.get());
    if (live.empty()) return node;  // unmapped (shouldn't happen for
                                    // reachable Scans) or nothing live

    std::vector<std::string> new_names;
    Schema new_schema;
    new_names.reserve(node->scan_columns.size());
    for (size_t i = 0; i < node->scan_columns.size(); ++i) {
        const auto& f = node->output_schema[i];
        if (live.count(f.id)) {
            new_names.push_back(node->scan_columns[i]);
            new_schema.append(f);
        }
    }
    if (new_schema.size() == node->scan_columns.size()) return node;

    // Scan factory doesn't allow zero columns. Bail if pruning would
    // erase everything — shouldn't happen for reachable Scans.
    if (new_schema.size() == 0) return node;

    auto result = std::make_shared<PlanNode>(*node);
    result->scan_columns = std::move(new_names);
    result->output_schema = std::move(new_schema);
    return result;
}

// ── Engine ────────────────────────────────────────────────────────

// Apply every enabled rule to `node` in sequence. Rules are
// independent (target disjoint NodeKinds), so any order works; we
// fix the order here for determinism across builds.
PlanPtr apply_local_rules(
    PlanPtr node, const RuleConfig& cfg, const LivenessResult& liveness)
{
    // Order is significant here: CountDistinct rewrite runs BEFORE
    // G4a so the CountAll it emits is picked up by G4a on the same
    // node in this sequence. Otherwise the fixpoint loop would still
    // converge, but in two passes instead of one.
    if (cfg.rewrite_count_distinct)          node = rule_count_distinct_to_distinct_plus_count(node);
    if (cfg.rewrite_count_to_sum)            node = rule_count_to_sum(node);
    // Order is significant here: G7 runs BEFORE the multi-key split so
    // that any non-equi residual is lifted out first. The multi-key
    // rule only fires on conditions that are already a pure conjunction
    // of equi clauses; G7's output satisfies that precondition.
    if (cfg.split_non_equi_join_conditions)  node = rule_split_non_equi_join(node);
    if (cfg.split_multi_key_equi_join)       node = rule_split_multi_key_equi_join(node);
    if (cfg.scan_column_pushdown)            node = rule_scan_column_pushdown(node, liveness);
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
    RuleEngine(const RuleConfig& cfg, const LivenessResult& liveness)
        : _cfg(cfg), _liveness(liveness) {}

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
        current = apply_local_rules(std::move(current), _cfg, _liveness);
        _memo.emplace(node.get(), current);
        return current;
    }

private:
    const RuleConfig& _cfg;
    const LivenessResult& _liveness;
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
        // Liveness is recomputed each pass: the tree can grow Filter
        // wrappers between passes (G7, G1-rule), and Scan pushdown
        // would incorrectly see live sets from the old tree if we
        // cached across passes.
        auto liveness = compute_liveness(*current);
        RuleEngine engine(cfg, liveness);
        PlanPtr next = engine.rewrite(current);
        if (next.get() == current.get()) return next;
        current = next;
    }
    throw std::runtime_error(
        "ir::apply_rules: did not converge within " +
        std::to_string(kMaxPasses) + " passes");
}

} // namespace gamax::ir
