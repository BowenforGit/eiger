#include <gtest/gtest.h>

#include "exec/executor.hpp"
#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/lower_to_exec_plan.hpp"
#include "ir/plan_node.hpp"
#include "ir/rules.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "phys/phys.hpp"

namespace gir = gamax::ir;
namespace gex = gamax::exec;
using gamax::data_type;
using gamax::type_id;

namespace {

gir::Catalog make_catalog() {
    gir::Catalog cat;
    gir::CatalogTable t;
    t.name = "orders";
    t.column_names = {"o_orderkey", "o_custkey", "o_orderdate"};
    t.column_types = {
        data_type{type_id::INT64},
        data_type{type_id::INT64},
        data_type{type_id::INT32},
    };
    t.column_nullable = {false, false, false};
    cat.add_table(std::move(t));

    gir::CatalogTable l;
    l.name = "lineitem";
    l.column_names = {"l_orderkey", "l_suppkey", "l_quantity"};
    l.column_types = {
        data_type{type_id::INT64},
        data_type{type_id::INT64},
        data_type{type_id::FLOAT64},
    };
    l.column_nullable = {false, false, false};
    cat.add_table(std::move(l));

    return cat;
}

} // namespace

// ── G4a: CountAll / Count → Sum(lit(1)) ──────────────────────────

TEST(RulesCountToSum, RewritesCountAllWhenEnabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::CountAll, gir::lit(int64_t{0})}}).plan();

    auto rewritten = gir::apply_rules(plan);
    ASSERT_EQ(rewritten->kind, gir::NodeKind::Aggregate);
    ASSERT_EQ(rewritten->agg_items.size(), 1u);
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::Sum);
    EXPECT_EQ(rewritten->agg_items[0].body.kind, gir::ExprKind::Literal);
    EXPECT_EQ(rewritten->agg_items[0].body.type.id(), type_id::INT64);
}

TEST(RulesCountToSum, RewritesCountWhenEnabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::Count, o.c("o_orderkey")}}).plan();

    auto rewritten = gir::apply_rules(plan);
    ASSERT_EQ(rewritten->agg_items.size(), 1u);
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::Sum);
    EXPECT_EQ(rewritten->agg_items[0].body.kind, gir::ExprKind::Literal);
}

TEST(RulesCountToSum, NoOpWhenDisabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::CountAll, gir::lit(int64_t{0})}}).plan();

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    // No rule applied → identical pointer (DAG-sharing invariant).
    EXPECT_EQ(rewritten.get(), plan.get());
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::CountAll);
}

TEST(RulesCountToSum, LeavesCountDistinctAlone) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::CountDistinct, o.c("o_orderkey")}}).plan();

    auto rewritten = gir::apply_rules(plan);
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::CountDistinct);
}

TEST(RulesCountToSum, PreservesSumItems) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto l = qb.scan("lineitem", {"l_orderkey", "l_quantity"});
    auto plan = l.aggregate(
        {"l_orderkey"},
        {
            {"sq",    gir::AggKind::Sum,       l.c("l_quantity")},
            {"count", gir::AggKind::CountAll,  gir::lit(int64_t{0})},
        }).plan();

    auto rewritten = gir::apply_rules(plan);
    // Sum is preserved by identity; CountAll is rewritten.
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::Sum);
    EXPECT_EQ(rewritten->agg_items[0].body.kind, gir::ExprKind::ColRef);
    EXPECT_EQ(rewritten->agg_items[1].agg, gir::AggKind::Sum);
    EXPECT_EQ(rewritten->agg_items[1].body.kind, gir::ExprKind::Literal);
}

TEST(RulesCountToSum, EndToEndCountsOrders) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        /*keys=*/{},
        {{"n", gir::AggKind::CountAll, gir::lit(int64_t{0})}}).plan();

    // Fixture table with 4 rows.
    std::vector<int64_t> ok   = {1, 2, 3, 4};
    std::vector<int64_t> ck   = {10, 10, 20, 20};
    std::vector<int32_t> date = {9000, 9100, 9200, 9300};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(gamax::make_column_from_host(
        ok.data(), static_cast<gamax::size_type>(ok.size())));
    cols.push_back(gamax::make_column_from_host(
        ck.data(), static_cast<gamax::size_type>(ck.size())));
    cols.push_back(gamax::make_column_from_host(
        date.data(), static_cast<gamax::size_type>(date.size())));
    auto tbl = gamax::make_table(std::move(cols));
    auto view = tbl->view();

    auto rewritten = gir::apply_rules(plan);
    auto ep = gir::lower_to_exec_plan(rewritten, cat);
    gamax::phys::Context ctx{cudaStreamDefault};
    gex::SourceResolver resolver =
        [&view](const std::string&) { return view; };
    auto out = gex::execute(ep, resolver, ctx);
    ASSERT_EQ(out->num_rows(), 1);
    ASSERT_EQ(out->num_columns(), 1);
    auto h = gamax::copy_column_to_host<int64_t>(out->view()[0]);
    EXPECT_EQ(h[0], 4);
}

// ── G7: split non-equi join conditions ───────────────────────────

TEST(RulesSplitNonEquiJoin, WrapsJoinInFilterForInnerWithResidual) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey", "l_quantity"});
    // Inner join with equi prefix + residual numeric condition.
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_orderdate") > gir::lit(9100))).plan();

    auto rewritten = gir::apply_rules(plan);
    // Root is now a Filter wrapping the Join.
    ASSERT_EQ(rewritten->kind, gir::NodeKind::Filter);
    ASSERT_EQ(rewritten->inputs.size(), 1u);
    ASSERT_EQ(rewritten->inputs[0]->kind, gir::NodeKind::Join);
    // The join's condition is the pure equi part.
    const auto& on = *rewritten->inputs[0]->join_condition;
    EXPECT_EQ(on.kind, gir::ExprKind::BinCompare);
    EXPECT_EQ(on.compare_op, gir::CompareOp::Eq);
}

TEST(RulesSplitNonEquiJoin, NoOpWhenDisabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey", "l_quantity"});
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_orderdate") > gir::lit(9100))).plan();

    gir::RuleConfig cfg;
    cfg.split_non_equi_join_conditions = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    EXPECT_EQ(rewritten.get(), plan.get());
    EXPECT_EQ(rewritten->kind, gir::NodeKind::Join);
}

TEST(RulesSplitNonEquiJoin, KeepsPureEquiUntouchedWhenOnlyG7Enabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  == l.c("l_suppkey"))).plan();  // two equi

    // Scope the assertion to G7 alone — with the multi-key rule off,
    // a pure-equi multi-key join is a no-op fixed point for G7.
    gir::RuleConfig cfg;
    cfg.split_multi_key_equi_join = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    EXPECT_EQ(rewritten->kind, gir::NodeKind::Join);
    EXPECT_EQ(rewritten.get(), plan.get());
}

TEST(RulesSplitNonEquiJoin, LeavesSemiJoinAlone) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.left_semi_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  != l.c("l_suppkey"))).plan();

    auto rewritten = gir::apply_rules(plan);
    // G7 is conservative: Semi residual semantics differ from Inner,
    // so the rule leaves Semi joins alone.
    EXPECT_EQ(rewritten->kind, gir::NodeKind::Join);
    EXPECT_EQ(rewritten.get(), plan.get());
}

TEST(RulesSplitNonEquiJoin, LeavesZeroEquiAlone) {
    // If there's no equi clause, splitting would produce an empty
    // join condition — we'd need a dedicated cross-join rewrite.
    // G7 stays conservative and leaves such joins alone.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.inner_join(l,
            o.c("o_orderkey") > l.c("l_orderkey")).plan();

    auto rewritten = gir::apply_rules(plan);
    EXPECT_EQ(rewritten->kind, gir::NodeKind::Join);
    EXPECT_EQ(rewritten.get(), plan.get());
}

TEST(RulesSplitNonEquiJoin, EndToEndMatchesHandFiltered) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey", "l_quantity"});
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_orderdate") >= gir::lit(9100))).plan();

    // Orders: 4 rows (keys 1..4, dates 9000/9100/9200/9300).
    std::vector<int64_t> o_ok   = {1, 2, 3, 4};
    std::vector<int64_t> o_ck   = {10, 10, 20, 20};
    std::vector<int32_t> o_date = {9000, 9100, 9200, 9300};
    std::vector<gamax::column_ptr> oc;
    oc.push_back(gamax::make_column_from_host(
        o_ok.data(), static_cast<gamax::size_type>(o_ok.size())));
    oc.push_back(gamax::make_column_from_host(
        o_ck.data(), static_cast<gamax::size_type>(o_ck.size())));
    oc.push_back(gamax::make_column_from_host(
        o_date.data(), static_cast<gamax::size_type>(o_date.size())));
    auto orders = gamax::make_table(std::move(oc));

    // Lineitem: 4 rows keyed 1..4.
    std::vector<int64_t> l_ok = {1, 2, 3, 4};
    std::vector<int64_t> l_sk = {100, 200, 300, 400};
    std::vector<double>  l_q  = {1.0, 2.0, 3.0, 4.0};
    std::vector<gamax::column_ptr> lc;
    lc.push_back(gamax::make_column_from_host(
        l_ok.data(), static_cast<gamax::size_type>(l_ok.size())));
    lc.push_back(gamax::make_column_from_host(
        l_sk.data(), static_cast<gamax::size_type>(l_sk.size())));
    lc.push_back(gamax::make_column_from_host(
        l_q.data(),  static_cast<gamax::size_type>(l_q.size())));
    auto lineitem = gamax::make_table(std::move(lc));

    auto o_view = orders->view();
    auto l_view = lineitem->view();
    gex::SourceResolver resolver = [&](const std::string& name) {
        if (name == "orders") return o_view;
        if (name == "lineitem") return l_view;
        throw std::runtime_error("unknown");
    };

    // With rules enabled, the non-equi clause moves into a Filter,
    // and the plan lowers and executes cleanly.
    auto rewritten = gir::apply_rules(plan);
    auto ep = gir::lower_to_exec_plan(rewritten, cat);
    gamax::phys::Context ctx{cudaStreamDefault};
    auto out = gex::execute(ep, resolver, ctx);
    // Only rows where o_orderdate >= 9100: orderkeys 2, 3, 4 → 3 rows.
    EXPECT_EQ(out->num_rows(), 3);
}

// ── G1-rule: multi-key equi → single-key + filter ───────────────

TEST(RulesSplitMultiKeyEqui, WrapsJoinInFilterForTwoEquiClauses) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  == l.c("l_suppkey"))).plan();

    auto rewritten = gir::apply_rules(plan);
    ASSERT_EQ(rewritten->kind, gir::NodeKind::Filter);
    ASSERT_EQ(rewritten->inputs[0]->kind, gir::NodeKind::Join);
    // Join's on-clause is now a single equi clause.
    const auto& on = *rewritten->inputs[0]->join_condition;
    EXPECT_EQ(on.kind, gir::ExprKind::BinCompare);
    EXPECT_EQ(on.compare_op, gir::CompareOp::Eq);
}

TEST(RulesSplitMultiKeyEqui, ReducesThreeEquiToSingleKeyPlusTwoFilters) {
    // Three-key inner join → single-key join + Filter(two equi clauses).
    // Verifies the rule collapses multi-key cleanly regardless of arity.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey", "l_quantity"});
    // Craft a 3-equi condition (contrived — types must match).
    auto plan = o.inner_join(l,
            (o.c("o_orderkey")  == l.c("l_orderkey"))
         && (o.c("o_custkey")   == l.c("l_suppkey"))
         && (o.c("o_orderdate") == l.c("l_quantity"))).plan();
    // (Typing note: equality between INT32 and FLOAT64 is legal in the
    // IR factory — it only requires mutual-comparability; the backend
    // would widen. We're only checking the rewriting machinery here.)

    auto rewritten = gir::apply_rules(plan);
    ASSERT_EQ(rewritten->kind, gir::NodeKind::Filter);
    // The Filter wraps a single-key Join.
    ASSERT_EQ(rewritten->inputs[0]->kind, gir::NodeKind::Join);
    const auto& on = *rewritten->inputs[0]->join_condition;
    EXPECT_EQ(on.kind, gir::ExprKind::BinCompare);
    EXPECT_EQ(on.compare_op, gir::CompareOp::Eq);
    // The Filter's predicate is the conjunction of the other two.
    const auto& pred = rewritten->filter_predicate;
    EXPECT_EQ(pred.kind, gir::ExprKind::BinLogical);
    EXPECT_EQ(pred.logical_op, gir::LogicalOp::And);
}

TEST(RulesSplitMultiKeyEqui, NoOpWhenDisabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  == l.c("l_suppkey"))).plan();

    gir::RuleConfig cfg;
    cfg.split_multi_key_equi_join = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    EXPECT_EQ(rewritten.get(), plan.get());
    EXPECT_EQ(rewritten->kind, gir::NodeKind::Join);
}

TEST(RulesSplitMultiKeyEqui, LeavesSingleKeyJoinAlone) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.inner_join(l,
            o.c("o_orderkey") == l.c("l_orderkey")).plan();

    auto rewritten = gir::apply_rules(plan);
    EXPECT_EQ(rewritten.get(), plan.get());
}

TEST(RulesSplitMultiKeyEqui, LeavesSemiJoinAlone) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.left_semi_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  == l.c("l_suppkey"))).plan();

    auto rewritten = gir::apply_rules(plan);
    // Multi-key semi residual isn't semantics-preserving post-join.
    EXPECT_EQ(rewritten.get(), plan.get());
}

TEST(RulesSplitMultiKeyEqui, ComposesWithG7) {
    // An Inner join with equi + non-equi residual + multi-key equi:
    //   (ok == ok) && (ck == sk) && (od > lit(9100))
    // G7 strips the non-equi residual:
    //   Filter(od > 9100, Join(on = (ok==ok) && (ck==sk)))
    // G1-rule then reduces multi-key equi:
    //   Filter(od > 9100, Filter(ck==sk, Join(on = ok==ok)))
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey", "l_quantity"});
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  == l.c("l_suppkey"))
         && (o.c("o_orderdate") > gir::lit(9100))).plan();

    auto rewritten = gir::apply_rules(plan);
    ASSERT_EQ(rewritten->kind, gir::NodeKind::Filter);
    ASSERT_EQ(rewritten->inputs[0]->kind, gir::NodeKind::Filter);
    ASSERT_EQ(rewritten->inputs[0]->inputs[0]->kind, gir::NodeKind::Join);
    const auto& join = *rewritten->inputs[0]->inputs[0];
    const auto& on = *join.join_condition;
    EXPECT_EQ(on.kind, gir::ExprKind::BinCompare);  // single equi
}

TEST(RulesSplitMultiKeyEqui, EndToEndTwoKeyInnerJoin) {
    // Fixture matching a two-key join: orders has 4 rows, lineitem
    // has 4 rows, exactly one row-pair satisfies BOTH equi clauses.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey"});
    auto plan = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  == l.c("l_suppkey"))).plan();

    std::vector<int64_t> o_ok = {1, 2, 3, 4};
    std::vector<int64_t> o_ck = {10, 20, 30, 40};
    std::vector<gamax::column_ptr> oc;
    oc.push_back(gamax::make_column_from_host(
        o_ok.data(), static_cast<gamax::size_type>(o_ok.size())));
    oc.push_back(gamax::make_column_from_host(
        o_ck.data(), static_cast<gamax::size_type>(o_ck.size())));
    auto orders = gamax::make_table(std::move(oc));

    // Lineitem with matching orderkeys but only one matching suppkey.
    std::vector<int64_t> l_ok = {1, 2, 3, 4};
    std::vector<int64_t> l_sk = {99, 20, 88, 77};   // row 2 matches both
    std::vector<gamax::column_ptr> lc;
    lc.push_back(gamax::make_column_from_host(
        l_ok.data(), static_cast<gamax::size_type>(l_ok.size())));
    lc.push_back(gamax::make_column_from_host(
        l_sk.data(), static_cast<gamax::size_type>(l_sk.size())));
    auto lineitem = gamax::make_table(std::move(lc));

    auto o_view = orders->view();
    auto l_view = lineitem->view();
    gex::SourceResolver resolver = [&](const std::string& name) {
        if (name == "orders") return o_view;
        if (name == "lineitem") return l_view;
        throw std::runtime_error("unknown");
    };

    // Without any rule, lowering throws on multi-key.
    {
        gir::RuleConfig cfg;
        cfg.split_multi_key_equi_join = false;
        cfg.split_non_equi_join_conditions = false;  // irrelevant here
        auto rewritten = gir::apply_rules(plan, cfg);
        EXPECT_THROW(gir::lower_to_exec_plan(rewritten, cat),
                     std::runtime_error);
    }
    // With the rule on, lowering succeeds and execution produces the
    // single matching pair (ok=2, ck/sk=20).
    {
        auto rewritten = gir::apply_rules(plan);
        auto ep = gir::lower_to_exec_plan(rewritten, cat);
        gamax::phys::Context ctx{cudaStreamDefault};
        auto out = gex::execute(ep, resolver, ctx);
        EXPECT_EQ(out->num_rows(), 1);
    }
}

TEST(RulesEngine, HonorsDagSharing) {
    // A node referenced by two parents must be rewritten once; the
    // rewritten pointer appears at both references in the output.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    // Aggregate referencing the same Scan twice via self-join:
    auto plan = o.inner_join(o,
            (o.c("o_orderkey") == o.c("o_orderkey"))).plan();
    // inner_join validates that the on-clause is in the combined
    // schema (which treats both sides' ids the same since it's a
    // self-join), so the condition above is well-formed.

    // Rules don't touch Scan / Join-with-pure-equi in this shape;
    // the only question is whether the shared Scan DAG survives.
    auto rewritten = gir::apply_rules(plan);
    EXPECT_EQ(rewritten->inputs[0].get(), rewritten->inputs[1].get());
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
