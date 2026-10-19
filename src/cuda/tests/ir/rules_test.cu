// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

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

// Build a SourceResolver that narrows `view` (assumed to be laid out
// in the same column order as the catalog table named `table_name`)
// to the caller-requested subset, preserving order.
gex::SourceResolver make_resolver(
    const gir::Catalog& cat,
    const std::string& table_name,
    const gamax::table_view& view)
{
    return [&cat, table_name, view](
               const std::string& name,
               const std::vector<std::string>& cols) -> gamax::table_view {
        if (name != table_name) {
            throw std::runtime_error("unknown table: " + name);
        }
        const auto* ct = cat.find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto it = std::find(ct->column_names.begin(),
                                ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), it)));
        }
        return view.select_columns(ords);
    };
}

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

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = true;
    auto rewritten = gir::apply_rules(plan, cfg);
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

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = true;
    auto rewritten = gir::apply_rules(plan, cfg);
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

    // Disable every other rule so we can assert CountToSum's knob
    // in isolation — otherwise Scan-pushdown would rewrite the Scan
    // (o_orderkey is dead after this Aggregate), changing the root
    // pointer even when CountToSum itself doesn't fire.
    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = false;
    cfg.split_non_equi_join_conditions = false;
    cfg.split_multi_key_equi_join = false;
    cfg.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    EXPECT_EQ(rewritten.get(), plan.get());
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::CountAll);
}

TEST(RulesCountToSum, LeavesCountDistinctAloneWhenOnlyG4aIsEnabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::CountDistinct, o.c("o_orderkey")}}).plan();

    // Disable the CountDistinct-specific rewrite so we can assert G4a
    // (rewrite_count_to_sum) doesn't touch CountDistinct on its own —
    // that's the separate rewrite_count_distinct rule's job.
    gir::RuleConfig cfg;
    cfg.rewrite_count_distinct = false;
    cfg.split_non_equi_join_conditions = false;
    cfg.split_multi_key_equi_join = false;
    cfg.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    EXPECT_EQ(rewritten.get(), plan.get());
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

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = true;
    auto rewritten = gir::apply_rules(plan, cfg);
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
    cols.push_back(gamax::make_column_from_host(ok.data(), static_cast<gamax::size_type>(ok.size()), cudaStreamDefault));
    cols.push_back(gamax::make_column_from_host(ck.data(), static_cast<gamax::size_type>(ck.size()), cudaStreamDefault));
    cols.push_back(gamax::make_column_from_host(date.data(), static_cast<gamax::size_type>(date.size()), cudaStreamDefault));
    auto tbl = gamax::make_table(std::move(cols));
    auto view = tbl->view();

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = true;
    auto rewritten = gir::apply_rules(plan, cfg);
    auto ep = gir::lower_to_exec_plan(rewritten, cat);
    gamax::phys::Context ctx{cudaStreamDefault};
    auto resolver = make_resolver(cat, "orders", view);
    auto out = gex::execute(ep, resolver, ctx);
    ASSERT_EQ(out->num_rows(), 1);
    ASSERT_EQ(out->num_columns(), 1);
    auto h = gamax::copy_column_to_host<int64_t>(out->view()[0], cudaStreamDefault);
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
    oc.push_back(gamax::make_column_from_host(o_ok.data(), static_cast<gamax::size_type>(o_ok.size()), cudaStreamDefault));
    oc.push_back(gamax::make_column_from_host(o_ck.data(), static_cast<gamax::size_type>(o_ck.size()), cudaStreamDefault));
    oc.push_back(gamax::make_column_from_host(o_date.data(), static_cast<gamax::size_type>(o_date.size()), cudaStreamDefault));
    auto orders = gamax::make_table(std::move(oc));

    // Lineitem: 4 rows keyed 1..4.
    std::vector<int64_t> l_ok = {1, 2, 3, 4};
    std::vector<int64_t> l_sk = {100, 200, 300, 400};
    std::vector<double>  l_q  = {1.0, 2.0, 3.0, 4.0};
    std::vector<gamax::column_ptr> lc;
    lc.push_back(gamax::make_column_from_host(l_ok.data(), static_cast<gamax::size_type>(l_ok.size()), cudaStreamDefault));
    lc.push_back(gamax::make_column_from_host(l_sk.data(), static_cast<gamax::size_type>(l_sk.size()), cudaStreamDefault));
    lc.push_back(gamax::make_column_from_host(l_q.data(), static_cast<gamax::size_type>(l_q.size()), cudaStreamDefault));
    auto lineitem = gamax::make_table(std::move(lc));

    auto o_view = orders->view();
    auto l_view = lineitem->view();
    gex::SourceResolver resolver =
        [&](const std::string& name,
            const std::vector<std::string>& cols) -> gamax::table_view {
        const gamax::table_view* src = nullptr;
        if (name == "orders")         src = &o_view;
        else if (name == "lineitem")  src = &l_view;
        else throw std::runtime_error("unknown");
        const auto* ct = cat.find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto it = std::find(ct->column_names.begin(),
                                ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), it)));
        }
        return src->select_columns(ords);
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
    oc.push_back(gamax::make_column_from_host(o_ok.data(), static_cast<gamax::size_type>(o_ok.size()), cudaStreamDefault));
    oc.push_back(gamax::make_column_from_host(o_ck.data(), static_cast<gamax::size_type>(o_ck.size()), cudaStreamDefault));
    auto orders = gamax::make_table(std::move(oc));

    // Lineitem with matching orderkeys but only one matching suppkey.
    std::vector<int64_t> l_ok = {1, 2, 3, 4};
    std::vector<int64_t> l_sk = {99, 20, 88, 77};   // row 2 matches both
    std::vector<gamax::column_ptr> lc;
    lc.push_back(gamax::make_column_from_host(l_ok.data(), static_cast<gamax::size_type>(l_ok.size()), cudaStreamDefault));
    lc.push_back(gamax::make_column_from_host(l_sk.data(), static_cast<gamax::size_type>(l_sk.size()), cudaStreamDefault));
    auto lineitem = gamax::make_table(std::move(lc));

    auto o_view = orders->view();
    auto l_view = lineitem->view();
    gex::SourceResolver resolver =
        [&](const std::string& name,
            const std::vector<std::string>& cols) -> gamax::table_view {
        const gamax::table_view* src = nullptr;
        if (name == "orders")         src = &o_view;
        else if (name == "lineitem")  src = &l_view;
        else throw std::runtime_error("unknown");
        const auto* ct = cat.find(name);
        std::vector<gamax::size_type> ords;
        ords.reserve(cols.size());
        for (const auto& c : cols) {
            auto it = std::find(ct->column_names.begin(),
                                ct->column_names.end(), c);
            ords.push_back(static_cast<gamax::size_type>(
                std::distance(ct->column_names.begin(), it)));
        }
        return src->select_columns(ords);
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

// ── Scan column pushdown ─────────────────────────────────────────

TEST(RulesScanPushdown, ShrinksScanToLiveColumnsOnly) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    // Only o_orderkey is consumed downstream (filter + project).
    auto filtered = o.filter(o.c("o_orderkey") > gir::lit(int64_t{1}));
    auto projected = filtered.project({
        {"k", filtered.c("o_orderkey")},
    });

    auto rewritten = gir::apply_rules(projected.plan());

    // Walk down to the Scan in the rewritten tree.
    auto node = rewritten;
    while (node->kind != gir::NodeKind::Scan) {
        ASSERT_FALSE(node->inputs.empty());
        node = node->inputs[0];
    }
    EXPECT_EQ(node->scan_columns.size(), 1u);
    EXPECT_EQ(node->scan_columns[0], "o_orderkey");
}

TEST(RulesScanPushdown, PreservesPredicateOnlyColumns) {
    // Columns referenced only by a filter predicate are still live at
    // the Scan (liveness forces them) — they must stay in scan_columns.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto filtered = o.filter(o.c("o_orderdate") > gir::lit(9100));
    auto projected = filtered.project({
        {"k", filtered.c("o_orderkey")},
    });

    auto rewritten = gir::apply_rules(projected.plan());

    auto node = rewritten;
    while (node->kind != gir::NodeKind::Scan) node = node->inputs[0];

    // o_custkey is dead; o_orderkey and o_orderdate are live.
    EXPECT_EQ(node->scan_columns.size(), 2u);
    EXPECT_EQ(std::find(node->scan_columns.begin(), node->scan_columns.end(),
                        "o_custkey"),
              node->scan_columns.end());
}

TEST(RulesScanPushdown, NoOpWhenDisabled) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto plan = o.project({{"k", o.c("o_orderkey")}}).plan();

    gir::RuleConfig cfg;
    cfg.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, cfg);

    // With the rule off, the Scan retains all requested columns.
    auto node = rewritten;
    while (node->kind != gir::NodeKind::Scan) node = node->inputs[0];
    EXPECT_EQ(node->scan_columns.size(), 3u);
}

TEST(RulesScanPushdown, LeavesAllLiveScanAlone) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    // Project uses both → both live at Scan → no pruning.
    auto plan = o.project({
        {"a", o.c("o_orderkey")},
        {"b", o.c("o_custkey")},
    }).plan();

    auto rewritten = gir::apply_rules(plan);
    EXPECT_EQ(rewritten.get(), plan.get());
}

TEST(RulesScanPushdown, ComposesWithJoinSplit) {
    // Multi-key join with a dead payload column on the right side.
    // G1-rule splits the join into single-key + Filter; Scan-pushdown
    // removes the dead payload column from the right Scan.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto l = qb.scan("lineitem", {"l_orderkey", "l_suppkey", "l_quantity"});
    auto joined = o.inner_join(l,
            (o.c("o_orderkey") == l.c("l_orderkey"))
         && (o.c("o_custkey")  == l.c("l_suppkey")));
    // Project only references o_orderkey + l_suppkey → l_quantity is
    // dead after the join. Both join keys are live at the respective
    // Scans (join condition needs them).
    auto plan = joined.project({
        {"ok", joined.c("o_orderkey")},
        {"sk", joined.c("l_suppkey")},
    }).plan();

    auto rewritten = gir::apply_rules(plan);

    // Walk both Scan branches.
    std::vector<gir::PlanPtr> scans;
    std::function<void(const gir::PlanPtr&)> visit =
        [&](const gir::PlanPtr& n) {
            if (n->kind == gir::NodeKind::Scan) {
                scans.push_back(n);
                return;
            }
            for (const auto& c : n->inputs) visit(c);
        };
    visit(rewritten);
    ASSERT_EQ(scans.size(), 2u);

    for (const auto& s : scans) {
        if (s->scan_table == "lineitem") {
            EXPECT_EQ(std::find(s->scan_columns.begin(),
                                s->scan_columns.end(), "l_quantity"),
                      s->scan_columns.end());
        }
    }
}

TEST(RulesScanPushdown, EndToEndProducesCorrectResult) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto plan = o.filter(o.c("o_orderdate") > gir::lit(9100))
                  .project({{"ok", o.c("o_orderkey")}}).plan();

    std::vector<int64_t> ok  = {1, 2, 3, 4};
    std::vector<int64_t> ck  = {10, 20, 30, 40};
    std::vector<int32_t> dt  = {9000, 9100, 9200, 9300};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(gamax::make_column_from_host(ok.data(), static_cast<gamax::size_type>(ok.size()), cudaStreamDefault));
    cols.push_back(gamax::make_column_from_host(ck.data(), static_cast<gamax::size_type>(ck.size()), cudaStreamDefault));
    cols.push_back(gamax::make_column_from_host(dt.data(), static_cast<gamax::size_type>(dt.size()), cudaStreamDefault));
    auto tbl = gamax::make_table(std::move(cols));
    auto view = tbl->view();

    auto rewritten = gir::apply_rules(plan);
    auto ep = gir::lower_to_exec_plan(rewritten, cat);
    gamax::phys::Context ctx{cudaStreamDefault};
    auto resolver = make_resolver(cat, "orders", view);
    auto out = gex::execute(ep, resolver, ctx);
    // o_orderdate > 9100 → rows 3, 4 → o_orderkey = 3, 4.
    ASSERT_EQ(out->num_rows(), 2);
    ASSERT_EQ(out->num_columns(), 1);
    auto h = gamax::copy_column_to_host<int64_t>(out->view()[0], cudaStreamDefault);
    EXPECT_EQ(h[0], 3);
    EXPECT_EQ(h[1], 4);
}

// ── LowerOptions: prune_filter_materialization knob ──────────────

TEST(LowerOptions, PruneFilterMaterializationOnEmitsOutputOrdinals) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    // Predicate touches o_orderdate; only o_orderkey is live downstream.
    auto filtered = o.filter(o.c("o_orderdate") > gir::lit(9100));
    auto plan = filtered.project({{"k", filtered.c("o_orderkey")}}).plan();

    // Disable all rules so Scan pushdown doesn't hide the effect.
    gir::RuleConfig no_rules;
    no_rules.rewrite_count_to_sum = false;
    no_rules.split_non_equi_join_conditions = false;
    no_rules.split_multi_key_equi_join = false;
    no_rules.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, no_rules);

    gir::LowerOptions opts;
    opts.prune_filter_materialization = true;
    auto ep = gir::lower_to_exec_plan(rewritten, cat, opts);

    // Find the Filter step; expect non-empty output_ordinals.
    const gex::Step* fstep = nullptr;
    for (const auto& s : ep.steps) {
        if (s.kind == gex::StepKind::Filter) { fstep = &s; break; }
    }
    ASSERT_NE(fstep, nullptr);
    const auto& fp = std::get<gex::FilterPayload>(fstep->payload);
    EXPECT_FALSE(fp.output_ordinals.empty());
}

TEST(LowerOptions, PruneFilterMaterializationOffPreservesAllColumns) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto filtered = o.filter(o.c("o_orderdate") > gir::lit(9100));
    auto plan = filtered.project({{"k", filtered.c("o_orderkey")}}).plan();

    // Disable scan pushdown too so we can isolate the Filter behavior.
    gir::RuleConfig no_rules;
    no_rules.rewrite_count_to_sum = false;
    no_rules.split_non_equi_join_conditions = false;
    no_rules.split_multi_key_equi_join = false;
    no_rules.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, no_rules);

    gir::LowerOptions opts;
    opts.prune_filter_materialization = false;
    auto ep = gir::lower_to_exec_plan(rewritten, cat, opts);

    const gex::Step* fstep = nullptr;
    for (const auto& s : ep.steps) {
        if (s.kind == gex::StepKind::Filter) { fstep = &s; break; }
    }
    ASSERT_NE(fstep, nullptr);
    const auto& fp = std::get<gex::FilterPayload>(fstep->payload);
    EXPECT_TRUE(fp.output_ordinals.empty());
}

TEST(LowerOptions, BothKnobsOffStillProducesCorrectResult) {
    // Correctness check: turning every knob off must not break
    // execution — it should just produce a less-optimized ExecPlan.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});
    auto plan = o.filter(o.c("o_orderdate") > gir::lit(9100))
                  .project({{"k", o.c("o_orderkey")}}).plan();

    std::vector<int64_t> ok  = {1, 2, 3, 4};
    std::vector<int64_t> ck  = {10, 20, 30, 40};
    std::vector<int32_t> dt  = {9000, 9100, 9200, 9300};
    std::vector<gamax::column_ptr> cols;
    cols.push_back(gamax::make_column_from_host(ok.data(), static_cast<gamax::size_type>(ok.size()), cudaStreamDefault));
    cols.push_back(gamax::make_column_from_host(ck.data(), static_cast<gamax::size_type>(ck.size()), cudaStreamDefault));
    cols.push_back(gamax::make_column_from_host(dt.data(), static_cast<gamax::size_type>(dt.size()), cudaStreamDefault));
    auto tbl = gamax::make_table(std::move(cols));
    auto view = tbl->view();

    gir::RuleConfig no_rules;
    no_rules.rewrite_count_to_sum = false;
    no_rules.split_non_equi_join_conditions = false;
    no_rules.split_multi_key_equi_join = false;
    no_rules.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, no_rules);

    gir::LowerOptions opts;
    opts.prune_filter_materialization = false;
    auto ep = gir::lower_to_exec_plan(rewritten, cat, opts);

    gamax::phys::Context ctx{cudaStreamDefault};
    auto resolver = make_resolver(cat, "orders", view);
    auto out = gex::execute(ep, resolver, ctx);
    ASSERT_EQ(out->num_rows(), 2);
    auto h = gamax::copy_column_to_host<int64_t>(out->view()[0], cudaStreamDefault);
    EXPECT_EQ(h[0], 3);
    EXPECT_EQ(h[1], 4);
}

// ── rewrite_count_distinct ───────────────────────────────────────

TEST(RulesCountDistinct, RewritesSingleAggCountDistinctToDistinctPlusCount) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::CountDistinct, o.c("o_orderkey")}}).plan();

    // Isolate the CountDistinct rule so we can see its immediate
    // output shape (Aggregate(CountAll) over Distinct) without G4a's
    // subsequent CountAll → Sum(lit(1)) rewrite collapsing it.
    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = false;
    cfg.split_non_equi_join_conditions = false;
    cfg.split_multi_key_equi_join = false;
    cfg.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, cfg);

    ASSERT_EQ(rewritten->kind, gir::NodeKind::Aggregate);
    ASSERT_EQ(rewritten->agg_items.size(), 1u);
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::CountAll);
    ASSERT_EQ(rewritten->inputs.size(), 1u);
    EXPECT_EQ(rewritten->inputs[0]->kind, gir::NodeKind::Distinct);
    // Distinct's keys are K ∪ {x} = {o_custkey, o_orderkey}.
    EXPECT_EQ(rewritten->inputs[0]->distinct_keys.size(), 2u);
}

TEST(RulesCountDistinct, PreservesOriginalAggregateOutputId) {
    // The outer CountAll must emit the original CountDistinct's
    // ColumnId so ancestor nodes (Sort, Project) still resolve.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto orig = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::CountDistinct, o.c("o_orderkey")}}).plan();
    const auto orig_agg_id = orig->output_schema.back().id;

    auto rewritten = gir::apply_rules(orig);
    EXPECT_EQ(rewritten->output_schema.back().id, orig_agg_id);
}

TEST(RulesCountDistinct, ComposesWithG4aInSinglePass) {
    // CountDistinct rule emits CountAll; G4a then rewrites CountAll
    // → Sum(lit(1)) on the same apply_local_rules call.
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto o = qb.scan("orders", {"o_orderkey", "o_custkey"});
    auto plan = o.aggregate(
        {"o_custkey"},
        {{"n", gir::AggKind::CountDistinct, o.c("o_orderkey")}}).plan();

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = true;
    auto rewritten = gir::apply_rules(plan, cfg);
    ASSERT_EQ(rewritten->kind, gir::NodeKind::Aggregate);
    ASSERT_EQ(rewritten->agg_items.size(), 1u);
    // G4a must have turned the emitted CountAll into Sum(lit(1)).
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::Sum);
}

TEST(RulesCountDistinct, LeavesMultiAggAggregateAlone) {
    // Narrow scope: only single-agg CountDistinct fires. Mixed agg
    // sets pass through untouched and will fail at lowering (tested
    // separately in test_emit / run_tpch_ir error paths).
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto l = qb.scan("lineitem", {"l_orderkey", "l_quantity"});
    auto plan = l.aggregate(
        {"l_orderkey"},
        {
            {"n",  gir::AggKind::CountDistinct, l.c("l_quantity")},
            {"sq", gir::AggKind::Sum,           l.c("l_quantity")},
        }).plan();

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = false;
    cfg.split_non_equi_join_conditions = false;
    cfg.split_multi_key_equi_join = false;
    cfg.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    EXPECT_EQ(rewritten.get(), plan.get());
    ASSERT_EQ(rewritten->agg_items.size(), 2u);
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::CountDistinct);
    EXPECT_EQ(rewritten->agg_items[1].agg, gir::AggKind::Sum);
}

TEST(RulesCountDistinct, LeavesNonColRefBodyAlone) {
    auto cat = make_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto l = qb.scan("lineitem", {"l_orderkey", "l_quantity"});
    // CountDistinct over a computed expression — not a plain ColRef.
    auto body = l.c("l_quantity") + gir::lit(int64_t{1});
    auto plan = l.aggregate(
        {"l_orderkey"},
        {{"n", gir::AggKind::CountDistinct, std::move(body)}}).plan();

    gir::RuleConfig cfg;
    cfg.rewrite_count_to_sum = false;
    cfg.split_non_equi_join_conditions = false;
    cfg.split_multi_key_equi_join = false;
    cfg.scan_column_pushdown = false;
    auto rewritten = gir::apply_rules(plan, cfg);
    EXPECT_EQ(rewritten.get(), plan.get());
    EXPECT_EQ(rewritten->agg_items[0].agg, gir::AggKind::CountDistinct);
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
