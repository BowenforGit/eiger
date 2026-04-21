#include <gtest/gtest.h>

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/expr.hpp"
#include "ir/plan_node.hpp"

namespace gir = gamax::ir;
using gamax::data_type;
using gamax::type_id;

namespace {
data_type t(type_id id) { return data_type{id}; }

gir::Catalog make_tpch_catalog() {
    gir::Catalog cat;
    gir::CatalogTable lineitem;
    lineitem.name = "lineitem";
    lineitem.column_names = {
        "l_shipdate", "l_extendedprice", "l_discount", "l_quantity",
        "l_returnflag", "l_linestatus", "l_orderkey",
    };
    lineitem.column_types = {
        t(type_id::INT32), t(type_id::FLOAT64),
        t(type_id::FLOAT64), t(type_id::INT32),
        t(type_id::STRING), t(type_id::STRING), t(type_id::INT64),
    };
    lineitem.column_nullable = {false, false, false, false, false, false, false};
    cat.add_table(std::move(lineitem));

    gir::CatalogTable orders;
    orders.name = "orders";
    orders.column_names  = {"o_orderkey", "o_orderdate", "o_orderpriority"};
    orders.column_types  = {t(type_id::INT64), t(type_id::INT32), t(type_id::STRING)};
    orders.column_nullable = {false, false, false};
    cat.add_table(std::move(orders));
    return cat;
}
} // namespace

TEST(IrBuilder, ScanResolvesColumnsByName) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    auto li = qb.scan("lineitem", {"l_shipdate", "l_discount"});
    EXPECT_EQ(li.schema().size(), 2u);

    auto ref = li.c("l_discount");
    EXPECT_EQ(ref.type, t(type_id::FLOAT64));
    EXPECT_EQ(ref.col_ref, li.schema()[1].id);
}

TEST(IrBuilder, UnknownColumnThrows) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate"});
    EXPECT_THROW(li.c("no_such_column"), std::runtime_error);
}

TEST(IrBuilder, FilterChain) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    auto li = qb.scan("lineitem", {"l_shipdate", "l_discount"});
    auto f = li.filter(
        li.c("l_shipdate") >= gir::lit(8766)
     && li.c("l_discount") <= gir::lit(0.07));

    EXPECT_EQ(f.plan()->kind, gir::NodeKind::Filter);
    EXPECT_EQ(f.schema().size(), 2u);
    // Filter preserves ColumnIds.
    EXPECT_EQ(f.schema()[0].id, li.schema()[0].id);
}

TEST(IrBuilder, ProjectComputedAndPure) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    auto li = qb.scan("lineitem", {"l_extendedprice", "l_discount"});
    auto p = li.project({
        {"revenue",  li.c("l_extendedprice") * li.c("l_discount")},
        {"discount", li.c("l_discount")},
    });

    ASSERT_EQ(p.schema().size(), 2u);
    EXPECT_EQ(p.schema()[0].type, t(type_id::FLOAT64));
    // The pure ColRef item ("discount") preserves the input ColumnId.
    EXPECT_EQ(p.schema()[1].id, li.schema()[1].id);
    // The computed one gets a fresh ColumnId.
    EXPECT_NE(p.schema()[0].id, li.schema()[0].id);
}

TEST(IrBuilder, AggregateByName) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    auto li = qb.scan("lineitem",
        {"l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice"});

    auto a = li.aggregate(
        {"l_returnflag", "l_linestatus"},
        {
            {"sum_qty",   gir::AggKind::Sum, li.c("l_quantity")},
            {"sum_price", gir::AggKind::Sum, li.c("l_extendedprice")},
            {"cnt",       gir::AggKind::CountAll, gir::lit(int64_t{0})},
        });

    ASSERT_EQ(a.schema().size(), 5u);
    // Keys are carried through with their original ids.
    EXPECT_EQ(a.schema()[0].id, li.schema()[0].id);
    EXPECT_EQ(a.schema()[1].id, li.schema()[1].id);
    EXPECT_EQ(a.schema()[2].type, t(type_id::INT64));   // sum_qty
    EXPECT_EQ(a.schema()[3].type, t(type_id::FLOAT64)); // sum_price
    EXPECT_EQ(a.schema()[4].type, t(type_id::INT64));   // cnt
}

TEST(IrBuilder, SortWithHelpers) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate", "l_discount"});

    auto s = li.sort({gir::asc("l_shipdate"), gir::desc("l_discount")});
    ASSERT_EQ(s.plan()->kind, gir::NodeKind::Sort);
    ASSERT_EQ(s.plan()->sort_keys.size(), 2u);
    EXPECT_EQ(s.plan()->sort_keys[0].order, gamax::sort_order::ASCENDING);
    EXPECT_EQ(s.plan()->sort_keys[1].order, gamax::sort_order::DESCENDING);
}

TEST(IrBuilder, LimitAndDistinct) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate"});
    auto d = li.distinct();        // all-columns distinct
    auto l = d.limit(5);
    EXPECT_EQ(d.plan()->kind, gir::NodeKind::Distinct);
    EXPECT_EQ(l.plan()->kind, gir::NodeKind::Limit);
    EXPECT_EQ(l.plan()->limit_n, 5);
}

TEST(IrBuilder, InnerJoinAcrossBuilders) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    auto li = qb.scan("lineitem", {"l_orderkey", "l_extendedprice"});
    auto o  = qb.scan("orders",   {"o_orderkey", "o_orderdate"});

    auto j = li.inner_join(o, li.c("l_orderkey") == o.c("o_orderkey"));
    EXPECT_EQ(j.plan()->kind, gir::NodeKind::Join);
    EXPECT_EQ(j.plan()->join_kind, gir::JoinKind::Inner);
    EXPECT_EQ(j.schema().size(), 4u);
}

TEST(IrBuilder, CrossJoinNoCondition) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);
    auto li = qb.scan("lineitem", {"l_shipdate"});
    auto o  = qb.scan("orders",   {"o_orderkey"});
    auto j = li.cross_join(o);
    EXPECT_EQ(j.plan()->join_kind, gir::JoinKind::Cross);
    EXPECT_FALSE(j.plan()->join_condition.has_value());
}

TEST(IrBuilder, SelfJoinYieldsDistinctColumnIds) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    // Self-join of lineitem with itself — each scan is a fresh column
    // source, so l_orderkey on the left and right sides have different
    // ColumnIds, which the join condition disambiguates.
    auto l1 = qb.scan("lineitem", {"l_orderkey"});
    auto l2 = qb.scan("lineitem", {"l_orderkey"});
    EXPECT_NE(l1.schema()[0].id, l2.schema()[0].id);

    auto j = l1.inner_join(l2, l1.c("l_orderkey") == l2.c("l_orderkey"));
    ASSERT_EQ(j.schema().size(), 2u);
    EXPECT_NE(j.schema()[0].id, j.schema()[1].id);
}

// ── End-to-end: Q6 expressed as a single declarative pipeline ──
TEST(IrBuilder, Q6DeclarativePipeline) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    auto li = qb.scan("lineitem",
        {"l_shipdate", "l_extendedprice", "l_discount", "l_quantity"});

    auto plan = li
        .filter(
            li.c("l_shipdate") >= gir::lit(8766)
         && li.c("l_shipdate") <  gir::lit(9131)
         && li.c("l_discount") >= gir::lit(0.05)
         && li.c("l_discount") <= gir::lit(0.07)
         && li.c("l_quantity") <  gir::lit(24))
        .project({
            {"revenue", li.c("l_extendedprice") * li.c("l_discount")}
        });
    plan = plan.aggregate(
        /*keys=*/{},
        {{"total", gir::AggKind::Sum, plan.c("revenue")}});

    ASSERT_EQ(plan.schema().size(), 1u);
    EXPECT_EQ(plan.schema()[0].name, "total");
    EXPECT_EQ(plan.schema()[0].type, t(type_id::FLOAT64));
    EXPECT_EQ(plan.plan()->kind, gir::NodeKind::Aggregate);
}

// End-to-end: Q3-style join + group-by + sort + limit in builder form.
TEST(IrBuilder, Q3DeclarativePipeline) {
    gir::IdGenerator gen;
    auto cat = make_tpch_catalog();
    gir::QueryBuilder qb(cat, gen);

    auto li = qb.scan("lineitem",
        {"l_orderkey", "l_extendedprice", "l_discount", "l_shipdate"});
    auto o  = qb.scan("orders",
        {"o_orderkey", "o_orderdate", "o_orderpriority"});

    auto joined = li.inner_join(
        o, li.c("l_orderkey") == o.c("o_orderkey"));
    auto filtered = joined.filter(
        joined.c("l_shipdate")   > gir::lit(9000)
     && joined.c("o_orderdate") <  gir::lit(9000));
    auto projected = filtered.project({
        {"revenue",
            filtered.c("l_extendedprice") * filtered.c("l_discount")},
        {"o_orderpriority", filtered.c("o_orderpriority")},
    });
    auto agg = projected.aggregate(
        {"o_orderpriority"},
        {{"rev_sum", gir::AggKind::Sum, projected.c("revenue")}});
    auto top = agg.sort({gir::desc("rev_sum")}).limit(10);

    ASSERT_EQ(top.schema().size(), 2u);
    EXPECT_EQ(top.schema()[0].name, "o_orderpriority");
    EXPECT_EQ(top.schema()[1].name, "rev_sum");
    EXPECT_EQ(top.plan()->kind, gir::NodeKind::Limit);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
