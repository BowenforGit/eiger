#include <gtest/gtest.h>

#include <string>

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/expr.hpp"
#include "ir/plan_node.hpp"
#include "ir/print.hpp"
#include "tpch/tpch_ir.hpp"

namespace gir = gamax::ir;
using gamax::data_type;
using gamax::type_id;

namespace {
data_type t(type_id id) { return data_type{id}; }

// Small catalog reused by several tests.
gir::Catalog make_small_catalog() {
    gir::Catalog cat;
    gir::CatalogTable t1;
    t1.name = "t1";
    t1.column_names  = {"a", "b", "c"};
    t1.column_types  = {t(type_id::INT32), t(type_id::FLOAT64), t(type_id::STRING)};
    t1.column_nullable = {false, false, false};
    cat.add_table(std::move(t1));

    gir::CatalogTable t2;
    t2.name = "t2";
    t2.column_names  = {"a", "d"};
    t2.column_types  = {t(type_id::INT32), t(type_id::FLOAT64)};
    t2.column_nullable = {false, false};
    cat.add_table(std::move(t2));
    return cat;
}

// Helper: does `haystack` contain `needle`?
bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}
} // namespace

// ── Expression rendering ───────────────────────────────────────────

TEST(IrPrint, ExprLiteralsAndOps) {
    gir::IdGenerator g;
    auto id = g.next();
    std::unordered_map<gir::ColumnId, std::string> names = {{id, "x"}};
    auto col = gir::col_ref(id, t(type_id::INT32), false);
    auto e = (col + gir::lit(1)) * gir::lit(2.5);
    EXPECT_EQ(gir::format_expr(e, names), "((x + 1) * 2.5)");
}

TEST(IrPrint, ExprColRefFallsBackToIdWithoutNames) {
    gir::IdGenerator g;
    auto id = g.next();  // value=0
    auto col = gir::col_ref(id, t(type_id::INT32), false);
    EXPECT_EQ(gir::format_expr(col), "#0");
}

TEST(IrPrint, ExprLogicalAndCompare) {
    gir::IdGenerator g;
    auto a = g.next(); auto b = g.next();
    std::unordered_map<gir::ColumnId, std::string> names = {
        {a, "a"}, {b, "b"},
    };
    auto refa = gir::col_ref(a, t(type_id::INT32), false);
    auto refb = gir::col_ref(b, t(type_id::INT32), false);
    auto e = (refa >= gir::lit(0)) && (refb < gir::lit(10));
    EXPECT_EQ(gir::format_expr(e, names), "((a >= 0) && (b < 10))");
}

TEST(IrPrint, ExprStringOpAndCase) {
    gir::IdGenerator g;
    auto id = g.next();
    std::unordered_map<gir::ColumnId, std::string> names = {{id, "p_type"}};
    auto s = gir::col_ref(id, t(type_id::STRING), false);
    auto pred = gir::string_op(
        gir::StringOpKind::Startswith, std::move(s), {"PROMO"});
    auto cs = gir::case_when(std::move(pred), gir::lit(1.0), gir::lit(0.0));
    auto out = gir::format_expr(cs, names);
    EXPECT_TRUE(contains(out, "case("));
    EXPECT_TRUE(contains(out, "startswith(p_type, \"PROMO\")"));
    EXPECT_TRUE(contains(out, "1"));
    EXPECT_TRUE(contains(out, "0"));
}

TEST(IrPrint, ExprFunctionCall) {
    gir::IdGenerator g;
    auto id = g.next();
    std::unordered_map<gir::ColumnId, std::string> names = {{id, "d"}};
    auto col = gir::col_ref(id, t(type_id::INT32), false);
    auto e = gir::func("year", {std::move(col)}, t(type_id::INT32));
    EXPECT_EQ(gir::format_expr(e, names), "year(d)");
}

TEST(IrPrint, ExprUnaryNot) {
    gir::IdGenerator g;
    auto id = g.next();
    std::unordered_map<gir::ColumnId, std::string> names = {{id, "s"}};
    auto s = gir::col_ref(id, t(type_id::STRING), false);
    auto pred = gir::string_op(gir::StringOpKind::Equal, std::move(s), {"x"});
    auto n = !pred;
    EXPECT_EQ(gir::format_expr(n, names), "!str_eq(s, \"x\")");
}

// ── Schema rendering ───────────────────────────────────────────────

TEST(IrPrint, SchemaBasic) {
    gir::IdGenerator g;
    gir::Schema s({
        {g.next(), t(type_id::INT32),   false, "a"},
        {g.next(), t(type_id::FLOAT64), false, "b"},
    });
    EXPECT_EQ(gir::format_schema(s), "[a:INT32, b:FLOAT64]");
}

TEST(IrPrint, SchemaUnnamedFallsBackToId) {
    gir::IdGenerator g;
    auto id = g.next();
    gir::Schema s({ {id, t(type_id::INT64), false, ""} });
    EXPECT_EQ(gir::format_schema(s),
              "[#" + std::to_string(id.value) + ":INT64]");
}

// ── Per-node rendering ─────────────────────────────────────────────

TEST(IrPrint, PlanScanAndFilter) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("t1", {"a", "b"});
    auto f = li.filter(li.c("a") > gir::lit(0));
    auto out = gir::format_plan(*f.plan());
    EXPECT_TRUE(contains(out, "Filter((a > 0))"));
    EXPECT_TRUE(contains(out, "Scan(t1 [a, b])"));
}

TEST(IrPrint, PlanProjectShowsNamedAssignments) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("t1", {"a", "b"});
    auto p = li.project({
        {"sum", li.c("a") + gir::lit(1)},
        {"b",   li.c("b")},
    });
    auto out = gir::format_plan(*p.plan());
    EXPECT_TRUE(contains(out, "Project(sum = (a + 1), b = b)"));
}

TEST(IrPrint, PlanAggregateShowsKeysAndAggs) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("t1", {"a", "b"});
    auto a = li.aggregate(
        {"a"},
        {
            {"sum_b", gir::AggKind::Sum, li.c("b")},
            {"cnt",   gir::AggKind::CountAll, gir::lit(int64_t{0})},
        });
    auto out = gir::format_plan(*a.plan());
    EXPECT_TRUE(contains(out, "Aggregate(keys=[a]; aggs=[sum(b) AS sum_b, count_all(*) AS cnt])"));
}

TEST(IrPrint, PlanSortAndLimitIncludesOrders) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("t1", {"a", "b"});
    auto top = li.sort({gir::desc("a"), gir::asc("b")}).limit(5);
    auto out = gir::format_plan(*top.plan());
    EXPECT_TRUE(contains(out, "Sort[a DESC, b ASC]"));
    EXPECT_TRUE(contains(out, "Limit(n=5)"));
}

TEST(IrPrint, PlanInnerJoinAndCrossJoin) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    // Distinct column names on each side so the join condition reads
    // clearly without triggering id-disambiguation.
    auto A = qb.scan("t1", {"b"});         // b on t1 is FLOAT64
    auto B = qb.scan("t1", {"b"});         // separate scan, separate ids
    // Both columns are named "b" -> printer will disambiguate with #id
    // (exercised by the dedicated test below).
    auto inner = A.inner_join(B, A.c("b") == B.c("b"));
    auto cross = A.cross_join(B);

    auto inner_out = gir::format_plan(*inner.plan());
    EXPECT_TRUE(contains(inner_out, "InnerJoin"));
    EXPECT_TRUE(contains(inner_out, "=="));
    auto cross_out = gir::format_plan(*cross.plan());
    EXPECT_TRUE(contains(cross_out, "CrossJoin\n"));
}

TEST(IrPrint, ColRefDisambiguatesOnNameCollision) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    // Two scans of the same table — each column gets a fresh
    // ColumnId, so the joined output has *two* fields named "a" and
    // two named "b".
    auto L = qb.scan("t1", {"a", "b"});
    auto R = qb.scan("t1", {"a", "b"});
    auto joined = L.inner_join(R, L.c("a") == R.c("a"));
    auto out = gir::format_plan(*joined.plan());

    // Each colliding name is rendered as `name#<id>`; the two sides
    // of the equality are distinct.
    auto pos1 = out.find("a#");
    ASSERT_NE(pos1, std::string::npos);
    auto pos2 = out.find("a#", pos1 + 1);
    ASSERT_NE(pos2, std::string::npos);
    // And the two "a#..." tokens have different ids.
    auto tok1 = out.substr(pos1, 5);  // enough to capture a#<digit(s)>
    auto tok2 = out.substr(pos2, 5);
    EXPECT_NE(tok1, tok2);
}

TEST(IrPrint, NonAmbiguousNamesStayClean) {
    // The common case: no cross-scan name collisions. Names should
    // render without a `#<id>` suffix.
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("t1", {"a", "b"});
    auto f = li.filter(li.c("a") >= gir::lit(0));
    auto out = gir::format_plan(*f.plan());
    // Expression renders as "(a >= 0)", not "(a#N >= 0)".
    EXPECT_TRUE(contains(out, "(a >= 0)"));
    EXPECT_EQ(out.find("a#"), std::string::npos);
}

TEST(IrPrint, PlanDistinct) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("t1", {"a", "b"});
    auto d_all  = li.distinct();
    auto d_keys = li.distinct({"a"});

    EXPECT_TRUE(contains(gir::format_plan(*d_all.plan()),  "Distinct(all)"));
    EXPECT_TRUE(contains(gir::format_plan(*d_keys.plan()), "Distinct(keys=[a])"));
}

// ── Nesting / indentation ─────────────────────────────────────────

TEST(IrPrint, NestingIndentsWithTwoSpacesPerDepth) {
    auto cat = make_small_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto li = qb.scan("t1", {"a"});
    auto f = li.filter(li.c("a") >= gir::lit(0));
    auto s = f.sort({gir::asc("a")});
    auto out = gir::format_plan(*s.plan());

    // Expected shape:
    //   Sort[a ASC]
    //     Filter((a >= 0))
    //       Scan(t1 [a])
    EXPECT_TRUE(contains(out, "Sort[a ASC]\n  Filter((a >= 0))\n    Scan(t1 [a])\n"));
}

// ── End-to-end snapshot on a TPC-H query ──────────────────────────

TEST(IrPrint, Q6PrintsReadablePlan) {
    auto cat = gamax::ir::tpch::make_tpch_catalog();
    gir::IdGenerator g;
    gir::QueryBuilder qb(cat, g);
    auto plan = gamax::ir::tpch::build_q6(qb);
    auto out = gir::format_plan(*plan);

    // Should contain the headline nodes.
    EXPECT_TRUE(contains(out, "Aggregate("));
    EXPECT_TRUE(contains(out, "sum(revenue) AS total"));
    EXPECT_TRUE(contains(out, "Project(revenue = (l_extendedprice * l_discount))"));
    EXPECT_TRUE(contains(out, "Filter("));
    EXPECT_TRUE(contains(out, "Scan(lineitem [l_shipdate, l_extendedprice, l_discount, l_quantity])"));
    // And nested structure: the Scan line should be indented 3 levels
    // below the root Aggregate (6 spaces of leading whitespace).
    EXPECT_TRUE(contains(out, "\n      Scan(lineitem"));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
