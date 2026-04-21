#include <gtest/gtest.h>

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/plan_node.hpp"
#include "tpch/tpch_ir.hpp"

namespace gir = gamax::ir;
namespace gir_tpch = gamax::ir::tpch;

// These tests exercise IR expressivity: each plan must be buildable
// without throwing. Some plans use IR constructs whose physical paths
// aren't implemented yet (multi-key joins, cross_join, FunctionCall
// scalars, COUNT(*)); those land cleanly at IR-construction time and
// would only fail at emit time. That's the designed behavior.

namespace {
gir::QueryBuilder make_qb(gir::Catalog& cat, gir::IdGenerator& gen) {
    return gir::QueryBuilder(cat, gen);
}
} // namespace

TEST(IrTpchExpressivity, CatalogHasAllEightTables) {
    auto cat = gir_tpch::make_tpch_catalog();
    for (auto* n : {
        "region", "nation", "part", "supplier",
        "partsupp", "customer", "orders", "lineitem"})
    {
        EXPECT_NE(cat.find(n), nullptr) << "missing table: " << n;
    }
}

TEST(IrTpchExpressivity, Q1Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q1(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    // Output: 2 keys + 7 aggregates + count_order = 10 columns.
    EXPECT_EQ(plan->output_schema.size(), 10u);
}

TEST(IrTpchExpressivity, Q2Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q2(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Limit);
    EXPECT_EQ(plan->limit_n, 100);
    // SELECT list: s_acctbal, s_name, n_name, p_partkey, p_mfgr,
    // s_address, s_phone, s_comment = 8 columns.
    EXPECT_EQ(plan->output_schema.size(), 8u);
}

TEST(IrTpchExpressivity, Q3Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q3(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Limit);
    EXPECT_EQ(plan->limit_n, 10);
    // 3 keys + revenue = 4 columns.
    EXPECT_EQ(plan->output_schema.size(), 4u);
}

TEST(IrTpchExpressivity, Q4Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q4(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 2u);  // priority + count
}

TEST(IrTpchExpressivity, Q5Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q5(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 2u);  // n_name + revenue
}

TEST(IrTpchExpressivity, Q6Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q6(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Aggregate);
    EXPECT_EQ(plan->output_schema.size(), 1u);  // scalar total
}

TEST(IrTpchExpressivity, Q7Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q7(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    // supp_nation, cust_nation, l_year, revenue.
    EXPECT_EQ(plan->output_schema.size(), 4u);
}

TEST(IrTpchExpressivity, Q8Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q8(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    // o_year, mkt_share.
    EXPECT_EQ(plan->output_schema.size(), 2u);
}

TEST(IrTpchExpressivity, Q9Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q9(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 3u);  // nation, year, sum_profit
}

TEST(IrTpchExpressivity, Q10Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q10(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Limit);
    EXPECT_EQ(plan->limit_n, 20);
    // 7 key fields + revenue.
    EXPECT_EQ(plan->output_schema.size(), 8u);
}

TEST(IrTpchExpressivity, Q11Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q11(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 2u);  // partkey + value
}

// A sanity check on FunctionCall: Q7 and Q9 both use year(); make sure
// the resulting Expr is well-formed.
TEST(IrTpchExpressivity, FunctionCallConstructionAndNullable) {
    auto date_col = gir::col_ref(
        gir::ColumnId{0}, gamax::data_type{gamax::type_id::INT32}, false);
    auto y = gir::func("year", {std::move(date_col)},
                       gamax::data_type{gamax::type_id::INT32});
    EXPECT_EQ(y.kind, gir::ExprKind::FunctionCall);
    EXPECT_EQ(y.function_name, "year");
    EXPECT_EQ(y.children.size(), 1u);
    EXPECT_FALSE(y.nullable);

    auto nn_col = gir::col_ref(
        gir::ColumnId{1}, gamax::data_type{gamax::type_id::INT32}, true);
    auto y_null = gir::func("year", {std::move(nn_col)},
                            gamax::data_type{gamax::type_id::INT32});
    EXPECT_TRUE(y_null.nullable);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
