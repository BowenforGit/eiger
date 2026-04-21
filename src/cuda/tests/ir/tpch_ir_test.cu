#include <gtest/gtest.h>

#include <iostream>

#include "ir/builder.hpp"
#include "ir/catalog.hpp"
#include "ir/plan_node.hpp"
#include "ir/print.hpp"
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

TEST(IrTpchExpressivity, Q12Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q12(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 3u);  // shipmode + high + low
}

TEST(IrTpchExpressivity, Q13Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q13(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 2u);  // c_count + custdist
}

TEST(IrTpchExpressivity, Q14Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q14(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Project);
    EXPECT_EQ(plan->output_schema.size(), 1u);  // promo_revenue
}

TEST(IrTpchExpressivity, Q15Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q15(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    // s_suppkey, s_name, s_address, s_phone, total_revenue
    EXPECT_EQ(plan->output_schema.size(), 5u);
}

TEST(IrTpchExpressivity, Q16Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q16(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    // p_brand, p_type, p_size, supplier_cnt
    EXPECT_EQ(plan->output_schema.size(), 4u);
}

TEST(IrTpchExpressivity, Q17Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q17(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Project);
    EXPECT_EQ(plan->output_schema.size(), 1u);  // avg_yearly
}

TEST(IrTpchExpressivity, Q18Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q18(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Limit);
    EXPECT_EQ(plan->limit_n, 100);
    // c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum_qty
    EXPECT_EQ(plan->output_schema.size(), 6u);
}

TEST(IrTpchExpressivity, Q19Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q19(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Aggregate);
    EXPECT_EQ(plan->output_schema.size(), 1u);  // revenue
}

TEST(IrTpchExpressivity, Q20Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q20(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 2u);  // s_name, s_address
}

TEST(IrTpchExpressivity, Q21Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q21(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Limit);
    EXPECT_EQ(plan->limit_n, 100);
    EXPECT_EQ(plan->output_schema.size(), 2u);  // s_name, numwait
}

TEST(IrTpchExpressivity, Q22Builds) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    auto qb = make_qb(cat, gen);
    auto plan = gir_tpch::build_q22(qb);
    ASSERT_NE(plan, nullptr);
    EXPECT_EQ(plan->kind, gir::NodeKind::Sort);
    EXPECT_EQ(plan->output_schema.size(), 3u);  // cntrycode, numcust, totacctbal
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

// ── Visual-inspection dump of every TPC-H plan ─────────────────────
//
// This test asserts nothing; it prints the pretty-printed form of
// every TPC-H query we've built. Meant for running by hand:
//
//     build/src/cuda/tests/test_tpch_ir --gtest_filter='*DumpAll*'
//
// It's not a correctness check — just makes it cheap to eyeball the
// shape of each plan after an IR change.
TEST(IrTpchExpressivity, DumpAllPlans) {
    auto cat = gir_tpch::make_tpch_catalog();
    gir::IdGenerator gen;
    gir::QueryBuilder qb(cat, gen);

    struct Entry {
        const char* name;
        gir::PlanPtr (*build)(const gir::QueryBuilder&);
    };
    const Entry queries[] = {
        {"Q1",  gir_tpch::build_q1},
        {"Q2",  gir_tpch::build_q2},
        {"Q3",  gir_tpch::build_q3},
        {"Q4",  gir_tpch::build_q4},
        {"Q5",  gir_tpch::build_q5},
        {"Q6",  gir_tpch::build_q6},
        {"Q7",  gir_tpch::build_q7},
        {"Q8",  gir_tpch::build_q8},
        {"Q9",  gir_tpch::build_q9},
        {"Q10", gir_tpch::build_q10},
        {"Q11", gir_tpch::build_q11},
        {"Q12", gir_tpch::build_q12},
        {"Q13", gir_tpch::build_q13},
        {"Q14", gir_tpch::build_q14},
        {"Q15", gir_tpch::build_q15},
        {"Q16", gir_tpch::build_q16},
        {"Q17", gir_tpch::build_q17},
        {"Q18", gir_tpch::build_q18},
        {"Q19", gir_tpch::build_q19},
        {"Q20", gir_tpch::build_q20},
        {"Q21", gir_tpch::build_q21},
        {"Q22", gir_tpch::build_q22},
    };

    for (const auto& q : queries) {
        std::cout << "\n======================== TPC-H " << q.name
                  << " ========================\n";
        auto plan = q.build(qb);
        std::cout << gir::format_plan(*plan);
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
