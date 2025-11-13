#include <gtest/gtest.h>
#include <memory>
#include <cuda.h>

#include "expr/expression.cuh"

#include "tests/test_helper.hpp"
#include "memory/column.hpp"
#include "types/data_type.hpp"
#include "utils/utils.cuh"

void test_expr_eval_unary() {
    int val[] = {1, 2, 3, 4};
    auto col = gamax::make_column_from_host(val, 4);

    gamax::expression::ExprTree et;
    auto id = et.add_node(gamax::expression::ColumnExpr(col->view()));
    id = et.add_node(gamax::expression::LeftShiftExpr(col->type(), 2), id);
    et.add_node(gamax::expression::NegateExpr(col->type()), id, true);

    auto res = gamax::expression::evaluate_expr(et);

    int expected[] = {-(1 << 2), -(2 << 2), -(3 << 2), -(4 << 2)};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

void test_expr_eval_binary() {
    int val[] = {1, 2, 3, 4};
    auto col = gamax::make_column_from_host(val, 4);

    gamax::expression::ExprTree et;
    int id1 = et.add_node(gamax::expression::ColumnExpr(col->view()));
    int id2 = et.add_node(gamax::expression::ConstantExpr(gamax::scalar(2)));
    et.add_node(gamax::expression::AddExpr(col->type()), id1, id2, true);

    auto res = gamax::expression::evaluate_expr(et);
    print_gpu_arr(res->view().data<int>(), 4);

    int expected[] = {3, 4, 5, 6};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

void test_expr_eval_any() {
    int val1[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val1, 4);

    int val2[] = {5, 6, 7, 8};
    auto col2 = gamax::make_column_from_host(val2, 4);

    gamax::expression::ExprTree et;
    auto id1 = et.add_node(gamax::expression::ColumnExpr(col1->view()));
    auto id2 = et.add_node(gamax::expression::ConstantExpr(gamax::scalar(1)));
    auto id3 = et.add_node(gamax::expression::SubstractExpr(col1->type()), id2, id1);
    auto id4 = et.add_node(gamax::expression::ColumnExpr(col2->view()));
    et.add_node(gamax::expression::MultiplyExpr(col1->type()), id3, id4, true);

    auto res = gamax::expression::evaluate_expr(et);
    print_gpu_arr(res->view().data<int>(), 4);

    int expected[] = {0, -6, -14, -24};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

TEST(ExpressionTest, EvalUnary) {
    test_expr_eval_unary();
}

TEST(ExpressionTest, EvalBinary) {
    test_expr_eval_binary();
}

TEST(ExpressionTest, EvalAny) {
    test_expr_eval_any();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}