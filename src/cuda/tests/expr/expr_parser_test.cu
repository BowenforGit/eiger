// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "expr/expr_parser.hpp"
#include "expr/expr_fast.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "tests/test_helper.hpp"
#include "types/data_type.hpp"
#include "utils/utils.cuh"

using namespace gamax::expression;

TEST(ExprParserTest, ArithmeticWithCast) {
    float val1[] = {1.5f, 2.5f, 3.5f, 4.5f};
    float val2[] = {0.5f, 1.0f, 1.5f, 2.0f};
    auto col1 = gamax::make_column_from_host(val1, 4, cudaStreamDefault);
    auto col2 = gamax::make_column_from_host(val2, 4, cudaStreamDefault);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    auto schema = std::vector<gamax::data_type>{table->view()[0].type(), table->view()[1].type()};

    auto expr = parse_expression_to_tree("col0 * col1 + float32(1.0)", schema);
    auto res = gamax::expression::evaluate_expr(table->view(), expr, cudaStreamDefault);

    float expected[] = {1.5f * 0.5f + 1.0f, 2.5f * 1.0f + 1.0f, 3.5f * 1.5f + 1.0f, 4.5f * 2.0f + 1.0f};
    auto expected_col = gamax::make_column_from_host(expected, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

TEST(ExprParserTest, TernarySelect) {
    int val0[] = {1, -2, 3, -4};
    int val1[] = {10, 20, 30, 40};
    int val2[] = {100, 200, 300, 400};
    auto col0 = gamax::make_column_from_host(val0, 4, cudaStreamDefault);
    auto col1 = gamax::make_column_from_host(val1, 4, cudaStreamDefault);
    auto col2 = gamax::make_column_from_host(val2, 4, cudaStreamDefault);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col0));
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    auto schema = std::vector<gamax::data_type>{table->view()[0].type(), table->view()[1].type(), table->view()[2].type()};

    auto expr = parse_expression_to_tree("col0 > 0 ? col1 : col2", schema);
    auto res = gamax::expression::evaluate_expr(table->view(), expr, cudaStreamDefault);

    int expected[] = {10, 200, 30, 400};
    auto expected_col = gamax::make_column_from_host(expected, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

TEST(ExprParserTest, PrecedenceAndParentheses) {
    int val0[] = {1, 2, 3, 4};
    int val1[] = {5, 6, 7, 8};
    int val2[] = {2, 3, 4, 5};
    auto col0 = gamax::make_column_from_host(val0, 4, cudaStreamDefault);
    auto col1 = gamax::make_column_from_host(val1, 4, cudaStreamDefault);
    auto col2 = gamax::make_column_from_host(val2, 4, cudaStreamDefault);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col0));
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    auto schema = std::vector<gamax::data_type>{table->view()[0].type(), table->view()[1].type(), table->view()[2].type()};

    auto expr = parse_expression_to_tree("col0 + col1 * col2", schema);
    auto res = gamax::expression::evaluate_expr(table->view(), expr, cudaStreamDefault);

    int expected[] = {1 + 5 * 2, 2 + 6 * 3, 3 + 7 * 4, 4 + 8 * 5};
    auto expected_col = gamax::make_column_from_host(expected, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));

    auto expr_with_parens = parse_expression_to_tree("(col0 + col1) * col2", schema);
    auto res_paren = gamax::expression::evaluate_expr(table->view(), expr_with_parens, cudaStreamDefault);

    int expected_paren[] = {(1 + 5) * 2, (2 + 6) * 3, (3 + 7) * 4, (4 + 8) * 5};
    auto expected_col_paren = gamax::make_column_from_host(expected_paren, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res_paren->view(), expected_col_paren->view()));
}

TEST(ExprParserTest, UnaryAndShift) {
    int val0[] = {1, 2, 3, 4};
    auto col0 = gamax::make_column_from_host(val0, 4, cudaStreamDefault);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col0));
    auto table = gamax::make_table(std::move(columns));
    auto schema = std::vector<gamax::data_type>{table->view()[0].type()};

    auto expr = parse_expression_to_tree("-(col0 << 2)", schema);
    auto res = gamax::expression::evaluate_expr(table->view(), expr, cudaStreamDefault);

    int expected[] = {-(1 << 2), -(2 << 2), -(3 << 2), -(4 << 2)};
    auto expected_col = gamax::make_column_from_host(expected, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

TEST(ExprParserTest, DefaultLiteralTypeIsInt32) {
    int val0[] = {10, 20, 30, 40};
    auto col0 = gamax::make_column_from_host(val0, 4, cudaStreamDefault);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col0));
    auto table = gamax::make_table(std::move(columns));
    auto schema = std::vector<gamax::data_type>{table->view()[0].type()};

    auto expr = parse_expression_to_tree("col0 + 1", schema);
    auto res = gamax::expression::evaluate_expr(table->view(), expr, cudaStreamDefault);

    int expected[] = {11, 21, 31, 41};
    auto expected_col = gamax::make_column_from_host(expected, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

TEST(ExprParserTest, LogicalChaining) {
    int val0[] = {0, 1, 2, 3};
    int val1[] = {2, 2, 1, 4};
    int val2[] = {5, 4, 5, 0};
    auto col0 = gamax::make_column_from_host(val0, 4, cudaStreamDefault);
    auto col1 = gamax::make_column_from_host(val1, 4, cudaStreamDefault);
    auto col2 = gamax::make_column_from_host(val2, 4, cudaStreamDefault);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col0));
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    auto schema = std::vector<gamax::data_type>{table->view()[0].type(), table->view()[1].type(), table->view()[2].type()};

    // Expression: (col0 > 1 && col1 == 2) || (col2 < 5)
    auto expr = parse_expression_to_tree("(col0 > 1 && col1 == 2) || (col2 < 5)", schema);
    auto res = gamax::expression::evaluate_expr(table->view(), expr, cudaStreamDefault);

    bool expected[] = {
        /* row0 */ (0 > 1 && 2 == 2) || (5 < 5),
        /* row1 */ (1 > 1 && 2 == 2) || (4 < 5),
        /* row2 */ (2 > 1 && 1 == 2) || (5 < 5),
        /* row3 */ (3 > 1 && 4 == 2) || (0 < 5)
    };
    auto expected_col = gamax::make_column_from_host(expected, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

TEST(ExprParserTest, LogicalWithTernary) {
    int val0[] = {1, -1, 2, -2};
    int val1[] = {5, 6, 7, 8};
    int val2[] = {3, 3, 10, 0};
    int val3[] = {1, 2, 3, 4};
    auto col0 = gamax::make_column_from_host(val0, 4, cudaStreamDefault);
    auto col1 = gamax::make_column_from_host(val1, 4, cudaStreamDefault);
    auto col2 = gamax::make_column_from_host(val2, 4, cudaStreamDefault);
    auto col3 = gamax::make_column_from_host(val3, 4, cudaStreamDefault);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col0));
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    columns.push_back(std::move(col3));
    auto table = gamax::make_table(std::move(columns));
    auto schema = std::vector<gamax::data_type>{
        table->view()[0].type(), table->view()[1].type(), table->view()[2].type(), table->view()[3].type()};

    // Expression: col0 > 0 ? (col1 > col2 && col2 > col3) : (col1 < col2 || col3 == 0)
    auto expr = parse_expression_to_tree(
        "col0 > 0 ? (col1 > col2 && col2 > col3) : (col1 < col2 || col3 == 0)", schema);
    auto res = gamax::expression::evaluate_expr(table->view(), expr, cudaStreamDefault);

    bool expected[] = {
        /* row0: cond true  */ (5 > 3 && 3 > 1),
        /* row1: cond false */ (6 < 3 || 2 == 0),
        /* row2: cond true  */ (7 > 10 && 10 > 3),
        /* row3: cond false */ (8 < 0 || 4 == 0)
    };
    auto expected_col = gamax::make_column_from_host(expected, 4, cudaStreamDefault);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}
