#include <gtest/gtest.h>
#include <memory>
#include <cuda.h>

#include "expr/expr_fast.hpp"
#include "tests/test_helper.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"
#include "utils/utils.cuh"

using namespace gamax::expression;

// Test: Constant expression
TEST(ExprFastTest, ConstantExpr) {
    int val[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: constant value 5
    auto expr = ExprTree::constant(gamax::scalar(5));
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {5, 5, 5, 5};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Float data type arithmetic
TEST(ExprFastTest, FloatArithmetic) {
    float val1[] = {1.5f, 2.5f, 3.5f, 4.5f};
    float val2[] = {0.5f, 1.0f, 1.5f, 2.0f};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] * column[1] + 1.0
    auto float_type = gamax::data_type(gamax::type_id::FLOAT32);
    auto expr = ExprTree::add(
        ExprTree::multiply(
            ExprTree::column(0, float_type),
            ExprTree::column(1, float_type)
        ),
        ExprTree::constant(gamax::scalar(1.0f))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    float expected[] = {1.5f * 0.5f + 1.0f, 2.5f * 1.0f + 1.0f, 3.5f * 1.5f + 1.0f, 4.5f * 2.0f + 1.0f};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Double data type arithmetic
TEST(ExprFastTest, DoubleArithmetic) {
    double val1[] = {1.5, 2.5, 3.5, 4.5};
    double val2[] = {0.5, 1.0, 1.5, 2.0};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: (column[0] - column[1]) / 2.0
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::divide(
        ExprTree::subtract(
            ExprTree::column(0, double_type),
            ExprTree::column(1, double_type)
        ),
        ExprTree::constant(gamax::scalar(2.0))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    double expected[] = {(1.5 - 0.5) / 2.0, (2.5 - 1.0) / 2.0, (3.5 - 1.5) / 2.0, (4.5 - 2.0) / 2.0};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Float comparison
TEST(ExprFastTest, FloatComparison) {
    float val1[] = {1.5f, 2.5f, 3.5f, 4.5f};
    float val2[] = {2.0f, 2.5f, 3.0f, 5.0f};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] >= column[1]
    auto float_type = gamax::data_type(gamax::type_id::FLOAT32);
    auto expr = ExprTree::greater_equal(
        ExprTree::column(0, float_type),
        ExprTree::column(1, float_type)
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {false, true, true, false};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Double comparison
TEST(ExprFastTest, DoubleComparison) {
    double val1[] = {1.5, 2.5, 3.5, 4.5};
    double val2[] = {2.0, 2.5, 3.0, 5.0};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] < column[1]
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::less_than(
        ExprTree::column(0, double_type),
        ExprTree::column(1, double_type)
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {true, false, false, true};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Column access
TEST(ExprFastTest, ColumnExpr) {
    int val[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0]
    auto expr = ExprTree::column(0, gamax::data_type(gamax::type_id::INT32));
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {1, 2, 3, 4};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Unary expression (Negate)
TEST(ExprFastTest, NegateExpr) {
    int val[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: -column[0]
    auto expr = ExprTree::negate(
        ExprTree::column(0, gamax::data_type(gamax::type_id::INT32))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {-1, -2, -3, -4};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Unary expression with chaining (Negate after LeftShift)
TEST(ExprFastTest, UnaryChainExpr) {
    int val[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: -(column[0] << 2)
    auto expr = ExprTree::negate(
        ExprTree::left_shift(
            ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
            2
        )
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {-(1 << 2), -(2 << 2), -(3 << 2), -(4 << 2)};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Binary expression (Add)
TEST(ExprFastTest, AddExpr) {
    int val[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] + 2
    auto expr = ExprTree::add(
        ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
        ExprTree::constant(gamax::scalar(2))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {3, 4, 5, 6};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Complex expression with multiple operations
TEST(ExprFastTest, ComplexExpr) {
    int val1[] = {1, 2, 3, 4};
    int val2[] = {5, 6, 7, 8};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: (1 - column[0]) * column[1]
    auto expr = ExprTree::multiply(
        ExprTree::subtract(
            ExprTree::constant(gamax::scalar(1)),
            ExprTree::column(0, gamax::data_type(gamax::type_id::INT32))
        ),
        ExprTree::column(1, gamax::data_type(gamax::type_id::INT32))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {0, -6, -14, -24};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Comparison expression (Equal)
TEST(ExprFastTest, EqualExpr) {
    int val1[] = {1, 2, 3, 4};
    int val2[] = {1, 3, 3, 5};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] == column[1]
    auto expr = ExprTree::equal(
        ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
        ExprTree::column(1, gamax::data_type(gamax::type_id::INT32))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {true, false, true, false};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Comparison expression (GreaterThan)
TEST(ExprFastTest, GreaterThanExpr) {
    int val1[] = {1, 3, 3, 5};
    int val2[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] > column[1]
    auto expr = ExprTree::greater_than(
        ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
        ExprTree::column(1, gamax::data_type(gamax::type_id::INT32))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {false, true, false, true};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Logical expression (And)
TEST(ExprFastTest, AndExpr) {
    bool val1[] = {true, true, false, false};
    bool val2[] = {true, false, true, false};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] && column[1]
    auto expr = ExprTree::logical_and(
        ExprTree::column(0, gamax::data_type(gamax::type_id::BOOL8)),
        ExprTree::column(1, gamax::data_type(gamax::type_id::BOOL8))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {true, false, false, false};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Logical expression (Or)
TEST(ExprFastTest, OrExpr) {
    bool val1[] = {true, true, false, false};
    bool val2[] = {true, false, true, false};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] || column[1]
    auto expr = ExprTree::logical_or(
        ExprTree::column(0, gamax::data_type(gamax::type_id::BOOL8)),
        ExprTree::column(1, gamax::data_type(gamax::type_id::BOOL8))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {true, true, true, false};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Combined comparison and logical operations
TEST(ExprFastTest, CombinedLogicalExpr) {
    int val1[] = {1, 2, 3, 4};
    int val2[] = {2, 2, 2, 2};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: (column[0] > 1) && (column[1] == 2)
    auto expr = ExprTree::logical_and(
        ExprTree::greater_than(
            ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
            ExprTree::constant(gamax::scalar(1))
        ),
        ExprTree::equal(
            ExprTree::column(1, gamax::data_type(gamax::type_id::INT32)),
            ExprTree::constant(gamax::scalar(2))
        )
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {false, true, true, true};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Multiple columns with arithmetic
TEST(ExprFastTest, MultiColumnArithmetic) {
    int val1[] = {10, 20, 30, 40};
    int val2[] = {5, 10, 15, 20};
    int val3[] = {2, 3, 4, 5};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    auto col3 = gamax::make_column_from_host(val3, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    columns.push_back(std::move(col3));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: (column[0] + column[1]) * column[2]
    auto expr = ExprTree::multiply(
        ExprTree::add(
            ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
            ExprTree::column(1, gamax::data_type(gamax::type_id::INT32))
        ),
        ExprTree::column(2, gamax::data_type(gamax::type_id::INT32))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {30, 90, 180, 300};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Register pressure (deep expression tree)
TEST(ExprFastTest, RegisterPressure) {
    int val[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: ((col + 1) + 2) + 3
    auto expr = ExprTree::add(
        ExprTree::add(
            ExprTree::add(
                ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
                ExprTree::constant(gamax::scalar(1))
            ),
            ExprTree::constant(gamax::scalar(2))
        ),
        ExprTree::constant(gamax::scalar(3))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {7, 8, 9, 10};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Using operator overloading
TEST(ExprFastTest, OperatorOverloading) {
    int val1[] = {1, 2, 3, 4};
    int val2[] = {5, 6, 7, 8};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    
    // Create expression using operator overloading: (1 - column[0]) * column[1]
    auto expr = (ExprTree::constant(gamax::scalar(1)) - ExprTree::column(0, int_type)) 
              * ExprTree::column(1, int_type);
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {0, -6, -14, -24};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Operator overloading with unary minus and left shift
TEST(ExprFastTest, OperatorOverloadingUnary) {
    int val[] = {1, 2, 3, 4};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression using operator overloading: -(column[0] << 2)
    auto expr = -(ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)) << 2);
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {-(1 << 2), -(2 << 2), -(3 << 2), -(4 << 2)};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Right shift operator
TEST(ExprFastTest, RightShiftExpr) {
    int val[] = {16, 32, 64, 128};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: column[0] >> 2
    auto expr = ExprTree::right_shift(
        ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)),
        2
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {16 >> 2, 32 >> 2, 64 >> 2, 128 >> 2};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Right shift with operator overloading
TEST(ExprFastTest, RightShiftOperatorOverloading) {
    int val[] = {16, 32, 64, 128};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression using operator overloading: column[0] >> 2
    auto expr = ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)) >> 2;
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    int expected[] = {4, 8, 16, 32};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Square root function
TEST(ExprFastTest, SqrtExpr) {
    double val[] = {4.0, 9.0, 16.0, 25.0};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: sqrt(column[0])
    auto expr = ExprTree::sqrt(
        ExprTree::column(0, gamax::data_type(gamax::type_id::FLOAT64))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    double expected[] = {2.0, 3.0, 4.0, 5.0};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Natural logarithm function
TEST(ExprFastTest, LogExpr) {
    double val[] = {1.0, 2.718281828459045, 7.389056098930650, 20.085536923187668};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: log(column[0])
    auto expr = ExprTree::log(
        ExprTree::column(0, gamax::data_type(gamax::type_id::FLOAT64))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    // log(1) = 0, log(e) = 1, log(e^2) = 2, log(e^3) = 3
    double expected[] = {0.0, 1.0, 2.0, 3.0};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Exponential function
TEST(ExprFastTest, ExpExpr) {
    double val[] = {0.0, 1.0, 2.0, 3.0};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: exp(column[0])
    auto expr = ExprTree::exp(
        ExprTree::column(0, gamax::data_type(gamax::type_id::FLOAT64))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    // exp(0) = 1, exp(1) = e, exp(2) = e^2, exp(3) = e^3
    double expected[] = {1.0, 2.718281828459045, 7.389056098930650, 20.085536923187668};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Float square root
TEST(ExprFastTest, SqrtFloatExpr) {
    float val[] = {1.0f, 4.0f, 9.0f, 16.0f};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: sqrt(column[0])
    auto expr = ExprTree::sqrt(
        ExprTree::column(0, gamax::data_type(gamax::type_id::FLOAT32))
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Test: Complex expression with sqrt, log, and exp
TEST(ExprFastTest, ComplexMathExpr) {
    double val[] = {4.0, 9.0, 16.0, 25.0};
    auto col1 = gamax::make_column_from_host(val, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));
    
    // Create expression tree: exp(log(sqrt(column[0])))
    auto expr = ExprTree::exp(
        ExprTree::log(
            ExprTree::sqrt(
                ExprTree::column(0, gamax::data_type(gamax::type_id::FLOAT64))
            )
        )
    );
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    double expected[] = {2.0, 3.0, 4.0, 5.0};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    auto res_host = gamax::copy_column_to_host<double>(res->view());
    for(size_t i = 0; i < 4; ++i) {
        ASSERT_NEAR(res_host[i], expected[i], 1e-6);
    }
}

// Test: Operator overloading with comparisons
TEST(ExprFastTest, OperatorOverloadingComparison) {
    int val1[] = {1, 2, 3, 4};
    int val2[] = {2, 2, 2, 2};
    auto col1 = gamax::make_column_from_host(val1, 4);
    auto col2 = gamax::make_column_from_host(val2, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    columns.push_back(std::move(col2));
    auto table = gamax::make_table(std::move(columns));
    
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    
    // Create expression: (column[0] > 1) && (column[1] == 2)
    auto expr = (ExprTree::column(0, int_type) > ExprTree::constant(gamax::scalar(1)))
             && (ExprTree::column(1, int_type) == ExprTree::constant(gamax::scalar(2)));
    
    auto res = gamax::expression::evaluate_expr(table->view(), expr);
    
    bool expected[] = {false, true, true, true};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// ── YearOf: extract year from YYYYMMDD int32 date ──

// Single-row-per-year coverage: each input is a different date in a
// different year, so the output is just the year component.
TEST(ExprFastTest, YearOfExpr) {
    int32_t val[] = {19950101, 19951231, 19970615, 19981201};
    auto col1 = gamax::make_column_from_host(val, 4);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));

    auto expr = ExprTree::year_of(
        ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)));

    auto res = gamax::expression::evaluate_expr(table->view(), expr);

    int32_t expected[] = {1995, 1995, 1997, 1998};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

// Compose year_of with a comparison — the common use case
// (`year(o_orderdate) >= 1993`). Regression for the result flowing
// through a downstream comparison kernel.
TEST(ExprFastTest, YearOfInComparison) {
    int32_t val[] = {19920101, 19930101, 19940101, 19950101};
    auto col1 = gamax::make_column_from_host(val, 4);

    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1));
    auto table = gamax::make_table(std::move(columns));

    // year_of(col) >= 1994
    auto expr =
        ExprTree::year_of(ExprTree::column(0, gamax::data_type(gamax::type_id::INT32)))
            >= ExprTree::constant(gamax::scalar(int32_t{1994}));

    auto res = gamax::expression::evaluate_expr(table->view(), expr);

    bool expected[] = {false, false, true, true};
    auto expected_col = gamax::make_column_from_host(expected, 4);
    ASSERT_TRUE(gamax::test::check_cols_eq(res->view(), expected_col->view()));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
