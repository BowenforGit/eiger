#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cuda.h>

#include "operators/select/select.hpp"
#include "expr/expr_fast.cuh"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "types/data_type.hpp"
#include "tests/test_helper.hpp"
#include "utils/utils.cuh"

namespace {

using namespace gamax::expression;

// Helper function to create a table from host data arrays
template<typename T>
std::unique_ptr<gamax::table> make_test_table(
    const std::vector<std::vector<T>>& column_data)
{
    std::vector<std::unique_ptr<gamax::column>> columns;
    for (const auto& col_data : column_data) {
        auto col = gamax::make_column_from_host(col_data.data(), col_data.size());
        columns.push_back(std::move(col));
    }
    return gamax::make_table(std::move(columns));
}

// Helper function to verify table results
template<typename T>
bool verify_table_result(
    gamax::table_view result,
    const std::vector<std::vector<T>>& expected_data)
{
    if (result.num_columns() != expected_data.size()) {
        std::cout << "Number of columns mismatch" << std::endl;
        return false;
    }
    
    if (expected_data.empty() || result.num_rows() != expected_data[0].size()) {
        std::cout << "Number of rows mismatch" << std::endl;
        return false;
    }

    for (gamax::size_type col_idx = 0; col_idx < result.num_columns(); ++col_idx) {
        auto expected_col = gamax::make_column_from_host(
            expected_data[col_idx].data(), 
            expected_data[col_idx].size());
        if (!gamax::test::check_cols_eq(result[col_idx], expected_col->view())) {
            std::cout << "Column " << col_idx << " data mismatch" << std::endl;
            return false;
        }
    }
    return true;
}

// Test: Float data type with arithmetic predicate
void test_select_float_arithmetic() {
    std::vector<std::vector<float>> input_data = {
        {1.5f, 2.5f, 3.5f, 4.5f, 5.5f},  // col1
        {0.5f, 1.0f, 1.5f, 2.0f, 2.5f}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] * col[1] > 5.0
    auto float_type = gamax::data_type(gamax::type_id::FLOAT32);
    auto expr = ExprTree::greater_than(
        ExprTree::multiply(
            ExprTree::column(0, float_type),
            ExprTree::column(1, float_type)
        ),
        ExprTree::constant(gamax::scalar(5.0f))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 * col2 > 5.0
    // Row 0: 1.5*0.5=0.75 (no), Row 1: 2.5*1.0=2.5 (no), Row 2: 3.5*1.5=5.25 (yes)
    // Row 3: 4.5*2.0=9.0 (yes), Row 4: 5.5*2.5=13.75 (yes)
    std::vector<std::vector<float>> expected_data = {
        {3.5f, 4.5f, 5.5f},  // col1
        {1.5f, 2.0f, 2.5f}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test: Double data type with comparison predicate
void test_select_double_comparison() {
    std::vector<std::vector<double>> input_data = {
        {1.5, 2.5, 3.5, 4.5, 5.5},  // col1
        {2.0, 2.5, 3.0, 5.0, 4.0}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] >= col[1]
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::greater_equal(
        ExprTree::column(0, double_type),
        ExprTree::column(1, double_type)
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 >= col2
    // Row 0: 1.5>=2.0 (no), Row 1: 2.5>=2.5 (yes), Row 2: 3.5>=3.0 (yes)
    // Row 3: 4.5>=5.0 (no), Row 4: 5.5>=4.0 (yes)
    std::vector<std::vector<double>> expected_data = {
        {2.5, 3.5, 5.5},  // col1
        {2.5, 3.0, 4.0}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test: Float with complex expression and logical operators
void test_select_float_complex() {
    std::vector<std::vector<float>> input_data = {
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},  // col1
        {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: (col[0] + col[1] > 6.0) && (col[0] < 5.0)
    auto float_type = gamax::data_type(gamax::type_id::FLOAT32);
    auto expr = ExprTree::logical_and(
        ExprTree::greater_than(
            ExprTree::add(
                ExprTree::column(0, float_type),
                ExprTree::column(1, float_type)
            ),
            ExprTree::constant(gamax::scalar(6.0f))
        ),
        ExprTree::less_than(
            ExprTree::column(0, float_type),
            ExprTree::constant(gamax::scalar(5.0f))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where (col1 + col2 > 6.0) AND (col1 < 5.0)
    // Row 0: 1.0+1.5=2.5>6.0 (no), Row 1: 2.0+2.5=4.5>6.0 (no)
    // Row 2: 3.0+3.5=6.5>6.0 && 3.0<5.0 (yes), Row 3: 4.0+4.5=8.5>6.0 && 4.0<5.0 (yes)
    // Row 4: 5.0+5.5=10.5>6.0 && 5.0<5.0 (no), Row 5: 6.0+6.5=12.5>6.0 && 6.0<5.0 (no)
    std::vector<std::vector<float>> expected_data = {
        {3.0f, 4.0f},  // col1
        {3.5f, 4.5f}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test: Double with division and comparison
void test_select_double_division() {
    std::vector<std::vector<double>> input_data = {
        {10.0, 20.0, 30.0, 40.0, 50.0},  // col1
        {2.0, 4.0, 5.0, 8.0, 10.0}       // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] / col[1] >= 5.0
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::greater_equal(
        ExprTree::divide(
            ExprTree::column(0, double_type),
            ExprTree::column(1, double_type)
        ),
        ExprTree::constant(gamax::scalar(5.0))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 / col2 >= 5.0
    // Row 0: 10.0/2.0=5.0 (yes), Row 1: 20.0/4.0=5.0 (yes), Row 2: 30.0/5.0=6.0 (yes)
    // Row 3: 40.0/8.0=5.0 (yes), Row 4: 50.0/10.0=5.0 (yes)
    std::vector<std::vector<double>> expected_data = {
        {10.0, 20.0, 30.0, 40.0, 50.0},  // col1
        {2.0, 4.0, 5.0, 8.0, 10.0}       // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 1: Single column predicate with high selectivity (>80%)
// Predicate: col1 > 1 (selects 8 out of 10 rows = 80%)
void test_select_single_column_high_selectivity() {
    // Create a table with one column
    std::vector<std::vector<int32_t>> input_data = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] > 1
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::greater_than(
        ExprTree::column(0, int_type),
        ExprTree::constant(gamax::scalar(1))
    );

    // Execute select
    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col[0] > 1 (i.e., 2, 3, 4, 5, 6, 7, 8, 9)
    std::vector<std::vector<int>> expected_data = {
        {2, 3, 4, 5, 6, 7, 8, 9}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 2: Single column predicate with low selectivity (<20%)
// Predicate: col1 < 2 (selects 2 out of 10 rows = 20%)
void test_select_single_column_low_selectivity() {
    std::vector<std::vector<int>> input_data = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] < 2
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::less_than(
        ExprTree::column(0, int_type),
        ExprTree::constant(gamax::scalar(2))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col[0] < 2 (i.e., 0, 1)
    std::vector<std::vector<int>> expected_data = {
        {0, 1}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 3: Single column predicate with medium selectivity (~50%)
// Predicate: col1 >= 5 (selects 5 out of 10 rows = 50%)
void test_select_single_column_medium_selectivity() {
    std::vector<std::vector<int>> input_data = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] >= 5
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::greater_equal(
        ExprTree::column(0, int_type),
        ExprTree::constant(gamax::scalar(5))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col[0] >= 5 (i.e., 5, 6, 7, 8, 9)
    std::vector<std::vector<int>> expected_data = {
        {5, 6, 7, 8, 9}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 4: Two column predicate with high selectivity
// Predicate: col1 + col2 > 5 (high selectivity)
void test_select_two_columns_high_selectivity() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5, 6, 7, 8},  // col1
        {0, 1, 2, 3, 4, 5, 6, 7}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] + col[1] > 5
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::greater_than(
        ExprTree::add(
            ExprTree::column(0, int_type),
            ExprTree::column(1, int_type)
        ),
        ExprTree::constant(gamax::scalar(5))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 + col2 > 5
    // Row 0: 1+0=1 (no), Row 1: 2+1=3 (no), Row 2: 3+2=5 (no)
    // Row 3: 4+3=7 (yes), Row 4: 5+4=9 (yes), Row 5: 6+5=11 (yes)
    // Row 6: 7+6=13 (yes), Row 7: 8+7=15 (yes)
    std::vector<std::vector<int>> expected_data = {
        {4, 5, 6, 7, 8},  // col1
        {3, 4, 5, 6, 7}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 5: Two column predicate with low selectivity
// Predicate: col1 - col2 < 1 (low selectivity)
void test_select_two_columns_low_selectivity() {
    std::vector<std::vector<int>> input_data = {
        {5, 6, 7, 8, 9, 10, 11, 12},  // col1
        {1, 2, 3, 4, 5, 6,  7,  8}    // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] - col[1] < 1
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::less_than(
        ExprTree::subtract(
            ExprTree::column(0, int_type),
            ExprTree::column(1, int_type)
        ),
        ExprTree::constant(gamax::scalar(1))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 - col2 < 1 (i.e., difference is 0 or negative)
    // All differences are 4, so no rows should be selected
    ASSERT_EQ(result->num_rows(), 0);
}

// Test 6: Two column predicate with medium selectivity
// Predicate: col1 <= col2 (~50% selectivity)
void test_select_two_columns_medium_selectivity() {
    std::vector<std::vector<int>> input_data = {
        {1, 5, 3, 7, 2, 8, 4, 9},  // col1
        {4, 3, 6, 2, 8, 1, 9, 0}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] <= col[1]
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::less_equal(
        ExprTree::column(0, int_type),
        ExprTree::column(1, int_type)
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 <= col2
    // Row 0: 1<=4 (yes), Row 1: 5<=3 (no), Row 2: 3<=6 (yes), Row 3: 7<=2 (no)
    // Row 4: 2<=8 (yes), Row 5: 8<=1 (no), Row 6: 4<=9 (yes), Row 7: 9<=0 (no)
    std::vector<std::vector<int>> expected_data = {
        {1, 3, 2, 4},  // col1
        {4, 6, 8, 9}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 7: Three column predicate
// Predicate: (col1 + col2) * col3 > 20
void test_select_three_columns() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5},  // col1
        {2, 3, 4, 5, 6},  // col2
        {1, 2, 3, 2, 1}   // col3
    };
    auto table = make_test_table(input_data);

    // Build expression: (col[0] + col[1]) * col[2] > 20
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::greater_than(
        ExprTree::multiply(
            ExprTree::add(
                ExprTree::column(0, int_type),
                ExprTree::column(1, int_type)
            ),
            ExprTree::column(2, int_type)
        ),
        ExprTree::constant(gamax::scalar(20))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where (col1 + col2) * col3 > 20
    // Row 0: (1+2)*1=3 (no), Row 1: (2+3)*2=10 (no), Row 2: (3+4)*3=21 (yes)
    // Row 3: (4+5)*2=18 (no), Row 4: (5+6)*1=11 (no)
    std::vector<std::vector<int>> expected_data = {
        {3},  // col1
        {4},  // col2
        {3}   // col3
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 8: Four column complex predicate with medium selectivity
// Predicate: col1 + col2 == col3 + col4
void test_select_four_columns() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5, 6},  // col1
        {2, 3, 4, 5, 6, 7},  // col2
        {0, 1, 2, 3, 4, 5},  // col3
        {3, 4, 5, 6, 7, 8}   // col4
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] + col[1] == col[2] + col[3]
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::equal(
        ExprTree::add(
            ExprTree::column(0, int_type),
            ExprTree::column(1, int_type)
        ),
        ExprTree::add(
            ExprTree::column(2, int_type),
            ExprTree::column(3, int_type)
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 + col2 == col3 + col4
    // All rows satisfy this condition
    std::vector<std::vector<int>> expected_data = {
        {1, 2, 3, 4, 5, 6},  // col1
        {2, 3, 4, 5, 6, 7},  // col2
        {0, 1, 2, 3, 4, 5},  // col3
        {3, 4, 5, 6, 7, 8}   // col4
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 9: Edge case - all rows selected (100% selectivity)
void test_select_all_rows() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] >= 0 (all values are >= 0)
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::greater_equal(
        ExprTree::column(0, int_type),
        ExprTree::constant(gamax::scalar(0))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: all rows
    ASSERT_TRUE(verify_table_result(result->view(), input_data));
}

// Test 10: Edge case - no rows selected (0% selectivity)
void test_select_no_rows() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] < 0 (all values are >= 1)
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::less_than(
        ExprTree::column(0, int_type),
        ExprTree::constant(gamax::scalar(0))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: no rows
    ASSERT_EQ(result->num_rows(), 0);
}

// Test 11: AND predicate with two conditions
// Predicate: col1 > 2 AND col1 < 7
void test_select_and_predicate() {
    std::vector<std::vector<int>> input_data = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] > 2 AND col[0] < 7
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::logical_and(
        ExprTree::greater_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(2))
        ),
        ExprTree::less_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(7))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col[0] > 2 AND col[0] < 7 (i.e., 3, 4, 5, 6)
    std::vector<std::vector<int>> expected_data = {
        {3, 4, 5, 6}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 12: OR predicate with two conditions
// Predicate: col1 < 2 OR col1 > 8
void test_select_or_predicate() {
    std::vector<std::vector<int>> input_data = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] < 2 OR col[0] > 8
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::logical_or(
        ExprTree::less_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(2))
        ),
        ExprTree::greater_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(8))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col[0] < 2 OR col[0] > 8 (i.e., 0, 1, 9, 10)
    std::vector<std::vector<int>> expected_data = {
        {0, 1, 9, 10}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 13: AND predicate on two columns
// Predicate: col1 > 3 AND col2 < 5
void test_select_and_two_columns() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5, 6, 7, 8},  // col1
        {8, 7, 6, 5, 4, 3, 2, 1}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] > 3 AND col[1] < 5
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::logical_and(
        ExprTree::greater_than(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(3))
        ),
        ExprTree::less_than(
            ExprTree::column(1, int_type),
            ExprTree::constant(gamax::scalar(5))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 > 3 AND col2 < 5
    std::vector<std::vector<int>> expected_data = {
        {5, 6, 7, 8},  // col1
        {4, 3, 2, 1}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 14: OR predicate on two columns
// Predicate: col1 <= 2 OR col2 >= 7
void test_select_or_two_columns() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5, 6, 7, 8},  // col1
        {8, 7, 6, 5, 4, 3, 2, 1}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] <= 2 OR col[1] >= 7
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::logical_or(
        ExprTree::less_equal(
            ExprTree::column(0, int_type),
            ExprTree::constant(gamax::scalar(2))
        ),
        ExprTree::greater_equal(
            ExprTree::column(1, int_type),
            ExprTree::constant(gamax::scalar(7))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 <= 2 OR col2 >= 7
    std::vector<std::vector<int>> expected_data = {
        {1, 2},  // col1
        {8, 7}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 15: Complex nested predicate - (col1 > 2 AND col1 < 6) OR col2 == 9
void test_select_nested_and_or() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5, 6, 7, 8},  // col1
        {9, 0, 1, 2, 3, 4, 5, 9}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: (col[0] > 2 AND col[0] < 6) OR col[1] == 9
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::logical_or(
        ExprTree::logical_and(
            ExprTree::greater_than(
                ExprTree::column(0, int_type),
                ExprTree::constant(gamax::scalar(2))
            ),
            ExprTree::less_than(
                ExprTree::column(0, int_type),
                ExprTree::constant(gamax::scalar(6))
            )
        ),
        ExprTree::equal(
            ExprTree::column(1, int_type),
            ExprTree::constant(gamax::scalar(9))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where (col1 > 2 AND col1 < 6) OR col2 == 9
    std::vector<std::vector<int>> expected_data = {
        {1, 3, 4, 5, 8},  // col1
        {9, 1, 2, 3, 9}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 16: Complex nested predicate - (col1 < 3 OR col1 > 7) AND col2 >= 5
void test_select_nested_or_and() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9},  // col1
        {3, 5, 7, 4, 6, 8, 2, 9, 5}   // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: (col[0] < 3 OR col[0] > 7) AND col[1] >= 5
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::logical_and(
        ExprTree::logical_or(
            ExprTree::less_than(
                ExprTree::column(0, int_type),
                ExprTree::constant(gamax::scalar(3))
            ),
            ExprTree::greater_than(
                ExprTree::column(0, int_type),
                ExprTree::constant(gamax::scalar(7))
            )
        ),
        ExprTree::greater_equal(
            ExprTree::column(1, int_type),
            ExprTree::constant(gamax::scalar(5))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where (col1 < 3 OR col1 > 7) AND col2 >= 5
    std::vector<std::vector<int>> expected_data = {
        {2, 8, 9},  // col1
        {5, 9, 5}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 17: Three-way AND predicate
// Predicate: col1 > 2 AND col2 < 8 AND col3 >= 5
void test_select_three_way_and() {
    std::vector<std::vector<int>> input_data = {
        {1, 3, 4, 5, 6, 7},  // col1
        {9, 7, 6, 5, 8, 3},  // col2
        {4, 5, 6, 7, 4, 8}   // col3
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] > 2 AND col[1] < 8 AND col[2] >= 5
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = ExprTree::logical_and(
        ExprTree::logical_and(
            ExprTree::greater_than(
                ExprTree::column(0, int_type),
                ExprTree::constant(gamax::scalar(2))
            ),
            ExprTree::less_than(
                ExprTree::column(1, int_type),
                ExprTree::constant(gamax::scalar(8))
            )
        ),
        ExprTree::greater_equal(
            ExprTree::column(2, int_type),
            ExprTree::constant(gamax::scalar(5))
        )
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 > 2 AND col2 < 8 AND col3 >= 5
    std::vector<std::vector<int>> expected_data = {
        {3, 4, 5, 7},  // col1
        {7, 6, 5, 3},  // col2
        {5, 6, 7, 8}   // col3
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 18: Using operator overloading for natural syntax
void test_select_with_operators() {
    std::vector<std::vector<int>> input_data = {
        {1, 2, 3, 4, 5},  // col1
        {2, 3, 4, 5, 6},  // col2
    };
    auto table = make_test_table(input_data);

    // Build expression using operators: col[0] + col[1] > 5
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = (ExprTree::column(0, int_type) + ExprTree::column(1, int_type)) 
              > ExprTree::constant(gamax::scalar(5));

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col1 + col2 > 5
    // Row 0: 1+2=3 (no), Row 1: 2+3=5 (no), Row 2: 3+4=7 (yes)
    // Row 3: 4+5=9 (yes), Row 4: 5+6=11 (yes)
    std::vector<std::vector<int>> expected_data = {
        {3, 4, 5},  // col1
        {4, 5, 6}   // col2
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 19: Using operator overloading with logical operators
void test_select_with_logical_operators() {
    std::vector<std::vector<int>> input_data = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    };
    auto table = make_test_table(input_data);

    // Build expression using operators: col[0] > 2 && col[0] < 7
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = (ExprTree::column(0, int_type) > ExprTree::constant(gamax::scalar(2)))
             && (ExprTree::column(0, int_type) < ExprTree::constant(gamax::scalar(7)));

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col[0] > 2 AND col[0] < 7 (i.e., 3, 4, 5, 6)
    std::vector<std::vector<int>> expected_data = {
        {3, 4, 5, 6}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test 20: Complex expression with shift operators
void test_select_with_shift_operators() {
    std::vector<std::vector<int>> input_data = {
        {8, 16, 32, 64, 128}
    };
    auto table = make_test_table(input_data);

    // Build expression: col[0] >> 2 > 10
    auto int_type = gamax::data_type(gamax::type_id::INT32);
    auto expr = (ExprTree::column(0, int_type) >> 2) > ExprTree::constant(gamax::scalar(10));

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where col[0] >> 2 > 10
    // Row 0: 8>>2=2 (no), Row 1: 16>>2=4 (no), Row 2: 32>>2=8 (no)
    // Row 3: 64>>2=16 (yes), Row 4: 128>>2=32 (yes)
    std::vector<std::vector<int>> expected_data = {
        {64, 128}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

} // anonymous namespace

// Google Test cases
TEST(SelectFastTest, FloatArithmetic) {
    test_select_float_arithmetic();
}

TEST(SelectFastTest, DoubleComparison) {
    test_select_double_comparison();
}

TEST(SelectFastTest, FloatComplex) {
    test_select_float_complex();
}

TEST(SelectFastTest, DoubleDivision) {
    test_select_double_division();
}

TEST(SelectFastTest, SingleColumnHighSelectivity) {
    test_select_single_column_high_selectivity();
}

TEST(SelectFastTest, SingleColumnLowSelectivity) {
    test_select_single_column_low_selectivity();
}

TEST(SelectFastTest, SingleColumnMediumSelectivity) {
    test_select_single_column_medium_selectivity();
}

TEST(SelectFastTest, TwoColumnsHighSelectivity) {
    test_select_two_columns_high_selectivity();
}

TEST(SelectFastTest, TwoColumnsLowSelectivity) {
    test_select_two_columns_low_selectivity();
}

TEST(SelectFastTest, TwoColumnsMediumSelectivity) {
    test_select_two_columns_medium_selectivity();
}

TEST(SelectFastTest, ThreeColumns) {
    test_select_three_columns();
}

TEST(SelectFastTest, FourColumns) {
    test_select_four_columns();
}

TEST(SelectFastTest, AllRowsSelected) {
    test_select_all_rows();
}

TEST(SelectFastTest, NoRowsSelected) {
    test_select_no_rows();
}

TEST(SelectFastTest, AndPredicate) {
    test_select_and_predicate();
}

TEST(SelectFastTest, OrPredicate) {
    test_select_or_predicate();
}

TEST(SelectFastTest, AndTwoColumns) {
    test_select_and_two_columns();
}

TEST(SelectFastTest, OrTwoColumns) {
    test_select_or_two_columns();
}

TEST(SelectFastTest, NestedAndOr) {
    test_select_nested_and_or();
}

TEST(SelectFastTest, NestedOrAnd) {
    test_select_nested_or_and();
}

TEST(SelectFastTest, ThreeWayAnd) {
    test_select_three_way_and();
}

TEST(SelectFastTest, WithOperators) {
    test_select_with_operators();
}

TEST(SelectFastTest, WithLogicalOperators) {
    test_select_with_logical_operators();
}

TEST(SelectFastTest, WithShiftOperators) {
    test_select_with_shift_operators();
}

// Test: Select with sqrt function
TEST(SelectFastTest, WithSqrtFunction) {
    std::vector<std::vector<double>> input_data = {
        {4.0, 9.0, 16.0, 25.0, 36.0}
    };
    auto table = make_test_table(input_data);

    // Build expression: sqrt(col[0]) > 4.0
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::greater_than(
        ExprTree::sqrt(ExprTree::column(0, double_type)),
        ExprTree::constant(gamax::scalar(4.0))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where sqrt(col[0]) > 4.0
    // sqrt(4)=2 (no), sqrt(9)=3 (no), sqrt(16)=4 (no), sqrt(25)=5 (yes), sqrt(36)=6 (yes)
    std::vector<std::vector<double>> expected_data = {
        {25.0, 36.0}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test: Select with log function
TEST(SelectFastTest, WithLogFunction) {
    std::vector<std::vector<double>> input_data = {
        {1.0, 2.718281828459045, 7.389056098930650, 20.085536923187668, 54.598150033144236}
    };
    auto table = make_test_table(input_data);

    // Build expression: log(col[0]) >= 2.0
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::greater_equal(
        ExprTree::log(ExprTree::column(0, double_type)),
        ExprTree::constant(gamax::scalar(2.0))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where log(col[0]) >= 2.0
    // log(1)=0 (no), log(e)=1 (no), log(e^2)=2 (yes), log(e^3)=3 (yes), log(e^4)=4 (yes)
    std::vector<std::vector<double>> expected_data = {
        {7.389056098930650, 20.085536923187668, 54.598150033144236}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test: Select with exp function
TEST(SelectFastTest, WithExpFunction) {
    std::vector<std::vector<double>> input_data = {
        {0.0, 0.5, 1.0, 1.5, 2.0}
    };
    auto table = make_test_table(input_data);

    // Build expression: exp(col[0]) > 3.0
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::greater_than(
        ExprTree::exp(ExprTree::column(0, double_type)),
        ExprTree::constant(gamax::scalar(3.0))
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where exp(col[0]) > 3.0
    // exp(0)=1 (no), exp(0.5)≈1.65 (no), exp(1)≈2.72 (no), exp(1.5)≈4.48 (yes), exp(2)≈7.39 (yes)
    std::vector<std::vector<double>> expected_data = {
        {1.5, 2.0}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

// Test: Complex select with math functions combined
TEST(SelectFastTest, ComplexMathFunctions) {
    std::vector<std::vector<double>> input_data = {
        {1.0, 4.0, 9.0, 16.0, 25.0},  // col1
        {1.0, 2.0, 3.0, 4.0, 5.0}     // col2
    };
    auto table = make_test_table(input_data);

    // Build expression: sqrt(col[0]) == col[1]
    auto double_type = gamax::data_type(gamax::type_id::FLOAT64);
    auto expr = ExprTree::equal(
        ExprTree::sqrt(ExprTree::column(0, double_type)),
        ExprTree::column(1, double_type)
    );

    auto result = gamax::select::select(table->view(), expr);

    // Expected: rows where sqrt(col[0]) == col[1]
    // All rows should match
    std::vector<std::vector<double>> expected_data = {
        {1.0, 4.0, 9.0, 16.0, 25.0},
        {1.0, 2.0, 3.0, 4.0, 5.0}
    };

    ASSERT_TRUE(verify_table_result(result->view(), expected_data));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
