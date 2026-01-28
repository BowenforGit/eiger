#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include "operators/sort/sort.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"
#include "tests/test_helper.hpp"

namespace gamax {
namespace test {

class SortTableTest : public ::testing::Test {
protected:
    void SetUp() override { 
        cudaStreamCreate(&stream_); 
    }
    
    void TearDown() override { 
        cudaStreamDestroy(stream_); 
    }

    cudaStream_t stream_{};

    // Helper to create string column from vector of strings
    static std::unique_ptr<gamax::string_column> makeStringColumn(
        const std::vector<std::string>& strings,
        cudaStream_t stream = cudaStreamDefault)
    {
        std::vector<char> flat;
        std::vector<int32_t> lens;
        for (const auto& s : strings) {
            lens.push_back(static_cast<int32_t>(s.size()));
            flat.insert(flat.end(), s.begin(), s.end());
        }
        auto col = gamax::make_string_column(
            flat.data(),
            lens.data(),
            static_cast<size_type>(strings.size()),
            stream,
            gamax::ARROW_STRING);
        return std::unique_ptr<gamax::string_column>(
            static_cast<gamax::string_column*>(col.release()));
    }

    // Helper to verify string column content
    static std::vector<std::string> getStringColumnContent(
        const gamax::string_column& col,
        cudaStream_t stream = cudaStreamDefault)
    {
        using offset_t = gamax::size_type;
        
        std::vector<offset_t> offsets(col.size() + 1);
        cudaMemcpyAsync(
            offsets.data(),
            col.meta_buffer().data(),
            (col.size() + 1) * sizeof(offset_t),
            cudaMemcpyDeviceToHost, stream);
        
        std::vector<char> data_buffer(col.bytes());
        cudaMemcpyAsync(
            data_buffer.data(),
            col.buffer().data(),
            col.bytes(),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        std::vector<std::string> result;
        for (size_t i = 0; i < static_cast<size_t>(col.size()); ++i) {
            offset_t start = offsets[i];
            offset_t end = offsets[i + 1];
            result.emplace_back(data_buffer.begin() + start, data_buffer.begin() + end);
        }
        return result;
    }
};

// =============================================================================
// Test Case 1: Numeric key with numeric non-key columns (ASCENDING)
// =============================================================================
TEST_F(SortTableTest, NumericKeyWithNumericColumns_Ascending) {
    // Key column: [3, 1, 4, 1, 5]
    // Val column: [30, 10, 40, 11, 50]
    int32_t keys[] = {3, 1, 4, 1, 5};
    int32_t vals[] = {30, 10, 40, 11, 50};
    
    auto key_col = gamax::make_column_from_host(keys, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 2);
    EXPECT_EQ(result->num_rows(), 5);
    
    // Expected sorted by key ascending: [1, 1, 3, 4, 5]
    // Corresponding values: [10, 11, 30, 40, 50] (stable sort behavior may vary)
    int32_t expected_keys[] = {1, 1, 3, 4, 5};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 5);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    
    // Verify values are permuted correctly (sum should be preserved)
    auto result_vals = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    cudaStreamSynchronize(stream_);
    int32_t sum = 0;
    for (int i = 0; i < 5; ++i) sum += result_vals[i];
    EXPECT_EQ(sum, 30 + 10 + 40 + 11 + 50);
}

// =============================================================================
// Test Case 2: Numeric key with numeric non-key columns (DESCENDING)
// =============================================================================
TEST_F(SortTableTest, NumericKeyWithNumericColumns_Descending) {
    int32_t keys[] = {3, 1, 4, 1, 5};
    int32_t vals[] = {30, 10, 40, 11, 50};
    
    auto key_col = gamax::make_column_from_host(keys, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::DESCENDING, stream_);
    
    // Expected sorted by key descending: [5, 4, 3, 1, 1]
    int32_t expected_keys[] = {5, 4, 3, 1, 1};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 5);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
}

// =============================================================================
// Test Case 3: Numeric key with string non-key columns
// =============================================================================
TEST_F(SortTableTest, NumericKeyWithStringColumns) {
    // Key column (numeric): [3, 1, 2]
    // Val column (string): ["cherry", "apple", "banana"]
    int32_t keys[] = {3, 1, 2};
    std::vector<std::string> vals = {"cherry", "apple", "banana"};
    
    auto key_col = gamax::make_column_from_host(keys, 3);
    auto val_col = makeStringColumn(vals, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 2);
    
    // Expected sorted by key: [1, 2, 3] -> strings: ["apple", "banana", "cherry"]
    int32_t expected_keys[] = {1, 2, 3};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3);
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    
    // Verify string column
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(1));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    
    std::vector<std::string> expected_strings = {"apple", "banana", "cherry"};
    ASSERT_EQ(result_strings.size(), expected_strings.size());
    for (size_t i = 0; i < expected_strings.size(); ++i) {
        EXPECT_EQ(result_strings[i], expected_strings[i]) << "Mismatch at index " << i;
    }
}

// =============================================================================
// Test Case 4: String key with numeric non-key columns
// =============================================================================
TEST_F(SortTableTest, StringKeyWithNumericColumns) {
    // Key column (string): ["cherry", "apple", "banana"]
    // Val column (numeric): [30, 10, 20]
    std::vector<std::string> keys = {"cherry", "apple", "banana"};
    int32_t vals[] = {30, 10, 20};
    
    auto key_col = makeStringColumn(keys, stream_);
    auto val_col = gamax::make_column_from_host(vals, 3);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 2);
    
    // Expected sorted by string key: ["apple", "banana", "cherry"]
    // Corresponding values: [10, 20, 30]
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(0));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    
    std::vector<std::string> expected_keys_str = {"apple", "banana", "cherry"};
    ASSERT_EQ(result_strings.size(), expected_keys_str.size());
    for (size_t i = 0; i < expected_keys_str.size(); ++i) {
        EXPECT_EQ(result_strings[i], expected_keys_str[i]);
    }
    
    int32_t expected_vals[] = {10, 20, 30};
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3);
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 5: String key with string non-key columns
// =============================================================================
TEST_F(SortTableTest, StringKeyWithStringColumns) {
    // Key column: ["zebra", "apple", "mango"]
    // Val column: ["Z", "A", "M"]
    std::vector<std::string> keys = {"zebra", "apple", "mango"};
    std::vector<std::string> vals = {"Z", "A", "M"};
    
    auto key_col = makeStringColumn(keys, stream_);
    auto val_col = makeStringColumn(vals, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    // Expected sorted: ["apple", "mango", "zebra"] -> ["A", "M", "Z"]
    auto& result_key_col = dynamic_cast<const gamax::string_column&>(result->get_column(0));
    auto result_key_strings = getStringColumnContent(result_key_col, stream_);
    std::vector<std::string> expected_keys_str = {"apple", "mango", "zebra"};
    ASSERT_EQ(result_key_strings, expected_keys_str);
    
    auto& result_val_col = dynamic_cast<const gamax::string_column&>(result->get_column(1));
    auto result_val_strings = getStringColumnContent(result_val_col, stream_);
    std::vector<std::string> expected_vals_str = {"A", "M", "Z"};
    ASSERT_EQ(result_val_strings, expected_vals_str);
}

// =============================================================================
// Test Case 6: Mixed columns (numeric key, numeric + string non-key)
// =============================================================================
TEST_F(SortTableTest, NumericKeyWithMixedColumns) {
    // Key (numeric): [2, 1, 3]
    // Col1 (numeric): [20, 10, 30]
    // Col2 (string): ["two", "one", "three"]
    int32_t keys[] = {2, 1, 3};
    int32_t col1[] = {20, 10, 30};
    std::vector<std::string> col2 = {"two", "one", "three"};
    
    auto key_col = gamax::make_column_from_host(keys, 3);
    auto col1_ptr = gamax::make_column_from_host(col1, 3);
    auto col2_ptr = makeStringColumn(col2, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(col1_ptr));
    columns.push_back(std::move(col2_ptr));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    // Expected: keys [1, 2, 3], col1 [10, 20, 30], col2 ["one", "two", "three"]
    int32_t expected_keys[] = {1, 2, 3};
    int32_t expected_col1[] = {10, 20, 30};
    std::vector<std::string> expected_col2 = {"one", "two", "three"};
    
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3);
    auto expected_col1_ptr = gamax::make_column_from_host(expected_col1, 3);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_col1_ptr->view()));
    
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(2));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    ASSERT_EQ(result_strings, expected_col2);
}

// =============================================================================
// Test Case 7: Empty table
// =============================================================================
TEST_F(SortTableTest, EmptyTable) {
    auto key_col = gamax::make_column<int32_t>(0, false, stream_);
    auto val_col = gamax::make_column<int32_t>(0, false, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 2);
    EXPECT_EQ(result->num_rows(), 0);
}

// =============================================================================
// Test Case 8: Single column table (only key, no non-key columns)
// =============================================================================
TEST_F(SortTableTest, SingleNumericColumn) {
    int32_t keys[] = {5, 2, 8, 1, 9};
    auto key_col = gamax::make_column_from_host(keys, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    int32_t expected[] = {1, 2, 5, 8, 9};
    auto expected_col = gamax::make_column_from_host(expected, 5);
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_col->view()));
}

// =============================================================================
// Test Case 9: Single string column
// =============================================================================
TEST_F(SortTableTest, SingleStringColumn) {
    std::vector<std::string> keys = {"delta", "alpha", "charlie", "bravo"};
    auto key_col = makeStringColumn(keys, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    auto table = gamax::make_table(std::move(columns));
    
    auto result = gamax::sort::sort_table(table->view(), 0, gamax::sort_order::ASCENDING, stream_);
    
    auto& result_col = dynamic_cast<const gamax::string_column&>(result->get_column(0));
    auto result_strings = getStringColumnContent(result_col, stream_);
    
    std::vector<std::string> expected = {"alpha", "bravo", "charlie", "delta"};
    ASSERT_EQ(result_strings, expected);
}

// =============================================================================
// Test Case 10: Multi-key sorting with two numeric keys
// =============================================================================
TEST_F(SortTableTest, MultiKeySort_TwoNumericKeys) {
    // Primary key: [1, 1, 2, 2, 1]
    // Secondary key: [3, 1, 2, 1, 2]
    // Value column: [13, 11, 22, 21, 12]
    int32_t key1[] = {1, 1, 2, 2, 1};
    int32_t key2[] = {3, 1, 2, 1, 2};
    int32_t vals[] = {13, 11, 22, 21, 12};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, stream_);
    
    // Expected order by (key1, key2): 
    // (1,1)->11, (1,2)->12, (1,3)->13, (2,1)->21, (2,2)->22
    int32_t expected_key1[] = {1, 1, 1, 2, 2};
    int32_t expected_key2[] = {1, 2, 3, 1, 2};
    int32_t expected_vals[] = {11, 12, 13, 21, 22};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 5);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 5);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 5);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 11: Multi-key sorting with mixed sort orders
// =============================================================================
TEST_F(SortTableTest, MultiKeySort_MixedSortOrders) {
    // Primary key (ASC): [1, 1, 2, 2]
    // Secondary key (DESC): [1, 2, 1, 2]
    // Value column: [11, 12, 21, 22]
    int32_t key1[] = {1, 1, 2, 2};
    int32_t key2[] = {1, 2, 1, 2};
    int32_t vals[] = {11, 12, 21, 22};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING,   // key1 ASC
        gamax::sort_order::DESCENDING   // key2 DESC
    };
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, stream_);
    
    // Expected order: (1,2)->12, (1,1)->11, (2,2)->22, (2,1)->21
    int32_t expected_key1[] = {1, 1, 2, 2};
    int32_t expected_key2[] = {2, 1, 2, 1};
    int32_t expected_vals[] = {12, 11, 22, 21};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 12: Multi-key sorting with string and numeric keys
// =============================================================================
TEST_F(SortTableTest, MultiKeySort_StringAndNumericKeys) {
    // String key (primary): ["b", "a", "b", "a"]
    // Numeric key (secondary): [2, 2, 1, 1]
    // Value column: [12, 22, 11, 21]
    std::vector<std::string> key1 = {"b", "a", "b", "a"};
    int32_t key2[] = {2, 2, 1, 1};
    int32_t vals[] = {12, 22, 11, 21};
    
    auto key1_col = makeStringColumn(key1, stream_);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, stream_);
    
    // Expected order: (a,1)->21, (a,2)->22, (b,1)->11, (b,2)->12
    std::vector<std::string> expected_key1 = {"a", "a", "b", "b"};
    int32_t expected_key2[] = {1, 2, 1, 2};
    int32_t expected_vals[] = {21, 22, 11, 12};
    
    auto& result_key1_col = dynamic_cast<const gamax::string_column&>(result->get_column(0));
    auto result_key1_strings = getStringColumnContent(result_key1_col, stream_);
    ASSERT_EQ(result_key1_strings, expected_key1);
    
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 13: Multi-key sorting with empty keys (should throw)
// =============================================================================
TEST_F(SortTableTest, MultiKeySorting_EmptyKeys_Throws) {
    int32_t col1[] = {1, 2, 3};
    auto col1_ptr = gamax::make_column_from_host(col1, 3);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1_ptr));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> empty_keys = {};
    std::vector<gamax::sort_order> empty_orders = {};
    
    EXPECT_THROW(
        gamax::sort::sort_table(table->view(), empty_keys, empty_orders, stream_),
        std::invalid_argument
    );
}

// =============================================================================
// Test Case 14: Multi-key sorting with mismatched sizes (should throw)
// =============================================================================
TEST_F(SortTableTest, MultiKeySorting_MismatchedSizes_Throws) {
    int32_t col1[] = {1, 2, 3};
    auto col1_ptr = gamax::make_column_from_host(col1, 3);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1_ptr));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> keys = {0};
    std::vector<gamax::sort_order> orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::DESCENDING  // Extra order
    };
    
    EXPECT_THROW(
        gamax::sort::sort_table(table->view(), keys, orders, stream_),
        std::invalid_argument
    );
}

// =============================================================================
// Test Case 15: Multi-key with single key falls back to single-key sort
// =============================================================================
TEST_F(SortTableTest, MultiKeySorting_SingleKey_FallsBack) {
    int32_t keys[] = {3, 1, 2};
    int32_t vals[] = {30, 10, 20};
    
    auto key_col = gamax::make_column_from_host(keys, 3);
    auto val_col = gamax::make_column_from_host(vals, 3);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::ASCENDING};
    
    // Should not throw - falls back to single key sorting
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, stream_);
    
    int32_t expected_keys[] = {1, 2, 3};
    int32_t expected_vals[] = {10, 20, 30};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_val_col->view()));
}

// =============================================================================
// =============================================================================
// SMART KEY FUSION TESTS
// Tests for sort_table with smart_keys_fusion option enabled
// =============================================================================
// =============================================================================

// =============================================================================
// Test Case 16: Basic smart key fusion with two int32 keys
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_BasicTwoKeys) {
    // Primary key: [1, 1, 2, 2, 1]
    // Secondary key: [3, 1, 2, 1, 2]
    // Value column: [13, 11, 22, 21, 12]
    int32_t key1[] = {1, 1, 2, 2, 1};
    int32_t key2[] = {3, 1, 2, 1, 2};
    int32_t vals[] = {13, 11, 22, 21, 12};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected order by (key1, key2): 
    // (1,1)->11, (1,2)->12, (1,3)->13, (2,1)->21, (2,2)->22
    int32_t expected_key1[] = {1, 1, 1, 2, 2};
    int32_t expected_key2[] = {1, 2, 3, 1, 2};
    int32_t expected_vals[] = {11, 12, 13, 21, 22};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 5);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 5);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 5);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 17: Smart key fusion with signed integers (negative values)
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_SignedIntegers) {
    // Keys with negative values
    int32_t key1[] = {-5, 0, 5, -5, 0};
    int32_t key2[] = {2, -1, 0, 1, 1};
    int32_t vals[] = {-52, 0-1, 50, -51, 01};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected sorted order: (-5,1)->-51, (-5,2)->-52, (0,-1)->-1, (0,1)->1, (5,0)->50
    int32_t expected_key1[] = {-5, -5, 0, 0, 5};
    int32_t expected_key2[] = {1, 2, -1, 1, 0};
    int32_t expected_vals[] = {-51, -52, -1, 1, 50};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 5);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 5);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 5);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 18: Smart key fusion with int8 and int16 (narrow types)
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_NarrowTypes) {
    // int8 key1, int16 key2
    int8_t key1[] = {2, 1, 2, 1, 1};
    int16_t key2[] = {100, 200, 50, 100, 150};
    int32_t vals[] = {1, 2, 3, 4, 5};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected: (1,100)->4, (1,150)->5, (1,200)->2, (2,50)->3, (2,100)->1
    int8_t expected_key1[] = {1, 1, 1, 2, 2};
    int16_t expected_key2[] = {100, 150, 200, 50, 100};
    int32_t expected_vals[] = {4, 5, 2, 3, 1};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 5);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 5);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 5);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 19: Smart key fusion with int64 keys
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_Int64Keys) {
    // int64 keys - will use range compression
    int64_t key1[] = {1000000000LL, 1000000001LL, 1000000000LL, 1000000001LL};
    int64_t key2[] = {2000000002LL, 2000000001LL, 2000000001LL, 2000000002LL};
    int32_t vals[] = {1, 2, 3, 4};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected: (1e9, 2e9+1)->3, (1e9, 2e9+2)->1, (1e9+1, 2e9+1)->2, (1e9+1, 2e9+2)->4
    int64_t expected_key1[] = {1000000000LL, 1000000000LL, 1000000001LL, 1000000001LL};
    int64_t expected_key2[] = {2000000001LL, 2000000002LL, 2000000001LL, 2000000002LL};
    int32_t expected_vals[] = {3, 1, 2, 4};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 20: Smart key fusion with mixed ASC/DESC orders
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_MixedOrders) {
    // Primary key (ASC): [1, 1, 2, 2]
    // Secondary key (DESC): [1, 2, 1, 2]
    int32_t key1[] = {1, 1, 2, 2};
    int32_t key2[] = {1, 2, 1, 2};
    int32_t vals[] = {11, 12, 21, 22};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING,   // key1 ASC
        gamax::sort_order::DESCENDING   // key2 DESC
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected order: (1,2)->12, (1,1)->11, (2,2)->22, (2,1)->21
    int32_t expected_key1[] = {1, 1, 2, 2};
    int32_t expected_key2[] = {2, 1, 2, 1};
    int32_t expected_vals[] = {12, 11, 22, 21};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 21: Smart key fusion with empty table
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_EmptyTable) {
    auto key1_col = gamax::make_column<int32_t>(0, false, stream_);
    auto key2_col = gamax::make_column<int32_t>(0, false, stream_);
    auto val_col = gamax::make_column<int32_t>(0, false, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 3);
    EXPECT_EQ(result->num_rows(), 0);
}

// =============================================================================
// Test Case 22: Smart key fusion with single row
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_SingleRow) {
    int32_t key1[] = {42};
    int32_t key2[] = {100};
    int32_t vals[] = {999};
    
    auto key1_col = gamax::make_column_from_host(key1, 1);
    auto key2_col = gamax::make_column_from_host(key2, 1);
    auto val_col = gamax::make_column_from_host(vals, 1);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    auto expected_key1_col = gamax::make_column_from_host(key1, 1);
    auto expected_key2_col = gamax::make_column_from_host(key2, 1);
    auto expected_val_col = gamax::make_column_from_host(vals, 1);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 23: Smart key fusion with all same values (0 bits needed)
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_AllSameValues) {
    // All rows have the same key values
    int32_t key1[] = {5, 5, 5, 5};
    int32_t key2[] = {10, 10, 10, 10};
    int32_t vals[] = {1, 2, 3, 4};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // All keys same, so any order is valid - just verify key values
    auto expected_key1_col = gamax::make_column_from_host(key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(key2, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    
    // Verify value sum is preserved (order may vary since all keys are equal)
    auto result_vals = gamax::copy_column_to_host<int32_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    int32_t sum = 0;
    for (int i = 0; i < 4; ++i) sum += result_vals[i];
    EXPECT_EQ(sum, 1 + 2 + 3 + 4);
}

// =============================================================================
// Test Case 24: Smart key fusion fallback - float key
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_FallbackFloatKey) {
    // Float keys are not fusable, should fall back to default sorting
    float key1[] = {3.14f, 1.41f, 2.71f};
    int32_t key2[] = {1, 2, 3};
    int32_t vals[] = {31, 14, 27};
    
    auto key1_col = gamax::make_column_from_host(key1, 3);
    auto key2_col = gamax::make_column_from_host(key2, 3);
    auto val_col = gamax::make_column_from_host(vals, 3);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    // Should fall back to default sort and still produce correct results
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 3);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Verify sorted order: 1.41 < 2.71 < 3.14
    float expected_key1[] = {1.41f, 2.71f, 3.14f};
    int32_t expected_vals[] = {14, 27, 31};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 3);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 25: Smart key fusion fallback - string key
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_FallbackStringKey) {
    // String keys are not fusable, should fall back to default sorting
    std::vector<std::string> key1 = {"cherry", "apple", "banana"};
    int32_t key2[] = {1, 2, 3};
    int32_t vals[] = {30, 10, 20};
    
    auto key1_col = makeStringColumn(key1, stream_);
    auto key2_col = gamax::make_column_from_host(key2, 3);
    auto val_col = gamax::make_column_from_host(vals, 3);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Verify sorted by string key: apple, banana, cherry
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(0));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    
    std::vector<std::string> expected_strings = {"apple", "banana", "cherry"};
    ASSERT_EQ(result_strings, expected_strings);
    
    int32_t expected_vals[] = {10, 20, 30};
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3);
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 26: Smart key fusion with low cardinality (dictionary encoding path)
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_LowCardinality) {
    // Generate keys with low cardinality but large range values
    // This should trigger dictionary encoding path
    // NDV = 4, but values span a wide range (would require many bits without dict)
    const int N = 100;
    std::vector<int32_t> key1_vec(N);
    std::vector<int32_t> key2_vec(N);
    std::vector<int32_t> vals_vec(N);
    
    // key1: cycles through 0, 1000000, 2000000, 3000000 (4 distinct values, wide range)
    // key2: cycles through 0, 100000, 200000, 300000 (4 distinct values, wide range)
    for (int i = 0; i < N; ++i) {
        key1_vec[i] = (i % 4) * 1000000;
        key2_vec[i] = ((i / 4) % 4) * 100000;
        vals_vec[i] = i;
    }
    
    auto key1_col = gamax::make_column_from_host(key1_vec.data(), N);
    auto key2_col = gamax::make_column_from_host(key2_vec.data(), N);
    auto val_col = gamax::make_column_from_host(vals_vec.data(), N);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Verify result is sorted
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), N);
    
    auto result_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto result_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    // Verify sorted order
    for (int i = 1; i < N; ++i) {
        bool is_ordered = (result_key1[i-1] < result_key1[i]) || 
                          (result_key1[i-1] == result_key1[i] && result_key2[i-1] <= result_key2[i]);
        EXPECT_TRUE(is_ordered) << "Order violation at index " << i 
                                << ": (" << result_key1[i-1] << "," << result_key2[i-1] 
                                << ") vs (" << result_key1[i] << "," << result_key2[i] << ")";
    }
}

// =============================================================================
// Test Case 27: Smart key fusion with three keys
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_ThreeKeys) {
    // Three keys to test multiple key fusion
    int8_t key1[] = {1, 1, 1, 2, 2, 2};
    int8_t key2[] = {1, 1, 2, 1, 1, 2};
    int8_t key3[] = {1, 2, 1, 1, 2, 1};
    int32_t vals[] = {111, 112, 121, 211, 212, 221};
    
    auto key1_col = gamax::make_column_from_host(key1, 6);
    auto key2_col = gamax::make_column_from_host(key2, 6);
    auto key3_col = gamax::make_column_from_host(key3, 6);
    auto val_col = gamax::make_column_from_host(vals, 6);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(key3_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1, 2};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING,
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected order by (key1, key2, key3): 
    // (1,1,1)->111, (1,1,2)->112, (1,2,1)->121, (2,1,1)->211, (2,1,2)->212, (2,2,1)->221
    int8_t expected_key1[] = {1, 1, 1, 2, 2, 2};
    int8_t expected_key2[] = {1, 1, 2, 1, 1, 2};
    int8_t expected_key3[] = {1, 2, 1, 1, 2, 1};
    int32_t expected_vals[] = {111, 112, 121, 211, 212, 221};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 6);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 6);
    auto expected_key3_col = gamax::make_column_from_host(expected_key3, 6);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 6);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_key3_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(3).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 28: Smart key fusion with non-key string column
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_WithStringNonKeyColumn) {
    // Numeric keys, string value column
    int32_t key1[] = {2, 1, 3, 1};
    int32_t key2[] = {2, 2, 1, 1};
    std::vector<std::string> vals = {"two-two", "one-two", "three-one", "one-one"};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = makeStringColumn(vals, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected: (1,1)->"one-one", (1,2)->"one-two", (2,2)->"two-two", (3,1)->"three-one"
    int32_t expected_key1[] = {1, 1, 2, 3};
    int32_t expected_key2[] = {1, 2, 2, 1};
    std::vector<std::string> expected_vals = {"one-one", "one-two", "two-two", "three-one"};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(2));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    ASSERT_EQ(result_strings, expected_vals);
}

// =============================================================================
// Test Case 29: Smart key fusion with large input
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_LargeInput) {
    const int N = 100000;  // Large enough to stress test
    
    std::vector<int32_t> key1_vec(N);
    std::vector<int32_t> key2_vec(N);
    std::vector<int32_t> vals_vec(N);
    
    // Generate random-looking but deterministic data
    for (int i = 0; i < N; ++i) {
        // Values that create a moderate range but compressible
        key1_vec[i] = (i * 17) % 1000;    // 0-999
        key2_vec[i] = (i * 31) % 500;     // 0-499
        vals_vec[i] = i;
    }
    
    auto key1_col = gamax::make_column_from_host(key1_vec.data(), N);
    auto key2_col = gamax::make_column_from_host(key2_vec.data(), N);
    auto val_col = gamax::make_column_from_host(vals_vec.data(), N);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), N);
    
    // Verify result is correctly sorted
    auto result_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto result_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    for (int i = 1; i < N; ++i) {
        bool is_ordered = (result_key1[i-1] < result_key1[i]) || 
                          (result_key1[i-1] == result_key1[i] && result_key2[i-1] <= result_key2[i]);
        EXPECT_TRUE(is_ordered) << "Order violation at index " << i;
    }
}

// =============================================================================
// Test Case 30: Smart key fusion with all descending
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_AllDescending) {
    int32_t key1[] = {1, 2, 1, 2};
    int32_t key2[] = {1, 1, 2, 2};
    int32_t vals[] = {11, 21, 12, 22};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::DESCENDING, 
        gamax::sort_order::DESCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected order (DESC, DESC): (2,2)->22, (2,1)->21, (1,2)->12, (1,1)->11
    int32_t expected_key1[] = {2, 2, 1, 1};
    int32_t expected_key2[] = {2, 1, 2, 1};
    int32_t expected_vals[] = {22, 21, 12, 11};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 31: Smart key fusion option disabled (verify fallback behavior)
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_Disabled) {
    int32_t key1[] = {2, 1, 2, 1};
    int32_t key2[] = {2, 2, 1, 1};
    int32_t vals[] = {22, 12, 21, 11};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = false;  // Explicitly disabled
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Should still produce correct results via default path
    int32_t expected_key1[] = {1, 1, 2, 2};
    int32_t expected_key2[] = {1, 2, 1, 2};
    int32_t expected_vals[] = {11, 12, 21, 22};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 32: Smart key fusion with unsigned integer types
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_UnsignedTypes) {
    uint32_t key1[] = {3, 1, 2, 1};
    uint16_t key2[] = {100, 300, 200, 200};
    int32_t vals[] = {1, 2, 3, 4};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected: (1,200)->4, (1,300)->2, (2,200)->3, (3,100)->1
    uint32_t expected_key1[] = {1, 1, 2, 3};
    uint16_t expected_key2[] = {200, 300, 200, 100};
    int32_t expected_vals[] = {4, 2, 3, 1};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 4);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 4);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 4);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 33: Smart key fusion with negative range spanning zero
// =============================================================================
TEST_F(SortTableTest, SmartKeyFusion_NegativeRangeSpanningZero) {
    // Keys that span from negative to positive values
    int32_t key1[] = {-10, 10, 0, -5, 5};
    int32_t key2[] = {-100, 100, 0, 50, -50};
    int32_t vals[] = {1, 2, 3, 4, 5};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key1_col));
    columns.push_back(std::move(key2_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0, 1};
    std::vector<gamax::sort_order> sort_orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::ASCENDING
    };
    
    gamax::sort::sort_table_options opts;
    opts.smart_keys_fusion = true;
    
    auto result = gamax::sort::sort_table(table->view(), sort_keys, sort_orders, opts, stream_);
    
    // Expected: (-10,-100)->1, (-5,50)->4, (0,0)->3, (5,-50)->5, (10,100)->2
    int32_t expected_key1[] = {-10, -5, 0, 5, 10};
    int32_t expected_key2[] = {-100, 50, 0, -50, 100};
    int32_t expected_vals[] = {1, 4, 3, 5, 2};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 5);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 5);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 5);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

} // namespace test
} // namespace gamax
