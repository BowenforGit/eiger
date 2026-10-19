// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

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

class TopKTableTest : public ::testing::Test {
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
// Test Case 1: TopK with single numeric key (DESCENDING) - uses DeviceTopK
// =============================================================================
TEST_F(TopKTableTest, SingleNumericKey_Descending_Sorted) {
    // Key column: [3, 1, 4, 1, 5, 9, 2, 6]
    // Val column: [30, 10, 40, 11, 50, 90, 20, 60]
    int32_t keys[] = {3, 1, 4, 1, 5, 9, 2, 6};
    int32_t vals[] = {30, 10, 40, 11, 50, 90, 20, 60};
    
    auto key_col = gamax::make_column_from_host(keys, 8, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 8, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 2);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Top 3 largest keys in sorted order: [9, 6, 5]
    // Corresponding values: [90, 60, 50]
    int32_t expected_keys[] = {9, 6, 5};
    int32_t expected_vals[] = {90, 60, 50};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3, cudaStreamDefault);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3, cudaStreamDefault);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 2: TopK with single numeric key (ASCENDING) - uses DeviceTopK
// =============================================================================
TEST_F(TopKTableTest, SingleNumericKey_Ascending_Sorted) {
    // Key column: [3, 1, 4, 1, 5, 9, 2, 6]
    // Val column: [30, 10, 40, 11, 50, 90, 20, 60]
    int32_t keys[] = {3, 1, 4, 1, 5, 9, 2, 6};
    int32_t vals[] = {30, 10, 40, 11, 50, 90, 20, 60};
    
    auto key_col = gamax::make_column_from_host(keys, 8, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 8, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::ASCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 2);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Top 3 smallest keys in sorted order: [1, 1, 2]
    // Check the sum of keys and values
    auto result_keys = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto result_vals = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    // Keys should be sorted ascending
    EXPECT_LE(result_keys[0], result_keys[1]);
    EXPECT_LE(result_keys[1], result_keys[2]);
    
    // The 3 smallest keys are 1, 1, 2
    int32_t key_sum = result_keys[0] + result_keys[1] + result_keys[2];
    EXPECT_EQ(key_sum, 1 + 1 + 2);
}

// =============================================================================
// Test Case 3: TopK with single numeric key (unsorted result)
// =============================================================================
TEST_F(TopKTableTest, SingleNumericKey_Unsorted) {
    int32_t keys[] = {3, 1, 4, 1, 5, 9, 2, 6};
    int32_t vals[] = {30, 10, 40, 11, 50, 90, 20, 60};
    
    auto key_col = gamax::make_column_from_host(keys, 8, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 8, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, false, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Results may be unsorted, but should contain the top 3 largest keys
    auto result_keys = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    std::vector<int32_t> sorted_result(result_keys.get(), result_keys.get() + 3);
    std::sort(sorted_result.begin(), sorted_result.end(), std::greater<int32_t>());
    
    // Should be 9, 6, 5 when sorted
    EXPECT_EQ(sorted_result[0], 9);
    EXPECT_EQ(sorted_result[1], 6);
    EXPECT_EQ(sorted_result[2], 5);
}

// =============================================================================
// Test Case 4: TopK with single numeric key and string non-key column
// =============================================================================
TEST_F(TopKTableTest, SingleNumericKey_WithStringColumn) {
    // Key (numeric): [3, 1, 4, 2, 5]
    // Val (string): ["three", "one", "four", "two", "five"]
    int32_t keys[] = {3, 1, 4, 2, 5};
    std::vector<std::string> vals = {"three", "one", "four", "two", "five"};
    
    auto key_col = gamax::make_column_from_host(keys, 5, cudaStreamDefault);
    auto val_col = makeStringColumn(vals, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Top 3 largest: [5, 4, 3] -> ["five", "four", "three"]
    int32_t expected_keys[] = {5, 4, 3};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3, cudaStreamDefault);
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(1));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    std::vector<std::string> expected_strings = {"five", "four", "three"};
    ASSERT_EQ(result_strings, expected_strings);
}

// =============================================================================
// Test Case 5: TopK with string key (falls back to full sort)
// =============================================================================
TEST_F(TopKTableTest, SingleStringKey_Sorted) {
    // Key (string): ["cherry", "apple", "banana", "date", "elderberry"]
    // Val (numeric): [30, 10, 20, 40, 50]
    std::vector<std::string> keys = {"cherry", "apple", "banana", "date", "elderberry"};
    int32_t vals[] = {30, 10, 20, 40, 50};
    
    auto key_col = makeStringColumn(keys, stream_);
    auto val_col = gamax::make_column_from_host(vals, 5, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::ASCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Top 3 smallest strings: ["apple", "banana", "cherry"]
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(0));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    std::vector<std::string> expected_strings = {"apple", "banana", "cherry"};
    ASSERT_EQ(result_strings, expected_strings);
    
    // Corresponding values: [10, 20, 30]
    int32_t expected_vals[] = {10, 20, 30};
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3, cudaStreamDefault);
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 6: TopK with multi-key (falls back to full sort)
// =============================================================================
TEST_F(TopKTableTest, MultiKey_Sorted) {
    // Primary key: [1, 1, 2, 2, 1]
    // Secondary key: [3, 1, 2, 1, 2]
    // Value column: [13, 11, 22, 21, 12]
    int32_t key1[] = {1, 1, 2, 2, 1};
    int32_t key2[] = {3, 1, 2, 1, 2};
    int32_t vals[] = {13, 11, 22, 21, 12};
    
    auto key1_col = gamax::make_column_from_host(key1, 5, cudaStreamDefault);
    auto key2_col = gamax::make_column_from_host(key2, 5, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 5, cudaStreamDefault);
    
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
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Full sorted order: (1,1)->11, (1,2)->12, (1,3)->13, (2,1)->21, (2,2)->22
    // Top 3: (1,1)->11, (1,2)->12, (1,3)->13
    int32_t expected_key1[] = {1, 1, 1};
    int32_t expected_key2[] = {1, 2, 3};
    int32_t expected_vals[] = {11, 12, 13};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 3, cudaStreamDefault);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 3, cudaStreamDefault);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3, cudaStreamDefault);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 7: TopK where k >= num_rows (returns full sorted table)
// =============================================================================
TEST_F(TopKTableTest, K_GreaterThanOrEqual_NumRows) {
    int32_t keys[] = {3, 1, 2};
    int32_t vals[] = {30, 10, 20};
    
    auto key_col = gamax::make_column_from_host(keys, 3, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 3, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::ASCENDING};
    
    // Request k=10 but only 3 rows exist
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 10, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);  // Should return all rows
    
    // Should be fully sorted
    int32_t expected_keys[] = {1, 2, 3};
    int32_t expected_vals[] = {10, 20, 30};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3, cudaStreamDefault);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3, cudaStreamDefault);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 8: TopK with k = 1
// =============================================================================
TEST_F(TopKTableTest, K_Equals_One) {
    int32_t keys[] = {3, 1, 4, 1, 5};
    int32_t vals[] = {30, 10, 40, 11, 50};
    
    auto key_col = gamax::make_column_from_host(keys, 5, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 5, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 1, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 1);
    
    // Largest key is 5
    int32_t expected_keys[] = {5};
    int32_t expected_vals[] = {50};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 1, cudaStreamDefault);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 1, cudaStreamDefault);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 9: TopK with empty keys (should throw)
// =============================================================================
TEST_F(TopKTableTest, EmptyKeys_Throws) {
    int32_t col1[] = {1, 2, 3};
    auto col1_ptr = gamax::make_column_from_host(col1, 3, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1_ptr));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> empty_keys = {};
    std::vector<gamax::sort_order> empty_orders = {};
    
    EXPECT_THROW(
        gamax::sort::topk_table(table->view(), empty_keys, empty_orders, 2, true, stream_),
        std::invalid_argument
    );
}

// =============================================================================
// Test Case 10: TopK with mismatched sizes (should throw)
// =============================================================================
TEST_F(TopKTableTest, MismatchedSizes_Throws) {
    int32_t col1[] = {1, 2, 3};
    auto col1_ptr = gamax::make_column_from_host(col1, 3, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1_ptr));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> keys = {0};
    std::vector<gamax::sort_order> orders = {
        gamax::sort_order::ASCENDING, 
        gamax::sort_order::DESCENDING  // Extra order
    };
    
    EXPECT_THROW(
        gamax::sort::topk_table(table->view(), keys, orders, 2, true, stream_),
        std::invalid_argument
    );
}

// =============================================================================
// Test Case 11: TopK with k <= 0 (should throw)
// =============================================================================
TEST_F(TopKTableTest, K_LessThanOrEqualZero_Throws) {
    int32_t col1[] = {1, 2, 3};
    auto col1_ptr = gamax::make_column_from_host(col1, 3, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(col1_ptr));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> keys = {0};
    std::vector<gamax::sort_order> orders = {gamax::sort_order::ASCENDING};
    
    EXPECT_THROW(
        gamax::sort::topk_table(table->view(), keys, orders, 0, true, stream_),
        std::invalid_argument
    );
    
    EXPECT_THROW(
        gamax::sort::topk_table(table->view(), keys, orders, -1, true, stream_),
        std::invalid_argument
    );
}

// =============================================================================
// Test Case 12: TopK with different numeric types (int64)
// =============================================================================
TEST_F(TopKTableTest, Int64Key) {
    int64_t keys[] = {300, 100, 400, 200, 500};
    int32_t vals[] = {30, 10, 40, 20, 50};
    
    auto key_col = gamax::make_column_from_host(keys, 5, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 5, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Top 3 largest: [500, 400, 300]
    int64_t expected_keys[] = {500, 400, 300};
    int32_t expected_vals[] = {50, 40, 30};
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3, cudaStreamDefault);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 3, cudaStreamDefault);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 13: TopK with float key
// =============================================================================
TEST_F(TopKTableTest, FloatKey) {
    float keys[] = {3.14f, 1.41f, 2.71f, 1.73f, 0.57f};
    int32_t vals[] = {314, 141, 271, 173, 57};
    
    auto key_col = gamax::make_column_from_host(keys, 5, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 5, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 2, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 2);
    
    // Top 2 largest: [3.14, 2.71]
    auto result_keys = gamax::copy_column_to_host<float>(result->get_column(0).view(), stream_);
    auto result_vals = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    EXPECT_FLOAT_EQ(result_keys[0], 3.14f);
    EXPECT_FLOAT_EQ(result_keys[1], 2.71f);
    EXPECT_EQ(result_vals[0], 314);
    EXPECT_EQ(result_vals[1], 271);
}

// =============================================================================
// Test Case 14: TopK with double key
// =============================================================================
TEST_F(TopKTableTest, DoubleKey) {
    double keys[] = {3.14159, 1.41421, 2.71828, 1.73205, 0.57721};
    int32_t vals[] = {314, 141, 271, 173, 57};
    
    auto key_col = gamax::make_column_from_host(keys, 5, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 5, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::ASCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 2, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 2);
    
    // Top 2 smallest: [0.57721, 1.41421]
    auto result_keys = gamax::copy_column_to_host<double>(result->get_column(0).view(), stream_);
    auto result_vals = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    EXPECT_DOUBLE_EQ(result_keys[0], 0.57721);
    EXPECT_DOUBLE_EQ(result_keys[1], 1.41421);
    EXPECT_EQ(result_vals[0], 57);
    EXPECT_EQ(result_vals[1], 141);
}

// =============================================================================
// Test Case 15: TopK with mixed columns (numeric key, numeric + string non-key)
// =============================================================================
TEST_F(TopKTableTest, NumericKeyWithMixedColumns) {
    // Key (numeric): [4, 2, 5, 1, 3]
    // Col1 (numeric): [40, 20, 50, 10, 30]
    // Col2 (string): ["four", "two", "five", "one", "three"]
    int32_t keys[] = {4, 2, 5, 1, 3};
    int32_t col1[] = {40, 20, 50, 10, 30};
    std::vector<std::string> col2 = {"four", "two", "five", "one", "three"};
    
    auto key_col = gamax::make_column_from_host(keys, 5, cudaStreamDefault);
    auto col1_ptr = gamax::make_column_from_host(col1, 5, cudaStreamDefault);
    auto col2_ptr = makeStringColumn(col2, stream_);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(col1_ptr));
    columns.push_back(std::move(col2_ptr));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::ASCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Top 3 smallest: keys [1, 2, 3], col1 [10, 20, 30], col2 ["one", "two", "three"]
    int32_t expected_keys[] = {1, 2, 3};
    int32_t expected_col1[] = {10, 20, 30};
    std::vector<std::string> expected_col2 = {"one", "two", "three"};
    
    auto expected_key_col = gamax::make_column_from_host(expected_keys, 3, cudaStreamDefault);
    auto expected_col1_ptr = gamax::make_column_from_host(expected_col1, 3, cudaStreamDefault);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_col1_ptr->view()));
    
    auto& result_str_col = dynamic_cast<const gamax::string_column&>(result->get_column(2));
    auto result_strings = getStringColumnContent(result_str_col, stream_);
    ASSERT_EQ(result_strings, expected_col2);
}

// =============================================================================
// Test Case 16: TopK with multi-key and mixed sort orders
// =============================================================================
TEST_F(TopKTableTest, MultiKey_MixedSortOrders) {
    // Primary key (ASC): [1, 1, 2, 2]
    // Secondary key (DESC): [1, 2, 1, 2]
    // Value column: [11, 12, 21, 22]
    int32_t key1[] = {1, 1, 2, 2};
    int32_t key2[] = {1, 2, 1, 2};
    int32_t vals[] = {11, 12, 21, 22};
    
    auto key1_col = gamax::make_column_from_host(key1, 4, cudaStreamDefault);
    auto key2_col = gamax::make_column_from_host(key2, 4, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 4, cudaStreamDefault);
    
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
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 2, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 2);
    
    // Full sort order: (1,2)->12, (1,1)->11, (2,2)->22, (2,1)->21
    // Top 2: (1,2)->12, (1,1)->11
    int32_t expected_key1[] = {1, 1};
    int32_t expected_key2[] = {2, 1};
    int32_t expected_vals[] = {12, 11};
    
    auto expected_key1_col = gamax::make_column_from_host(expected_key1, 2, cudaStreamDefault);
    auto expected_key2_col = gamax::make_column_from_host(expected_key2, 2, cudaStreamDefault);
    auto expected_val_col = gamax::make_column_from_host(expected_vals, 2, cudaStreamDefault);
    
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_key1_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(1).view(), expected_key2_col->view()));
    EXPECT_TRUE(check_cols_eq(result->get_column(2).view(), expected_val_col->view()));
}

// =============================================================================
// Test Case 17: TopK with single column table (only key)
// =============================================================================
TEST_F(TopKTableTest, SingleColumnTable) {
    int32_t keys[] = {5, 2, 8, 1, 9, 3};
    auto key_col = gamax::make_column_from_host(keys, 6, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 3, true, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 1);
    EXPECT_EQ(result->num_rows(), 3);
    
    // Top 3 largest: [9, 8, 5]
    int32_t expected[] = {9, 8, 5};
    auto expected_col = gamax::make_column_from_host(expected, 3, cudaStreamDefault);
    EXPECT_TRUE(check_cols_eq(result->get_column(0).view(), expected_col->view()));
}

// =============================================================================
// Test Case 18: TopK preserves key-value relationship
// =============================================================================
TEST_F(TopKTableTest, PreservesKeyValueRelationship) {
    // Each key has a unique corresponding value
    int32_t keys[] = {100, 200, 300, 400, 500, 600, 700, 800};
    int32_t vals[] = {1, 2, 3, 4, 5, 6, 7, 8};
    
    auto key_col = gamax::make_column_from_host(keys, 8, cudaStreamDefault);
    auto val_col = gamax::make_column_from_host(vals, 8, cudaStreamDefault);
    
    std::vector<std::unique_ptr<gamax::column>> columns;
    columns.push_back(std::move(key_col));
    columns.push_back(std::move(val_col));
    auto table = gamax::make_table(std::move(columns));
    
    std::vector<int> sort_keys = {0};
    std::vector<gamax::sort_order> sort_orders = {gamax::sort_order::DESCENDING};
    
    auto result = gamax::sort::topk_table(
        table->view(), sort_keys, sort_orders, 4, true, stream_);
    
    auto result_keys = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto result_vals = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    // Verify relationship: val = key / 100
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(result_vals[i], result_keys[i] / 100) 
            << "Key-value mismatch at index " << i;
    }
}

} // namespace test
} // namespace gamax
