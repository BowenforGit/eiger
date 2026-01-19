#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

#include "operators/groupby/groupby.hpp"
#include "operators/groupby/aggregation.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "tests/test_helper.hpp"

namespace gamax {
namespace test {

class GroupbyMultiKeysTest : public ::testing::Test {
protected:
    void SetUp() override { 
        cudaStreamCreate(&stream_); 
    }
    
    void TearDown() override { 
        cudaStreamDestroy(stream_); 
    }

    cudaStream_t stream_{};
};

// =============================================================================
// Test Case 1: Two integer key columns with SUM aggregation
// =============================================================================
TEST_F(GroupbyMultiKeysTest, TwoIntKeyColumns_Sum) {
    // Key1: [1, 1, 2, 2, 1]
    // Key2: [10, 20, 10, 20, 10]
    // Val:  [100, 200, 300, 400, 500]
    // Expected groups: (1,10)->600, (1,20)->200, (2,10)->300, (2,20)->400
    
    int32_t key1[] = {1, 1, 2, 2, 1};
    int32_t key2[] = {10, 20, 10, 20, 10};
    int32_t vals[] = {100, 200, 300, 400, 500};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 3);  // 2 keys + 1 agg result
    EXPECT_EQ(result->num_rows(), 4);     // 4 unique (key1, key2) combinations
    
    // Copy results to host and verify
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    // Build expected map and verify results
    std::map<std::pair<int32_t, int32_t>, int64_t> expected = {
        {{1, 10}, 600}, {{1, 20}, 200}, {{2, 10}, 300}, {{2, 20}, 400}
    };
    
    std::map<std::pair<int32_t, int32_t>, int64_t> actual;
    for (size_t i = 0; i < 4; ++i) {
        actual[{h_key1[i], h_key2[i]}] = h_vals[i];
    }
    
    EXPECT_EQ(actual, expected);
}

// =============================================================================
// Test Case 2: Three key columns with SUM aggregation
// =============================================================================
TEST_F(GroupbyMultiKeysTest, ThreeIntKeyColumns_Sum) {
    // Key1: [1, 1, 1, 1, 2, 2]
    // Key2: [1, 1, 2, 2, 1, 1]
    // Key3: [1, 2, 1, 2, 1, 2]
    // Val:  [10, 20, 30, 40, 50, 60]
    // Each combination is unique
    
    int32_t key1[] = {1, 1, 1, 1, 2, 2};
    int32_t key2[] = {1, 1, 2, 2, 1, 1};
    int32_t key3[] = {1, 2, 1, 2, 1, 2};
    int32_t vals[] = {10, 20, 30, 40, 50, 60};
    
    auto key1_col = gamax::make_column_from_host(key1, 6);
    auto key2_col = gamax::make_column_from_host(key2, 6);
    auto key3_col = gamax::make_column_from_host(key3, 6);
    auto val_col = gamax::make_column_from_host(vals, 6);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    key_columns.push_back(std::move(key3_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 4);  // 3 keys + 1 agg result
    EXPECT_EQ(result->num_rows(), 6);     // 6 unique combinations
    
    // Verify sum of all aggregated values equals sum of input
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(3).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    int64_t total = 0;
    for (size_t i = 0; i < 6; ++i) {
        total += h_vals[i];
    }
    EXPECT_EQ(total, 10 + 20 + 30 + 40 + 50 + 60);
}

// =============================================================================
// Test Case 3: Two key columns with MAX aggregation
// =============================================================================
TEST_F(GroupbyMultiKeysTest, TwoIntKeyColumns_Max) {
    // Key1: [1, 1, 1, 2, 2, 2]
    // Key2: [1, 1, 2, 1, 2, 2]
    // Val:  [10, 50, 30, 40, 20, 60]
    // Expected: (1,1)->50, (1,2)->30, (2,1)->40, (2,2)->60
    
    int32_t key1[] = {1, 1, 1, 2, 2, 2};
    int32_t key2[] = {1, 1, 2, 1, 2, 2};
    int32_t vals[] = {10, 50, 30, 40, 20, 60};
    
    auto key1_col = gamax::make_column_from_host(key1, 6);
    auto key2_col = gamax::make_column_from_host(key2, 6);
    auto val_col = gamax::make_column_from_host(vals, 6);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::MAX});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 3);
    EXPECT_EQ(result->num_rows(), 4);
    
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_vals = gamax::copy_column_to_host<int32_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    std::map<std::pair<int32_t, int32_t>, int32_t> expected = {
        {{1, 1}, 50}, {{1, 2}, 30}, {{2, 1}, 40}, {{2, 2}, 60}
    };
    
    std::map<std::pair<int32_t, int32_t>, int32_t> actual;
    for (size_t i = 0; i < 4; ++i) {
        actual[{h_key1[i], h_key2[i]}] = h_vals[i];
    }
    
    EXPECT_EQ(actual, expected);
}

// =============================================================================
// Test Case 4: Mixed type keys (int32 and int64) with SUM
// =============================================================================
TEST_F(GroupbyMultiKeysTest, MixedTypeKeys_Int32AndInt64) {
    // Key1 (int32): [1, 1, 2, 2]
    // Key2 (int64): [100, 200, 100, 200]
    // Val (int32):  [10, 20, 30, 40]
    
    int32_t key1[] = {1, 1, 2, 2};
    int64_t key2[] = {100, 200, 100, 200};
    int32_t vals[] = {10, 20, 30, 40};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 3);
    EXPECT_EQ(result->num_rows(), 4);  // All combinations unique
    
    // Verify total sum preserved
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    int64_t total = 0;
    for (size_t i = 0; i < 4; ++i) {
        total += h_vals[i];
    }
    EXPECT_EQ(total, 10 + 20 + 30 + 40);
}

// =============================================================================
// Test Case 5: Two keys with multiple value columns and aggregations
// =============================================================================
TEST_F(GroupbyMultiKeysTest, TwoKeys_MultipleAggregations) {
    // Key1: [1, 1, 2, 2, 1]
    // Key2: [1, 2, 1, 2, 1]
    // Val1: [10, 20, 30, 40, 50]
    // Val2: [5, 15, 25, 35, 45]
    // Groups: (1,1)->SUM(Val1)=60, SUM(Val2)=50
    //         (1,2)->SUM(Val1)=20, SUM(Val2)=15
    //         (2,1)->SUM(Val1)=30, SUM(Val2)=25
    //         (2,2)->SUM(Val1)=40, SUM(Val2)=35
    
    int32_t key1[] = {1, 1, 2, 2, 1};
    int32_t key2[] = {1, 2, 1, 2, 1};
    int32_t val1[] = {10, 20, 30, 40, 50};
    int32_t val2[] = {5, 15, 25, 35, 45};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val1_col = gamax::make_column_from_host(val1, 5);
    auto val2_col = gamax::make_column_from_host(val2, 5);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val1_col));
    val_columns.push_back(std::move(val2_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    requests.push_back({1, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 4);  // 2 keys + 2 agg results
    EXPECT_EQ(result->num_rows(), 4);
    
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_agg1 = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    auto h_agg2 = gamax::copy_column_to_host<int64_t>(result->get_column(3).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    std::map<std::pair<int32_t, int32_t>, std::pair<int64_t, int64_t>> expected = {
        {{1, 1}, {60, 50}}, {{1, 2}, {20, 15}}, {{2, 1}, {30, 25}}, {{2, 2}, {40, 35}}
    };
    
    std::map<std::pair<int32_t, int32_t>, std::pair<int64_t, int64_t>> actual;
    for (size_t i = 0; i < 4; ++i) {
        actual[{h_key1[i], h_key2[i]}] = {h_agg1[i], h_agg2[i]};
    }
    
    EXPECT_EQ(actual, expected);
}

// =============================================================================
// Test Case 6: All rows have the same composite key (single group)
// =============================================================================
TEST_F(GroupbyMultiKeysTest, SingleGroup_AllSameKeys) {
    // All keys are (1, 100)
    int32_t key1[] = {1, 1, 1, 1, 1};
    int32_t key2[] = {100, 100, 100, 100, 100};
    int32_t vals[] = {10, 20, 30, 40, 50};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 3);
    EXPECT_EQ(result->num_rows(), 1);  // Single group
    
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    EXPECT_EQ(h_key1[0], 1);
    EXPECT_EQ(h_key2[0], 100);
    EXPECT_EQ(h_vals[0], 150);  // 10+20+30+40+50
}

// =============================================================================
// Test Case 7: Each row is its own group (all unique composite keys)
// =============================================================================
TEST_F(GroupbyMultiKeysTest, AllUniqueKeys_EachRowIsGroup) {
    int32_t key1[] = {1, 2, 3, 4, 5};
    int32_t key2[] = {10, 20, 30, 40, 50};
    int32_t vals[] = {100, 200, 300, 400, 500};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 3);
    EXPECT_EQ(result->num_rows(), 5);  // Each row is a group
    
    // Verify sum of all output values equals sum of input
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    int64_t total = 0;
    for (size_t i = 0; i < 5; ++i) {
        total += h_vals[i];
    }
    EXPECT_EQ(total, 100 + 200 + 300 + 400 + 500);
}

// =============================================================================
// Test Case 8: Keys with negative values
// =============================================================================
TEST_F(GroupbyMultiKeysTest, NegativeKeyValues) {
    int32_t key1[] = {-1, -1, 1, 1};
    int32_t key2[] = {-10, 10, -10, 10};
    int32_t vals[] = {100, 200, 300, 400};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 4);  // 4 unique combinations
    
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    std::map<std::pair<int32_t, int32_t>, int64_t> expected = {
        {{-1, -10}, 100}, {{-1, 10}, 200}, {{1, -10}, 300}, {{1, 10}, 400}
    };
    
    std::map<std::pair<int32_t, int32_t>, int64_t> actual;
    for (size_t i = 0; i < 4; ++i) {
        actual[{h_key1[i], h_key2[i]}] = h_vals[i];
    }
    
    EXPECT_EQ(actual, expected);
}

// =============================================================================
// Test Case 9: Keys where first column matches but second differs
// =============================================================================
TEST_F(GroupbyMultiKeysTest, FirstKeyMatchesSecondDiffers) {
    // All key1 values are 1, key2 provides the grouping
    int32_t key1[] = {1, 1, 1, 1, 1, 1};
    int32_t key2[] = {1, 2, 3, 1, 2, 3};
    int32_t vals[] = {10, 20, 30, 40, 50, 60};
    
    auto key1_col = gamax::make_column_from_host(key1, 6);
    auto key2_col = gamax::make_column_from_host(key2, 6);
    auto val_col = gamax::make_column_from_host(vals, 6);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 3);  // 3 groups: (1,1), (1,2), (1,3)
    
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    std::map<std::pair<int32_t, int32_t>, int64_t> expected = {
        {{1, 1}, 50}, {{1, 2}, 70}, {{1, 3}, 90}
    };
    
    std::map<std::pair<int32_t, int32_t>, int64_t> actual;
    for (size_t i = 0; i < 3; ++i) {
        actual[{h_key1[i], h_key2[i]}] = h_vals[i];
    }
    
    EXPECT_EQ(actual, expected);
}

// =============================================================================
// Test Case 10: Large number of groups (stress test)
// =============================================================================
TEST_F(GroupbyMultiKeysTest, LargeNumberOfGroups) {
    // Create 100 groups from 10x10 key combinations
    const int N = 100;
    std::vector<int32_t> key1_data(N), key2_data(N), val_data(N);
    
    for (int i = 0; i < N; ++i) {
        key1_data[i] = i / 10;  // 0-9, each repeated 10 times
        key2_data[i] = i % 10;  // 0-9, cycling
        val_data[i] = 1;        // All ones for easy sum verification
    }
    
    auto key1_col = gamax::make_column_from_host(key1_data.data(), N);
    auto key2_col = gamax::make_column_from_host(key2_data.data(), N);
    auto val_col = gamax::make_column_from_host(val_data.data(), N);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 200;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 100);  // 10x10 = 100 unique combinations
    
    // Each group should have sum = 1 (each combination appears once)
    auto h_vals = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(h_vals[i], 1);
    }
}

// =============================================================================
// Test Case 11: Two keys with SUM and MAX on same value column
// =============================================================================
TEST_F(GroupbyMultiKeysTest, TwoKeys_SumAndMaxOnSameColumn) {
    // Key1: [1, 1, 1, 2, 2]
    // Key2: [1, 1, 2, 1, 1]
    // Val:  [10, 30, 20, 50, 40]
    // (1,1) -> SUM=40, MAX=30
    // (1,2) -> SUM=20, MAX=20
    // (2,1) -> SUM=90, MAX=50
    
    int32_t key1[] = {1, 1, 1, 2, 2};
    int32_t key2[] = {1, 1, 2, 1, 1};
    int32_t vals[] = {10, 30, 20, 50, 40};
    
    auto key1_col = gamax::make_column_from_host(key1, 5);
    auto key2_col = gamax::make_column_from_host(key2, 5);
    auto val_col = gamax::make_column_from_host(vals, 5);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    requests.push_back({0, gamax::groupby::aggregation_id::MAX});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_columns(), 4);  // 2 keys + 2 agg results
    EXPECT_EQ(result->num_rows(), 3);
    
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_sum = gamax::copy_column_to_host<int64_t>(result->get_column(2).view(), stream_);
    auto h_max = gamax::copy_column_to_host<int32_t>(result->get_column(3).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    std::map<std::pair<int32_t, int32_t>, std::pair<int64_t, int32_t>> expected = {
        {{1, 1}, {40, 30}}, {{1, 2}, {20, 20}}, {{2, 1}, {90, 50}}
    };
    
    std::map<std::pair<int32_t, int32_t>, std::pair<int64_t, int32_t>> actual;
    for (size_t i = 0; i < 3; ++i) {
        actual[{h_key1[i], h_key2[i]}] = {h_sum[i], h_max[i]};
    }
    
    EXPECT_EQ(actual, expected);
}

// =============================================================================
// Test Case 12: Float value column with integer keys
// =============================================================================
TEST_F(GroupbyMultiKeysTest, FloatValues_IntegerKeys) {
    int32_t key1[] = {1, 1, 2, 2};
    int32_t key2[] = {1, 2, 1, 2};
    float vals[] = {1.5f, 2.5f, 3.5f, 4.5f};
    
    auto key1_col = gamax::make_column_from_host(key1, 4);
    auto key2_col = gamax::make_column_from_host(key2, 4);
    auto val_col = gamax::make_column_from_host(vals, 4);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 4);
    
    // Total sum should be preserved
    auto h_vals = gamax::copy_column_to_host<float>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    float total = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        total += h_vals[i];
    }
    EXPECT_NEAR(total, 1.5 + 2.5 + 3.5 + 4.5, 0.001);
}

// =============================================================================
// Test Case 13: Double value column with MAX aggregation
// =============================================================================
TEST_F(GroupbyMultiKeysTest, DoubleValues_Max) {
    int32_t key1[] = {1, 1, 1, 2, 2, 2};
    int32_t key2[] = {1, 1, 2, 1, 2, 2};
    double vals[] = {1.1, 3.3, 2.2, 5.5, 4.4, 6.6};
    
    auto key1_col = gamax::make_column_from_host(key1, 6);
    auto key2_col = gamax::make_column_from_host(key2, 6);
    auto val_col = gamax::make_column_from_host(vals, 6);
    
    std::vector<std::unique_ptr<gamax::column>> key_columns;
    key_columns.push_back(std::move(key1_col));
    key_columns.push_back(std::move(key2_col));
    auto key_table = gamax::make_table(std::move(key_columns));
    
    std::vector<std::unique_ptr<gamax::column>> val_columns;
    val_columns.push_back(std::move(val_col));
    auto val_table = gamax::make_table(std::move(val_columns));
    
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::MAX});
    
    gamax::groupby::groupby_hint hint;
    hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
    hint.groups_upper_bound = 10;
    
    auto result = gamax::groupby::groupby(
        key_table->view(), val_table->view(), requests, hint, stream_);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_rows(), 4);
    
    auto h_key1 = gamax::copy_column_to_host<int32_t>(result->get_column(0).view(), stream_);
    auto h_key2 = gamax::copy_column_to_host<int32_t>(result->get_column(1).view(), stream_);
    auto h_vals = gamax::copy_column_to_host<double>(result->get_column(2).view(), stream_);
    cudaStreamSynchronize(stream_);
    
    std::map<std::pair<int32_t, int32_t>, double> expected = {
        {{1, 1}, 3.3}, {{1, 2}, 2.2}, {{2, 1}, 5.5}, {{2, 2}, 6.6}
    };
    
    std::map<std::pair<int32_t, int32_t>, double> actual;
    for (size_t i = 0; i < 4; ++i) {
        actual[{h_key1[i], h_key2[i]}] = h_vals[i];
    }
    
    for (const auto& [key, val] : expected) {
        EXPECT_NEAR(actual[key], val, 0.001);
    }
}

} // namespace test
} // namespace gamax

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
