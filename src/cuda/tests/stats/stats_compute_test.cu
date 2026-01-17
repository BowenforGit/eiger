#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <set>

#include "stats/stats_compute.hpp"
#include "stats/stats_storage.hpp"
#include "memory/column.hpp"
#include "string/string.hpp"
#include "string/string_column.hpp"

namespace gamax {
namespace test {

class StatsComputeTest : public ::testing::Test {
protected:
    void SetUp() override { 
        cudaStreamCreate(&stream_);
        // Clear any cached stats from previous tests
        gamax::stats::StatsStorage::instance().clear_all();
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
};

// =============================================================================
// Test compute_min
// =============================================================================

TEST_F(StatsComputeTest, ComputeMin_Int32) {
    int32_t data[] = {5, 2, 8, 1, 9, 3};
    auto col = gamax::make_column_from_host(data, 6, stream_);
    
    auto result = gamax::stats::compute_min(col->view(), stream_);
    
    EXPECT_EQ(result.get<int32_t>(), 1);
}

TEST_F(StatsComputeTest, ComputeMin_Int64) {
    int64_t data[] = {100, -50, 200, -100, 50};
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto result = gamax::stats::compute_min(col->view(), stream_);
    
    EXPECT_EQ(result.get<int64_t>(), -100);
}

TEST_F(StatsComputeTest, ComputeMin_Float) {
    float data[] = {3.5f, 1.2f, 4.8f, 0.5f, 2.1f};
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto result = gamax::stats::compute_min(col->view(), stream_);
    
    EXPECT_FLOAT_EQ(result.get<float>(), 0.5f);
}

TEST_F(StatsComputeTest, ComputeMin_Double) {
    double data[] = {3.14159, 2.71828, 1.41421, 1.73205};
    auto col = gamax::make_column_from_host(data, 4, stream_);
    
    auto result = gamax::stats::compute_min(col->view(), stream_);
    
    EXPECT_DOUBLE_EQ(result.get<double>(), 1.41421);
}

TEST_F(StatsComputeTest, ComputeMin_SingleElement) {
    int32_t data[] = {42};
    auto col = gamax::make_column_from_host(data, 1, stream_);
    
    auto result = gamax::stats::compute_min(col->view(), stream_);
    
    EXPECT_EQ(result.get<int32_t>(), 42);
}

// =============================================================================
// Test compute_max
// =============================================================================

TEST_F(StatsComputeTest, ComputeMax_Int32) {
    int32_t data[] = {5, 2, 8, 1, 9, 3};
    auto col = gamax::make_column_from_host(data, 6, stream_);
    
    auto result = gamax::stats::compute_max(col->view(), stream_);
    
    EXPECT_EQ(result.get<int32_t>(), 9);
}

TEST_F(StatsComputeTest, ComputeMax_Int64) {
    int64_t data[] = {100, -50, 200, -100, 50};
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto result = gamax::stats::compute_max(col->view(), stream_);
    
    EXPECT_EQ(result.get<int64_t>(), 200);
}

TEST_F(StatsComputeTest, ComputeMax_Float) {
    float data[] = {3.5f, 1.2f, 4.8f, 0.5f, 2.1f};
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto result = gamax::stats::compute_max(col->view(), stream_);
    
    EXPECT_FLOAT_EQ(result.get<float>(), 4.8f);
}

TEST_F(StatsComputeTest, ComputeMax_Unsigned) {
    uint32_t data[] = {100, 200, 50, 300, 150};
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto result = gamax::stats::compute_max(col->view(), stream_);
    
    EXPECT_EQ(result.get<uint32_t>(), 300u);
}

// =============================================================================
// Test compute_avg
// =============================================================================

TEST_F(StatsComputeTest, ComputeAvg_Int32) {
    int32_t data[] = {10, 20, 30, 40, 50};  // sum = 150, avg = 30
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto result = gamax::stats::compute_avg(col->view(), stream_);
    
    EXPECT_DOUBLE_EQ(result.get<double>(), 30.0);
}

TEST_F(StatsComputeTest, ComputeAvg_Float) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};  // sum = 10, avg = 2.5
    auto col = gamax::make_column_from_host(data, 4, stream_);
    
    auto result = gamax::stats::compute_avg(col->view(), stream_);
    
    EXPECT_NEAR(result.get<double>(), 2.5, 1e-6);
}

TEST_F(StatsComputeTest, ComputeAvg_WithNegatives) {
    int32_t data[] = {-10, 10, -20, 20, 0};  // sum = 0, avg = 0
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto result = gamax::stats::compute_avg(col->view(), stream_);
    
    EXPECT_DOUBLE_EQ(result.get<double>(), 0.0);
}

TEST_F(StatsComputeTest, ComputeAvg_LargeValues) {
    // Test that 64-bit accumulator prevents overflow
    int32_t data[] = {2000000000, 2000000000, 2000000000};  // Would overflow int32
    auto col = gamax::make_column_from_host(data, 3, stream_);
    
    auto result = gamax::stats::compute_avg(col->view(), stream_);
    
    EXPECT_DOUBLE_EQ(result.get<double>(), 2000000000.0);
}

TEST_F(StatsComputeTest, ComputeAvg_EmptyColumn) {
    auto col = gamax::make_column<int32_t>(0, stream_);
    
    auto result = gamax::stats::compute_avg(col->view(), stream_);
    
    EXPECT_DOUBLE_EQ(result.get<double>(), 0.0);
}

// =============================================================================
// Test compute_ndv_approx for numeric types
// =============================================================================

TEST_F(StatsComputeTest, ComputeNdvApprox_Int32_AllDistinct) {
    int32_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto col = gamax::make_column_from_host(data, 10, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    // HLL is approximate, but for 10 distinct values it should be close
    EXPECT_GE(ndv, 8);   // Allow some error
    EXPECT_LE(ndv, 12);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_Int32_AllSame) {
    int32_t data[] = {42, 42, 42, 42, 42, 42, 42, 42, 42, 42};
    auto col = gamax::make_column_from_host(data, 10, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    // Should be exactly 1 or very close
    EXPECT_EQ(ndv, 1);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_Int32_SomeDuplicates) {
    // 5 distinct values: 1, 2, 3, 4, 5 - each appears twice
    int32_t data[] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    auto col = gamax::make_column_from_host(data, 10, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    EXPECT_GE(ndv, 4);
    EXPECT_LE(ndv, 6);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_LargerDataset) {
    // Create 1000 values with 100 distinct
    std::vector<int32_t> data(1000);
    for (int i = 0; i < 1000; ++i) {
        data[i] = i % 100;
    }
    auto col = gamax::make_column_from_host(data.data(), 1000, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    // HLL should estimate ~100, allow 20% error
    EXPECT_GE(ndv, 80);
    EXPECT_LE(ndv, 120);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_EmptyColumn) {
    auto col = gamax::make_column<int32_t>(0, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    
    EXPECT_EQ(result.get<gamax::size_type>(), 0);
}

// =============================================================================
// Test compute_ndv_approx for string types (4-byte prefix HLL)
// =============================================================================

TEST_F(StatsComputeTest, ComputeNdvApprox_String_AllDistinct) {
    std::vector<std::string> strings = {
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon"
    };
    auto col = makeStringColumn(strings, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    // All strings have different 4-byte prefixes, so should be ~10
    EXPECT_GE(ndv, 8);
    EXPECT_LE(ndv, 12);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_String_AllSame) {
    std::vector<std::string> strings = {
        "hello", "hello", "hello", "hello", "hello"
    };
    auto col = makeStringColumn(strings, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    EXPECT_EQ(ndv, 1);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_String_SamePrefixDifferentSuffix) {
    // These all have the same 4-byte prefix "test", so NDV will be 1
    std::vector<std::string> strings = {
        "test1", "test2", "test3", "test4", "test5"
    };
    auto col = makeStringColumn(strings, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    // All have same 4-byte prefix "test", so HLL sees them as identical
    EXPECT_EQ(ndv, 1);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_String_ShortStrings) {
    // Strings shorter than 4 bytes
    std::vector<std::string> strings = {"a", "b", "c", "ab", "bc", "abc"};
    auto col = makeStringColumn(strings, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    auto ndv = result.get<gamax::size_type>();
    
    // All have distinct prefixes (padded with zeros)
    EXPECT_GE(ndv, 5);
    EXPECT_LE(ndv, 7);
}

TEST_F(StatsComputeTest, ComputeNdvApprox_String_Empty) {
    std::vector<std::string> strings = {};
    auto col = makeStringColumn(strings, stream_);
    
    auto result = gamax::stats::compute_ndv_approx(col->view(), stream_);
    
    EXPECT_EQ(result.get<gamax::size_type>(), 0);
}

// =============================================================================
// Test stats caching via StatsStorage
// =============================================================================

TEST_F(StatsComputeTest, StatsStorage_CachesResults) {
    int32_t data[] = {5, 2, 8, 1, 9, 3};
    auto col = gamax::make_column_from_host(data, 6, stream_);
    
    // First call computes and caches
    auto result1 = gamax::stats::get_or_compute_scalar_stat(
        col->view(), gamax::stats::StatsType::MIN, stream_);
    
    // Second call should return cached value
    auto result2 = gamax::stats::get_or_compute_scalar_stat(
        col->view(), gamax::stats::StatsType::MIN, stream_);
    
    EXPECT_EQ(result1.get<int32_t>(), 1);
    EXPECT_EQ(result2.get<int32_t>(), 1);
}

TEST_F(StatsComputeTest, StatsStorage_MultipleStatsTypes) {
    int32_t data[] = {10, 20, 30, 40, 50};
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto min_result = gamax::stats::get_or_compute_scalar_stat(
        col->view(), gamax::stats::StatsType::MIN, stream_);
    auto max_result = gamax::stats::get_or_compute_scalar_stat(
        col->view(), gamax::stats::StatsType::MAX, stream_);
    auto avg_result = gamax::stats::get_or_compute_scalar_stat(
        col->view(), gamax::stats::StatsType::AVG, stream_);
    
    EXPECT_EQ(min_result.get<int32_t>(), 10);
    EXPECT_EQ(max_result.get<int32_t>(), 50);
    EXPECT_DOUBLE_EQ(avg_result.get<double>(), 30.0);
}

// =============================================================================
// Test compute_stats with StatsRequest
// =============================================================================

TEST_F(StatsComputeTest, ComputeStats_WithRequest) {
    int32_t data[] = {1, 2, 3, 4, 5};
    auto col = gamax::make_column_from_host(data, 5, stream_);
    
    auto request = gamax::stats::StatsRequestBuilder(col->view())
        .as_base()
        .with_min()
        .with_max()
        .with_avg()
        .build();
    
    gamax::stats::compute_stats(col->view(), request, stream_);
    
    // Verify all stats are now cached
    auto& storage = gamax::stats::StatsStorage::instance();
    
    auto min_opt = storage.get_scalar_stat(col->view(), gamax::stats::StatsType::MIN);
    auto max_opt = storage.get_scalar_stat(col->view(), gamax::stats::StatsType::MAX);
    auto avg_opt = storage.get_scalar_stat(col->view(), gamax::stats::StatsType::AVG);
    
    ASSERT_TRUE(min_opt.has_value());
    ASSERT_TRUE(max_opt.has_value());
    ASSERT_TRUE(avg_opt.has_value());
    
    EXPECT_EQ(min_opt->get<int32_t>(), 1);
    EXPECT_EQ(max_opt->get<int32_t>(), 5);
    EXPECT_DOUBLE_EQ(avg_opt->get<double>(), 3.0);
}

} // namespace test
} // namespace gamax
