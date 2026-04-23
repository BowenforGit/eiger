#include <iostream>
#include <gtest/gtest.h>
#include <vector>

#include "data_gen/table_gen.hpp"
#include "operators/groupby/aggregation.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"

namespace {
    template<typename T>
    void aggregate_test(
        const gamax::size_type nrows,
        gamax::groupby::aggregation_id agg_op)
    {
        auto type = gamax::data_type(gamax::type_to_id<T>());
        
        auto t = gamax::datagen::device::pk_uniform_id(type, nrows, 2, 42);

        std::vector<gamax::groupby::aggregation_request> agg_reqs;
        agg_reqs.push_back({0, agg_op});
        agg_reqs.push_back({1, agg_op});
        
        auto agg_res = gamax::groupby::aggregate::aggregate(t->view(), agg_reqs);
        auto agg_view = agg_res->view();
        
        ASSERT_EQ(agg_view.num_columns(), 2);
        ASSERT_EQ(agg_view.num_rows(), 1);

        auto res0 = gamax::copy_column_to_host<T>(agg_view[0]);
        auto res1 = gamax::copy_column_to_host<T>(agg_view[1]);
        
        if(agg_op == gamax::groupby::aggregation_id::SUM) {
            auto sum = nrows*(nrows-1)/2;
            ASSERT_EQ(res0[0], sum);
            ASSERT_EQ(res1[0], sum);
        } else if(agg_op == gamax::groupby::aggregation_id::MAX) {
            ASSERT_EQ(res0[0], nrows-1);
            ASSERT_EQ(res1[0], nrows-1);
        }
    }
}

TEST(AggregateTest, Sum32) {
    aggregate_test<int32_t>((1<<14), gamax::groupby::aggregation_id::SUM);
}

TEST(AggregateTest, Max32) {
    aggregate_test<int32_t>((1<<25), gamax::groupby::aggregation_id::MAX);
}

TEST(AggregateTest, Sum64) {
    aggregate_test<int64_t>((1<<14), gamax::groupby::aggregation_id::SUM);
}

TEST(AggregateTest, Max64) {
    aggregate_test<int64_t>((1<<25), gamax::groupby::aggregation_id::MAX);
}

namespace {
template <typename T>
void scalar_avg_test(gamax::size_type nrows) {
    auto type = gamax::data_type(gamax::type_to_id<T>());
    auto t = gamax::datagen::device::pk_uniform_id(type, nrows, 2, 42);

    std::vector<gamax::groupby::aggregation_request> agg_reqs;
    agg_reqs.push_back({0, gamax::groupby::aggregation_id::AVG});
    agg_reqs.push_back({1, gamax::groupby::aggregation_id::AVG});

    auto agg_res = gamax::groupby::aggregate::aggregate(t->view(), agg_reqs);
    auto agg_view = agg_res->view();

    ASSERT_EQ(agg_view.num_columns(), 2);
    ASSERT_EQ(agg_view.num_rows(), 1);
    ASSERT_EQ(agg_view[0].type().id(), gamax::type_id::FLOAT64);
    ASSERT_EQ(agg_view[1].type().id(), gamax::type_id::FLOAT64);

    auto res0 = gamax::copy_column_to_host<double>(agg_view[0]);
    auto res1 = gamax::copy_column_to_host<double>(agg_view[1]);
    // pk_uniform_id column 0 = 0..nrows-1; column 1 = 0..nrows-1; sums
    // are nrows*(nrows-1)/2, so avg = (nrows-1)/2.
    double expected = static_cast<double>(nrows - 1) / 2.0;
    ASSERT_NEAR(res0[0], expected, 1e-6);
    ASSERT_NEAR(res1[0], expected, 1e-6);
}
} // namespace

TEST(AggregateTest, Avg32) {
    scalar_avg_test<int32_t>(1 << 14);
}

TEST(AggregateTest, Avg64) {
    scalar_avg_test<int64_t>(1 << 14);
}

TEST(AggregateTest, AvgFloat64) {
    gamax::size_type nrows = 1 << 12;
    auto type = gamax::data_type(gamax::type_id::FLOAT64);
    auto t = gamax::datagen::device::pk_uniform_id(type, nrows, 1, 7);

    std::vector<gamax::groupby::aggregation_request> agg_reqs;
    agg_reqs.push_back({0, gamax::groupby::aggregation_id::AVG});

    auto agg_res = gamax::groupby::aggregate::aggregate(t->view(), agg_reqs);
    auto agg_view = agg_res->view();
    ASSERT_EQ(agg_view[0].type().id(), gamax::type_id::FLOAT64);

    auto res0 = gamax::copy_column_to_host<double>(agg_view[0]);
    double expected = static_cast<double>(nrows - 1) / 2.0;
    ASSERT_NEAR(res0[0], expected, 1e-4);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}