// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "operators/groupby/groupby.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "utils/utils.cuh"
#include "tests/test_helper.hpp"

#define TEST_GROUPBY() \
    auto out_table_ptr = gamax::groupby::groupby(key_table_ptr->view(), value_table_ptr->view(), req, hint, cudaStreamDefault); \
    ASSERT_EQ(out_table_ptr->num_columns(), 2); \
    for(int i = 0; i < 2; i++) { \
        EXPECT_TRUE(gamax::test::check_cols_eq_after_sort(out_table_ptr->get_column(0).view(), res_table_ptr->get_column(0).view())); \
    } 

template<typename T>
void test_floating_point_groupby() {
    int h_key[] = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    auto key = gamax::make_column_from_host(h_key, 10, cudaStreamDefault);
    auto key_table_ptr = std::make_unique<gamax::table>();
    key_table_ptr->add_column(std::move(key));

    T h_value[] = {1.1, 2.1, 2.2, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4};
    auto value = gamax::make_column_from_host(h_value, 10, cudaStreamDefault);
    auto value_table_ptr = std::make_unique<gamax::table>();
    value_table_ptr->add_column(std::move(value));

    int h_res_key[] = {1, 2, 3, 4};
    auto res_key = gamax::make_column_from_host(h_res_key, 4, cudaStreamDefault);
    T h_res_value[] = {1.1, 2.2, 3.3, 4.4};
    auto res_value = gamax::make_column_from_host(h_res_value, 4, cudaStreamDefault);
    auto res_table_ptr = std::make_unique<gamax::table>();
    res_table_ptr->add_column(std::move(res_key));
    res_table_ptr->add_column(std::move(res_value));
    
    std::vector<gamax::groupby::aggregation_request> req;
    req.push_back({0, gamax::groupby::aggregation_id::MAX});

    gamax::groupby::groupby_hint hint;
    hint.groups_upper_bound = 4;

    {
        hint.groupby_algo = gamax::groupby::groupby_algo::HASH;
        TEST_GROUPBY();
    }

    {
        hint.groupby_algo = gamax::groupby::groupby_algo::SORT;
        TEST_GROUPBY()
    }

    {
        hint.groupby_algo = gamax::groupby::groupby_algo::SORT_GFUR;
        TEST_GROUPBY()
    }

    {
        hint.groupby_algo = gamax::groupby::groupby_algo::SORT_OPT;
        TEST_GROUPBY()
    }
}

TEST(FP_TESTS, DOUBLE_GROUP_BY) {
    test_floating_point_groupby<double>();
}

TEST(FP_TESTS, FLOAT_GROUP_BY) {
    test_floating_point_groupby<float>();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
