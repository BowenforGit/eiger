// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>

#include "operators/join/join.hpp"
#include "operators/join/semi_join.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "data_gen/generator.cuh"
#include "utils/transform.hpp"
#include "utils/utils.cuh"
#include "data_gen/table_gen.hpp"
#include "tests/test_helper.hpp"

namespace {
    template<typename T>
    void semi_join_test(
        const gamax::size_type npk,
        const double match_ratio,
        gamax::join::join_algo algo,
        gamax::join::materialization_algo mat_algo) 
    {
        constexpr int cols = 2;
        const gamax::size_type nitems = 2 * npk;
        
        auto type = gamax::data_type(gamax::type_to_id<T>());

        auto r = gamax::datagen::device::fk_uniform_id(
            type, npk, nitems, cols, match_ratio, 98, cudaStreamDefault);
        auto s = gamax::datagen::device::fk_uniform_id(
            type, npk, nitems, cols, 1.0, 42, cudaStreamDefault);

        gamax::join::join_hint hint;
        hint.join_algo = algo;
        hint.mat_algo = mat_algo;
        
        auto res = gamax::join::semi_join(r->view(), s->view(), hint, cudaStreamDefault);
        auto res_view = res->view();

        ASSERT_EQ(res_view.num_columns(), cols);
        ASSERT_EQ(res_view.num_rows(), nitems * match_ratio);
        auto policy = gamax::rmm_sync_exec_policy_handle{cudaStreamDefault};
        ASSERT_TRUE(thrust::all_of(policy.get(), res_view[0].data<T>(), res_view[0].data<T>() + res_view[0].size(), [npk] __device__ (T const& v) -> bool { return v < npk && v >= 0; }));
        for(gamax::size_type i = 1; i < res_view.num_columns(); i++) {
            ASSERT_TRUE(gamax::test::check_cols_eq(res_view[0], res_view[i]));
        }
    }
}

#define SEMI_JOIN_TEST_CASE_SUITE(SUITE_NAME, JOIN_ALGO, MAT_ALGO) \
    TEST(SUITE_NAME, SJ_ALL_MATCH_1) { \
        semi_join_test<int32_t>(1024, 1.0, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, SJ_ALL_MATCH_2) { \
        semi_join_test<int64_t>(1024, 1.0, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, SJ_ALL_MATCH_3) { \
        semi_join_test<int32_t>((1<<27), 1.0, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, SJ_HALF_MATCH_1) { \
        semi_join_test<int32_t>(1024, 0.5, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, SJ_NO_MATCH_1) { \
        semi_join_test<int32_t>(32, 0.0, JOIN_ALGO, MAT_ALGO); \
    }

SEMI_JOIN_TEST_CASE_SUITE(SEMI_HASH_JOIN, gamax::join::join_algo::HASH, gamax::join::materialization_algo::GFTR)

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
