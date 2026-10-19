// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "operators/join/join.hpp"
#include "operators/join/inner_join.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "memory/mem_manager.hpp"
#include "data_gen/generator.cuh"
#include "utils/transform.hpp"
#include "utils/utils.cuh"
#include "data_gen/table_gen.hpp"
#include "tests/test_helper.hpp"

namespace {
    template <typename T>
    std::vector<T> copy_to_host(gamax::column_view column)
    {
        std::vector<T> result(static_cast<std::size_t>(column.size()));
        if (!result.empty()) {
            EXPECT_EQ(
                cudaMemcpy(
                    result.data(), column.data<T>(),
                    result.size() * sizeof(T), cudaMemcpyDeviceToHost),
                cudaSuccess);
        }
        return result;
    }

    template<typename T>
    void pkpk_join_test(
        const gamax::size_type npk,
        const gamax::size_type rcols,
        const gamax::size_type scols,
        gamax::join::join_algo algo,
        gamax::join::materialization_algo mat_algo) 
    {
        auto type = gamax::data_type(gamax::type_to_id<T>());
        
        auto r = gamax::datagen::device::pk_uniform_id(
            type, npk, rcols, 42, cudaStreamDefault);
        auto s = gamax::datagen::device::pk_uniform_id(
            type, npk, scols, 98, cudaStreamDefault);

        gamax::join::join_hint hint;
        hint.unique_left = true;
        hint.unique_right = true;
        hint.join_algo = algo;
        hint.mat_algo = mat_algo;
        
        auto res = gamax::join::inner_eq_join(r->view(), s->view(), hint, cudaStreamDefault);
        auto res_view = res->view();

        ASSERT_EQ(res_view.num_columns(), rcols+scols-1);
        ASSERT_EQ(res_view.num_rows(), npk);
        ASSERT_TRUE(gamax::test::check_cols_eq_after_sort(res_view[0], r->view()[0]));
        for(gamax::size_type i = 1; i < res_view.num_columns(); i++) {
            ASSERT_TRUE(gamax::test::check_cols_eq(res_view[0], res_view[i]));
        }
    }

    template<typename T>
    void pkfk_join_test(
        const gamax::size_type npk,
        const gamax::size_type nfk,
        const gamax::size_type rcols,
        const gamax::size_type scols,
        const double match_ratio,
        gamax::join::join_algo algo,
        gamax::join::materialization_algo mat_algo) 
    {
        auto type = gamax::data_type(gamax::type_to_id<T>());
        
        auto r = gamax::datagen::device::pk_uniform_id(
            type, npk, rcols, 42, cudaStreamDefault);
        auto s = gamax::datagen::device::fk_uniform_id(
            type, npk, nfk, scols, match_ratio, 98, cudaStreamDefault);

        gamax::join::join_hint hint;
        hint.unique_left = true;
        hint.join_algo = algo;
        hint.mat_algo = mat_algo;
        
        auto res = gamax::join::inner_eq_join(r->view(), s->view(), hint, cudaStreamDefault);
        auto res_view = res->view();
        std::cout << "res_view.num_rows() = " << res_view.num_rows() << std::endl;

        ASSERT_EQ(res_view.num_columns(), rcols+scols-1);
        if(match_ratio == 1.0) {
            ASSERT_EQ(res_view.num_rows(), nfk);
            ASSERT_TRUE(gamax::test::check_cols_eq_after_sort(res_view[0], s->view()[0]));
        }
        else {
            ASSERT_EQ(res_view.num_rows(), nfk / (static_cast<gamax::size_type>(1.0/match_ratio)));
        }
        for(gamax::size_type i = 1; i < res_view.num_columns(); i++) {
            ASSERT_TRUE(gamax::test::check_cols_eq(res_view[0], res_view[i]));
        }
    }
}

TEST(IEQ_HASH_JOIN_INDICES, PKFK_MATCH_PAIRS) {
    constexpr gamax::size_type npk = 1024;
    constexpr gamax::size_type nfk = 4096;
    const auto type = gamax::data_type(gamax::type_to_id<int32_t>());
    auto left = gamax::datagen::device::pk_uniform_id(
        type, npk, 2, 42, cudaStreamDefault);
    auto right = gamax::datagen::device::fk_uniform_id(
        type, npk, nfk, 2, 1.0, 98, cudaStreamDefault);

    gamax::join::join_hint hint;
    hint.unique_left = true;
    hint.join_algo = gamax::join::HASH;
    hint.mat_algo = gamax::join::GFTR;
    auto matches = gamax::join::eq_join_indices(
        left->get_column(0).view(), right->get_column(0).view(),
        hint, cudaStreamDefault);

    ASSERT_EQ(matches.left_indices->size(), nfk);
    ASSERT_EQ(matches.right_indices->size(), nfk);
    const auto left_indices =
        copy_to_host<gamax::size_type>(matches.left_indices->view());
    const auto right_indices =
        copy_to_host<gamax::size_type>(matches.right_indices->view());
    const auto left_keys = copy_to_host<int32_t>(left->get_column(0).view());
    const auto right_keys = copy_to_host<int32_t>(right->get_column(0).view());
    for (std::size_t row = 0; row < left_indices.size(); ++row) {
        ASSERT_GE(left_indices[row], 0);
        ASSERT_LT(left_indices[row], npk);
        ASSERT_GE(right_indices[row], 0);
        ASSERT_LT(right_indices[row], nfk);
        ASSERT_EQ(
            left_keys[static_cast<std::size_t>(left_indices[row])],
            right_keys[static_cast<std::size_t>(right_indices[row])]);
    }
}

TEST(IEQ_HASH_JOIN_INDICES, EMPTY_INPUT) {
    auto left = gamax::make_column<int32_t>(0, cudaStreamDefault);
    auto right = gamax::make_column<int32_t>(8, cudaStreamDefault);
    gamax::join::join_hint hint;
    hint.unique_left = true;
    hint.join_algo = gamax::join::HASH;
    hint.mat_algo = gamax::join::GFTR;
    auto matches = gamax::join::eq_join_indices(
        left->view(), right->view(), hint, cudaStreamDefault);
    ASSERT_EQ(matches.left_indices->size(), 0);
    ASSERT_EQ(matches.right_indices->size(), 0);
}

#define IEQ_JOIN_TEST_CASE_SUITE(SUITE_NAME, JOIN_ALGO, MAT_ALGO) \
    TEST(SUITE_NAME, IEQ_PKPK_1) { \
        pkpk_join_test<int32_t>(1024, 2, 2, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, IEQ_PKFK_1) { \
        pkfk_join_test<int32_t>(1024, 2048, 2, 2, 1.0, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, IEQ_PKFK_2) { \
        pkfk_join_test<int32_t>((1<<27), (1<<28), 2, 2, 1.0, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, IEQ_PKFK_3) { \
        pkfk_join_test<int32_t>((1<<27), (1<<28), 3, 3, 1.0, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, IEQ_PKFK_4) { \
        pkfk_join_test<int32_t>((1<<27), (1<<28), 2, 2, 0.5, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, IEQ_PKFK_5) { \
        pkfk_join_test<int32_t>((1<<27), (1<<28), 2, 2, 0.125, JOIN_ALGO, MAT_ALGO); \
    } \
    TEST(SUITE_NAME, IEQ_PKFK_6) { \
        pkfk_join_test<int64_t>((1<<26), (1<<27), 2, 2, 1.0, JOIN_ALGO, MAT_ALGO); \
    }

IEQ_JOIN_TEST_CASE_SUITE(IEQ_PARTITION_HASH_JOIN_GFTR, gamax::join::join_algo::PARTITION, gamax::join::materialization_algo::GFTR)
IEQ_JOIN_TEST_CASE_SUITE(IEQ_PARTITION_HASH_JOIN_GFUR, gamax::join::join_algo::PARTITION, gamax::join::materialization_algo::GFUR)
IEQ_JOIN_TEST_CASE_SUITE(IEQ_HASH_JOIN, gamax::join::join_algo::HASH, gamax::join::materialization_algo::GFTR) // materialization does not matter 
IEQ_JOIN_TEST_CASE_SUITE(IEQ_SORT_MERGE_JOIN_GFTR, gamax::join::join_algo::SORT, gamax::join::materialization_algo::GFTR)
IEQ_JOIN_TEST_CASE_SUITE(IEQ_SORT_MERGE_JOIN_GFUR, gamax::join::join_algo::SORT, gamax::join::materialization_algo::GFUR)

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
