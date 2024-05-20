#pragma once

#include <iostream>
#include <omp.h>
#include <gtest/gtest.h>
#include <vector>

#include "benchmarks/groupby/groupby_util.cuh"
#include "operators/groupby/groupby.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/groupby_partition.hpp"
#include "operators/groupby/groupby_hash.hpp"
#include "types/type_dispatcher.hpp"

using namespace gamax::groupby;

namespace gamax::groupby {
template <typename KeyT, typename ValueT, typename AggT>
void invoke_groupby_sort(KeyT* h_keys, 
                         ValueT* h_vals, 
                         const int num_items,
                         gamax::groupby::groupby_algo algo,
                         KeyT*& h_out_keys,
                         typename AggT::OutputType*& h_out_vals,
                         int& num_groups,
                         gamax::groupby::groupby_hint& hint) {
    auto d_keys = gamax::make_column_from_host(h_keys, num_items);
    auto d_vals = gamax::make_column_from_host(h_vals, num_items);
    auto d_vals_2 = gamax::make_column_from_host(h_vals, num_items);

    std::vector<decltype(d_keys)> keys_vec;
    keys_vec.push_back(std::move(d_keys));
    std::vector<decltype(d_vals)> vals_vec;
    vals_vec.push_back(std::move(d_vals));
    vals_vec.push_back(std::move(d_vals_2));

    gamax::table key_table(std::move(keys_vec));
    gamax::table val_table(std::move(vals_vec));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    requests.push_back({1, gamax::groupby::aggregation_id::SUM});

    std::unique_ptr<gamax::table> out_table_ptr = nullptr;

    if(algo == gamax::groupby::groupby_algo::GROUPBY_SORT) {
        out_table_ptr = gamax::groupby::sort::groupby(key_table, val_table, requests, hint);
    } else if(algo == gamax::groupby::groupby_algo::GROUPBY_PARTITION) {
        out_table_ptr = gamax::groupby::partition::groupby(key_table, val_table, requests, hint);
    } else if(algo == gamax::groupby::groupby_algo::GROUPBY_SORT_GFUR) {
        out_table_ptr = gamax::groupby::sort::groupby_gfur(key_table, val_table, requests, hint);
    } else if(algo == gamax::groupby::groupby_algo::GROUPBY_PARTITION_GFUR) {
        out_table_ptr = gamax::groupby::partition::groupby_gfur(key_table, val_table, requests, hint);
    } else if(algo == gamax::groupby::groupby_algo::GROUPBY_HASH) {
        out_table_ptr = gamax::groupby::hash::groupby(key_table, val_table, requests, hint);
    } else {
        throw std::runtime_error("Unsupported groupby algorithm");
    }

    auto& d_out_keys = out_table_ptr->get_column(0);
    auto& d_out_vals = out_table_ptr->get_column(2);

    h_out_keys = new KeyT[d_out_keys.size()];
    h_out_vals = new typename AggT::OutputType[d_out_vals.size()];

    gamax::copy_column_to_host(d_out_keys, h_out_keys);
    gamax::copy_column_to_host(d_out_vals, h_out_vals);
    num_groups = d_out_keys.size();
}

template <typename KeyT>
void check_groupby_result(KeyT* h_keys, KeyT* ret_keys, int* ret_vals, int num_groups, const groupby_params& params) {
    ASSERT_EQ(num_groups, params.actual_num_groups);
    unsigned long long sum = 0;

    if(params.zipf_factor == 0.0) {
        for(int i = 0; i < num_groups; i++) {
            sum += ret_keys[i];
            ASSERT_EQ(ret_vals[i], params.num_items / params.num_groups);
        }
        EXPECT_EQ(sum, ((unsigned long long)params.num_groups - 1) * (unsigned long long)params.num_groups / 2);
    } else {
        std::vector<int> group_sizes(params.num_groups, 0);
        for(int i = 0; i < params.num_items; i++) {
            group_sizes[h_keys[i]]++;
        }
        for(int i = 0; i < num_groups; i++) {
            ASSERT_EQ(ret_vals[i], group_sizes[ret_keys[i]]);
        }
    }
}

template <typename KeyT, typename AggT>
void test_groupby_sort(groupby_params& params) {
    KeyT*  h_keys = (KeyT*)malloc(params.num_items * sizeof(KeyT));
    int* h_vals = (int*)malloc(params.num_items * sizeof(int));

    generate_groupby_data(h_keys, h_vals, params);

    // check the actual number of groups generated if zipf factor is not zero
    if(params.zipf_factor > 0.0) {
        std::vector<int> group_sizes(params.num_groups, 0);
        for(int i = 0; i < params.num_items; i++) {
            group_sizes[h_keys[i]]++;
        }
        
        int num_groups_generated = 0;
        for(int i = 0; i < params.num_groups; i++) {
            if(group_sizes[i] > 0) {
                num_groups_generated++;
            }
        }

        if(params.num_groups > num_groups_generated) {
            std::cout << "Warning: only " << num_groups_generated << " groups generated out of " << params.num_groups << std::endl;
        }

        params.actual_num_groups = num_groups_generated;
    }
    else {
        params.actual_num_groups = params.num_groups;
    }


    gamax::groupby::groupby_hint hint;
    hint.groups_upper_bound = params.num_groups;

    KeyT* h_out_keys = nullptr;
    typename AggT::OutputType* h_out_vals = nullptr;
    int num_groups = 0;
    
    invoke_groupby_sort<KeyT, int, AggT>(h_keys, h_vals, params.num_items, params.algo, h_out_keys, h_out_vals, num_groups, hint);
    check_groupby_result(h_keys, h_out_keys, h_out_vals, num_groups, params);
    
    free(h_keys);
    free(h_vals);
    free(h_out_keys);
    free(h_out_vals);
}

#define GROUPBY_TEST_CASE(K, AGG, N, G, ALGO, Z) \
    { \
        groupby_params params; \
        params.num_items = (N); \
        params.num_groups = (G); \
        params.zipf_factor = (Z); \
        params.algo = (ALGO); \
        params.num_aggs = 1; \
        test_groupby_sort<K, AGG<int>>(params); \
    }

#define GROUPBY_TEST_CASE_SUITE(SUITE_NAME, ALGO) \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_4_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 4, ALGO, 0) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_32_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 32, ALGO, 0) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_1024_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 1024, ALGO, 0) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_1048576_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<20), ALGO, 0) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_16777216_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<24), ALGO, 0) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_33554432_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<25), ALGO, 0) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_268435356_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<28), ALGO, 0) \
    } \
    TEST(SUITE_NAME, LONG_GROUPKEY_SUM_1024_GROUPS) { \
        GROUPBY_TEST_CASE(int64_t, Sum, (1<<28), 1024, ALGO, 0) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_1024_GROUPS_SKEW_1) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 1024, ALGO, (0.5)) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_1024_GROUPS_SKEW_2) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 1024, ALGO, (1.0)) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_1024_GROUPS_SKEW_3) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 1024, ALGO, (2.0)) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_4194304_GROUPS_SKEW_4) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<22), ALGO, (1.0)) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_4194304_GROUPS_SKEW_5) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<22), ALGO, (2.0)) \
    }
}