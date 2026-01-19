#pragma once

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include <cuda.h>
#include <memory>

#include "benchmarks/groupby/groupby_util.cuh"
#include "operators/groupby/groupby.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "types/type_dispatcher.hpp"

using namespace gamax::groupby;

namespace gamax::groupby {

/**
 * Invoke groupby operation using GPU-generated data.
 * 
 * @param key_table: Table containing the key columns (generated on GPU).
 * @param val_table: Table containing the value columns (generated on GPU).
 * @param h_out_keys: Output host array for result keys.
 * @param h_out_vals: Output host array for result values.
 * @param num_groups: Output number of groups found.
 * @param hint: Groupby hint for algorithm selection.
 * @param stream: CUDA stream to use.
 */
template <typename KeyT, typename AggT>
void invoke_groupby_gpu(gamax::table& key_table,
                        gamax::table& val_table,
                        KeyT*& h_out_keys,
                        typename AggT::OutputType*& h_out_vals,
                        gamax::size_type& num_groups,
                        gamax::groupby::groupby_hint& hint,
                        cudaStream_t stream) {
    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});
    requests.push_back({1, gamax::groupby::aggregation_id::SUM});

    auto out_table_ptr = gamax::groupby::groupby(key_table.view(), val_table.view(), requests, hint, stream);

    auto& d_out_keys = out_table_ptr->get_column(0);
    auto& d_out_vals = out_table_ptr->get_column(2);

    h_out_keys = new KeyT[d_out_keys.size()];
    h_out_vals = new typename AggT::OutputType[d_out_vals.size()];

    gamax::copy_column_to_host(d_out_keys, h_out_keys, stream);
    gamax::copy_column_to_host(d_out_vals, h_out_vals, stream);
    cudaStreamSynchronize(stream);
    num_groups = d_out_keys.size();
}

/**
 * Check groupby result for uniform distribution case (no skew).
 * For uniform distribution with all-ones values:
 * - Each group should have exactly (num_items / num_groups) items
 * - Sum of all keys should equal sum of 0..(num_groups-1)
 */
template <typename KeyT, typename RetType>
void check_groupby_result_uniform(KeyT* ret_keys, RetType* ret_vals, gamax::size_type num_groups, const groupby_params& params) {
    ASSERT_EQ(num_groups, params.actual_num_groups);
    unsigned long long sum = 0;

    for(gamax::size_type i = 0; i < num_groups; i++) {
        sum += ret_keys[i];
        ASSERT_EQ(ret_vals[i], params.num_items / params.num_groups);
    }
    EXPECT_EQ(sum, ((unsigned long long)params.num_groups - 1) * (unsigned long long)params.num_groups / 2);
}

/**
 * Check groupby result for zipf distribution case (skewed).
 * For zipf distribution with all-ones values:
 * - Each group's sum should equal the count of that key in input
 */
template <typename KeyT, typename RetType>
void check_groupby_result_zipf(KeyT* h_keys, KeyT* ret_keys, RetType* ret_vals, gamax::size_type num_groups, const groupby_params& params) {
    ASSERT_EQ(num_groups, params.actual_num_groups);
    
    std::vector<gamax::size_type> group_sizes(params.num_groups, 0);
    for(gamax::size_type i = 0; i < params.num_items; i++) {
        group_sizes[h_keys[i]]++;
    }
    for(gamax::size_type i = 0; i < num_groups; i++) {
        ASSERT_EQ(ret_vals[i], group_sizes[ret_keys[i]]);
    }
}

/**
 * Test groupby with GPU-based data generation.
 * 
 * Data generation semantics:
 * - Uniform (zipf_factor == 0): Keys are [0, num_groups) repeated and shuffled,
 *   values are all 1s. Each group has exactly (num_items / num_groups) items.
 * - Zipf (zipf_factor > 0): Keys follow zipf distribution over [0, num_groups),
 *   values are all 1s. Some groups may have no items due to skew.
 */
template <typename KeyT, typename AggT>
void test_groupby_sort(groupby_params& params, cudaStream_t stream = cudaStreamDefault) {
    // Set value option to ALL_ONES to match original test semantics
    params.val_opt = val_option::ALL_ONES;
    
    // Generate data on GPU using the GPU-based generator
    gamax::table key_table;
    gamax::table val_table;
    generate_groupby_data<KeyT, int>(key_table, val_table, params);
    
    // For zipf distribution, we need to copy keys to host to count actual groups
    std::unique_ptr<KeyT[]> h_keys;
    if(params.zipf_factor > 0.0) {
        h_keys = gamax::copy_column_to_host<KeyT>(key_table.get_column(0).view(), stream);
        cudaStreamSynchronize(stream);
        
        std::vector<gamax::size_type> group_sizes(params.num_groups, 0);
        for(gamax::size_type i = 0; i < params.num_items; i++) {
            group_sizes[h_keys[i]]++;
        }
        
        gamax::size_type num_groups_generated = 0;
        for(gamax::size_type i = 0; i < params.num_groups; i++) {
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
    hint.groupby_algo = params.algo;

    KeyT* h_out_keys = nullptr;
    typename AggT::OutputType* h_out_vals = nullptr;
    gamax::size_type num_groups = 0;
    
    invoke_groupby_gpu<KeyT, AggT>(key_table, val_table, h_out_keys, h_out_vals, num_groups, hint, stream);
    
    if(params.zipf_factor == 0.0) {
        check_groupby_result_uniform(h_out_keys, h_out_vals, num_groups, params);
    } else {
        check_groupby_result_zipf(h_keys.get(), h_out_keys, h_out_vals, num_groups, params);
    }
    
    delete[] h_out_keys;
    delete[] h_out_vals;
}

#define GROUPBY_TEST_CASE(K, AGG, N, G, ALGO, Z) \
    { \
        groupby_params params; \
        params.num_items = (N); \
        params.num_groups = (G); \
        params.zipf_factor = (Z); \
        params.algo = (ALGO); \
        params.num_aggs = 2; \
        cudaStream_t stream; \
        cudaStreamCreate(&stream); \
        test_groupby_sort<K, AGG<int>>(params, stream); \
        cudaStreamDestroy(stream); \
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