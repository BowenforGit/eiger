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
#include "operators/groupby/groupby_sort.cuh"
#include "operators/groupby/groupby_partition.cuh"
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

    std::vector<decltype(d_keys)> keys_vec;
    keys_vec.push_back(std::move(d_keys));
    std::vector<decltype(d_vals)> vals_vec;
    vals_vec.push_back(std::move(d_vals));

    gamax::table key_table(std::move(keys_vec));
    gamax::table val_table(std::move(vals_vec));

    std::vector<gamax::groupby::aggregation_request> requests;
    requests.push_back({0, gamax::groupby::aggregation_id::SUM});

    std::unique_ptr<gamax::table> out_table_ptr = nullptr;

    if(algo == gamax::groupby::groupby_algo::GROUPBY_SORT) {
        out_table_ptr = gamax::groupby::sort::groupby(key_table, val_table, requests, hint);
    } else if(algo == gamax::groupby::groupby_algo::GROUPBY_PARTITION) {
        out_table_ptr = gamax::groupby::partition::groupby(key_table, val_table, requests, hint);
    } else {
        throw std::runtime_error("Unsupported groupby algorithm");
    }

    auto& d_out_keys = out_table_ptr->get_column(0);
    auto& d_out_vals = out_table_ptr->get_column(1);

    h_out_keys = new KeyT[d_out_keys.size()];
    h_out_vals = new typename AggT::OutputType[d_out_vals.size()];

    gamax::copy_column_to_host(d_out_keys, h_out_keys);
    gamax::copy_column_to_host(d_out_vals, h_out_vals);
    num_groups = d_out_keys.size();
}

template <typename KeyT, typename AggT>
void test_groupby_sort(const groupby_params& params, KeyT*& h_out_keys, int*& h_out_vals, int& num_groups) {
    KeyT*  h_keys = (KeyT*)malloc(params.num_items * sizeof(KeyT));
    int* h_vals = (int*)malloc(params.num_items * sizeof(int));

    generate_groupby_data(h_keys, h_vals, params);

    gamax::groupby::groupby_hint hint;
    hint.groups_upper_bound = params.num_groups;
    invoke_groupby_sort<KeyT, int, AggT>(h_keys, h_vals, params.num_items, params.algo, h_out_keys, h_out_vals, num_groups, hint);
    
    free(h_keys);
    free(h_vals);
}

#define GROUPBY_TEST_CASE(K, AGG, N, G, ALGO) \
    { \
        groupby_params params; \
        params.num_items = (N); \
        params.num_groups = (G); \
        params.zipf_factor = 0.0; \
        params.algo = (ALGO); \
        K* ret_keys = nullptr; \
        int* ret_vals = nullptr; \
        int num_groups; \
        test_groupby_sort<K, AGG<int>>(params, ret_keys, ret_vals, num_groups); \
        EXPECT_EQ(num_groups, params.num_groups); \
        unsigned long long sum = 0; \
        for(int i = 0; i < num_groups; i++) { \
            sum += ret_keys[i]; \
            EXPECT_EQ(ret_vals[i], params.num_items / params.num_groups); \
        } \
        EXPECT_EQ(sum, ((unsigned long long)params.num_groups - 1) * (unsigned long long)params.num_groups / 2); \
        free(ret_keys); \
        free(ret_vals); \
    }

#define GROUPBY_TEST_CASE_SUITE(SUITE_NAME, ALGO) \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_4_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 4, ALGO) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_1024_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), 1024, ALGO) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_1048576_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<20), ALGO) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_33554432_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<25), ALGO) \
    } \
    TEST(SUITE_NAME, INT_GROUPKEY_SUM_268435356_GROUPS) { \
        GROUPBY_TEST_CASE(int, Sum, (1<<28), (1<<28), ALGO) \
    } \
    TEST(SUITE_NAME, LONG_GROUPKEY_SUM_1024_GROUPS) { \
        GROUPBY_TEST_CASE(int64_t, Sum, (1<<28), 1024, ALGO) \
    }
}