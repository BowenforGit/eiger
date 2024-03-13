#include <iostream>
#include <omp.h>
#include <gtest/gtest.h>

#include "operators/groupby/groupby_sort.cuh"
#include "operators/groupby/groupby_partition.cuh"
#include "types/tuple.cuh"
#include "benchmarks/groupby/groupby_util.cuh"

using namespace gamax::groupby;

template <typename KeyT, typename ValueT, typename AggT>
void invoke_groupby_sort(KeyT* h_keys, 
                         ValueT* h_vals, 
                         const int num_items,
                         gamax::groupby::groupby_algo algo,
                         KeyT*& h_out_keys,
                         typename AggT::OutputType*& h_out_vals,
                         int& num_groups,
                         const struct groupby_hint& hint) {
    using keys_tuple_t = struct Chunk<KeyT>;
    using vals_tuple_t = struct Chunk<ValueT>;
    using agg_t = typename AggT::OutputType;
    using out_tuple_t = struct Chunk<KeyT, agg_t>;

    keys_tuple_t tkeys(num_items, 0, h_keys);
    vals_tuple_t tvals(num_items, 0, h_vals);
    release_mem(tkeys.select_vec);
    release_mem(tvals.select_vec);
    tkeys.select_vec = nullptr;
    tvals.select_vec = nullptr;
    
    const bool use_hint = hint.NDV != -1;
    out_tuple_t tout;
    tout.allocate(use_hint ? hint.NDV : num_items);
    cudaDeviceSynchronize();
    
    if(algo == gamax::groupby::groupby_algo::GROUPBY_SORT) {
        gamax::groupby::sort::grouped_agg<AggT>(tkeys, tvals, tout);
    } else {
        gamax::groupby::partition::grouped_agg<AggT>(tkeys, tvals, tout);
    }
    // gamax::groupby::partition::grouped_agg<AggT>(tkeys, tvals, tout);

    tout.peek(16);

    tkeys.free_mem();
    tvals.free_mem();

    num_groups = tout.num_items;
    h_out_keys = (KeyT*)malloc(num_groups * sizeof(KeyT));
    h_out_vals = (agg_t*)malloc(num_groups * sizeof(agg_t));
    cudaMemcpy(h_out_keys, COL(tout, 0), num_groups * sizeof(KeyT), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_vals, COL(tout, 1), num_groups * sizeof(agg_t), cudaMemcpyDeviceToHost);
    tout.free_mem();

    cudaDeviceSynchronize();
}

template <typename KeyT, typename AggT>
void test_groupby_sort(const groupby_params& params, KeyT*& h_out_keys, int*& h_out_vals, int& num_groups) {
    KeyT*  h_keys = (KeyT*)malloc(params.num_items * sizeof(KeyT));
    int* h_vals = (int*)malloc(params.num_items * sizeof(int));

    generate_groupby_data(h_keys, h_vals, params);

    struct groupby_hint hint;
    hint.NDV = params.num_groups;
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
        GROUPBY_TEST_CASE(long, Sum, (1<<28), 1024, ALGO) \
    }

GROUPBY_TEST_CASE_SUITE(GROUPBY_SORT, gamax::groupby::groupby_algo::GROUPBY_SORT)
GROUPBY_TEST_CASE_SUITE(GROUPBY_PARTITION, gamax::groupby::groupby_algo::GROUPBY_PARTITION)

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}