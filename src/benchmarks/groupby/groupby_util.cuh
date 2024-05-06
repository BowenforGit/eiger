#pragma once

#include <iostream>
#include <omp.h>
#include <cassert>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include "operators/groupby/groupby.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/groupby_partition.hpp"
#include "types/type_dispatcher.hpp"

#include "data_gen/generator.cuh"

namespace gamax {
namespace groupby {
enum groupby_algo {
    GROUPBY_PARTITION,
    GROUPBY_SORT,
    GROUPBY_PARTITION_GFUR,
    GROUPBY_SORT_GFUR,
    GROUPBY_HASH,
    GROUPBY_NOT_SUPPORTED
};

struct groupby_params {
    int num_items {4096};
    int num_groups {1024};
    double zipf_factor {0.0};
    enum groupby_algo algo {GROUPBY_SORT};
    int actual_num_groups {-1};
    int num_aggs {1};

    void print() const {
        std::cout << "num_items: " << num_items << std::endl;
        std::cout << "num_groups: " << num_groups << std::endl;
        std::cout << "zipf_factor: " << zipf_factor << std::endl;
        std::cout << "actual_num_groups: " << actual_num_groups << std::endl;
        std::cout << "num_aggs: " << num_aggs << std::endl;
    }
};

template <typename KeyT, typename ValueT>
void generate_groupby_data(KeyT* h_keys, ValueT* h_vals, const groupby_params& params) {
    if(params.zipf_factor == 0.0) {
        #pragma omp parallel for
        for (int i = 0; i < params.num_items; i++) {
            h_keys[i] = i % params.num_groups;
            h_vals[i] = 1;
        }
        knuth_shuffle(h_keys, params.num_items);
    } else {
        auto lut = gen_zipf_lut(params.zipf_factor, params.num_groups);
        zipf_distribution_t<KeyT> dist(lut.begin(), lut.end());
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::default_random_engine e(42+tid);
            
            #pragma omp for
            for(size_t i = 0; i < params.num_items; i++) {
                h_keys[i] = dist(e);
                h_vals[i] = 1;
            }
        }
    }
}

template <typename KeyT>
void generate_zipf_keys(KeyT* h_keys, const groupby_params& params) {
    auto lut = gen_zipf_lut(params.zipf_factor, params.num_groups);
    zipf_distribution_t<KeyT> dist(lut.begin(), lut.end());
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::default_random_engine e(42+tid);
        
        #pragma omp for
        for(size_t i = 0; i < params.num_items; i++) {
            h_keys[i] = dist(e);
        }
    }
}

template <typename Key, typename Value>
void generate_groupby_data(gamax::table& key_table, gamax::table& val_table, const groupby_params& params) {
    std::vector<std::unique_ptr<gamax::column>> key_vec;
    if(params.zipf_factor == 0.0) {
        auto keys = gamax::make_sequence<Key>(params.num_items, 0, 1);
        auto raw_key_ptr = keys->template data<Key>();
        thrust::for_each_n(thrust::device, raw_key_ptr, params.num_items, [=] __device__ (Key& key) { key = key % params.num_groups; });
        thrust::default_random_engine g;
        thrust::shuffle(thrust::device, raw_key_ptr, raw_key_ptr + params.num_items, g);
        key_vec.push_back(std::move(keys));
    } else {
        auto  h_keys = new Key[params.num_items];
        generate_zipf_keys(h_keys, params);
        auto d_keys = gamax::make_column_from_host(h_keys, params.num_items);
        key_vec.push_back(std::move(d_keys));
        delete [] h_keys;
    }
    gamax::table kt{std::move(key_vec)};
    key_table = std::move(kt);
    
    std::vector<std::unique_ptr<gamax::column>> vals_vec;
    for(int i = 0; i < params.num_aggs; i++) {
        auto vals = gamax::make_sequence<Value>(params.num_items, 0, 1);
        vals_vec.push_back(std::move(vals));
    }
    gamax::table vt{std::move(vals_vec)};
    val_table = std::move(vt);

    std::cout << val_table.num_rows() << " " << val_table.num_columns() << std::endl;

    cudaDeviceSynchronize();
    std::cout << "Data generated" << std::endl;
}
}
}