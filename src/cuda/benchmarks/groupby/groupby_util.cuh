#pragma once

#include <iostream>
#include <omp.h>
#include <cassert>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <thrust/fill.h>

#include "operators/groupby/groupby.hpp"
#include "memory/column.hpp"
#include "memory/table.hpp"
#include "operators/groupby/aggregation.hpp"
#include "operators/groupby/agg_func.cuh"
#include "operators/groupby/groupby_sort.hpp"
#include "operators/groupby/groupby_partition.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/mem_manager.hpp"

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

enum val_option {
    SEQ,
    ALL_ONES,
    FROM_KEYS
};

struct groupby_params {
    int num_items {4096};
    int num_groups {1024};
    double zipf_factor {0.0};
    enum groupby_algo algo {GROUPBY_SORT};
    enum val_option val_opt {SEQ};
    int actual_num_groups {-1};
    int num_aggs {1};

    template<typename Stream>
    void print(Stream& out) const {
        out << "num_items: " << num_items << ", ";
        out << "num_groups: " << num_groups << ", ";
        out << "zipf_factor: " << zipf_factor << ", ";
        out << "actual_num_groups: " << actual_num_groups << ", ";
        out << "num_aggs: " << num_aggs << std::endl;
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
void generate_zipf_keys_host(KeyT* h_keys, const groupby_params& params) {
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

template<typename Key>
void generate_groupby_keys(gamax::table& key_table, const groupby_params& params) {
    std::vector<std::unique_ptr<gamax::column>> key_vec;
    if(params.zipf_factor == 0.0) {
        auto keys = gamax::make_sequence<Key>(params.num_items, 0, 1);
        auto raw_key_ptr = keys->template data<Key>();
        thrust::for_each_n(thrust::device, raw_key_ptr, params.num_items, [=] __device__ (Key& key) { key = key % params.num_groups; });
        thrust::default_random_engine g;
        thrust::shuffle(thrust::cuda::par(*get_thrust_allocator()), raw_key_ptr, raw_key_ptr + params.num_items, g);
        key_vec.push_back(std::move(keys));
    } else {
        auto  h_keys = new Key[params.num_items];
        generate_zipf_keys_host(h_keys, params);
        auto d_keys = gamax::make_column_from_host(h_keys, params.num_items);
        key_vec.push_back(std::move(d_keys));
        delete [] h_keys;
    }
    gamax::table kt{std::move(key_vec)};
    key_table = std::move(kt);

    cudaDeviceSynchronize();
}

template<typename Value>
void generate_groupby_vals_seq(gamax::table& val_table, const groupby_params& params) {
    std::vector<std::unique_ptr<gamax::column>> vals_vec;
    for(int i = 0; i < params.num_aggs; i++) {
        auto vals = gamax::make_sequence<Value>(params.num_items, 0, 1);
        vals_vec.push_back(std::move(vals));
    }
    gamax::table vt{std::move(vals_vec)};
    val_table = std::move(vt);

    cudaDeviceSynchronize();
}

template<typename Value>
void generate_groupby_vals_all_ones(gamax::table& val_table, const groupby_params& params) {
    std::vector<std::unique_ptr<gamax::column>> vals_vec;
    for(int i = 0; i < params.num_aggs; i++) {
        auto vals = gamax::make_column<Value>(params.num_items);
        thrust::fill_n(thrust::device, reinterpret_cast<Value*>(vals->data()), params.num_items, static_cast<Value>(1));
        vals_vec.push_back(std::move(vals));
    }
    gamax::table vt{std::move(vals_vec)};
    val_table = std::move(vt);

    cudaDeviceSynchronize();
}

template<typename Key, typename Value, std::enable_if_t<std::is_same_v<Key, Value>, void>* = nullptr>
void generate_groupby_vals_from_keys(gamax::column& key, gamax::table& val_table, const groupby_params& params) {
    std::vector<std::unique_ptr<gamax::column>> vals_vec;
    for(int i = 0; i < params.num_aggs; i++) {
        auto vals = key.deep_copy();
        vals_vec.push_back(std::move(vals));
    }
    gamax::table vt{std::move(vals_vec)};
    val_table = std::move(vt);

    cudaDeviceSynchronize();
}

template <typename Key, typename Value>
void generate_groupby_data(gamax::table& key_table, gamax::table& val_table, const groupby_params& params) {
    generate_groupby_keys<Key>(key_table, params);

    switch(params.val_opt) {
        case val_option::SEQ:
            generate_groupby_vals_seq<Value>(val_table, params);
            break;
        case val_option::ALL_ONES:
            generate_groupby_vals_all_ones<Value>(val_table, params);
            break;
        case val_option::FROM_KEYS:
            if constexpr(std::is_same_v<Key, Value>) {
                generate_groupby_vals_from_keys<Key, Value>(key_table.get_column(0), val_table, params);
            } else {
                throw std::runtime_error("FROM_KEYS is only supported for Key == Value");
            }
            break;
        default:
            throw std::runtime_error("Unknown val_option");
    }

    cudaDeviceSynchronize();
    std::cout << "Data generated" << std::endl;
}
}
}