#pragma once

#include <iostream>
#include <omp.h>
#include <cassert>

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
    GROUPBY_NOT_SUPPORTED
};

struct groupby_params {
    int num_items {4096};
    int num_groups {1024};
    double zipf_factor {0.0};
    enum groupby_algo algo {GROUPBY_SORT};
    int actual_num_groups {-1};

    void print() const {
        std::cout << "num_items: " << num_items << std::endl;
        std::cout << "num_groups: " << num_groups << std::endl;
        std::cout << "zipf_factor: " << zipf_factor << std::endl;
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
    std::cout << "Data generated" << std::endl;
}
}
}