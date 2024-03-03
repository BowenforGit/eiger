#pragma once

#include <iostream>

#include "data_gen/generator.cuh"

namespace gamax {
namespace groupby {
    struct groupby_params {
        int num_items {4096};
        int num_groups {1024};
        double zipf_factor {0.0};

        void print() const {
            std::cout << "num_items: " << num_items << std::endl;
            std::cout << "num_groups: " << num_groups << std::endl;
            std::cout << "zipf_factor: " << zipf_factor << std::endl;
        }
    };

    struct groupby_hint {
        int NDV {-1};
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
            std::cout << "Zipfian distribution not implemented yet" << std::endl;
            std::exit(1);
        }
        std::cout << "Data generated" << std::endl;
    }

    // some common aggregation functions
    template<typename ValT>
    struct Sum {
        using OutputType = ValT;
        
        static constexpr bool is_monoid = true;
        static constexpr ValT identity = 0;

        template <typename IdxT, typename ResultT, typename IntermediateT>
        __device__ __forceinline__ void global_update(IdxT idx, 
                                            ResultT* result, 
                                            IntermediateT* inter, 
                                            ValT val) {
            atomicAdd(&result[idx], val);
        }

        template <typename ResultT>
        __device__ __forceinline__ ResultT thread_local_update(
                                            ResultT local_agg,
                                            ValT val) {
            return local_agg + val;
        }

        template <typename ResultT>
        __device__ __forceinline__ ResultT operator() (
                                            const ResultT& local_agg,
                                            const ValT& val) const {
            return local_agg + val;
        }
    };

    // template<typename ValT>
    // struct Max {
    //     using OutputType = ValT;
    //     template <typename IdxT, typename ResultT, typename IntermediateT>
    //     __device__ __forceinline__ void update(IdxT idx, 
    //                                         ResultT* result, 
    //                                         IntermediateT* inter, 
    //                                         ValT val) {
    //         atomicMax(&result[idx], val);
    //     }
    // };
    template<typename ValT>
    struct Max {
        using OutputType = ValT;
        
        static constexpr bool is_monoid = true;
        static constexpr ValT identity = std::numeric_limits<OutputType>::min();

        template <typename IdxT, typename ResultT, typename IntermediateT>
        __device__ __forceinline__ void global_update(IdxT idx, 
                                            ResultT* result, 
                                            IntermediateT* inter, 
                                            ValT val) {
            atomicMax(&result[idx], val);
        }

        template <typename ResultT>
        __device__ __forceinline__ ResultT thread_local_update(
                                            ResultT local_agg,
                                            ValT val) {
            return cub::max(local_agg, val);
        }

        template <typename ResultT>
        __device__ __forceinline__ ResultT operator() (
                                            const ResultT& local_agg,
                                            const ValT& val) const {
            return cub::max(local_agg, val);
        }
    };
}
}