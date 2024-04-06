#pragma once

#include "atomics/atomics.cuh"

namespace gamax::groupby {
template<typename ValT>
struct Sum {
    using InputType = ValT;
    using OutputType = ValT; // FIXME
    
    static constexpr bool is_monoid = true;
    static constexpr bool is_atomic_updatable = true;
    static constexpr ValT identity = 0;

    template <typename IdxT, typename ResultT>
    __device__ __forceinline__ void global_update(IdxT idx, 
                                        ResultT* result, 
                                        ValT val) {
        gamax::atomic::atomic_add(&result[idx], val);
    }

    template <typename ResultT>
    __device__ __forceinline__ void atomic_update(ResultT* result, 
                                                    ValT val) {
        gamax::atomic::atomic_add(result, val);
    }

    template <typename ResultT>
    __device__ __forceinline__ ResultT thread_local_update(
                                        const ResultT& local_agg,
                                        const ValT& val) {
        return local_agg + val;
    }

    template <typename ResultT>
    __device__ __forceinline__ ResultT operator() (
                                        const ResultT& local_agg,
                                        const ValT& val) const {
        // return template thread_local_update<ResultT>(local_agg, val);
        return local_agg + val;
    }
};

template<typename ValT>
struct Max {
    using InputType = ValT;
    using OutputType = ValT;
    
    static constexpr bool is_monoid = true;
    static constexpr bool is_atomic_updatable = true;
    static constexpr ValT identity = std::numeric_limits<OutputType>::min();

    template <typename IdxT, typename ResultT>
    __device__ __forceinline__ void global_update(IdxT idx, 
                                        ResultT* result, 
                                        ValT val) {
        gamax::atomic::atomic_max(&result[idx], val);
    }

    template <typename ResultT>
    __device__ __forceinline__ void atomic_update(ResultT* result, 
                                                    ValT val) {
        gamax::atomic::atomic_max(result, val);
    }

    template <typename ResultT>
    __device__ __forceinline__ ResultT thread_local_update(
                                        const ResultT& local_agg,
                                        const ValT& val) {
        return cub::max(local_agg, val);
    }

    template <typename ResultT>
    __device__ __forceinline__ ResultT operator() (
                                        const ResultT& local_agg,
                                        const ValT& val) const {
        return cub::max(local_agg, val);
    }
};
} // namespace gamax::groupby