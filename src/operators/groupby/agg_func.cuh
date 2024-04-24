#pragma once

#include <cuda.h>
#include <limits>

#include "atomics/atomics.cuh"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"

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

    template<typename T, typename U>
    __device__ __forceinline__ OutputType operator() (
                                        const T& a,
                                        const U& b) const {
        return static_cast<OutputType>(a) + static_cast<OutputType>(b);
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

    template<typename T, typename U>
    __device__ __forceinline__ OutputType operator() (
                                        const T& a,
                                        const U& b) const {
        return cub::max(static_cast<OutputType>(a), static_cast<OutputType>(b));
    }
};
} // namespace gamax::groupby