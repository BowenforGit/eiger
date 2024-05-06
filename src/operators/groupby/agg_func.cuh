#pragma once

#include <cuda.h>
#include <limits>

#include "atomics/atomics.cuh"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/column.hpp"

namespace gamax::groupby {
template<typename ValT>
struct Sum {
    using InputType = ValT;
    using OutputType = ValT; // FIXME
    
    static constexpr bool is_monoid = true;
    static constexpr bool is_atomic_updatable = true;
    static constexpr OutputType identity = 0;

    template <typename IdxT, typename ResultT>
    __device__ __forceinline__ void global_update(IdxT idx, 
                                        ResultT* result, 
                                        ValT val) const {
        gamax::atomic::atomic_add(&result[idx], val);
    }

    template <typename ResultT>
    __device__ __forceinline__ void atomic_update(ResultT* result, 
                                                    ValT val) const {
        gamax::atomic::atomic_add(result, val);
    }

    template <typename ResultT>
    __device__ __forceinline__ ResultT thread_local_update(
                                        const ResultT& local_agg,
                                        const ValT& val) const {
        return local_agg + val;
    }

    template<typename T, typename U>
    __device__ __forceinline__ OutputType operator() (
                                        const T& a,
                                        const U& b) const {
        return static_cast<OutputType>(a) + static_cast<OutputType>(b);
    }

    __device__ __forceinline__ void operator() (
                                        column& src, gamax::size_type src_idx, 
                                        column& dst, gamax::size_type dst_idx) const {
        this->global_update(dst_idx, dst.data<OutputType>(), src.data<ValT>()[src_idx]);
    }
};

template<typename ValT>
struct Max {
    using InputType = ValT;
    using OutputType = ValT;
    
    static constexpr bool is_monoid = true;
    static constexpr bool is_atomic_updatable = true;
    static constexpr OutputType identity = std::numeric_limits<OutputType>::min();

    template <typename IdxT, typename ResultT>
    __device__ __forceinline__ void global_update(IdxT idx, 
                                        ResultT* result, 
                                        ValT val) const {
        gamax::atomic::atomic_max(&result[idx], val);
    }

    template <typename ResultT>
    __device__ __forceinline__ void atomic_update(ResultT* result, 
                                                    ValT val) const {
        gamax::atomic::atomic_max(result, val);
    }

    template <typename ResultT>
    __device__ __forceinline__ ResultT thread_local_update(
                                        const ResultT& local_agg,
                                        const ValT& val) const {
        return cub::max(local_agg, val);
    }

    template<typename T, typename U>
    __device__ __forceinline__ OutputType operator() (
                                        const T& a,
                                        const U& b) const {
        return cub::max(static_cast<OutputType>(a), static_cast<OutputType>(b));
    }

    __device__ __forceinline__ void operator() (
                                        column& src, gamax::size_type src_idx, 
                                        column& dst, gamax::size_type dst_idx) const {
        this->global_update(dst_idx, dst.data<OutputType>(), src.data<ValT>()[src_idx]);
    }
};
} // namespace gamax::groupby