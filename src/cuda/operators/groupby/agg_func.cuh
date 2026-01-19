#pragma once

#include <cuda.h>
#include <limits>

#include "atomics/atomics.cuh"
#include "types/data_type.hpp"
#include "types/type_dispatcher.hpp"
#include "memory/column.hpp"

// The output data type of the aggregation function follows the rules from
// https://www.postgresql.org/docs/current/functions-aggregate.html#FUNCTIONS-AGGREGATE-TABLE
// except that 
// (1) SUM(bigint) still returns bigint instead of numeric for simplicity
// (2) Single precision still returns single precision for simplicity
namespace gamax::groupby {

namespace {
    template<typename T>
    struct sum_output_type {
        using type = T;
    };
#define SPECIALIZE_SUM_OUTPUT_TYPE(src_type, dst_type) \
    template<> \
    struct sum_output_type<src_type> { \
        using type = dst_type; \
    };

    SPECIALIZE_SUM_OUTPUT_TYPE(int8_t, int64_t)
    SPECIALIZE_SUM_OUTPUT_TYPE(int16_t, int64_t)
    SPECIALIZE_SUM_OUTPUT_TYPE(int32_t, int64_t)
    SPECIALIZE_SUM_OUTPUT_TYPE(uint8_t, uint64_t)
    SPECIALIZE_SUM_OUTPUT_TYPE(uint16_t, uint64_t)
    SPECIALIZE_SUM_OUTPUT_TYPE(uint32_t, uint64_t)
#undef SPECIALIZE_SUM_OUTPUT_TYPE
}
template<typename ValT>
struct Sum {
    using InputType = ValT;
    using OutputType = typename sum_output_type<ValT>::type;
    
    static constexpr bool is_monoid = true;
    static constexpr bool is_atomic_updatable = true;
    static constexpr OutputType identity = 0;

    template <typename IdxT, typename ResultT>
    __device__ __forceinline__ void global_update(IdxT idx, 
                                        ResultT* result, 
                                        ValT val) const {
        gamax::atomic::atomic_add(&result[idx], static_cast<ResultT>(val));
    }

    template <typename ResultT>
    __device__ __forceinline__ void atomic_update(ResultT* result, 
                                                    ValT val) const {
        gamax::atomic::atomic_add(result, static_cast<ResultT>(val));
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
                                        gamax::column_view src, gamax::size_type src_idx, 
                                        gamax::mutable_column_view dst, gamax::size_type dst_idx) const {
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
        return local_agg > val ? local_agg : val;
    }

    template<typename T, typename U>
    __device__ __forceinline__ OutputType operator() (
                                        const T& a,
                                        const U& b) const {
        return static_cast<OutputType>(a) > static_cast<OutputType>(b) ? static_cast<OutputType>(a) : static_cast<OutputType>(b);
    }

    __device__ __forceinline__ void operator() (
                                        gamax::column_view src, gamax::size_type src_idx, 
                                        gamax::mutable_column_view dst, gamax::size_type dst_idx) const {
        this->global_update(dst_idx, dst.data<OutputType>(), src.data<ValT>()[src_idx]);
    }
};

template<gamax::groupby::aggregation_id agg_kind, typename ValT>
struct agg_fn_mapper {
    using F = void;
};

#define AGG_FN_MAPPER(agg_kind, agg_name) \
    template<typename ValT> \
    struct agg_fn_mapper<gamax::groupby::aggregation_id::agg_kind, ValT> { \
        using F = gamax::groupby::agg_name<ValT>; \
    };

AGG_FN_MAPPER(SUM, Sum)
AGG_FN_MAPPER(MAX, Max)
} // namespace gamax::groupby