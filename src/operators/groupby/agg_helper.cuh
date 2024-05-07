#pragma once

#include <cuda.h>

#include "aggregation.hpp"
#include "memory/column.hpp"
#include "memory/table_device_view.hpp"
#include "types/type_dispatcher.hpp"
#include "types/data_type.hpp"

namespace gamax::groupby::details {
template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_val_type_helper {
    template<typename T>
    __device__ __forceinline__ void operator()(
        gamax::column& src,
        gamax::size_type src_idx,
        gamax::column& dst,
        gamax::size_type dst_idx)
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
        F{}(src, src_idx, dst, dst_idx);
    }

    // initialize the aggregation value
    template<typename T>
    __device__ __forceinline__ void operator()(
        gamax::column& dst,
        gamax::size_type dst_idx)
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
        dst.data<F::OutputType>()[dst_idx] = F::identity;
    }

    __device__ __forceinline__ void operator()(
        gamax::column& src,
        gamax::size_type src_idx,
        gamax::column& dst,
        gamax::size_type dst_idx)
    {

        type_dispatcher(
            src.type(),
            dispatch_val_type_helper<agg_kind>{},
            src, 
            src_idx, 
            dst, 
            dst_idx);
    }

    __device__ __forceinline__ void operator()(
        gamax::column& dst,
        gamax::size_type dst_idx)
    {

        type_dispatcher(
            dst.type(),
            dispatch_val_type_helper<agg_kind>{},
            dst, 
            dst_idx);
    }
};

template<typename... Ts>
__device__ __forceinline__ void dispatch_type_aggregate(gamax::groupby::aggregation_id agg_kind, Ts&&... args)
{
    switch(agg_kind) {
        case gamax::groupby::aggregation_id::SUM:
            dispatch_val_type_helper<gamax::groupby::aggregation_id::SUM>{}(std::forward<Ts>(args)...);
            break;
        case gamax::groupby::aggregation_id::MAX:
            dispatch_val_type_helper<gamax::groupby::aggregation_id::MAX>{}(std::forward<Ts>(args)...);
            break;
        default:
            GAMAX_UNREACHABLE("Unsupported aggregation function");
    }
}

__device__ __forceinline__ void aggregate(
    gamax::groupby::aggregation_id agg_kind,
    gamax::column& src,
    gamax::size_type src_idx,
    gamax::column& dst,
    gamax::size_type dst_idx)
{
    dispatch_type_aggregate(
        agg_kind,
        src, 
        src_idx, 
        dst, 
        dst_idx);
}

__device__ __forceinline__ void init_agg(
    gamax::groupby::aggregation_id agg_kind,
    gamax::column& dst,
    gamax::size_type dst_idx)
{
    dispatch_type_aggregate(
        agg_kind,
        dst, 
        dst_idx);
}

__inline__ __global__ void initialize_aggregation(
    table_device_view res,
    gamax::groupby::aggregation_request* aggs)
{
    auto num_groups = res.num_rows();
    auto num_aggs = res.num_columns() - 1;

    for(gamax::size_type i = get_cuda_tid(); i < num_groups; i += nthreads()) {
        for(gamax::size_type j = 0; j < num_aggs; j++) {
            init_agg(aggs[j].agg_kind, res.get_column(j+1), i);
        }
    }
}
} // namespace gamax::groupby::details