#pragma once

#include <cuda.h>

#ifdef __NVCC__
#include <cooperative_groups/memcpy_async.h>
#endif

#include "aggregation.hpp"
#include "memory/column.hpp"
// #include "memory/table_device_view.hpp"
#include "types/type_dispatcher.hpp"
#include "types/data_type.hpp"

namespace gamax::groupby::details {
template<gamax::groupby::aggregation_id agg_kind>
struct dispatch_val_type_helper {
    template<typename T>
    __device__ __forceinline__ void operator()(
        gamax::column_view src,
        gamax::size_type src_idx,
        gamax::mutable_column_view dst,
        gamax::size_type dst_idx)
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
        F{}(src, src_idx, dst, dst_idx);
    }

    template<typename T>
    __device__ __forceinline__ void operator()(
        gamax::column_view src,
        gamax::size_type src_idx,
        void* dst,
        gamax::size_type dst_idx)
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;

        F f;
        f.atomic_update(reinterpret_cast<typename F::OutputType*>(dst) + dst_idx, src.data<T>()[src_idx]);
    }

    // reduce the values from src_idx[src_end] to src_idx[src_start] into a single value
    // this can be used by a thread block or a grid to perform the aggregation
    // depending on the group parameter you passed in
    template<typename T, typename Group, typename IdxIt>
    __device__ __forceinline__ void operator()(
        Group const& group,
        gamax::column_view src,
        IdxIt& src_idx,
        gamax::size_type src_start,
        gamax::size_type src_end,
        gamax::mutable_column_view dst,
        gamax::size_type dst_idx)
    {
        namespace cg = cooperative_groups;
        
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
        using output_t  = typename F::OutputType;

        auto block = cg::this_thread_block();
        
        __shared__ output_t agg_val;
        if(block.thread_rank() == 0) {
            agg_val = F::identity;
        }
        block.sync();

        F f;
        output_t tl_agg = F::identity;
        #pragma unroll
        // for(int i = threadIdx.x; i < src_end - src_start; i += blockDim.x) {
        for(gamax::size_type i = group.thread_rank(); i < src_end - src_start; i += group.size()) {
            tl_agg = f.thread_local_update(tl_agg, src.data<T>()[src_idx[i]]);
        }
        f.atomic_update(&agg_val, tl_agg);
        block.sync();

        if(block.thread_rank() == 0) {
            // dst.data<output_t>()[dst_idx] = agg_val;
            f.global_update(dst_idx, dst.data<output_t>(), agg_val);
        }
    }

    // initialize the aggregation value
    template<typename T>
    __device__ __forceinline__ void operator()(
        void* dst,
        gamax::size_type dst_idx)
    {
        using F = typename gamax::groupby::agg_fn_mapper<agg_kind, T>::F;
        reinterpret_cast<typename F::OutputType*>(dst)[dst_idx] = F::identity;
    }

    __device__ __forceinline__ void operator()(
        gamax::column_view src,
        gamax::size_type src_idx,
        gamax::mutable_column_view dst,
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

    template<typename Group, typename IdxIt>
    __device__ __forceinline__ void operator()(
        Group const& group,
        gamax::column_view src,
        IdxIt& src_idx,
        gamax::size_type src_start,
        gamax::size_type src_end,
        gamax::mutable_column_view dst,
        gamax::size_type dst_idx)
    {

        type_dispatcher(
            src.type(),
            dispatch_val_type_helper<agg_kind>{},
            group,
            src, 
            src_idx, 
            src_start,
            src_end,
            dst, 
            dst_idx);
    }

    __device__ __forceinline__ void operator()(
        gamax::data_type src_type,
        void* dst,
        gamax::size_type dst_idx)
    {

        type_dispatcher(
            src_type,
            dispatch_val_type_helper<agg_kind>{},
            dst, 
            dst_idx);
    }

    __device__ __forceinline__ void operator()(
        gamax::column_view src,
        gamax::size_type src_idx,
        void* dst,
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
    gamax::column_view src,
    gamax::size_type src_idx,
    gamax::mutable_column_view dst,
    gamax::size_type dst_idx)
{
    dispatch_type_aggregate(
        agg_kind,
        src, 
        src_idx, 
        dst, 
        dst_idx);
}

template<typename Group, typename IdxIt>
__device__ __forceinline__ void reduce(
    Group const& group,
    gamax::groupby::aggregation_id agg_kind,
    gamax::column_view src,
    IdxIt& src_idx,
    gamax::size_type src_start,
    gamax::size_type src_end,
    gamax::mutable_column_view dst,
    gamax::size_type dst_idx
)
{
    dispatch_type_aggregate(
        agg_kind,
        group,
        src, 
        src_idx, 
        src_start, 
        src_end, 
        dst, 
        dst_idx);
}

__device__ __forceinline__ void block_aggregate(
    gamax::groupby::aggregation_id agg_kind,
    gamax::column_view src,
    gamax::size_type src_idx,
    void* dst,
    gamax::size_type dst_idx)
{
    dispatch_type_aggregate(
        agg_kind,
        src, 
        src_idx, 
        dst, 
        dst_idx);
}

#ifdef __NVCC__
template<typename TyGroup>
struct copy_agg_from_smem_to_global_helper {
    template<typename T>
    __device__ __forceinline__ void operator()(
        const TyGroup& group,
        gamax::mutable_column_view agg,
        const gamax::size_type offset,
        void* block_agg,
        const gamax::size_type num_items)
    {
        cooperative_groups::memcpy_async(group, agg.data<T>()+offset, reinterpret_cast<T*>(block_agg), num_items * sizeof(T));
    }
};

template<typename TyGroup>
__device__ __forceinline__ void copy_agg_from_smem_to_global(
    const TyGroup& group,
    gamax::mutable_column_view agg,
    const gamax::size_type offset,
    void* block_agg, 
    const gamax::size_type num_items)
{
    type_dispatcher(
        agg.type(),
        copy_agg_from_smem_to_global_helper<TyGroup>{},
        group,
        agg,
        offset,
        block_agg,
        num_items);
}

template<typename TyGroup>
__device__ __forceinline__ void copy_agg_from_smem_to_global(
    const TyGroup& group,
    gamax::experimental::mutable_table_device_view res,
    const gamax::size_type offset,
    void* block_agg, 
    const gamax::size_type num_groups,
    const gamax::size_type num_aggs)
{
    auto cur_agg_buffer = block_agg;

    #pragma unroll
    for(gamax::size_type k = 0; k < num_aggs; k++) {
        copy_agg_from_smem_to_global(
            group,
            res.get_column(k+1), 
            offset, 
            cur_agg_buffer, 
            num_groups);
        cur_agg_buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(cur_agg_buffer) + (num_groups)*gamax::size_of(res.get_column(k+1).type()));
    }

    cooperative_groups::wait(group);
}
#else
template<typename TyGroup>
struct copy_agg_from_smem_to_global_helper {
    template<typename T>
    __device__ __forceinline__ void operator()(
        const TyGroup& group,
        gamax::mutable_column_view agg,
        const gamax::size_type offset,
        void* block_agg,
        const gamax::size_type num_items)
    {
        auto src = reinterpret_cast<T*>(block_agg);
        for(gamax::size_type idx = group.thread_rank(); idx < num_items; idx += group.size()) {
            agg.data<T>()[offset + idx] = src[idx];
        }
        group.sync();
    }
};

template<typename TyGroup>
__device__ __forceinline__ void copy_agg_from_smem_to_global(
    const TyGroup& group,
    gamax::mutable_column_view agg,
    const gamax::size_type offset,
    void* block_agg, 
    const gamax::size_type num_items)
{
    type_dispatcher(
        agg.type(),
        copy_agg_from_smem_to_global_helper<TyGroup>{},
        group,
        agg,
        offset,
        block_agg,
        num_items);
}

template<typename TyGroup>
__device__ __forceinline__ void copy_agg_from_smem_to_global(
    const TyGroup& group,
    gamax::experimental::mutable_table_device_view res,
    const gamax::size_type offset,
    void* block_agg, 
    const gamax::size_type num_groups,
    const gamax::size_type num_aggs)
{
    auto cur_agg_buffer = block_agg;

    #pragma unroll
    for(gamax::size_type k = 0; k < num_aggs; k++) {
        copy_agg_from_smem_to_global(
            group,
            res.get_column(k+1), 
            offset, 
            cur_agg_buffer, 
            num_groups);
        cur_agg_buffer = reinterpret_cast<void*>(reinterpret_cast<char*>(cur_agg_buffer) + (num_groups)*gamax::size_of(res.get_column(k+1).type()));
    }

    group.sync();
}
#endif

__device__ __forceinline__ void init_agg(
    gamax::groupby::aggregation_id agg_kind,
    gamax::data_type src_type,
    gamax::mutable_column_view dst,
    gamax::size_type dst_idx)
{
    dispatch_type_aggregate(
        agg_kind,
        src_type,
        dst.data(),
        dst_idx);
}

__device__ __forceinline__ void init_agg(
    gamax::groupby::aggregation_id agg_kind,
    gamax::data_type src_type,
    void* dst,
    gamax::size_type dst_idx)
{
    dispatch_type_aggregate(
        agg_kind,
        src_type, // agg fn and src type can together deduce the output type
        dst, 
        dst_idx);
}

__inline__ __global__ void initialize_aggregation(
    gamax::experimental::table_device_view src,
    gamax::experimental::mutable_table_device_view res,
    gamax::groupby::aggregation_request* aggs)
{
    auto num_groups = res.num_rows();
    auto num_aggs = res.num_columns();

    for(gamax::size_type i = get_cuda_tid(); i < num_groups; i += nthreads()) {
        for(gamax::size_type j = 0; j < num_aggs; j++) {
            init_agg(
                aggs[j].agg_kind, 
                src.get_column(aggs[j].column_index).type(), 
                res.get_column(j), 
                i);
        }
    }
}
} // namespace gamax::groupby::details