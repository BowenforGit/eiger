// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// See LICENSE and THIRD_PARTY_NOTICES.md for license information.

#include <thrust/iterator/transform_iterator.h>

#include "operators/partition/partition.hpp"
#include "operators/sort/sort.hpp"
#include "utils/cub_wrapper.cuh"
#include "types/type_dispatcher.hpp"
#include "types/data_type.hpp"
#include "memory/column.hpp"

namespace gamax::partition {
std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
partition_column_pair(
    gamax::column& key_col, 
    gamax::column& val_col, 
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    void* d_temp_storage, 
    size_t temp_storage_bytes)
{
    return partition_column_pair(
        key_col.view(), val_col.view(), begin_bit, end_bit, stream,
        d_temp_storage, temp_storage_bytes);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
partition_column_pair(
    gamax::column_view key_col, 
    gamax::column_view val_col, 
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    void* d_temp_storage, 
    size_t temp_storage_bytes)
{
    return gamax::sort::sort_column_pair(
        key_col, 
        val_col, 
        stream,
        begin_bit, 
        end_bit, 
        gamax::sort_order::ASCENDING, 
        d_temp_storage, 
        temp_storage_bytes);
}

std::unique_ptr<gamax::column>
partition_column(
    gamax::column& col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    return partition_column(
        col.view(), begin_bit, end_bit, stream, d_temp_storage, temp_storage_bytes);
}

std::unique_ptr<gamax::column>
partition_column(
    gamax::column_view col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    return gamax::sort::sort_column(
        col, 
        stream,
        begin_bit, 
        end_bit, 
        gamax::sort_order::ASCENDING, 
        d_temp_storage, 
        temp_storage_bytes);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
arg_partition(
    gamax::column& col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    return arg_partition(
        col.view(), begin_bit, end_bit, stream, d_temp_storage, temp_storage_bytes);
}

std::pair<std::unique_ptr<gamax::column>, std::unique_ptr<gamax::column>> 
arg_partition(
    gamax::column_view col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream,
    void* d_temp_storage,
    size_t temp_storage_bytes)
{
    auto idx_col = gamax::make_sequence(col.size(), stream);
    return partition_column_pair(
        col, idx_col->view(), begin_bit, end_bit, stream,
        d_temp_storage, temp_storage_bytes);
}

namespace {
template<typename T>
struct RadixExtractor
{   
    int begin_bit;
    int mask;
    __host__ __device__ RadixExtractor(int begin_bit, int end_bit) : begin_bit(begin_bit), mask((1 << (end_bit - begin_bit)) - 1) {}
    
    template<typename U = T, std::enable_if_t<std::is_integral<U>::value, int> = 0>
    __host__ __device__ __forceinline__
    T operator()(const T &a) const {
        return (a >> begin_bit) & mask;
    }

    template<typename U = T, std::enable_if_t<!std::is_integral<U>::value, int> = 0>
    __device__ __forceinline__
    T operator()(const T &a) const {
        GAMAX_UNREACHABLE("Cannot extract radix from non-integral type");
    }
};

struct compute_partition_offsets_impl {
    template<typename T>
    std::unique_ptr<gamax::column> operator()(
        gamax::column_view partitioned_key_col,
        int begin_bit,
        int end_bit,
        cudaStream_t stream)
    {
        const auto n_partitions = (1 << (end_bit - begin_bit));
        const auto num_items = partitioned_key_col.size();

        // Allocate AND run CUB on the caller's `stream` (not the default stream). CUB's
        // trailing stream arg defaults to 0; omitting it ran the histogram/scan on stream 0
        // while the partitioned input was produced on `stream`. A normal blocking stream
        // masks that (it serializes with stream 0), but a non-default stream -- e.g. a CUDA
        // green-context stream -- does not, so the counts raced the partition -> wrong
        // offsets. Threading `stream` everywhere makes the ordering correct on any stream.
        auto d_counts_out_ptr = gamax::make_idx_column(n_partitions, stream);
        auto offsets = gamax::make_idx_column(n_partitions+1, stream);

        RadixExtractor<T> conversion_op(begin_bit, end_bit);
        auto itr = thrust::make_transform_iterator(partitioned_key_col.data<T>(), conversion_op);

        // HistogramEven requires the counter variable must have a type supported by atomicAdd(_block)
        if constexpr (gamax::use_64bit_size_type) {
            CALL_CUB_FUNC(stream, cub::DeviceHistogram::HistogramEven, itr, d_counts_out_ptr->data<unsigned long long>(), n_partitions+1, 0, n_partitions, num_items, stream);
        } else {
            CALL_CUB_FUNC(stream, cub::DeviceHistogram::HistogramEven, itr, d_counts_out_ptr->data<gamax::size_type>(), n_partitions+1, 0, n_partitions, num_items, stream);
        }
        CALL_CUB_FUNC(stream, cub::DeviceScan::InclusiveSum, d_counts_out_ptr->data<gamax::size_type>(), offsets->data<gamax::size_type>(), n_partitions, stream);

        return offsets;
    }
};
    
} // unnamed namespace

std::unique_ptr<gamax::column>
compute_partition_offsets(
    gamax::column& partitioned_key_col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream)
{
    return compute_partition_offsets(partitioned_key_col.view(), begin_bit, end_bit, stream);
}

std::unique_ptr<gamax::column>
compute_partition_offsets(
    gamax::column_view partitioned_key_col,
    int begin_bit,
    int end_bit,
    cudaStream_t stream)
{
    return gamax::type_dispatcher(partitioned_key_col.type(), compute_partition_offsets_impl{}, partitioned_key_col, begin_bit, end_bit, stream);
}
} // namespace gamax::partition
